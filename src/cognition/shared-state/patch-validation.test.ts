import { describe, expect, it } from "vitest";

import type { SharedStateArtifact, SharedStateEntry } from "../../memory/shared-state/index.js";
import {
  createEntityId,
  createGoalId,
  createRelationalSlotId,
  createSharedStateEntryId,
  createStreamEntryId,
  type EntityId,
  type StreamEntryId,
} from "../../util/ids.js";
import { unknownMemoryDisclosureLabel } from "../../memory/common/disclosure-label.js";
import type { RelationshipClaim } from "../../memory/common/relationship-claims.js";
import { memoryDisclosurePayloadFields } from "../../memory/common/disclosure-serializers.js";
import type { EmitSharedStatePatch, SharedStateCanonicalizationCandidates } from "./types.js";
import {
  allowedCanonicalizationIds,
  dedupeCanonicalizesAcrossOperations,
  normalizePatch,
} from "./patch-validation.js";
import {
  applySharedStateArtifactLifecycleCap,
  expandPruneDependencies,
  materializeSharedStateOperationIds,
} from "./lifecycle-cap.js";
import { stateKeysAreNearDuplicate } from "./state-key.js";
import {
  buildSharedStateArtifactPromptSummaryWithExposure,
  type SharedStatePromptExposure,
  type SharedStatePromptSummaryOptions,
} from "./summary.js";

const EMPTY_CANONICALIZES = {
  goal_ids: [],
  commitment_ids: [],
  action_ids: [],
  open_question_ids: [],
};

function relationshipClaim(overrides: Partial<RelationshipClaim> = {}): RelationshipClaim {
  return {
    label_family: "kinship",
    subject_entity_id: null,
    object_entity_id: null,
    object_text: "relación familiar",
    requires_grounding: true,
    evidence_relational_slot_ids: [],
    evidence_stream_entry_ids: [],
    ...overrides,
  };
}

function makeArtifact(input: {
  audienceEntityId: EntityId;
  entries: readonly SharedStateEntry[];
}): SharedStateArtifact {
  return {
    audience_entity_id: input.audienceEntityId,
    record_version: 1,
    created_at: 1_000,
    updated_at: 1_000,
    last_compiled_at: null,
    last_compiled_stream_entry_id: null,
    entries: [...input.entries],
  };
}

function makeEntry(input: {
  audienceEntityId: EntityId;
  sourceStreamEntryId: StreamEntryId;
  stateKey: string | null;
  kind?: SharedStateEntry["kind"];
  rank?: number;
  text?: string;
  ownerEntityId?: EntityId | null;
  canonicalizes?: SharedStateEntry["canonicalizes"];
}): SharedStateEntry {
  const rank = input.rank ?? 0;

  return {
    id: createSharedStateEntryId(),
    audience_entity_id: input.audienceEntityId,
    state_key: input.stateKey,
    kind: input.kind ?? "live",
    text: input.text ?? `Entry ${rank}`,
    owner_entity_id: input.ownerEntityId ?? null,
    provenance_stream_entry_ids: [input.sourceStreamEntryId],
    last_updated_stream_entry_ids: [input.sourceStreamEntryId],
    created_at: 1_000 + rank,
    last_updated_at: 1_000 + rank,
    last_updated_turn_global: null,
    superseded_by_id: null,
    rank,
    canonicalizes: input.canonicalizes ?? EMPTY_CANONICALIZES,
  };
}

function normalizeKeyedPatch(input: {
  previousEntries: readonly SharedStateEntry[];
  operations: EmitSharedStatePatch["operations"];
  audienceEntityId: EntityId;
  sourceStreamEntryId: StreamEntryId;
  allowedSourceStreamEntryIds?: readonly StreamEntryId[];
  participantRoster?: Parameters<typeof normalizePatch>[0]["participantRoster"];
  relationshipEvidenceStreamEntryTrust?: Parameters<
    typeof normalizePatch
  >[0]["relationshipEvidenceStreamEntryTrust"];
  canonicalizationCandidates?: SharedStateCanonicalizationCandidates;
  summaryOptions?: SharedStatePromptSummaryOptions;
  promptExposure?: SharedStatePromptExposure | null;
}) {
  const selfEntityId = createEntityId();
  const speakerEntityId = createEntityId();
  const previousArtifact = makeArtifact({
    audienceEntityId: input.audienceEntityId,
    entries: input.previousEntries,
  });

  return normalizePatch({
    patch: {
      operations: input.operations,
    },
    previousArtifact,
    promptExposure:
      input.promptExposure === undefined
        ? buildSharedStateArtifactPromptSummaryWithExposure(previousArtifact, input.summaryOptions)
            .exposure
        : input.promptExposure,
    audienceEntityId: input.audienceEntityId,
    selfEntityId,
    speakerEntityId,
    participants: [],
    participantRoster: input.participantRoster,
    relationshipEvidenceStreamEntryTrust: input.relationshipEvidenceStreamEntryTrust,
    allowedSourceStreamEntryIds: new Set(
      input.allowedSourceStreamEntryIds ?? [input.sourceStreamEntryId],
    ),
    allowedCanonicalizationIds: allowedCanonicalizationIds(input.canonicalizationCandidates),
    maxLiveEntriesPerKey: 2,
    nowMs: 2_000,
    materializePlan: (operations, retainedEntryIds) => {
      const compilerOperations = materializeSharedStateOperationIds(operations);
      const lifecycle = applySharedStateArtifactLifecycleCap({
        previousArtifact,
        operations: compilerOperations,
        retainedEntryIds,
        nowMs: 2_000,
      });
      return {
        compilerOperations,
        ...dedupeCanonicalizesAcrossOperations(
          expandPruneDependencies({
            previousArtifact,
            operations: lifecycle.operations,
            nowMs: 2_000,
          }),
        ),
      };
    },
  });
}

function canonicalizationCandidate<T extends { id: string; text: string }>(
  candidate: T,
): T & ReturnType<typeof memoryDisclosurePayloadFields> {
  return {
    ...candidate,
    ...memoryDisclosurePayloadFields(unknownMemoryDisclosureLabel()),
  };
}

function addOperation(input: {
  stateKey: string;
  kind?: Extract<EmitSharedStatePatch["operations"][number], { type: "add" }>["kind"];
  sourceStreamEntryId: StreamEntryId;
  newKeyReason?: string | null;
}): Extract<EmitSharedStatePatch["operations"][number], { type: "add" }> {
  return {
    type: "add",
    state_key: input.stateKey,
    ...(input.newKeyReason === null
      ? {}
      : { new_key_reason: input.newKeyReason ?? "test fixture new key" }),
    kind: input.kind ?? "live",
    text: "New keyed entry",
    source_stream_entry_ids: [input.sourceStreamEntryId],
  };
}

describe("normalizePatch locked text exposure", () => {
  it.each(
    (["omitted", "truncated", "full"] as const).flatMap((exposure) =>
      (["prune", "rekey", "demote", "supersede"] as const).flatMap((displacement) =>
        (["add", "update", "supersede"] as const).flatMap((write) =>
          [false, true].map((reverse) => ({ exposure, displacement, write, reverse })),
        ),
      ),
    ),
  )(
    "checks the post-state of $displacement + $write ($exposure, reverse=$reverse)",
    ({ exposure, displacement, write, reverse }) => {
      const audienceEntityId = createEntityId();
      const sourceStreamEntryId = createStreamEntryId();
      const goalId = createGoalId();
      const entry = makeEntry({
        audienceEntityId,
        sourceStreamEntryId,
        stateKey: "decision.rule",
        kind: "locked",
        text: "規則の条件と例。".repeat(100),
        canonicalizes: { ...EMPTY_CANONICALIZES, goal_ids: [goalId] },
      });
      const other = makeEntry({
        audienceEntityId,
        sourceStreamEntryId,
        stateKey: "observation.reference",
      });
      const displacementOperation: EmitSharedStatePatch["operations"][number] =
        displacement === "prune"
          ? { type: "prune", id: entry.id }
          : displacement === "supersede"
            ? {
                type: "supersede",
                id: entry.id,
                replacement: {
                  state_key: "archive.retired",
                  kind: "live",
                  text: "Replacement",
                  source_stream_entry_ids: [sourceStreamEntryId],
                },
              }
            : {
                type: "update",
                id: entry.id,
                state_key: displacement === "rekey" ? "archive.retired" : "decision.rule",
                ...(displacement === "demote" ? { kind: "live" as const } : {}),
                source_stream_entry_ids: [sourceStreamEntryId],
              };
      const canonicalizes = { goal_ids: [goalId] };
      const writeOperation: EmitSharedStatePatch["operations"][number] =
        write === "add"
          ? {
              ...addOperation({ stateKey: "decision.rule", kind: "locked", sourceStreamEntryId }),
              canonicalizes,
            }
          : write === "supersede"
            ? {
                type: "supersede",
                id: other.id,
                canonicalizes,
                replacement: {
                  state_key: "decision.rule",
                  kind: "locked",
                  text: "Replacement",
                  source_stream_entry_ids: [sourceStreamEntryId],
                },
              }
            : {
                type: "update",
                id: other.id,
                state_key: "decision.rule",
                kind: "locked",
                canonicalizes,
                source_stream_entry_ids: [sourceStreamEntryId],
              };
      const paired = [displacementOperation, writeOperation];
      const result = normalizeKeyedPatch({
        audienceEntityId,
        sourceStreamEntryId,
        previousEntries: [entry, other],
        canonicalizationCandidates: {
          goals: [canonicalizationCandidate({ id: goalId, text: "Goal handle" })],
        },
        summaryOptions: {
          maxEntries: { locked: exposure === "omitted" ? 0 : 1 },
          maxEntryTextTokens: exposure === "full" ? 1_000 : 20,
        },
        operations: [
          ...(reverse ? paired.reverse() : paired),
          addOperation({ stateKey: "plan.meeting", sourceStreamEntryId }),
        ],
      });

      const keyCollision = write === "add" && reverse;
      const exposureViolation =
        exposure !== "full" &&
        (displacement === "supersede" || (displacement === "prune" && !keyCollision));
      const exposureIndices = exposureViolation ? (keyCollision ? [1] : [0, 1]) : [];
      expect(result.operations).toHaveLength(3 - exposureIndices.length - Number(keyCollision));
      expect(result.operations.at(-1)).toMatchObject({ type: "add", state_key: "plan.meeting" });
      expect(result.rejected).toEqual([
        ...(keyCollision
          ? [expect.objectContaining({ reason: "locked_state_key_collision", operationIndex: 0 })]
          : []),
        ...exposureIndices.map((operationIndex) =>
          expect.objectContaining({
            reason: "locked_target_not_fully_rendered",
            operationIndex,
            targetEntryId: entry.id,
            stateKey: "decision.rule",
            targetExposure: exposure,
          }),
        ),
      ]);
      expect(result.modelPruneRequests).toHaveLength(
        displacement === "prune" && !exposureViolation ? 1 : 0,
      );
    },
  );

  it.each(["decision.rule", null])(
    "compares only the original exact key %s after re-keying before prune + add",
    (stateKey) => {
      const audienceEntityId = createEntityId();
      const sourceStreamEntryId = createStreamEntryId();
      const entry = makeEntry({ audienceEntityId, sourceStreamEntryId, stateKey, kind: "locked" });
      const result = normalizeKeyedPatch({
        audienceEntityId,
        sourceStreamEntryId,
        previousEntries: [entry],
        summaryOptions: { maxEntries: { locked: 0 } },
        operations: [
          {
            type: "update",
            id: entry.id,
            state_key: "archive.retired",
            source_stream_entry_ids: [sourceStreamEntryId],
          },
          { type: "prune", id: entry.id },
          addOperation({ stateKey: "archive.retired", sourceStreamEntryId }),
          addOperation({ stateKey: "plan.meeting", sourceStreamEntryId }),
        ],
      });
      expect(result.operations).toHaveLength(4);
      expect(result.rejected).toEqual([]);
    },
  );

  it("checks locked history removed by a prune dependency before adding under its key", () => {
    const audienceEntityId = createEntityId();
    const sourceStreamEntryId = createStreamEntryId();
    const other = makeEntry({
      audienceEntityId,
      sourceStreamEntryId,
      stateKey: "observation.reference",
    });
    const historical = {
      ...makeEntry({
        audienceEntityId,
        sourceStreamEntryId,
        stateKey: "decision.rule",
        kind: "locked",
      }),
      superseded_by_id: other.id,
    };
    const result = normalizeKeyedPatch({
      audienceEntityId,
      sourceStreamEntryId,
      previousEntries: [historical, other],
      summaryOptions: { summaryTokenBudget: 1, maxEntries: { locked: 0, live: 0 } },
      operations: [
        { type: "prune", id: other.id },
        addOperation({ stateKey: "decision.rule", sourceStreamEntryId }),
      ],
    });
    // Removing the same-key writer leaves a bare dependency prune, which remains allowed.
    expect(result.operations).toEqual([{ type: "prune", id: other.id }]);
    expect(result.plan.operations).toEqual([
      { type: "prune", id: historical.id },
      { type: "prune", id: other.id },
    ]);
    expect(result.rejected).toEqual([
      expect.objectContaining({
        reason: "locked_target_not_fully_rendered",
        operationIndex: 1,
        targetEntryId: historical.id,
        stateKey: "decision.rule",
        targetExposure: "omitted",
      }),
    ]);
  });

  it("allows canonicalization-only updates and tentative same-key additions without displacement", () => {
    const audienceEntityId = createEntityId();
    const sourceStreamEntryId = createStreamEntryId();
    const goalId = createGoalId();
    const entry = makeEntry({
      audienceEntityId,
      sourceStreamEntryId,
      stateKey: "decision.rule",
      kind: "locked",
    });
    const result = normalizeKeyedPatch({
      audienceEntityId,
      sourceStreamEntryId,
      previousEntries: [entry],
      canonicalizationCandidates: {
        goals: [canonicalizationCandidate({ id: goalId, text: "Goal handle" })],
      },
      summaryOptions: { maxEntries: { locked: 0 } },
      operations: [
        {
          type: "update",
          id: entry.id,
          state_key: "decision.rule",
          canonicalizes: { goal_ids: [goalId] },
          source_stream_entry_ids: [sourceStreamEntryId],
        },
        addOperation({ stateKey: "decision.rule", kind: "tentative", sourceStreamEntryId }),
      ],
    });
    expect(result.rejected).toEqual([]);
    expect(result.operations).toEqual([
      expect.objectContaining({
        type: "update",
        id: entry.id,
        canonicalizes: { ...EMPTY_CANONICALIZES, goal_ids: [goalId] },
      }),
      expect.objectContaining({ type: "add", kind: "tentative", state_key: "decision.rule" }),
    ]);
  });

  it.each(
    ["Rule with constraints and examples. ", "規則には制約、例、条件があります。"].flatMap((text) =>
      (["full", "truncated", "omitted"] as const).flatMap((completeness) =>
        (["update", "supersede"] as const).map((type) => ({ text, completeness, type })),
      ),
    ),
  )("requires full exposure for $type ($completeness): $text", ({ text, completeness, type }) => {
    const audienceEntityId = createEntityId();
    const sourceStreamEntryId = createStreamEntryId();
    const entry = makeEntry({
      audienceEntityId,
      sourceStreamEntryId,
      stateKey: "decision.rule",
      kind: "locked",
      text: text.repeat(30),
    });
    const replacement = {
      state_key: "decision.rule",
      kind: "locked" as const,
      text: "全く別の本文。",
      source_stream_entry_ids: [sourceStreamEntryId],
    };
    const result = normalizeKeyedPatch({
      audienceEntityId,
      sourceStreamEntryId,
      previousEntries: [entry],
      summaryOptions: {
        maxEntries: { locked: completeness === "omitted" ? 0 : 1 },
        maxEntryTextTokens: completeness === "truncated" ? 20 : 1_000,
      },
      operations: [
        type === "update"
          ? { type, id: entry.id, ...replacement }
          : { type, id: entry.id, replacement },
        {
          type: "update",
          id: entry.id,
          state_key: "decision.rule",
          owner_entity_id: audienceEntityId,
          source_stream_entry_ids: [sourceStreamEntryId],
        },
        addOperation({ stateKey: "plan.meeting", sourceStreamEntryId }),
      ],
    });

    expect(result.operations).toHaveLength(completeness === "full" ? 3 : 1);
    if (completeness === "full")
      expect(result.operations.at(-2)).toMatchObject({
        type: "update",
        owner_entity_id: audienceEntityId,
      });
    expect(result.operations.at(-1)).toMatchObject({ type: "add", state_key: "plan.meeting" });
    expect(result.rejected).toEqual(
      completeness === "full"
        ? []
        : [0, 1].map((operationIndex) =>
            expect.objectContaining({
              reason: "locked_target_not_fully_rendered",
              operationType: operationIndex === 0 ? type : "update",
              operationIndex,
              targetEntryId: entry.id,
              targetExposure: completeness,
            }),
          ),
    );
  });

  it.each(["missing", "stale_version", "wrong_artifact"] as const)(
    "fails closed with %s exposure, even after a non-text demotion",
    (exposureCase) => {
      const audienceEntityId = createEntityId();
      const sourceStreamEntryId = createStreamEntryId();
      const entry = makeEntry({
        audienceEntityId,
        sourceStreamEntryId,
        stateKey: "decision.rule",
        kind: "locked",
      });
      const artifact = makeArtifact({ audienceEntityId, entries: [entry] });
      const { exposure } = buildSharedStateArtifactPromptSummaryWithExposure(artifact);
      const result = normalizeKeyedPatch({
        audienceEntityId,
        sourceStreamEntryId,
        previousEntries: [entry],
        promptExposure:
          exposureCase === "missing"
            ? null
            : {
                ...exposure!,
                ...(exposureCase === "stale_version"
                  ? { artifact_version: 2 }
                  : { audience_entity_id: createEntityId() }),
              },
        operations: [
          {
            type: "update",
            id: entry.id,
            state_key: "decision.rule",
            kind: "live",
            source_stream_entry_ids: [sourceStreamEntryId],
          },
          {
            type: "update",
            id: entry.id,
            state_key: "decision.rule",
            text: "Replacement",
            source_stream_entry_ids: [sourceStreamEntryId],
          },
        ],
      });
      expect(result.operations).toEqual([]);
      expect(result.rejected).toEqual(
        [0, 1].map((operationIndex) =>
          expect.objectContaining({
            reason: "locked_target_not_fully_rendered",
            targetExposure: "unavailable",
            operationIndex,
          }),
        ),
      );
    },
  );
});

describe("normalizePatch exposure recheck", () => {
  it("rejects the whole remaining patch when removing a write reconnects a prune dependency", () => {
    const audienceEntityId = createEntityId();
    const sourceStreamEntryId = createStreamEntryId();
    const successor = makeEntry({
      audienceEntityId,
      sourceStreamEntryId,
      stateKey: "observation.successor",
    });
    const middle = {
      ...makeEntry({ audienceEntityId, sourceStreamEntryId, stateKey: "observation.middle" }),
      superseded_by_id: successor.id,
    };
    const historical = {
      ...makeEntry({
        audienceEntityId,
        sourceStreamEntryId,
        stateKey: "decision.history",
        kind: "locked",
      }),
      superseded_by_id: middle.id,
    };
    const locked = makeEntry({
      audienceEntityId,
      sourceStreamEntryId,
      stateKey: "boundary.access",
      kind: "locked",
    });
    const result = normalizeKeyedPatch({
      audienceEntityId,
      sourceStreamEntryId,
      previousEntries: [historical, middle, successor, locked],
      summaryOptions: { summaryTokenBudget: 1, maxEntries: { locked: 0, live: 0 } },
      operations: [
        {
          type: "update",
          id: locked.id,
          state_key: "boundary.access",
          text: "Replacement",
          source_stream_entry_ids: [sourceStreamEntryId],
        },
        {
          type: "supersede",
          id: middle.id,
          replacement: {
            state_key: "boundary.access",
            kind: "tentative",
            text: "New middle",
            source_stream_entry_ids: [sourceStreamEntryId],
          },
        },
        { type: "prune", id: successor.id },
        addOperation({ stateKey: "decision.history", sourceStreamEntryId }),
        addOperation({ stateKey: "plan.meeting", sourceStreamEntryId }),
      ],
    });
    // Initially H remains linked to M, whose replacement disconnected it from S.
    // Rejecting the writes to L reconnects H -> M -> S; pruning S would now remove H.
    expect(result.operations).toEqual([]);
    expect(result.plan.operations).toEqual([]);
    expect(result.rejected).toEqual(
      [0, 1, 2, 3, 4].map((operationIndex) =>
        expect.objectContaining({
          reason: "locked_target_not_fully_rendered",
          operationIndex,
          targetExposure: "omitted",
          targetEntryId: operationIndex < 2 ? locked.id : historical.id,
          stateKey: operationIndex < 2 ? "boundary.access" : "decision.history",
        }),
      ),
    );
  });
});

describe("normalizePatch empty update no-op handling", () => {
  it("drops an update with no material fields", () => {
    const audienceEntityId = createEntityId();
    const sourceStreamEntryId = createStreamEntryId();
    const entry = makeEntry({
      audienceEntityId,
      sourceStreamEntryId,
      stateKey: "decision.route",
      text: "Madrid 3 is locked.",
    });

    const result = normalizeKeyedPatch({
      previousEntries: [entry],
      operations: [
        {
          type: "update",
          id: entry.id,
          state_key: entry.state_key ?? "decision.route",
          source_stream_entry_ids: [sourceStreamEntryId],
        },
      ],
      audienceEntityId,
      sourceStreamEntryId,
    });

    expect(result.operations).toEqual([]);
    expect(result.rejected).toEqual([]);
    expect(result.emptyUpdateAttemptedCount).toBe(1);
    expect(result.emptyUpdateDrops).toEqual([
      {
        operationIndex: 0,
        operationId: entry.id,
        stateKey: "decision.route",
        fieldPresence: {
          kind: false,
          text: false,
          owner_entity_id: false,
          canonicalizes: false,
        },
      },
    ]);
  });

  it("drops an update with text equal to the existing entry", () => {
    const audienceEntityId = createEntityId();
    const sourceStreamEntryId = createStreamEntryId();
    const entry = makeEntry({
      audienceEntityId,
      sourceStreamEntryId,
      stateKey: "decision.route",
      text: "Madrid 3 is locked.",
    });

    const result = normalizeKeyedPatch({
      previousEntries: [entry],
      operations: [
        {
          type: "update",
          id: entry.id,
          state_key: "decision.route",
          text: "Madrid 3 is locked.",
          source_stream_entry_ids: [sourceStreamEntryId],
        },
      ],
      audienceEntityId,
      sourceStreamEntryId,
    });

    expect(result.operations).toEqual([]);
    expect(result.rejected).toEqual([]);
    expect(result.emptyUpdateDrops).toHaveLength(1);
    expect(result.emptyUpdateDrops[0]?.fieldPresence.text).toBe(true);
  });

  it("applies an update with text different from the existing entry", () => {
    const audienceEntityId = createEntityId();
    const sourceStreamEntryId = createStreamEntryId();
    const entry = makeEntry({
      audienceEntityId,
      sourceStreamEntryId,
      stateKey: "decision.route",
      text: "Madrid 3 is locked.",
    });

    const result = normalizeKeyedPatch({
      previousEntries: [entry],
      operations: [
        {
          type: "update",
          id: entry.id,
          state_key: "decision.route",
          text: "Madrid 4 is locked.",
          source_stream_entry_ids: [sourceStreamEntryId],
        },
      ],
      audienceEntityId,
      sourceStreamEntryId,
    });

    expect(result.emptyUpdateDrops).toEqual([]);
    expect(result.emptyUpdateAttemptedCount).toBe(1);
    expect(result.rejected).toEqual([]);
    expect(result.operations).toEqual([
      expect.objectContaining({
        type: "update",
        id: entry.id,
        text: "Madrid 4 is locked.",
      }),
    ]);
  });

  it("drops an update that repeats an existing owner outside the current allowed set", () => {
    const audienceEntityId = createEntityId();
    const historicalOwnerEntityId = createEntityId();
    const sourceStreamEntryId = createStreamEntryId();
    const entry = makeEntry({
      audienceEntityId,
      sourceStreamEntryId,
      stateKey: "decision.route",
      text: "Madrid 3 is locked.",
      ownerEntityId: historicalOwnerEntityId,
    });

    const result = normalizeKeyedPatch({
      previousEntries: [entry],
      operations: [
        {
          type: "update",
          id: entry.id,
          state_key: "decision.route",
          owner_entity_id: historicalOwnerEntityId,
          source_stream_entry_ids: [sourceStreamEntryId],
        },
      ],
      audienceEntityId,
      sourceStreamEntryId,
    });

    expect(result.operations).toEqual([]);
    expect(result.rejected).toEqual([]);
    expect(result.emptyUpdateDrops).toEqual([
      expect.objectContaining({
        operationId: entry.id,
        fieldPresence: expect.objectContaining({
          owner_entity_id: true,
        }),
      }),
    ]);
  });

  it("rejects an update that introduces a new owner outside the current allowed set", () => {
    const audienceEntityId = createEntityId();
    const historicalOwnerEntityId = createEntityId();
    const disallowedOwnerEntityId = createEntityId();
    const sourceStreamEntryId = createStreamEntryId();
    const entry = makeEntry({
      audienceEntityId,
      sourceStreamEntryId,
      stateKey: "decision.route",
      text: "Madrid 3 is locked.",
      ownerEntityId: historicalOwnerEntityId,
    });

    const result = normalizeKeyedPatch({
      previousEntries: [entry],
      operations: [
        {
          type: "update",
          id: entry.id,
          state_key: "decision.route",
          owner_entity_id: disallowedOwnerEntityId,
          source_stream_entry_ids: [sourceStreamEntryId],
        },
      ],
      audienceEntityId,
      sourceStreamEntryId,
    });

    expect(result.operations).toEqual([]);
    expect(result.emptyUpdateDrops).toEqual([]);
    expect(result.emptyUpdateAttemptedCount).toBe(0);
    expect(result.rejected).toEqual([
      expect.objectContaining({
        reason: "invalid_owner_entity_id",
        operationIndex: 0,
      }),
    ]);
  });

  it("applies a text change that repeats an existing owner outside the current allowed set", () => {
    const audienceEntityId = createEntityId();
    const historicalOwnerEntityId = createEntityId();
    const sourceStreamEntryId = createStreamEntryId();
    const entry = makeEntry({
      audienceEntityId,
      sourceStreamEntryId,
      stateKey: "decision.route",
      text: "Madrid 3 is locked.",
      ownerEntityId: historicalOwnerEntityId,
    });

    const result = normalizeKeyedPatch({
      previousEntries: [entry],
      operations: [
        {
          type: "update",
          id: entry.id,
          state_key: "decision.route",
          owner_entity_id: historicalOwnerEntityId,
          text: "Madrid 4 is locked.",
          source_stream_entry_ids: [sourceStreamEntryId],
        },
      ],
      audienceEntityId,
      sourceStreamEntryId,
    });

    expect(result.emptyUpdateDrops).toEqual([]);
    expect(result.rejected).toEqual([]);
    expect(result.operations).toEqual([
      expect.objectContaining({
        type: "update",
        id: entry.id,
        owner_entity_id: historicalOwnerEntityId,
        text: "Madrid 4 is locked.",
      }),
    ]);
  });

  it("applies an update with same text and a new canonicalizes id", () => {
    const audienceEntityId = createEntityId();
    const sourceStreamEntryId = createStreamEntryId();
    const goalId = createGoalId();
    const entry = makeEntry({
      audienceEntityId,
      sourceStreamEntryId,
      stateKey: "decision.route",
      kind: "locked",
      text: "Madrid 3 is locked.",
    });

    const result = normalizeKeyedPatch({
      previousEntries: [entry],
      operations: [
        {
          type: "update",
          id: entry.id,
          state_key: "decision.route",
          text: "Madrid 3 is locked.",
          canonicalizes: {
            goal_ids: [goalId],
          },
          source_stream_entry_ids: [sourceStreamEntryId],
        },
      ],
      audienceEntityId,
      sourceStreamEntryId,
      canonicalizationCandidates: {
        goals: [canonicalizationCandidate({ id: goalId, text: "Route goal" })],
      },
    });

    expect(result.emptyUpdateDrops).toEqual([]);
    expect(result.rejected).toEqual([]);
    expect(result.operations).toEqual([
      expect.objectContaining({
        type: "update",
        id: entry.id,
        canonicalizes: expect.objectContaining({
          goal_ids: [goalId],
        }),
      }),
    ]);
  });

  it("drops an update with same text and only existing canonicalizes ids", () => {
    const audienceEntityId = createEntityId();
    const sourceStreamEntryId = createStreamEntryId();
    const goalId = createGoalId();
    const entry = makeEntry({
      audienceEntityId,
      sourceStreamEntryId,
      stateKey: "decision.route",
      kind: "locked",
      text: "Madrid 3 is locked.",
      canonicalizes: {
        ...EMPTY_CANONICALIZES,
        goal_ids: [goalId],
      },
    });

    const result = normalizeKeyedPatch({
      previousEntries: [entry],
      operations: [
        {
          type: "update",
          id: entry.id,
          state_key: "decision.route",
          text: "Madrid 3 is locked.",
          canonicalizes: {
            goal_ids: [goalId],
          },
          source_stream_entry_ids: [sourceStreamEntryId],
        },
      ],
      audienceEntityId,
      sourceStreamEntryId,
      canonicalizationCandidates: {
        goals: [canonicalizationCandidate({ id: goalId, text: "Route goal" })],
      },
    });

    expect(result.operations).toEqual([]);
    expect(result.rejected).toEqual([]);
    expect(result.emptyUpdateAttemptedCount).toBe(1);
    expect(result.emptyUpdateDrops).toHaveLength(1);
    expect(result.emptyUpdateDrops[0]?.fieldPresence.canonicalizes).toBe(true);
  });

  it("drops a provenance-only update with new source citations", () => {
    const audienceEntityId = createEntityId();
    const originalSourceStreamEntryId = createStreamEntryId();
    const citationOnlySourceStreamEntryId = createStreamEntryId();
    const entry = makeEntry({
      audienceEntityId,
      sourceStreamEntryId: originalSourceStreamEntryId,
      stateKey: "decision.route",
      text: "Madrid 3 is locked.",
    });

    const result = normalizeKeyedPatch({
      previousEntries: [entry],
      operations: [
        {
          type: "update",
          id: entry.id,
          state_key: "decision.route",
          source_stream_entry_ids: [citationOnlySourceStreamEntryId],
        },
      ],
      audienceEntityId,
      sourceStreamEntryId: originalSourceStreamEntryId,
      allowedSourceStreamEntryIds: [originalSourceStreamEntryId, citationOnlySourceStreamEntryId],
    });

    expect(result.operations).toEqual([]);
    expect(result.rejected).toEqual([]);
    expect(result.emptyUpdateAttemptedCount).toBe(1);
    expect(result.emptyUpdateDrops).toHaveLength(1);
  });

  it("drops empty updates while preserving valid operations in a mixed patch", () => {
    const audienceEntityId = createEntityId();
    const sourceStreamEntryId = createStreamEntryId();
    const entry = makeEntry({
      audienceEntityId,
      sourceStreamEntryId,
      stateKey: "decision.route",
      text: "Madrid 3 is locked.",
    });

    const result = normalizeKeyedPatch({
      previousEntries: [entry],
      operations: [
        {
          type: "update",
          id: entry.id,
          state_key: "decision.route",
          source_stream_entry_ids: [sourceStreamEntryId],
        },
        {
          type: "update",
          id: entry.id,
          state_key: "decision.route",
          text: "Madrid 4 is locked.",
          source_stream_entry_ids: [sourceStreamEntryId],
        },
      ],
      audienceEntityId,
      sourceStreamEntryId,
    });

    expect(result.rejected).toEqual([]);
    expect(result.emptyUpdateAttemptedCount).toBe(2);
    expect(result.emptyUpdateDrops).toHaveLength(1);
    expect(result.operations).toEqual([
      expect.objectContaining({
        type: "update",
        text: "Madrid 4 is locked.",
      }),
    ]);
  });
});

describe("normalizePatch state_key validation", () => {
  it("renames an entry in place when an update carries a different state_key", () => {
    const audienceEntityId = createEntityId();
    const sourceStreamEntryId = createStreamEntryId();
    const previousStateKey = "audit.inference.supersede_as_durable_correction_path";
    const nextStateKey = "audit.inference.supersede_as_durable_correction_route";
    const entry = makeEntry({
      audienceEntityId,
      sourceStreamEntryId,
      stateKey: previousStateKey,
      kind: "locked",
      text: "Supersede was recorded as the durable correction path.",
    });

    const result = normalizeKeyedPatch({
      previousEntries: [entry],
      operations: [
        {
          type: "update",
          id: entry.id,
          state_key: nextStateKey,
          source_stream_entry_ids: [sourceStreamEntryId],
        },
      ],
      audienceEntityId,
      sourceStreamEntryId,
    });

    // The key guards -- near-duplicate, locked collision, new_key_reason -- live in
    // validateAddStateKey, which only the add case calls. An add carrying this key against this
    // artifact would be rejected as a near duplicate; the same key lands on an update.
    expect(stateKeysAreNearDuplicate(nextStateKey, previousStateKey)).toBe(true);
    expect(result.rejected).toEqual([]);
    // A key-only change is material, so it does not fall into the empty-update drop.
    expect(result.emptyUpdateDrops).toEqual([]);
    expect(result.operations).toEqual([
      expect.objectContaining({
        type: "update",
        id: entry.id,
        state_key: nextStateKey,
      }),
    ]);
  });

  it("rejects required relationship claims without grounding evidence", () => {
    const audienceEntityId = createEntityId();
    const sourceStreamEntryId = createStreamEntryId();
    const result = normalizeKeyedPatch({
      audienceEntityId,
      sourceStreamEntryId,
      previousEntries: [],
      operations: [
        {
          ...addOperation({
            stateKey: "plan.attendees",
            sourceStreamEntryId,
          }),
          text: "The care plan depends on a sensitive relationship assertion.",
          relationship_claims: [relationshipClaim({ object_text: "mi esposa" })],
        },
      ],
    });

    expect(result.operations).toEqual([]);
    expect(result.rejected).toEqual([
      expect.objectContaining({
        reason: "relationship_claim_ungrounded",
        operationType: "add",
        operationIndex: 0,
        relationshipClaims: [expect.objectContaining({ object_text: "mi esposa" })],
        ungroundedRelationshipClaims: [expect.objectContaining({ object_text: "mi esposa" })],
      }),
    ]);
  });

  it("accepts contextual role labels without relationship grounding evidence", () => {
    const audienceEntityId = createEntityId();
    const sourceStreamEntryId = createStreamEntryId();
    const result = normalizeKeyedPatch({
      audienceEntityId,
      sourceStreamEntryId,
      previousEntries: [],
      operations: [
        {
          ...addOperation({
            stateKey: "plan.rollout",
            sourceStreamEntryId,
          }),
          text: "The design partner and rollout owner are tracked for this project.",
        },
      ],
    });

    expect(result.rejected).toEqual([]);
    expect(result.operations).toEqual([
      expect.objectContaining({
        type: "add",
        text: "The design partner and rollout owner are tracked for this project.",
      }),
    ]);
  });

  it("accepts medical and professional context nouns without relationship grounding evidence", () => {
    const audienceEntityId = createEntityId();
    const sourceStreamEntryId = createStreamEntryId();
    const result = normalizeKeyedPatch({
      audienceEntityId,
      sourceStreamEntryId,
      previousEntries: [],
      operations: [
        {
          ...addOperation({
            stateKey: "plan.appointments",
            sourceStreamEntryId,
          }),
          text: "Doctor appointment is pending; patient portal is down; dentist appointment is not booked.",
        },
      ],
    });

    expect(result.rejected).toEqual([]);
    expect(result.operations).toEqual([
      expect.objectContaining({
        type: "add",
        text: "Doctor appointment is pending; patient portal is down; dentist appointment is not booked.",
      }),
    ]);
  });

  it("accepts relationship claims grounded by roster relational slot evidence", () => {
    const audienceEntityId = createEntityId();
    const sourceStreamEntryId = createStreamEntryId();
    const slotId = createRelationalSlotId();
    const result = normalizeKeyedPatch({
      audienceEntityId,
      sourceStreamEntryId,
      previousEntries: [],
      participantRoster: {
        participants: [
          {
            entity_id: audienceEntityId,
            display_name: "Avery",
            known_relationships: ["spouse.name:Priya"],
            audience_role: "audience",
            relationship_source: `relational_slot:${slotId}`,
          },
        ],
        non_chat_subjects: [],
        unknown_or_uncertain: [],
      },
      operations: [
        {
          ...addOperation({
            stateKey: "plan.attendees",
            sourceStreamEntryId,
          }),
          text: "Priya is Avery's spouse for care-planning context.",
          relationship_claims: [
            relationshipClaim({
              object_text: "Priya",
              evidence_relational_slot_ids: [slotId],
            }),
          ],
        },
      ],
    });

    expect(result.rejected).toEqual([]);
    expect(result.operations).toEqual([
      expect.objectContaining({
        type: "add",
        text: "Priya is Avery's spouse for care-planning context.",
      }),
    ]);
  });

  it("accepts relationship claims grounded by trusted user-message stream evidence", () => {
    const audienceEntityId = createEntityId();
    const sourceStreamEntryId = createStreamEntryId();
    const result = normalizeKeyedPatch({
      audienceEntityId,
      sourceStreamEntryId,
      previousEntries: [],
      relationshipEvidenceStreamEntryTrust: (streamEntryId) =>
        streamEntryId === sourceStreamEntryId
          ? { allowed: true }
          : { allowed: false, reason: "missing" },
      operations: [
        {
          ...addOperation({
            stateKey: "plan.attendees",
            sourceStreamEntryId,
          }),
          text: "Use the parent constraint for care planning.",
          relationship_claims: [
            relationshipClaim({
              evidence_stream_entry_ids: [sourceStreamEntryId],
            }),
          ],
        },
      ],
    });

    expect(result.rejected).toEqual([]);
    expect(result.operations).toEqual([
      expect.objectContaining({
        type: "add",
        text: "Use the parent constraint for care planning.",
      }),
    ]);
  });

  it("rejects relationship claims grounded only by assistant stream evidence", () => {
    const audienceEntityId = createEntityId();
    const sourceStreamEntryId = createStreamEntryId();
    const result = normalizeKeyedPatch({
      audienceEntityId,
      sourceStreamEntryId,
      previousEntries: [],
      relationshipEvidenceStreamEntryTrust: () => ({
        allowed: false,
        reason: "not_user_msg",
      }),
      operations: [
        {
          ...addOperation({
            stateKey: "plan.attendees",
            sourceStreamEntryId,
          }),
          text: "Use the parent constraint for care planning.",
          relationship_claims: [
            relationshipClaim({
              evidence_stream_entry_ids: [sourceStreamEntryId],
            }),
          ],
        },
      ],
    });

    expect(result.operations).toEqual([]);
    expect(result.rejected).toEqual([
      expect.objectContaining({
        reason: "relationship_claim_ungrounded",
        relationshipClaims: [
          expect.objectContaining({
            evidence_stream_entry_ids: [sourceStreamEntryId],
          }),
        ],
        rejectedRelationshipClaimEvidenceStreamEntryIds: [
          {
            id: sourceStreamEntryId,
            reason: "not_user_msg",
          },
        ],
      }),
    ]);
  });

  it("bypasses relationship grounding when operation has no relationship claims", () => {
    const audienceEntityId = createEntityId();
    const sourceStreamEntryId = createStreamEntryId();
    const result = normalizeKeyedPatch({
      audienceEntityId,
      sourceStreamEntryId,
      previousEntries: [],
      operations: [
        {
          ...addOperation({
            stateKey: "plan.attendees",
            sourceStreamEntryId,
          }),
          text: "Care planning constraint is locked for the itinerary.",
        },
      ],
    });

    expect(result.rejected).toEqual([]);
    expect(result.operations).toEqual([
      expect.objectContaining({
        type: "add",
        text: "Care planning constraint is locked for the itinerary.",
      }),
    ]);
  });

  it("accepts supersede replacement text with no declared relationship claims", () => {
    const audienceEntityId = createEntityId();
    const sourceStreamEntryId = createStreamEntryId();
    const previous = makeEntry({
      audienceEntityId,
      sourceStreamEntryId,
      stateKey: "plan.care",
      text: "Legacy care planning entry.",
    });
    const result = normalizeKeyedPatch({
      audienceEntityId,
      sourceStreamEntryId,
      previousEntries: [previous],
      relationshipEvidenceStreamEntryTrust: (streamEntryId) =>
        streamEntryId === sourceStreamEntryId
          ? { allowed: true }
          : { allowed: false, reason: "missing" },
      operations: [
        {
          type: "supersede",
          id: previous.id,
          replacement: {
            state_key: "plan.care",
            kind: "locked",
            text: "Use the care-planning constraint.",
            source_stream_entry_ids: [sourceStreamEntryId],
          },
        },
      ],
    });

    expect(result.rejected).toEqual([]);
    expect(result.operations).toEqual([
      expect.objectContaining({
        type: "supersede",
        id: previous.id,
      }),
    ]);
  });

  it("accepts supersede replacement relationship claims grounded by replacement-entry evidence", () => {
    const audienceEntityId = createEntityId();
    const sourceStreamEntryId = createStreamEntryId();
    const previous = makeEntry({
      audienceEntityId,
      sourceStreamEntryId,
      stateKey: "plan.care",
      text: "Legacy care planning entry.",
    });
    const result = normalizeKeyedPatch({
      audienceEntityId,
      sourceStreamEntryId,
      previousEntries: [previous],
      relationshipEvidenceStreamEntryTrust: (streamEntryId) =>
        streamEntryId === sourceStreamEntryId
          ? { allowed: true }
          : { allowed: false, reason: "missing" },
      operations: [
        {
          type: "supersede",
          id: previous.id,
          replacement: {
            state_key: "plan.care",
            kind: "locked",
            text: "Use the parent constraint for care planning.",
            source_stream_entry_ids: [sourceStreamEntryId],
            relationship_claims: [
              relationshipClaim({
                evidence_stream_entry_ids: [sourceStreamEntryId],
              }),
            ],
          },
        },
      ],
    });

    expect(result.rejected).toEqual([]);
    expect(result.operations).toEqual([
      expect.objectContaining({
        type: "supersede",
        id: previous.id,
      }),
    ]);
  });

  it("rejects supersede replacement relationship claims with no accepted evidence", () => {
    const audienceEntityId = createEntityId();
    const sourceStreamEntryId = createStreamEntryId();
    const previous = makeEntry({
      audienceEntityId,
      sourceStreamEntryId,
      stateKey: "plan.care",
      text: "Legacy care planning entry.",
    });
    const result = normalizeKeyedPatch({
      audienceEntityId,
      sourceStreamEntryId,
      previousEntries: [previous],
      operations: [
        {
          type: "supersede",
          id: previous.id,
          replacement: {
            state_key: "plan.care",
            kind: "locked",
            text: "Use the parent constraint for care planning.",
            source_stream_entry_ids: [sourceStreamEntryId],
            relationship_claims: [relationshipClaim()],
          },
        },
      ],
    });

    expect(result.operations).toEqual([]);
    expect(result.rejected).toEqual([
      expect.objectContaining({
        reason: "relationship_claim_ungrounded",
        operationType: "supersede",
        operationIndex: 0,
        targetEntryId: previous.id,
      }),
    ]);
  });

  it("accepts update text grounded by trusted user-message stream evidence", () => {
    const audienceEntityId = createEntityId();
    const sourceStreamEntryId = createStreamEntryId();
    const previous = makeEntry({
      audienceEntityId,
      sourceStreamEntryId,
      stateKey: "plan.care",
      text: "Legacy care planning entry.",
    });
    const result = normalizeKeyedPatch({
      audienceEntityId,
      sourceStreamEntryId,
      previousEntries: [previous],
      relationshipEvidenceStreamEntryTrust: (streamEntryId) =>
        streamEntryId === sourceStreamEntryId
          ? { allowed: true }
          : { allowed: false, reason: "missing" },
      operations: [
        {
          type: "update",
          id: previous.id,
          state_key: "plan.care",
          text: "Use the parent constraint for care planning.",
          source_stream_entry_ids: [sourceStreamEntryId],
          relationship_claims: [
            relationshipClaim({
              evidence_stream_entry_ids: [sourceStreamEntryId],
            }),
          ],
        },
      ],
    });

    expect(result.rejected).toEqual([]);
    expect(result.operations).toEqual([
      expect.objectContaining({
        type: "update",
        id: previous.id,
        text: "Use the parent constraint for care planning.",
      }),
    ]);
  });

  it("accepts a same-key live add below the per-key cap", () => {
    const audienceEntityId = createEntityId();
    const sourceStreamEntryId = createStreamEntryId();
    const result = normalizeKeyedPatch({
      audienceEntityId,
      sourceStreamEntryId,
      previousEntries: [
        makeEntry({
          audienceEntityId,
          sourceStreamEntryId,
          stateKey: "plan.attendees",
          rank: 0,
        }),
      ],
      operations: [
        addOperation({
          stateKey: "plan.attendees",
          sourceStreamEntryId,
        }),
      ],
    });

    expect(result.rejected).toEqual([]);
    expect(result.operations).toEqual([
      expect.objectContaining({
        type: "add",
        state_key: "plan.attendees",
        kind: "live",
      }),
    ]);
  });

  it("rejects a near-duplicate never-seen state key", () => {
    const audienceEntityId = createEntityId();
    const sourceStreamEntryId = createStreamEntryId();
    const result = normalizeKeyedPatch({
      audienceEntityId,
      sourceStreamEntryId,
      previousEntries: [
        makeEntry({
          audienceEntityId,
          sourceStreamEntryId,
          stateKey: "observation.nora.video_call_repeated_question",
          rank: 0,
        }),
      ],
      operations: [
        addOperation({
          stateKey: "observation.nora.video_call_repeated_question_reconfirm",
          sourceStreamEntryId,
        }),
      ],
    });

    expect(result.operations).toEqual([]);
    expect(result.rejected).toEqual([
      expect.objectContaining({
        reason: "near_duplicate_state_key",
        operationType: "add",
        operationIndex: 0,
        stateKey: "observation.nora.video_call_repeated_question_reconfirm",
        similarStateKeys: ["observation.nora.video_call_repeated_question"],
        sharedStateKeyTokens: expect.arrayContaining([
          "observation",
          "nora",
          "video",
          "call",
          "repeated",
          "question",
        ]),
      }),
    ]);
  });

  it("rejects a near-duplicate key accumulated within an empty-registry patch", () => {
    const audienceEntityId = createEntityId();
    const sourceStreamEntryId = createStreamEntryId();
    const result = normalizeKeyedPatch({
      audienceEntityId,
      sourceStreamEntryId,
      previousEntries: [],
      operations: [
        addOperation({
          stateKey: "observation.nora.video_call_repeated_question",
          sourceStreamEntryId,
          newKeyReason: null,
        }),
        addOperation({
          stateKey: "observation.nora.video_call_repeated_question_reconfirm",
          sourceStreamEntryId,
          newKeyReason: null,
        }),
      ],
    });

    expect(result.operations).toEqual([
      expect.objectContaining({
        type: "add",
        state_key: "observation.nora.video_call_repeated_question",
      }),
    ]);
    expect(result.rejected).toEqual([
      expect.objectContaining({
        reason: "near_duplicate_state_key",
        operationType: "add",
        operationIndex: 1,
        stateKey: "observation.nora.video_call_repeated_question_reconfirm",
        similarStateKeys: ["observation.nora.video_call_repeated_question"],
      }),
    ]);
  });

  it("accepts a never-seen state key without new_key_reason when the active registry is empty", () => {
    const audienceEntityId = createEntityId();
    const sourceStreamEntryId = createStreamEntryId();
    const result = normalizeKeyedPatch({
      audienceEntityId,
      sourceStreamEntryId,
      previousEntries: [],
      operations: [
        addOperation({
          stateKey: "decision.architecture",
          sourceStreamEntryId,
          newKeyReason: null,
        }),
      ],
    });

    expect(result.rejected).toEqual([]);
    expect(result.operations).toEqual([
      expect.objectContaining({
        type: "add",
        state_key: "decision.architecture",
      }),
    ]);
  });

  it("rejects a never-seen state key without new_key_reason when the active registry is populated", () => {
    const audienceEntityId = createEntityId();
    const sourceStreamEntryId = createStreamEntryId();
    const result = normalizeKeyedPatch({
      audienceEntityId,
      sourceStreamEntryId,
      previousEntries: [
        makeEntry({
          audienceEntityId,
          sourceStreamEntryId,
          stateKey: "plan.attendees",
          rank: 0,
        }),
      ],
      operations: [
        addOperation({
          stateKey: "decision.architecture",
          sourceStreamEntryId,
          newKeyReason: null,
        }),
      ],
    });

    expect(result.operations).toEqual([]);
    expect(result.rejected).toEqual([
      expect.objectContaining({
        reason: "missing_new_key_reason",
        operationType: "add",
        operationIndex: 0,
        stateKey: "decision.architecture",
      }),
    ]);
  });

  it("does not require new_key_reason for an exact existing state key", () => {
    const audienceEntityId = createEntityId();
    const sourceStreamEntryId = createStreamEntryId();
    const result = normalizeKeyedPatch({
      audienceEntityId,
      sourceStreamEntryId,
      previousEntries: [
        makeEntry({
          audienceEntityId,
          sourceStreamEntryId,
          stateKey: "plan.attendees",
          rank: 0,
        }),
      ],
      operations: [
        addOperation({
          stateKey: "plan.attendees",
          sourceStreamEntryId,
          newKeyReason: null,
        }),
      ],
    });

    expect(result.rejected).toEqual([]);
    expect(result.operations).toEqual([
      expect.objectContaining({
        type: "add",
        state_key: "plan.attendees",
      }),
    ]);
  });

  it("rejects a same-key live add when the per-key cap is already reached", () => {
    const audienceEntityId = createEntityId();
    const sourceStreamEntryId = createStreamEntryId();
    const result = normalizeKeyedPatch({
      audienceEntityId,
      sourceStreamEntryId,
      previousEntries: [
        makeEntry({
          audienceEntityId,
          sourceStreamEntryId,
          stateKey: "plan.attendees",
          rank: 0,
        }),
        makeEntry({
          audienceEntityId,
          sourceStreamEntryId,
          stateKey: "plan.attendees",
          rank: 1,
        }),
      ],
      operations: [
        addOperation({
          stateKey: "plan.attendees",
          sourceStreamEntryId,
        }),
      ],
    });

    expect(result.operations).toEqual([]);
    expect(result.rejected).toEqual([
      expect.objectContaining({
        reason: "live_entry_cap_exceeded_for_key",
        operationType: "add",
        operationIndex: 0,
        stateKey: "plan.attendees",
        currentCount: 2,
        proposedCount: 3,
        maxLiveEntriesPerKey: 2,
      }),
    ]);
  });

  it("does not count legacy null-key entries toward a keyed live add", () => {
    const audienceEntityId = createEntityId();
    const sourceStreamEntryId = createStreamEntryId();
    const result = normalizeKeyedPatch({
      audienceEntityId,
      sourceStreamEntryId,
      previousEntries: [
        makeEntry({
          audienceEntityId,
          sourceStreamEntryId,
          stateKey: null,
          rank: 0,
        }),
        makeEntry({
          audienceEntityId,
          sourceStreamEntryId,
          stateKey: null,
          rank: 1,
        }),
      ],
      operations: [
        addOperation({
          stateKey: "plan.attendees",
          sourceStreamEntryId,
        }),
      ],
    });

    expect(result.rejected).toEqual([]);
    expect(result.operations).toEqual([
      expect.objectContaining({
        type: "add",
        state_key: "plan.attendees",
      }),
    ]);
  });

  it("rejects a live add when a locked entry already owns the state key", () => {
    const audienceEntityId = createEntityId();
    const sourceStreamEntryId = createStreamEntryId();
    const lockedEntry = makeEntry({
      audienceEntityId,
      sourceStreamEntryId,
      stateKey: "plan.attendees",
      kind: "locked",
      rank: 0,
    });
    const result = normalizeKeyedPatch({
      audienceEntityId,
      sourceStreamEntryId,
      previousEntries: [lockedEntry],
      operations: [
        addOperation({
          stateKey: "plan.attendees",
          sourceStreamEntryId,
        }),
      ],
    });

    expect(result.operations).toEqual([]);
    expect(result.rejected).toEqual([
      expect.objectContaining({
        reason: "locked_state_key_collision",
        operationType: "add",
        operationIndex: 0,
        stateKey: "plan.attendees",
        currentCount: 1,
        lockedEntryIds: [lockedEntry.id],
      }),
    ]);
  });

  it("accepts tentative same-key adds when live entries are already at cap", () => {
    const audienceEntityId = createEntityId();
    const sourceStreamEntryId = createStreamEntryId();
    const result = normalizeKeyedPatch({
      audienceEntityId,
      sourceStreamEntryId,
      previousEntries: [
        makeEntry({
          audienceEntityId,
          sourceStreamEntryId,
          stateKey: "plan.attendees",
          rank: 0,
        }),
        makeEntry({
          audienceEntityId,
          sourceStreamEntryId,
          stateKey: "plan.attendees",
          rank: 1,
        }),
      ],
      operations: [
        addOperation({
          stateKey: "plan.attendees",
          kind: "tentative",
          sourceStreamEntryId,
        }),
      ],
    });

    expect(result.rejected).toEqual([]);
    expect(result.operations).toEqual([
      expect.objectContaining({
        type: "add",
        state_key: "plan.attendees",
        kind: "tentative",
      }),
    ]);
  });
});
