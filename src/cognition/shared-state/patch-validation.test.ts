import { describe, expect, it } from "vitest";

import type {
  SharedStateArtifact,
  SharedStateEntry,
} from "../../memory/decision-artifacts/index.js";
import {
  createEntityId,
  createGoalId,
  createRelationalSlotId,
  createSharedStateEntryId,
  createStreamEntryId,
  type EntityId,
  type StreamEntryId,
} from "../../util/ids.js";
import type { EmitSharedStatePatch, SharedStateCanonicalizationCandidates } from "./schema.js";
import { allowedCanonicalizationIds, normalizePatch } from "./patch-validation.js";

const EMPTY_CANONICALIZES = {
  goal_ids: [],
  commitment_ids: [],
  action_ids: [],
  open_question_ids: [],
};

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
}) {
  const selfEntityId = createEntityId();
  const speakerEntityId = createEntityId();

  return normalizePatch({
    patch: {
      operations: input.operations,
    },
    previousArtifact: makeArtifact({
      audienceEntityId: input.audienceEntityId,
      entries: input.previousEntries,
    }),
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
  });
}

function addOperation(input: {
  stateKey: string;
  kind?: SharedStateEntry["kind"];
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
        goals: [{ id: goalId, text: "Route goal" }],
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
        goals: [{ id: goalId, text: "Route goal" }],
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
      allowedSourceStreamEntryIds: [
        originalSourceStreamEntryId,
        citationOnlySourceStreamEntryId,
      ],
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
  it("rejects protected relationship labels without grounding evidence", () => {
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
          text: "Mom's spouse is asleep.",
        },
      ],
    });

    expect(result.operations).toEqual([]);
    expect(result.rejected).toEqual([
      expect.objectContaining({
        reason: "relationship_label_ungrounded",
        operationType: "add",
        operationIndex: 0,
        protectedRelationshipLabels: ["spouse"],
        relationshipEvidenceRelationalSlotIds: [],
        relationshipEvidenceStreamEntryIds: [],
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

  it("accepts protected relationship labels grounded by roster relational slot evidence", () => {
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
          relationship_evidence_relational_slot_ids: [slotId],
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

  it("accepts protected relationship labels grounded by trusted user-message stream evidence", () => {
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
          relationship_evidence_stream_entry_ids: [sourceStreamEntryId],
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

  it("rejects protected relationship labels grounded only by assistant stream evidence", () => {
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
          relationship_evidence_stream_entry_ids: [sourceStreamEntryId],
        },
      ],
    });

    expect(result.operations).toEqual([]);
    expect(result.rejected).toEqual([
      expect.objectContaining({
        reason: "relationship_label_ungrounded",
        relationshipEvidenceStreamEntryIds: [sourceStreamEntryId],
        rejectedRelationshipEvidenceStreamEntryIds: [
          {
            id: sourceStreamEntryId,
            reason: "not_user_msg",
          },
        ],
      }),
    ]);
  });

  it("bypasses relationship grounding when operation text has no protected label", () => {
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

  it("accepts supersede replacement text grounded by operation-level evidence", () => {
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
          relationship_evidence_stream_entry_ids: [sourceStreamEntryId],
          replacement: {
            state_key: "plan.care",
            kind: "locked",
            text: "Use the parent constraint for care planning.",
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

  it("accepts supersede replacement text grounded by replacement-entry evidence", () => {
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
            relationship_evidence_stream_entry_ids: [sourceStreamEntryId],
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

  it("rejects supersede replacement text with no relationship evidence", () => {
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
          },
        },
      ],
    });

    expect(result.operations).toEqual([]);
    expect(result.rejected).toEqual([
      expect.objectContaining({
        reason: "relationship_label_ungrounded",
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
          relationship_evidence_stream_entry_ids: [sourceStreamEntryId],
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
