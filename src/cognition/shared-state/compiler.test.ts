import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import { FakeLLMClient } from "../../llm/test-support/fake-client.js";
import type { ActionRecord } from "../../memory/actions/index.js";
import { sharedStateMigrations } from "../../memory/decision-artifacts/migrations.js";
import { SharedStateRepository } from "../../memory/decision-artifacts/repository.js";
import type { SharedStateEntryKind } from "../../memory/decision-artifacts/types.js";
import {
  createEpisodeFixture,
  createOfflineTestHarness,
  createSemanticNodeFixture,
  TestEmbeddingClient,
} from "../../offline/test-support.js";
import { selfMigrations } from "../../memory/self/migrations.js";
import { GoalsRepository } from "../../memory/self/goals-repository.js";
import { resolveSemanticContext, toRetrievedSemantic } from "../../retrieval/semantic-retrieval.js";
import {
  composeMigrations,
  openDatabase,
  type SqliteDatabase,
} from "../../storage/sqlite/index.js";
import { FixedClock } from "../../util/clock.js";
import type { JsonValue } from "../../util/json-value.js";
import {
  createActionId,
  createCommitmentId,
  createEntityId,
  createGoalId,
  createOpenQuestionId,
  createRelationalSlotId,
  createStreamEntryId,
  type EntityId,
  type StreamEntryId,
} from "../../util/ids.js";
import type { EvidenceLedger, EvidenceLedgerEntry } from "../evidence-ledger/index.js";
import { renderSharedStateArtifact, renderEvidenceLedger } from "../evidence-ledger/index.js";
import { summarizeSemanticContext } from "../deliberation/prompt/retrieval.js";
import {
  advanceSharedStateCompileSkipAnchor,
  buildSharedStateLedgerPromptContext,
} from "../lifecycle/turn-phase-coordinator.js";
import type { TurnTraceData, TurnTraceEventName, TurnTracer } from "../tracing/tracer.js";
import {
  compileSharedStateArtifact,
  DECISION_ARTIFACT_TOOL_NAME,
  SHARED_STATE_SYSTEM_PROMPT,
  SHARED_STATE_TOOL_NAME,
} from "./compiler.js";
import type { SemanticRevisionVerdictCache } from "./reconciliation.js";

let defaultStateKeyCounter = 0;

function emitSharedStateArtifactPatchResponse(patch: unknown, toolName = SHARED_STATE_TOOL_NAME) {
  return emitRawSharedStateArtifactPatchResponse(withDefaultStateKeys(patch), toolName);
}

function withDefaultStateKeys(input: unknown): unknown {
  if (input === null || typeof input !== "object" || Array.isArray(input)) {
    return input;
  }

  const patch = input as { operations?: unknown };

  if (!Array.isArray(patch.operations)) {
    return input;
  }

  return {
    ...patch,
    operations: patch.operations.map((operation) => {
      if (operation === null || typeof operation !== "object" || Array.isArray(operation)) {
        return operation;
      }

      const typedOperation = operation as {
        type?: unknown;
        state_key?: unknown;
        replacement?: unknown;
      };

      if (typedOperation.type === "add" || typedOperation.type === "update") {
        return {
          state_key: `decision.fixture_${defaultStateKeyCounter++}`,
          ...(typedOperation.type === "add" ? { new_key_reason: "test fixture new key" } : {}),
          ...typedOperation,
        };
      }

      if (
        typedOperation.type === "supersede" &&
        typedOperation.replacement !== null &&
        typeof typedOperation.replacement === "object" &&
        !Array.isArray(typedOperation.replacement)
      ) {
        return {
          ...typedOperation,
          replacement: {
            state_key: `decision.fixture_${defaultStateKeyCounter++}`,
            ...(typedOperation.replacement as Record<string, unknown>),
          },
        };
      }

      return operation;
    }),
  };
}

function emitRawSharedStateArtifactPatchResponse(
  input: unknown,
  toolName = SHARED_STATE_TOOL_NAME,
) {
  return {
    text: "",
    input_tokens: 12,
    output_tokens: 8,
    stop_reason: "tool_use",
    tool_calls: [
      {
        id: "toolu_decision_patch",
        name: toolName,
        input,
      },
    ],
  };
}

function emitSemanticRevisionResponse(input: {
  verdicts: Array<{ node_id: string; verdict: "supersede" | "contradict" | "keep" | "uncertain" }>;
}) {
  return {
    text: "",
    input_tokens: 7,
    output_tokens: 5,
    stop_reason: "tool_use",
    tool_calls: [
      {
        id: "toolu_decision_artifact_semantic_revision",
        name: "EmitDecisionArtifactSemanticRevision",
        input: {
          verdicts: input.verdicts,
        },
      },
    ],
  };
}

function throwingResponse(): never {
  throw new Error("llm down");
}

function createTraceRecorder(): TurnTracer & {
  events: Array<{ event: TurnTraceEventName; data: TurnTraceData }>;
} {
  const events: Array<{ event: TurnTraceEventName; data: TurnTraceData }> = [];

  return {
    enabled: true,
    includePayloads: true,
    events,
    emit: vi.fn((event: TurnTraceEventName, data: TurnTraceData) => {
      events.push({ event, data });
    }),
  };
}

function expectSingleSuccessfulRepair(
  trace: Pick<ReturnType<typeof createTraceRecorder>, "events">,
): void {
  expect(
    trace.events.filter((event) => event.event === "shared_state.compile.repair_attempted"),
  ).toHaveLength(1);
  expect(
    trace.events.filter((event) => event.event === "shared_state.compile.repair_succeeded"),
  ).toHaveLength(1);
  expect(
    trace.events.filter((event) => event.event === "shared_state.compile.repair_failed"),
  ).toHaveLength(0);
}

function ledgerEntry(input: {
  streamEntryId: StreamEntryId;
  streamIndex: number;
  text: string;
  taint?: EvidenceLedgerEntry["taint"];
}): EvidenceLedgerEntry {
  return {
    id: `current_session_stream:${input.streamEntryId}`,
    source_type: "current_session_stream",
    session_scope: "current_session",
    actor: "user",
    trust_rank: 95,
    text: input.text,
    taint: input.taint ?? "none",
    stream_index: input.streamIndex,
  };
}

function evidenceLedger(entries: readonly EvidenceLedgerEntry[]): EvidenceLedger {
  return {
    transcriptIncluded: true,
    transcriptCompacted: false,
    originalTranscriptTokenEstimate: 0,
    compactedTranscriptEntryCount: 0,
    rawPreservedUserTranscriptEntryCount: 0,
    estimatedTokens: 0,
    sections: [
      {
        id: "current_session_transcript",
        label: "2. Current-Session Transcript",
        entries: [...entries],
      },
    ],
  };
}

describe("compileSharedStateArtifact", () => {
  let db: SqliteDatabase;
  let repository: SharedStateRepository;
  let clock: FixedClock;
  let audience: EntityId;
  let self: EntityId;
  let alice: EntityId;
  let currentStreamEntryId: StreamEntryId;

  beforeEach(() => {
    db = openDatabase(":memory:", {
      migrations: composeMigrations(sharedStateMigrations, selfMigrations),
    });
    clock = new FixedClock(2_000);
    repository = new SharedStateRepository({
      db,
      clock,
    });
    audience = createEntityId();
    self = createEntityId();
    alice = createEntityId();
    currentStreamEntryId = createStreamEntryId();
  });

  afterEach(() => {
    db.close();
  });

  function baseInput(llmClient: FakeLLMClient) {
    return {
      llmClient,
      model: "claude-haiku-test",
      repository,
      audienceEntityId: audience,
      selfEntityId: self,
      speakerEntityId: alice,
      participants: [{ entityId: alice, displayName: "Alice" }],
      currentUserMessage: "Madrid 3, SS 3, Seville 4, Granada 3 is locked.",
      currentUserStreamEntryId: currentStreamEntryId,
      promptVisibleLedger: "Commitments: route order confirmed.",
      allowedSourceStreamEntryIds: [currentStreamEntryId],
      clock,
      turnId: "turn_decision_artifact_test",
    };
  }

  function activeEntries() {
    return (repository.get(audience)?.entries ?? []).filter(
      (entry) => entry.superseded_by_id === null,
    );
  }

  it("frames the compiler prompt as shared audience state instead of planning state", () => {
    expect(SHARED_STATE_SYSTEM_PROMPT).toContain("canonical shared audience state");
    expect(SHARED_STATE_SYSTEM_PROMPT).toContain(
      "durable decisions and constraints for this audience",
    );
    expect(SHARED_STATE_SYSTEM_PROMPT).not.toContain("canonical planning state");
    expect(SHARED_STATE_SYSTEM_PROMPT).not.toContain("canonical shared planning state");
    expect(SHARED_STATE_SYSTEM_PROMPT).not.toContain("shared planning decision state");
    expect(SHARED_STATE_SYSTEM_PROMPT).toContain("Relationship label grounding");
    expect(SHARED_STATE_SYSTEM_PROMPT).toContain(
      "cite the supporting relational slot or stream entry",
    );
    expect(SHARED_STATE_SYSTEM_PROMPT).toContain(
      "Before proposing add, scan previous_artifact_summary.active_entries",
    );
    expect(SHARED_STATE_SYSTEM_PROMPT).toContain(
      "third or later live entry with the same state_key",
    );
    expect(SHARED_STATE_SYSTEM_PROMPT).toContain("Do not emit cosmetic or provenance-only updates");
    expect(SHARED_STATE_SYSTEM_PROMPT).toContain("Every add, update, and supersede replacement");
  });

  it("passes relational slots as structured compiler context", async () => {
    const slotSource = createStreamEntryId();
    const llmClient = new FakeLLMClient({
      responses: [emitSharedStateArtifactPatchResponse({ operations: [] })],
    });

    await compileSharedStateArtifact({
      ...baseInput(llmClient),
      allowedSourceStreamEntryIds: [currentStreamEntryId, slotSource],
      relationalSlotsContext: [
        {
          id: createRelationalSlotId(),
          subject_entity_id: alice,
          slot_key: "partner.name",
          value: "Priya",
          state: "established",
          evidence_stream_entry_ids: [slotSource],
          contradicted_by_stream_entry_ids: [],
          alternate_values: [],
        },
      ],
    });

    const content = JSON.parse(String(llmClient.requests[0]?.messages[0]?.content ?? "{}")) as {
      relational_slots_context?: unknown;
    };

    expect(content.relational_slots_context).toEqual([
      expect.objectContaining({
        subject_entity_id: alice,
        slot_key: "partner.name",
        value: "Priya",
        state: "established",
        evidence_stream_entry_ids: [slotSource],
      }),
    ]);
  });

  it("allows prompt-visible relational-slot evidence ids as shared-state citations", async () => {
    const slotSource = createStreamEntryId();
    const slotId = createRelationalSlotId();
    const llmClient = new FakeLLMClient({
      responses: [
        emitSharedStateArtifactPatchResponse({
          operations: [
            {
              type: "add",
              kind: "locked",
              text: "Priya is Avery's partner for care-planning context.",
              owner_entity_id: audience,
              source_stream_entry_ids: [slotSource],
              relationship_evidence_relational_slot_ids: [slotId],
            },
          ],
        }),
      ],
    });

    await compileSharedStateArtifact({
      ...baseInput(llmClient),
      allowedSourceStreamEntryIds: [currentStreamEntryId],
      participantRoster: {
        participants: [
          {
            entity_id: alice,
            display_name: "Avery",
            known_relationships: ["partner.name:Priya"],
            audience_role: "speaker",
            relationship_source: `relational_slot:${slotId}`,
          },
        ],
        non_chat_subjects: [],
        unknown_or_uncertain: [],
      },
      relationalSlotsContext: [
        {
          id: slotId,
          subject_entity_id: alice,
          slot_key: "partner.name",
          value: "Priya",
          state: "established",
          evidence_stream_entry_ids: [slotSource],
          contradicted_by_stream_entry_ids: [],
          alternate_values: [],
        },
      ],
    });

    expect(activeEntries()).toEqual([
      expect.objectContaining({
        text: "Priya is Avery's partner for care-planning context.",
        provenance_stream_entry_ids: [slotSource],
      }),
    ]);
  });

  it("adds a locked decision emitted by the LLM", async () => {
    const llmClient = new FakeLLMClient({
      responses: [
        emitSharedStateArtifactPatchResponse({
          operations: [
            {
              type: "add",
              kind: "locked",
              text: "Madrid 3 / SS 3 / Seville 4 / Granada 3",
              owner_entity_id: audience,
              source_stream_entry_ids: [currentStreamEntryId],
            },
          ],
        }),
      ],
    });

    await compileSharedStateArtifact(baseInput(llmClient));

    const artifact = repository.get(audience);
    expect(artifact?.entries).toHaveLength(1);
    expect(artifact?.entries[0]).toMatchObject({
      kind: "locked",
      text: "Madrid 3 / SS 3 / Seville 4 / Granada 3",
      owner_entity_id: audience,
      provenance_stream_entry_ids: [currentStreamEntryId],
    });
    expect(llmClient.requests[0]).toMatchObject({
      model: "claude-haiku-test",
      max_tokens: 8_000,
      budget: "decision-artifact-compiler",
      tool_choice: { type: "tool", name: SHARED_STATE_TOOL_NAME },
    });
  });

  it("accepts the legacy decision-artifact patch tool name as an alias", async () => {
    const llmClient = new FakeLLMClient({
      responses: [
        emitSharedStateArtifactPatchResponse(
          {
            operations: [
              {
                type: "add",
                kind: "live",
                text: "Avery is waiting on the clinic callback.",
                owner_entity_id: audience,
                source_stream_entry_ids: [currentStreamEntryId],
              },
            ],
          },
          DECISION_ARTIFACT_TOOL_NAME,
        ),
      ],
    });

    await compileSharedStateArtifact(baseInput(llmClient));

    expect(activeEntries()).toEqual([
      expect.objectContaining({
        kind: "live",
        text: "Avery is waiting on the clinic callback.",
      }),
    ]);
  });

  it("retains valid canonicalization ids in the normalized patch", async () => {
    const goalId = createGoalId();
    const commitmentId = createCommitmentId();
    const actionId = createActionId();
    const openQuestionId = createOpenQuestionId();
    const llmClient = new FakeLLMClient({
      responses: [
        emitSharedStateArtifactPatchResponse({
          operations: [
            {
              type: "add",
              kind: "locked",
              text: "Granada is locked for 3 nights",
              owner_entity_id: audience,
              source_stream_entry_ids: [currentStreamEntryId],
              canonicalizes: {
                goal_ids: [goalId],
                commitment_ids: [commitmentId],
                action_ids: [actionId],
                open_question_ids: [openQuestionId],
              },
            },
          ],
        }),
      ],
    });

    const patch = await compileSharedStateArtifact({
      ...baseInput(llmClient),
      canonicalizationCandidates: {
        goals: [{ id: goalId, text: "Lock Granada for 3 nights" }],
        commitments: [
          {
            id: commitmentId,
            text: "Keep the release window locked.",
            kind: "assistant_commitment",
            type: "promise",
            directive_family: "release_window",
            enforcement_class: "advisory",
          },
        ],
        actions: [{ id: actionId, text: "Track Granada decision" }],
        openQuestions: [{ id: openQuestionId, text: "Is Granada final?" }],
      },
    });

    expect(patch.operations[0]).toMatchObject({
      canonicalizes: {
        goal_ids: [goalId],
        commitment_ids: [commitmentId],
        action_ids: [actionId],
        open_question_ids: [openQuestionId],
      },
    });
    expect(repository.get(audience)?.entries[0]?.canonicalizes).toEqual({
      goal_ids: [goalId],
      commitment_ids: [commitmentId],
      action_ids: [actionId],
      open_question_ids: [openQuestionId],
    });
  });

  it("drops invalid canonicalization ids and reports them in reconciliation trace", async () => {
    const trace = createTraceRecorder();
    const unknownGoalId = createGoalId();
    const llmClient = new FakeLLMClient({
      responses: [
        emitSharedStateArtifactPatchResponse({
          operations: [
            {
              type: "add",
              kind: "locked",
              text: "Granada is locked for 3 nights",
              owner_entity_id: audience,
              source_stream_entry_ids: [currentStreamEntryId],
              canonicalizes: {
                goal_ids: ["goal_invalid", unknownGoalId],
              },
            },
          ],
        }),
      ],
    });

    const patch = await compileSharedStateArtifact({
      ...baseInput(llmClient),
      tracer: trace,
      canonicalizationCandidates: {
        goals: [],
      },
    });

    expect(patch.operations[0]).toMatchObject({
      canonicalizes: {
        goal_ids: [],
        commitment_ids: [],
        action_ids: [],
        open_question_ids: [],
      },
    });
    expect(trace.events).toContainEqual(
      expect.objectContaining({
        event: "shared_state.reconcile.completed",
        data: expect.objectContaining({
          goals_retired: 0,
          unknown_ids: [
            expect.objectContaining({
              channel: "goal",
              id: "goal_invalid",
              reason: "invalid_id",
            }),
            expect.objectContaining({
              channel: "goal",
              id: unknownGoalId,
              reason: "unknown_id",
            }),
          ] satisfies JsonValue,
        }),
      }),
    );
  });

  it("drops duplicate canonicalization ids across artifact operations before persisting", async () => {
    const trace = createTraceRecorder();
    const goalId = createGoalId();
    const llmClient = new FakeLLMClient({
      responses: [
        emitSharedStateArtifactPatchResponse({
          operations: [
            {
              type: "add",
              kind: "locked",
              text: "Granada is locked for 3 nights",
              owner_entity_id: audience,
              source_stream_entry_ids: [currentStreamEntryId],
              canonicalizes: {
                goal_ids: [goalId],
              },
            },
            {
              type: "add",
              kind: "locked",
              text: "Granada nights are canonical",
              owner_entity_id: audience,
              source_stream_entry_ids: [currentStreamEntryId],
              canonicalizes: {
                goal_ids: [goalId],
              },
            },
          ],
        }),
      ],
    });

    await compileSharedStateArtifact({
      ...baseInput(llmClient),
      tracer: trace,
      canonicalizationCandidates: {
        goals: [{ id: goalId, text: "Lock Granada for 3 nights" }],
      },
    });

    const entries = repository.get(audience)?.entries ?? [];
    expect(entries).toHaveLength(2);
    expect(entries[0]?.canonicalizes.goal_ids).toEqual([goalId]);
    expect(entries[1]?.canonicalizes.goal_ids).toEqual([]);
    expect(trace.events).toContainEqual(
      expect.objectContaining({
        event: "shared_state.reconcile.completed",
        data: expect.objectContaining({
          canonicalization_duplicates_dropped: [
            expect.objectContaining({
              kind: "locked",
              dropped_ids: expect.objectContaining({
                goal_ids: [goalId],
              }),
            }),
          ],
        }),
      }),
    );
  });

  it("lets a surviving entry keep canonicalization ids claimed by a pruned entry", async () => {
    const trace = createTraceRecorder();
    const goalsRepository = new GoalsRepository({
      db,
      clock,
    });
    const goal = goalsRepository.add({
      description: "Lock Granada for 3 nights",
      priority: 1,
      provenance: {
        kind: "online",
        process: "test",
      },
      audienceEntityId: audience,
      sourceStreamEntryIds: [currentStreamEntryId],
    });
    const llmClient = new FakeLLMClient({
      responses: [
        emitSharedStateArtifactPatchResponse({
          operations: [
            {
              type: "add",
              kind: "locked",
              text: "Older Granada lock duplicate",
              owner_entity_id: audience,
              source_stream_entry_ids: [currentStreamEntryId],
              canonicalizes: {
                goal_ids: [goal.id],
              },
            },
            {
              type: "add",
              kind: "locked",
              text: "Surviving Granada lock",
              owner_entity_id: audience,
              source_stream_entry_ids: [currentStreamEntryId],
              canonicalizes: {
                goal_ids: [goal.id],
              },
            },
          ],
        }),
      ],
    });

    await compileSharedStateArtifact({
      ...baseInput(llmClient),
      tracer: trace,
      canonicalizationCandidates: {
        goals: [{ id: goal.id, text: goal.description }],
      },
      reconciliation: {
        goalsRepository,
      },
      lifecycle: {
        maxActiveEntries: 1,
      },
    });

    const entries = repository.get(audience)?.entries ?? [];
    expect(entries).toHaveLength(1);
    expect(entries[0]).toMatchObject({
      text: "Surviving Granada lock",
      canonicalizes: {
        goal_ids: [goal.id],
      },
    });
    expect(goalsRepository.get(goal.id)).toMatchObject({
      status: "done",
      canonicalized_by_artifact_entry_id: entries[0]?.id,
    });
    expect(trace.events).toContainEqual(
      expect.objectContaining({
        event: "shared_state.reconcile.completed",
        data: expect.objectContaining({
          canonicalization_duplicates_dropped: [],
        }),
      }),
    );
  });

  it("canonicalizes a locked entry and retires an active goal after upsert", async () => {
    const goalsRepository = new GoalsRepository({
      db,
      clock,
    });
    const goal = goalsRepository.add({
      description: "Lock Granada for 3 nights",
      priority: 1,
      provenance: {
        kind: "online",
        process: "test",
      },
      audienceEntityId: audience,
      sourceStreamEntryIds: [currentStreamEntryId],
    });
    const llmClient = new FakeLLMClient({
      responses: [
        emitSharedStateArtifactPatchResponse({
          operations: [
            {
              type: "add",
              kind: "locked",
              text: "Granada is locked for 3 nights",
              owner_entity_id: audience,
              source_stream_entry_ids: [currentStreamEntryId],
              canonicalizes: {
                goal_ids: [goal.id],
              },
            },
          ],
        }),
      ],
    });

    await compileSharedStateArtifact({
      ...baseInput(llmClient),
      canonicalizationCandidates: {
        goals: [{ id: goal.id, text: goal.description }],
      },
      reconciliation: {
        goalsRepository,
      },
    });

    const artifactEntry = repository.get(audience)?.entries[0];
    expect(goalsRepository.get(goal.id)).toMatchObject({
      status: "done",
      canonicalized_by_artifact_entry_id: artifactEntry?.id,
    });
  });

  it("touches canonicalized actions with the supplied global turn counter", async () => {
    const actionId = createActionId();
    const update = vi.fn();
    const action = {
      id: actionId,
      description: "Follow up with the clinic",
      actor: "user",
      audience_entity_id: audience,
      goal_id: null,
      open_question_id: null,
      state: "committed_to_do",
      confidence: 0.8,
      provenance_episode_ids: [],
      provenance_stream_entry_ids: [currentStreamEntryId],
      created_at: 1_000,
      updated_at: 1_000,
      considering_at: null,
      committed_at: 1_000,
      scheduled_at: null,
      completed_at: null,
      not_done_at: null,
      expired_at: null,
      archived_at: null,
      unknown_at: null,
      canonicalized_by_artifact_entry_id: null,
      session_scope: null,
      session_anchor_id: null,
      last_referenced_at_ms: null,
      last_referenced_turn_counter: 2,
      last_referenced_turn_global: null,
    } satisfies ActionRecord;
    const actionRepository = {
      get: vi.fn(() => action),
      update,
    };
    const llmClient = new FakeLLMClient({
      responses: [
        emitSharedStateArtifactPatchResponse({
          operations: [
            {
              type: "add",
              kind: "locked",
              text: "Alice owns the clinic callback follow-up",
              owner_entity_id: audience,
              source_stream_entry_ids: [currentStreamEntryId],
              canonicalizes: {
                action_ids: [actionId],
              },
            },
          ],
        }),
      ],
    });

    await compileSharedStateArtifact({
      ...baseInput(llmClient),
      turnCounter: 42,
      canonicalizationCandidates: {
        actions: [{ id: actionId, text: "Follow up with the clinic" }],
      },
      reconciliation: {
        actionRepository,
      },
    });

    expect(update).toHaveBeenCalledWith(
      actionId,
      expect.objectContaining({
        last_referenced_turn_counter: 42,
        last_referenced_turn_global: 42,
      }),
      { skipSideEffects: true },
    );
  });

  it("runs semantic belief revision from an accepted locked artifact entry", async () => {
    const embeddingClient = new TestEmbeddingClient(
      new Map([["Project runtime is Node 22", [1, 0, 0, 0]]]),
    );
    const harness = await createOfflineTestHarness({
      clock,
      embeddingClient,
    });
    const artifactRepository = new SharedStateRepository({
      db: harness.db,
      clock,
    });

    try {
      const sourceEpisode = createEpisodeFixture({
        audience_entity_id: audience,
        shared: false,
      });
      await harness.episodicRepository.insert(sourceEpisode);
      const staleNode = await harness.semanticNodeRepository.insert(
        createSemanticNodeFixture(
          {
            label: "Project runtime is Node 20",
            description: "The project runtime is Node 20.",
            source_episode_ids: [sourceEpisode.id],
          },
          [1, 0, 0, 0],
        ),
      );
      const llmClient = new FakeLLMClient({
        responses: [
          emitSharedStateArtifactPatchResponse({
            operations: [
              {
                type: "add",
                kind: "locked",
                text: "Project runtime is Node 22",
                owner_entity_id: audience,
                source_stream_entry_ids: [currentStreamEntryId],
              },
            ],
          }),
          emitSemanticRevisionResponse({
            verdicts: [
              {
                node_id: staleNode.id,
                verdict: "supersede",
              },
            ],
          }),
        ],
      });

      await compileSharedStateArtifact({
        ...baseInput(llmClient),
        repository: artifactRepository,
        currentUserMessage: "Lock the project runtime as Node 22; the Node 20 note is stale.",
        promptVisibleLedger: "Semantic context says the project runtime is Node 20.",
        semanticBeliefRevision: {
          semanticNodeRepository: harness.semanticNodeRepository,
          episodicRepository: harness.episodicRepository,
          embeddingClient,
          model: "semantic-revision-test",
        },
      });

      const artifactEntry = artifactRepository.get(audience)?.entries[0];
      const updatedStaleNode = await harness.semanticNodeRepository.get(staleNode.id);
      expect(artifactEntry).toMatchObject({
        kind: "locked",
        text: "Project runtime is Node 22",
      });
      expect(updatedStaleNode).toMatchObject({
        status: "superseded",
        corrected_by: currentStreamEntryId,
        superseded_at: 2_000,
      });

      const semantic = toRetrievedSemantic(
        await resolveSemanticContext(
          "Project runtime",
          {
            audienceEntityId: audience,
            queryVector: Float32Array.from([1, 0, 0, 0]),
            graphWalkDepth: 1,
            maxGraphNodes: 4,
          },
          {
            embeddingClient,
            episodicRepository: harness.episodicRepository,
            semanticNodeRepository: harness.semanticNodeRepository,
            semanticGraph: harness.semanticGraph,
          },
        ),
      );
      const retrievedStaleNode = semantic.matched_nodes.find((node) => node.id === staleNode.id);
      const summary = summarizeSemanticContext(semantic, 2_000, clock.now());

      expect(retrievedStaleNode).toMatchObject({
        status: "superseded",
        status_retrieval_multiplier: 0.5,
      });
      expect(summary).toContain("[status=superseded, t=2000]");
      expect(summary).not.toContain(currentStreamEntryId);
    } finally {
      await harness.cleanup();
    }
  });

  it.each(["search", "judge"] as const)(
    "accepts the artifact when semantic revision %s fails",
    async (failure) => {
      const embeddingClient = new TestEmbeddingClient(
        new Map([["Project runtime is Node 22", [1, 0, 0, 0]]]),
      );
      const harness = await createOfflineTestHarness({
        clock,
        embeddingClient,
      });
      const artifactRepository = new SharedStateRepository({
        db: harness.db,
        clock,
      });
      const trace = createTraceRecorder();

      try {
        const sourceEpisode = createEpisodeFixture({
          audience_entity_id: audience,
          shared: false,
        });
        await harness.episodicRepository.insert(sourceEpisode);
        const staleNode = await harness.semanticNodeRepository.insert(
          createSemanticNodeFixture(
            {
              label: "Project runtime is Node 20",
              description: "The project runtime is Node 20.",
              source_episode_ids: [sourceEpisode.id],
            },
            [1, 0, 0, 0],
          ),
        );
        const llmClient = new FakeLLMClient({
          responses: [
            emitSharedStateArtifactPatchResponse({
              operations: [
                {
                  type: "add",
                  kind: "locked",
                  text: "Project runtime is Node 22",
                  owner_entity_id: audience,
                  source_stream_entry_ids: [currentStreamEntryId],
                },
              ],
            }),
            ...(failure === "judge" ? [throwingResponse] : []),
          ],
        });
        const semanticNodeRepository = {
          searchByVector:
            failure === "search"
              ? vi.fn(async () => {
                  throw new Error("semantic vector search failed");
                })
              : harness.semanticNodeRepository.searchByVector.bind(harness.semanticNodeRepository),
          markSuperseded: harness.semanticNodeRepository.markSuperseded.bind(
            harness.semanticNodeRepository,
          ),
          markContradicted: harness.semanticNodeRepository.markContradicted.bind(
            harness.semanticNodeRepository,
          ),
        };

        await compileSharedStateArtifact({
          ...baseInput(llmClient),
          repository: artifactRepository,
          currentUserMessage: "Lock the project runtime as Node 22.",
          semanticBeliefRevision: {
            semanticNodeRepository,
            episodicRepository: harness.episodicRepository,
            embeddingClient,
            model: "semantic-revision-test",
          },
          tracer: trace,
        });

        expect(artifactRepository.get(audience)?.entries[0]).toMatchObject({
          kind: "locked",
          text: "Project runtime is Node 22",
        });
        await expect(harness.semanticNodeRepository.get(staleNode.id)).resolves.toMatchObject({
          status: "active",
        });
        expect(trace.events).toContainEqual(
          expect.objectContaining({
            event: "semantic_revision.degraded",
          }),
        );
      } finally {
        await harness.cleanup();
      }
    },
  );

  it("traces top-level semantic revision fail-open errors separately from no-op revision", async () => {
    const embeddingClient = new TestEmbeddingClient(
      new Map([["Project runtime is Node 22", [1, 0, 0, 0]]]),
    );
    const harness = await createOfflineTestHarness({
      clock,
      embeddingClient,
    });
    const artifactRepository = new SharedStateRepository({
      db: harness.db,
      clock,
    });
    const trace = createTraceRecorder();
    const throwingVerdictCache = {
      get() {
        throw new Error("semantic revision cache failed");
      },
      set: vi.fn(),
      clear: vi.fn(),
      size: 0,
    } as unknown as SemanticRevisionVerdictCache;

    try {
      const sourceEpisode = createEpisodeFixture({
        audience_entity_id: audience,
        shared: false,
      });
      await harness.episodicRepository.insert(sourceEpisode);
      await harness.semanticNodeRepository.insert(
        createSemanticNodeFixture(
          {
            label: "Project runtime is Node 20",
            description: "The project runtime is Node 20.",
            source_episode_ids: [sourceEpisode.id],
          },
          [1, 0, 0, 0],
        ),
      );
      const llmClient = new FakeLLMClient({
        responses: [
          emitSharedStateArtifactPatchResponse({
            operations: [
              {
                type: "add",
                kind: "locked",
                text: "Project runtime is Node 22",
                owner_entity_id: audience,
                source_stream_entry_ids: [currentStreamEntryId],
              },
            ],
          }),
        ],
      });

      await compileSharedStateArtifact({
        ...baseInput(llmClient),
        repository: artifactRepository,
        currentUserMessage: "Lock the project runtime as Node 22.",
        semanticBeliefRevision: {
          semanticNodeRepository: harness.semanticNodeRepository,
          episodicRepository: harness.episodicRepository,
          embeddingClient,
          model: "semantic-revision-test",
          verdictCache: throwingVerdictCache,
        },
        tracer: trace,
      });

      expect(artifactRepository.get(audience)?.entries[0]).toMatchObject({
        kind: "locked",
        text: "Project runtime is Node 22",
      });
      expect(trace.events).toContainEqual(
        expect.objectContaining({
          event: "shared_state.semantic_revision.degraded",
          data: expect.objectContaining({
            reason: "semantic revision cache failed",
            skipped_due_to_error: 1,
          }),
        }),
      );
    } finally {
      await harness.cleanup();
    }
  });

  it("retries stranded canonicalizes on an otherwise no-op compile", async () => {
    const trace = createTraceRecorder();
    const goalsRepository = new GoalsRepository({
      db,
      clock,
    });
    const strandedSource = createStreamEntryId();
    const goal = goalsRepository.add({
      description: "Lock Granada for 3 nights",
      priority: 1,
      provenance: {
        kind: "online",
        process: "test",
      },
      audienceEntityId: audience,
      sourceStreamEntryIds: [strandedSource],
    });
    repository.upsert(
      audience,
      [
        {
          type: "add",
          state_key: "decision.granada_duration",
          kind: "locked",
          text: "Granada is locked for 3 nights",
          provenance_stream_entry_ids: [strandedSource],
          canonicalizes: {
            goal_ids: [goal.id],
            commitment_ids: [],
            action_ids: [],
            open_question_ids: [],
          },
        },
      ],
      {
        lastCompiledStreamEntryId: strandedSource,
      },
    );
    const llmClient = new FakeLLMClient({
      responses: [emitSharedStateArtifactPatchResponse({ operations: [] })],
    });

    await compileSharedStateArtifact({
      ...baseInput(llmClient),
      tracer: trace,
      allowedSourceStreamEntryIds: [strandedSource, currentStreamEntryId],
      reconciliation: {
        goalsRepository,
      },
    });

    const artifactEntry = repository.get(audience)?.entries[0];
    expect(goalsRepository.get(goal.id)).toMatchObject({
      status: "done",
      canonicalized_by_artifact_entry_id: artifactEntry?.id,
    });
    expect(trace.events).toContainEqual(
      expect.objectContaining({
        event: "shared_state.reconcile.completed",
        data: expect.objectContaining({
          goals_retired: 1,
          current_operation_canonicalization_count: 0,
          retried_stranded_canonicalization_count: 1,
          retry_unsettled_summary: expect.objectContaining({
            unsettled_goal_count: 1,
            unsettled_total_count: 1,
          }) as JsonValue,
        }),
      }),
    );
  });

  it("reconciles retried stranded handles and new canonicalizes in one compile", async () => {
    const trace = createTraceRecorder();
    const goalsRepository = new GoalsRepository({
      db,
      clock,
    });
    const strandedSource = createStreamEntryId();
    const strandedGoal = goalsRepository.add({
      description: "Lock Granada for 3 nights",
      priority: 1,
      provenance: {
        kind: "online",
        process: "test",
      },
      audienceEntityId: audience,
      sourceStreamEntryIds: [strandedSource],
    });
    const newGoal = goalsRepository.add({
      description: "Lock Seville for 4 nights",
      priority: 1,
      provenance: {
        kind: "online",
        process: "test",
      },
      audienceEntityId: audience,
      sourceStreamEntryIds: [currentStreamEntryId],
    });
    repository.upsert(
      audience,
      [
        {
          type: "add",
          state_key: "decision.granada_duration",
          kind: "locked",
          text: "Granada is locked for 3 nights",
          provenance_stream_entry_ids: [strandedSource],
          canonicalizes: {
            goal_ids: [strandedGoal.id],
            commitment_ids: [],
            action_ids: [],
            open_question_ids: [],
          },
        },
      ],
      {
        lastCompiledStreamEntryId: strandedSource,
      },
    );
    const llmClient = new FakeLLMClient({
      responses: [
        emitSharedStateArtifactPatchResponse({
          operations: [
            {
              type: "add",
              kind: "locked",
              text: "Seville is locked for 4 nights",
              owner_entity_id: audience,
              source_stream_entry_ids: [currentStreamEntryId],
              canonicalizes: {
                goal_ids: [newGoal.id],
              },
            },
          ],
        }),
      ],
    });

    await compileSharedStateArtifact({
      ...baseInput(llmClient),
      tracer: trace,
      allowedSourceStreamEntryIds: [strandedSource, currentStreamEntryId],
      canonicalizationCandidates: {
        goals: [{ id: newGoal.id, text: newGoal.description }],
      },
      reconciliation: {
        goalsRepository,
      },
    });

    expect(goalsRepository.get(strandedGoal.id)).toMatchObject({
      status: "done",
    });
    expect(goalsRepository.get(newGoal.id)).toMatchObject({
      status: "done",
    });
    expect(trace.events).toContainEqual(
      expect.objectContaining({
        event: "shared_state.reconcile.completed",
        data: expect.objectContaining({
          goals_retired: 2,
          current_operation_canonicalization_count: 1,
          retried_stranded_canonicalization_count: 1,
        }),
      }),
    );
  });

  it("drops canonicalizes emitted on non-locked entries before reconciliation", async () => {
    const trace = createTraceRecorder();
    const goalsRepository = new GoalsRepository({
      db,
      clock,
    });
    const goal = goalsRepository.add({
      description: "Lock Granada for 3 nights",
      priority: 1,
      provenance: {
        kind: "online",
        process: "test",
      },
      audienceEntityId: audience,
      sourceStreamEntryIds: [currentStreamEntryId],
    });
    const llmClient = new FakeLLMClient({
      responses: [
        emitSharedStateArtifactPatchResponse({
          operations: [
            {
              type: "add",
              kind: "live",
              text: "Granada is under discussion",
              owner_entity_id: audience,
              source_stream_entry_ids: [currentStreamEntryId],
              canonicalizes: {
                goal_ids: [goal.id],
              },
            },
          ],
        }),
      ],
    });

    await compileSharedStateArtifact({
      ...baseInput(llmClient),
      tracer: trace,
      canonicalizationCandidates: {
        goals: [{ id: goal.id, text: goal.description }],
      },
      reconciliation: {
        goalsRepository,
      },
    });

    expect(repository.get(audience)?.entries[0]).toMatchObject({
      kind: "live",
      canonicalizes: {
        goal_ids: [],
        commitment_ids: [],
        action_ids: [],
        open_question_ids: [],
      },
    });
    expect(goalsRepository.get(goal.id)).toMatchObject({
      status: "active",
      canonicalized_by_artifact_entry_id: null,
    });
    expect(trace.events).toContainEqual(
      expect.objectContaining({
        event: "shared_state.compile.completed",
        data: expect.objectContaining({
          canonicalizes_rejected_non_locked: [
            {
              operation_index: 0,
              kind: "live",
              dropped_ids: {
                goal_ids: [goal.id],
                commitment_ids: [],
                action_ids: [],
                open_question_ids: [],
              },
            },
          ] satisfies JsonValue,
        }),
      }),
    );
  });

  it("rejects an invalid owner entity id with a traced reason", async () => {
    const trace = createTraceRecorder();
    const llmClient = new FakeLLMClient({
      responses: [
        emitSharedStateArtifactPatchResponse({
          operations: [
            {
              type: "add",
              kind: "locked",
              text: "Madrid 3 / SS 3 / Seville 4 / Granada 3",
              owner_entity_id: createEntityId(),
              source_stream_entry_ids: [currentStreamEntryId],
            },
          ],
        }),
      ],
    });

    await compileSharedStateArtifact({
      ...baseInput(llmClient),
      tracer: trace,
    });

    expect(repository.get(audience)).toBeNull();
    expect(trace.events).toContainEqual(
      expect.objectContaining({
        event: "shared_state.compile.completed",
        data: expect.objectContaining({
          rejectedCount: 1,
          rejectionReasons: ["invalid_owner_entity_id"] satisfies JsonValue,
          applied: false,
        }),
      }),
    );
  });

  it("supersedes an existing entry with a replacement entry", async () => {
    const firstSource = createStreamEntryId();
    const initial = repository.upsert(
      audience,
      [
        {
          type: "add",
          state_key: "decision.route",
          kind: "locked",
          text: "Locked route order: Madrid 3 / SS 3 / Seville 4 / Granada 2",
          provenance_stream_entry_ids: [firstSource],
        },
      ],
      {
        lastCompiledStreamEntryId: firstSource,
      },
    );
    const oldEntryId = initial?.entries[0]?.id;

    expect(oldEntryId).toBeDefined();

    const llmClient = new FakeLLMClient({
      responses: [
        emitSharedStateArtifactPatchResponse({
          operations: [
            {
              type: "supersede",
              id: oldEntryId!,
              replacement: {
                kind: "locked",
                text: "Locked route order: Madrid 3 / SS 3 / Seville 4 / Granada 3",
                owner_entity_id: audience,
                source_stream_entry_ids: [currentStreamEntryId],
              },
              source_stream_entry_ids: [currentStreamEntryId],
            },
          ],
        }),
      ],
    });

    await compileSharedStateArtifact({
      ...baseInput(llmClient),
      allowedSourceStreamEntryIds: [firstSource, currentStreamEntryId],
    });

    const artifact = repository.get(audience);
    const oldEntry = artifact?.entries.find((entry) => entry.id === oldEntryId);
    const replacement = artifact?.entries.find((entry) => entry.id !== oldEntryId);

    expect(artifact?.entries).toHaveLength(2);
    expect(oldEntry?.superseded_by_id).toBe(replacement?.id);
    expect(replacement).toMatchObject({
      kind: "locked",
      text: "Locked route order: Madrid 3 / SS 3 / Seville 4 / Granada 3",
      provenance_stream_entry_ids: [currentStreamEntryId],
    });
  });

  it("skips gracefully when the LLM call fails", async () => {
    const onDegraded = vi.fn();
    const llmClient = new FakeLLMClient({
      responses: [throwingResponse],
    });

    await compileSharedStateArtifact({
      ...baseInput(llmClient),
      onDegraded,
    });

    expect(repository.get(audience)).toBeNull();
    expect(onDegraded).toHaveBeenCalledWith("llm_failed", expect.any(Error));
  });

  it("advances compile metadata and record version for a no-op compile", async () => {
    const initial = repository.upsert(audience, [
      {
        type: "add",
        state_key: "decision.live",
        kind: "live",
        text: "Question: Granada pacing",
        provenance_stream_entry_ids: [currentStreamEntryId],
      },
    ]);
    const llmClient = new FakeLLMClient({
      responses: [emitSharedStateArtifactPatchResponse({ operations: [] })],
    });

    await compileSharedStateArtifact(baseInput(llmClient));

    expect(repository.get(audience)?.record_version).toBe((initial?.record_version ?? 0) + 1);
    expect(repository.get(audience)?.last_compiled_stream_entry_id).toBe(currentStreamEntryId);
  });

  it("succeeds without degradation when all operations are dropped empty updates", async () => {
    const trace = createTraceRecorder();
    const onDegraded = vi.fn();
    const initial = repository.upsert(audience, [
      {
        type: "add",
        state_key: "decision.route",
        kind: "live",
        text: "Madrid 3 is locked.",
        provenance_stream_entry_ids: [currentStreamEntryId],
      },
      {
        type: "add",
        state_key: "decision.pacing",
        kind: "locked",
        text: "Keep the first leg short.",
        provenance_stream_entry_ids: [currentStreamEntryId],
      },
      {
        type: "add",
        state_key: "decision.rooms",
        kind: "pending",
        text: "Room assignments are unresolved.",
        provenance_stream_entry_ids: [currentStreamEntryId],
      },
    ]);
    const entries = repository.get(audience)?.entries ?? [];
    const llmClient = new FakeLLMClient({
      responses: [
        emitSharedStateArtifactPatchResponse({
          operations: entries.map((entry) => ({
            type: "update",
            id: entry.id,
            state_key: entry.state_key,
            source_stream_entry_ids: [currentStreamEntryId],
          })),
        }),
      ],
    });

    const patch = await compileSharedStateArtifact({
      ...baseInput(llmClient),
      tracer: trace,
      onDegraded,
    });

    expect(patch.operations).toEqual([]);
    expect(onDegraded).not.toHaveBeenCalled();
    expect(repository.get(audience)?.record_version).toBe((initial?.record_version ?? 0) + 1);
    expect(repository.get(audience)?.entries).toHaveLength(3);
    expect(
      trace.events.filter((event) => event.event === "shared_state.compile.empty_update_dropped"),
    ).toHaveLength(3);
    expect(
      trace.events.some(
        (event) =>
          event.event === "shared_state.compile.degraded" && event.data.reason === "invalid_patch",
      ),
    ).toBe(false);
    expect(trace.events).toContainEqual(
      expect.objectContaining({
        event: "shared_state.compile.completed",
        data: expect.objectContaining({
          applied: false,
          operationCount: 0,
          rejectedCount: 0,
          update_checked_for_empty_count: 3,
          empty_update_attempted_count: 3,
          empty_update_dropped_count: 3,
          empty_update_repaired_count: 0,
        }),
      }),
    );
  });

  it("drops empty updates while applying other valid operations", async () => {
    const trace = createTraceRecorder();
    repository.upsert(audience, [
      {
        type: "add",
        state_key: "decision.route",
        kind: "live",
        text: "Madrid 3 is locked.",
        provenance_stream_entry_ids: [currentStreamEntryId],
      },
    ]);
    const existing = repository.get(audience)?.entries[0];
    expect(existing).toBeDefined();
    const llmClient = new FakeLLMClient({
      responses: [
        emitSharedStateArtifactPatchResponse({
          operations: [
            {
              type: "update",
              id: existing?.id,
              state_key: existing?.state_key,
              text: existing?.text,
              source_stream_entry_ids: [currentStreamEntryId],
            },
            {
              type: "add",
              state_key: "decision.hotel",
              new_key_reason: "test fixture hotel decision",
              kind: "live",
              text: "Hotel selection is still open.",
              source_stream_entry_ids: [currentStreamEntryId],
            },
          ],
        }),
      ],
    });

    const patch = await compileSharedStateArtifact({
      ...baseInput(llmClient),
      tracer: trace,
    });

    expect(patch.operations).toEqual([
      expect.objectContaining({
        type: "add",
        state_key: "decision.hotel",
      }),
    ]);
    expect(activeEntries()).toHaveLength(2);
    expect(
      trace.events.filter((event) => event.event === "shared_state.compile.empty_update_dropped"),
    ).toEqual([
      expect.objectContaining({
        data: expect.objectContaining({
          operation_index: 0,
          operation_id: existing?.id,
          state_key: "decision.route",
          field_presence: {
            kind: false,
            text: true,
            owner_entity_id: false,
            canonicalizes: false,
          },
        }),
      }),
    ]);
    expect(trace.events).toContainEqual(
      expect.objectContaining({
        event: "shared_state.compile.completed",
        data: expect.objectContaining({
          applied: true,
          operationCount: 1,
          rejectedCount: 0,
          update_checked_for_empty_count: 1,
          empty_update_attempted_count: 1,
          empty_update_dropped_count: 1,
          empty_update_repaired_count: 0,
        }),
      }),
    );
  });

  it("creates an empty artifact on a first no-op compile so later turns can delta from it", async () => {
    const firstSource = createStreamEntryId();
    const secondSource = createStreamEntryId();
    const llmClient = new FakeLLMClient({
      responses: [emitSharedStateArtifactPatchResponse({ operations: [] })],
    });

    await compileSharedStateArtifact({
      ...baseInput(llmClient),
      currentUserStreamEntryId: firstSource,
      allowedSourceStreamEntryIds: [firstSource],
    });

    const artifact = repository.get(audience);
    const ledger = evidenceLedger([
      ledgerEntry({ streamEntryId: firstSource, streamIndex: 0, text: "first no-op turn" }),
      ledgerEntry({ streamEntryId: secondSource, streamIndex: 1, text: "second turn" }),
    ]);
    const context = buildSharedStateLedgerPromptContext({
      ledger,
      previousArtifact: artifact,
      fullPromptVisibleLedger: renderEvidenceLedger(ledger) ?? "",
      enabled: true,
      minTailPerSection: 1,
    });

    expect(artifact).toMatchObject({
      record_version: 1,
      last_compiled_stream_entry_id: firstSource,
      entries: [],
    });
    expect(context.ledgerMode).toBe("delta");
    expect(context.promptVisibleLedger).not.toContain("first no-op turn");
    expect(context.promptVisibleLedger).toContain("second turn");
  });

  it("advances no-op compile anchors so the next ledger delta starts after the no-op turn", async () => {
    const firstSource = createStreamEntryId();
    const secondSource = createStreamEntryId();
    const thirdSource = createStreamEntryId();
    const llmClient = new FakeLLMClient({
      responses: [
        emitSharedStateArtifactPatchResponse({
          operations: [
            {
              type: "add",
              kind: "live",
              text: "Live shared-state decision",
              owner_entity_id: audience,
              source_stream_entry_ids: [firstSource],
            },
          ],
        }),
        emitSharedStateArtifactPatchResponse({ operations: [] }),
      ],
    });

    await compileSharedStateArtifact({
      ...baseInput(llmClient),
      currentUserStreamEntryId: firstSource,
      allowedSourceStreamEntryIds: [firstSource],
    });
    const afterFirst = repository.get(audience);

    await compileSharedStateArtifact({
      ...baseInput(llmClient),
      currentUserStreamEntryId: secondSource,
      allowedSourceStreamEntryIds: [firstSource, secondSource],
    });

    const afterNoOp = repository.get(audience);
    const ledger = evidenceLedger([
      ledgerEntry({ streamEntryId: firstSource, streamIndex: 0, text: "first compile turn" }),
      ledgerEntry({ streamEntryId: secondSource, streamIndex: 1, text: "no-op compile turn" }),
      ledgerEntry({ streamEntryId: thirdSource, streamIndex: 2, text: "third compile turn" }),
    ]);
    const context = buildSharedStateLedgerPromptContext({
      ledger,
      previousArtifact: afterNoOp,
      fullPromptVisibleLedger: renderEvidenceLedger(ledger) ?? "",
      enabled: true,
      minTailPerSection: 1,
    });

    expect(afterNoOp?.record_version).toBe((afterFirst?.record_version ?? 0) + 1);
    expect(afterNoOp?.last_compiled_stream_entry_id).toBe(secondSource);
    expect(context.ledgerMode).toBe("delta");
    expect(context.promptVisibleLedger).not.toContain("first compile turn");
    expect(context.promptVisibleLedger).not.toContain("no-op compile turn");
    expect(context.promptVisibleLedger).toContain("third compile turn");
  });

  it("rejects citations hidden by the delta-rendered ledger context", async () => {
    const trace = createTraceRecorder();
    const olderSource = createStreamEntryId();
    const anchorSource = createStreamEntryId();
    const deltaSource = createStreamEntryId();
    repository.upsert(audience, [], {
      lastCompiledStreamEntryId: anchorSource,
    });
    const ledger = evidenceLedger([
      ledgerEntry({ streamEntryId: olderSource, streamIndex: 0, text: "older hidden turn" }),
      ledgerEntry({ streamEntryId: anchorSource, streamIndex: 1, text: "anchor turn" }),
      ledgerEntry({ streamEntryId: deltaSource, streamIndex: 2, text: "visible delta turn" }),
    ]);
    const context = buildSharedStateLedgerPromptContext({
      ledger,
      previousArtifact: repository.get(audience),
      fullPromptVisibleLedger: renderEvidenceLedger(ledger) ?? "",
      enabled: true,
      minTailPerSection: 1,
    });
    const llmClient = new FakeLLMClient({
      responses: [
        emitSharedStateArtifactPatchResponse({
          operations: [
            {
              type: "add",
              kind: "locked",
              text: "Hidden citation should not be accepted",
              owner_entity_id: audience,
              source_stream_entry_ids: [olderSource],
            },
          ],
        }),
      ],
    });

    await compileSharedStateArtifact({
      ...baseInput(llmClient),
      currentUserStreamEntryId: deltaSource,
      promptVisibleLedger: context.promptVisibleLedger,
      previousArtifact: repository.get(audience),
      allowedSourceStreamEntryIds: context.visibleStreamEntryIds,
      tracer: trace,
      ledgerMode: context.ledgerMode,
    });

    expect(context.ledgerMode).toBe("delta");
    expect(context.visibleStreamEntryIds).toEqual([deltaSource]);
    expect(repository.get(audience)?.entries).toHaveLength(0);
    expect(trace.events).toContainEqual(
      expect.objectContaining({
        event: "shared_state.compile.completed",
        data: expect.objectContaining({
          rejectedCount: 1,
          rejectionReasons: ["disallowed_source_stream_entry_id"] satisfies JsonValue,
          applied: false,
        }),
      }),
    );
  });

  it("keeps quarantined ledger entries visible while removing their stream ids from the source allow-list", () => {
    const quarantinedSource = createStreamEntryId();
    const trustedSource = createStreamEntryId();
    const ledger = evidenceLedger([
      ledgerEntry({
        streamEntryId: quarantinedSource,
        streamIndex: 0,
        text: "Quarantined context remains visible.",
        taint: "quarantined",
      }),
      ledgerEntry({
        streamEntryId: trustedSource,
        streamIndex: 1,
        text: "Trusted context remains citable.",
      }),
    ]);
    const context = buildSharedStateLedgerPromptContext({
      ledger,
      previousArtifact: null,
      fullPromptVisibleLedger: renderEvidenceLedger(ledger) ?? "",
      enabled: true,
      minTailPerSection: 1,
      sourceTrustValidator: (streamEntryId) =>
        streamEntryId === quarantinedSource
          ? { allowed: false, reason: "quarantined" }
          : { allowed: true },
    });

    expect(context.ledgerMode).toBe("full_fallback");
    expect(context.visibleStreamEntryIds).toEqual([trustedSource]);
    expect(context.offLimitsSourceStreamEntryIds).toEqual([quarantinedSource]);
    expect(context.promptVisibleLedger).toContain("Quarantined context remains visible.");
    expect(context.promptVisibleLedger).toContain(quarantinedSource);
  });

  it("marks citation-guard rejections for quarantined stream ids with source trust details", async () => {
    const trace = createTraceRecorder();
    const quarantinedSource = createStreamEntryId();
    const trustedSource = createStreamEntryId();
    const llmClient = new FakeLLMClient({
      responses: [
        emitSharedStateArtifactPatchResponse({
          operations: [
            {
              type: "add",
              kind: "locked",
              text: "Canonical decision from an off-limits source",
              owner_entity_id: audience,
              source_stream_entry_ids: [quarantinedSource],
            },
          ],
        }),
      ],
    });

    await compileSharedStateArtifact({
      ...baseInput(llmClient),
      promptVisibleLedger: "Trusted context and off-limits context are both visible.",
      allowedSourceStreamEntryIds: [trustedSource],
      offLimitsSourceStreamEntryIds: [quarantinedSource],
      sourceTrustValidator: (streamEntryId) =>
        streamEntryId === quarantinedSource
          ? { allowed: false, reason: "quarantined" }
          : { allowed: true },
      tracer: trace,
    });

    const requestPayload = JSON.parse(String(llmClient.requests[0]?.messages[0]?.content)) as {
      source_trust?: unknown;
    };
    const completed = trace.events.find(
      (event) => event.event === "shared_state.compile.completed",
    );

    expect(repository.get(audience)?.entries ?? []).toHaveLength(0);
    expect(requestPayload.source_trust).toEqual({
      citation_eligible_source_stream_entry_id_count: 1,
      off_limits_source_stream_entry_ids: [quarantinedSource],
    });
    expect(completed?.data).toEqual(
      expect.objectContaining({
        rejectedCount: 1,
        rejectionReasons: ["quarantined_source_stream_entry_id"] satisfies JsonValue,
        source_trust_rejections: [
          {
            operation_index: 0,
            operation_type: "add",
            source_stream_entry_id: quarantinedSource,
            source_trust_reason: "quarantined",
          },
        ] satisfies JsonValue,
      }),
    );
  });

  it("does not re-allow the current user turn through relational-slot evidence", async () => {
    const trace = createTraceRecorder();
    const trustedSource = createStreamEntryId();
    const llmClient = new FakeLLMClient({
      responses: [
        emitSharedStateArtifactPatchResponse({
          operations: [
            {
              type: "add",
              kind: "locked",
              text: "Current user sourced decision",
              owner_entity_id: audience,
              source_stream_entry_ids: [currentStreamEntryId],
            },
          ],
        }),
      ],
    });

    await compileSharedStateArtifact({
      ...baseInput(llmClient),
      allowedSourceStreamEntryIds: [trustedSource],
      offLimitsSourceStreamEntryIds: [currentStreamEntryId],
      relationalSlotsContext: [
        {
          id: createRelationalSlotId(),
          subject_entity_id: alice,
          slot_key: "context.current_turn",
          value: "current turn",
          state: "established",
          evidence_stream_entry_ids: [currentStreamEntryId],
          contradicted_by_stream_entry_ids: [],
          alternate_values: [],
        },
      ],
      sourceTrustValidator: () => ({ allowed: true }),
      tracer: trace,
    });

    const requestPayload = JSON.parse(String(llmClient.requests[0]?.messages[0]?.content)) as {
      source_trust?: unknown;
    };
    const completed = trace.events.find(
      (event) => event.event === "shared_state.compile.completed",
    );

    expect(repository.get(audience)?.entries ?? []).toHaveLength(0);
    expect(requestPayload.source_trust).toEqual({
      citation_eligible_source_stream_entry_id_count: 1,
      off_limits_source_stream_entry_ids: [currentStreamEntryId],
    });
    expect(completed?.data).toEqual(
      expect.objectContaining({
        rejectedCount: 1,
        rejectionReasons: ["disallowed_source_stream_entry_id"] satisfies JsonValue,
      }),
    );
  });

  it("advances safe prefilter skip markers so the next ledger delta starts after the skip turn", async () => {
    const firstSource = createStreamEntryId();
    const skippedSource = createStreamEntryId();
    const thirdSource = createStreamEntryId();
    const llmClient = new FakeLLMClient({
      responses: [
        emitSharedStateArtifactPatchResponse({
          operations: [
            {
              type: "add",
              kind: "live",
              text: "Live shared-state decision",
              owner_entity_id: audience,
              source_stream_entry_ids: [firstSource],
            },
          ],
        }),
      ],
    });

    await compileSharedStateArtifact({
      ...baseInput(llmClient),
      currentUserStreamEntryId: firstSource,
      allowedSourceStreamEntryIds: [firstSource],
    });

    const afterFirst = repository.get(audience);
    const skipped = advanceSharedStateCompileSkipAnchor({
      repository,
      audienceEntityId: audience,
      previousArtifact: afterFirst,
      currentUserStreamEntryId: skippedSource,
      nowMs: clock.now(),
    });
    const ledger = evidenceLedger([
      ledgerEntry({ streamEntryId: firstSource, streamIndex: 0, text: "first compile turn" }),
      ledgerEntry({ streamEntryId: skippedSource, streamIndex: 1, text: "skipped closure turn" }),
      ledgerEntry({ streamEntryId: thirdSource, streamIndex: 2, text: "third compile turn" }),
    ]);
    const context = buildSharedStateLedgerPromptContext({
      ledger,
      previousArtifact: skipped.artifact,
      fullPromptVisibleLedger: renderEvidenceLedger(ledger) ?? "",
      enabled: true,
      minTailPerSection: 1,
    });

    expect(skipped.advanced).toBe(true);
    expect(skipped.artifact?.record_version).toBe((afterFirst?.record_version ?? 0) + 1);
    expect(skipped.artifact?.last_compiled_stream_entry_id).toBe(skippedSource);
    expect(context.ledgerMode).toBe("delta");
    expect(context.promptVisibleLedger).not.toContain("first compile turn");
    expect(context.promptVisibleLedger).not.toContain("skipped closure turn");
    expect(context.promptVisibleLedger).toContain("third compile turn");
  });

  it("sends a summarized previous artifact instead of the full artifact JSON", async () => {
    repository.upsert(audience, [
      {
        type: "add",
        state_key: "decision.live",
        kind: "live",
        text: "Live shared-state decision",
        provenance_stream_entry_ids: [currentStreamEntryId],
      },
    ]);
    const llmClient = new FakeLLMClient({
      responses: [emitSharedStateArtifactPatchResponse({ operations: [] })],
    });

    await compileSharedStateArtifact(baseInput(llmClient));

    const prompt = JSON.parse(llmClient.requests[0]?.messages[0]?.content ?? "{}") as {
      previous_artifact?: unknown;
      previous_artifact_summary?: {
        active_entries?: {
          live?: Array<{ text: string }>;
        };
        active_entries_by_state_key?: Record<string, Array<{ text: string }>>;
      };
    };

    expect(prompt.previous_artifact).toBeUndefined();
    expect(prompt.previous_artifact_summary?.active_entries?.live).toEqual([
      expect.objectContaining({
        state_key: "decision.live",
        text: "Live shared-state decision",
      }),
    ]);
    expect(prompt.previous_artifact_summary?.active_entries_by_state_key).toMatchObject({
      "decision.live": [
        expect.objectContaining({
          text: "Live shared-state decision",
        }),
      ],
    });
  });

  it("renders every active state_key in a prominent registry before the artifact summary", async () => {
    repository.upsert(audience, [
      {
        type: "add",
        state_key: "observation.nora.video_call_repeated_question",
        kind: "live",
        text: "The active observation thread is preserved.",
        provenance_stream_entry_ids: [currentStreamEntryId],
      },
      {
        type: "add",
        state_key: "decision.architecture.api_boundary",
        kind: "pending",
        text: "The active pending decision is preserved.",
        provenance_stream_entry_ids: [currentStreamEntryId],
      },
    ]);
    const llmClient = new FakeLLMClient({
      responses: [emitSharedStateArtifactPatchResponse({ operations: [] })],
    });

    await compileSharedStateArtifact({
      ...baseInput(llmClient),
      previousArtifactSummaryOptions: {
        maxEntries: {
          live: 0,
          pending: 0,
        },
      },
    });

    const promptContent = llmClient.requests[0]?.messages[0]?.content ?? "{}";
    const prompt = JSON.parse(promptContent) as {
      existing_state_key_registry?: Array<{
        state_key: string;
        bucket: string;
        active_entry_ids: string[];
        active_entry_count: number;
        kinds: SharedStateEntryKind[];
        most_recent_update_at: number;
        most_recent_stream_entry_id: string | null;
      }>;
      previous_artifact_summary?: {
        active_entries?: {
          live?: Array<{ text: string }>;
          pending?: Array<{ text: string }>;
        };
      };
    };

    expect(promptContent.indexOf('"existing_state_key_registry"')).toBeLessThan(
      promptContent.indexOf('"previous_artifact_summary"'),
    );
    expect(prompt.existing_state_key_registry).toEqual([
      expect.objectContaining({
        state_key: "decision.architecture.api_boundary",
        bucket: "decision.architecture",
        active_entry_count: 1,
        kinds: ["pending"],
        active_entry_ids: [expect.any(String)],
        most_recent_update_at: expect.any(Number),
        most_recent_stream_entry_id: currentStreamEntryId,
      }),
      expect.objectContaining({
        state_key: "observation.nora.video_call_repeated_question",
        bucket: "observation.nora",
        active_entry_count: 1,
        kinds: ["live"],
        active_entry_ids: [expect.any(String)],
        most_recent_update_at: expect.any(Number),
        most_recent_stream_entry_id: currentStreamEntryId,
      }),
    ]);
    expect(prompt.previous_artifact_summary?.active_entries?.live).toEqual([]);
    expect(prompt.previous_artifact_summary?.active_entries?.pending).toEqual([]);
  });

  it("warns when the compiler input estimate exceeds the prompt budget", async () => {
    const trace = createTraceRecorder();
    const llmClient = new FakeLLMClient({
      responses: [emitSharedStateArtifactPatchResponse({ operations: [] })],
    });

    await compileSharedStateArtifact({
      ...baseInput(llmClient),
      promptVisibleLedger: "large ledger entry ".repeat(180_000),
      tracer: trace,
      ledgerMode: "delta",
    });

    const warning = trace.events.find((event) => event.event === "shared_state.compile.degraded");
    const completed = trace.events.find(
      (event) => event.event === "shared_state.compile.completed",
    );

    expect(warning).toBeDefined();
    expect(warning?.data.ledger_mode).toBe("delta");
    expect(typeof warning?.data.input_token_estimate).toBe("number");
    expect(warning?.data.input_token_estimate as number).toBeGreaterThan(35_000);
    expect(warning?.data.breakdown).toEqual(
      expect.objectContaining({
        prompt_visible_ledger: expect.any(Number),
      }),
    );
    expect(completed?.data).toEqual(
      expect.objectContaining({
        ledger_mode: "delta",
        input_token_estimate: warning?.data.input_token_estimate,
      }),
    );
  });

  it("repairs an invalid compiler payload once and applies the corrected patch", async () => {
    const trace = createTraceRecorder();
    const llmClient = new FakeLLMClient({
      responses: [
        emitRawSharedStateArtifactPatchResponse({
          operations: [
            {
              type: "replace",
              text: "Invalid operation type",
            },
          ],
        }),
        emitSharedStateArtifactPatchResponse({
          operations: [
            {
              type: "add",
              kind: "live",
              text: "Corrected live shared-state entry",
              owner_entity_id: audience,
              source_stream_entry_ids: [currentStreamEntryId],
            },
          ],
        }),
      ],
    });

    const patch = await compileSharedStateArtifact({
      ...baseInput(llmClient),
      tracer: trace,
    });
    const repairPayload = JSON.parse(String(llmClient.requests[1]?.messages[0]?.content)) as {
      additional_prompt_sections?: string[];
    };

    expect(llmClient.requests).toHaveLength(2);
    expect(repairPayload.additional_prompt_sections?.[0]).toContain(
      "Your previous patch was invalid:",
    );
    expect(repairPayload.additional_prompt_sections?.[0]).toContain("Emit a corrected patch.");
    expect(patch.operations).toHaveLength(1);
    expect(activeEntries().map((entry) => entry.text)).toContain(
      "Corrected live shared-state entry",
    );
    expect(trace.events.map((event) => event.event)).toEqual(
      expect.arrayContaining([
        "shared_state.compile.repair_attempted",
        "shared_state.compile.repair_succeeded",
      ]),
    );
    expect(trace.events.some((event) => event.event === "shared_state.compile.repair_failed")).toBe(
      false,
    );
    expectSingleSuccessfulRepair(trace);
  });

  it("repairs live adds that exceed the per-key active-entry cap", async () => {
    const firstSource = createStreamEntryId();
    const secondSource = createStreamEntryId();
    repository.upsert(audience, [
      {
        type: "add",
        state_key: "observation.recurring",
        kind: "live",
        text: "Live observation cluster A",
        owner_entity_id: audience,
        provenance_stream_entry_ids: [firstSource],
        created_at: 1_000,
        last_updated_at: 1_000,
        rank: 0,
      },
      {
        type: "add",
        state_key: "observation.recurring",
        kind: "live",
        text: "Live observation cluster B",
        owner_entity_id: audience,
        provenance_stream_entry_ids: [secondSource],
        created_at: 1_100,
        last_updated_at: 1_100,
        rank: 1,
      },
    ]);
    const targetEntryId = activeEntries()[1]?.id;
    const trace = createTraceRecorder();
    const llmClient = new FakeLLMClient({
      responses: [
        emitSharedStateArtifactPatchResponse({
          operations: [
            {
              type: "add",
              state_key: "observation.recurring",
              kind: "live",
              text: "Third parallel live observation",
              owner_entity_id: audience,
              source_stream_entry_ids: [currentStreamEntryId],
            },
          ],
        }),
        emitSharedStateArtifactPatchResponse({
          operations: [
            {
              type: "update",
              id: targetEntryId,
              state_key: "observation.recurring",
              text: "Merged recurring observation cluster",
              source_stream_entry_ids: [currentStreamEntryId],
            },
          ],
        }),
      ],
    });

    const patch = await compileSharedStateArtifact({
      ...baseInput(llmClient),
      tracer: trace,
      lifecycle: {
        maxLiveEntriesPerKey: 2,
      },
    });
    const repairPayload = JSON.parse(String(llmClient.requests[1]?.messages[0]?.content)) as {
      additional_prompt_sections?: string[];
    };

    expect(targetEntryId).toBeDefined();
    expect(llmClient.requests).toHaveLength(2);
    expect(repairPayload.additional_prompt_sections?.[0]).toContain(
      "structural shared-state key compaction",
    );
    expect(repairPayload.additional_prompt_sections?.[0]).toContain("observation.recurring");
    expect(patch.operations).toEqual([
      expect.objectContaining({
        type: "update",
        id: targetEntryId,
        state_key: "observation.recurring",
        text: "Merged recurring observation cluster",
      }),
    ]);
    expect(
      activeEntries().filter((entry) => entry.state_key === "observation.recurring"),
    ).toHaveLength(2);
    expect(activeEntries().map((entry) => entry.text)).toContain(
      "Merged recurring observation cluster",
    );
    expect(trace.events).toEqual(
      expect.arrayContaining([
        expect.objectContaining({
          event: "shared_state.compile.add_rejected_cap_exceeded",
          data: expect.objectContaining({
            state_key: "observation.recurring",
            current_count: 2,
            proposed_count: 3,
            max_live_entries_per_key: 2,
            target_entry_id: targetEntryId,
          }),
        }),
        expect.objectContaining({ event: "shared_state.compile.repair_succeeded" }),
        expect.objectContaining({
          event: "shared_state.compile.completed",
          data: expect.objectContaining({
            add_rejected_cap_exceeded_count: 1,
          }),
        }),
      ]),
    );
    expectSingleSuccessfulRepair(trace);
  });

  it("repairs near-duplicate add state_keys by reusing the active key", async () => {
    repository.upsert(audience, [
      {
        type: "add",
        state_key: "observation.nora.video_call_repeated_question",
        kind: "live",
        text: "Existing repeated-question video call observation.",
        provenance_stream_entry_ids: [currentStreamEntryId],
      },
    ]);
    const targetEntryId = activeEntries()[0]?.id;
    const trace = createTraceRecorder();
    const llmClient = new FakeLLMClient({
      responses: [
        emitRawSharedStateArtifactPatchResponse({
          operations: [
            {
              type: "add",
              state_key: "observation.nora.video_call_repeated_question_reconfirm",
              kind: "live",
              text: "Reconfirmed repeated-question video call observation.",
              source_stream_entry_ids: [currentStreamEntryId],
            },
          ],
        }),
        emitSharedStateArtifactPatchResponse({
          operations: [
            {
              type: "update",
              id: targetEntryId,
              state_key: "observation.nora.video_call_repeated_question",
              text: "Merged repeated-question video call observation.",
              source_stream_entry_ids: [currentStreamEntryId],
            },
          ],
        }),
      ],
    });

    const patch = await compileSharedStateArtifact({
      ...baseInput(llmClient),
      tracer: trace,
    });
    const repairPayload = JSON.parse(String(llmClient.requests[1]?.messages[0]?.content)) as {
      additional_prompt_sections?: string[];
    };

    expect(targetEntryId).toBeDefined();
    expect(llmClient.requests).toHaveLength(2);
    expect(repairPayload.additional_prompt_sections?.[0]).toContain(
      "appears to cover the same thread",
    );
    expect(repairPayload.additional_prompt_sections?.[0]).toContain(
      "observation.nora.video_call_repeated_question_reconfirm",
    );
    expect(repairPayload.additional_prompt_sections?.[0]).toContain(
      "observation.nora.video_call_repeated_question",
    );
    expect(patch.operations).toEqual([
      expect.objectContaining({
        type: "update",
        id: targetEntryId,
        state_key: "observation.nora.video_call_repeated_question",
        text: "Merged repeated-question video call observation.",
      }),
    ]);
    expect(
      activeEntries().filter(
        (entry) => entry.state_key === "observation.nora.video_call_repeated_question",
      ),
    ).toHaveLength(1);
    expect(trace.events).toEqual(
      expect.arrayContaining([
        expect.objectContaining({
          event: "shared_state.compile.add_rejected_near_duplicate_state_key",
          data: expect.objectContaining({
            state_key: "observation.nora.video_call_repeated_question_reconfirm",
            similar_state_keys: ["observation.nora.video_call_repeated_question"],
            shared_state_key_tokens: expect.arrayContaining(["observation", "nora", "video"]),
          }),
        }),
        expect.objectContaining({ event: "shared_state.compile.repair_succeeded" }),
      ]),
    );
    expectSingleSuccessfulRepair(trace);
  });

  it("repairs never-seen add state_keys without new_key_reason", async () => {
    const trace = createTraceRecorder();
    repository.upsert(
      audience,
      [
        {
          type: "add",
          state_key: "decision.existing",
          kind: "locked",
          text: "Existing shared-state key.",
          provenance_stream_entry_ids: [currentStreamEntryId],
        },
      ],
      {
        lastCompiledStreamEntryId: currentStreamEntryId,
      },
    );
    const llmClient = new FakeLLMClient({
      responses: [
        emitRawSharedStateArtifactPatchResponse({
          operations: [
            {
              type: "add",
              state_key: "decision.architecture.api_boundary",
              kind: "live",
              text: "New architecture decision boundary.",
              source_stream_entry_ids: [currentStreamEntryId],
            },
          ],
        }),
        emitSharedStateArtifactPatchResponse({
          operations: [
            {
              type: "add",
              state_key: "decision.architecture.api_boundary",
              kind: "live",
              text: "New architecture decision boundary.",
              source_stream_entry_ids: [currentStreamEntryId],
              new_key_reason: "Represents a new architecture boundary thread.",
            },
          ],
        }),
      ],
    });

    const patch = await compileSharedStateArtifact({
      ...baseInput(llmClient),
      tracer: trace,
    });
    const repairPayload = JSON.parse(String(llmClient.requests[1]?.messages[0]?.content)) as {
      additional_prompt_sections?: string[];
    };

    expect(llmClient.requests).toHaveLength(2);
    expect(repairPayload.additional_prompt_sections?.[0]).toContain("new_key_reason");
    expect(repairPayload.additional_prompt_sections?.[0]).toContain(
      "decision.architecture.api_boundary",
    );
    expect(patch.operations).toEqual([
      expect.objectContaining({
        type: "add",
        state_key: "decision.architecture.api_boundary",
        text: "New architecture decision boundary.",
      }),
    ]);
    expect(trace.events).toEqual(
      expect.arrayContaining([
        expect.objectContaining({
          event: "shared_state.compile.add_rejected_missing_new_key_reason",
          data: expect.objectContaining({
            state_key: "decision.architecture.api_boundary",
          }),
        }),
        expect.objectContaining({ event: "shared_state.compile.repair_succeeded" }),
      ]),
    );
    expectSingleSuccessfulRepair(trace);
  });

  it("repairs shared-state operations with ungrounded protected relationship labels", async () => {
    const trace = createTraceRecorder();
    const llmClient = new FakeLLMClient({
      responses: [
        emitSharedStateArtifactPatchResponse({
          operations: [
            {
              type: "add",
              kind: "locked",
              text: "Use the parent constraint for care planning.",
              owner_entity_id: audience,
              source_stream_entry_ids: [currentStreamEntryId],
            },
          ],
        }),
        emitSharedStateArtifactPatchResponse({
          operations: [
            {
              type: "add",
              kind: "locked",
              text: "Use the parent constraint for care planning.",
              owner_entity_id: audience,
              source_stream_entry_ids: [currentStreamEntryId],
              relationship_evidence_stream_entry_ids: [currentStreamEntryId],
            },
          ],
        }),
      ],
    });

    const patch = await compileSharedStateArtifact({
      ...baseInput(llmClient),
      tracer: trace,
      relationshipEvidenceStreamEntryTrust: (streamEntryId) =>
        streamEntryId === currentStreamEntryId
          ? { allowed: true }
          : { allowed: false, reason: "missing" },
    });
    const repairPayload = JSON.parse(String(llmClient.requests[1]?.messages[0]?.content)) as {
      additional_prompt_sections?: string[];
    };

    expect(llmClient.requests).toHaveLength(2);
    expect(repairPayload.additional_prompt_sections?.[0]).toContain(
      "relationship_evidence_relational_slot_ids",
    );
    expect(patch.operations).toEqual([
      expect.objectContaining({
        type: "add",
        text: "Use the parent constraint for care planning.",
      }),
    ]);
    expect(trace.events).toEqual(
      expect.arrayContaining([
        expect.objectContaining({
          event: "shared_state.compile.label_ungrounded",
          data: expect.objectContaining({
            protected_relationship_labels: ["parent"],
            relationship_evidence_stream_entry_ids: [],
          }),
        }),
        expect.objectContaining({ event: "shared_state.compile.repair_succeeded" }),
      ]),
    );
    expectSingleSuccessfulRepair(trace);
  });

  it("degrades when compiler payload repair is still invalid", async () => {
    const trace = createTraceRecorder();
    const invalidResponse = emitRawSharedStateArtifactPatchResponse({
      operations: [
        {
          type: "replace",
          text: "Still invalid operation type",
        },
      ],
    });
    const llmClient = new FakeLLMClient({
      responses: [invalidResponse, invalidResponse],
    });

    const patch = await compileSharedStateArtifact({
      ...baseInput(llmClient),
      tracer: trace,
    });

    expect(llmClient.requests).toHaveLength(2);
    expect(patch.operations).toHaveLength(0);
    expect(activeEntries()).toHaveLength(0);
    expect(trace.events.map((event) => event.event)).toEqual(
      expect.arrayContaining([
        "shared_state.compile.repair_attempted",
        "shared_state.compile.repair_failed",
        "shared_state.compile.degraded",
      ]),
    );
    expect(
      trace.events.find((event) => event.event === "shared_state.compile.degraded")?.data.reason,
    ).toBe("invalid_payload");
  });

  it("does not repair max-token-truncated compiler payloads", async () => {
    const trace = createTraceRecorder();
    const llmClient = new FakeLLMClient({
      responses: [
        {
          ...emitRawSharedStateArtifactPatchResponse({
            operations: [
              {
                type: "replace",
                text: "Partial invalid operation from truncation",
              },
            ],
          }),
          stop_reason: "max_tokens",
        },
      ],
    });

    const patch = await compileSharedStateArtifact({
      ...baseInput(llmClient),
      tracer: trace,
    });

    expect(llmClient.requests).toHaveLength(1);
    expect(patch.operations).toHaveLength(0);
    expect(
      trace.events.some((event) => event.event === "shared_state.compile.repair_attempted"),
    ).toBe(false);
    expect(
      trace.events.find((event) => event.event === "shared_state.compile.degraded")?.data.reason,
    ).toBe("invalid_payload");
  });

  it("does not attempt compiler payload repair when the first patch is valid", async () => {
    const trace = createTraceRecorder();
    const llmClient = new FakeLLMClient({
      responses: [emitSharedStateArtifactPatchResponse({ operations: [] })],
    });

    await compileSharedStateArtifact({
      ...baseInput(llmClient),
      tracer: trace,
    });

    expect(llmClient.requests).toHaveLength(1);
    expect(
      trace.events.some((event) => event.event === "shared_state.compile.repair_attempted"),
    ).toBe(false);
  });

  it("enforces the active-entry lifecycle budget on a no-op patch", async () => {
    const source = createStreamEntryId();

    repository.upsert(
      audience,
      Array.from({ length: 50 }, (_, index) => ({
        type: "add" as const,
        state_key: `decision.lifecycle_${index}`,
        kind: "locked" as const,
        text: `Locked planning entry ${index}`,
        provenance_stream_entry_ids: [source],
        created_at: 1_000 + index,
        last_updated_at: 1_000 + index,
        rank: index,
      })),
      {
        lastCompiledStreamEntryId: source,
      },
    );

    const llmClient = new FakeLLMClient({
      responses: [emitSharedStateArtifactPatchResponse({ operations: [] })],
    });
    const trace = createTraceRecorder();
    const patch = await compileSharedStateArtifact({
      ...baseInput(llmClient),
      allowedSourceStreamEntryIds: [source, currentStreamEntryId],
      tracer: trace,
    });

    expect(activeEntries()).toHaveLength(40);
    expect(patch.operations.filter((operation) => operation.type === "prune")).toHaveLength(10);
    expect(trace.events).toContainEqual(
      expect.objectContaining({
        event: "shared_state.compile.completed",
        data: expect.objectContaining({
          artifact_total_entry_count: 40,
          artifact_active_entry_count: 40,
          artifact_omitted_entry_count: 26,
          artifact_pruned_entry_count_this_turn: 10,
          artifact_superseded_count_this_turn: 0,
          rendered_by_kind: expect.objectContaining({
            locked: 14,
          }),
        }),
      }),
    );
  });

  it("keeps long compile sequences inside the active artifact budget while reserving live render slots", async () => {
    const maxActiveEntries = 40;
    const liveRenderReservation = 8;
    const responses = Array.from({ length: 60 }, (_, index) =>
      emitSharedStateArtifactPatchResponse({
        operations: [
          {
            type: "add",
            kind: "locked",
            text: `Locked long-plan route invariant ${index}`,
            owner_entity_id: audience,
            source_stream_entry_ids: [currentStreamEntryId],
          },
          {
            type: "add",
            kind: "live",
            text: `Live long-plan detail ${index}`,
            owner_entity_id: audience,
            source_stream_entry_ids: [currentStreamEntryId],
          },
          {
            type: "add",
            kind: "pending",
            text: `Pending long-plan decision ${index}`,
            owner_entity_id: audience,
            source_stream_entry_ids: [currentStreamEntryId],
          },
        ],
      }),
    );
    const llmClient = new FakeLLMClient({ responses });
    const trace = createTraceRecorder();
    let sawRenderedOmission = false;
    let sawLifecyclePrune = false;

    for (let index = 0; index < responses.length; index += 1) {
      await compileSharedStateArtifact({
        ...baseInput(llmClient),
        currentUserMessage: `Long planning turn ${index}`,
        tracer: trace,
      });

      const artifact = repository.get(audience);
      const active = activeEntries();
      const activeLive = active.filter((entry) => entry.kind === "live");
      const expectedLiveEntries = [...activeLive]
        .sort(
          (left, right) =>
            right.last_updated_at - left.last_updated_at ||
            left.rank - right.rank ||
            right.created_at - left.created_at ||
            left.id.localeCompare(right.id),
        )
        .slice(0, Math.min(activeLive.length, liveRenderReservation));
      const rendered = renderSharedStateArtifact(artifact) ?? "";
      const renderedLiveEntryCount = rendered.match(/kind=live/g)?.length ?? 0;
      const completed = trace.events
        .filter((event) => event.event === "shared_state.compile.completed")
        .at(-1);
      const artifactTotalEntryCount = completed?.data.artifact_total_entry_count;
      const artifactActiveEntryCount = completed?.data.artifact_active_entry_count;
      const artifactOmittedEntryCount = completed?.data.artifact_omitted_entry_count;
      const artifactRenderedEntryCount = completed?.data.artifactEntryCount;
      const artifactPrunedEntryCount = completed?.data.artifact_pruned_entry_count_this_turn;

      expect(active.length).toBeLessThanOrEqual(maxActiveEntries);
      expect(renderedLiveEntryCount).toBeGreaterThanOrEqual(expectedLiveEntries.length);
      expect(expectedLiveEntries.every((entry) => rendered.includes(entry.text))).toBe(true);
      expect(completed?.data).toEqual(
        expect.objectContaining({
          artifact_total_entry_count: expect.any(Number),
          artifact_active_entry_count: expect.any(Number),
          artifact_omitted_entry_count: expect.any(Number),
          artifact_pruned_entry_count_this_turn: expect.any(Number),
          artifact_superseded_count_this_turn: expect.any(Number),
          rendered_by_kind: expect.any(Object),
        }),
      );
      expect(artifactTotalEntryCount as number).toBeGreaterThanOrEqual(active.length);
      expect(artifactActiveEntryCount as number).toBe(active.length);
      if (typeof artifactOmittedEntryCount === "number" && artifactOmittedEntryCount > 0) {
        sawRenderedOmission = true;
      }
      if (typeof artifactPrunedEntryCount === "number" && artifactPrunedEntryCount > 0) {
        sawLifecyclePrune = true;
      }
      if (
        typeof artifactRenderedEntryCount === "number" &&
        typeof artifactOmittedEntryCount === "number" &&
        active.length > artifactRenderedEntryCount
      ) {
        expect(artifactOmittedEntryCount).toBe(active.length - artifactRenderedEntryCount);
      }
    }

    expect(sawRenderedOmission).toBe(true);
    expect(sawLifecyclePrune).toBe(true);
  });

  it("prunes superseded dependencies before pruning a referenced replacement", async () => {
    const firstSource = createStreamEntryId();
    const secondSource = createStreamEntryId();
    const extraSource = createStreamEntryId();
    const initial = repository.upsert(audience, [
      {
        type: "add",
        state_key: "decision.original_route",
        kind: "locked",
        text: "Original locked route",
        provenance_stream_entry_ids: [firstSource],
        created_at: 1_000,
        last_updated_at: 1_000,
        rank: 0,
      },
    ]);
    const originalId = initial?.entries[0]?.id;

    expect(originalId).toBeDefined();
    const superseded = repository.upsert(audience, [
      {
        type: "supersede",
        id: originalId!,
        replacement: {
          state_key: "decision.original_route",
          kind: "locked",
          text: "Replacement locked route",
          provenance_stream_entry_ids: [secondSource],
          created_at: 1_100,
          last_updated_at: 1_100,
          rank: 1,
        },
        last_updated_stream_entry_ids: [secondSource],
      },
      {
        type: "add",
        state_key: "decision.extra_route_1",
        kind: "locked",
        text: "Extra locked route 1",
        provenance_stream_entry_ids: [extraSource],
        created_at: 2_000,
        last_updated_at: 2_000,
        rank: 2,
      },
      {
        type: "add",
        state_key: "decision.extra_route_2",
        kind: "locked",
        text: "Extra locked route 2",
        provenance_stream_entry_ids: [extraSource],
        created_at: 3_000,
        last_updated_at: 3_000,
        rank: 3,
      },
    ]);
    const replacementId = superseded?.entries.find((entry) => entry.id !== originalId)?.id;

    expect(replacementId).toBeDefined();

    const llmClient = new FakeLLMClient({
      responses: [emitSharedStateArtifactPatchResponse({ operations: [] })],
    });

    const patch = await compileSharedStateArtifact({
      ...baseInput(llmClient),
      allowedSourceStreamEntryIds: [firstSource, secondSource, extraSource, currentStreamEntryId],
      lifecycle: {
        maxActiveEntries: 1,
        kindSoftCaps: {
          locked: 0,
        },
      },
    });

    const artifact = repository.get(audience);

    expect(activeEntries()).toHaveLength(1);
    expect(activeEntries()[0]?.text).toBe("Extra locked route 2");
    expect(artifact?.entries.find((entry) => entry.id === originalId)).toBeUndefined();
    expect(artifact?.entries.find((entry) => entry.id === replacementId)).toBeUndefined();
    expect(
      patch.operations
        .filter((operation) => operation.type === "prune")
        .map((operation) => operation.id),
    ).toEqual(expect.arrayContaining([originalId, replacementId]));
  });

  it("keeps the active-entry cap hard when every replacement has a superseded referrer", async () => {
    const source = createStreamEntryId();
    const originalIds: string[] = [];
    const replacementIds: string[] = [];

    for (let index = 0; index < 42; index += 1) {
      const originalText = `Original locked route ${index}`;
      const replacementText = `Replacement locked route ${index}`;
      const original = repository.upsert(audience, [
        {
          type: "add",
          state_key: `decision.route_${index}`,
          kind: "locked",
          text: originalText,
          provenance_stream_entry_ids: [source],
          created_at: 1_000 + index,
          last_updated_at: 1_000 + index,
          rank: index,
        },
      ]);
      const originalId = original?.entries.find((entry) => entry.text === originalText)?.id;

      expect(originalId).toBeDefined();

      const superseded = repository.upsert(audience, [
        {
          type: "supersede",
          id: originalId!,
          replacement: {
            state_key: `decision.route_${index}`,
            kind: "locked",
            text: replacementText,
            provenance_stream_entry_ids: [source],
            created_at: 10_000 + index,
            last_updated_at: 10_000 + index,
            rank: index,
          },
          last_updated_stream_entry_ids: [source],
        },
      ]);
      const replacementId = superseded?.entries.find((entry) => entry.text === replacementText)?.id;

      expect(replacementId).toBeDefined();
      originalIds.push(originalId!);
      replacementIds.push(replacementId!);
    }

    const trace = createTraceRecorder();
    const llmClient = new FakeLLMClient({
      responses: [emitSharedStateArtifactPatchResponse({ operations: [] })],
    });
    const patch = await compileSharedStateArtifact({
      ...baseInput(llmClient),
      allowedSourceStreamEntryIds: [source, currentStreamEntryId],
      tracer: trace,
      lifecycle: {
        maxActiveEntries: 40,
        kindSoftCaps: {
          locked: 40,
        },
      },
    });
    const pruneIds = patch.operations
      .filter((operation) => operation.type === "prune")
      .map((operation) => operation.id);

    expect(activeEntries()).toHaveLength(40);
    expect(pruneIds).toHaveLength(4);
    for (const replacementId of replacementIds.filter((id) => pruneIds.includes(id))) {
      const originalId = originalIds[replacementIds.indexOf(replacementId)];

      expect(pruneIds.indexOf(originalId!)).toBeGreaterThanOrEqual(0);
      expect(pruneIds.indexOf(originalId!)).toBeLessThan(pruneIds.indexOf(replacementId));
    }
    expect(
      trace.events.find((event) => event.event === "shared_state.lifecycle.degraded"),
    ).toBeUndefined();
  });

  it("expands dependencies for an LLM-emitted prune of a referenced replacement", async () => {
    const firstSource = createStreamEntryId();
    const secondSource = createStreamEntryId();
    const initial = repository.upsert(audience, [
      {
        type: "add",
        state_key: "decision.original_route",
        kind: "locked",
        text: "Original locked route",
        provenance_stream_entry_ids: [firstSource],
        created_at: 1_000,
        last_updated_at: 1_000,
        rank: 0,
      },
    ]);
    const originalId = initial?.entries[0]?.id;

    expect(originalId).toBeDefined();

    const superseded = repository.upsert(audience, [
      {
        type: "supersede",
        id: originalId!,
        replacement: {
          state_key: "decision.original_route",
          kind: "locked",
          text: "Replacement locked route",
          provenance_stream_entry_ids: [secondSource],
          created_at: 1_100,
          last_updated_at: 1_100,
          rank: 1,
        },
        last_updated_stream_entry_ids: [secondSource],
      },
    ]);
    const replacementId = superseded?.entries.find((entry) => entry.id !== originalId)?.id;

    expect(replacementId).toBeDefined();

    const llmClient = new FakeLLMClient({
      responses: [
        emitSharedStateArtifactPatchResponse({
          operations: [
            {
              type: "prune",
              id: replacementId!,
            },
          ],
        }),
      ],
    });
    const patch = await compileSharedStateArtifact({
      ...baseInput(llmClient),
      allowedSourceStreamEntryIds: [firstSource, secondSource, currentStreamEntryId],
    });
    const pruneIds = patch.operations
      .filter((operation) => operation.type === "prune")
      .map((operation) => operation.id);
    const artifact = repository.get(audience);

    expect(pruneIds).toEqual([originalId, replacementId]);
    expect(artifact?.entries.find((entry) => entry.id === originalId)).toBeUndefined();
    expect(artifact?.entries.find((entry) => entry.id === replacementId)).toBeUndefined();
  });

  it("accepts all shared state kinds emitted by the compiler", async () => {
    const kinds = [
      "locked",
      "live",
      "tentative",
      "invalidated",
      "pending",
    ] as const satisfies readonly SharedStateEntryKind[];
    const trace = createTraceRecorder();
    const llmClient = new FakeLLMClient({
      responses: [
        emitSharedStateArtifactPatchResponse({
          operations: kinds.map((kind) => ({
            type: "add" as const,
            kind,
            text: `Artifact entry kind ${kind}`,
            owner_entity_id: audience,
            source_stream_entry_ids: [currentStreamEntryId],
          })),
        }),
      ],
    });

    await compileSharedStateArtifact({
      ...baseInput(llmClient),
      tracer: trace,
    });

    expect(
      repository
        .get(audience)
        ?.entries.map((entry) => entry.kind)
        .sort(),
    ).toEqual([...kinds].sort());
    expect(trace.events).toContainEqual(
      expect.objectContaining({
        event: "shared_state.compile.completed",
        data: expect.objectContaining({
          rejectedCount: 0,
          rejectionReasons: [],
          applied: true,
          operation_counts_by_kind: {
            add: 5,
            update: 0,
            supersede: 0,
            prune: 0,
          },
        }),
      }),
    );
  });

  it("applies internal lifecycle demotion after patch normalization without refreshing last update metadata", async () => {
    const oldSource = createStreamEntryId();
    const initial = repository.upsert(
      audience,
      [
        {
          type: "add",
          state_key: "state.placeholder",
          kind: "live",
          text: "Placeholder shared state for the fixture",
          provenance_stream_entry_ids: [oldSource],
          last_updated_stream_entry_ids: [oldSource],
          created_at: 100,
          last_updated_at: 100,
        },
      ],
      {
        now: 100,
      },
    );
    const entryId = initial?.entries[0]?.id;
    const trace = createTraceRecorder();
    const llmClient = new FakeLLMClient({
      responses: [emitSharedStateArtifactPatchResponse({ operations: [] })],
    });

    await compileSharedStateArtifact({
      ...baseInput(llmClient),
      tracer: trace,
      turnCounter: 12,
      renderOptions: {
        currentTurnCounter: 12,
        lastUpdatedTurnByStreamEntryId: {
          [oldSource]: 1,
        },
      },
      lifecycle: {
        recentTurnThreshold: 5,
        dormantTurnThreshold: 15,
      },
    });

    const entryAfterDemotion = repository
      .get(audience)
      ?.entries.find((entry) => entry.id === entryId);

    expect(entryAfterDemotion).toMatchObject({
      kind: "low_salience_live",
      last_updated_at: 100,
      last_updated_stream_entry_ids: [oldSource],
    });
    expect(trace.events).toContainEqual({
      event: "shared_state.lifecycle.demoted",
      data: expect.objectContaining({
        entry_id: entryId,
        from_kind: "live",
        to_kind: "low_salience_live",
        reason: "old_live_without_structural_pull",
      }),
    });
    expect(
      trace.events.find((event) => event.event === "shared_state.compile.completed")?.data,
    ).toMatchObject({
      lifecycle_demoted_live_to_low_salience_count: 1,
      active_by_kind: expect.objectContaining({
        low_salience_live: 1,
      }),
    });
  });

  it("emits lifecycle aging blocker diagnostics in compile completion trace", async () => {
    const ledgerSource = createStreamEntryId();
    const unknownAgeSource = createStreamEntryId();
    const initial = repository.upsert(
      audience,
      [
        {
          type: "add",
          state_key: "state.blocked",
          kind: "live",
          text: "A structurally protected shared state entry.",
          provenance_stream_entry_ids: [ledgerSource],
          last_updated_stream_entry_ids: [ledgerSource],
        },
        {
          type: "add",
          state_key: "state.unknown-age",
          kind: "live",
          text: "A shared state entry without turn age.",
          provenance_stream_entry_ids: [unknownAgeSource],
          last_updated_stream_entry_ids: [unknownAgeSource],
        },
      ],
      { now: 100 },
    );
    const entryId = initial?.entries.find((entry) => entry.state_key === "state.blocked")?.id;
    const unknownAgeEntryId = initial?.entries.find(
      (entry) => entry.state_key === "state.unknown-age",
    )?.id;
    const trace = createTraceRecorder();
    const llmClient = new FakeLLMClient({
      responses: [emitSharedStateArtifactPatchResponse({ operations: [] })],
    });

    await compileSharedStateArtifact({
      ...baseInput(llmClient),
      tracer: trace,
      turnCounter: 12,
      renderOptions: {
        ledgerStreamEntryIds: [ledgerSource],
        currentTurnCounter: 12,
        lastUpdatedTurnByStreamEntryId: {
          [ledgerSource]: 1,
        },
      },
      lifecycle: {
        recentTurnThreshold: 5,
        dormantTurnThreshold: 15,
      },
    });

    expect(
      trace.events.find((event) => event.event === "shared_state.compile.completed")?.data,
    ).toMatchObject({
      lifecycle_aging_blocker_counts_live_to_low_salience: {
        demotable_count: 1,
        demoted_count: 0,
        blocked_by_ledger_overlap: 1,
      },
      lifecycle_aging_blocker_counts_low_salience_to_dormant: {
        demotable_count: 0,
        demoted_count: 0,
      },
      lifecycle_aging_blocked_sample: [
        {
          entry_id: entryId,
          state_key: "state.blocked",
          age_turns: 11,
          rendered: true,
          block_reasons: ["ledger_overlap"],
          block_strengths: ["hard"],
          block_reasons_with_strength: [{ reason: "ledger_overlap", strength: "hard" }],
          active_canonicalizer_kinds: null,
        },
      ],
      lifecycle_aging_unknown_age_sample: [
        {
          entry_id: unknownAgeEntryId,
          state_key: "state.unknown-age",
          kind: "live",
          last_updated_stream_entry_ids_count: 1,
          last_updated_turn_global: null,
          rendered: true,
        },
      ],
    });
  });

  it("maps omitted update kinds for demoted entries back to public live kind", async () => {
    const oldSource = createStreamEntryId();
    const initial = repository.upsert(
      audience,
      [
        {
          type: "add",
          state_key: "state.placeholder",
          kind: "dormant_live",
          text: "Placeholder shared state before direct update",
          provenance_stream_entry_ids: [oldSource],
          last_updated_stream_entry_ids: [oldSource],
          created_at: 100,
          last_updated_at: 100,
        },
      ],
      {
        now: 100,
      },
    );
    const entryId = initial?.entries[0]?.id;
    const trace = createTraceRecorder();
    const llmClient = new FakeLLMClient({
      responses: [
        emitSharedStateArtifactPatchResponse({
          operations: [
            {
              type: "update",
              id: entryId,
              text: "Updated placeholder shared state",
              source_stream_entry_ids: [oldSource],
            },
          ],
        }),
      ],
    });

    const patch = await compileSharedStateArtifact({
      ...baseInput(llmClient),
      allowedSourceStreamEntryIds: [currentStreamEntryId, oldSource],
      tracer: trace,
      turnCounter: 30,
      renderOptions: {
        currentTurnCounter: 30,
        lastUpdatedTurnByStreamEntryId: {
          [oldSource]: 1,
        },
      },
      lifecycle: {
        recentTurnThreshold: 5,
        dormantTurnThreshold: 15,
      },
    });

    expect(patch.operations).toEqual([
      expect.objectContaining({
        type: "update",
        id: entryId,
        kind: "live",
        text: "Updated placeholder shared state",
      }),
    ]);
    expect(repository.get(audience)?.entries.find((entry) => entry.id === entryId)?.kind).toBe(
      "live",
    );
    expect(trace.events).toContainEqual({
      event: "shared_state.lifecycle.reactivated",
      data: expect.objectContaining({
        entry_id: entryId,
        from_kind: "dormant_live",
        to_kind: "live",
        reason: "touched_by_patch",
      }),
    });
  });

  it("advances live entries through low-salience and dormant states across compiles", async () => {
    const oldSource = createStreamEntryId();
    const initial = repository.upsert(
      audience,
      [
        {
          type: "add",
          state_key: "state.placeholder",
          kind: "live",
          text: "Placeholder shared state before aging",
          provenance_stream_entry_ids: [oldSource],
          last_updated_stream_entry_ids: [oldSource],
          created_at: 100,
          last_updated_at: 100,
        },
      ],
      {
        now: 100,
      },
    );
    const entryId = initial?.entries[0]?.id;
    const llmClient = new FakeLLMClient({
      responses: [
        emitSharedStateArtifactPatchResponse({ operations: [] }),
        emitSharedStateArtifactPatchResponse({ operations: [] }),
      ],
    });

    await compileSharedStateArtifact({
      ...baseInput(llmClient),
      turnCounter: 10,
      renderOptions: {
        currentTurnCounter: 10,
        lastUpdatedTurnByStreamEntryId: {
          [oldSource]: 1,
        },
      },
      lifecycle: {
        recentTurnThreshold: 5,
        dormantTurnThreshold: 15,
      },
    });

    expect(repository.get(audience)?.entries.find((entry) => entry.id === entryId)?.kind).toBe(
      "low_salience_live",
    );

    await compileSharedStateArtifact({
      ...baseInput(llmClient),
      turnCounter: 20,
      renderOptions: {
        currentTurnCounter: 20,
        lastUpdatedTurnByStreamEntryId: {
          [oldSource]: 1,
        },
      },
      lifecycle: {
        recentTurnThreshold: 5,
        dormantTurnThreshold: 15,
      },
    });

    expect(repository.get(audience)?.entries.find((entry) => entry.id === entryId)?.kind).toBe(
      "dormant_live",
    );
  });
});
