import { ActionRepository, actionMigrations } from "../../memory/actions/index.js";
import { openDatabase } from "../../storage/sqlite/index.js";
import { makeActionRecord } from "../../test-support/factories/memory.js";
import { makeLockedSharedStateEntry } from "../../test-support/factories/shared-state.js";
import { TurnLifecycleTracker } from "../lifecycle/turn-lifecycle-tracker.js";
import { createWorkingMemory } from "../../memory/working/index.js";
import { describe, expect, it, vi } from "vitest";

import { FakeLLMClient } from "../../llm/test-support/fake-client.js";
import { TestEmbeddingClient, createOfflineTestHarness } from "../../offline/test-support.js";
import { FixedClock } from "../../util/clock.js";
import {
  createActionId,
  createEntityId,
  createSessionId,
  createStreamEntryId,
} from "../../util/ids.js";
import { NOOP_TRACER } from "../../tracing/tracer.js";
import { ActionStateExtractor } from "./action-state-extractor.js";
import { TurnActionStateService } from "./turn-action-state-service.js";

function createActionStateResponse(input: unknown) {
  return {
    text: "",
    input_tokens: 10,
    output_tokens: 5,
    stop_reason: "tool_use" as const,
    tool_calls: [
      {
        id: "toolu_action_state",
        name: "EmitActionStates",
        input,
      },
    ],
  };
}

describe("TurnActionStateService", () => {
  it.each([
    { disposition: "annotate", hasAnomaly: true, extractionCalls: 1 },
    { disposition: "quarantine", hasAnomaly: true, extractionCalls: 0 },
    { disposition: undefined, hasAnomaly: true, extractionCalls: 0 },
    { disposition: "none", hasAnomaly: false, extractionCalls: 1 },
  ] as const)(
    "action extraction with $disposition and anomaly=$hasAnomaly",
    async ({ disposition, hasAnomaly, extractionCalls }) => {
      const extractSpy = vi.spyOn(ActionStateExtractor.prototype, "extract");
      const llm = new FakeLLMClient({
        responses: [createActionStateResponse({ action_states: [] })],
      });
      const service = new TurnActionStateService({
        model: "test-recall",
        actionRepository: {
          add: vi.fn(),
          list: vi.fn(() => []),
          count: () => 0,
          findByDescription: async () => [],
        } as never,
        embeddingClient: new TestEmbeddingClient(),
        clock: new FixedClock(1_000),
        tracer: NOOP_TRACER,
      });

      try {
        const ids = await service.extract({
          llmClient: llm,
          turnId: "turn_anomaly",
          isUserTurn: true,
          userMessage: "You were playing Tom.",
          persistedUserEntryId: createStreamEntryId(),
          recentHistory: [],
          audienceEntityId: null,
          frameAnomalyDisposition: disposition,
          frameAnomaly: hasAnomaly
            ? {
                status: "ok",
                kind: "frame_assignment_claim",
                confidence: 0.96,
                rationale: "The user-role message assigns the prior exchange to a roleplay frame.",
              }
            : null,
        });

        expect(ids).toEqual([]);
        expect(extractSpy).toHaveBeenCalledTimes(extractionCalls);
        expect(llm.requests).toHaveLength(extractionCalls);
      } finally {
        extractSpy.mockRestore();
      }
    },
  );

  it("emits degraded extraction traces with session scope", async () => {
    const streamEntryId = createStreamEntryId();
    const sessionId = createSessionId();
    const emit = vi.fn();
    const invalidResponse = createActionStateResponse({
      action_states: "invalid",
    });
    const llm = new FakeLLMClient({
      responses: [invalidResponse, invalidResponse],
    });
    const service = new TurnActionStateService({
      model: "test-recall",
      actionRepository: {
        add: vi.fn(),
        list: vi.fn(() => []),
        count: () => 0,
        findByDescription: async () => [],
      } as never,
      embeddingClient: new TestEmbeddingClient(),
      clock: new FixedClock(1_000),
      tracer: { enabled: true, includePayloads: false, emit },
    });

    const ids = await service.extract({
      llmClient: llm,
      turnId: "turn_action_degraded_session",
      sessionId,
      isUserTurn: true,
      userMessage: "I will send the Atlas follow-up.",
      persistedUserEntryId: streamEntryId,
      recentHistory: [],
      audienceEntityId: null,
    });

    expect(ids).toEqual([]);
    expect(emit).toHaveBeenCalledWith(
      "extraction.actions.degraded",
      expect.objectContaining({
        turnId: "turn_action_degraded_session",
        session_id: sessionId,
        reason: "invalid_payload",
      }),
    );
  });

  it("passes private active actions to the extractor reference context with disclosure metadata", async () => {
    const streamEntryId = createStreamEntryId();
    const actionSourceEntryId = createStreamEntryId();
    const alice = createEntityId();
    const privateDescription = "Prepare Alice private launch note";
    const llm = new FakeLLMClient({
      responses: [
        createActionStateResponse({
          action_states: [],
        }),
      ],
    });
    const harness = await createOfflineTestHarness({ llmClient: llm });

    try {
      harness.actionRepository.add({
        id: createActionId(),
        description: privateDescription,
        actor: "borg",
        audience_entity_id: alice,
        goal_id: null,
        open_question_id: null,
        state: "scheduled",
        confidence: 0.9,
        provenance_episode_ids: [],
        provenance_stream_entry_ids: [actionSourceEntryId],
        created_at: 1_000,
        updated_at: 1_000,
        considering_at: null,
        committed_at: null,
        scheduled_at: 1_000,
        completed_at: null,
        not_done_at: null,
        declined_at: null,
        expired_at: null,
        archived_at: null,
        unknown_at: null,
        canonicalized_by_artifact_entry_id: null,
        session_scope: null,
        session_anchor_id: null,
        last_referenced_at_ms: 1_000,
        last_referenced_turn_counter: null,
      });
      const service = new TurnActionStateService({
        model: "test-recall",
        actionRepository: harness.actionRepository,
        embeddingClient: harness.embeddingClient,
        clock: harness.clock,
        tracer: NOOP_TRACER,
      });

      await service.extract({
        llmClient: llm,
        turnId: "turn_private_action_reference",
        isUserTurn: true,
        userMessage: "What action memory is active?",
        persistedUserEntryId: streamEntryId,
        recentHistory: [],
        audienceEntityId: null,
      });

      const payload = JSON.parse(String(llm.requests[0]?.messages[0]?.content ?? "{}")) as {
        active_actions_for_reference?: Array<Record<string, unknown>>;
      };

      expect(payload.active_actions_for_reference).toEqual([
        expect.objectContaining({
          description: privateDescription,
          disclosure: expect.stringContaining("disclosure_class=relationship_private"),
          disclosure_label: expect.objectContaining({
            disclosure_class: "relationship_private",
            private_to_entity_ids: [alice],
          }),
        }),
      ]);
    } finally {
      await harness.cleanup();
    }
  });

  it("persists goal-linked extracted actions so completion resolves linked open questions", async () => {
    const streamEntryId = createStreamEntryId();
    const llm = new FakeLLMClient();
    const harness = await createOfflineTestHarness({ llmClient: llm });

    try {
      const goal = harness.goalsRepository.add({
        description: "Resolve the Atlas follow-up",
        priority: 0.8,
        provenance: { kind: "manual" },
      });
      llm.pushResponse(
        createActionStateResponse({
          action_states: [
            {
              classification: "concrete_action",
              description: "Send the Atlas follow-up",
              actor: "borg",
              state: "committed_to_do",
              goal_id: goal.id,
              audience_entity_id: null,
              evidence_stream_entry_ids: [streamEntryId],
              confidence: 0.91,
            },
          ],
        }),
      );
      const question = harness.openQuestionsRepository.add({
        question: "Was the Atlas follow-up sent?",
        urgency: 0.7,
        goal_id: goal.id,
        source: "reflection",
        provenance: { kind: "manual" },
      });
      const service = new TurnActionStateService({
        model: "test-recall",
        actionRepository: harness.actionRepository,
        embeddingClient: harness.embeddingClient,
        clock: harness.clock,
        tracer: NOOP_TRACER,
      });

      const ids = await service.extract({
        llmClient: llm,
        turnId: "turn_goal_link",
        isUserTurn: true,
        userMessage: "I will send the Atlas follow-up.",
        persistedUserEntryId: streamEntryId,
        recentHistory: [],
        audienceEntityId: null,
        suppliedGoals: [goal],
      });

      expect(ids).toHaveLength(1);
      const actionId = ids[0];

      if (actionId === undefined) {
        throw new Error("Expected an extracted action id");
      }

      expect(harness.actionRepository.get(actionId)).toMatchObject({
        goal_id: goal.id,
        open_question_id: null,
        state: "committed_to_do",
      });

      harness.actionRepository.update(actionId, {
        state: "completed",
      });

      expect(harness.openQuestionsRepository.get(question.id)).toMatchObject({
        status: "resolved",
        resolution_evidence_stream_entry_ids: [streamEntryId],
      });
    } finally {
      await harness.cleanup();
    }
  });
  it.each([
    "unrelated",
    "continued",
    "new",
    "completed",
    "shared_state",
    "empty",
    "empty_shared_state",
  ] as const)("uses the existing extraction call sites for %s", async (scenario) => {
    const db = openDatabase(":memory:", { migrations: actionMigrations });
    try {
      const repository = new ActionRepository({ db });
      const source = createStreamEntryId();
      const prior = makeActionRecord({ actor: "borg" });
      if (scenario !== "new" && scenario !== "empty" && scenario !== "empty_shared_state")
        repository.add(prior);
      const assertsAction = ["continued", "new", "completed"].includes(scenario);
      const llm = new FakeLLMClient({
        responses: [
          createActionStateResponse({
            action_states: assertsAction
              ? [
                  {
                    classification: "concrete_action",
                    description: "Current wording",
                    actor: "borg",
                    state: scenario === "completed" ? "completed" : "scheduled",
                    matched_existing_action_id: scenario === "new" ? null : prior.id,
                    evidence_stream_entry_ids: [source],
                    confidence: 0.9,
                  },
                ]
              : [],
          }),
          createActionStateResponse({ action_states: [] }),
        ],
      });
      const service = new TurnActionStateService({
        actionRepository: repository,
        embeddingClient: new TestEmbeddingClient(),
        clock: new FixedClock(2_000),
        model: "mock",
        tracer: NOOP_TRACER,
      });
      await service.extract({
        llmClient: llm,
        turnId: "budget",
        isUserTurn: true,
        userMessage: "opaque",
        persistedUserEntryId: source,
        recentHistory: [],
        audienceEntityId: null,
      });
      if (scenario === "continued")
        expect(repository.get(prior.id)?.provenance_stream_entry_ids).toContain(source);
      await service.closeBorgSelfPerformedActions({
        llmClient: llm,
        turnId: "budget",
        userMessage: "opaque",
        persistedUserEntryId: source,
        persistedAgentEntryId: createStreamEntryId(),
        agentResponse: "opaque",
        recentHistory: [],
        audienceEntityId: null,
        currentTurnSharedStateEntries:
          scenario === "shared_state" || scenario === "empty_shared_state"
            ? [makeLockedSharedStateEntry()]
            : [],
      });
      expect(llm.requests).toHaveLength(scenario === "new" || scenario === "shared_state" ? 2 : 1);
    } finally {
      db.close();
    }
  });

  it.each([false, true])(
    "returns committed post-turn IDs for abort cleanup when completion hook throws=%s",
    async (hookThrows) => {
      const db = openDatabase(":memory:", { migrations: actionMigrations });
      try {
        const onCompleted = vi.fn(() => {
          if (hookThrows) throw new Error("question resolution failed");
        });
        const repository = new ActionRepository({ db, onCompleted });
        const emit = vi.fn();
        const source = createStreamEntryId(),
          agentSource = createStreamEntryId();
        const prior = makeActionRecord({ actor: "borg", provenance_stream_entry_ids: [source] });
        repository.add(prior);
        const llm = new FakeLLMClient({
          responses: [
            createActionStateResponse({
              action_states: [
                {
                  classification: "concrete_action",
                  description: "Closing wording",
                  actor: "borg",
                  state: "completed",
                  matched_existing_action_id: createActionId(),
                  evidence_stream_entry_ids: [agentSource],
                  confidence: 0.9,
                },
              ],
            }),
          ],
        });
        const service = new TurnActionStateService({
          actionRepository: repository,
          embeddingClient: new TestEmbeddingClient(),
          clock: new FixedClock(2_000),
          model: "mock",
          tracer: { enabled: true, includePayloads: false, emit },
        });
        const inserted = await service.closeBorgSelfPerformedActions({
          llmClient: llm,
          turnId: "aborted",
          userMessage: "opaque",
          persistedUserEntryId: source,
          persistedAgentEntryId: agentSource,
          agentResponse: "opaque",
          recentHistory: [],
          audienceEntityId: null,
          currentTurnSharedStateEntries: [],
        });
        expect(inserted).toHaveLength(1);
        expect(onCompleted).toHaveBeenCalledOnce();
        if (hookThrows)
          expect(emit).toHaveBeenCalledWith(
            "action_state.completion_hook_failed",
            expect.objectContaining({
              action_id: inserted[0],
              error: "question resolution failed",
            }),
          );
        expect(emit).toHaveBeenCalledWith(
          "extraction.actions.selection_summary",
          expect.objectContaining({ inserted: 1, matched: 0 }),
        );
        expect(repository.get(inserted[0]!)?.selection_audit?.rejection_reason).toBe("not_offered");
        const wm = createWorkingMemory(createSessionId(), 1_000);
        const tracker = new TurnLifecycleTracker({
          actionRepository: repository,
          workingMemoryStore: {
            load: () => wm,
            save: (value) => value,
            recordPendingActionMerges: () => wm,
          },
          executiveStepsRepository: {} as never,
          goalsRepository: {} as never,
          openQuestionsRepository: {} as never,
          episodicRepository: {} as never,
          relationalSlotRepository: {} as never,
        });
        tracker.captureInitialWorkingMemory(wm);
        tracker.trackCreatedActionIds(inserted);
        expect(
          await tracker.cleanupAbortedTurnState({ turnId: "aborted", sessionId: wm.session_id }),
        ).toEqual([]);
        expect(repository.get(inserted[0]!)).toBeNull();
        expect(repository.get(prior.id)).toEqual({ ...prior, last_referenced_turn_global: null });
      } finally {
        db.close();
      }
    },
  );
});
