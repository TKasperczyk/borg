import { describe, expect, it, vi } from "vitest";

import { FakeLLMClient } from "../../llm/test-support/fake-client.js";
import { TestEmbeddingClient, createOfflineTestHarness } from "../../offline/test-support.js";
import { FixedClock } from "../../util/clock.js";
import { createSessionId, createStreamEntryId } from "../../util/ids.js";
import { NOOP_TRACER } from "../tracing/tracer.js";
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
  it("skips action extraction when the current turn has a frame anomaly", async () => {
    const extractSpy = vi.spyOn(ActionStateExtractor.prototype, "extract");
    const llm = new FakeLLMClient();
    const service = new TurnActionStateService({
      model: "test-recall",
      actionRepository: { add: vi.fn() } as never,
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
        frameAnomaly: {
          status: "ok",
          kind: "frame_assignment_claim",
          confidence: 0.96,
          rationale: "The user-role message assigns the prior exchange to a roleplay frame.",
        },
      });

      expect(ids).toEqual([]);
      expect(extractSpy).not.toHaveBeenCalled();
      expect(llm.requests).toHaveLength(0);
    } finally {
      extractSpy.mockRestore();
    }
  });

  it("emits degraded extraction traces with session scope", async () => {
    const streamEntryId = createStreamEntryId();
    const sessionId = createSessionId();
    const emit = vi.fn();
    const llm = new FakeLLMClient({
      responses: [
        createActionStateResponse({
          action_states: "invalid",
        }),
      ],
    });
    const service = new TurnActionStateService({
      model: "test-recall",
      actionRepository: {
        add: vi.fn(),
        list: vi.fn(() => []),
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

  it("persists goal-linked extracted actions so completion resolves linked open questions", async () => {
    const streamEntryId = createStreamEntryId();
    const llm = new FakeLLMClient({
      responses: [
        createActionStateResponse({
          action_states: [
            {
              classification: "concrete_action",
              description: "Send the Atlas follow-up",
              actor: "borg",
              state: "committed_to_do",
              audience_entity_id: null,
              evidence_stream_entry_ids: [streamEntryId],
              confidence: 0.91,
            },
          ],
        }),
      ],
    });
    const harness = await createOfflineTestHarness({ llmClient: llm });

    try {
      const goal = harness.goalsRepository.add({
        description: "Resolve the Atlas follow-up",
        priority: 0.8,
        provenance: { kind: "manual" },
      });
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
        goalId: goal.id,
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
});
