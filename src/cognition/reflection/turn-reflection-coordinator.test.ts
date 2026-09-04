import { describe, expect, it, vi } from "vitest";

import { createWorkingMemory } from "../../memory/working/index.js";
import type { StreamEntry } from "../../stream/index.js";
import { FixedClock } from "../../util/clock.js";
import { createEntityId, createSessionId, createStreamEntryId } from "../../util/ids.js";
import type { PerceptionResult } from "../types.js";
import type { Reflector } from "./index.js";
import { TurnReflectionCoordinator } from "./turn-reflection-coordinator.js";

describe("TurnReflectionCoordinator", () => {
  it("separates turn-produced stream and journal artifacts from inbound evidence", async () => {
    const sessionId = createSessionId();
    const inboundEntryId = createStreamEntryId();
    const persistedAgentEntry = {
      id: createStreamEntryId(),
      timestamp: 1_000,
      kind: "internal_event",
      content: { kind: "train_of_thought_continued" },
      session_id: sessionId,
    } as StreamEntry;
    const workingMemory = createWorkingMemory(sessionId, 1_000);
    const effects = {
      createdActionIds: [],
      createdExecutiveStepIds: [],
      createdOpenQuestionIds: [],
      updatedExecutiveSteps: [],
      updatedGoals: [],
      retiredGoalIds: [],
      resolvedOpenQuestions: [],
      updatedEpisodeStats: [],
    };
    const reflect = vi.fn(async () => ({ workingMemory, effects }));
    const coordinator = new TurnReflectionCoordinator({
      moodRepository: { update: vi.fn() },
      socialRepository: { recordInteractionWithId: vi.fn() },
      openQuestionsRepository: { list: vi.fn(() => []) },
      workingMemoryStore: { save: vi.fn() },
      pendingProceduralAttemptTracker: { update: vi.fn(() => []) },
      createReflector: vi.fn(() => ({ reflect }) as unknown as Reflector),
      clock: new FixedClock(1_000),
      tracer: {
        enabled: false,
        includePayloads: false,
        emit: vi.fn(),
      },
    });

    await coordinator.run({
      llmClient: {} as never,
      sessionId,
      turnId: "turn-autonomous",
      origin: "autonomous",
      userMessage: "",
      perception: {} as PerceptionResult,
      workingMood: {
        valence: 0,
        arousal: 0,
        dominant_emotion: "neutral",
      },
      postActionWorkingMemory: workingMemory,
      selfSnapshot: {} as never,
      deliberation: { retrievedEpisodes: [] } as never,
      actionResult: {
        response: "",
        tool_calls: [],
        intents: [],
        workingMemory,
      },
      retrievedEpisodes: [],
      retrievalConfidence: {
        confidence: 1,
        degraded: false,
        reason: null,
      } as never,
      executiveFocus: null as never,
      selectedSkill: null,
      proceduralContext: null,
      audienceEntityId: null,
      socialInteractionEntityId: null,
      pendingSocialAttribution: null,
      suppressionSet: { snapshot: () => [] } as never,
      persistedUserEntryId: inboundEntryId,
      persistedAgentEntry,
      currentTurnJournalEntryIds: [23],
      isUserTurn: false,
      streamWriter: {} as never,
      onHookFailure: vi.fn(),
      trackReflectionEffects: vi.fn(),
    });

    expect(reflect).toHaveBeenCalledWith(
      expect.objectContaining({
        currentTurnStreamEntryIds: [inboundEntryId, persistedAgentEntry.id],
        currentTurnProducedStreamEntryIds: [persistedAgentEntry.id],
        currentTurnJournalEntryIds: [23],
      }),
      expect.anything(),
    );
  });

  it("skips target-memory reflection persistence for directed_outbound internal instructions", async () => {
    const addOpenQuestion = vi.fn();
    const resolveOpenQuestion = vi.fn();
    const recordInteractionWithId = vi.fn();
    const reflect = vi.fn(async () => {
      addOpenQuestion();
      resolveOpenQuestion();
      return {
        workingMemory: createWorkingMemory(createSessionId(), 1_000),
        effects: {
          createdActionIds: [],
          createdExecutiveStepIds: [],
          createdOpenQuestionIds: [],
          updatedExecutiveSteps: [],
          updatedGoals: [],
          retiredGoalIds: [],
          resolvedOpenQuestions: [],
          updatedEpisodeStats: [],
        },
      };
    });
    const trackReflectionEffects = vi.fn();
    const workingMemory = createWorkingMemory(createSessionId(), 1_000);
    const coordinator = new TurnReflectionCoordinator({
      moodRepository: {
        update: vi.fn(),
      },
      socialRepository: {
        recordInteractionWithId,
      },
      openQuestionsRepository: {
        list: vi.fn(() => []),
      },
      workingMemoryStore: {
        save: vi.fn(),
      },
      pendingProceduralAttemptTracker: {
        update: vi.fn(() => []),
      },
      createReflector: vi.fn(() => ({ reflect }) as unknown as Reflector),
      clock: new FixedClock(1_000),
      tracer: {
        enabled: false,
        includePayloads: false,
        emit: vi.fn(),
      },
    });
    const persistedAgentEntry = {
      id: createStreamEntryId(),
      timestamp: 1_000,
      kind: "agent_msg",
      content: "target message",
      session_id: createSessionId(),
    } as StreamEntry;

    const result = await coordinator.run({
      llmClient: {} as never,
      sessionId: createSessionId(),
      turnId: "turn-directed-outbound",
      origin: "directed_outbound",
      userMessage: "internal directing instruction with operator-only context",
      perception: {} as PerceptionResult,
      workingMood: {
        valence: 0,
        arousal: 0,
        dominant_emotion: "neutral",
      },
      postActionWorkingMemory: workingMemory,
      selfSnapshot: {} as never,
      deliberation: {
        retrievedEpisodes: [],
      } as never,
      actionResult: {
        response: "target message",
      } as never,
      retrievedEpisodes: [],
      retrievalConfidence: {
        confidence: 1,
        degraded: false,
        reason: null,
      } as never,
      executiveFocus: null as never,
      selectedSkill: null,
      proceduralContext: null,
      audienceEntityId: createEntityId(),
      socialInteractionEntityId: createEntityId(),
      pendingSocialAttribution: null,
      suppressionSet: {
        snapshot: () => [],
      } as never,
      persistedAgentEntry,
      isUserTurn: false,
      streamWriter: {} as never,
      onHookFailure: vi.fn(),
      trackReflectionEffects,
    });

    expect(result.workingMemory).toBe(workingMemory);
    expect(result.effects).toEqual({
      createdActionIds: [],
      createdExecutiveStepIds: [],
      createdOpenQuestionIds: [],
      updatedExecutiveSteps: [],
      updatedGoals: [],
      retiredGoalIds: [],
      resolvedOpenQuestions: [],
      updatedEpisodeStats: [],
    });
    expect(addOpenQuestion).not.toHaveBeenCalled();
    expect(resolveOpenQuestion).not.toHaveBeenCalled();
    expect(recordInteractionWithId).not.toHaveBeenCalled();
    expect(reflect).not.toHaveBeenCalled();
    expect(trackReflectionEffects).toHaveBeenCalledWith(result.effects);
  });
});
