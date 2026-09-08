import { describe, expect, it, vi } from "vitest";

import { createWorkingMemory } from "../../memory/working/index.js";
import type { StreamEntry } from "../../stream/index.js";
import { FixedClock } from "../../util/clock.js";
import { createEntityId, createSessionId, createStreamEntryId } from "../../util/ids.js";
import { renderInboundBatch } from "../turn-input.js";
import type { PerceptionResult } from "../types.js";
import type { Reflector } from "./index.js";
import {
  MOOD_TRIGGER_REASON_LIMIT,
  TurnReflectionCoordinator,
} from "./turn-reflection-coordinator.js";

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

  // `mood_history.reason` renders as `trigger=` on `borg_affective_trajectory`, quoted
  // beside the numbers as if it named what moved them. Sliced off the rendered batch it
  // named the envelope instead: the `<inbound_batch>` tag plus an `<inbound_message>`
  // attribute list runs past 120 characters before the first body character, so every
  // stored trigger on a wrapping transport stopped inside a stream id. Read the entries
  // the renderer wrapped instead of a prefix of its output.
  describe("mood trigger text", () => {
    function runUserTurn(input: {
      userMessage: string;
      sourceUserEntries?: readonly StreamEntry[];
    }): ReturnType<typeof vi.fn> {
      const sessionId = createSessionId();
      const workingMemory = createWorkingMemory(sessionId, 1_000);
      const update = vi.fn(() => ({
        session_id: sessionId,
        valence: 0.2,
        arousal: 0.3,
        updated_at: 1_000,
        half_life_hours: 6,
        recent_triggers: [] as string[],
      }));
      const coordinator = new TurnReflectionCoordinator({
        moodRepository: { update },
        socialRepository: { recordInteractionWithId: vi.fn() },
        openQuestionsRepository: { list: vi.fn(() => []) },
        workingMemoryStore: { save: vi.fn() },
        pendingProceduralAttemptTracker: { update: vi.fn(() => []) },
        createReflector: vi.fn(
          () =>
            ({
              reflect: vi.fn(async () => ({
                workingMemory,
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
              })),
            }) as unknown as Reflector,
        ),
        clock: new FixedClock(1_000),
        tracer: { enabled: false, includePayloads: false, emit: vi.fn() },
      });

      // The mood write happens before `run` reaches its first await, so the spy is
      // populated by the time this returns; the rest of the turn needs wiring this
      // case does not exercise, so let it settle unobserved.
      void coordinator
        .run({
        llmClient: {} as never,
        sessionId,
        turnId: "turn-user",
        origin: "user",
        userMessage: input.userMessage,
        perception: {
          affectiveSignal: { valence: 0.2, arousal: 0.3, dominant_emotion: "neutral" },
        } as PerceptionResult,
        workingMood: { valence: 0, arousal: 0, dominant_emotion: "neutral" },
        postActionWorkingMemory: workingMemory,
        selfSnapshot: {} as never,
        deliberation: { retrievedEpisodes: [] } as never,
        actionResult: { response: "", tool_calls: [], intents: [], workingMemory },
        retrievedEpisodes: [],
        retrievalConfidence: { confidence: 1, degraded: false, reason: null } as never,
        executiveFocus: null as never,
        selectedSkill: null,
        proceduralContext: null,
        audienceEntityId: null,
        socialInteractionEntityId: null,
        pendingSocialAttribution: null,
        suppressionSet: { snapshot: () => [] } as never,
        ...(input.sourceUserEntries === undefined
          ? {}
          : { sourceUserEntries: input.sourceUserEntries }),
        persistedAgentEntry: {
          id: createStreamEntryId(),
          timestamp: 1_000,
          kind: "agent_msg",
          content: "ok",
          session_id: sessionId,
        } as StreamEntry,
        isUserTurn: true,
        streamWriter: {} as never,
        onHookFailure: vi.fn(),
        trackReflectionEffects: vi.fn(),
        })
        .catch(() => undefined);

      return update;
    }

    function batchEntry(input: { entryIndex: number; content: unknown }): StreamEntry {
      return {
        id: createStreamEntryId(),
        session_id: createSessionId(),
        entry_index: input.entryIndex,
        timestamp: 1_000 + input.entryIndex,
        kind: "user_msg",
        content: input.content,
      } as unknown as StreamEntry;
    }

    it("names the arrived bodies rather than the envelope that carried them", () => {
      const entries = [
        batchEntry({ entryIndex: 1, content: "The deadline moved." }),
      ];
      const update = runUserTurn({
        userMessage: renderInboundBatch({ entries: entries as never }),
        sourceUserEntries: entries,
      });

      expect(update).toHaveBeenCalledWith(
        expect.anything(),
        expect.objectContaining({ reason: "The deadline moved." }),
      );
    });

    it("orders bodies by entry index so the head slice starts at the first message", () => {
      const update = runUserTurn({
        userMessage: "<inbound_batch kind=\"stream_backlog\" count=\"2\">",
        sourceUserEntries: [
          batchEntry({ entryIndex: 2, content: "second" }),
          batchEntry({ entryIndex: 1, content: "first" }),
        ],
      });

      expect(update).toHaveBeenCalledWith(
        expect.anything(),
        expect.objectContaining({ reason: "first\nsecond" }),
      );
    });

    it("falls back to the message when no batch entries were wrapped", () => {
      const update = runUserTurn({ userMessage: "a scalar message" });

      expect(update).toHaveBeenCalledWith(
        expect.anything(),
        expect.objectContaining({ reason: "a scalar message" }),
      );
    });

    it("falls back to the message when no entry carries text content", () => {
      const update = runUserTurn({
        userMessage: "rendered fallback",
        sourceUserEntries: [
          batchEntry({ entryIndex: 1, content: { kind: "structured" } }),
        ],
      });

      expect(update).toHaveBeenCalledWith(
        expect.anything(),
        expect.objectContaining({ reason: "rendered fallback" }),
      );
    });

    it("caps the trigger at the stored width", () => {
      const body = "x".repeat(400);
      const update = runUserTurn({
        userMessage: "envelope",
        sourceUserEntries: [batchEntry({ entryIndex: 1, content: body })],
      });

      expect(update).toHaveBeenCalledWith(
        expect.anything(),
        expect.objectContaining({ reason: "x".repeat(MOOD_TRIGGER_REASON_LIMIT) }),
      );
    });
  });
});
