import { describe, expect, it, vi } from "vitest";

import { createWorkingMemory } from "../../../memory/working/index.js";
import { createEntityId, createSessionId } from "../../../util/ids.js";

import { runExtractionPhase } from "./extraction-phase.js";

function firstMockCallInput(mock: { mock: { calls: unknown[][] } }): unknown {
  return mock.mock.calls[0]?.[0];
}

describe("runExtractionPhase", () => {
  it("skips user-world extraction and mutation for directed outbound instructions", async () => {
    const sessionId = createSessionId();
    const workingMemory = createWorkingMemory(sessionId, 1_000);
    const extractAndApply = vi.fn();
    const extractActionState = vi.fn();
    const extractAndPersistGoals = vi.fn();
    const extractAndPersistDirectives = vi.fn();
    const buildSelfContext = vi.fn();

    const result = await runExtractionPhase({
      options: {
        selfContextBuilder: {
          build: buildSelfContext,
          listActiveGoalsForCognition: vi.fn(),
        },
        correctivePreferenceTurnService: {
          extractAndApply,
        },
        turnActionStateService: {
          extract: extractActionState,
        },
        turnGoalPromotionService: {
          extractAndPersist: extractAndPersistGoals,
        },
        creatorDirectiveTurnService: {
          extractAndPersist: extractAndPersistDirectives,
        },
      } as never,
      appendHookFailureEvent: vi.fn(),
      llmClient: {} as never,
      turnId: "turn-directed-outbound",
      sessionId,
      turnInput: {
        userMessage: "Internal outbound composition instruction",
        origin: "directed_outbound",
      },
      isUserTurn: false,
      cognitionInput: "Internal outbound composition instruction",
      perception: {
        mode: "relational",
        entities: [],
        affectiveSignal: {
          valence: 0,
          arousal: 0,
          dominant_emotion: null,
        },
        temporalCue: null,
      },
      workingMemory,
      recentHistory: [],
      audienceEntityId: null,
      groupSpeakerEntityId: null,
      groupSpeakerDisplayName: null,
      currentSenderEntityId: null,
      currentSenderDisplayName: null,
      currentSenderBorgRole: null,
      sessionAudienceRole: "participant",
      participantRoster: null,
      currentTurnFrameAnomaly: null,
      streamWriter: {} as never,
      trackAppliedSlotNegation: vi.fn(),
    });

    expect(result).toEqual({
      actionLinkSelfContext: null,
      correctiveCommitment: null,
      correctiveCommitmentSupersession: null,
      correctiveCommitmentRetirement: null,
      workingMemory,
      createdActionIds: [],
      persistedPromotions: {
        goalIds: [],
        executiveStepIds: [],
      },
      creatorDirectives: [],
    });
    expect(buildSelfContext).not.toHaveBeenCalled();
    expect(extractAndApply).not.toHaveBeenCalled();
    expect(extractActionState).not.toHaveBeenCalled();
    expect(extractAndPersistGoals).not.toHaveBeenCalled();
    expect(extractAndPersistDirectives).not.toHaveBeenCalled();
  });

  it("suppresses cross-audience authority for multi-sender batches", async () => {
    const sessionId = createSessionId();
    const workingMemory = createWorkingMemory(sessionId, 1_000);
    const extractAndApply = vi.fn(async () => ({
      commitment: null,
      commitmentSupersession: null,
      commitmentRetirement: null,
      workingMemory,
    }));
    const extractActionState = vi.fn(async () => []);
    const extractGoals = vi.fn(async () => ({
      goalIds: [],
      executiveStepIds: [],
    }));
    const extractDirectives = vi.fn(async () => []);
    const creatorId = createEntityId();

    await runExtractionPhase({
      options: {
        selfContextBuilder: {
          build: vi.fn(async () => ({
            executiveFocus: { selected_goal: null },
          })),
          listActiveGoalsForCognition: vi.fn(async () => []),
        },
        correctivePreferenceTurnService: {
          extractAndApply,
        },
        turnActionStateService: {
          extract: extractActionState,
        },
        turnGoalPromotionService: {
          extractAndPersist: extractGoals,
        },
        creatorDirectiveTurnService: {
          extractAndPersist: extractDirectives,
        },
      } as never,
      appendHookFailureEvent: vi.fn(),
      llmClient: {} as never,
      turnId: "turn-multi-sender",
      sessionId,
      turnInput: {
        userMessage: "<inbound_batch>...</inbound_batch>",
        origin: "user",
      },
      isUserTurn: true,
      cognitionInput: "<inbound_batch>...</inbound_batch>",
      perception: {
        mode: "relational",
        entities: [],
        affectiveSignal: {
          valence: 0,
          arousal: 0,
          dominant_emotion: null,
        },
        temporalCue: null,
      },
      workingMemory,
      recentHistory: [],
      audienceEntityId: null,
      groupSpeakerEntityId: null,
      groupSpeakerDisplayName: null,
      currentSenderEntityId: creatorId,
      currentSenderDisplayName: "Creator",
      currentSenderBorgRole: "creator",
      sessionAudienceRole: "operator",
      participantRoster: null,
      persistedUserEntryId: undefined,
      sourceUserEntryIds: [],
      distinctSenderCount: 2,
      currentTurnFrameAnomaly: null,
      streamWriter: {} as never,
      trackAppliedSlotNegation: vi.fn(),
    });

    expect(extractAndApply).toHaveBeenCalledWith(
      expect.objectContaining({
        isUserTurn: true,
        currentSenderEntityId: null,
        currentSenderBorgRole: null,
        sessionAudienceRole: "operator",
        crossAudienceTargeting: expect.objectContaining({
          allowed: false,
          candidateAudiences: [],
        }),
      }),
    );
    expect(extractDirectives).toHaveBeenCalledWith(
      expect.objectContaining({
        currentSenderEntityId: null,
        currentSenderBorgRole: null,
        currentSenderDisplayName: null,
      }),
    );
  });

  it("preserves creator/operator authority for a single creator sender", async () => {
    const sessionId = createSessionId();
    const workingMemory = createWorkingMemory(sessionId, 1_000);
    const extractAndApply = vi.fn(async () => ({
      commitment: null,
      commitmentSupersession: null,
      commitmentRetirement: null,
      workingMemory,
    }));
    const extractActionState = vi.fn(async () => []);
    const extractGoals = vi.fn(async () => ({
      goalIds: [],
      executiveStepIds: [],
    }));
    const extractDirectives = vi.fn(async () => []);
    const creatorId = createEntityId();

    await runExtractionPhase({
      options: {
        sessionsRepository: {
          list: vi.fn(() => []),
        },
        selfContextBuilder: {
          build: vi.fn(async () => ({
            executiveFocus: { selected_goal: null },
          })),
          listActiveGoalsForCognition: vi.fn(async () => []),
        },
        correctivePreferenceTurnService: {
          extractAndApply,
        },
        turnActionStateService: {
          extract: extractActionState,
        },
        turnGoalPromotionService: {
          extractAndPersist: extractGoals,
        },
        creatorDirectiveTurnService: {
          extractAndPersist: extractDirectives,
        },
      } as never,
      appendHookFailureEvent: vi.fn(),
      llmClient: {} as never,
      turnId: "turn-single-creator",
      sessionId,
      turnInput: {
        userMessage: "<inbound_batch>...</inbound_batch>",
        origin: "user",
      },
      isUserTurn: true,
      cognitionInput: "<inbound_batch>...</inbound_batch>",
      perception: {
        mode: "relational",
        entities: [],
        affectiveSignal: {
          valence: 0,
          arousal: 0,
          dominant_emotion: null,
        },
        temporalCue: null,
      },
      workingMemory,
      recentHistory: [],
      audienceEntityId: null,
      groupSpeakerEntityId: creatorId,
      groupSpeakerDisplayName: "Creator",
      currentSenderEntityId: creatorId,
      currentSenderDisplayName: "Creator",
      currentSenderBorgRole: "creator",
      sessionAudienceRole: "operator",
      participantRoster: null,
      sourceUserEntryIds: [],
      distinctSenderCount: 1,
      currentTurnFrameAnomaly: null,
      streamWriter: {} as never,
      trackAppliedSlotNegation: vi.fn(),
    });

    expect(extractAndApply).toHaveBeenCalledWith(
      expect.objectContaining({
        isUserTurn: true,
        currentSenderEntityId: creatorId,
        currentSenderBorgRole: "creator",
        sessionAudienceRole: "operator",
        crossAudienceTargeting: expect.objectContaining({
          allowed: true,
        }),
      }),
    );
    expect(extractDirectives).toHaveBeenCalledWith(
      expect.objectContaining({
        currentSenderEntityId: creatorId,
        currentSenderBorgRole: "creator",
        currentSenderDisplayName: "Creator",
      }),
    );
    for (const extractorInput of [
      firstMockCallInput(extractAndApply),
      firstMockCallInput(extractActionState),
      firstMockCallInput(extractGoals),
      firstMockCallInput(extractDirectives),
    ]) {
      expect(extractorInput).toBeDefined();
      expect(extractorInput).not.toHaveProperty("evidenceLedger");
      expect(extractorInput).not.toHaveProperty("audienceStanding");
      expect(extractorInput).not.toHaveProperty("observedEventIntrospection");
    }
  });
});
