import { describe, expect, it, vi } from "vitest";

import { createWorkingMemory } from "../../../memory/working/index.js";
import { createSessionId } from "../../../util/ids.js";

import { runExtractionPhase } from "./extraction-phase.js";

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
          listActiveGoalsVisibleToAudience: vi.fn(),
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
});
