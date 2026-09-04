import { describe, expect, it, vi } from "vitest";

import { createWorkingMemory, type WorkingMemory } from "../../../memory/working/index.js";
import type { StreamEntry } from "../../../stream/index.js";
import { createEntityId, createSessionId, createStreamEntryId } from "../../../util/ids.js";
import type { SessionId } from "../../../util/ids.js";

import { runExtractionPhase } from "./extraction-phase.js";

function firstMockCallInput(mock: { mock: { calls: unknown[][] } }): unknown {
  return mock.mock.calls[0]?.[0];
}

function baseExtractionPhaseInput(input: {
  sessionId: SessionId;
  workingMemory: WorkingMemory;
  extractActionState: Parameters<
    typeof runExtractionPhase
  >[0]["options"]["turnActionStateService"]["extract"];
}): Parameters<typeof runExtractionPhase>[0] {
  return {
    options: {
      selfContextBuilder: {
        build: vi.fn(async () => ({ executiveFocus: { selected_goal: null } })),
        listActiveGoalsForCognition: vi.fn(async () => []),
      },
      correctivePreferenceTurnService: {
        extractAndApply: vi.fn(async () => ({
          commitment: null,
          commitmentSupersession: null,
          commitmentRetirement: null,
          workingMemory: input.workingMemory,
        })),
      },
      turnActionStateService: { extract: input.extractActionState },
      turnGoalPromotionService: {
        extractAndPersist: vi.fn(async () => ({ goalIds: [], executiveStepIds: [] })),
      },
      creatorDirectiveTurnService: { extractAndPersist: vi.fn(async () => []) },
    } as never,
    appendHookFailureEvent: vi.fn(),
    llmClient: {} as never,
    turnId: "turn-speaker",
    sessionId: input.sessionId,
    turnInput: { userMessage: "message", origin: "user" } as never,
    isUserTurn: true,
    cognitionInput: "message",
    perception: {
      mode: "relational",
      entities: [],
      affectiveSignal: { valence: 0, arousal: 0, dominant_emotion: null },
      temporalCue: null,
    } as never,
    workingMemory: input.workingMemory,
    recentHistory: [],
    audienceEntityId: null,
    groupSpeakerEntityId: null,
    groupSpeakerDisplayName: null,
    currentSenderEntityId: null,
    currentSenderDisplayName: null,
    currentSenderBorgRole: null,
    sessionAudienceRole: "participant",
    participantRoster: null,
    persistedUserEntryId: undefined,
    sourceUserEntryIds: [],
    currentTurnFrameAnomaly: null,
    streamWriter: {} as never,
    trackAppliedSlotNegation: vi.fn(),
  };
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

  it("names the one-to-one sender as the action-state speaker", async () => {
    const sessionId = createSessionId();
    const workingMemory = createWorkingMemory(sessionId, 1_000);
    const extractActionState = vi.fn(async () => []);
    const senderId = createEntityId();
    const entryId = createStreamEntryId();
    const sourceEntry = {
      id: entryId,
      timestamp: 900,
      kind: "user_msg",
      content: "message",
      turn_status: "active",
      sender_entity_id: senderId,
      reply_target_entity_id: null,
      session_id: sessionId,
      compressed: false,
    } satisfies StreamEntry;
    const phaseInput = baseExtractionPhaseInput({ sessionId, workingMemory, extractActionState });

    await runExtractionPhase({
      ...phaseInput,
      // No group speaker: a one-to-one audience has none, and the sender is
      // only recoverable from the turn's own entries.
      groupSpeakerEntityId: null,
      groupSpeakerDisplayName: null,
      currentSenderEntityId: senderId,
      currentSenderDisplayName: "Peer",
      sourceUserEntries: [sourceEntry],
      sourceUserEntryIds: [entryId],
      senderAttribution: [{ entryId, senderEntityId: senderId, senderDisplayName: "Peer" }],
      distinctSenderCount: 1,
    });

    expect(extractActionState).toHaveBeenCalledWith(
      expect.objectContaining({
        speakerEntityId: senderId,
        speakerDisplayName: "Peer",
      }),
    );
    expect(phaseInput.options.turnGoalPromotionService.extractAndPersist).toHaveBeenCalledWith(
      expect.objectContaining({
        speakerEntityId: senderId,
        speakerDisplayName: "Peer",
        sourceUserEntries: [sourceEntry],
        senderAttribution: [{ entryId, senderEntityId: senderId, senderDisplayName: "Peer" }],
      }),
    );
    expect(phaseInput.options.correctivePreferenceTurnService.extractAndApply).toHaveBeenCalledWith(
      expect.objectContaining({
        committedByEntityId: senderId,
        speakerDisplayName: "Peer",
        sourceUserEntries: [sourceEntry],
        senderAttribution: [{ entryId, senderEntityId: senderId, senderDisplayName: "Peer" }],
      }),
    );
  });

  it("passes a group speaker's attributed source entry to both durable extractors", async () => {
    const sessionId = createSessionId();
    const workingMemory = createWorkingMemory(sessionId, 1_000);
    const extractActionState = vi.fn(async () => []);
    const speakerId = createEntityId();
    const audienceId = createEntityId();
    const entryId = createStreamEntryId();
    const sourceEntry = {
      id: entryId,
      timestamp: 910,
      kind: "user_msg",
      content: "group message",
      turn_status: "active",
      audience: "Group room",
      sender_entity_id: speakerId,
      reply_target_entity_id: null,
      session_id: sessionId,
      compressed: false,
    } satisfies StreamEntry;
    const senderAttribution = [
      { entryId, senderEntityId: speakerId, senderDisplayName: "Group speaker" },
    ];
    const phaseInput = baseExtractionPhaseInput({ sessionId, workingMemory, extractActionState });

    await runExtractionPhase({
      ...phaseInput,
      audienceEntityId: audienceId,
      groupSpeakerEntityId: speakerId,
      groupSpeakerDisplayName: "Group speaker",
      currentSenderEntityId: speakerId,
      currentSenderDisplayName: "Group speaker",
      sourceUserEntries: [sourceEntry],
      sourceUserEntryIds: [entryId],
      senderAttribution,
      distinctSenderCount: 1,
    });

    expect(phaseInput.options.turnGoalPromotionService.extractAndPersist).toHaveBeenCalledWith(
      expect.objectContaining({
        audienceEntityId: audienceId,
        speakerEntityId: speakerId,
        speakerDisplayName: "Group speaker",
        sourceUserEntries: [sourceEntry],
        sourceUserEntryIds: [entryId],
        senderAttribution,
      }),
    );
    expect(phaseInput.options.correctivePreferenceTurnService.extractAndApply).toHaveBeenCalledWith(
      expect.objectContaining({
        audienceEntityId: audienceId,
        committedByEntityId: speakerId,
        speakerDisplayName: "Group speaker",
        sourceUserEntries: [sourceEntry],
        sourceUserEntryIds: [entryId],
        senderAttribution,
      }),
    );
  });

  it("leaves the action-state speaker unnamed when a batch mixes senders", async () => {
    const sessionId = createSessionId();
    const workingMemory = createWorkingMemory(sessionId, 1_000);
    const extractActionState = vi.fn(async () => []);
    const firstEntryId = createStreamEntryId();
    const secondEntryId = createStreamEntryId();
    const firstSenderId = createEntityId();
    const secondSenderId = createEntityId();
    const sourceUserEntries = [
      {
        id: firstEntryId,
        timestamp: 920,
        kind: "user_msg",
        content: "first message",
        turn_status: "active",
        sender_entity_id: firstSenderId,
        reply_target_entity_id: null,
        session_id: sessionId,
        compressed: false,
      },
      {
        id: secondEntryId,
        timestamp: 930,
        kind: "user_msg",
        content: "second message",
        turn_status: "active",
        sender_entity_id: secondSenderId,
        reply_target_entity_id: null,
        session_id: sessionId,
        compressed: false,
      },
    ] satisfies StreamEntry[];
    const senderAttribution = [
      { entryId: firstEntryId, senderEntityId: firstSenderId, senderDisplayName: "One" },
      { entryId: secondEntryId, senderEntityId: secondSenderId, senderDisplayName: "Two" },
    ];
    const phaseInput = baseExtractionPhaseInput({ sessionId, workingMemory, extractActionState });

    await runExtractionPhase({
      ...phaseInput,
      groupSpeakerEntityId: null,
      groupSpeakerDisplayName: null,
      currentSenderEntityId: null,
      currentSenderDisplayName: null,
      sourceUserEntries,
      sourceUserEntryIds: [firstEntryId, secondEntryId],
      senderAttribution,
      distinctSenderCount: 2,
    });

    expect(extractActionState).toHaveBeenCalledWith(
      expect.objectContaining({
        speakerEntityId: null,
        speakerDisplayName: null,
        sourceUserEntryIds: [firstEntryId, secondEntryId],
        senderAttribution,
      }),
    );
    expect(phaseInput.options.turnGoalPromotionService.extractAndPersist).toHaveBeenCalledWith(
      expect.objectContaining({
        speakerEntityId: null,
        speakerDisplayName: null,
        sourceUserEntries,
        sourceUserEntryIds: [firstEntryId, secondEntryId],
        senderAttribution,
      }),
    );
    expect(phaseInput.options.correctivePreferenceTurnService.extractAndApply).toHaveBeenCalledWith(
      expect.objectContaining({
        committedByEntityId: null,
        speakerDisplayName: null,
        sourceUserEntries,
        sourceUserEntryIds: [firstEntryId, secondEntryId],
        senderAttribution,
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
