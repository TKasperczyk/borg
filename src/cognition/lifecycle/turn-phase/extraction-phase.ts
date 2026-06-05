import type { CorrectivePreferenceTurnService } from "../../commitments/corrective-preference-service.js";
import type { CreatorDirectiveTurnService } from "../../creator-directives/service.js";
import type { ActualFrameAnomalyClassification } from "../../frame-anomaly/index.js";
import type { ParticipantRoster } from "../../perception/index.js";
import type { RecencyMessage } from "../../recency/index.js";
import type { PerceptionResult } from "../../types.js";
import type { LLMClient } from "../../../llm/index.js";
import type { StreamEntry, StreamWriter } from "../../../stream/index.js";
import type { EntityId, SessionId, StreamEntryId } from "../../../util/ids.js";
import type { WorkingMemory } from "../../../memory/working/index.js";
import type { BorgRole } from "../../../memory/commitments/index.js";
import type { SessionAudienceRole } from "../../../sessions/index.js";
import { runsExtraction } from "../../types.js";
import { isCreatorInOperatorContext } from "../../authority.js";
import type { CurrentTurnUserInputSenderAttribution } from "../../turn-input.js";
import type { TurnPhaseCoordinatorOptions, TurnPhaseInput } from "./types.js";
import type { AppendHookFailureEvent } from "./utils.js";

// Most recently active other audiences offered to the corrective-preference
// extractor as cross-audience targets (only on creator-in-operator turns).
const CROSS_AUDIENCE_TARGET_CAP = 12;

function dedupeCrossAudienceTargets(
  targets: readonly { entity_id: EntityId; label: string }[],
): { entity_id: EntityId; label: string }[] {
  const byEntityId = new Map<EntityId, { entity_id: EntityId; label: string }>();

  for (const target of targets) {
    if (!byEntityId.has(target.entity_id)) {
      byEntityId.set(target.entity_id, target);
    }
  }

  return [...byEntityId.values()];
}

export type TurnExtractionPhaseResult = {
  actionLinkSelfContext: Awaited<
    ReturnType<TurnPhaseCoordinatorOptions["selfContextBuilder"]["build"]>
  > | null;
  correctiveCommitment: Parameters<
    CorrectivePreferenceTurnService["persistCommitment"]
  >[0]["commitment"];
  correctiveCommitmentSupersession: Parameters<
    CorrectivePreferenceTurnService["persistCommitment"]
  >[0]["supersession"];
  correctiveCommitmentRetirement: Parameters<
    CorrectivePreferenceTurnService["persistCommitment"]
  >[0]["retirement"];
  workingMemory: WorkingMemory;
  createdActionIds: Awaited<
    ReturnType<TurnPhaseCoordinatorOptions["turnActionStateService"]["extract"]>
  >;
  persistedPromotions: Awaited<
    ReturnType<TurnPhaseCoordinatorOptions["turnGoalPromotionService"]["extractAndPersist"]>
  >;
  creatorDirectives: Awaited<ReturnType<CreatorDirectiveTurnService["extractAndPersist"]>>;
};

export async function runExtractionPhase(input: {
  options: TurnPhaseCoordinatorOptions;
  appendHookFailureEvent: AppendHookFailureEvent;
  llmClient: LLMClient;
  turnId: string;
  sessionId: SessionId;
  turnInput: TurnPhaseInput;
  isUserTurn: boolean;
  cognitionInput: string;
  perception: PerceptionResult;
  workingMemory: WorkingMemory;
  recentHistory: readonly RecencyMessage[];
  audienceEntityId: EntityId | null;
  groupSpeakerEntityId: EntityId | null;
  groupSpeakerDisplayName: string | null;
  currentSenderEntityId: EntityId | null;
  currentSenderDisplayName: string | null;
  currentSenderBorgRole: BorgRole | null;
  sessionAudienceRole: SessionAudienceRole;
  participantRoster: ParticipantRoster | null;
  persistedUserEntryId?: StreamEntry["id"];
  sourceUserEntryIds?: readonly StreamEntryId[];
  senderAttribution?: readonly CurrentTurnUserInputSenderAttribution[];
  distinctSenderCount?: number;
  currentTurnFrameAnomaly: ActualFrameAnomalyClassification | null;
  streamWriter: StreamWriter;
  trackAppliedSlotNegation: Parameters<
    CorrectivePreferenceTurnService["extractAndApply"]
  >[0]["trackAppliedSlotNegation"];
}): Promise<TurnExtractionPhaseResult> {
  if (!runsExtraction(input.turnInput.origin)) {
    return {
      actionLinkSelfContext: null,
      correctiveCommitment: null,
      correctiveCommitmentSupersession: null,
      correctiveCommitmentRetirement: null,
      workingMemory: input.workingMemory,
      createdActionIds: [],
      persistedPromotions: {
        goalIds: [],
        executiveStepIds: [],
      },
      creatorDirectives: [],
    };
  }

  const actionLinkSelfContext =
    input.isUserTurn && input.currentTurnFrameAnomaly === null
      ? await input.options.selfContextBuilder.build({
          turnId: input.turnId,
          sessionId: input.sessionId,
          cognitionInput: input.cognitionInput,
          perception: input.perception,
          autonomyTrigger: input.turnInput.autonomyTrigger,
          audienceEntityId: input.audienceEntityId,
        })
      : null;
  const actionLinkGoalId = actionLinkSelfContext?.executiveFocus.selected_goal?.id ?? null;
  const activeGoalsForPromotion = input.isUserTurn
    ? await input.options.selfContextBuilder.listActiveGoalsForCognition(input.audienceEntityId)
    : [];
  // Cross-audience scope is offered only when the current sender is a creator
  // in an operator context (same authority gate as manual outbound). Other
  // turns get an empty candidate set, so the extractor cannot redirect a
  // commitment away from the current audience.
  const effectiveDistinctSenderCount =
    input.distinctSenderCount ?? (input.currentSenderEntityId === null ? 0 : 1);
  const authoritySenderEntityId =
    effectiveDistinctSenderCount === 1 ? input.currentSenderEntityId : null;
  const authoritySenderDisplayName =
    effectiveDistinctSenderCount === 1 ? input.currentSenderDisplayName : null;
  const authoritySenderBorgRole =
    effectiveDistinctSenderCount === 1 ? input.currentSenderBorgRole : null;
  const crossAudienceAllowed =
    effectiveDistinctSenderCount === 1 &&
    isCreatorInOperatorContext({
      currentSenderBorgRole: authoritySenderBorgRole,
      sessionAudienceRole: input.sessionAudienceRole,
    });
  const crossAudienceCandidateAudiences =
    crossAudienceAllowed && input.options.sessionsRepository !== undefined
      ? dedupeCrossAudienceTargets(
          input.options.sessionsRepository
            .list({
              status: "active",
              excludeSessionId: input.sessionId,
              limit: CROSS_AUDIENCE_TARGET_CAP,
            })
            .flatMap((session) => {
              // Label-only sessions carry a null audience_entity_id column but
              // still scope memory to an entity resolved from the label at turn
              // time. Resolve the same way (existing entities only) so the
              // candidate id matches what the target session deliberates under.
              const audienceEntityId =
                session.audience_entity_id ??
                input.options.entityRepository.findByName(session.audience_label);
              return audienceEntityId === null || audienceEntityId === input.audienceEntityId
                ? []
                : [{ entity_id: audienceEntityId, label: session.audience_label }];
            }),
        )
      : [];

  const [correctivePreferenceTurn, createdActionIds, persistedPromotions, creatorDirectives] =
    await Promise.all([
      input.currentTurnFrameAnomaly === null
        ? input.options.correctivePreferenceTurnService.extractAndApply({
            llmClient: input.llmClient,
            turnId: input.turnId,
            isUserTurn: input.isUserTurn,
            userMessage: input.turnInput.userMessage,
            persistedUserEntryId: input.persistedUserEntryId,
            sourceUserEntryIds: input.sourceUserEntryIds,
            recentHistory: input.recentHistory,
            audienceEntityId: input.audienceEntityId,
            committedByEntityId: input.groupSpeakerEntityId,
            currentSenderEntityId: authoritySenderEntityId,
            currentSenderBorgRole: authoritySenderBorgRole,
            sessionAudienceRole: input.sessionAudienceRole,
            speakerDisplayName: input.groupSpeakerDisplayName,
            participantRoster: input.participantRoster,
            crossAudienceTargeting: {
              allowed: crossAudienceAllowed,
              candidateAudiences: crossAudienceCandidateAudiences,
            },
            sessionId: input.sessionId,
            onHookFailure: (hook, error, details) =>
              input.appendHookFailureEvent(input.streamWriter, hook, error, details),
            trackAppliedSlotNegation: input.trackAppliedSlotNegation,
          })
        : Promise.resolve({
            commitment: null,
            commitmentSupersession: null,
            commitmentRetirement: null,
            workingMemory: input.workingMemory,
          }),
      input.options.turnActionStateService.extract({
        llmClient: input.llmClient,
        turnId: input.turnId,
        isUserTurn: input.isUserTurn,
        userMessage: input.turnInput.userMessage,
        persistedUserEntryId: input.persistedUserEntryId,
        sourceUserEntryIds: input.sourceUserEntryIds,
        senderAttribution: input.senderAttribution,
        recentHistory: input.recentHistory,
        audienceEntityId: input.audienceEntityId,
        sessionId: input.sessionId,
        speakerEntityId: input.groupSpeakerEntityId,
        speakerDisplayName: input.groupSpeakerDisplayName,
        goalId: actionLinkGoalId,
        turnCounter: input.turnInput.globalTurnCounter ?? input.workingMemory.turn_counter,
        frameAnomaly: input.currentTurnFrameAnomaly,
      }),
      input.currentTurnFrameAnomaly === null
        ? input.options.turnGoalPromotionService.extractAndPersist({
            llmClient: input.llmClient,
            turnId: input.turnId,
            sessionId: input.sessionId,
            isUserTurn: input.isUserTurn,
            userMessage: input.turnInput.userMessage,
            recentHistory: input.recentHistory,
            audienceEntityId: input.audienceEntityId,
            ownerEntityId: input.groupSpeakerEntityId,
            speakerDisplayName: input.groupSpeakerDisplayName,
            temporalCue: input.perception.temporalCue,
            activeGoals: activeGoalsForPromotion,
            persistedUserEntryId: input.persistedUserEntryId,
            sourceUserEntryIds: input.sourceUserEntryIds,
            onHookFailure: (hook, error, details) =>
              input.appendHookFailureEvent(input.streamWriter, hook, error, details),
          })
        : Promise.resolve({
            goalIds: [],
            executiveStepIds: [],
          }),
      input.options.creatorDirectiveTurnService.extractAndPersist({
        llmClient: input.llmClient,
        turnId: input.turnId,
        isUserTurn: input.isUserTurn,
        userMessage: input.turnInput.userMessage,
        audienceEntityId: input.audienceEntityId,
        currentSenderEntityId: authoritySenderEntityId,
        currentSenderBorgRole: authoritySenderBorgRole,
        currentSenderDisplayName: authoritySenderDisplayName,
        sourceSessionId: input.sessionId,
        persistedUserEntryId: input.persistedUserEntryId,
        sourceUserEntryIds: input.sourceUserEntryIds,
        recentHistory: input.recentHistory,
        sessionId: input.sessionId,
        sessionAudienceRole: input.sessionAudienceRole,
        participantRoster: input.participantRoster,
      }),
    ]);

  return {
    actionLinkSelfContext,
    correctiveCommitment: correctivePreferenceTurn.commitment,
    correctiveCommitmentSupersession: correctivePreferenceTurn.commitmentSupersession,
    correctiveCommitmentRetirement: correctivePreferenceTurn.commitmentRetirement,
    workingMemory: correctivePreferenceTurn.workingMemory,
    createdActionIds,
    persistedPromotions,
    creatorDirectives,
  };
}
