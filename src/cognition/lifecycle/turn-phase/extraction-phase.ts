import type { CorrectivePreferenceTurnService } from "../../commitments/corrective-preference-service.js";
import type { FrameAnomalyClassification } from "../../frame-anomaly/index.js";
import { isFrameAnomaly } from "../../frame-anomaly/index.js";
import type { RecencyMessage } from "../../recency/index.js";
import type { PerceptionResult } from "../../types.js";
import type { LLMClient } from "../../../llm/index.js";
import type { StreamEntry, StreamWriter } from "../../../stream/index.js";
import type { EntityId, SessionId } from "../../../util/ids.js";
import type { WorkingMemory } from "../../../memory/working/index.js";
import type { TurnPhaseCoordinatorOptions, TurnPhaseInput } from "./types.js";
import type { AppendHookFailureEvent } from "./utils.js";

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
  workingMemory: WorkingMemory;
  createdActionIds: Awaited<
    ReturnType<TurnPhaseCoordinatorOptions["turnActionStateService"]["extract"]>
  >;
  persistedPromotions: Awaited<
    ReturnType<TurnPhaseCoordinatorOptions["turnGoalPromotionService"]["extractAndPersist"]>
  >;
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
  persistedUserEntryId?: StreamEntry["id"];
  frameAnomalyClassification: FrameAnomalyClassification | null;
  streamWriter: StreamWriter;
  trackAppliedSlotNegation: Parameters<
    CorrectivePreferenceTurnService["extractAndApply"]
  >[0]["trackAppliedSlotNegation"];
}): Promise<TurnExtractionPhaseResult> {
  const currentTurnFrameAnomaly = isFrameAnomaly(input.frameAnomalyClassification)
    ? input.frameAnomalyClassification
    : null;
  const actionLinkSelfContext =
    input.isUserTurn && currentTurnFrameAnomaly === null
      ? await input.options.selfContextBuilder.build({
          turnId: input.turnId,
          cognitionInput: input.cognitionInput,
          perception: input.perception,
          autonomyTrigger: input.turnInput.autonomyTrigger,
          audienceEntityId: input.audienceEntityId,
        })
      : null;
  const actionLinkGoalId = actionLinkSelfContext?.executiveFocus.selected_goal?.id ?? null;
  const activeGoalsForPromotion = input.isUserTurn
    ? await input.options.selfContextBuilder.listActiveGoalsVisibleToAudience(
        input.audienceEntityId,
      )
    : [];
  const [correctivePreferenceTurn, createdActionIds, persistedPromotions] = await Promise.all([
    currentTurnFrameAnomaly === null
      ? input.options.correctivePreferenceTurnService.extractAndApply({
          llmClient: input.llmClient,
          turnId: input.turnId,
          userMessage: input.turnInput.userMessage,
          persistedUserEntryId: input.persistedUserEntryId,
          recentHistory: input.recentHistory,
          audienceEntityId: input.audienceEntityId,
          committedByEntityId: input.groupSpeakerEntityId,
          speakerDisplayName: input.groupSpeakerDisplayName,
          sessionId: input.sessionId,
          onHookFailure: (hook, error, details) =>
            input.appendHookFailureEvent(input.streamWriter, hook, error, details),
          trackAppliedSlotNegation: input.trackAppliedSlotNegation,
        })
      : Promise.resolve({
          commitment: null,
          commitmentSupersession: null,
          workingMemory: input.workingMemory,
        }),
    input.options.turnActionStateService.extract({
      llmClient: input.llmClient,
      turnId: input.turnId,
      isUserTurn: input.isUserTurn,
      userMessage: input.turnInput.userMessage,
      persistedUserEntryId: input.persistedUserEntryId,
      recentHistory: input.recentHistory,
      audienceEntityId: input.audienceEntityId,
      sessionId: input.sessionId,
      speakerEntityId: input.groupSpeakerEntityId,
      speakerDisplayName: input.groupSpeakerDisplayName,
      goalId: actionLinkGoalId,
      turnCounter: input.workingMemory.turn_counter,
      frameAnomaly: input.frameAnomalyClassification,
    }),
    currentTurnFrameAnomaly === null
      ? input.options.turnGoalPromotionService.extractAndPersist({
          llmClient: input.llmClient,
          turnId: input.turnId,
          isUserTurn: input.isUserTurn,
          userMessage: input.turnInput.userMessage,
          recentHistory: input.recentHistory,
          audienceEntityId: input.audienceEntityId,
          ownerEntityId: input.groupSpeakerEntityId,
          speakerDisplayName: input.groupSpeakerDisplayName,
          temporalCue: input.perception.temporalCue,
          activeGoals: activeGoalsForPromotion,
          persistedUserEntryId: input.persistedUserEntryId,
          onHookFailure: (hook, error, details) =>
            input.appendHookFailureEvent(input.streamWriter, hook, error, details),
        })
      : Promise.resolve({
          goalIds: [],
          executiveStepIds: [],
        }),
  ]);

  return {
    actionLinkSelfContext,
    correctiveCommitment: correctivePreferenceTurn.commitment,
    correctiveCommitmentSupersession: correctivePreferenceTurn.commitmentSupersession,
    workingMemory: correctivePreferenceTurn.workingMemory,
    createdActionIds,
    persistedPromotions,
  };
}
