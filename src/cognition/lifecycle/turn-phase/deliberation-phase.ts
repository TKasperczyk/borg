import { Deliberator } from "../../deliberation/deliberator.js";
import type { ContradictionRoutingCooldown } from "../../deliberation/contradiction-routing-cooldown.js";
import type { ActualFrameAnomalyClassification } from "../../frame-anomaly/index.js";
import type { ActiveParticipant, ParticipantProfileContext } from "../../participants.js";
import type { ParticipantRoster } from "../../perception/index.js";
import type { RecencyMessage } from "../../recency/index.js";
import type { PerceptionResult } from "../../types.js";
import type { LLMClient } from "../../../llm/index.js";
import type { EntityId, SessionId } from "../../../util/ids.js";
import type { StreamEntry, StreamWriter } from "../../../stream/index.js";
import type { WorkingMemory } from "../../../memory/working/index.js";
import type { TurnPhaseCoordinatorOptions, TurnPhaseInput } from "./types.js";
import { sharedStateRenderOptions } from "./utils.js";
import type { TurnRetrievalPhaseResult } from "./retrieval-phase.js";

export type TurnDeliberationPhaseResult = {
  deliberation: Awaited<ReturnType<Deliberator["run"]>>;
  workingMemory: WorkingMemory;
};

export async function runDeliberationPhase(input: {
  options: TurnPhaseCoordinatorOptions;
  llmClient: LLMClient;
  sessionId: SessionId;
  turnId: string;
  turnInput: TurnPhaseInput;
  streamWriter: StreamWriter;
  audienceEntityId: EntityId | null;
  persistedUserEntryId?: StreamEntry["id"];
  perception: PerceptionResult;
  workingMemory: WorkingMemory;
  activeParticipants: readonly ActiveParticipant[];
  participantProfiles: readonly ParticipantProfileContext[];
  audienceProfile: ReturnType<TurnPhaseCoordinatorOptions["socialRepository"]["getProfile"]>;
  recencyMessages: readonly RecencyMessage[];
  currentTurnFrameAnomaly: ActualFrameAnomalyClassification | null;
  retrievalPhase: TurnRetrievalPhaseResult;
  contradictionRoutingCooldown: ContradictionRoutingCooldown;
  participantRoster: ParticipantRoster | null;
}): Promise<TurnDeliberationPhaseResult> {
  const deliberator = new Deliberator({
    llmClient: input.llmClient,
    toolDispatcher: input.options.toolDispatcher,
    cognitionModel: input.options.config.anthropic.models.cognition,
    cognitionThinking: input.options.config.generation.cognition.thinking,
    clock: input.options.clock,
    tracer: input.options.tracer,
    hostCapabilities: input.options.config.host_capabilities,
    sharedStateRenderOptions: sharedStateRenderOptions(input.options.config),
  });
  const deliberation = await deliberator.run(
    {
      sessionId: input.sessionId,
      turnId: input.turnId,
      audience: input.turnInput.audience,
      audienceEntityId: input.audienceEntityId,
      senderEntityId: input.turnInput.senderEntityId,
      userMessage: input.turnInput.userMessage,
      userEntryId: input.persistedUserEntryId,
      autonomyTrigger: input.turnInput.autonomyTrigger ?? null,
      perception: input.perception,
      retrievalResult: input.retrievalPhase.retrievedEpisodes,
      retrievedSemantic: input.retrievalPhase.retrievedSemantic,
      retrievedEvidence: input.retrievalPhase.retrieval.evidence,
      contradictionPresent: input.retrievalPhase.retrieval.contradiction_present,
      contradictionRouting: input.retrievalPhase.retrieval.contradictionRouting,
      retrievalConfidence: input.retrievalPhase.retrieval.confidence,
      applicableCommitments: input.retrievalPhase.applicableCommitments,
      openQuestionsContext: input.retrievalPhase.retrieval.open_questions,
      pendingCorrectionsContext: input.retrievalPhase.pendingCorrections,
      relationalSlots: input.retrievalPhase.relationalSlots,
      activeParticipants: input.activeParticipants,
      participantRoster: input.participantRoster,
      participantProfiles: input.participantProfiles,
      selectedSkill: input.retrievalPhase.selectedSkill,
      entityRepository: input.options.entityRepository,
      workingMemory: input.workingMemory,
      recentCompletedActions: input.options.postGenerationGuardRunner.listRecentCompletedActions(
        input.audienceEntityId,
      ),
      affectiveTrajectory: input.retrievalPhase.affectiveTrajectory,
      selfSnapshot: input.retrievalPhase.selfSnapshot,
      executiveFocus: input.retrievalPhase.executiveFocusWithStep,
      audienceProfile: input.audienceProfile,
      recencyMessages: input.recencyMessages,
      frameAnomaly: input.currentTurnFrameAnomaly,
      evidenceLedgerPromptSection: input.retrievalPhase.evidenceLedgerContext.promptSection,
      evidenceLedger: input.retrievalPhase.evidenceLedgerContext.ledger,
      routingOverride: input.retrievalPhase.routingOverride,
      contradictionRoutingCooldown: input.contradictionRoutingCooldown,
      contradictionRoutingConfig: input.options.config.deliberation.contradictionRouting,
      options: {
        stakes: input.turnInput.stakes,
      },
      reRetrieve: input.retrievalPhase.retrievalContext.reRetrieve,
    },
    input.streamWriter,
  );

  if (deliberation.emissionRecommendation === "no_output") {
    return {
      deliberation,
      workingMemory: input.options.discourseStateService.setStopState({
        workingMemory: input.workingMemory,
        provenance: "s2_planner_no_output",
        sourceStreamEntryId: deliberation.thoughtStreamEntryIds?.[0],
        reason: "S2 planner recommended no assistant message for this turn.",
        turnId: input.turnId,
      }),
    };
  }

  return {
    deliberation,
    workingMemory: input.workingMemory,
  };
}
