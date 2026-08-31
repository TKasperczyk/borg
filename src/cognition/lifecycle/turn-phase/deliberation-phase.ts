import { Deliberator } from "../../deliberation/deliberator.js";
import type { CreatorIdentityContext, TrustedCreatorContext } from "../../deliberation/types.js";
import type { AutonomousOutboundPromptContext } from "../../../outbound/index.js";
import type { OperatorSessionSnapshot } from "./session-snapshot.js";
import { PROMPT_KEYS, type PromptKey } from "../../prompts/registry.js";
import type { PromptOverrideRepository } from "../../prompts/override-repository.js";
import type { ContradictionRoutingCooldown } from "../../deliberation/contradiction-routing-cooldown.js";
import type { ActualFrameAnomalyClassification } from "../../frame-anomaly/index.js";
import type { ActiveParticipant, ParticipantProfileContext } from "../../participants.js";
import type { ParticipantRoster } from "../../perception/index.js";
import type { RecencyMessage } from "../../recency/index.js";
import { isUserTurnOrigin, type PerceptionResult } from "../../types.js";
import type { LLMClient } from "../../../llm/index.js";
import type { BorgUserContentBlock } from "../../../attachments/index.js";
import type { EntityId, SessionId } from "../../../util/ids.js";
import type { StreamEntry, StreamWriter } from "../../../stream/index.js";
import type { WorkingMemory } from "../../../memory/working/index.js";
import type { SessionParticipationPolicy } from "../../../sessions/index.js";
import type { TurnPhaseCoordinatorOptions, TurnPhaseInput } from "./types.js";
import { sharedStateRenderOptions } from "./utils.js";
import type { TurnRetrievalPhaseResult } from "./retrieval-phase.js";

export type TurnDeliberationPhaseResult = {
  deliberation: Awaited<ReturnType<Deliberator["run"]>>;
  workingMemory: WorkingMemory;
};

function resolvePromptBlockOverrides(
  repo: Pick<PromptOverrideRepository, "get"> | undefined,
): Partial<Record<PromptKey, string>> | undefined {
  if (repo === undefined) {
    return undefined;
  }

  const blocks: Partial<Record<PromptKey, string>> = {};
  for (const key of PROMPT_KEYS) {
    const override = repo.get(key);
    if (override !== null) {
      blocks[key] = override;
    }
  }
  return Object.keys(blocks).length === 0 ? undefined : blocks;
}

function resolveCommitmentEntityLabels(
  commitments: TurnRetrievalPhaseResult["applicableCommitments"],
  entityRepository: TurnPhaseCoordinatorOptions["entityRepository"],
): Readonly<Record<string, string>> {
  const ids = new Set(
    commitments.flatMap((commitment) =>
      [
        commitment.made_to_entity,
        commitment.restricted_audience,
        commitment.about_entity,
        commitment.committed_by_entity_id ?? null,
      ].filter((entityId): entityId is EntityId => entityId !== null),
    ),
  );
  return Object.fromEntries(
    [...ids].flatMap((entityId) => {
      const label = entityRepository.get(entityId)?.canonical_name;
      return label === undefined ? [] : [[entityId, label]];
    }),
  );
}

export async function runDeliberationPhase(input: {
  options: TurnPhaseCoordinatorOptions;
  llmClient: LLMClient;
  sessionId: SessionId;
  turnId: string;
  turnInput: TurnPhaseInput;
  streamWriter: StreamWriter;
  isSelfAudience?: boolean;
  audienceEntityId: EntityId | null;
  participationPolicy: SessionParticipationPolicy;
  creatorIdentity: CreatorIdentityContext | null;
  creatorContext: TrustedCreatorContext | null;
  autonomousOutbound?: AutonomousOutboundPromptContext | null;
  operatorSessionSnapshot: OperatorSessionSnapshot | null;
  persistedUserEntryId?: StreamEntry["id"];
  sourceUserEntryIds?: readonly StreamEntry["id"][];
  currentUserContent?: readonly BorgUserContentBlock[];
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
  const promptBlocks = resolvePromptBlockOverrides(input.options.promptOverrideRepository);
  const deliberator = new Deliberator({
    llmClient: input.llmClient,
    toolDispatcher: input.options.toolDispatcher,
    cognitionModel: input.options.config.anthropic.models.cognition,
    cognitionThinking: input.options.config.generation.cognition.thinking,
    clock: input.options.clock,
    tracer: input.options.tracer,
    hostCapabilities: input.options.config.host_capabilities,
    promptBlocks,
    finalizerDynamicPromptCacheEnabled:
      input.options.config.deliberation.finalizerDynamicPromptCacheEnabled,
    finalizerSurfaceVariant: input.options.config.deliberation.finalizerSurfaceVariant,
    planRequestedVerificationMembershipTokenBudget:
      input.options.config.deliberation.planRequestedVerificationMembershipTokenBudget,
    finalizerContextCapture: input.options.finalizerContextCapture,
    plannerSurfaceVariant: input.options.config.deliberation.plannerSurfaceVariant,
    plannerContextCapture: input.options.plannerContextCapture,
    sharedStateRenderOptions: sharedStateRenderOptions(input.options.config),
    maxImagesPerLlmCall: input.options.config.attachments.maxImagesPerLedger,
  });
  const deliberation = await deliberator.run(
    {
      sessionId: input.sessionId,
      currentTimeContext: input.retrievalPhase.currentTimeContext,
      participationPolicy: input.participationPolicy,
      creatorIdentity: input.creatorIdentity,
      creatorContext: input.creatorContext,
      autonomousOutbound: input.autonomousOutbound ?? null,
      creatorDirectiveBriefing: input.retrievalPhase.creatorDirectiveBriefing,
      operatorSessionSnapshot: input.operatorSessionSnapshot,
      turnId: input.turnId,
      turnOrigin: input.turnInput.origin,
      audience: input.turnInput.audience,
      isSelfAudience: input.isSelfAudience ?? false,
      audienceEntityId: input.audienceEntityId,
      senderEntityId: input.turnInput.senderEntityId,
      userMessage: input.turnInput.userMessage,
      userEntryId: input.persistedUserEntryId,
      currentUserContent: input.currentUserContent,
      autonomyTrigger: input.turnInput.autonomyTrigger ?? null,
      perception: input.perception,
      retrievalResult: input.retrievalPhase.retrievedEpisodes,
      retrievedSemantic: input.retrievalPhase.retrievedSemantic,
      retrievedEvidence: input.retrievalPhase.retrieval.evidence,
      contradictionPresent: input.retrievalPhase.retrieval.contradiction_present,
      contradictionRouting: input.retrievalPhase.retrieval.contradictionRouting,
      retrievalConfidence: input.retrievalPhase.retrieval.confidence,
      applicableCommitments: input.retrievalPhase.applicableCommitments,
      commitmentEntityLabels: resolveCommitmentEntityLabels(
        input.retrievalPhase.applicableCommitments,
        input.options.entityRepository,
      ),
      openQuestionsContext: input.retrievalPhase.retrieval.open_questions,
      pendingCorrectionsContext: input.retrievalPhase.pendingCorrections,
      relationalSlots: input.retrievalPhase.relationalSlots,
      activeParticipants: input.activeParticipants,
      participantRoster: input.participantRoster,
      participantProfiles: input.participantProfiles,
      selectedSkill: input.retrievalPhase.selectedSkill,
      entityRepository: input.options.entityRepository,
      workingMemory: input.workingMemory,
      turnMechanismEvidence: input.retrievalPhase.turnMechanismEvidence,
      recentCompletedActions:
        input.options.postGenerationGuardRunner.listRecentCompletedActionsForCognition(
          input.audienceEntityId,
        ),
      affectiveTrajectory: input.retrievalPhase.affectiveTrajectory,
      selfSnapshot: input.retrievalPhase.selfSnapshot,
      executiveFocus: input.retrievalPhase.executiveFocusWithStep,
      audienceProfile: input.audienceProfile,
      recencyMessages: input.recencyMessages,
      frameAnomaly: input.currentTurnFrameAnomaly,
      evidenceLedgerPromptSection: input.retrievalPhase.evidenceLedgerContext.promptSection,
      sessionReentryContinuityPromptSection:
        input.retrievalPhase.evidenceLedgerContext.sessionReentryContinuityPromptSection,
      evidenceLedger: input.retrievalPhase.evidenceLedgerContext.ledger,
      sharedStateAppliedOperationCount:
        input.retrievalPhase.evidenceLedgerContext.sharedStateAppliedOperationCount,
      openQuestionsRenderedToFinalizerCount:
        input.retrievalPhase.evidenceLedgerContext.openQuestionsRenderedToFinalizerCount,
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

  if (
    deliberation.emissionRecommendation === "no_output" &&
    isUserTurnOrigin(input.turnInput.origin)
  ) {
    return {
      deliberation,
      workingMemory: input.options.discourseStateService.setStopState({
        workingMemory: input.workingMemory,
        provenance: "s2_planner_no_output",
        sourceStreamEntryId: deliberation.thoughtStreamEntryIds?.[0],
        reason: "S2 planner recommended no assistant message for this turn.",
        turnId: input.turnId,
        sessionId: input.sessionId,
      }),
    };
  }

  return {
    deliberation,
    workingMemory: input.workingMemory,
  };
}
