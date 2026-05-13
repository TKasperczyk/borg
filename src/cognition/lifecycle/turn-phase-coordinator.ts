import { performAction } from "../action/index.js";
import type { ToolLoopCallRecord } from "../action/index.js";
import type { TurnActionCoordinator } from "../action/turn-action-coordinator.js";
import type { TurnActionStateService } from "../actions/turn-action-state-service.js";
import { SuppressionSet } from "../attention/index.js";
import type { AttributionLifecycleService } from "../attribution/lifecycle-service.js";
import { formatAutonomyTriggerContext, type AutonomyTriggerContext } from "../autonomy-trigger.js";
import {
  appendCommitmentIfMissing,
  type CorrectivePreferenceTurnService,
} from "../commitments/corrective-preference-service.js";
import { Deliberator, type TurnStakes } from "../deliberation/deliberator.js";
import {
  EvidenceLedgerBuilder,
  compactEvidenceLedger,
  estimateEvidenceLedgerPromptTokens,
  renderEvidenceLedger,
  summarizeDecisionStateArtifactRender,
  summarizeEvidenceLedgerTrace,
  type EvidenceLedger,
  type EvidenceLedgerBuildInput,
  type EvidenceLedgerCompactionTraceSummary,
} from "../evidence-ledger/index.js";
import { compileDecisionArtifact } from "../decision-artifact/index.js";
import type { TurnDiscourseStateService } from "../generation/turn-discourse-state.js";
import {
  replyTargetEntityId,
  type PendingTurnEmission,
  type TurnEmission,
} from "../generation/types.js";
import { GenerationGate } from "../generation/generation-gate.js";
import { StopCommitmentExtractor } from "../generation/self-stop-commitment.js";
import {
  FrameAnomalyClassifier,
  classifyFrameAnomalyDegradedFallback,
  isFrameAnomaly,
  type ActualFrameAnomalyClassification,
  type FrameAnomalyClassification,
} from "../frame-anomaly/index.js";
import type { TurnGoalPromotionService } from "../goals/turn-goal-promotion-service.js";
import type { PerceptionGateway } from "../perception/gateway.js";
import {
  loadRecentParticipantStreamEntries,
  resolveActiveParticipants,
  resolveParticipantProfiles,
  type ActiveParticipant,
  type ParticipantProfileContext,
} from "../participants.js";
import type { TurnOpeningPersistence } from "../persistence/turn-opening.js";
import type { RecencyMessage } from "../recency/index.js";
import type { TurnReflectionCoordinator } from "../reflection/turn-reflection-coordinator.js";
import type { TurnRetrievalCoordinator } from "../retrieval/turn-coordinator.js";
import type { TurnSelfContextBuilder } from "../self/turn-self-context.js";
import { toTraceJsonValue, type TurnTracer } from "../tracing/tracer.js";
import type { CognitiveMode, IntentRecord } from "../types.js";
import type { Config } from "../../config/index.js";
import type { EmbeddingClient } from "../../embeddings/index.js";
import type { LLMClient } from "../../llm/index.js";
import type { ActionRepository } from "../../memory/actions/index.js";
import type { CommitmentRepository, EntityRepository } from "../../memory/commitments/index.js";
import type {
  DecisionArtifact,
  DecisionArtifactRepository,
} from "../../memory/decision-artifacts/index.js";
import type { RelationalSlotRepository } from "../../memory/relational-slots/index.js";
import {
  appendInternalFailureEvent,
  type GoalsRepository,
  type OpenQuestionsRepository,
} from "../../memory/self/index.js";
import type { SocialRepository } from "../../memory/social/index.js";
import type { WorkingMemory, WorkingMemoryStore } from "../../memory/working/index.js";
import {
  QUARANTINED_USER_ENTRY_EVENT,
  type StreamEntry,
  type StreamReader,
  type StreamWriter,
} from "../../stream/index.js";
import type { ToolDispatcher } from "../../tools/index.js";
import type { Clock } from "../../util/clock.js";
import { CognitionError } from "../../util/errors.js";
import {
  streamEntryIdHelpers,
  type EntityId,
  type SessionId,
  type StreamEntryId,
} from "../../util/ids.js";
import type { StreamIngestionCoordinator } from "../ingestion/index.js";
import type { TurnPostGenerationGuardRunner } from "../generation/turn-post-generation-guard.js";
import type { TurnLifecycleTracker } from "./turn-lifecycle-tracker.js";
import {
  ClosureLoopClassifier,
  assessDegradedClosureLoopFallback,
  assessClosureLoopClassification,
  buildClosureLoopMessageWindow,
  type ClosureLoopAssessment,
} from "../generation/closure-loop.js";

const ACTIVE_TURN_STATUS = "active";
const DELIBERATION_RELATIONAL_SLOT_LIMIT = 24;

type EvidenceLedgerFinalizerContext = {
  ledger: EvidenceLedger | null;
  promptSection: string | null;
};

type EvidenceLedgerFinalizerBuildInput = EvidenceLedgerBuildInput & {
  isUserTurn: boolean;
};

const DECISION_ARTIFACT_LEDGER_STREAM_METADATA_KEYS = [
  "stream_ids",
  "source_stream_ids",
  "evidence_stream_entry_ids",
] as const;

function listConstrainedRelationalSlotsForParticipants(
  repository: RelationalSlotRepository,
  participants: readonly ActiveParticipant[],
) {
  if (participants.length === 0) {
    return repository.listConstrained({
      limit: DELIBERATION_RELATIONAL_SLOT_LIMIT,
    });
  }

  return participants
    .flatMap((participant) =>
      repository.listConstrained({
        subjectEntityId: participant.entityId,
        limit: DELIBERATION_RELATIONAL_SLOT_LIMIT,
      }),
    )
    .slice(0, DELIBERATION_RELATIONAL_SLOT_LIMIT);
}

function audienceProfileForParticipants(
  participantProfiles: readonly ParticipantProfileContext[],
  audienceEntityId: EntityId | null,
) {
  if (participantProfiles.length === 1) {
    return participantProfiles[0]?.profile ?? null;
  }

  if (audienceEntityId === null) {
    return null;
  }

  return (
    participantProfiles.find((participant) => participant.entityId === audienceEntityId)?.profile ??
    null
  );
}

function addDecisionArtifactAllowedStreamId(ids: Set<StreamEntryId>, value: unknown): void {
  if (typeof value === "string" && streamEntryIdHelpers.is(value)) {
    ids.add(value);
  }
}

function addDecisionArtifactAllowedStreamIds(ids: Set<StreamEntryId>, value: unknown): void {
  if (typeof value === "string") {
    addDecisionArtifactAllowedStreamId(ids, value);
    return;
  }

  if (!Array.isArray(value)) {
    return;
  }

  for (const item of value) {
    addDecisionArtifactAllowedStreamId(ids, item);
  }
}

function addDecisionArtifactEntryIdStreamHandle(ids: Set<StreamEntryId>, entryId: string): void {
  const currentSessionPrefix = "current_session_stream:";
  const currentUserPrefix = "current_user_message:";
  const source = entryId.startsWith(currentSessionPrefix)
    ? entryId.slice(currentSessionPrefix.length)
    : entryId.startsWith(currentUserPrefix)
      ? entryId.slice(currentUserPrefix.length)
      : null;

  addDecisionArtifactAllowedStreamId(ids, source);
}

function collectDecisionArtifactAllowedStreamEntryIds(
  ledger: EvidenceLedger,
  currentUserStreamEntryId: StreamEntryId,
): StreamEntryId[] {
  const ids = new Set<StreamEntryId>([currentUserStreamEntryId]);

  for (const section of ledger.sections) {
    for (const entry of section.entries) {
      addDecisionArtifactEntryIdStreamHandle(ids, entry.id);
      addDecisionArtifactAllowedStreamIds(ids, entry.citations);

      for (const key of DECISION_ARTIFACT_LEDGER_STREAM_METADATA_KEYS) {
        addDecisionArtifactAllowedStreamIds(ids, entry.state_metadata?.[key]);
      }
    }
  }

  return [...ids];
}

export type TurnPhaseInput = {
  userMessage: string;
  audience?: string;
  senderEntityId?: EntityId;
  stakes?: TurnStakes;
  sessionId?: SessionId;
  origin?: "user" | "autonomous";
  autonomyTrigger?: AutonomyTriggerContext | null;
};

export type TurnPhaseResult = {
  mode: CognitiveMode;
  path: "system_1" | "system_2" | "suppressed";
  response: string;
  emitted: boolean;
  emission: TurnEmission;
  thoughts: string[];
  usage: {
    input_tokens: number;
    output_tokens: number;
    stop_reason: string | null;
  };
  retrievedEpisodeIds: string[];
  referencedEpisodeIds: string[];
  intents: IntentRecord[];
  toolCalls: ToolLoopCallRecord[];
  agentMessageId?: string;
};

export type TurnPhaseCoordinatorOptions = {
  config: Config;
  embeddingClient: EmbeddingClient;
  workingMemoryStore: WorkingMemoryStore;
  entityRepository: EntityRepository;
  socialRepository: SocialRepository;
  relationalSlotRepository: RelationalSlotRepository;
  actionRepository: Pick<ActionRepository, "get" | "list"> &
    Partial<Pick<ActionRepository, "findSimilarDescriptionPairs">>;
  commitmentRepository: CommitmentRepository;
  decisionArtifactRepository: Pick<DecisionArtifactRepository, "get" | "upsert">;
  goalsRepository: GoalsRepository;
  openQuestionsRepository: Pick<OpenQuestionsRepository, "findByHandles">;
  toolDispatcher: ToolDispatcher;
  createStreamReader: (sessionId: SessionId) => StreamReader;
  streamIngestionCoordinator?: StreamIngestionCoordinator;
  llmFactory: () => LLMClient;
  perceptionGateway: PerceptionGateway;
  turnOpeningPersistence: TurnOpeningPersistence;
  attributionLifecycleService: AttributionLifecycleService;
  correctivePreferenceTurnService: CorrectivePreferenceTurnService;
  turnActionStateService: TurnActionStateService;
  turnGoalPromotionService: TurnGoalPromotionService;
  selfContextBuilder: TurnSelfContextBuilder;
  turnRetrievalCoordinator: TurnRetrievalCoordinator;
  discourseStateService: TurnDiscourseStateService;
  postGenerationGuardRunner: Pick<TurnPostGenerationGuardRunner, "listRecentCompletedActions">;
  turnActionCoordinator: TurnActionCoordinator;
  turnReflectionCoordinator: TurnReflectionCoordinator;
  clock: Clock;
  tracer: TurnTracer;
};

export type RunTurnPhasesInput = {
  input: TurnPhaseInput;
  sessionId: SessionId;
  turnId: string;
  streamWriter: StreamWriter;
  lifecycleTracker: TurnLifecycleTracker;
};

function evidenceLedgerCompactionChanged(summary: EvidenceLedgerCompactionTraceSummary): boolean {
  return (
    summary.dedupedEntryCount > 0 ||
    summary.droppedSections.length > 0 ||
    summary.postCapTokens < summary.preCapTokens ||
    Object.values(summary.omittedEntryCountsBySection).some((count) => count > 0)
  );
}

export class TurnPhaseCoordinator {
  constructor(private readonly options: TurnPhaseCoordinatorOptions) {}

  async run(input: RunTurnPhasesInput): Promise<TurnPhaseResult> {
    const turnInput = input.input;
    const sessionId = input.sessionId;
    const turnId = input.turnId;
    const streamWriter = input.streamWriter;
    const lifecycleTracker = input.lifecycleTracker;
    const isSelfAudience = turnInput.audience === "self";
    const isUserTurn = turnInput.origin !== "autonomous";
    const preflightAudienceEntityId =
      turnInput.audience === undefined || isSelfAudience
        ? null
        : this.options.entityRepository.findByName(turnInput.audience);
    const preflightAudienceEntity =
      preflightAudienceEntityId === null
        ? null
        : this.options.entityRepository.get(preflightAudienceEntityId);

    if (
      preflightAudienceEntity?.kind === "group" &&
      isUserTurn &&
      (turnInput.senderEntityId === null || turnInput.senderEntityId === undefined)
    ) {
      throw new CognitionError("Group-audience user turns require senderEntityId", {
        code: "GROUP_SENDER_REQUIRED",
      });
    }

    await this.catchUpStreamIngestion(sessionId, streamWriter);
    let workingMemory = this.options.workingMemoryStore.load(sessionId);
    lifecycleTracker.captureInitialWorkingMemory(workingMemory);
    const turnPerception = this.options.perceptionGateway.beginTurn({
      turnId,
      onHookFailure: (hook, error, details) =>
        this.appendHookFailureEvent(streamWriter, hook, error, details),
    });
    const llmClient = this.options.llmFactory();
    const cognitionInput =
      turnInput.autonomyTrigger === null || turnInput.autonomyTrigger === undefined
        ? turnInput.userMessage
        : formatAutonomyTriggerContext(turnInput.autonomyTrigger);
    const audienceEntityId =
      turnInput.audience === undefined || isSelfAudience
        ? null
        : this.options.entityRepository.resolve(turnInput.audience, {
            provenance: "transport_audience_label",
          });
    const audienceEntity =
      audienceEntityId === null ? null : this.options.entityRepository.get(audienceEntityId);
    let audienceProfile =
      audienceEntityId === null ? null : this.options.socialRepository.getProfile(audienceEntityId);

    // In a group channel the social exchange belongs to the current speaker,
    // not to the abstract channel entity. Updating the group too is deferred.
    const socialInteractionEntityId =
      audienceEntity?.kind === "group" ? (turnInput.senderEntityId ?? null) : audienceEntityId;
    const groupSpeakerEntityId =
      audienceEntity?.kind === "group" ? (turnInput.senderEntityId ?? null) : null;
    const groupSpeakerDisplayName =
      groupSpeakerEntityId === null
        ? null
        : (this.options.entityRepository.get(groupSpeakerEntityId)?.canonical_name ?? null);
    const perceptionResult = await turnPerception.perceive({
      sessionId,
      isSelfAudience,
      origin: turnInput.origin,
      cognitionInput,
      workingMemory,
    });
    const perception = perceptionResult.perception;
    for (const userIdentityName of perception.userIdentityNames ?? []) {
      this.options.entityRepository.resolve(userIdentityName, {
        kind: "person",
        provenance: "user_declared",
      });
    }
    const recencyWindow = perceptionResult.recencyWindow;
    const workingMood = perceptionResult.workingMood;
    workingMemory = perceptionResult.workingMemory;
    const suppressionSet = SuppressionSet.fromEntries(
      workingMemory.suppressed,
      workingMemory.turn_counter,
    );
    const attributionResult = await this.options.attributionLifecycleService.settle({
      isUserTurn,
      audienceEntityId,
      socialEntityId: socialInteractionEntityId,
      perception,
      pendingSocialAttribution: workingMemory.pending_social_attribution,
      pendingTraitAttribution: workingMemory.pending_trait_attribution,
      audienceProfile,
      streamWriter,
      onHookFailure: (hook, error) => this.appendHookFailureEvent(streamWriter, hook, error),
    });
    const pendingSocialAttribution = attributionResult.pendingSocialAttribution;
    const pendingTraitAttribution = attributionResult.pendingTraitAttribution;
    audienceProfile = attributionResult.audienceProfile;

    const openingPersistence = await this.options.turnOpeningPersistence.persist({
      streamWriter,
      turnId,
      userMessage: turnInput.userMessage,
      persistUserMessage: isUserTurn,
      audience: turnInput.audience,
      senderEntityId: turnInput.senderEntityId,
      workingMemory,
      pendingSocialAttribution,
      pendingTraitAttribution,
      suppressionSet,
      perception,
      now: () => this.options.clock.now(),
    });
    const persistedUserEntry = openingPersistence.persistedUserEntry;
    const persistedUserEntryId = persistedUserEntry?.id;
    const persistedPerceptionEntry = openingPersistence.persistedPerceptionEntry;
    workingMemory = openingPersistence.workingMemory;
    const activeParticipantLimit = this.options.config.generation.activeParticipantLimit;
    const activeParticipants = resolveActiveParticipants({
      audienceEntityId,
      senderEntityId: turnInput.senderEntityId ?? null,
      streamEntries: loadRecentParticipantStreamEntries(
        this.options.createStreamReader(sessionId),
        activeParticipantLimit,
      ),
      entityRepository: this.options.entityRepository,
      limit: activeParticipantLimit,
    });
    const participantProfiles = resolveParticipantProfiles(
      activeParticipants,
      this.options.socialRepository,
    );
    if (activeParticipants.length > 0) {
      audienceProfile = audienceProfileForParticipants(participantProfiles, audienceEntityId);
    }

    const frameAnomalyClassification = await this.classifyFrameAnomaly({
      llmClient,
      turnId,
      isUserTurn,
      userMessage: turnInput.userMessage,
      recentHistory: recencyWindow.messages,
      persistedUserEntryId,
      streamWriter,
    });
    const currentTurnFrameAnomaly = isFrameAnomaly(frameAnomalyClassification)
      ? frameAnomalyClassification
      : null;

    const actionLinkSelfContext =
      isUserTurn && currentTurnFrameAnomaly === null
        ? await this.options.selfContextBuilder.build({
            turnId,
            cognitionInput,
            perception,
            autonomyTrigger: turnInput.autonomyTrigger,
            audienceEntityId,
          })
        : null;
    const actionLinkGoalId = actionLinkSelfContext?.executiveFocus.selected_goal?.id ?? null;
    const activeGoalsForPromotion = isUserTurn
      ? await this.options.selfContextBuilder.listActiveGoalsVisibleToAudience(audienceEntityId)
      : [];
    const [correctivePreferenceTurn, createdActionIds, persistedPromotions] = await Promise.all([
      currentTurnFrameAnomaly === null
        ? this.options.correctivePreferenceTurnService.extractAndApply({
            llmClient,
            turnId,
            userMessage: turnInput.userMessage,
            persistedUserEntryId,
            recentHistory: recencyWindow.messages,
            audienceEntityId,
            committedByEntityId: groupSpeakerEntityId,
            speakerDisplayName: groupSpeakerDisplayName,
            sessionId,
            onHookFailure: (hook, error, details) =>
              this.appendHookFailureEvent(streamWriter, hook, error, details),
            trackAppliedSlotNegation: (slot) => lifecycleTracker.trackAppliedSlotNegation(slot),
          })
        : Promise.resolve({
            commitment: null,
            workingMemory,
          }),
      this.options.turnActionStateService.extract({
        llmClient,
        turnId,
        isUserTurn,
        userMessage: turnInput.userMessage,
        persistedUserEntryId,
        recentHistory: recencyWindow.messages,
        audienceEntityId,
        speakerEntityId: groupSpeakerEntityId,
        speakerDisplayName: groupSpeakerDisplayName,
        goalId: actionLinkGoalId,
        frameAnomaly: frameAnomalyClassification,
      }),
      currentTurnFrameAnomaly === null
        ? this.options.turnGoalPromotionService.extractAndPersist({
            llmClient,
            turnId,
            isUserTurn,
            userMessage: turnInput.userMessage,
            recentHistory: recencyWindow.messages,
            audienceEntityId,
            ownerEntityId: groupSpeakerEntityId,
            speakerDisplayName: groupSpeakerDisplayName,
            temporalCue: perception.temporalCue,
            activeGoals: activeGoalsForPromotion,
            persistedUserEntryId,
            onHookFailure: (hook, error, details) =>
              this.appendHookFailureEvent(streamWriter, hook, error, details),
          })
        : Promise.resolve({
            goalIds: [],
            executiveStepIds: [],
          }),
    ]);
    const correctiveCommitment = correctivePreferenceTurn.commitment;
    workingMemory = correctivePreferenceTurn.workingMemory;
    lifecycleTracker.trackCreatedActionIds(createdActionIds);
    lifecycleTracker.trackCreatedGoalIds(persistedPromotions.goalIds);
    lifecycleTracker.trackCreatedExecutiveStepIds(persistedPromotions.executiveStepIds);

    const closureLoopAssessment = await this.classifyClosureLoop({
      llmClient,
      turnId,
      isUserTurn,
      userMessage: turnInput.userMessage,
      recentHistory: recencyWindow.messages,
      persistedUserEntryId,
      workingMemory,
      streamWriter,
    });

    if (closureLoopAssessment?.currentUserSubstantive === true) {
      workingMemory = this.options.discourseStateService.clearClosureLoop({
        workingMemory,
        reason: closureLoopAssessment.reason,
        turnId,
      });
    } else if (
      closureLoopAssessment?.currentUserClosureShaped === true &&
      workingMemory.discourse_state?.closure_loop?.status === "named"
    ) {
      return this.suppressFromClosureLoop({
        turnId,
        turnInput,
        streamWriter,
        workingMemory,
        persistedUserEntryId,
        correctiveCommitment,
        perceptionMode: perception.mode,
        reason: closureLoopAssessment.reason,
      });
    } else if (closureLoopAssessment?.closureLoopDetected === true) {
      workingMemory = this.options.discourseStateService.setClosureLoopDetected({
        workingMemory,
        sourceStreamEntryIds: closureLoopAssessment.sourceStreamEntryIds,
        reason: closureLoopAssessment.reason,
        turnId,
      });
    }

    const generationGate = new GenerationGate({
      llmClient,
      embeddingClient: this.options.embeddingClient,
      model: this.options.config.anthropic.models.background,
      hardCapTurns: this.options.config.generation.discourseStateHardCapTurns,
      onDegraded: (reason, error) =>
        this.appendHookFailureEvent(streamWriter, "generation_gate", error ?? reason, {
          reason,
        }),
    });
    const gateResult = await generationGate.evaluate({
      userMessage: turnInput.userMessage,
      workingMemory,
      recencyMessages: recencyWindow.messages,
    });

    if (gateResult.signals.hardCapDue) {
      await this.options.discourseStateService.appendHardCapEvent({
        streamWriter,
        turnId,
        activeTurns: gateResult.signals.hardCapActiveTurns,
        hardCapTurns: this.options.config.generation.discourseStateHardCapTurns,
        stateReason:
          workingMemory.discourse_state?.stop_until_substantive_content?.reason ?? "unknown",
      });
    }

    if (gateResult.clearDiscourseStop) {
      workingMemory = this.options.discourseStateService.clearStopState({
        workingMemory,
        reason: gateResult.explanation,
        turnId,
      });
    }

    if (gateResult.action === "suppress") {
      return this.suppressFromGenerationGate({
        turnId,
        turnInput,
        streamWriter,
        workingMemory,
        persistedUserEntryId,
        gateResult,
        correctiveCommitment,
        perceptionMode: perception.mode,
      });
    }

    const selfContext =
      actionLinkSelfContext !== null &&
      persistedPromotions.goalIds.length === 0 &&
      persistedPromotions.executiveStepIds.length === 0
        ? actionLinkSelfContext
        : await this.options.selfContextBuilder.build({
            turnId,
            cognitionInput,
            perception,
            autonomyTrigger: turnInput.autonomyTrigger,
            audienceEntityId,
          });
    const selfSnapshot = selfContext.selfSnapshot;
    const activeScoringValues = selfContext.activeScoringValues;
    const retrievalScoringFeatures = selfContext.retrievalScoringFeatures;
    const executiveFocusWithStep = selfContext.executiveFocus;

    const retrievalContext = await this.options.turnRetrievalCoordinator.coordinate({
      sessionId,
      turnId,
      userMessage: turnInput.userMessage,
      recentMessages: recencyWindow.messages.map((message) => ({
        role: message.role,
        content: message.content,
      })),
      cognitionInput,
      inputAudience: turnInput.audience,
      isSelfAudience,
      audienceEntityId,
      audienceEntity,
      audienceProfile,
      perception,
      workingMemory,
      selfSnapshot,
      executiveFocus: executiveFocusWithStep,
      activeValues: activeScoringValues,
      scoringFeatures: retrievalScoringFeatures,
      suppressionSet,
      findEntityByName: (name) => this.options.entityRepository.findByName(name),
      llmClient,
      proceduralContextModel: this.options.config.anthropic.models.background,
    });
    const applicableCommitments = appendCommitmentIfMissing(
      retrievalContext.applicableCommitments,
      correctiveCommitment,
    );
    const pendingCorrections = retrievalContext.pendingCorrections;
    const affectiveTrajectory = retrievalContext.affectiveTrajectory;
    const retrieval = retrievalContext.retrieval;
    const retrievedEpisodes = retrievalContext.retrievedEpisodes;
    const retrievedSemantic = retrievalContext.retrievedSemantic;
    const proceduralContext = retrievalContext.proceduralContext;
    const selectedSkill = retrievalContext.selectedSkill;
    const relationalSlots = listConstrainedRelationalSlotsForParticipants(
      this.options.relationalSlotRepository,
      activeParticipants,
    );
    const evidenceLedgerContext = await this.buildEvidenceLedgerFinalizerContext({
      sessionId,
      turnId,
      audienceEntityId,
      currentUserMessage: turnInput.userMessage,
      currentUserEntry: persistedUserEntry ?? undefined,
      workingMemory,
      applicableCommitments,
      retrievedEvidence: retrieval.evidence,
      retrievedEpisodes,
      retrievedSemantic,
      openQuestions: retrieval.open_questions,
      pendingCorrections,
      frameAnomaly: currentTurnFrameAnomaly,
      activeParticipants,
      isUserTurn,
    });
    const deliberator = new Deliberator({
      llmClient,
      toolDispatcher: this.options.toolDispatcher,
      cognitionModel: this.options.config.anthropic.models.cognition,
      cognitionThinking: this.options.config.generation.cognition.thinking,
      clock: this.options.clock,
      tracer: this.options.tracer,
      hostCapabilities: this.options.config.host_capabilities,
    });
    const deliberation = await deliberator.run(
      {
        sessionId,
        turnId,
        audience: turnInput.audience,
        audienceEntityId,
        senderEntityId: turnInput.senderEntityId,
        userMessage: turnInput.userMessage,
        userEntryId: persistedUserEntryId,
        autonomyTrigger: turnInput.autonomyTrigger ?? null,
        perception,
        retrievalResult: retrievedEpisodes,
        retrievedSemantic,
        retrievedEvidence: retrieval.evidence,
        contradictionPresent: retrieval.contradiction_present,
        retrievalConfidence: retrieval.confidence,
        applicableCommitments,
        openQuestionsContext: retrieval.open_questions,
        pendingCorrectionsContext: pendingCorrections,
        relationalSlots,
        activeParticipants,
        participantProfiles,
        selectedSkill,
        entityRepository: this.options.entityRepository,
        workingMemory,
        recentCompletedActions:
          this.options.postGenerationGuardRunner.listRecentCompletedActions(audienceEntityId),
        affectiveTrajectory,
        selfSnapshot,
        executiveFocus: executiveFocusWithStep,
        audienceProfile,
        recencyMessages: recencyWindow.messages,
        frameAnomaly: currentTurnFrameAnomaly,
        evidenceLedgerPromptSection: evidenceLedgerContext.promptSection,
        evidenceLedger: evidenceLedgerContext.ledger,
        options: {
          stakes: turnInput.stakes,
        },
        reRetrieve: retrievalContext.reRetrieve,
      },
      streamWriter,
    );

    if (deliberation.emissionRecommendation === "no_output") {
      workingMemory = this.options.discourseStateService.setStopState({
        workingMemory,
        provenance: "s2_planner_no_output",
        sourceStreamEntryId: deliberation.thoughtStreamEntryIds?.[0],
        reason: "S2 planner recommended no assistant message for this turn.",
        turnId,
      });
    }

    workingMemory = {
      ...workingMemory,
      updated_at: this.options.clock.now(),
    };
    const actionCoordinatorResult = await this.options.turnActionCoordinator.run({
      llmClient,
      turnId,
      sessionId,
      deliberation,
      workingMemory,
      userMessage: turnInput.userMessage,
      cognitionInput,
      origin: turnInput.origin,
      autonomyTrigger: turnInput.autonomyTrigger,
      applicableCommitments,
      perceptionEntities: perception.entities,
      persistedUserEntry: persistedUserEntry ?? undefined,
      retrievedEpisodes,
      currentUserClosureKind: closureLoopAssessment?.currentUserAct ?? null,
      audienceEntityId,
    });
    const actionResult = actionCoordinatorResult.actionResult;
    const actionEmission: PendingTurnEmission = actionCoordinatorResult.actionEmission;
    lifecycleTracker.trackPendingActionMerges(actionResult.pending_action_merge_count ?? 0);
    const persistedAgentEntry =
      actionEmission.kind === "message"
        ? await streamWriter.append({
            kind: "agent_msg",
            turn_id: turnId,
            turn_status: ACTIVE_TURN_STATUS,
            content: actionResult.response,
            tool_calls: actionResult.tool_calls,
            reply_target_entity_id: replyTargetEntityId(actionEmission.reply_target),
            ...(actionEmission.persistence_class === undefined
              ? {}
              : { persistence_class: actionEmission.persistence_class }),
            ...(turnInput.audience === undefined ? {} : { audience: turnInput.audience }),
          })
        : actionEmission.kind === "observed"
          ? await this.options.discourseStateService.appendObservationMarker({
              streamWriter,
              reason: actionEmission.reason,
              userEntryId: persistedUserEntryId,
              turnId,
              audience: turnInput.audience,
            })
          : await this.options.discourseStateService.appendSuppressionMarker({
              streamWriter,
              reason: actionEmission.reason,
              userEntryId: persistedUserEntryId,
              turnId,
              audience: turnInput.audience,
            });

    if (actionEmission.kind === "suppressed") {
      return this.suppressFromAction({
        turnId,
        turnInput,
        streamWriter,
        actionResult,
        actionEmission,
        persistedAgentEntry,
        correctiveCommitment,
        perceptionMode: perception.mode,
        deliberation,
      });
    }

    const turnEmission: TurnEmission =
      actionEmission.kind === "observed"
        ? {
            kind: "observed",
            reason: actionEmission.reason,
            markerEntryId: persistedAgentEntry.id,
          }
        : {
            kind: "message",
            content: actionResult.response,
            agentMessageId: persistedAgentEntry.id,
            ...(actionEmission.reply_target === undefined
              ? {}
              : { reply_target: actionEmission.reply_target }),
            ...(actionEmission.persistence_class === undefined
              ? {}
              : { persistence_class: actionEmission.persistence_class }),
          };
    let postActionWorkingMemory = actionResult.workingMemory;
    if (
      actionEmission.kind === "message" &&
      actionEmission.closure_pressure_history_reason !== undefined
    ) {
      postActionWorkingMemory = this.options.discourseStateService.appendClosurePressureHistory({
        workingMemory: postActionWorkingMemory,
        turnId,
        reason: actionEmission.closure_pressure_history_reason,
      });
    }
    if (actionEmission.kind === "message") {
      const stopCommitmentExtractor = new StopCommitmentExtractor({
        llmClient,
        model: this.options.config.anthropic.models.background,
        onDegraded: (reason, error) =>
          this.appendHookFailureEvent(
            streamWriter,
            "self_stop_commitment_extraction",
            error ?? reason,
            {
              reason,
            },
          ),
      });
      const stopCommitment = await stopCommitmentExtractor.extract({
        userMessage: turnInput.userMessage,
        agentResponse: actionResult.response,
      });

      if (stopCommitment !== null) {
        postActionWorkingMemory = this.options.discourseStateService.setStopState({
          workingMemory: postActionWorkingMemory,
          provenance: "self_commitment_extractor",
          sourceStreamEntryId: persistedAgentEntry.id,
          reason: stopCommitment.reason,
          turnId,
        });
      }

      if (postActionWorkingMemory.discourse_state?.closure_loop?.status === "detected") {
        postActionWorkingMemory = this.options.discourseStateService.markClosureLoopNamed({
          workingMemory: postActionWorkingMemory,
          sourceStreamEntryId: persistedAgentEntry.id,
          reason: "Closure loop detected; assistant used the single allowed naming/output turn.",
          turnId,
        });
        postActionWorkingMemory = this.options.discourseStateService.setStopState({
          workingMemory: postActionWorkingMemory,
          provenance: "finalizer_no_output",
          sourceStreamEntryId: persistedAgentEntry.id,
          reason:
            "Closure loop was already named once; suppress further closure-only turns until substantive content.",
          turnId,
        });
      }
    }

    await this.options.turnReflectionCoordinator.run({
      llmClient,
      sessionId,
      turnId,
      origin: turnInput.origin,
      userMessage: turnInput.userMessage,
      perception,
      workingMood,
      postActionWorkingMemory,
      selfSnapshot,
      deliberation,
      actionResult,
      retrievedEpisodes: deliberation.retrievedEpisodes,
      retrievalConfidence: retrieval.confidence,
      executiveFocus: executiveFocusWithStep,
      selectedSkill,
      proceduralContext,
      audienceEntityId,
      audienceIsGroup: audienceEntity?.kind === "group",
      senderEntityId: turnInput.senderEntityId ?? null,
      socialInteractionEntityId,
      pendingSocialAttribution,
      suppressionSet,
      persistedUserEntryId,
      persistedPerceptionEntry,
      persistedAgentEntry,
      isUserTurn,
      frameAnomaly: frameAnomalyClassification,
      streamWriter,
      onHookFailure: (hook, error) => this.appendHookFailureEvent(streamWriter, hook, error),
      trackReflectionEffects: (effects) => lifecycleTracker.trackReflectionEffects(effects),
    });
    await this.persistCorrectiveCommitment(streamWriter, correctiveCommitment);
    this.startLiveIngestion(sessionId);

    return {
      mode: perception.mode,
      path: deliberation.path,
      response: actionResult.response,
      emitted: actionEmission.kind === "message",
      emission: turnEmission,
      thoughts: deliberation.thoughts,
      usage: deliberation.usage,
      retrievedEpisodeIds: deliberation.retrievedEpisodes.map((result) => result.episode.id),
      referencedEpisodeIds: [...(deliberation.referencedEpisodeIds ?? [])],
      intents: actionResult.intents,
      toolCalls: [...actionResult.tool_calls],
      ...(actionEmission.kind === "message" ? { agentMessageId: persistedAgentEntry.id } : {}),
    };
  }

  private async classifyFrameAnomaly(input: {
    llmClient: LLMClient;
    turnId: string;
    isUserTurn: boolean;
    userMessage: string;
    recentHistory: readonly RecencyMessage[];
    persistedUserEntryId?: StreamEntryId;
    streamWriter: StreamWriter;
  }): Promise<FrameAnomalyClassification | null> {
    if (!input.isUserTurn || input.persistedUserEntryId === undefined) {
      return null;
    }

    const classifier = new FrameAnomalyClassifier({
      llmClient: input.llmClient,
      model: this.options.config.anthropic.models.recallExpansion,
      tracer: this.options.tracer,
      turnId: input.turnId,
      onDegraded: (reason, error) => {
        if (!this.options.tracer.enabled) {
          return;
        }

        this.options.tracer.emit("frame_anomaly_classifier_degraded", {
          turnId: input.turnId,
          reason,
          ...(this.options.tracer.includePayloads && error !== undefined
            ? { error: error instanceof Error ? error.message : String(error) }
            : {}),
        });
      },
    });
    let classification = await classifier.classify({
      userMessage: input.userMessage,
      recentHistory: input.recentHistory,
    });

    if (classification.status === "degraded") {
      const fallback = classifyFrameAnomalyDegradedFallback(input.userMessage);

      if (fallback.matched) {
        if (this.options.tracer.enabled) {
          this.options.tracer.emit("frame_anomaly_degraded_fallback_match", {
            turnId: input.turnId,
            pattern: fallback.pattern,
            kind: fallback.kind,
          });
        }

        classification = fallback.classification;
      } else if (this.options.tracer.enabled) {
        this.options.tracer.emit("frame_anomaly_degraded_fallback_normal", {
          turnId: input.turnId,
        });
      }
    }

    if (isFrameAnomaly(classification)) {
      await this.appendFrameAnomalyEvents({
        streamWriter: input.streamWriter,
        turnId: input.turnId,
        persistedUserEntryId: input.persistedUserEntryId,
        classification,
      });
    }

    return classification;
  }

  private async classifyClosureLoop(input: {
    llmClient: LLMClient;
    turnId: string;
    isUserTurn: boolean;
    userMessage: string;
    recentHistory: readonly RecencyMessage[];
    persistedUserEntryId?: StreamEntryId;
    workingMemory: WorkingMemory;
    streamWriter: StreamWriter;
  }): Promise<ClosureLoopAssessment | null> {
    if (!input.isUserTurn || input.persistedUserEntryId === undefined) {
      return null;
    }

    const activeClosureLoop = input.workingMemory.discourse_state?.closure_loop ?? null;
    const closurePressureHistory =
      input.workingMemory.discourse_state?.closure_pressure_history ?? [];

    if (
      activeClosureLoop === null &&
      closurePressureHistory.length === 0 &&
      input.recentHistory.length < 4
    ) {
      return null;
    }

    const messages = buildClosureLoopMessageWindow({
      recentHistory: input.recentHistory,
      currentUserMessage: input.userMessage,
      currentUserEntryId: input.persistedUserEntryId,
    });
    const classifier = new ClosureLoopClassifier({
      llmClient: input.llmClient,
      model: this.options.config.anthropic.models.recallExpansion,
      tracer: this.options.tracer,
      turnId: input.turnId,
      onDegraded: (reason, error) => {
        if (!this.options.tracer.enabled) {
          return;
        }

        this.options.tracer.emit("closure_loop_classifier_degraded", {
          turnId: input.turnId,
          reason,
          ...(this.options.tracer.includePayloads && error !== undefined
            ? { error: error instanceof Error ? error.message : String(error) }
            : {}),
        });
      },
    });
    const classification = await classifier.classify({
      messages,
    });

    if (classification.degraded) {
      await this.appendHookFailureEvent(input.streamWriter, "closure_loop_classifier", null, {
        turnId: input.turnId,
        reason: classification.rationale,
      });

      return assessDegradedClosureLoopFallback({
        suppliedMessages: messages,
        currentUserRef: input.persistedUserEntryId,
        priorClosureLoopActive: activeClosureLoop !== null,
      });
    }

    return assessClosureLoopClassification({
      classification,
      suppliedMessages: messages,
      currentUserRef: input.persistedUserEntryId,
    });
  }

  private async appendFrameAnomalyEvents(input: {
    streamWriter: StreamWriter;
    turnId: string;
    persistedUserEntryId: StreamEntryId;
    classification: ActualFrameAnomalyClassification;
  }): Promise<void> {
    try {
      await input.streamWriter.appendMany([
        {
          kind: "internal_event",
          turn_id: input.turnId,
          content: {
            event: "frame_anomaly_gate",
            turn_id: input.turnId,
            source_stream_entry_id: input.persistedUserEntryId,
            cited_stream_entry_ids: [input.persistedUserEntryId],
            kind: input.classification.kind,
            confidence: input.classification.confidence,
            rationale: input.classification.rationale,
          },
        },
        {
          kind: "internal_event",
          turn_id: input.turnId,
          content: {
            event: QUARANTINED_USER_ENTRY_EVENT,
            turn_id: input.turnId,
            source_stream_entry_id: input.persistedUserEntryId,
            cited_stream_entry_ids: [input.persistedUserEntryId],
            kind: input.classification.kind,
            confidence: input.classification.confidence,
            rationale: input.classification.rationale,
          },
        },
      ]);

      if (this.options.tracer.enabled) {
        this.options.tracer.emit("frame_anomaly_quarantine_appended", {
          turnId: input.turnId,
          kind: input.classification.kind,
          sourceStreamEntryId: input.persistedUserEntryId,
        });
      }
    } catch (error) {
      await this.appendHookFailureEvent(input.streamWriter, "frame_anomaly_gate_event", error, {
        turnId: input.turnId,
      });
    }
  }

  private async suppressFromClosureLoop(input: {
    turnId: string;
    turnInput: TurnPhaseInput;
    streamWriter: StreamWriter;
    workingMemory: WorkingMemory;
    persistedUserEntryId?: StreamEntry["id"];
    correctiveCommitment: Parameters<
      CorrectivePreferenceTurnService["persistCommitment"]
    >[0]["commitment"];
    perceptionMode: CognitiveMode;
    reason: string;
  }): Promise<TurnPhaseResult> {
    let workingMemory = this.options.discourseStateService.markClosureLoopNamed({
      workingMemory: input.workingMemory,
      reason: input.reason,
      turnId: input.turnId,
      sourceStreamEntryId: input.persistedUserEntryId,
    });
    workingMemory = this.options.discourseStateService.setStopState({
      workingMemory,
      provenance: "finalizer_no_output",
      sourceStreamEntryId: input.persistedUserEntryId,
      reason: "Closure loop already named; suppressing another closure-only turn.",
      turnId: input.turnId,
    });
    const suppressionActionResult = await performAction({
      response: "",
      emission: {
        kind: "suppressed",
        reason: "finalizer_no_output",
      },
      toolCalls: [],
      intents: [],
      workingMemory: {
        ...workingMemory,
        updated_at: this.options.clock.now(),
      },
    });
    const suppressionMarker = await this.options.discourseStateService.appendSuppressionMarker({
      streamWriter: input.streamWriter,
      reason: "finalizer_no_output",
      userEntryId: input.persistedUserEntryId,
      turnId: input.turnId,
      audience: input.turnInput.audience,
    });
    const suppressionEmission: TurnEmission = {
      kind: "suppressed",
      reason: "finalizer_no_output",
      markerEntryId: suppressionMarker.id,
    };
    const suppressedWorkingMemory = this.options.discourseStateService.applySuppressedEmissionState(
      {
        workingMemory: suppressionActionResult.workingMemory,
        reason: "finalizer_no_output",
        sourceStreamEntryId: suppressionMarker.id,
        turnId: input.turnId,
      },
    );

    if (this.options.tracer.enabled) {
      this.options.tracer.emit("generation_suppressed", {
        turnId: input.turnId,
        reason: "finalizer_no_output",
        streamEntryId: suppressionMarker.id,
        source: "closure_loop",
        classified: true,
      });
    }

    this.options.workingMemoryStore.save({
      ...suppressedWorkingMemory,
      updated_at: this.options.clock.now(),
    });
    await this.persistCorrectiveCommitment(input.streamWriter, input.correctiveCommitment);

    return {
      mode: input.perceptionMode,
      path: "suppressed",
      response: "",
      emitted: false,
      emission: suppressionEmission,
      thoughts: [],
      usage: {
        input_tokens: 0,
        output_tokens: 0,
        stop_reason: "suppressed",
      },
      retrievedEpisodeIds: [],
      referencedEpisodeIds: [],
      intents: [],
      toolCalls: [],
    };
  }

  private async suppressFromGenerationGate(input: {
    turnId: string;
    turnInput: TurnPhaseInput;
    streamWriter: StreamWriter;
    workingMemory: WorkingMemory;
    persistedUserEntryId?: StreamEntry["id"];
    gateResult: Awaited<ReturnType<GenerationGate["evaluate"]>>;
    correctiveCommitment: Parameters<
      CorrectivePreferenceTurnService["persistCommitment"]
    >[0]["commitment"];
    perceptionMode: CognitiveMode;
  }): Promise<TurnPhaseResult> {
    let workingMemory = input.workingMemory;
    const suppressionReason = input.gateResult.reason ?? "generation_gate";
    const activeStop = workingMemory.discourse_state?.stop_until_substantive_content ?? null;

    if (activeStop === null) {
      workingMemory = this.options.discourseStateService.setStopState({
        workingMemory,
        provenance: "generation_gate",
        sourceStreamEntryId: input.persistedUserEntryId,
        reason: input.gateResult.explanation,
        turnId: input.turnId,
      });
    }

    const suppressionActionResult = await performAction({
      response: "",
      emission: {
        kind: "suppressed",
        reason: suppressionReason,
      },
      toolCalls: [],
      intents: [],
      workingMemory: {
        ...workingMemory,
        updated_at: this.options.clock.now(),
      },
    });
    const suppressionMarker = await this.options.discourseStateService.appendSuppressionMarker({
      streamWriter: input.streamWriter,
      reason: suppressionReason,
      userEntryId: input.persistedUserEntryId,
      turnId: input.turnId,
      audience: input.turnInput.audience,
    });
    const suppressionEmission: TurnEmission = {
      kind: "suppressed",
      reason: suppressionReason,
      markerEntryId: suppressionMarker.id,
    };
    const suppressedWorkingMemory = this.options.discourseStateService.applySuppressedEmissionState(
      {
        workingMemory: suppressionActionResult.workingMemory,
        reason: suppressionReason,
        sourceStreamEntryId: suppressionMarker.id,
        turnId: input.turnId,
      },
    );

    if (this.options.tracer.enabled) {
      this.options.tracer.emit("generation_suppressed", {
        turnId: input.turnId,
        reason: suppressionReason,
        streamEntryId: suppressionMarker.id,
        source: "generation_gate",
        classified: input.gateResult.classified,
      });
    }

    this.options.workingMemoryStore.save({
      ...suppressedWorkingMemory,
      updated_at: this.options.clock.now(),
    });
    await this.persistCorrectiveCommitment(input.streamWriter, input.correctiveCommitment);

    return {
      mode: input.perceptionMode,
      path: "suppressed",
      response: "",
      emitted: false,
      emission: suppressionEmission,
      thoughts: [],
      usage: {
        input_tokens: 0,
        output_tokens: 0,
        stop_reason: "suppressed",
      },
      retrievedEpisodeIds: [],
      referencedEpisodeIds: [],
      intents: [],
      toolCalls: [],
    };
  }

  private async suppressFromAction(input: {
    turnId: string;
    turnInput: TurnPhaseInput;
    streamWriter: StreamWriter;
    actionResult: Awaited<ReturnType<TurnActionCoordinator["run"]>>["actionResult"];
    actionEmission: Extract<PendingTurnEmission, { kind: "suppressed" }>;
    persistedAgentEntry: StreamEntry;
    correctiveCommitment: Parameters<
      CorrectivePreferenceTurnService["persistCommitment"]
    >[0]["commitment"];
    perceptionMode: CognitiveMode;
    deliberation: Awaited<ReturnType<Deliberator["run"]>>;
  }): Promise<TurnPhaseResult> {
    const suppressionEmission: TurnEmission = {
      kind: "suppressed",
      reason: input.actionEmission.reason,
      markerEntryId: input.persistedAgentEntry.id,
    };
    let suppressedWorkingMemory = this.options.discourseStateService.applySuppressedEmissionState({
      workingMemory: input.actionResult.workingMemory,
      reason: input.actionEmission.reason,
      sourceStreamEntryId: input.persistedAgentEntry.id,
      turnId: input.turnId,
    });
    if (
      input.actionEmission.closure_pressure_history_reason !== undefined &&
      input.actionEmission.reason !== "closure_pressure_only" &&
      input.actionEmission.reason !== "closure_response_audit_failed_closed"
    ) {
      suppressedWorkingMemory = this.options.discourseStateService.appendClosurePressureHistory({
        workingMemory: suppressedWorkingMemory,
        turnId: input.turnId,
        reason: input.actionEmission.closure_pressure_history_reason,
      });
    }

    if (this.options.tracer.enabled) {
      this.options.tracer.emit("generation_suppressed", {
        turnId: input.turnId,
        reason: input.actionEmission.reason,
        streamEntryId: input.persistedAgentEntry.id,
      });
    }

    this.options.workingMemoryStore.save({
      ...suppressedWorkingMemory,
      updated_at: this.options.clock.now(),
    });
    await this.persistCorrectiveCommitment(input.streamWriter, input.correctiveCommitment);

    return {
      mode: input.perceptionMode,
      path: "suppressed",
      response: "",
      emitted: false,
      emission: suppressionEmission,
      thoughts: input.deliberation.thoughts,
      usage: input.deliberation.usage,
      retrievedEpisodeIds: input.deliberation.retrievedEpisodes.map((result) => result.episode.id),
      referencedEpisodeIds: [...(input.deliberation.referencedEpisodeIds ?? [])],
      intents: [],
      toolCalls: [...input.actionResult.tool_calls],
    };
  }

  private async buildEvidenceLedgerFinalizerContext(
    input: EvidenceLedgerFinalizerBuildInput,
  ): Promise<EvidenceLedgerFinalizerContext> {
    const config = this.options.config.generation.evidenceLedger;

    if (!config.enabled) {
      return {
        ledger: null,
        promptSection: null,
      };
    }

    const builder = new EvidenceLedgerBuilder({
      createStreamReader: this.options.createStreamReader,
      relationalSlotRepository: this.options.relationalSlotRepository,
      actionRepository: this.options.actionRepository,
      commitmentRepository: this.options.commitmentRepository,
      goalsRepository: this.options.goalsRepository,
      openQuestionsRepository: this.options.openQuestionsRepository,
      currentSessionTranscriptTokenBudget: config.currentSessionTranscriptTokenBudget,
      actionThreadRenderLimit: config.actionThreadRenderLimit,
      actionThreadSimilarityThreshold: config.actionThreadSimilarityThreshold,
      actionThreadSourceRecordLimit: config.actionThreadSourceRecordLimit,
      entityRepository: this.options.entityRepository,
    });
    const builtLedger = await builder.build(input);
    const compacted = compactEvidenceLedger(builtLedger, {
      targetTokens: config.finalizerTargetTokens,
      hardCapTokens: config.finalizerHardCapTokens,
      maxEntryTextTokens: config.finalizerMaxEntryTextTokens,
      sectionOptions: config.sectionOptions,
    });
    const ledgerWithoutArtifact = compacted.ledger;
    const renderedWithoutArtifact = renderEvidenceLedger(ledgerWithoutArtifact);
    const decisionArtifact = await this.compileDecisionArtifactForEvidenceLedger({
      input,
      ledger: ledgerWithoutArtifact,
      promptVisibleLedger: renderedWithoutArtifact ?? "",
    });
    const ledger = this.withDecisionArtifact(ledgerWithoutArtifact, decisionArtifact);
    const rendered = renderEvidenceLedger(ledger);
    const decisionArtifactSummary = summarizeDecisionStateArtifactRender(ledger.decisionArtifact);
    const traceSummary = summarizeEvidenceLedgerTrace({
      ...ledger,
      estimatedTokens: estimateEvidenceLedgerPromptTokens(ledger),
    });

    if (
      this.options.tracer.enabled &&
      input.turnId !== undefined &&
      evidenceLedgerCompactionChanged(compacted.traceSummary)
    ) {
      this.options.tracer.emit("evidence_ledger_compacted", {
        turnId: input.turnId,
        pre_dedupe_tokens: compacted.traceSummary.preDedupeTokens,
        post_dedupe_tokens: compacted.traceSummary.postDedupeTokens,
        pre_cap_tokens: compacted.traceSummary.preCapTokens,
        post_section_cap_tokens: compacted.traceSummary.postSectionCapTokens,
        post_cap_tokens: compacted.traceSummary.postCapTokens,
        deduped_entry_count: compacted.traceSummary.dedupedEntryCount,
        omitted_entry_counts: toTraceJsonValue(compacted.traceSummary.omittedEntryCountsBySection),
        dropped_sections: compacted.traceSummary.droppedSections,
        target_tokens: compacted.traceSummary.targetTokens,
        hard_cap_tokens: compacted.traceSummary.hardCapTokens,
      });
    }

    if (this.options.tracer.enabled && input.turnId !== undefined) {
      this.options.tracer.emit("evidence_ledger_built", {
        turnId: input.turnId,
        entry_counts: toTraceJsonValue(traceSummary.entryCountsBySection),
        transcript_included: traceSummary.transcriptIncluded,
        transcript_compacted: traceSummary.transcriptCompacted,
        transcript_omitted_reason: traceSummary.transcriptOmittedReason ?? null,
        original_transcript_token_estimate: traceSummary.originalTranscriptTokenEstimate,
        compacted_transcript_token_estimate: traceSummary.compactedTranscriptTokenEstimate,
        compacted_entry_count: traceSummary.compactedEntryCount,
        raw_preserved_user_entry_count: traceSummary.rawPreservedUserEntryCount,
        total_estimated_tokens: traceSummary.totalEstimatedTokens,
        estimated_tokens_by_section: toTraceJsonValue(traceSummary.estimatedTokensBySection),
        decision_artifact_entry_count: decisionArtifactSummary.renderedEntryCount,
        decision_artifact_rendered_token_estimate: decisionArtifactSummary.estimatedTokens,
      });
    }

    return {
      ledger,
      promptSection: rendered,
    };
  }

  private async compileDecisionArtifactForEvidenceLedger(input: {
    input: EvidenceLedgerFinalizerBuildInput;
    ledger: EvidenceLedger;
    promptVisibleLedger: string;
  }): Promise<DecisionArtifact | null> {
    const audienceEntityId = input.input.audienceEntityId;

    if (audienceEntityId === null) {
      return null;
    }

    const previousArtifact = this.options.decisionArtifactRepository.get(audienceEntityId);

    if (!input.input.isUserTurn || input.input.currentUserEntry === undefined) {
      return previousArtifact;
    }

    const selfEntityId = this.options.entityRepository.resolve("self", {
      kind: "self",
      provenance: "assistant_seeded",
    });

    await compileDecisionArtifact({
      llmClient: this.options.llmFactory(),
      model: this.options.config.anthropic.models.recallExpansion,
      repository: this.options.decisionArtifactRepository,
      audienceEntityId,
      selfEntityId,
      speakerEntityId: input.input.currentUserEntry.sender_entity_id,
      participants: (input.input.activeParticipants ?? []).map((participant) => ({
        entityId: participant.entityId,
        displayName: participant.displayName,
      })),
      currentUserMessage: input.input.currentUserMessage,
      currentUserStreamEntryId: input.input.currentUserEntry.id,
      promptVisibleLedger: input.promptVisibleLedger,
      previousArtifact,
      allowedSourceStreamEntryIds: collectDecisionArtifactAllowedStreamEntryIds(
        input.ledger,
        input.input.currentUserEntry.id,
      ),
      clock: this.options.clock,
      tracer: this.options.tracer,
      turnId: input.input.turnId,
    });

    return this.options.decisionArtifactRepository.get(audienceEntityId);
  }

  private withDecisionArtifact(
    ledger: EvidenceLedger,
    decisionArtifact: DecisionArtifact | null,
  ): EvidenceLedger {
    const ledgerWithArtifact = {
      ...ledger,
      decisionArtifact,
    };

    return {
      ...ledgerWithArtifact,
      estimatedTokens: estimateEvidenceLedgerPromptTokens(ledgerWithArtifact),
    };
  }

  private async persistCorrectiveCommitment(
    streamWriter: StreamWriter,
    commitment: Parameters<CorrectivePreferenceTurnService["persistCommitment"]>[0]["commitment"],
  ): Promise<void> {
    await this.options.correctivePreferenceTurnService.persistCommitment({
      commitment,
      onHookFailure: (hook, error, details) =>
        this.appendHookFailureEvent(streamWriter, hook, error, details),
    });
  }

  private async appendHookFailureEvent(
    streamWriter: StreamWriter,
    hook: string,
    error: unknown,
    details?: Record<string, unknown>,
  ): Promise<void> {
    await appendInternalFailureEvent(streamWriter, hook, error, details);
  }

  private async catchUpStreamIngestion(
    sessionId: SessionId,
    streamWriter: StreamWriter,
  ): Promise<void> {
    const coordinator = this.options.streamIngestionCoordinator;

    if (coordinator === undefined) {
      return;
    }

    try {
      const result = await coordinator.catchUp(sessionId, {
        maxEntries: this.options.config.streamIngestion.preTurnCatchup.maxEntries,
      });

      if (result.error !== undefined) {
        await this.appendHookFailureEvent(
          streamWriter,
          "stream_ingestion_pre_turn_catchup",
          result.error,
          {
            processedEntries: result.processedEntries,
          },
        );
      }
    } catch (error) {
      await this.appendHookFailureEvent(streamWriter, "stream_ingestion_pre_turn_catchup", error);
    }
  }

  private startLiveIngestion(sessionId: SessionId): void {
    if (this.options.streamIngestionCoordinator !== undefined) {
      void this.options.streamIngestionCoordinator.ingest(sessionId).catch((error) => {
        console.error("Live stream ingestion failed", error);
      });
    }
  }
}
