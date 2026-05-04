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
import type { TurnDiscourseStateService } from "../generation/turn-discourse-state.js";
import type { PendingTurnEmission, TurnEmission } from "../generation/types.js";
import { GenerationGate } from "../generation/generation-gate.js";
import { StopCommitmentExtractor } from "../generation/self-stop-commitment.js";
import type { TurnGoalPromotionService } from "../goals/turn-goal-promotion-service.js";
import type { PerceptionGateway } from "../perception/gateway.js";
import type { TurnOpeningPersistence } from "../persistence/turn-opening.js";
import type { TurnReflectionCoordinator } from "../reflection/turn-reflection-coordinator.js";
import type { TurnRetrievalCoordinator } from "../retrieval/turn-coordinator.js";
import type { TurnSelfContextBuilder } from "../self/turn-self-context.js";
import type { TurnTracer } from "../tracing/tracer.js";
import type { CognitiveMode, IntentRecord } from "../types.js";
import type { Config } from "../../config/index.js";
import type { EmbeddingClient } from "../../embeddings/index.js";
import type { LLMClient } from "../../llm/index.js";
import type { EntityRepository } from "../../memory/commitments/index.js";
import type { RelationalSlotRepository } from "../../memory/relational-slots/index.js";
import { appendInternalFailureEvent } from "../../memory/self/index.js";
import type { SocialRepository } from "../../memory/social/index.js";
import type { WorkingMemory, WorkingMemoryStore } from "../../memory/working/index.js";
import type { StreamEntry, StreamWriter } from "../../stream/index.js";
import type { ToolDispatcher } from "../../tools/index.js";
import type { Clock } from "../../util/clock.js";
import type { SessionId } from "../../util/ids.js";
import type { StreamIngestionCoordinator } from "../ingestion/index.js";
import type { TurnRelationalGuardRunner } from "../generation/turn-relational-guard.js";
import type { TurnLifecycleTracker } from "./turn-lifecycle-tracker.js";

const ACTIVE_TURN_STATUS = "active";

export type TurnPhaseInput = {
  userMessage: string;
  audience?: string;
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
  toolDispatcher: ToolDispatcher;
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
  relationalGuardRunner: Pick<TurnRelationalGuardRunner, "listRecentCompletedActions">;
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

export class TurnPhaseCoordinator {
  constructor(private readonly options: TurnPhaseCoordinatorOptions) {}

  async run(input: RunTurnPhasesInput): Promise<TurnPhaseResult> {
    const turnInput = input.input;
    const sessionId = input.sessionId;
    const turnId = input.turnId;
    const streamWriter = input.streamWriter;
    const lifecycleTracker = input.lifecycleTracker;
    const isSelfAudience = turnInput.audience === "self";

    await this.catchUpStreamIngestion(sessionId, streamWriter);
    let workingMemory = this.options.workingMemoryStore.load(sessionId);
    lifecycleTracker.captureInitialWorkingMemory(workingMemory);
    const turnPerception = this.options.perceptionGateway.beginTurn({
      turnId,
      onHookFailure: (hook, error, details) =>
        this.appendHookFailureEvent(streamWriter, hook, error, details),
    });
    const llmClient = this.options.llmFactory();
    const isUserTurn = turnInput.origin !== "autonomous";
    const cognitionInput =
      turnInput.autonomyTrigger === null || turnInput.autonomyTrigger === undefined
        ? turnInput.userMessage
        : formatAutonomyTriggerContext(turnInput.autonomyTrigger);
    const audienceEntityId =
      turnInput.audience === undefined || isSelfAudience
        ? null
        : this.options.entityRepository.resolve(turnInput.audience);
    const audienceEntity =
      audienceEntityId === null ? null : this.options.entityRepository.get(audienceEntityId);
    let audienceProfile =
      audienceEntityId === null ? null : this.options.socialRepository.getProfile(audienceEntityId);
    const perceptionResult = await turnPerception.perceive({
      sessionId,
      isSelfAudience,
      origin: turnInput.origin,
      cognitionInput,
      workingMemory,
    });
    const perception = perceptionResult.perception;
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

    const correctivePreferenceTurn =
      await this.options.correctivePreferenceTurnService.extractAndApply({
        llmClient,
        turnId,
        userMessage: turnInput.userMessage,
        persistedUserEntryId,
        recentHistory: recencyWindow.messages,
        audienceEntityId,
        sessionId,
        onHookFailure: (hook, error, details) =>
          this.appendHookFailureEvent(streamWriter, hook, error, details),
        trackAppliedSlotNegation: (slot) => lifecycleTracker.trackAppliedSlotNegation(slot),
      });
    const correctiveCommitment = correctivePreferenceTurn.commitment;
    workingMemory = correctivePreferenceTurn.workingMemory;
    const createdActionIds = await this.options.turnActionStateService.extract({
      llmClient,
      turnId,
      isUserTurn,
      userMessage: turnInput.userMessage,
      persistedUserEntryId,
      recentHistory: recencyWindow.messages,
      audienceEntityId,
    });
    lifecycleTracker.trackCreatedActionIds(createdActionIds);
    const activeGoalsForPromotion = isUserTurn
      ? await this.options.selfContextBuilder.listActiveGoalsVisibleToAudience(audienceEntityId)
      : [];
    const persistedPromotions = await this.options.turnGoalPromotionService.extractAndPersist({
      llmClient,
      turnId,
      isUserTurn,
      userMessage: turnInput.userMessage,
      recentHistory: recencyWindow.messages,
      audienceEntityId,
      temporalCue: perception.temporalCue,
      activeGoals: activeGoalsForPromotion,
      persistedUserEntryId,
      onHookFailure: (hook, error, details) =>
        this.appendHookFailureEvent(streamWriter, hook, error, details),
    });
    lifecycleTracker.trackCreatedGoalIds(persistedPromotions.goalIds);
    lifecycleTracker.trackCreatedExecutiveStepIds(persistedPromotions.executiveStepIds);

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

    const selfContext = await this.options.selfContextBuilder.build({
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
    const relationalSlots = this.options.relationalSlotRepository.listConstrained({
      limit: 24,
    });
    const deliberator = new Deliberator({
      llmClient,
      toolDispatcher: this.options.toolDispatcher,
      cognitionModel: this.options.config.anthropic.models.cognition,
      backgroundModel: this.options.config.anthropic.models.background,
      clock: this.options.clock,
      tracer: this.options.tracer,
    });
    const deliberation = await deliberator.run(
      {
        sessionId,
        turnId,
        audience: turnInput.audience,
        audienceEntityId,
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
        selectedSkill,
        entityRepository: this.options.entityRepository,
        workingMemory,
        recentCompletedActions:
          this.options.relationalGuardRunner.listRecentCompletedActions(audienceEntityId),
        affectiveTrajectory,
        selfSnapshot,
        executiveFocus: executiveFocusWithStep,
        audienceProfile,
        recencyMessages: recencyWindow.messages,
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
            ...(turnInput.audience === undefined ? {} : { audience: turnInput.audience }),
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

    const messageEmission: TurnEmission = {
      kind: "message",
      content: actionResult.response,
      agentMessageId: persistedAgentEntry.id,
    };
    let postActionWorkingMemory = actionResult.workingMemory;
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
      pendingSocialAttribution,
      suppressionSet,
      persistedUserEntryId,
      persistedPerceptionEntry,
      persistedAgentEntry,
      isUserTurn,
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
      emitted: true,
      emission: messageEmission,
      thoughts: deliberation.thoughts,
      usage: deliberation.usage,
      retrievedEpisodeIds: deliberation.retrievedEpisodes.map((result) => result.episode.id),
      referencedEpisodeIds: [...(deliberation.referencedEpisodeIds ?? [])],
      intents: actionResult.intents,
      toolCalls: [...actionResult.tool_calls],
      agentMessageId: persistedAgentEntry.id,
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
      ...suppressionActionResult.workingMemory,
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
    const suppressedWorkingMemory = this.options.discourseStateService.applySuppressedEmissionState(
      {
        workingMemory: input.actionResult.workingMemory,
        reason: input.actionEmission.reason,
        sourceStreamEntryId: input.persistedAgentEntry.id,
        turnId: input.turnId,
      },
    );

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
