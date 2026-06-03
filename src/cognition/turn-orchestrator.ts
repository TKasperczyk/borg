import { randomUUID } from "node:crypto";
import { performance } from "node:perf_hooks";

import type {
  AttachmentRepository,
  AttachmentService,
  ImagePerceptionRepository,
  ImagePerceptionService,
} from "../attachments/index.js";
import type { Config } from "../config/index.js";
import type { ExecutiveStepsRepository } from "../executive/index.js";
import type { EmbeddingClient } from "../embeddings/index.js";
import type { LLMClient } from "../llm/index.js";
import { MoodRepository } from "../memory/affective/index.js";
import type { ActivityRepository } from "../memory/activity/index.js";
import type { ActionRepository } from "../memory/actions/index.js";
import { CommitmentRepository, EntityRepository } from "../memory/commitments/index.js";
import type { CreatorDirectiveRepository } from "../memory/creator-directives/index.js";
import type { SharedStateRepository } from "../memory/decision-artifacts/index.js";
import { EpisodicRepository } from "../memory/episodic/index.js";
import type { IdentityService } from "../memory/identity/index.js";
import type { ObservedEventRepository } from "../memory/observed-events/index.js";
import { SkillSelector } from "../memory/procedural/index.js";
import { RelationalSlotRepository } from "../memory/relational-slots/index.js";
import type { SelfDecisionRepository } from "../memory/self-decisions/index.js";
import {
  AutobiographicalRepository,
  GoalsRepository,
  GrowthMarkersRepository,
  TraitsRepository,
  ValuesRepository,
  type OpenQuestionsRepository,
} from "../memory/self/index.js";
import { ReviewQueueRepository, type SemanticNodeRepository } from "../memory/semantic/index.js";
import { SocialRepository } from "../memory/social/index.js";
import { WorkingMemoryStore, type WorkingMemory } from "../memory/working/index.js";
import type { RetrievalPipeline } from "../retrieval/index.js";
import type {
  AutonomousOutboundPolicy,
  OutboundDelivery,
  OutboundDeliveryReceipt,
} from "../outbound/index.js";
import type { SessionSourceType, SessionsRepository } from "../sessions/index.js";
import {
  ABORTED_TURN_EVENT,
  StreamReader,
  StreamWriter,
  type StreamEntryIndexRepository,
} from "../stream/index.js";
import type { ToolDispatcher } from "../tools/index.js";
import { SystemClock, type Clock } from "../util/clock.js";
import { SessionBusyError } from "../util/errors.js";
import { DEFAULT_SESSION_ID, type EntityId, type SessionId } from "../util/ids.js";
import type { ToolLoopCallRecord } from "./turn-action/index.js";
import { TurnActionCoordinator } from "./turn-action/turn-action-coordinator.js";
import { TurnActionStateService } from "./actions/turn-action-state-service.js";
import { AttributionLifecycleService } from "./attribution/lifecycle-service.js";
import { CommitmentGuardRunner } from "./commitments/guard-runner.js";
import { CorrectivePreferenceTurnService } from "./commitments/corrective-preference-service.js";
import { CreatorDirectiveTurnService } from "./creator-directives/service.js";
import type { SelfSnapshot } from "./deliberation/deliberator.js";
import { TurnDiscourseStateService } from "./generation/turn-discourse-state.js";
import { TurnPostGenerationGuardRunner } from "./generation/turn-post-generation-guard.js";
import type { TurnEmission } from "./generation/types.js";
import { TurnGoalPromotionService } from "./goals/turn-goal-promotion-service.js";
import type {
  ChatResponseWatermarkCoordinator,
  StreamIngestionCoordinator,
} from "./ingestion/index.js";
import { TurnLifecycleTracker } from "./lifecycle/turn-lifecycle-tracker.js";
import { TurnPhaseCoordinator } from "./lifecycle/turn-phase-coordinator.js";
import type { TurnPhaseCoordinatorInput } from "./lifecycle/turn-phase/types.js";
import type { PromptOverrideRepository } from "./prompts/override-repository.js";
import { detectAffectiveSignal } from "./perception/affective-signal.js";
import { PerceptionGateway } from "./perception/gateway.js";
import { TurnOpeningPersistence } from "./persistence/turn-opening.js";
import { PendingProceduralAttemptTracker } from "./procedural/pending-attempt-tracker.js";
import { TurnContextCompiler } from "./recency/index.js";
import type { Reflector } from "./reflection/index.js";
import { TurnReflectionCoordinator } from "./reflection/turn-reflection-coordinator.js";
import { TurnRetrievalCoordinator } from "./retrieval/turn-coordinator.js";
import { SessionLock } from "./session-lock.js";
import { TurnSelfContextBuilder } from "./self/turn-self-context.js";
import { NOOP_TRACER, type TurnTerminalOutcome, type TurnTracer } from "./tracing/tracer.js";
import { isInboundBatchTurnInput, type TurnOrchestratorInput } from "./turn-input.js";
import { isAutonomousLikeTurnOrigin, type CognitiveMode, type IntentRecord } from "./types.js";

export type TurnResult = {
  turn_id: string;
  mode: CognitiveMode;
  path: "system_1" | "system_2" | "suppressed";
  response: string;
  emitted: boolean;
  emission: TurnEmission;
  thoughts: string[];
  usage: {
    input_tokens: number;
    output_tokens: number;
    cache_creation_input_tokens?: number;
    cache_read_input_tokens?: number;
    stop_reason: string | null;
  };
  retrievedEpisodeIds: string[];
  referencedEpisodeIds: string[];
  intents: IntentRecord[];
  toolCalls: ToolLoopCallRecord[];
  agentMessageId?: string;
  outboundDelivery?: OutboundDeliveryReceipt;
};

export type TurnOrchestratorOptions = {
  config: Config;
  retrievalPipeline: RetrievalPipeline;
  embeddingClient: EmbeddingClient;
  episodicRepository: EpisodicRepository;
  semanticNodeRepository?: SemanticNodeRepository;
  valuesRepository: ValuesRepository;
  goalsRepository: GoalsRepository;
  traitsRepository: TraitsRepository;
  autobiographicalRepository?: AutobiographicalRepository;
  growthMarkersRepository?: GrowthMarkersRepository;
  executiveStepsRepository: ExecutiveStepsRepository;
  moodRepository: MoodRepository;
  actionRepository: ActionRepository;
  socialRepository: SocialRepository;
  skillSelector: SkillSelector;
  relationalSlotRepository: RelationalSlotRepository;
  entityRepository: EntityRepository;
  commitmentRepository: CommitmentRepository;
  creatorDirectiveRepository: CreatorDirectiveRepository;
  sharedStateRepository: SharedStateRepository;
  activityRepository?: ActivityRepository;
  selfDecisionRepository?: SelfDecisionRepository;
  observedEventRepository?: ObservedEventRepository;
  identityService: IdentityService;
  reviewQueueRepository: ReviewQueueRepository;
  openQuestionsRepository: OpenQuestionsRepository;
  workingMemoryStore: WorkingMemoryStore;
  llmFactory: () => LLMClient;
  createReflector: (llmClient: LLMClient) => Reflector;
  toolDispatcher: ToolDispatcher;
  clock?: Clock;
  createStreamWriter: (sessionId: SessionId) => StreamWriter;
  entryIndex?: StreamEntryIndexRepository;
  attachmentRepository: Pick<
    AttachmentRepository,
    "get" | "isActiveForStreamEntry" | "listByParentEntry"
  >;
  imagePerceptionRepository: Pick<ImagePerceptionRepository, "listByParentEntries">;
  attachmentService: AttachmentService;
  imagePerceptionService?: ImagePerceptionService;
  /**
   * Build a reader for the given session's stream. The orchestrator uses
   * this to compile the recent-dialogue window before a turn starts, so the
   * LLM can see its own prior responses without the working-memory
   * scratchpad indirection. Defaults to the standard on-disk reader.
   */
  createStreamReader?: (sessionId: SessionId) => StreamReader;
  /**
   * Compiles the recency window (recent user/assistant messages) from the
   * stream for every turn. A default is constructed if not provided.
   */
  turnContextCompiler?: TurnContextCompiler;
  /**
   * If provided, fires best-effort live episodic extraction after each turn
   * and bounded catch-up before the next turn's retrieval. If omitted, the
   * orchestrator skips live extraction entirely and relies on explicit
   * `borg.episodic.extract()` / `borg.dream.consolidate()` calls.
   */
  streamIngestionCoordinator?: StreamIngestionCoordinator;
  chatResponseWatermarkCoordinator?: ChatResponseWatermarkCoordinator;
  outboundDelivery?: Pick<OutboundDelivery, "deliver">;
  autonomousOutboundPolicy?: Pick<AutonomousOutboundPolicy, "promptContext">;
  outboundSourceTypes?: readonly SessionSourceType[];
  affectiveSignalDetector?: typeof detectAffectiveSignal;
  sessionLock?: SessionLock;
  tracer?: TurnTracer;
  promptOverrideRepository?: Pick<PromptOverrideRepository, "get">;
  sessionsRepository?: Pick<SessionsRepository, "count" | "get" | "list">;
};

function stripTurnLockMode(input: TurnOrchestratorInput): TurnPhaseCoordinatorInput {
  if (isInboundBatchTurnInput(input)) {
    const { lockMode: _lockMode, ...phaseInput } = input;
    return phaseInput;
  }

  const { lockMode: _lockMode, ...phaseInput } = input;
  return phaseInput;
}

export class TurnOrchestrator {
  private readonly clock: Clock;
  private readonly sessionLock: SessionLock;
  private readonly tracer: TurnTracer;
  private readonly selfContextBuilder: TurnSelfContextBuilder;
  private readonly turnPhaseCoordinator: TurnPhaseCoordinator;

  constructor(private readonly options: TurnOrchestratorOptions) {
    this.clock = options.clock ?? new SystemClock();
    this.tracer = options.tracer ?? NOOP_TRACER;
    const turnContextCompiler = options.turnContextCompiler ?? new TurnContextCompiler();
    this.sessionLock =
      options.sessionLock ??
      new SessionLock({
        dataDir: options.config.dataDir,
      });
    const createStreamReader =
      options.createStreamReader ??
      ((sessionId: SessionId) =>
        new StreamReader({
          dataDir: options.config.dataDir,
          sessionId,
          entryIndex: options.entryIndex,
        }));
    const perceptionGateway = new PerceptionGateway({
      config: options.config,
      llmFactory: () => options.llmFactory(),
      clock: this.clock,
      tracer: this.tracer,
      getAffectiveSignalDetector: () => options.affectiveSignalDetector,
      turnContextCompiler,
      createStreamReader,
    });
    const turnOpeningPersistence = new TurnOpeningPersistence({
      workingMemoryStore: options.workingMemoryStore,
      activityRepository: options.activityRepository,
    });
    const attributionLifecycleService = new AttributionLifecycleService({
      socialRepository: options.socialRepository,
      traitsRepository: options.traitsRepository,
      episodicRepository: options.episodicRepository,
      clock: this.clock,
    });
    const turnRetrievalCoordinator = new TurnRetrievalCoordinator({
      commitmentRepository: options.commitmentRepository,
      reviewQueueRepository: options.reviewQueueRepository,
      moodRepository: options.moodRepository,
      retrievalPipeline: options.retrievalPipeline,
      skillSelector: options.skillSelector,
      clock: this.clock,
      tracer: this.tracer,
    });
    const commitmentGuardRunner = new CommitmentGuardRunner({
      detectionModel: options.config.anthropic.models.background,
      rewriteModel: options.config.anthropic.models.cognition,
      mode: options.config.generation.postGenerationGuards.commitment.mode,
      criticalKinds: options.config.commitments.enforce.criticalKinds,
      regenerateBeforeSuppress: options.config.commitments.enforce.regenerateBeforeSuppress,
      rewriteOnViolation: options.config.commitments.enforce.rewriteOnViolation,
      entityRepository: options.entityRepository,
      tracer: this.tracer,
    });
    const pendingProceduralAttemptTracker = new PendingProceduralAttemptTracker();
    this.selfContextBuilder = new TurnSelfContextBuilder({
      embeddingClient: options.embeddingClient,
      episodicRepository: options.episodicRepository,
      valuesRepository: options.valuesRepository,
      goalsRepository: options.goalsRepository,
      traitsRepository: options.traitsRepository,
      autobiographicalRepository: options.autobiographicalRepository,
      growthMarkersRepository: options.growthMarkersRepository,
      executiveStepsRepository: options.executiveStepsRepository,
      clock: this.clock,
      tracer: this.tracer,
      goalFocusThreshold: options.config.executive.goalFocusThreshold,
      goalFollowupLookaheadMs: options.config.autonomy.triggers.goalFollowupDue.lookaheadMs,
      goalFollowupStaleMs: options.config.autonomy.triggers.goalFollowupDue.staleMs,
    });
    const correctivePreferenceTurnService = new CorrectivePreferenceTurnService({
      model: options.config.anthropic.models.recallExpansion,
      commitmentRepository: options.commitmentRepository,
      identityService: options.identityService,
      relationalSlotRepository: options.relationalSlotRepository,
      workingMemoryStore: options.workingMemoryStore,
      clock: this.clock,
      tracer: this.tracer,
    });
    const creatorDirectiveTurnService = new CreatorDirectiveTurnService({
      model: options.config.anthropic.models.creatorDirective,
      creatorDirectiveRepository: options.creatorDirectiveRepository,
      entityRepository: options.entityRepository,
      tracer: this.tracer,
    });
    const turnActionStateService = new TurnActionStateService({
      model: options.config.anthropic.models.recallExpansion,
      actionRepository: options.actionRepository,
      embeddingClient: options.embeddingClient,
      clock: this.clock,
      tracer: this.tracer,
    });
    const turnGoalPromotionService = new TurnGoalPromotionService({
      model: options.config.anthropic.models.recallExpansion,
      identityService: options.identityService,
      goalsRepository: options.goalsRepository,
      executiveStepsRepository: options.executiveStepsRepository,
      embeddingClient: options.embeddingClient,
      clock: this.clock,
      tracer: this.tracer,
    });
    const discourseStateService = new TurnDiscourseStateService({
      tracer: this.tracer,
      clock: this.clock,
    });
    const postGenerationGuardRunner = new TurnPostGenerationGuardRunner({
      auditModel: options.config.anthropic.models.background,
      closurePressureMode: options.config.generation.postGenerationGuards.closurePressure.mode,
      createStreamReader,
      actionRepository: options.actionRepository,
      relationalSlotRepository: options.relationalSlotRepository,
      clock: this.clock,
      tracer: this.tracer,
    });
    const turnActionCoordinator = new TurnActionCoordinator({
      commitmentGuardRunner,
      postGenerationGuardRunner,
      embeddingClient: options.embeddingClient,
      pendingActionJudgeModel: options.config.anthropic.models.background,
      clock: this.clock,
      tracer: this.tracer,
    });
    const turnReflectionCoordinator = new TurnReflectionCoordinator({
      moodRepository: options.moodRepository,
      socialRepository: options.socialRepository,
      openQuestionsRepository: options.openQuestionsRepository,
      workingMemoryStore: options.workingMemoryStore,
      pendingProceduralAttemptTracker,
      createReflector: (llmClient) => options.createReflector(llmClient),
      clock: this.clock,
      tracer: this.tracer,
    });
    this.turnPhaseCoordinator = new TurnPhaseCoordinator({
      config: options.config,
      embeddingClient: options.embeddingClient,
      episodicRepository: options.episodicRepository,
      semanticNodeRepository: options.semanticNodeRepository,
      workingMemoryStore: options.workingMemoryStore,
      entityRepository: options.entityRepository,
      socialRepository: options.socialRepository,
      relationalSlotRepository: options.relationalSlotRepository,
      actionRepository: options.actionRepository,
      commitmentRepository: options.commitmentRepository,
      creatorDirectiveRepository: options.creatorDirectiveRepository,
      sharedStateRepository: options.sharedStateRepository,
      activityRepository: options.activityRepository,
      selfDecisionRepository: options.selfDecisionRepository,
      observedEventRepository: options.observedEventRepository,
      goalsRepository: options.goalsRepository,
      openQuestionsRepository: options.openQuestionsRepository,
      toolDispatcher: options.toolDispatcher,
      createStreamReader,
      entryIndex: options.entryIndex,
      attachmentRepository: options.attachmentRepository,
      imagePerceptionRepository: options.imagePerceptionRepository,
      attachmentService: options.attachmentService,
      imagePerceptionService: options.imagePerceptionService,
      streamIngestionCoordinator: options.streamIngestionCoordinator,
      chatResponseWatermarkCoordinator: options.chatResponseWatermarkCoordinator,
      outboundDelivery: options.outboundDelivery,
      autonomousOutboundPolicy: options.autonomousOutboundPolicy,
      outboundSourceTypes: options.outboundSourceTypes,
      llmFactory: () => options.llmFactory(),
      perceptionGateway,
      turnOpeningPersistence,
      attributionLifecycleService,
      correctivePreferenceTurnService,
      creatorDirectiveTurnService,
      turnActionStateService,
      turnGoalPromotionService,
      selfContextBuilder: this.selfContextBuilder,
      turnRetrievalCoordinator,
      discourseStateService,
      postGenerationGuardRunner,
      turnActionCoordinator,
      turnReflectionCoordinator,
      clock: this.clock,
      tracer: this.tracer,
      promptOverrideRepository: options.promptOverrideRepository,
      ...(options.sessionsRepository === undefined
        ? {}
        : { sessionsRepository: options.sessionsRepository }),
    });
  }

  loadWorkingMemory(sessionId: SessionId): WorkingMemory {
    return this.options.workingMemoryStore.load(sessionId);
  }

  clearWorkingMemory(sessionId: SessionId): void {
    this.options.workingMemoryStore.clear(sessionId);
  }

  private async buildSelfSnapshot(audienceEntityId: EntityId | null): Promise<SelfSnapshot> {
    return this.selfContextBuilder.buildSelfSnapshot(audienceEntityId);
  }

  private async appendFailureEvent(
    streamWriter: StreamWriter,
    error: unknown,
    sessionId: SessionId,
    turnId: string,
  ): Promise<void> {
    const message =
      error instanceof Error
        ? `${error.name}: ${error.message}`
        : `Unknown error: ${String(error)}`;

    try {
      await streamWriter.append({
        kind: "internal_event",
        turn_id: turnId,
        turn_status: "aborted",
        content: {
          event: ABORTED_TURN_EVENT,
          turn_id: turnId,
          session_id: sessionId,
          reason: message,
        },
      });
    } catch {
      // Best-effort logging only.
    }

    if (this.tracer.enabled) {
      this.tracer.emit("turn.rejected", {
        turnId,
        reason: message,
        session_id: sessionId,
      });
    }
  }

  private emitTerminalTurn(input: {
    turnId: string;
    sessionId: SessionId;
    outcome: TurnTerminalOutcome;
    startedWallMs: number;
  }): void {
    if (!this.tracer.enabled) {
      return;
    }

    this.tracer.emit("turn.terminal", {
      turnId: input.turnId,
      turn_id: input.turnId,
      session_id: input.sessionId,
      outcome: input.outcome,
      ts: this.clock.now(),
      duration_ms: Math.max(0, performance.now() - input.startedWallMs),
    });
  }

  async run(input: TurnOrchestratorInput): Promise<TurnResult> {
    const sessionId = input.sessionId ?? DEFAULT_SESSION_ID;
    this.options.attachmentService.validateAttachments(input.attachments ?? []);
    const effectiveLockMode =
      input.lockMode ?? (isAutonomousLikeTurnOrigin(input.origin) ? "try" : "block");
    const lease =
      effectiveLockMode === "try"
        ? await this.sessionLock.tryAcquire(sessionId)
        : effectiveLockMode === "block"
          ? await this.sessionLock.acquire(sessionId)
          : await this.sessionLock.acquire(sessionId, { timeoutMs: effectiveLockMode.timeoutMs });
    const phaseInput = stripTurnLockMode(input);

    if (lease === null) {
      throw new SessionBusyError(`Session ${sessionId} is busy`, {
        code: "SESSION_TURN_BUSY",
      });
    }

    const turnId = randomUUID();
    const globalTurnCounter =
      input.globalTurnCounter === undefined
        ? this.options.actionRepository.nextLifecycleTurnGlobal()
        : this.options.actionRepository.ensureLifecycleTurnGlobal(input.globalTurnCounter);
    const streamWriter = this.options.createStreamWriter(sessionId);
    const terminalStartedWallMs = performance.now();
    let terminalOutcome: TurnTerminalOutcome = "error";
    const lifecycleTracker = new TurnLifecycleTracker({
      workingMemoryStore: this.options.workingMemoryStore,
      actionRepository: this.options.actionRepository,
      executiveStepsRepository: this.options.executiveStepsRepository,
      goalsRepository: this.options.goalsRepository,
      openQuestionsRepository: this.options.openQuestionsRepository,
      episodicRepository: this.options.episodicRepository,
      relationalSlotRepository: this.options.relationalSlotRepository,
    });

    try {
      try {
        const result = await this.turnPhaseCoordinator.run({
          input: phaseInput,
          globalTurnCounter,
          sessionId,
          turnId,
          streamWriter,
          lifecycleTracker,
        });
        lifecycleTracker.commitTurnState();
        terminalOutcome =
          result.terminalOutcome ??
          (result.path === "suppressed" ? "suppressed_action" : "reflected");
        return result;
      } catch (error) {
        await lifecycleTracker.cleanupAbortedTurnState();
        await this.appendFailureEvent(streamWriter, error, sessionId, turnId);
        terminalOutcome = "aborted";
        throw error;
      } finally {
        this.emitTerminalTurn({
          turnId,
          sessionId,
          outcome: terminalOutcome,
          startedWallMs: terminalStartedWallMs,
        });
      }
    } finally {
      streamWriter.close();
      await lease.release();
    }
  }
}
