import { randomUUID } from "node:crypto";

import type { Config } from "../config/index.js";
import type { ExecutiveStepsRepository } from "../executive/index.js";
import type { EmbeddingClient } from "../embeddings/index.js";
import type { LLMClient } from "../llm/index.js";
import { MoodRepository } from "../memory/affective/index.js";
import type { ActionRepository } from "../memory/actions/index.js";
import { CommitmentRepository, EntityRepository } from "../memory/commitments/index.js";
import { EpisodicRepository } from "../memory/episodic/index.js";
import type { IdentityService } from "../memory/identity/index.js";
import { SkillSelector } from "../memory/procedural/index.js";
import { RelationalSlotRepository } from "../memory/relational-slots/index.js";
import {
  AutobiographicalRepository,
  GoalsRepository,
  GrowthMarkersRepository,
  TraitsRepository,
  ValuesRepository,
  type OpenQuestionsRepository,
} from "../memory/self/index.js";
import { ReviewQueueRepository } from "../memory/semantic/index.js";
import { SocialRepository } from "../memory/social/index.js";
import { WorkingMemoryStore, type WorkingMemory } from "../memory/working/index.js";
import type { RetrievalPipeline } from "../retrieval/index.js";
import { ABORTED_TURN_EVENT, StreamReader, StreamWriter } from "../stream/index.js";
import type { ToolDispatcher } from "../tools/index.js";
import { SystemClock, type Clock } from "../util/clock.js";
import { SessionBusyError } from "../util/errors.js";
import { DEFAULT_SESSION_ID, type EntityId, type SessionId } from "../util/ids.js";
import type { ToolLoopCallRecord } from "./action/index.js";
import { TurnActionCoordinator } from "./action/turn-action-coordinator.js";
import { TurnActionStateService } from "./actions/turn-action-state-service.js";
import { AttributionLifecycleService } from "./attribution/lifecycle-service.js";
import type { AutonomyTriggerContext } from "./autonomy-trigger.js";
import { CommitmentGuardRunner } from "./commitments/guard-runner.js";
import { CorrectivePreferenceTurnService } from "./commitments/corrective-preference-service.js";
import type { SelfSnapshot, TurnStakes } from "./deliberation/deliberator.js";
import { TurnDiscourseStateService } from "./generation/turn-discourse-state.js";
import { TurnRelationalGuardRunner } from "./generation/turn-relational-guard.js";
import type { TurnEmission } from "./generation/types.js";
import { TurnGoalPromotionService } from "./goals/turn-goal-promotion-service.js";
import type { StreamIngestionCoordinator } from "./ingestion/index.js";
import { TurnLifecycleTracker } from "./lifecycle/turn-lifecycle-tracker.js";
import { TurnPhaseCoordinator } from "./lifecycle/turn-phase-coordinator.js";
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
import { NOOP_TRACER, type TurnTracer } from "./tracing/tracer.js";
import type { CognitiveMode, IntentRecord } from "./types.js";

export type TurnInput = {
  userMessage: string;
  audience?: string;
  stakes?: TurnStakes;
  sessionId?: SessionId;
  origin?: "user" | "autonomous";
  autonomyTrigger?: AutonomyTriggerContext | null;
};

export type TurnResult = {
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

export type TurnOrchestratorOptions = {
  config: Config;
  retrievalPipeline: RetrievalPipeline;
  embeddingClient: EmbeddingClient;
  episodicRepository: EpisodicRepository;
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
  identityService: IdentityService;
  reviewQueueRepository: ReviewQueueRepository;
  openQuestionsRepository: OpenQuestionsRepository;
  workingMemoryStore: WorkingMemoryStore;
  llmFactory: () => LLMClient;
  createReflector: (llmClient: LLMClient) => Reflector;
  toolDispatcher: ToolDispatcher;
  clock?: Clock;
  createStreamWriter: (sessionId: SessionId) => StreamWriter;
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
  affectiveSignalDetector?: typeof detectAffectiveSignal;
  sessionLock?: SessionLock;
  tracer?: TurnTracer;
};

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
    const turnActionStateService = new TurnActionStateService({
      model: options.config.anthropic.models.recallExpansion,
      actionRepository: options.actionRepository,
      clock: this.clock,
      tracer: this.tracer,
    });
    const turnGoalPromotionService = new TurnGoalPromotionService({
      model: options.config.anthropic.models.recallExpansion,
      identityService: options.identityService,
      executiveStepsRepository: options.executiveStepsRepository,
      clock: this.clock,
      tracer: this.tracer,
    });
    const discourseStateService = new TurnDiscourseStateService({
      tracer: this.tracer,
    });
    const relationalGuardRunner = new TurnRelationalGuardRunner({
      auditModel: options.config.anthropic.models.background,
      rewriteModel: options.config.anthropic.models.cognition,
      createStreamReader,
      actionRepository: options.actionRepository,
      commitmentRepository: options.commitmentRepository,
      relationalSlotRepository: options.relationalSlotRepository,
      clock: this.clock,
      tracer: this.tracer,
    });
    const turnActionCoordinator = new TurnActionCoordinator({
      commitmentGuardRunner,
      relationalGuardRunner,
      embeddingClient: options.embeddingClient,
      pendingActionJudgeModel: options.config.anthropic.models.background,
      clock: this.clock,
      tracer: this.tracer,
      workingMemoryStore: options.workingMemoryStore,
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
      workingMemoryStore: options.workingMemoryStore,
      entityRepository: options.entityRepository,
      socialRepository: options.socialRepository,
      relationalSlotRepository: options.relationalSlotRepository,
      toolDispatcher: options.toolDispatcher,
      streamIngestionCoordinator: options.streamIngestionCoordinator,
      llmFactory: () => options.llmFactory(),
      perceptionGateway,
      turnOpeningPersistence,
      attributionLifecycleService,
      correctivePreferenceTurnService,
      turnActionStateService,
      turnGoalPromotionService,
      selfContextBuilder: this.selfContextBuilder,
      turnRetrievalCoordinator,
      discourseStateService,
      relationalGuardRunner,
      turnActionCoordinator,
      turnReflectionCoordinator,
      clock: this.clock,
      tracer: this.tracer,
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
      this.tracer.emit("turn_aborted", {
        turnId,
        reason: message,
        sessionId,
      });
    }
  }

  async run(input: TurnInput): Promise<TurnResult> {
    const sessionId = input.sessionId ?? DEFAULT_SESSION_ID;
    const lease =
      input.origin === "autonomous"
        ? await this.sessionLock.tryAcquire(sessionId)
        : await this.sessionLock.acquire(sessionId);

    if (lease === null) {
      throw new SessionBusyError(`Session ${sessionId} is busy`, {
        code: "SESSION_TURN_BUSY",
      });
    }

    const turnId = randomUUID();
    const streamWriter = this.options.createStreamWriter(sessionId);
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
        return await this.turnPhaseCoordinator.run({
          input,
          sessionId,
          turnId,
          streamWriter,
          lifecycleTracker,
        });
      } catch (error) {
        await lifecycleTracker.cleanupAbortedTurnState();
        await this.appendFailureEvent(streamWriter, error, sessionId, turnId);
        throw error;
      }
    } finally {
      streamWriter.close();
      await lease.release();
    }
  }
}
