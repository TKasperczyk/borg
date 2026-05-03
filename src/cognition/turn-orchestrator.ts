import { randomUUID } from "node:crypto";

import type { Config } from "../config/index.js";
import { SuppressionSet } from "./attention/index.js";
import { AttributionLifecycleService } from "./attribution/lifecycle-service.js";
import {
  LLMPendingActionJudge,
  performAction,
  type PendingActionRejection,
  type ToolLoopCallRecord,
} from "./action/index.js";
import { formatAutonomyTriggerContext, type AutonomyTriggerContext } from "./autonomy-trigger.js";
import {
  CorrectivePreferenceExtractor,
  type CorrectivePreferenceCandidate,
} from "./commitments/corrective-preference-extractor.js";
import { CommitmentGuardRunner } from "./commitments/guard-runner.js";
import { Deliberator, type SelfSnapshot, type TurnStakes } from "./deliberation/deliberator.js";
import {
  GoalPromotionExtractor,
  type GoalPromotionCandidate,
} from "./goals/goal-promotion-extractor.js";
import { detectAffectiveSignal } from "./perception/affective-signal.js";
import { PerceptionGateway } from "./perception/gateway.js";
import { TurnOpeningPersistence } from "./persistence/turn-opening.js";
import { PendingProceduralAttemptTracker } from "./procedural/pending-attempt-tracker.js";
import { TurnContextCompiler } from "./recency/index.js";
import { computeExecutiveContextFits, selectExecutiveFocus } from "../executive/index.js";
import type { ExecutiveFocus, ExecutiveStepsRepository } from "../executive/index.js";
import type { StreamIngestionCoordinator } from "./ingestion/index.js";
import type { Reflector } from "./reflection/index.js";
import { TurnRetrievalCoordinator } from "./retrieval/turn-coordinator.js";
import type { RetrievalPipeline } from "../retrieval/index.js";
import {
  buildSelfScoringFeatureSet,
  selectActiveScoringValues,
  toRetrievalScoringFeatures,
  type SelfScoringFeatureSet,
} from "../retrieval/scoring-features.js";
import type { LLMClient } from "../llm/index.js";
import type { EmbeddingClient } from "../embeddings/index.js";
import { MoodRepository } from "../memory/affective/index.js";
import type { ActionRecord, ActionRepository } from "../memory/actions/index.js";
import {
  commitmentSchema,
  CommitmentRepository,
  EntityRepository,
  type CommitmentRecord,
} from "../memory/commitments/index.js";
import { SkillSelector } from "../memory/procedural/index.js";
import type { IdentityService } from "../memory/identity/index.js";
import {
  appendInternalFailureEvent,
  AutobiographicalRepository,
  GrowthMarkersRepository,
  GoalsRepository,
  TraitsRepository,
  ValuesRepository,
  type GoalRecord,
  type OpenQuestionsRepository,
} from "../memory/self/index.js";
import { ReviewQueueRepository } from "../memory/semantic/index.js";
import { SocialRepository } from "../memory/social/index.js";
import { WorkingMemoryStore, type WorkingMemory } from "../memory/working/index.js";
import { EpisodicRepository, isEpisodeVisibleToAudience } from "../memory/episodic/index.js";
import { StreamReader, StreamWriter } from "../stream/index.js";
import type { ToolDispatcher } from "../tools/index.js";
import { SessionBusyError } from "../util/errors.js";
import { SystemClock, type Clock } from "../util/clock.js";
import {
  DEFAULT_SESSION_ID,
  createCommitmentId,
  goalIdHelpers,
  type EntityId,
  type EpisodeId,
  type GoalId,
  type SessionId,
  type StreamEntryId,
} from "../util/ids.js";
import { NOOP_TRACER, toTraceJsonValue, type TurnTracer } from "./tracing/tracer.js";
import type { CognitiveMode, IntentRecord } from "./types.js";
import { SessionLock } from "./session-lock.js";
import type {
  AgentSuppressedStreamContent,
  PendingTurnEmission,
  TurnEmission,
} from "./generation/types.js";
import {
  clearStopUntilSubstantiveContent,
  setStopUntilSubstantiveContent,
} from "./generation/discourse-state.js";
import { GenerationGate } from "./generation/generation-gate.js";
import { StopCommitmentExtractor } from "./generation/self-stop-commitment.js";

function flattenGoals(goals: ReadonlyArray<GoalRecord & { children?: unknown }>): GoalRecord[] {
  const flattened: GoalRecord[] = [];
  const stack = [...goals];

  while (stack.length > 0) {
    const next = stack.shift();

    if (next === undefined) {
      continue;
    }

    flattened.push(next);

    if ("children" in next && Array.isArray(next.children)) {
      stack.push(...(next.children as GoalRecord[]));
    }
  }

  return flattened;
}

function getForcedExecutiveFocusGoalId(
  autonomyTrigger: AutonomyTriggerContext | null | undefined,
): GoalId | null {
  if (
    autonomyTrigger?.source_name !== "executive_focus_due" ||
    autonomyTrigger.payload.reason !== "step_due"
  ) {
    return null;
  }

  const candidate = autonomyTrigger.payload.force_executive_focus_goal_id;

  return typeof candidate === "string" && goalIdHelpers.is(candidate) ? candidate : null;
}

function applyForcedExecutiveFocus(
  focus: ExecutiveFocus,
  forcedGoalId: GoalId | null,
): ExecutiveFocus {
  if (forcedGoalId === null) {
    return focus;
  }

  const forcedScore = focus.candidates.find((candidate) => candidate.goal_id === forcedGoalId);

  if (forcedScore === undefined) {
    return focus;
  }

  return {
    ...focus,
    selected_goal: forcedScore.goal,
    selected_score: forcedScore,
  };
}

type ProvenanceScopedSelfRecord = {
  provenance?: {
    kind: string;
    episode_ids?: readonly EpisodeId[];
  } | null;
  evidence_episode_ids?: readonly EpisodeId[] | null;
  key_episode_ids?: readonly EpisodeId[] | null;
};

function getSelfRecordEvidenceEpisodeIds(record: ProvenanceScopedSelfRecord): EpisodeId[] {
  if (record.provenance?.kind !== "episodes") {
    return [];
  }

  const hasExplicitEvidence =
    record.evidence_episode_ids !== undefined || record.key_episode_ids !== undefined;
  const explicitEpisodeIds = [
    ...(record.evidence_episode_ids ?? []),
    ...(record.key_episode_ids ?? []),
  ];

  if (hasExplicitEvidence && explicitEpisodeIds.length === 0) {
    return [];
  }

  return [...new Set([...(record.provenance.episode_ids ?? []), ...explicitEpisodeIds])];
}

function isSelfRecordVisible(
  record: ProvenanceScopedSelfRecord,
  visibleEpisodeIds: ReadonlySet<EpisodeId>,
): boolean {
  const episodeIds = getSelfRecordEvidenceEpisodeIds(record);

  if (episodeIds.length === 0) {
    return true;
  }

  return episodeIds.some((episodeId) => visibleEpisodeIds.has(episodeId));
}

function buildCorrectivePreferenceCommitment(input: {
  candidate: CorrectivePreferenceCandidate;
  audienceEntityId: EntityId | null;
  sourceStreamEntryIds?: CommitmentRecord["source_stream_entry_ids"];
  nowMs: number;
}): CommitmentRecord {
  return commitmentSchema.parse({
    id: createCommitmentId(),
    type: input.candidate.type,
    directive: input.candidate.directive,
    priority: input.candidate.priority,
    made_to_entity: null,
    restricted_audience: input.audienceEntityId,
    about_entity: null,
    provenance: {
      kind: "online",
      process: "corrective-preference-extractor",
    },
    ...(input.sourceStreamEntryIds === undefined || input.sourceStreamEntryIds.length === 0
      ? {}
      : { source_stream_entry_ids: input.sourceStreamEntryIds }),
    created_at: input.nowMs,
    expires_at: null,
    expired_at: null,
    revoked_at: null,
    revoked_reason: null,
    revoke_provenance: null,
    superseded_by: null,
  });
}

function appendCommitmentIfMissing(
  commitments: readonly CommitmentRecord[],
  commitment: CommitmentRecord | null,
): CommitmentRecord[] {
  if (commitment === null) {
    return [...commitments];
  }

  if (commitments.some((existing) => existing.id === commitment.id)) {
    return [...commitments];
  }

  return [...commitments, commitment].sort(
    (left, right) => right.priority - left.priority || left.created_at - right.created_at,
  );
}

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
  private readonly perceptionGateway: PerceptionGateway;
  private readonly turnOpeningPersistence: TurnOpeningPersistence;
  private readonly attributionLifecycleService: AttributionLifecycleService;
  private readonly turnRetrievalCoordinator: TurnRetrievalCoordinator;
  private readonly commitmentGuardRunner: CommitmentGuardRunner;
  private readonly pendingProceduralAttemptTracker: PendingProceduralAttemptTracker;

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
      ((sessionId) =>
        new StreamReader({
          dataDir: options.config.dataDir,
          sessionId,
        }));
    this.perceptionGateway = new PerceptionGateway({
      config: options.config,
      llmFactory: options.llmFactory,
      clock: this.clock,
      tracer: this.tracer,
      getAffectiveSignalDetector: () => options.affectiveSignalDetector,
      turnContextCompiler,
      createStreamReader,
    });
    this.turnOpeningPersistence = new TurnOpeningPersistence({
      workingMemoryStore: options.workingMemoryStore,
    });
    this.attributionLifecycleService = new AttributionLifecycleService({
      socialRepository: options.socialRepository,
      traitsRepository: options.traitsRepository,
      episodicRepository: options.episodicRepository,
      clock: this.clock,
    });
    this.turnRetrievalCoordinator = new TurnRetrievalCoordinator({
      commitmentRepository: options.commitmentRepository,
      reviewQueueRepository: options.reviewQueueRepository,
      moodRepository: options.moodRepository,
      retrievalPipeline: options.retrievalPipeline,
      skillSelector: options.skillSelector,
      clock: this.clock,
      tracer: this.tracer,
    });
    this.commitmentGuardRunner = new CommitmentGuardRunner({
      detectionModel: options.config.anthropic.models.background,
      rewriteModel: options.config.anthropic.models.cognition,
      entityRepository: options.entityRepository,
      tracer: this.tracer,
    });
    this.pendingProceduralAttemptTracker = new PendingProceduralAttemptTracker();
  }

  loadWorkingMemory(sessionId: SessionId): WorkingMemory {
    return this.options.workingMemoryStore.load(sessionId);
  }

  clearWorkingMemory(sessionId: SessionId): void {
    this.options.workingMemoryStore.clear(sessionId);
  }

  private async buildSelfSnapshot(audienceEntityId: EntityId | null): Promise<SelfSnapshot> {
    const values = this.options.valuesRepository.list();
    const goals = flattenGoals(
      this.options.goalsRepository.list({
        status: "active",
        visibleToAudienceEntityId: audienceEntityId,
      }),
    );
    const traits = this.options.traitsRepository.list();
    const currentPeriod = this.options.autobiographicalRepository?.currentPeriod() ?? null;
    const recentGrowthMarkers = this.options.growthMarkersRepository?.list({ limit: 3 }) ?? [];
    const visibleRecords = await this.filterSelfRecordsVisibleToAudience(
      [
        ...values,
        ...goals,
        ...traits,
        ...(currentPeriod === null ? [] : [currentPeriod]),
        ...recentGrowthMarkers,
      ],
      audienceEntityId,
    );
    const visibleIds = new Set(visibleRecords.map((record) => record.id));

    return {
      values: values.filter((value) => visibleIds.has(value.id)),
      goals: goals.filter((goal) => visibleIds.has(goal.id)),
      traits: traits.filter((trait) => visibleIds.has(trait.id)),
      currentPeriod:
        currentPeriod === null || visibleIds.has(currentPeriod.id) ? currentPeriod : null,
      recentGrowthMarkers: recentGrowthMarkers.filter((marker) => visibleIds.has(marker.id)),
    };
  }

  private async filterSelfRecordsVisibleToAudience<T extends ProvenanceScopedSelfRecord>(
    records: readonly T[],
    audienceEntityId: EntityId | null,
  ): Promise<T[]> {
    const evidenceEpisodeIds = [
      ...new Set(records.flatMap((record) => getSelfRecordEvidenceEpisodeIds(record))),
    ];
    const evidenceEpisodes = await this.options.episodicRepository.getMany(evidenceEpisodeIds);
    const visibleEpisodeIds = new Set(
      evidenceEpisodes
        .filter((episode) => isEpisodeVisibleToAudience(episode, audienceEntityId))
        .map((episode) => episode.id),
    );

    return records.filter((record) => isSelfRecordVisible(record, visibleEpisodeIds));
  }

  private async listActiveGoalsVisibleToAudience(
    audienceEntityId: EntityId | null,
  ): Promise<GoalRecord[]> {
    const goals = flattenGoals(
      this.options.goalsRepository.list({
        status: "active",
        visibleToAudienceEntityId: audienceEntityId,
      }),
    );

    return this.filterSelfRecordsVisibleToAudience(goals, audienceEntityId);
  }

  private async appendFailureEvent(
    streamWriter: StreamWriter,
    error: unknown,
    sessionId: SessionId,
  ): Promise<void> {
    const message =
      error instanceof Error
        ? `${error.name}: ${error.message}`
        : `Unknown error: ${String(error)}`;

    try {
      await streamWriter.append({
        kind: "internal_event",
        content: `Turn failed for ${sessionId}: ${message}`,
      });
    } catch {
      // Best-effort logging only.
    }
  }

  private async appendHookFailureEvent(
    streamWriter: StreamWriter,
    hook: string,
    error: unknown,
    details?: Record<string, unknown>,
  ): Promise<void> {
    await appendInternalFailureEvent(streamWriter, hook, error, details);
  }

  private emitGoalPromotionDegraded(input: {
    turnId: string;
    reason: string;
    error?: unknown;
    details?: Record<string, unknown>;
  }): void {
    if (!this.tracer.enabled) {
      return;
    }

    this.tracer.emit("goal_promotion_extractor_degraded", {
      turnId: input.turnId,
      reason: input.reason,
      ...(input.details ?? {}),
      ...(this.tracer.includePayloads && input.error !== undefined
        ? { error: input.error instanceof Error ? input.error.message : String(input.error) }
        : {}),
    });
  }

  private async persistGoalPromotions(input: {
    candidates: readonly GoalPromotionCandidate[];
    audienceEntityId: EntityId | null;
    persistedUserEntryId?: StreamEntryId;
    streamWriter: StreamWriter;
    turnId: string;
  }): Promise<void> {
    const provenance = {
      kind: "online" as const,
      process: "goal-promotion-extractor",
    };
    const sourceStreamEntryIds =
      input.persistedUserEntryId === undefined ? undefined : [input.persistedUserEntryId];

    for (const candidate of input.candidates) {
      let goal: GoalRecord;

      try {
        goal = this.options.identityService.addGoal({
          description: candidate.description,
          priority: candidate.priority,
          status: "active",
          targetAt: candidate.target_at,
          audienceEntityId: input.audienceEntityId,
          provenance,
          sourceStreamEntryIds,
        });
      } catch (error) {
        this.emitGoalPromotionDegraded({
          turnId: input.turnId,
          reason: "goal_persist_failed",
          error,
          details: {
            description: candidate.description,
          },
        });
        await this.appendHookFailureEvent(
          input.streamWriter,
          "goal_promotion_goal_persist",
          error,
          {
            description: candidate.description,
          },
        );
        continue;
      }

      if (candidate.initial_step === null) {
        continue;
      }

      try {
        this.options.executiveStepsRepository.add({
          goalId: goal.id,
          description: candidate.initial_step.description,
          kind: candidate.initial_step.kind,
          dueAt: candidate.initial_step.due_at,
          provenance,
        });
      } catch (error) {
        this.emitGoalPromotionDegraded({
          turnId: input.turnId,
          reason: "initial_step_persist_failed",
          error,
          details: {
            goalId: goal.id,
          },
        });
        await this.appendHookFailureEvent(
          input.streamWriter,
          "goal_promotion_initial_step_persist",
          error,
          {
            goalId: goal.id,
          },
        );
      }
    }
  }

  private async appendSuppressionMarker(input: {
    streamWriter: StreamWriter;
    reason: Extract<PendingTurnEmission, { kind: "suppressed" }>["reason"];
    userEntryId?: AgentSuppressedStreamContent["user_entry_id"];
    turnId: string;
    audience?: string;
  }) {
    return input.streamWriter.append({
      kind: "agent_suppressed",
      content: {
        reason: input.reason,
        user_entry_id: input.userEntryId,
        turn_id: input.turnId,
      } satisfies AgentSuppressedStreamContent,
      ...(input.audience === undefined ? {} : { audience: input.audience }),
    });
  }

  private setDiscourseStopState(input: {
    workingMemory: WorkingMemory;
    provenance: Parameters<typeof setStopUntilSubstantiveContent>[1]["provenance"];
    sourceStreamEntryId?: Parameters<
      typeof setStopUntilSubstantiveContent
    >[1]["sourceStreamEntryId"];
    reason: string;
    turnId: string;
  }): WorkingMemory {
    const next = setStopUntilSubstantiveContent(input.workingMemory, {
      provenance: input.provenance,
      sourceStreamEntryId: input.sourceStreamEntryId,
      reason: input.reason,
      sinceTurn: input.workingMemory.turn_counter,
    });

    if (this.tracer.enabled) {
      this.tracer.emit("discourse_state_set", {
        turnId: input.turnId,
        state: "stop_until_substantive_content",
        provenance: input.provenance,
        reason: input.reason,
        ...(input.sourceStreamEntryId === undefined
          ? {}
          : { sourceStreamEntryId: input.sourceStreamEntryId }),
      });
    }

    return next;
  }

  private clearDiscourseStopState(input: {
    workingMemory: WorkingMemory;
    reason: string;
    turnId: string;
  }): WorkingMemory {
    const active = input.workingMemory.discourse_state?.stop_until_substantive_content ?? null;
    const next = clearStopUntilSubstantiveContent(input.workingMemory);

    if (active !== null && this.tracer.enabled) {
      this.tracer.emit("discourse_state_cleared", {
        turnId: input.turnId,
        state: "stop_until_substantive_content",
        provenance: active.provenance,
        reason: input.reason,
      });
    }

    return next;
  }

  private async appendDiscourseHardCapEvent(input: {
    streamWriter: StreamWriter;
    turnId: string;
    activeTurns: number;
    hardCapTurns: number;
    stateReason: string;
  }): Promise<void> {
    if (this.tracer.enabled) {
      this.tracer.emit("discourse_state_hard_cap", {
        turnId: input.turnId,
        state: "stop_until_substantive_content",
        activeTurns: input.activeTurns,
        hardCapTurns: input.hardCapTurns,
      });
    }

    try {
      await input.streamWriter.append({
        kind: "internal_event",
        content: {
          hook: "discourse_state_hard_cap",
          turn_id: input.turnId,
          active_turns: input.activeTurns,
          hard_cap_turns: input.hardCapTurns,
          state_reason: input.stateReason,
        },
      });
    } catch {
      // Best-effort telemetry only.
    }
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

  private listRecentCompletedActions(audienceEntityId: EntityId | null): ActionRecord[] {
    const visibleActions =
      audienceEntityId === null
        ? this.options.actionRepository.list({
            state: "completed",
            audienceEntityId: null,
            limit: 8,
          })
        : [
            ...this.options.actionRepository.list({
              state: "completed",
              audienceEntityId: null,
              limit: 8,
            }),
            ...this.options.actionRepository.list({
              state: "completed",
              audienceEntityId,
              limit: 8,
            }),
          ];

    return visibleActions
      .sort((left, right) => right.updated_at - left.updated_at || left.id.localeCompare(right.id))
      .slice(0, 8);
  }

  async run(input: TurnInput): Promise<TurnResult> {
    const sessionId = input.sessionId ?? DEFAULT_SESSION_ID;
    const isSelfAudience = input.audience === "self";
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

    try {
      try {
        await this.catchUpStreamIngestion(sessionId, streamWriter);
        let workingMemory = this.options.workingMemoryStore.load(sessionId);
        const turnPerception = this.perceptionGateway.beginTurn({
          turnId,
          onHookFailure: (hook, error, details) =>
            this.appendHookFailureEvent(streamWriter, hook, error, details),
        });
        const llmClient = this.options.llmFactory();
        const isUserTurn = input.origin !== "autonomous";
        const cognitionInput =
          input.autonomyTrigger === null || input.autonomyTrigger === undefined
            ? input.userMessage
            : formatAutonomyTriggerContext(input.autonomyTrigger);
        const audienceEntityId =
          input.audience === undefined || isSelfAudience
            ? null
            : this.options.entityRepository.resolve(input.audience);
        const audienceEntity =
          audienceEntityId === null ? null : this.options.entityRepository.get(audienceEntityId);
        let audienceProfile =
          audienceEntityId === null
            ? null
            : this.options.socialRepository.getProfile(audienceEntityId);
        const perceptionResult = await turnPerception.perceive({
          sessionId,
          isSelfAudience,
          origin: input.origin,
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
        const attributionResult = await this.attributionLifecycleService.settle({
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

        const openingPersistence = await this.turnOpeningPersistence.persist({
          streamWriter,
          userMessage: input.userMessage,
          persistUserMessage: isUserTurn,
          audience: input.audience,
          workingMemory,
          pendingSocialAttribution,
          pendingTraitAttribution,
          suppressionSet,
          perception,
          now: () => this.clock.now(),
        });
        const persistedUserEntry = openingPersistence.persistedUserEntry;
        const persistedUserEntryId = persistedUserEntry?.id;
        const persistedPerceptionEntry = openingPersistence.persistedPerceptionEntry;
        workingMemory = openingPersistence.workingMemory;

        let correctiveCommitment: CommitmentRecord | null = null;
        const activeCommitmentsForExtractor = this.options.commitmentRepository.getApplicable({
          audience: audienceEntityId,
          nowMs: this.clock.now(),
        });
        const correctivePreferenceExtractor = new CorrectivePreferenceExtractor({
          llmClient,
          model: this.options.config.anthropic.models.recallExpansion,
          tracer: this.tracer,
          turnId,
          onDegraded: (reason, error) => {
            if (!this.tracer.enabled) {
              return;
            }

            this.tracer.emit("commitment_extractor_degraded", {
              turnId,
              reason,
              ...(this.tracer.includePayloads && error !== undefined
                ? { error: error instanceof Error ? error.message : String(error) }
                : {}),
            });
          },
        });
        const correctiveCandidate = await correctivePreferenceExtractor.extract({
          userMessage: input.userMessage,
          recentHistory: recencyWindow.messages,
          audienceEntityId,
          activeCommitments: activeCommitmentsForExtractor.map((commitment) => ({
            id: commitment.id,
            type: commitment.type,
            directive: commitment.directive,
            priority: commitment.priority,
          })),
        });

        if (correctiveCandidate !== null) {
          const inMemoryCommitment = buildCorrectivePreferenceCommitment({
            candidate: correctiveCandidate,
            audienceEntityId,
            sourceStreamEntryIds:
              persistedUserEntryId === undefined ? undefined : [persistedUserEntryId],
            nowMs: this.clock.now(),
          });

          correctiveCommitment = inMemoryCommitment;

          try {
            correctiveCommitment = this.options.identityService.addCommitment({
              id: inMemoryCommitment.id,
              type: inMemoryCommitment.type,
              directive: inMemoryCommitment.directive,
              priority: inMemoryCommitment.priority,
              madeToEntity: inMemoryCommitment.made_to_entity,
              restrictedAudience: inMemoryCommitment.restricted_audience,
              aboutEntity: inMemoryCommitment.about_entity,
              provenance: inMemoryCommitment.provenance,
              sourceStreamEntryIds: inMemoryCommitment.source_stream_entry_ids,
              createdAt: inMemoryCommitment.created_at,
              expiresAt: inMemoryCommitment.expires_at,
            });
          } catch (error) {
            await this.appendHookFailureEvent(
              streamWriter,
              "corrective_preference_commitment_persist",
              error,
              {
                commitmentId: inMemoryCommitment.id,
              },
            );
          }
        }

        if (isUserTurn) {
          const activeGoalsForPromotion =
            await this.listActiveGoalsVisibleToAudience(audienceEntityId);
          const goalPromotionExtractor = new GoalPromotionExtractor({
            llmClient,
            model: this.options.config.anthropic.models.recallExpansion,
            tracer: this.tracer,
            turnId,
            onDegraded: (reason, error) => {
              this.emitGoalPromotionDegraded({
                turnId,
                reason,
                error,
              });
            },
          });
          const goalPromotionCandidates = await goalPromotionExtractor.extract({
            userMessage: input.userMessage,
            recentHistory: recencyWindow.messages,
            audienceEntityId,
            temporalCue: perception.temporalCue,
            activeGoals: activeGoalsForPromotion.map((goal) => ({
              id: goal.id,
              description: goal.description,
              priority: goal.priority,
              target_at: goal.target_at,
            })),
          });

          if (goalPromotionCandidates.length > 0) {
            await this.persistGoalPromotions({
              candidates: goalPromotionCandidates,
              audienceEntityId,
              persistedUserEntryId,
              streamWriter,
              turnId,
            });
          }
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
          userMessage: input.userMessage,
          workingMemory,
          recencyMessages: recencyWindow.messages,
        });

        if (gateResult.signals.hardCapDue) {
          await this.appendDiscourseHardCapEvent({
            streamWriter,
            turnId,
            activeTurns: gateResult.signals.hardCapActiveTurns,
            hardCapTurns: this.options.config.generation.discourseStateHardCapTurns,
            stateReason:
              workingMemory.discourse_state?.stop_until_substantive_content?.reason ?? "unknown",
          });
        }

        if (gateResult.clearDiscourseStop) {
          workingMemory = this.clearDiscourseStopState({
            workingMemory,
            reason: gateResult.explanation,
            turnId,
          });
        }

        if (gateResult.action === "suppress") {
          const suppressionReason = gateResult.reason ?? "generation_gate";
          const activeStop = workingMemory.discourse_state?.stop_until_substantive_content ?? null;

          if (activeStop === null) {
            workingMemory = this.setDiscourseStopState({
              workingMemory,
              provenance: "generation_gate",
              sourceStreamEntryId: persistedUserEntryId,
              reason: gateResult.explanation,
              turnId,
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
              updated_at: this.clock.now(),
            },
          });
          const suppressionMarker = await this.appendSuppressionMarker({
            streamWriter,
            reason: suppressionReason,
            userEntryId: persistedUserEntryId,
            turnId,
            audience: input.audience,
          });
          const suppressionEmission: TurnEmission = {
            kind: "suppressed",
            reason: suppressionReason,
            markerEntryId: suppressionMarker.id,
          };

          if (this.tracer.enabled) {
            this.tracer.emit("generation_suppressed", {
              turnId,
              reason: suppressionReason,
              streamEntryId: suppressionMarker.id,
              source: "generation_gate",
              classified: gateResult.classified,
            });
          }

          this.options.workingMemoryStore.save({
            ...suppressionActionResult.workingMemory,
            updated_at: this.clock.now(),
          });

          return {
            mode: perception.mode,
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

        const selfSnapshot = await this.buildSelfSnapshot(audienceEntityId);
        const executiveContextText = [
          cognitionInput,
          ...perception.entities,
          input.autonomyTrigger === null || input.autonomyTrigger === undefined
            ? ""
            : JSON.stringify(input.autonomyTrigger.payload),
        ]
          .join(" ")
          .trim();
        const activeScoringValues = selectActiveScoringValues(selfSnapshot.values);
        let selfScoringFeatures: SelfScoringFeatureSet = {
          goalVectors: [],
          valueVectors: [],
        };
        let contextFitByGoalId: Awaited<ReturnType<typeof computeExecutiveContextFits>> = new Map();

        try {
          selfScoringFeatures = await buildSelfScoringFeatureSet({
            embeddingClient: this.options.embeddingClient,
            goals: selfSnapshot.goals,
            activeValues: activeScoringValues,
          });
        } catch (error) {
          if (this.tracer.enabled) {
            this.tracer.emit("retrieval_degraded", {
              turnId,
              subsystem: "scoring_features",
              reason: error instanceof Error ? error.message : String(error),
            });
          }
        }

        try {
          contextFitByGoalId = await computeExecutiveContextFits({
            embeddingClient: this.options.embeddingClient,
            goalVectors: selfScoringFeatures.goalVectors,
            contextText: executiveContextText,
          });
        } catch (error) {
          if (this.tracer.enabled) {
            this.tracer.emit("retrieval_degraded", {
              turnId,
              subsystem: "executive_context_fit",
              reason: error instanceof Error ? error.message : String(error),
            });
          }
        }

        const executiveFocus = applyForcedExecutiveFocus(
          selectExecutiveFocus({
            goals: selfSnapshot.goals,
            cognitionInput,
            perceptionEntities: perception.entities,
            autonomyPayload: input.autonomyTrigger?.payload ?? null,
            nowMs: this.clock.now(),
            threshold: this.options.config.executive.goalFocusThreshold,
            deadlineLookaheadMs: this.options.config.autonomy.triggers.goalFollowupDue.lookaheadMs,
            staleMs: this.options.config.autonomy.triggers.goalFollowupDue.staleMs,
            contextFitByGoalId,
          }),
          getForcedExecutiveFocusGoalId(input.autonomyTrigger),
        );
        const retrievalScoringFeatures = toRetrievalScoringFeatures({
          selfFeatures: selfScoringFeatures,
          primaryGoalId: executiveFocus.selected_goal?.id ?? null,
        });
        const executiveFocusWithStep =
          executiveFocus.selected_goal === null
            ? executiveFocus
            : {
                ...executiveFocus,
                next_step: this.options.executiveStepsRepository.topOpen(
                  executiveFocus.selected_goal.id,
                ),
              };

        const retrievalContext = await this.turnRetrievalCoordinator.coordinate({
          sessionId,
          turnId,
          userMessage: input.userMessage,
          recentMessages: recencyWindow.messages.map((message) => ({
            role: message.role,
            content: message.content,
          })),
          cognitionInput,
          inputAudience: input.audience,
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
        const deliberator = new Deliberator({
          llmClient,
          toolDispatcher: this.options.toolDispatcher,
          cognitionModel: this.options.config.anthropic.models.cognition,
          backgroundModel: this.options.config.anthropic.models.background,
          clock: this.clock,
          tracer: this.tracer,
        });
        const deliberation = await deliberator.run(
          {
            sessionId,
            turnId,
            audience: input.audience,
            audienceEntityId,
            userMessage: input.userMessage,
            userEntryId: persistedUserEntryId,
            autonomyTrigger: input.autonomyTrigger ?? null,
            perception,
            retrievalResult: retrievedEpisodes,
            retrievedSemantic,
            retrievedEvidence: retrieval.evidence,
            contradictionPresent: retrieval.contradiction_present,
            retrievalConfidence: retrieval.confidence,
            applicableCommitments,
            openQuestionsContext: retrieval.open_questions,
            pendingCorrectionsContext: pendingCorrections,
            selectedSkill,
            entityRepository: this.options.entityRepository,
            workingMemory,
            recentCompletedActions: this.listRecentCompletedActions(audienceEntityId),
            affectiveTrajectory,
            selfSnapshot,
            executiveFocus: executiveFocusWithStep,
            audienceProfile,
            recencyMessages: recencyWindow.messages,
            options: {
              stakes: input.stakes,
            },
            reRetrieve: retrievalContext.reRetrieve,
          },
          streamWriter,
        );

        if (deliberation.emissionRecommendation === "no_output") {
          workingMemory = this.setDiscourseStopState({
            workingMemory,
            provenance: "s2_planner_no_output",
            sourceStreamEntryId: deliberation.thoughtStreamEntryIds?.[0],
            reason: "S2 planner recommended no assistant message for this turn.",
            turnId,
          });
        }

        workingMemory = {
          ...workingMemory,
          updated_at: this.clock.now(),
        };
        const deliberationEmission: PendingTurnEmission =
          deliberation.emissionRecommendation === "no_output"
            ? {
                kind: "suppressed",
                reason: "s2_planner_no_output",
              }
            : (deliberation.emission ?? {
                kind: "message",
                content: deliberation.response,
              });
        const pendingActionJudge = new LLMPendingActionJudge({
          llmClient,
          model: this.options.config.anthropic.models.background,
        });
        const onPendingActionRejected = (event: PendingActionRejection) => {
          if (!this.tracer.enabled) {
            return;
          }

          this.tracer.emit("working_memory_degraded", {
            turnId,
            subsystem: "pending_actions",
            reason: event.reason,
            confidence: event.confidence,
            degraded: event.degraded,
            ...(this.tracer.includePayloads
              ? {
                  record: toTraceJsonValue(event.record),
                }
              : {}),
          });
        };
        const actionResult =
          deliberationEmission.kind === "suppressed"
            ? await performAction({
                response: "",
                emission: deliberationEmission,
                toolCalls: deliberation.tool_calls,
                intents: [],
                workingMemory,
              })
            : await (async () => {
                const commitmentCheck = await this.commitmentGuardRunner.run({
                  llmClient,
                  turnId,
                  response: deliberation.response,
                  userMessage: input.userMessage,
                  cognitionInput,
                  origin: input.origin,
                  autonomyTrigger: input.autonomyTrigger,
                  commitments: applicableCommitments,
                  relevantEntities: perception.entities,
                });

                const commitmentEmission = commitmentCheck.emission;
                return performAction({
                  response: commitmentEmission.kind === "message" ? commitmentEmission.content : "",
                  emission: commitmentEmission,
                  toolCalls: deliberation.tool_calls,
                  intents: deliberation.intents,
                  workingMemory,
                  pendingActionJudge,
                  onPendingActionRejected,
                });
              })();
        const actionEmission: PendingTurnEmission = actionResult.emission ?? {
          kind: "message",
          content: actionResult.response,
        };
        const persistedAgentEntry =
          actionEmission.kind === "message"
            ? await streamWriter.append({
                kind: "agent_msg",
                content: actionResult.response,
                tool_calls: actionResult.tool_calls,
                ...(input.audience === undefined ? {} : { audience: input.audience }),
              })
            : await this.appendSuppressionMarker({
                streamWriter,
                reason: actionEmission.reason,
                userEntryId: persistedUserEntryId,
                turnId,
                audience: input.audience,
              });

        if (actionEmission.kind === "suppressed") {
          const suppressionEmission: TurnEmission = {
            kind: "suppressed",
            reason: actionEmission.reason,
            markerEntryId: persistedAgentEntry.id,
          };
          let suppressedWorkingMemory = actionResult.workingMemory;

          if (actionEmission.reason === "no_output_tool") {
            suppressedWorkingMemory = this.setDiscourseStopState({
              workingMemory: suppressedWorkingMemory,
              provenance: "no_output_tool",
              sourceStreamEntryId: persistedAgentEntry.id,
              reason: "Finalizer called no_output for this turn.",
              turnId,
            });
          }

          if (
            actionEmission.reason === "commitment_revision_failed" ||
            actionEmission.reason === "rewrite_unsupported_or_empty"
          ) {
            suppressedWorkingMemory = this.setDiscourseStopState({
              workingMemory: suppressedWorkingMemory,
              provenance: "commitment_guard",
              sourceStreamEntryId: persistedAgentEntry.id,
              reason:
                actionEmission.reason === "commitment_revision_failed"
                  ? "Commitment guard suppressed this turn because revision still violated an active commitment."
                  : "Commitment guard suppressed this turn because rewrite produced no supported output.",
              turnId,
            });
          }

          if (this.tracer.enabled) {
            this.tracer.emit("generation_suppressed", {
              turnId,
              reason: actionEmission.reason,
              streamEntryId: persistedAgentEntry.id,
            });
          }

          this.options.workingMemoryStore.save({
            ...suppressedWorkingMemory,
            updated_at: this.clock.now(),
          });

          return {
            mode: perception.mode,
            path: "suppressed",
            response: "",
            emitted: false,
            emission: suppressionEmission,
            thoughts: deliberation.thoughts,
            usage: deliberation.usage,
            retrievedEpisodeIds: deliberation.retrievedEpisodes.map((result) => result.episode.id),
            referencedEpisodeIds: [...(deliberation.referencedEpisodeIds ?? [])],
            intents: [],
            toolCalls: [...actionResult.tool_calls],
          };
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
          userMessage: input.userMessage,
          agentResponse: actionResult.response,
        });

        if (stopCommitment !== null) {
          postActionWorkingMemory = this.setDiscourseStopState({
            workingMemory: postActionWorkingMemory,
            provenance: "self_commitment_extractor",
            sourceStreamEntryId: persistedAgentEntry.id,
            reason: stopCommitment.reason,
            turnId,
          });
        }
        const actionResultForReflection = {
          ...actionResult,
          workingMemory: postActionWorkingMemory,
        };
        let moodSnapshot = workingMood;

        if (input.origin !== "autonomous" && perception.affectiveSignalDegraded !== true) {
          try {
            const nextMood = this.options.moodRepository.update(sessionId, {
              valence: perception.affectiveSignal.valence,
              arousal: perception.affectiveSignal.arousal,
              reason: input.userMessage.slice(0, 120),
              provenance: {
                kind: "system",
              },
            });
            moodSnapshot = {
              valence: nextMood.valence,
              arousal: nextMood.arousal,
              dominant_emotion: perception.affectiveSignal.dominant_emotion,
            };
          } catch (error) {
            await this.appendHookFailureEvent(streamWriter, "mood_update", error);
          }
        }

        let interactionRecord: ReturnType<SocialRepository["recordInteractionWithId"]> | null =
          null;
        if (audienceEntityId !== null) {
          try {
            interactionRecord = this.options.socialRepository.recordInteractionWithId(
              audienceEntityId,
              {
                now: this.clock.now(),
                provenance: {
                  kind: "system",
                },
              },
            );
          } catch (error) {
            await this.appendHookFailureEvent(streamWriter, "social_update", error);
          }
        }
        const reflector = this.options.createReflector(llmClient);
        const activeOpenQuestions = this.options.openQuestionsRepository.list({
          status: "open",
          visibleToAudienceEntityId: audienceEntityId,
          limit: 20,
        });
        const reflectedWorkingMemory = await reflector.reflect(
          {
            turnId,
            origin: input.origin ?? "user",
            userMessage: input.userMessage,
            perception,
            workingMemory: {
              ...postActionWorkingMemory,
              mood: moodSnapshot,
            },
            selfSnapshot,
            deliberationResult: deliberation,
            actionResult: actionResultForReflection,
            retrievedEpisodes: deliberation.retrievedEpisodes,
            retrievalConfidence: retrieval.confidence,
            executiveFocus: executiveFocusWithStep,
            selectedSkillId: selectedSkill?.skill.id ?? null,
            audienceEntityId,
            activeOpenQuestions,
            suppressionSet,
            currentTurnStreamEntryIds:
              persistedUserEntryId === undefined
                ? [persistedPerceptionEntry.id, persistedAgentEntry.id]
                : [persistedUserEntryId, persistedAgentEntry.id],
          },
          streamWriter,
        );
        const nextPendingSocialAttribution =
          audienceEntityId !== null &&
          interactionRecord !== null &&
          pendingSocialAttribution === null
            ? {
                entity_id: audienceEntityId,
                interaction_id: interactionRecord.interaction_id,
                agent_response_summary:
                  actionResult.response.trim().length === 0
                    ? null
                    : actionResult.response.replace(/\s+/g, " ").trim().slice(0, 240),
                turn_completed_ts: persistedAgentEntry.timestamp,
              }
            : pendingSocialAttribution;
        const nextPendingProceduralAttempts = this.pendingProceduralAttemptTracker.update({
          isUserTurn,
          userMessage: input.userMessage,
          perception,
          actionResult: actionResultForReflection,
          selectedSkill,
          proceduralContext,
          reflectedWorkingMemory,
          persistedUserEntryId,
          persistedAgentEntryId: persistedAgentEntry.id,
          audienceEntityId,
        });

        if (this.tracer.enabled) {
          this.tracer.emit("reflection_emitted", {
            turnId,
            attributions: {
              pending_social: nextPendingSocialAttribution !== null,
              pending_trait: reflectedWorkingMemory.pending_trait_attribution !== null,
              pending_procedural: nextPendingProceduralAttempts.length > 0,
              pending_actions: reflectedWorkingMemory.pending_actions.length,
            },
          });
        }

        this.options.workingMemoryStore.save({
          ...reflectedWorkingMemory,
          mood: moodSnapshot,
          pending_social_attribution: nextPendingSocialAttribution,
          pending_procedural_attempts: nextPendingProceduralAttempts,
          suppressed: suppressionSet.snapshot(),
          updated_at: this.clock.now(),
        });

        // Live-extract just-finished stream entries on a best-effort basis
        // for next-turn freshness. If extraction fails, the stream +
        // watermark remain the durable retry queue; the next turn runs a
        // bounded pre-turn catch-up before retrieval.
        if (this.options.streamIngestionCoordinator !== undefined) {
          void this.options.streamIngestionCoordinator.ingest(sessionId).catch((error) => {
            console.error("Live stream ingestion failed", error);
          });
        }

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
      } catch (error) {
        await this.appendFailureEvent(streamWriter, error, sessionId);
        throw error;
      }
    } finally {
      streamWriter.close();
      await lease.release();
    }
  }
}
