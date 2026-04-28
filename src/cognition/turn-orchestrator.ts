import { randomUUID } from "node:crypto";

import type { Config } from "../config/index.js";
import { SuppressionSet } from "./attention/index.js";
import { AttributionLifecycleService } from "./attribution/lifecycle-service.js";
import { performAction, type ToolLoopCallRecord } from "./action/index.js";
import { formatAutonomyTriggerContext, type AutonomyTriggerContext } from "./autonomy-trigger.js";
import { CommitmentGuardRunner } from "./commitments/guard-runner.js";
import { Deliberator, type SelfSnapshot, type TurnStakes } from "./deliberation/deliberator.js";
import { detectAffectiveSignal } from "./perception/affective-signal.js";
import { PerceptionGateway } from "./perception/gateway.js";
import { TurnOpeningPersistence } from "./persistence/turn-opening.js";
import { PendingProceduralAttemptTracker } from "./procedural/pending-attempt-tracker.js";
import { TurnContextCompiler } from "./recency/index.js";
import { selectExecutiveFocus } from "../executive/index.js";
import type { ExecutiveFocus, ExecutiveStepsRepository } from "../executive/index.js";
import type { StreamIngestionCoordinator } from "./ingestion/index.js";
import type { Reflector } from "./reflection/index.js";
import { TurnRetrievalCoordinator } from "./retrieval/turn-coordinator.js";
import type { RetrievalPipeline } from "../retrieval/index.js";
import type { LLMClient } from "../llm/index.js";
import { MoodRepository } from "../memory/affective/index.js";
import { CommitmentRepository, EntityRepository } from "../memory/commitments/index.js";
import { SkillSelector } from "../memory/procedural/index.js";
import {
  appendInternalFailureEvent,
  AutobiographicalRepository,
  GrowthMarkersRepository,
  GoalsRepository,
  TraitsRepository,
  ValuesRepository,
  type GoalRecord,
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
  goalIdHelpers,
  type EntityId,
  type EpisodeId,
  type GoalId,
  type SessionId,
} from "../util/ids.js";
import { NOOP_TRACER, type TurnTracer } from "./tracing/tracer.js";
import type { CognitiveMode, IntentRecord } from "./types.js";
import { SessionLock } from "./session-lock.js";

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
  path: "system_1" | "system_2";
  response: string;
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
  episodicRepository: EpisodicRepository;
  valuesRepository: ValuesRepository;
  goalsRepository: GoalsRepository;
  traitsRepository: TraitsRepository;
  autobiographicalRepository?: AutobiographicalRepository;
  growthMarkersRepository?: GrowthMarkersRepository;
  executiveStepsRepository: ExecutiveStepsRepository;
  moodRepository: MoodRepository;
  socialRepository: SocialRepository;
  skillSelector: SkillSelector;
  entityRepository: EntityRepository;
  commitmentRepository: CommitmentRepository;
  reviewQueueRepository: ReviewQueueRepository;
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
   * If provided, fires live episodic extraction after each turn so the next
   * turn's retrieval has access to material from just-finished turns. If
   * omitted, the orchestrator skips live extraction entirely and relies on
   * explicit `borg.episodic.extract()` / `borg.dream.consolidate()` calls.
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
    const goals = flattenGoals(this.options.goalsRepository.list({ status: "active" }));
    const traits = this.options.traitsRepository.list();
    const currentPeriod = this.options.autobiographicalRepository?.currentPeriod() ?? null;
    const recentGrowthMarkers = this.options.growthMarkersRepository?.list({ limit: 3 }) ?? [];
    const scopedRecords: ProvenanceScopedSelfRecord[] = [
      ...values,
      ...goals,
      ...traits,
      ...(currentPeriod === null ? [] : [currentPeriod]),
      ...recentGrowthMarkers,
    ];
    const evidenceEpisodeIds = [
      ...new Set(scopedRecords.flatMap((record) => getSelfRecordEvidenceEpisodeIds(record))),
    ];
    const evidenceEpisodes = await this.options.episodicRepository.getMany(evidenceEpisodeIds);
    const visibleEpisodeIds = new Set(
      evidenceEpisodes
        .filter((episode) => isEpisodeVisibleToAudience(episode, audienceEntityId))
        .map((episode) => episode.id),
    );
    const filterVisible = <T extends ProvenanceScopedSelfRecord>(record: T): boolean =>
      isSelfRecordVisible(record, visibleEpisodeIds);

    return {
      values: values.filter(filterVisible),
      goals: goals.filter(filterVisible),
      traits: traits.filter(filterVisible),
      currentPeriod: currentPeriod === null || filterVisible(currentPeriod) ? currentPeriod : null,
      recentGrowthMarkers: recentGrowthMarkers.filter(filterVisible),
    };
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
        const selfSnapshot = await this.buildSelfSnapshot(audienceEntityId);
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
          }),
          getForcedExecutiveFocusGoalId(input.autonomyTrigger),
        );
        const executiveFocusWithStep =
          executiveFocus.selected_goal === null
            ? executiveFocus
            : {
                ...executiveFocus,
                next_step: this.options.executiveStepsRepository.topOpen(
                  executiveFocus.selected_goal.id,
                ),
              };

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
          audience: input.audience,
          workingMemory,
          pendingSocialAttribution,
          pendingTraitAttribution,
          suppressionSet,
          perception,
          now: () => this.clock.now(),
        });
        const persistedUserEntry = openingPersistence.persistedUserEntry;
        workingMemory = openingPersistence.workingMemory;
        const retrievalContext = await this.turnRetrievalCoordinator.coordinate({
          sessionId,
          turnId,
          userMessage: input.userMessage,
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
          suppressionSet,
          findEntityByName: (name) => this.options.entityRepository.findByName(name),
        });
        const applicableCommitments = retrievalContext.applicableCommitments;
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
            userEntryId: persistedUserEntry.id,
            autonomyTrigger: input.autonomyTrigger ?? null,
            perception,
            retrievalResult: retrievedEpisodes,
            retrievedSemantic,
            contradictionPresent: retrieval.contradiction_present,
            retrievalConfidence: retrieval.confidence,
            applicableCommitments,
            openQuestionsContext: retrieval.open_questions,
            pendingCorrectionsContext: pendingCorrections,
            selectedSkill,
            entityRepository: this.options.entityRepository,
            workingMemory,
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

        workingMemory = {
          ...workingMemory,
          updated_at: this.clock.now(),
        };
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
          streamWriter,
        });

        const actionResult = await performAction({
          response: commitmentCheck.final_response,
          toolCalls: deliberation.tool_calls,
          intents: deliberation.intents,
          audience: input.audience,
          perception,
          workingMemory,
        });
        const agentEntry = {
          kind: "agent_msg",
          content: actionResult.response,
          tool_calls: actionResult.tool_calls,
          ...(input.audience === undefined ? {} : { audience: input.audience }),
        } satisfies Parameters<StreamWriter["append"]>[0];

        const persistedAgentEntry = await streamWriter.append(agentEntry);
        let moodSnapshot = workingMood;

        if (input.origin !== "autonomous") {
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
        const reflectedWorkingMemory = await reflector.reflect(
          {
            origin: input.origin ?? "user",
            userMessage: input.userMessage,
            perception,
            workingMemory: {
              ...workingMemory,
              mood: moodSnapshot,
            },
            selfSnapshot,
            deliberationResult: deliberation,
            actionResult,
            retrievedEpisodes: deliberation.retrievedEpisodes,
            retrievalConfidence: retrieval.confidence,
            executiveFocus: executiveFocusWithStep,
            selectedSkillId: selectedSkill?.skill.id ?? null,
            audienceEntityId,
            suppressionSet,
            currentTurnStreamEntryIds: [persistedUserEntry.id, persistedAgentEntry.id],
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
          actionResult,
          selectedSkill,
          proceduralContext,
          reflectedWorkingMemory,
          persistedUserEntryId: persistedUserEntry.id,
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
              pending_intents: reflectedWorkingMemory.pending_intents.length,
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

        // Live-extract just-finished stream entries so the next turn's
        // retrieval sees episodes from this turn. Fire-and-forget: the
        // coordinator dedups concurrent calls per session, watermark keeps
        // it idempotent, and its own onError hook logs failures via its
        // own stream writer (the turn's writer is about to close below).
        if (this.options.streamIngestionCoordinator !== undefined) {
          void this.options.streamIngestionCoordinator.ingest(sessionId).catch((error) => {
            console.error("Live stream ingestion failed", error);
          });
        }

        return {
          mode: perception.mode,
          path: deliberation.path,
          response: actionResult.response,
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
