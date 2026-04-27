import { randomUUID } from "node:crypto";

import type { Config } from "../config/index.js";
import { SuppressionSet } from "./attention/index.js";
import { AttributionLifecycleService } from "./attribution/lifecycle-service.js";
import { performAction, type ToolLoopCallRecord } from "./action/index.js";
import { formatAutonomyTriggerContext, type AutonomyTriggerContext } from "./autonomy-trigger.js";
import { CommitmentGuardRunner } from "./commitments/guard-runner.js";
import { Deliberator, type SelfSnapshot, type TurnStakes } from "./deliberation/deliberator.js";
import { detectAffectiveSignal } from "./perception/affective-signal.js";
import { Perceiver } from "./perception/index.js";
import { PendingProceduralAttemptTracker } from "./procedural/pending-attempt-tracker.js";
import { TurnContextCompiler, type RecencyWindow } from "./recency/index.js";
import type { StreamIngestionCoordinator } from "./ingestion/index.js";
import { Reflector } from "./reflection/index.js";
import { TurnRetrievalCoordinator } from "./retrieval/turn-coordinator.js";
import type { RetrievalPipeline } from "../retrieval/index.js";
import type { LLMClient } from "../llm/index.js";
import { createNeutralAffectiveSignal, MoodRepository } from "../memory/affective/index.js";
import { CommitmentRepository, EntityRepository } from "../memory/commitments/index.js";
import {
  ProceduralEvidenceRepository,
  SkillRepository,
  SkillSelector,
} from "../memory/procedural/index.js";
import {
  appendInternalFailureEvent,
  AutobiographicalRepository,
  GrowthMarkersRepository,
  OpenQuestionsRepository,
  GoalsRepository,
  TraitsRepository,
  ValuesRepository,
  type GoalRecord,
} from "../memory/self/index.js";
import { ReviewQueueRepository } from "../memory/semantic/index.js";
import { SocialRepository } from "../memory/social/index.js";
import { WorkingMemoryStore, type WorkingMemory } from "../memory/working/index.js";
import { EpisodicRepository } from "../memory/episodic/index.js";
import { type IdentityService } from "../memory/identity/index.js";
import { StreamReader, StreamWriter } from "../stream/index.js";
import type { ToolDispatcher } from "../tools/index.js";
import { ConfigError, SessionBusyError } from "../util/errors.js";
import { SystemClock, type Clock } from "../util/clock.js";
import { DEFAULT_SESSION_ID, type SessionId } from "../util/ids.js";
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
  openQuestionsRepository: OpenQuestionsRepository;
  moodRepository: MoodRepository;
  socialRepository: SocialRepository;
  skillRepository: SkillRepository;
  proceduralEvidenceRepository: ProceduralEvidenceRepository;
  skillSelector: SkillSelector;
  entityRepository: EntityRepository;
  commitmentRepository: CommitmentRepository;
  reviewQueueRepository: ReviewQueueRepository;
  identityService: IdentityService;
  workingMemoryStore: WorkingMemoryStore;
  llmFactory: () => LLMClient;
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
  private readonly turnContextCompiler: TurnContextCompiler;
  private readonly createStreamReader: (sessionId: SessionId) => StreamReader;
  private readonly sessionLock: SessionLock;
  private readonly tracer: TurnTracer;
  private readonly attributionLifecycleService: AttributionLifecycleService;
  private readonly turnRetrievalCoordinator: TurnRetrievalCoordinator;
  private readonly commitmentGuardRunner: CommitmentGuardRunner;
  private readonly pendingProceduralAttemptTracker: PendingProceduralAttemptTracker;

  constructor(private readonly options: TurnOrchestratorOptions) {
    this.clock = options.clock ?? new SystemClock();
    this.tracer = options.tracer ?? NOOP_TRACER;
    this.turnContextCompiler = options.turnContextCompiler ?? new TurnContextCompiler();
    this.sessionLock =
      options.sessionLock ??
      new SessionLock({
        dataDir: options.config.dataDir,
      });
    this.createStreamReader =
      options.createStreamReader ??
      ((sessionId) =>
        new StreamReader({
          dataDir: options.config.dataDir,
          sessionId,
        }));
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

  private buildSelfSnapshot(): SelfSnapshot {
    const values = this.options.valuesRepository.list();
    const goals = flattenGoals(this.options.goalsRepository.list({ status: "active" }));
    const traits = this.options.traitsRepository.list();
    const currentPeriod = this.options.autobiographicalRepository?.currentPeriod() ?? null;
    const recentGrowthMarkers = this.options.growthMarkersRepository?.list({ limit: 3 }) ?? [];

    return {
      values,
      goals,
      traits,
      currentPeriod,
      recentGrowthMarkers,
    };
  }

  private getOptionalLlmClient(): LLMClient | undefined {
    try {
      return this.options.llmFactory();
    } catch (error) {
      if (error instanceof ConfigError) {
        return undefined;
      }

      throw error;
    }
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
  ): Promise<void> {
    await appendInternalFailureEvent(streamWriter, hook, error);
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
        const optionalPerceptionLlm =
          this.options.config.perception.useLlmFallback === true
            ? this.getOptionalLlmClient()
            : undefined;
        const perceiver = new Perceiver({
          llmClient: optionalPerceptionLlm,
          model: this.options.config.anthropic.models.background,
          useLlmFallback: this.options.config.perception.useLlmFallback,
          modeWhenLlmAbsent: this.options.config.perception.modeWhenLlmAbsent,
          affectiveUseLlmFallback: this.options.config.affective.useLlmFallback,
          // Temporal cue uses the same LLM gate as mode detection: both rely
          // on the perception-bound LLM client. Turning off perception LLM
          // fallback turns off temporal extraction too (degrades to null).
          temporalCueUseLlmFallback: this.options.config.perception.useLlmFallback,
          detectAffectiveSignal: this.options.affectiveSignalDetector,
          onAffectiveError: (error) =>
            this.appendHookFailureEvent(streamWriter, "affective_extraction", error),
          clock: this.clock,
          tracer: this.tracer,
          turnId,
        });
        const llmClient = this.options.llmFactory();
        const selfSnapshot = this.buildSelfSnapshot();
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
        // Compile recent dialogue BEFORE appending the current user message,
        // so the window contains prior turns only. The compiler guarantees
        // the window starts with a user role and ends with an assistant
        // role, making it safe to concatenate with a trailing
        // {role:"user", content: currentUserMessage}.
        const recencyWindow: RecencyWindow = this.turnContextCompiler.compile(
          this.createStreamReader(sessionId),
          {
            includeSelfTurns: isSelfAudience,
          },
        );
        if (this.tracer.enabled) {
          this.tracer.emit("recency_compiled", {
            turnId,
            messageCount: recencyWindow.messages.length,
            sourceEntryIds: recencyWindow.messages.map((message) => message.stream_entry_id),
          });
        }
        const recentHistoryStrings = recencyWindow.messages.map(
          (message) => `${message.role}: ${message.content}`,
        );
        const perception = await perceiver.perceive(cognitionInput, recentHistoryStrings);
        const workingMood =
          input.origin === "autonomous"
            ? (workingMemory.mood ?? createNeutralAffectiveSignal())
            : perception.affectiveSignal;

        workingMemory = {
          ...workingMemory,
          turn_counter: workingMemory.turn_counter + 1,
          current_focus: perception.entities[0] ?? (cognitionInput.slice(0, 80) || null),
          hot_entities: perception.entities,
          mood: workingMood,
          mode: perception.mode,
          updated_at: this.clock.now(),
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

        const userEntry = {
          kind: "user_msg",
          content: input.userMessage,
          ...(input.audience === undefined ? {} : { audience: input.audience }),
        } satisfies Parameters<StreamWriter["append"]>[0];

        const persistedUserEntry = await streamWriter.append(userEntry);

        workingMemory = this.options.workingMemoryStore.save({
          ...workingMemory,
          pending_social_attribution: pendingSocialAttribution,
          pending_trait_attribution: pendingTraitAttribution,
          suppressed: suppressionSet.snapshot(),
          updated_at: this.clock.now(),
        });

        await streamWriter.append({
          kind: "perception",
          content: {
            mode: perception.mode,
            entities: perception.entities,
            temporalCue: perception.temporalCue,
            affectiveSignal: perception.affectiveSignal,
          },
          ...(input.audience === undefined ? {} : { audience: input.audience }),
        });
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
          suppressionSet,
          findEntityByName: (name) => this.options.entityRepository.findByName(name),
        });
        const applicableCommitments = retrievalContext.applicableCommitments;
        const pendingCorrections = retrievalContext.pendingCorrections;
        const affectiveTrajectory = retrievalContext.affectiveTrajectory;
        const retrieval = retrievalContext.retrieval;
        const retrievedEpisodes = retrievalContext.retrievedEpisodes;
        const retrievedSemantic = retrievalContext.retrievedSemantic;
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
        const reflector = new Reflector({
          clock: this.clock,
          llmClient,
          model: this.options.config.anthropic.models.background,
        });
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
            episodicRepository: this.options.episodicRepository,
            goalsRepository: this.options.goalsRepository,
            traitsRepository: this.options.traitsRepository,
            openQuestionsRepository: this.options.openQuestionsRepository,
            identityService: this.options.identityService,
            reviewQueueRepository: this.options.reviewQueueRepository,
            skillRepository: this.options.skillRepository,
            proceduralEvidenceRepository: this.options.proceduralEvidenceRepository,
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
