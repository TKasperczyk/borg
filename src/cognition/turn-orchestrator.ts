import { randomUUID } from "node:crypto";

import type { Config } from "../config/index.js";
import { SuppressionSet, computeRetrievalLimit, computeWeights } from "./attention/index.js";
import { performAction, type ToolLoopCallRecord } from "./action/index.js";
import { formatAutonomyTriggerContext, type AutonomyTriggerContext } from "./autonomy-trigger.js";
import { Deliberator, type SelfSnapshot, type TurnStakes } from "./deliberation/deliberator.js";
import { detectAffectiveSignal } from "./perception/affective-signal.js";
import { Perceiver } from "./perception/index.js";
import { TurnContextCompiler, type RecencyWindow } from "./recency/index.js";
import type { StreamIngestionCoordinator } from "./ingestion/index.js";
import { Reflector } from "./reflection/index.js";
import type { RetrievalPipeline, RetrievalSearchOptions } from "../retrieval/index.js";
import type { LLMClient } from "../llm/index.js";
import { createNeutralAffectiveSignal, MoodRepository } from "../memory/affective/index.js";
import {
  CommitmentChecker,
  CommitmentRepository,
  EntityRepository,
  type CommitmentRecord,
} from "../memory/commitments/index.js";
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
  type ValueRecord,
} from "../memory/self/index.js";
import { ReviewQueueRepository } from "../memory/semantic/index.js";
import { SocialRepository } from "../memory/social/index.js";
import {
  PENDING_PROCEDURAL_ATTEMPT_TTL_TURNS,
  PENDING_PROCEDURAL_ATTEMPTS_LIMIT,
  WorkingMemoryStore,
  type WorkingMemory,
} from "../memory/working/index.js";
import { EpisodicRepository } from "../memory/episodic/index.js";
import { type IdentityService } from "../memory/identity/index.js";
import { StreamReader, StreamWriter } from "../stream/index.js";
import type { ToolDispatcher } from "../tools/index.js";
import { ConfigError, SessionBusyError } from "../util/errors.js";
import { SystemClock, type Clock } from "../util/clock.js";
import { DEFAULT_SESSION_ID, type EpisodeId, type SessionId } from "../util/ids.js";
import { NOOP_TRACER, type TurnTracer } from "./tracing/tracer.js";
import type { CognitiveMode, IntentRecord } from "./types.js";
import { SessionLock } from "./session-lock.js";

const PENDING_SOCIAL_ATTRIBUTION_TTL_MS = 60 * 60 * 1_000;
const PENDING_TRAIT_ATTRIBUTION_TTL_MS = PENDING_SOCIAL_ATTRIBUTION_TTL_MS;
const TRAIT_ATTRIBUTION_POSITIVE_VALENCE_THRESHOLD = 0.2;

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

function selectActiveValues(values: readonly ValueRecord[], candidateLimit = 2): ValueRecord[] {
  const established = values.filter((value) => value.state === "established");
  const candidates = values
    .filter((value) => value.state !== "established")
    .sort((left, right) => right.priority - left.priority || left.created_at - right.created_at)
    .slice(0, candidateLimit);

  return [...established, ...candidates];
}

function compactTurnText(text: string, maxLength: number): string {
  const compacted = text.replace(/\s+/g, " ").trim();

  return compacted.length <= maxLength ? compacted : compacted.slice(0, maxLength).trimEnd();
}

function buildSkillSelectionQuery(userMessage: string, entities: readonly string[]): string {
  return [userMessage, ...entities]
    .map((part) => part.trim())
    .filter((part) => part.length > 0)
    .join(" ");
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

  private collectApplicableCommitments(
    audienceEntityId: ReturnType<EntityRepository["resolve"]> | null,
    perceivedEntities: readonly string[],
  ): CommitmentRecord[] {
    const aboutEntityIds: Array<ReturnType<EntityRepository["resolve"]> | null> = [];
    const seenEntities = new Set<string>();

    for (const entity of perceivedEntities) {
      const normalized = entity.trim();

      if (normalized.length === 0) {
        continue;
      }

      const key = normalized.toLowerCase();

      if (seenEntities.has(key)) {
        continue;
      }

      seenEntities.add(key);
      const entityId = this.options.entityRepository.findByName(normalized);

      if (entityId !== null) {
        aboutEntityIds.push(entityId);
      }
    }

    if (aboutEntityIds.length === 0) {
      aboutEntityIds.push(null);
    }

    const byId = new Map<string, CommitmentRecord>();

    for (const aboutEntityId of aboutEntityIds) {
      const applicable = this.options.commitmentRepository.getApplicable({
        audience: audienceEntityId,
        aboutEntity: aboutEntityId,
        nowMs: this.clock.now(),
      });

      for (const commitment of applicable) {
        byId.set(commitment.id, commitment);
      }
    }

    return [...byId.values()].sort(
      (left, right) => right.priority - left.priority || left.created_at - right.created_at,
    );
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

  private async resolveTraitEvidenceEpisodes(
    attribution: NonNullable<WorkingMemory["pending_trait_attribution"]>,
  ): Promise<EpisodeId[]> {
    if (attribution.source_stream_entry_ids.length > 0) {
      const resolved = await this.options.episodicRepository.findBySourceStreamIds(
        attribution.source_stream_entry_ids,
      );

      if (resolved !== null) {
        return [resolved.id];
      }
    }

    return [...attribution.source_episode_ids];
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
        let pendingSocialAttribution = workingMemory.pending_social_attribution;
        let pendingTraitAttribution = workingMemory.pending_trait_attribution;

        if (isUserTurn && pendingSocialAttribution !== null) {
          const nowMs = this.clock.now();
          const expired =
            nowMs - pendingSocialAttribution.turn_completed_ts > PENDING_SOCIAL_ATTRIBUTION_TTL_MS;
          const audienceMatches =
            audienceEntityId !== null && pendingSocialAttribution.entity_id === audienceEntityId;

          if (expired || !audienceMatches) {
            await streamWriter.append({
              kind: "internal_event",
              content: {
                kind: "social_attribution_drop",
                reason: expired ? "expired" : "audience_mismatch",
                pending_entity_id: pendingSocialAttribution.entity_id,
                current_audience_entity_id: audienceEntityId,
                turn_completed_ts: pendingSocialAttribution.turn_completed_ts,
                agent_response_summary: pendingSocialAttribution.agent_response_summary,
              },
            });
            pendingSocialAttribution = null;
          } else {
            try {
              this.options.socialRepository.attachSentiment(
                pendingSocialAttribution.interaction_id,
                {
                  valence: perception.affectiveSignal.valence,
                  now: nowMs,
                },
              );
              audienceProfile = this.options.socialRepository.getProfile(audienceEntityId);
              pendingSocialAttribution = null;
            } catch (error) {
              await this.appendHookFailureEvent(streamWriter, "social_update", error);
            }
          }
        }

        if (isUserTurn && pendingTraitAttribution !== null) {
          const nowMs = this.clock.now();
          const expired =
            nowMs - pendingTraitAttribution.turn_completed_ts > PENDING_TRAIT_ATTRIBUTION_TTL_MS;
          const audienceMatches = pendingTraitAttribution.audience_entity_id === audienceEntityId;

          if (expired || !audienceMatches) {
            await streamWriter.append({
              kind: "internal_event",
              content: {
                kind: "trait_attribution_drop",
                reason: expired ? "expired" : "audience_mismatch",
                pending_trait_label: pendingTraitAttribution.trait_label,
                pending_audience_entity_id: pendingTraitAttribution.audience_entity_id,
                current_audience_entity_id: audienceEntityId,
                turn_completed_ts: pendingTraitAttribution.turn_completed_ts,
                source_episode_ids: pendingTraitAttribution.source_episode_ids,
                source_stream_entry_ids: pendingTraitAttribution.source_stream_entry_ids,
              },
            });
            pendingTraitAttribution = null;
          } else if (
            perception.affectiveSignal.valence > TRAIT_ATTRIBUTION_POSITIVE_VALENCE_THRESHOLD
          ) {
            // Sprint 56: resolve the demonstrating turn's stream entries
            // to the extracted episode. If extraction hasn't completed
            // yet, keep the attribution pending so a later turn can
            // reinforce once an episode exists; TTL still expires it
            // eventually. Legacy state may carry source_episode_ids
            // directly. Resolution is wrapped so a repository failure
            // can't take down the user turn.
            try {
              const evidenceEpisodeIds = await this.resolveTraitEvidenceEpisodes(
                pendingTraitAttribution,
              );

              if (evidenceEpisodeIds.length > 0) {
                this.options.traitsRepository.reinforce({
                  label: pendingTraitAttribution.trait_label,
                  delta: pendingTraitAttribution.strength_delta,
                  provenance: {
                    kind: "episodes",
                    episode_ids: evidenceEpisodeIds,
                  },
                  timestamp: nowMs,
                });
                pendingTraitAttribution = null;
              }
            } catch (error) {
              await this.appendHookFailureEvent(streamWriter, "trait_update", error);
            }
          } else {
            pendingTraitAttribution = null;
          }
        }

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
        const applicableCommitments = this.collectApplicableCommitments(
          audienceEntityId,
          perception.entities,
        );
        const pendingCorrections = this.options.reviewQueueRepository
          .list({
            kind: "correction",
            openOnly: true,
          })
          .filter((item) => {
            const correctionAudience =
              typeof item.refs.audience_entity_id === "string"
                ? item.refs.audience_entity_id
                : null;

            if (audienceEntityId === null) {
              return correctionAudience === null;
            }

            return correctionAudience === null || correctionAudience === audienceEntityId;
          });
        const perceivedMood = workingMemory.mood ?? createNeutralAffectiveSignal();
        const perceivedMoodActive =
          Math.abs(perceivedMood.valence) + Math.abs(perceivedMood.arousal) > 0.3;
        const retrievalMood = perceivedMoodActive
          ? perceivedMood
          : this.options.moodRepository.current(sessionId);
        const affectiveTrajectory = this.options.moodRepository.history(sessionId, {
          limit: 5,
        });
        const activeValues = selectActiveValues(selfSnapshot.values);

        const attentionWeights = computeWeights(perception.mode, {
          currentGoals: selfSnapshot.goals,
          hasActiveValues: activeValues.length > 0,
          hasTemporalCue: perception.temporalCue !== null,
          moodActive: Math.abs(retrievalMood.valence) + Math.abs(retrievalMood.arousal) > 0.3,
          audienceTrust: audienceProfile?.trust ?? null,
        });
        const retrievalOptions: RetrievalSearchOptions = {
          limit: computeRetrievalLimit(perception.mode),
          audienceEntityId,
          attentionWeights,
          goalDescriptions: selfSnapshot.goals.map((goal) => goal.description),
          activeValues,
          temporalCue: perception.temporalCue,
          strictTimeRange: perception.temporalCue !== null,
          moodState: retrievalMood,
          audienceProfile,
          audienceTerms: isSelfAudience
            ? []
            : audienceEntity === null
              ? input.audience === undefined
                ? []
                : [input.audience]
              : [
                  audienceEntity.canonical_name,
                  ...audienceEntity.aliases,
                  ...(input.audience === undefined ? [] : [input.audience]),
                ],
          entityTerms: perception.entities,
          suppressionSet,
          includeOpenQuestions: perception.mode === "reflective",
          traceTurnId: turnId,
        };
        const retrieval = await this.options.retrievalPipeline.searchWithContext(
          cognitionInput,
          retrievalOptions,
        );
        const retrievedEpisodes = retrieval.episodes;
        const retrievedSemantic = retrieval.semantic;
        const skillSelectionQuery = buildSkillSelectionQuery(
          input.userMessage,
          perception.entities,
        );
        const selectedSkill =
          perception.mode === "problem_solving"
            ? await this.options.skillSelector.select(skillSelectionQuery, {
                k: 5,
              })
            : null;
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
            reRetrieve: (query, overrides = {}) =>
              this.options.retrievalPipeline.search(query, {
                ...retrievalOptions,
                ...overrides,
              }),
          },
          streamWriter,
        );

        workingMemory = {
          ...workingMemory,
          updated_at: this.clock.now(),
        };
        const commitmentChecker = new CommitmentChecker({
          llmClient,
          detectionModel: this.options.config.anthropic.models.background,
          rewriteModel: this.options.config.anthropic.models.cognition,
          entityRepository: this.options.entityRepository,
        });
        const commitmentCheckerUserMessage =
          input.origin === "autonomous" ? input.userMessage : cognitionInput;
        const commitmentCheck = await commitmentChecker.check({
          response: deliberation.response,
          userMessage: commitmentCheckerUserMessage,
          untrustedContext:
            input.origin === "autonomous" &&
            input.autonomyTrigger !== null &&
            input.autonomyTrigger !== undefined
              ? cognitionInput
              : null,
          commitments: applicableCommitments,
          relevantEntities: perception.entities,
        });
        if (this.tracer.enabled) {
          this.tracer.emit("commitment_check", {
            turnId,
            verdict: commitmentCheck.fallback_applied
              ? "fallback_applied"
              : commitmentCheck.revised
                ? "rewritten"
                : "passed",
            rewriteTriggered: commitmentCheck.revised,
            violationCount: commitmentCheck.violations.length,
          });
        }

        if (commitmentCheck.fallback_applied) {
          await streamWriter.append({
            kind: "internal_event",
            content:
              "Commitment guard fell back to a softened response after revision still violated an active commitment.",
          });
        }

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
        // Sprint 53: bounded list of pending procedural attempts.
        // Reflection retires only grounded success/failure outcomes;
        // unclear ones survive here until they age out (TTL) or get
        // graded on a later turn.
        const carriedAttempts = reflectedWorkingMemory.pending_procedural_attempts.filter(
          (attempt) =>
            reflectedWorkingMemory.turn_counter - attempt.turn_counter <
            PENDING_PROCEDURAL_ATTEMPT_TTL_TURNS,
        );
        const shouldAppendNewAttempt =
          isUserTurn && perception.mode === "problem_solving";
        const newAttempt = shouldAppendNewAttempt
          ? {
              problem_text: compactTurnText(input.userMessage, 1_000),
              approach_summary:
                selectedSkill?.skill.approach ??
                (compactTurnText(actionResult.response, 1_000) || "No explicit approach stated."),
              selected_skill_id: selectedSkill?.skill.id ?? null,
              source_stream_ids: [persistedUserEntry.id, persistedAgentEntry.id],
              turn_counter: reflectedWorkingMemory.turn_counter,
              audience_entity_id: audienceEntityId,
            }
          : null;
        const combinedAttempts = newAttempt === null
          ? carriedAttempts
          : [...carriedAttempts, newAttempt];
        // Cap by dropping oldest if needed.
        const nextPendingProceduralAttempts =
          combinedAttempts.length > PENDING_PROCEDURAL_ATTEMPTS_LIMIT
            ? combinedAttempts.slice(-PENDING_PROCEDURAL_ATTEMPTS_LIMIT)
            : combinedAttempts;

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
