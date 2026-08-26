import type { AttentionWeights, TemporalCue } from "../contracts/cognitive-contracts.js";
import {
  commitmentMemoryDisclosureLabel,
  imagePerceptionMemoryDisclosureLabel,
  openQuestionMemoryDisclosureLabel,
} from "../memory/common/disclosure-serializers.js";
import type { ImagePerceptionRepository, ImagePerceptionSearchHit } from "../attachments/index.js";
import type { EmbeddingClient } from "../embeddings/index.js";
import type { LLMClient } from "../llm/index.js";
import {
  effectiveCommitmentCriticalDomain,
  effectiveCommitmentEnforcementClass,
  type CommitmentRecord,
  type CommitmentRepository,
  type EntityRepository,
} from "../memory/commitments/index.js";
import {
  isEpisodeVisibleToCapability,
  resolveViewerCapability,
} from "../memory/episodic/access.js";
import { parseEpisodeParticipantEntityIdTerm } from "../memory/episodic/participant-terms.js";
import type { EpisodicRepository } from "../memory/episodic/repository.js";
import type {
  Episode,
  EpisodeCognitionRecallOptions,
  EpisodeSearchCandidate,
  EpisodeSearchOptions,
} from "../memory/episodic/types.js";
import type { DecayOptions } from "../memory/episodic/decay.js";
import type { OpenQuestion, OpenQuestionsRepository, ValueRecord } from "../memory/self/index.js";
import type { SemanticGraph } from "../memory/semantic/graph.js";
import type {
  SemanticEdgeRepository,
  SemanticNodeRepository,
} from "../memory/semantic/repository.js";
import type { ReviewQueueRepository } from "../memory/review-queue/review-queue.js";
import type { SemanticNode } from "../memory/semantic/types.js";
import type { SocialProfile } from "../memory/social/index.js";
import type { StreamEntry, StreamEntryIndexRepository } from "../stream/index.js";
import { NOOP_TRACER, type TurnTracer } from "../tracing/tracer.js";
import { SystemClock, type Clock } from "../util/clock.js";
import { mapWithConcurrency } from "../util/collections.js";
import { StorageError } from "../util/errors.js";
import {
  DEFAULT_SESSION_ID,
  type AttachmentId,
  type EntityId,
  type EpisodeId,
  type SessionId,
  type StreamEntryId,
} from "../util/ids.js";
import type { JsonValue } from "../util/json-value.js";
import { formatRelativeAge } from "../util/relative-time.js";

import { CitationResolver, type CitationResolverOptions } from "./citations.js";
import { assembleRetrievedContext, type RetrievedContext } from "./context-assembly.js";
import { cosineSimilarity } from "./embedding-similarity.js";
import { rankEvidenceItems } from "./evidence-pool.js";
import {
  EXACT_TERM_RESERVED_SLOTS,
  projectEpisodes,
  projectOpenQuestions,
  projectSemantic,
  type EpisodeProjectionSource,
} from "./evidence-projections.js";
import { DEFAULT_MMR_LAMBDA } from "./mmr.js";
import { retrieveOpenQuestionsForQuery as retrieveOpenQuestionsForQueryFromRepository } from "./open-questions.js";
import { RawStreamAdapter } from "./raw-stream-adapter.js";
import {
  DEFAULT_RECALL_STATE_MAX_ACTIVE_HANDLES,
  DEFAULT_RECALL_STATE_MAX_NEW_HANDLES_PER_TURN,
  DEFAULT_RECALL_STATE_MAX_WARM_EVIDENCE_RENDERED,
  DEFAULT_RECALL_STATE_REINFORCEMENT_CAP,
  DEFAULT_RECALL_STATE_TTL_TURNS,
  DEFAULT_RECALL_STATE_WARM_SUPPRESSION_TURNS,
  createEmptyRecallState,
  effectiveRecallStateReinforcementCount,
  deriveRecallEvidenceHandle,
  normalizeRecallEvidenceHandle,
  recallEvidenceHandleKey,
  type RecallState,
  type RecallStateHandle,
  type RecallStateRepository,
} from "./recall-state.js";
import { DEFAULT_RECALL_EXPANSION_MODEL, expandRecall } from "./recall-expansion.js";
import type {
  EvidenceItem,
  EvidencePool,
  RecallEvidenceHandle,
  RecallIntent,
  RecallIntentKind,
} from "./recall-types.js";
import {
  buildRetrievedEpisode,
  clamp,
  DEFAULT_EPISODE_SCORE_WEIGHTS,
  participantEntityResolutionKey,
  scoreCandidate,
  type EpisodeScoreDefaults,
  type ParticipantEntityResolutionLookup,
  type RetrievalMoodState,
  type RetrievedEpisode,
  type ScoreWeights,
  type SuppressionLookup,
} from "./scoring.js";
import {
  buildRetrievalScoringFeatures,
  type RetrievalScoringFeatures,
} from "./scoring-features.js";
import {
  resolveSemanticContextForCognition,
  resolveSemanticContextForDisclosure,
  resolveSemanticDisclosureSourceAdapter,
  toRetrievedSemantic,
  type ResolvedSemanticRetrieval,
  type RetrievedSemantic,
  type SemanticSourceAdapter,
  type SemanticStatusMultipliers,
} from "./semantic-retrieval.js";
import { resolveTimeSignals } from "./time-signals.js";
import {
  SELF_RECALL_SCOPE,
  combineMemoryDisclosureLabels,
  memoryDisclosureLabelFromEpisodeAccess,
  unknownMemoryDisclosureLabel,
  type CognitionRecallContext,
  type DisclosureContext,
  type MemoryDisclosureLabel,
} from "./recall-context.js";

export type {
  RetrievedContext,
  RetrievedContradictionRouting,
  RetrievedContradictionRoutingItem,
  RetrievedContradictionSessionScope,
} from "./context-assembly.js";
export type { RetrievedEpisode } from "./scoring.js";
export type {
  RetrievedSemantic,
  RetrievedSemanticHit,
  RetrievedSemanticNode,
  RetrievedSemanticUnderReview,
} from "./semantic-retrieval.js";

export type RetrievalPipelineOptions = {
  embeddingClient: EmbeddingClient;
  llmClient?: LLMClient;
  recallExpansionModel?: string;
  episodicRepository: EpisodicRepository;
  dataDir: string;
  entryIndex?: StreamEntryIndexRepository;
  semanticNodeRepository?: SemanticNodeRepository;
  semanticEdgeRepository?: Pick<SemanticEdgeRepository, "getEdge">;
  semanticGraph?: SemanticGraph;
  reviewQueueRepository?: Pick<
    ReviewQueueRepository,
    "listOpenBeliefRevisionsByTarget" | "listOpenBeliefRevisionsByTargetForCognition"
  >;
  openQuestionsRepository?: OpenQuestionsRepository;
  imagePerceptionRepository?: ImagePerceptionRepository;
  entityRepository?: Pick<EntityRepository, "findByName">;
  commitmentRepository?: Pick<CommitmentRepository, "get" | "list">;
  recallStateRepository?: Pick<RecallStateRepository, "load" | "save">;
  clock?: Clock;
  tracer?: TurnTracer;
  scoreWeights?: ScoreWeights;
  mmrLambda?: number;
  decayOptions?: Omit<DecayOptions, "nowMs">;
  semanticUnderReviewMultiplier?: number;
  semanticStatusMultipliers?: Partial<SemanticStatusMultipliers>;
  semanticOverfetchMultiplier?: number;
  commitmentEvidenceSimilarityThreshold?: number;
  recallStateTtlTurns?: number;
  recallStateWarmSuppressionTurns?: number;
  recallStateMaxActiveHandles?: number;
  recallStateMaxNewHandlesPerTurn?: number;
  recallStateMaxWarmEvidenceRendered?: number;
  maxRetrievedImageRefs?: number;
  lexicalFusionEnabled?: boolean;
};

export type RetrievalSharedOptions = EpisodeCognitionRecallOptions & {
  rankingAudienceEntityId?: EntityId | null;
  mmrLambda?: number;
  scoreWeights?: ScoreWeights;
  decayOptions?: Omit<DecayOptions, "nowMs">;
  attentionWeights?: AttentionWeights;
  goalDescriptions?: readonly string[];
  primaryGoalDescription?: string;
  activeValues?: readonly ValueRecord[];
  scoringFeatures?: RetrievalScoringFeatures;
  temporalCue?: TemporalCue | null;
  strictTimeRange?: boolean;
  suppressionSet?: SuppressionLookup;
  graphWalkDepth?: number;
  maxGraphNodes?: number;
  asOf?: number;
  underReviewMultiplier?: number;
  statusMultipliers?: Partial<SemanticStatusMultipliers>;
  semanticOverfetchMultiplier?: number;
  includeOpenQuestions?: boolean;
  openQuestionsLimit?: number;
  moodState?: RetrievalMoodState | null;
  audienceProfile?: SocialProfile | null;
  audienceTerms?: readonly string[];
  entityTerms?: readonly string[];
  currentTurnAttachmentIds?: readonly AttachmentId[];
  sessionId?: SessionId;
  turnCounter?: number;
  traceTurnId?: string;
  recordRetrieval?: boolean;
};

export type CognitionRetrievalOptions = RetrievalSharedOptions & {
  recallContext: CognitionRecallContext;
  disclosureContext?: DisclosureContext;
  audienceEntityId?: never;
  crossAudience?: never;
};

export type DisclosureRetrievalOptions = RetrievalSharedOptions &
  Pick<EpisodeSearchOptions, "audienceEntityId" | "crossAudience"> & {
    disclosureContext?: DisclosureContext;
    recallContext?: never;
  };

export type CognitionRecallSearchOptions = CognitionRetrievalOptions;

/**
 * @deprecated Use DisclosureRetrievalOptions for disclosure reads or CognitionRetrievalOptions
 * for cognition recall.
 */
export type RetrievalSearchOptions = DisclosureRetrievalOptions;

type RetrievalExecutionOptions = CognitionRetrievalOptions | DisclosureRetrievalOptions;

type RetrievalExecutionMode = "cognition" | "disclosure";

// What the caller will actually consume from RetrievedContext.
//
// "episodes-only" is the searchEpisodesForDisclosure() contract: that entry
// returns `result.episodes` and discards the rest, and projectEpisodes() can
// only ever select candidates produced by the EPISODIC lane (it filters the
// evidence pool to source === "episode" and requires a hydrated projection
// source from collectEpisodicEvidenceCandidates — see evidence-projections.ts).
// The semantic, open-question, image-perception, and commitment-evidence lanes
// are therefore provably unobservable to episodes-only callers, yet they cost
// the bulk of a recall: each lane re-embeds its intents over the network and
// runs its own vector scans (~20+ embedding round-trips per call before the
// 2026-08 embedding cache, and most of the CPU after it). The memory sidecar's
// POST /memory/recall — team-agent's ambient-recall hot path with a 5s client
// deadline — is the main episodes-only caller, and these lanes were the reason
// it couldn't meet that deadline.
//
// Observable differences when "episodes-only" is active, all accepted:
//   - retrieval.completed trace: semanticHits is 0 and `confidence` reflects
//     episodic evidence only;
//   - recall_state (warm-recall continuity) is saved with episodic-lane fresh
//     evidence only — inconsequential for sidecar calls, which carry no
//     sessionId and thus use their own recall-state scope;
//   - RetrievedContext.semantic / open_questions / image perceptions come back
//     empty (the caller discards them by definition).
// Everything else is unchanged: recall expansion, the episodic lane itself,
// citation resolution, retrieval_log/episode_stats recording, MMR projection.
//
// "full" keeps the complete pipeline and stays the default for
// recallEpisodesForCognition() and searchWithContextForDisclosure().
type RetrievalProjection = "full" | "episodes-only";

export type RetrievalGetEpisodeOptions = {
  audienceEntityId?: EntityId | null;
  crossAudience?: boolean;
};

type ExpansionOutcome = {
  succeeded: boolean;
  facetIntents: RecallIntent[];
  namedTerms: string[];
};

type EpisodeEvidenceCandidate = {
  intent: RecallIntent;
  candidate: EpisodeSearchCandidate;
  matchedTerms: string[];
  score: EpisodeScoreDetails;
};

type RawEpisodeEvidenceCandidate = Omit<EpisodeEvidenceCandidate, "score">;

type EpisodeScoreDetails = {
  decayedSalience: number;
  heat: number;
  goalRelevance: number;
  valueAlignment: number;
  timeRelevance: number;
  moodBoost: number;
  socialRelevance: number;
  entityRelevance: number;
  suppressionPenalty: number;
  score: number;
  // Pre-clamp fused score including intent boosts (diagnostic export only).
  rawScore: number;
};

type SemanticEvidenceCandidate = {
  intent: RecallIntent;
  semantic: ResolvedSemanticRetrieval;
};

function semanticSourceAdapterPartialEvidenceFields(
  source: SemanticSourceAdapter,
): Pick<EvidenceItem, "partial_source_visibility" | "source_visibility_fraction"> {
  return source.partial
    ? {
        partial_source_visibility: true,
        source_visibility_fraction: source.sourceVisibilityFraction,
      }
    : {};
}

type OpenQuestionEvidenceCandidate = {
  intent: RecallIntent;
  question: OpenQuestion;
  score: number;
};

type ImagePerceptionEvidenceCandidate = {
  intent: RecallIntent;
  hit: ImagePerceptionSearchHit;
};

type RecallStateTurnContext = {
  state: RecallState;
  scopeKey: string;
  turn: number;
};

type WarmRecallCandidate = {
  key: string;
  handle: RecallEvidenceHandle;
  stateHandle: RecallStateHandle;
};

// Tunes the minimum similarity for commitment evidence admission.
const DEFAULT_COMMITMENT_EVIDENCE_SIMILARITY_THRESHOLD = 0.3;

// Tunes concurrent fanout across independent retrieval intent searches.
const RETRIEVAL_FANOUT_CONCURRENCY = 5;

// Names warm recall evidence so recurrence state can be traced separately from fresh recall.
const WARM_RECALL_INTENT_ID = "warm_recall";

// Tunes how much known-term intents lift exact episodic matches.
const KNOWN_TERM_INTENT_SCORE_BOOST = 0.25;

// Tunes how much recent intents lift recent episodic matches.
const RECENT_INTENT_SCORE_BOOST = 0.05;

// Tunes priority for episode handles retained in recall state.
const RECALL_HANDLE_EPISODE_RETENTION_RANK = 6;

// Tunes priority for raw stream handles with a parent episode retained in recall state.
const RECALL_HANDLE_PARENTED_RAW_STREAM_RETENTION_RANK = 5;

// Tunes priority for commitment handles retained in recall state.
const RECALL_HANDLE_COMMITMENT_RETENTION_RANK = 4;

// Tunes priority for open question handles retained in recall state.
const RECALL_HANDLE_OPEN_QUESTION_RETENTION_RANK = 3;

// Tunes priority for semantic and other secondary handles retained in recall state.
const RECALL_HANDLE_SECONDARY_RETENTION_RANK = 2;

// Tunes priority for orphan raw stream handles retained in recall state.
const RECALL_HANDLE_ORPHAN_RAW_STREAM_RETENTION_RANK = 1;

// Tunes the baseline score for warm recall evidence.
const WARM_RECALL_BASE_SCORE = 0.12;

// Tunes the maximum reinforcement bonus for warm recall evidence.
const WARM_RECALL_MAX_REINFORCEMENT_BONUS = 0.18;

// Tunes the per-reinforcement score increment for warm recall evidence.
const WARM_RECALL_REINFORCEMENT_STEP = 0.03;

// Tunes the maximum score allowed for warm recall evidence.
const WARM_RECALL_SCORE_CAP = 0.3;

// Tunes the default confidence for semantic evidence without an edge path.
const SEMANTIC_EVIDENCE_EMPTY_EDGE_PATH_CONFIDENCE = 0.3;

// Tunes provenance strength rendered for image perception evidence.
const IMAGE_PERCEPTION_PROVENANCE_SCORE = 0.7;

// Tunes raw stream evidence score for recent-turn recall intents.
const RECENT_RAW_STREAM_EVIDENCE_SCORE = 0.2;

// Tunes raw stream evidence score for non-recency recall intents.
const DEFAULT_RAW_STREAM_EVIDENCE_SCORE = 1;

export class RetrievalPipeline {
  private readonly clock: Clock;
  private readonly scoreWeights: ScoreWeights;
  private readonly mmrLambda: number;
  private readonly decayOptions?: Omit<DecayOptions, "nowMs">;
  private readonly tracer: TurnTracer;
  private readonly lexicalFusionEnabled: boolean;

  constructor(private readonly options: RetrievalPipelineOptions) {
    this.clock = options.clock ?? new SystemClock();
    this.tracer = options.tracer ?? NOOP_TRACER;
    this.scoreWeights = options.scoreWeights ?? { ...DEFAULT_EPISODE_SCORE_WEIGHTS };
    this.mmrLambda = options.mmrLambda ?? DEFAULT_MMR_LAMBDA;
    this.decayOptions = options.decayOptions;
    this.lexicalFusionEnabled = options.lexicalFusionEnabled ?? false;
  }

  retrieveOpenQuestionsForQuery(
    query: string,
    options: {
      relatedSemanticNodeIds?: readonly SemanticNode["id"][];
      limit?: number;
      queryVector?: Float32Array;
      traceTurnId?: string;
      sessionId?: SessionId;
    } = {},
  ): Promise<OpenQuestion[]> {
    return retrieveOpenQuestionsForQueryFromRepository(
      this.options.openQuestionsRepository,
      this.options.embeddingClient,
      query,
      {
        ...options,
        onDegraded:
          this.tracer.enabled && options.traceTurnId !== undefined
            ? (reason, error) => {
                this.tracer.emit("retrieval.degraded", {
                  turnId: options.traceTurnId!,
                  session_id: options.sessionId,
                  subsystem: "open_questions",
                  reason: error instanceof Error ? `${reason}: ${error.message}` : reason,
                });
              }
            : undefined,
      },
    );
  }

  async searchWithContextForDisclosure(
    query: string,
    options: DisclosureRetrievalOptions = {},
  ): Promise<RetrievedContext> {
    return this.searchWithContextInternal(query, options, "disclosure");
  }

  async recallEpisodesForCognition(
    query: string,
    options: CognitionRetrievalOptions,
  ): Promise<RetrievedContext> {
    return this.searchWithContextInternal(query, options, "cognition");
  }

  // Episodes-only contract: callers get RetrievedEpisode[] and nothing else,
  // so the context lanes are skipped wholesale — see RetrievalProjection for
  // the proof and the accepted observable differences. This is the memory
  // sidecar's /memory/recall path (borg.episodic.search via the facade).
  async searchEpisodesForDisclosure(
    query: string,
    options: DisclosureRetrievalOptions = {},
  ): Promise<RetrievedEpisode[]> {
    const result = await this.searchWithContextInternal(
      query,
      options,
      "disclosure",
      "episodes-only",
    );
    return result.episodes;
  }

  private async searchWithContextInternal(
    query: string,
    options: RetrievalExecutionOptions,
    mode: RetrievalExecutionMode,
    projection: RetrievalProjection = "full",
  ): Promise<RetrievedContext> {
    if (this.tracer.enabled && options.traceTurnId !== undefined) {
      this.tracer.emit("retrieval.started", {
        turnId: options.traceTurnId,
        session_id: options.sessionId,
        query,
        options: summarizeRetrievalOptions(options),
      });
    }

    const nowMs = this.clock.now();
    const limit = Math.max(1, options.limit ?? 5);
    const recallStateContext = this.loadRecallStateContext(options, nowMs, mode);
    const warmRecallEvidence =
      recallStateContext === null
        ? []
        : await this.rehydrateRecallStateEvidence(recallStateContext, options, nowMs, mode);
    let scoringFeatures = options.scoringFeatures;

    if (scoringFeatures === undefined) {
      try {
        scoringFeatures = await buildRetrievalScoringFeatures({
          embeddingClient: this.options.embeddingClient,
          goalDescriptions: options.goalDescriptions ?? [],
          primaryGoalDescription: options.primaryGoalDescription,
          activeValues: options.activeValues ?? [],
        });
      } catch (error) {
        if (this.tracer.enabled && options.traceTurnId !== undefined) {
          this.tracer.emit("retrieval.degraded", {
            turnId: options.traceTurnId,
            session_id: options.sessionId,
            subsystem: "scoring_features",
            reason: error instanceof Error ? error.message : String(error),
          });
        }

        scoringFeatures = {
          goalVectors: [],
          valueVectors: [],
        };
      }
    }

    const intents = await this.buildRecallIntents(query, options);
    const episodeCandidates = await this.collectEpisodicEvidenceCandidates(
      intents,
      options,
      scoringFeatures,
      nowMs,
      limit,
      mode,
    );
    this.emitIntentCandidateTrace(intents, episodeCandidates, options);
    const citationResolver = this.createCitationResolver();
    const citationEntries = await citationResolver.resolveCitationEntries(
      episodeCandidates.flatMap((item) => item.candidate.episode.source_stream_ids),
    );
    // Context lanes (semantic, open-question, image-perception, and — below —
    // commitment-evidence) cannot contribute episodes to projectEpisodes(), so
    // for episodes-only callers they are skipped entirely. Empty results flow
    // through the pool/projections unchanged. See RetrievalProjection.
    const collectContextLanes = projection !== "episodes-only";
    const semanticRetrievals = collectContextLanes
      ? await this.collectSemanticRetrievals(intents, options, mode)
      : [];
    const semantic = mergeSemanticRetrievals(semanticRetrievals.map((item) => item.semantic));
    const openQuestions = collectContextLanes
      ? await this.collectOpenQuestions(intents, semantic, options)
      : [];
    const imagePerceptions = collectContextLanes
      ? await this.collectImagePerceptionEvidenceWithDisclosureMode(intents, options, mode)
      : [];
    const episodeEvidenceSources = episodeCandidates.map((item) => ({
      evidence: episodeCandidateToEvidence(item),
      item,
    }));
    const episodeEvidence = episodeEvidenceSources.map((item) => item.evidence);
    const semanticEvidence = semanticRetrievals.flatMap((item) =>
      semanticRetrievalToEvidence(item.semantic, item.intent),
    );
    const openQuestionEvidenceSources = openQuestions.map((item) => ({
      evidence: openQuestionToEvidence(item.question, item.intent, item.score),
      item,
    }));
    const openQuestionEvidence = openQuestionEvidenceSources.map((item) => item.evidence);
    const imagePerceptionEvidence = imagePerceptions.map((item) =>
      imagePerceptionToEvidence(item, nowMs),
    );
    const commitmentEvidence = collectContextLanes
      ? await this.collectCommitmentEvidence(intents, options)
      : [];
    const rawStreamEvidence = [
      ...streamEntriesToEvidence(citationEntries, episodeCandidates),
      ...this.collectCurrentSessionRecentRawStreamEvidence(intents, options),
    ];
    const freshEvidence = [
      ...episodeEvidence,
      ...semanticEvidence,
      ...openQuestionEvidence,
      ...imagePerceptionEvidence,
      ...commitmentEvidence,
      ...rawStreamEvidence,
    ];
    const evidencePool: EvidencePool = {
      intents,
      items: rankEvidenceItems([...freshEvidence, ...warmRecallEvidence]),
    };
    const episodeProjectionSources = new Map<string, EpisodeProjectionSource>(
      episodeEvidenceSources.map(({ evidence, item }) => [
        evidence.id,
        {
          candidate: item.candidate,
          score: item.score,
          citationChain: () =>
            citationResolver.resolveCitationChainFromMap(
              item.candidate.episode.source_stream_ids,
              citationEntries,
              options.traceTurnId,
            ),
        },
      ]),
    );
    const episodeProjection = projectEpisodes(evidencePool, episodeProjectionSources, {
      limit,
      mmrLambda: options.mmrLambda ?? this.mmrLambda,
      exactTermReservedSlots: this.lexicalFusionEnabled ? EXACT_TERM_RESERVED_SLOTS : 0,
    });
    const semanticProjection = projectSemantic(evidencePool, toRetrievedSemantic(semantic));
    const openQuestionProjection = projectOpenQuestions(
      evidencePool,
      new Map(
        openQuestionEvidenceSources.map(({ evidence, item }) => [evidence.id, item.question]),
      ),
    );

    // Counted set = the MMR projection, not the evidence pool. Every episode
    // candidate is already in `evidencePool.items` as a source_type=episode
    // item, and that pool -- not this list -- is what
    // summarizeRetrievedEvidence() renders into the prompt. So an episode can
    // be shown to the model on many turns while incrementing retrieval_count on
    // none of them; heat, decay, the curator and the associator all read the
    // narrower number.
    if (options.recordRetrieval !== false) {
      for (const result of episodeProjection.episodes) {
        this.options.episodicRepository.recordRetrieval(result.episode.id, nowMs, result.score);
      }
    }

    const context = assembleRetrievedContext({
      episodes: episodeProjection.episodes,
      semantic: semanticProjection,
      openQuestions: openQuestionProjection,
      evidence: evidencePool.items,
      recallIntents: intents,
      contradictionPresent:
        semanticProjection.contradiction_hits.length > 0 ||
        semanticProjection.contradicts.length > 0,
      nowMs,
    });

    if (recallStateContext !== null && options.recordRetrieval !== false) {
      this.options.recallStateRepository?.save(
        this.refreshRecallState(recallStateContext, {
          freshEvidence,
          warmRecallEvidence,
          renderedEvidence: evidencePool.items,
          selectedEvidence: episodeProjection.selectedEvidence,
          nowMs,
        }),
      );
    }

    if (this.tracer.enabled && options.traceTurnId !== undefined) {
      this.tracer.emit("retrieval.completed", {
        turnId: options.traceTurnId,
        session_id: options.sessionId,
        episodeCount: context.episodes.length,
        semanticHits: countSemanticHits(context.semantic),
        asOf: options.asOf ?? null,
        confidence: context.confidence,
      });
    }

    return context;
  }

  private loadRecallStateContext(
    options: RetrievalExecutionOptions,
    nowMs: number,
    mode: RetrievalExecutionMode,
  ): RecallStateTurnContext | null {
    if (this.options.recallStateRepository === undefined) {
      return null;
    }

    const scopeKey = recallStateScopeKey(options, mode);
    const loaded =
      this.options.recallStateRepository.load(scopeKey) ??
      createEmptyRecallState({
        scopeKey,
        nowMs,
        ttlTurns: this.options.recallStateTtlTurns ?? DEFAULT_RECALL_STATE_TTL_TURNS,
      });

    return {
      state: loaded,
      scopeKey,
      turn: resolveRecallStateTurn(loaded, options.turnCounter),
    };
  }

  private async rehydrateRecallStateEvidence(
    context: RecallStateTurnContext,
    options: RetrievalExecutionOptions,
    nowMs: number,
    mode: RetrievalExecutionMode,
  ): Promise<EvidenceItem[]> {
    const evidence: EvidenceItem[] = [];
    const maxWarmEvidenceRendered = this.maxWarmEvidenceRendered();
    const currentTurnAttachmentIds = new Set(options.currentTurnAttachmentIds ?? []);

    if (maxWarmEvidenceRendered <= 0) {
      return evidence;
    }

    const activeHandles =
      currentTurnAttachmentIds.size === 0
        ? context.state.activeHandles
        : context.state.activeHandles.filter((stateHandle) => {
            const handle = normalizeRecallEvidenceHandle(stateHandle.handle);

            return (
              handle.source !== "image_perception" ||
              !currentTurnAttachmentIds.has(handle.attachmentId)
            );
          });
    const candidates = selectWarmRecallCandidates(
      activeHandles,
      context.state,
      context.turn,
      maxWarmEvidenceRendered,
    );

    for (const { handle, stateHandle } of candidates) {
      const item = await this.rehydrateRecallHandle(
        handle,
        {
          ...stateHandle,
          reinforcementCount: effectiveRecallStateReinforcementCount(stateHandle, context.turn),
        },
        options,
        nowMs,
        mode,
      );

      if (item !== null) {
        evidence.push(item);

        if (evidence.length >= maxWarmEvidenceRendered) {
          break;
        }
      }
    }

    return evidence;
  }

  private async rehydrateRecallHandle(
    handle: RecallEvidenceHandle,
    stateHandle: RecallStateHandle,
    options: RetrievalExecutionOptions,
    nowMs: number,
    mode: RetrievalExecutionMode,
  ): Promise<EvidenceItem | null> {
    if (handle.source === "episode") {
      return this.rehydrateEpisodeHandle(handle, stateHandle, options, mode);
    }

    if (handle.source === "raw_stream") {
      return this.rehydrateRawStreamHandle(handle, stateHandle, options, mode);
    }

    if (handle.source === "semantic_node") {
      return this.rehydrateSemanticNodeHandle(handle, stateHandle, options, mode);
    }

    if (handle.source === "semantic_edge") {
      return this.rehydrateSemanticEdgeHandle(handle, stateHandle, options, mode);
    }

    if (handle.source === "commitment") {
      return this.rehydrateCommitmentHandle(handle, stateHandle, nowMs);
    }

    if (handle.source === "image_perception") {
      return this.rehydrateImagePerceptionHandle(handle, stateHandle, options, nowMs, mode);
    }

    return this.rehydrateOpenQuestionHandle(handle, stateHandle);
  }

  private rehydrateImagePerceptionHandle(
    handle: Extract<RecallEvidenceHandle, { source: "image_perception" }>,
    stateHandle: RecallStateHandle,
    options: RetrievalExecutionOptions,
    nowMs: number,
    mode: RetrievalExecutionMode,
  ): EvidenceItem | null {
    const record = this.options.imagePerceptionRepository?.get(handle.perceptionId);

    if (record === undefined || record === null || !record.active) {
      return null;
    }

    if ((options.currentTurnAttachmentIds ?? []).includes(record.attachment_id)) {
      return null;
    }

    if (
      mode === "disclosure" &&
      options.crossAudience !== true &&
      (record.audience_entity_id === null ||
        record.audience_entity_id !== (options.audienceEntityId ?? null))
    ) {
      return null;
    }

    return imagePerceptionToEvidence(
      {
        intent: {
          id: WARM_RECALL_INTENT_ID,
          kind: "recent",
          query: record.caption,
          terms: [],
          priority: 0.4,
          source: "recency",
        },
        hit: {
          record,
          similarity: warmRecallScore(stateHandle),
        },
      },
      nowMs,
    );
  }

  private async rehydrateEpisodeHandle(
    handle: Extract<RecallEvidenceHandle, { source: "episode" }>,
    stateHandle: RecallStateHandle,
    options: RetrievalExecutionOptions,
    mode: RetrievalExecutionMode,
  ): Promise<EvidenceItem | null> {
    const episode = await this.options.episodicRepository.get(handle.episodeId);

    if (episode === null) {
      return null;
    }

    if (
      mode === "disclosure" &&
      !isEpisodeVisibleToRetrievalCapability(episode, episodeVisibilityOptions(options))
    ) {
      return null;
    }

    return {
      id: `warm_recall_episode_${episode.id}`,
      source: "warm_recall",
      text: `${episode.title}: ${episode.narrative}`,
      provenance: {
        episodeId: episode.id,
        streamIds: [...episode.source_stream_ids],
      },
      recallIntentId: WARM_RECALL_INTENT_ID,
      matchedTerms: [],
      score: warmRecallScore(stateHandle),
      scoreBreakdown: {
        provenance: 1,
        recency: computeRecencyEvidenceScore(episode.updated_at),
      },
      disclosureLabel: memoryDisclosureLabelFromEpisodeAccess(episode),
    };
  }

  private async rehydrateRawStreamHandle(
    handle: Extract<RecallEvidenceHandle, { source: "raw_stream" }>,
    stateHandle: RecallStateHandle,
    options: RetrievalExecutionOptions,
    mode: RetrievalExecutionMode,
  ): Promise<EvidenceItem | null> {
    let disclosureLabel: MemoryDisclosureLabel = unknownMemoryDisclosureLabel();

    if (handle.parentEpisodeId !== undefined) {
      const parent = await this.options.episodicRepository.get(handle.parentEpisodeId);

      if (
        parent === null ||
        (mode === "disclosure" &&
          !isEpisodeVisibleToRetrievalCapability(parent, episodeVisibilityOptions(options)))
      ) {
        return null;
      }

      disclosureLabel = memoryDisclosureLabelFromEpisodeAccess(parent);
    }

    const adapter = new RawStreamAdapter({
      dataDir: this.options.dataDir,
      entryIndex: this.options.entryIndex,
    });
    const entries = await adapter.resolveSourceIds(handle.streamIds);
    const orderedEntries = handle.streamIds
      .map((streamId) => entries.get(streamId))
      .filter((entry): entry is StreamEntry => entry !== undefined);

    if (orderedEntries.length === 0) {
      return null;
    }

    return {
      id: `warm_recall_raw_stream_${handle.streamIds.join("_")}`,
      source: "warm_recall",
      text: orderedEntries.map((entry) => streamEntryContentToText(entry)).join("\n"),
      provenance: {
        streamIds: [...handle.streamIds],
        ...(handle.parentEpisodeId === undefined
          ? {}
          : { parentEpisodeId: handle.parentEpisodeId }),
      },
      recallIntentId: WARM_RECALL_INTENT_ID,
      matchedTerms: [],
      score: warmRecallScore(stateHandle),
      scoreBreakdown: {
        provenance: 1,
      },
      disclosureLabel,
    };
  }

  private async rehydrateSemanticNodeHandle(
    handle: Extract<RecallEvidenceHandle, { source: "semantic_node" }>,
    stateHandle: RecallStateHandle,
    options: RetrievalExecutionOptions,
    mode: RetrievalExecutionMode,
  ): Promise<EvidenceItem | null> {
    const node = await this.options.semanticNodeRepository?.get(handle.nodeId);

    if (node === undefined || node === null || node.archived) {
      return null;
    }

    const source = await resolveSemanticDisclosureSourceAdapter({
      episodicRepository: this.options.episodicRepository,
      sourceEpisodeIds: node.source_episode_ids,
      mode,
      visibility: episodeVisibilityOptions(options),
    });

    if (source === null) {
      return null;
    }

    return {
      id: `warm_recall_semantic_node_${node.id}`,
      source: "warm_recall",
      text: `${node.label}: ${node.description}`,
      provenance: {
        nodeId: node.id,
      },
      recallIntentId: WARM_RECALL_INTENT_ID,
      matchedTerms: [],
      score: warmRecallScore(stateHandle),
      scoreBreakdown: {
        provenance: 1,
      },
      source_episode_ids: source.admittedSourceEpisodeIds,
      ...semanticSourceAdapterPartialEvidenceFields(source),
      disclosureLabel: source.disclosureLabel,
    };
  }

  private async rehydrateSemanticEdgeHandle(
    handle: Extract<RecallEvidenceHandle, { source: "semantic_edge" }>,
    stateHandle: RecallStateHandle,
    options: RetrievalExecutionOptions,
    mode: RetrievalExecutionMode,
  ): Promise<EvidenceItem | null> {
    const edge = this.options.semanticEdgeRepository?.getEdge(handle.edgeId);

    if (
      edge === undefined ||
      edge === null ||
      edge.valid_to !== null ||
      edge.invalidated_at !== null
    ) {
      return null;
    }

    const nodeId = handle.nodeId ?? edge.to_node_id;
    const node = await this.options.semanticNodeRepository?.get(nodeId);

    if (node === undefined || node === null || node.archived) {
      return null;
    }

    const nodeSource = await resolveSemanticDisclosureSourceAdapter({
      episodicRepository: this.options.episodicRepository,
      sourceEpisodeIds: node.source_episode_ids,
      mode,
      visibility: episodeVisibilityOptions(options),
    });
    const edgeSource = await resolveSemanticDisclosureSourceAdapter({
      episodicRepository: this.options.episodicRepository,
      sourceEpisodeIds: edge.evidence_episode_ids,
      mode,
      visibility: episodeVisibilityOptions(options),
    });

    if (nodeSource === null || edgeSource === null) {
      return null;
    }
    const semanticEdgeSourceEpisodeIds = [
      ...new Set([...nodeSource.admittedSourceEpisodeIds, ...edgeSource.admittedSourceEpisodeIds]),
    ];
    const partialSource = edgeSource.partial ? edgeSource : nodeSource.partial ? nodeSource : null;

    return {
      id: `warm_recall_semantic_edge_${edge.id}`,
      source: "warm_recall",
      text: `${node.label}: ${node.description}`,
      provenance: {
        edgeId: edge.id,
        nodeId: node.id,
      },
      recallIntentId: WARM_RECALL_INTENT_ID,
      matchedTerms: [],
      score: warmRecallScore(stateHandle),
      scoreBreakdown: {
        provenance: 1,
      },
      source_episode_ids: semanticEdgeSourceEpisodeIds,
      ...(partialSource === null ? {} : semanticSourceAdapterPartialEvidenceFields(partialSource)),
      disclosureLabel: combineMemoryDisclosureLabels([
        nodeSource.disclosureLabel,
        edgeSource.disclosureLabel,
      ]),
    };
  }

  private rehydrateCommitmentHandle(
    handle: Extract<RecallEvidenceHandle, { source: "commitment" }>,
    stateHandle: RecallStateHandle,
    nowMs: number,
  ): EvidenceItem | null {
    const commitment = this.options.commitmentRepository?.get(handle.commitmentId);

    if (commitment === undefined || commitment === null) {
      return null;
    }

    const activeVisible = this.options.commitmentRepository
      ?.list({
        activeOnly: true,
        nowMs,
      })
      .some((item) => item.id === commitment.id);

    if (activeVisible !== true) {
      return null;
    }

    return {
      id: `warm_recall_commitment_${commitment.id}`,
      source: "warm_recall",
      text: `${commitment.type}: ${commitment.directive}`,
      provenance: {
        commitmentId: commitment.id,
      },
      recallIntentId: WARM_RECALL_INTENT_ID,
      matchedTerms: [],
      score: warmRecallScore(stateHandle),
      scoreBreakdown: {
        provenance: 1,
      },
      disclosureLabel: commitmentMemoryDisclosureLabel(commitment),
      commitment_enforcement_class: effectiveCommitmentEnforcementClass(commitment),
      commitment_critical_domain: effectiveCommitmentCriticalDomain(commitment),
      commitment_directive_chars: commitment.directive.length,
    };
  }

  private rehydrateOpenQuestionHandle(
    handle: Extract<RecallEvidenceHandle, { source: "open_question" }>,
    stateHandle: RecallStateHandle,
  ): EvidenceItem | null {
    const question = this.options.openQuestionsRepository?.get(handle.openQuestionId);

    if (question === undefined || question === null || question.status !== "open") {
      return null;
    }

    return {
      id: `warm_recall_open_question_${question.id}`,
      source: "warm_recall",
      text: question.question,
      provenance: {
        openQuestionId: question.id,
      },
      recallIntentId: WARM_RECALL_INTENT_ID,
      matchedTerms: [],
      score: warmRecallScore(stateHandle),
      scoreBreakdown: {
        provenance: 1,
      },
      disclosureLabel: openQuestionMemoryDisclosureLabel(question),
    };
  }

  private refreshRecallState(
    context: RecallStateTurnContext,
    input: {
      freshEvidence: readonly EvidenceItem[];
      warmRecallEvidence: readonly EvidenceItem[];
      renderedEvidence: readonly EvidenceItem[];
      selectedEvidence: readonly EvidenceItem[];
      nowMs: number;
    },
  ): RecallState {
    const ttlTurns = context.state.ttlTurns;
    const freshHandles = collectEvidenceHandles(input.freshEvidence);
    const warmRecallHandles = collectEvidenceHandles(input.warmRecallEvidence);
    const renderedHandleKeys = new Set(collectEvidenceHandles(input.renderedEvidence).keys());
    const selectedHandlePriorities = new Map(
      [...collectEvidenceHandles(input.selectedEvidence).keys()].map((key, index) => [key, index]),
    );
    const existingByKey = new Map(
      context.state.activeHandles.map((stateHandle) => [
        recallEvidenceHandleKey(stateHandle.handle),
        stateHandle,
      ]),
    );
    const nextHandles = new Map<string, RecallStateHandle>();
    const nextSuppressedHandles = pruneSuppressedRecallHandles(
      context.state.suppressedHandles,
      context.turn,
    );

    for (const [key, stateHandle] of existingByKey) {
      if (stateHandle.expiresAfterTurn < context.turn) {
        continue;
      }

      nextHandles.set(key, {
        ...stateHandle,
        lastRenderedTurn: renderedHandleKeys.has(key) ? context.turn : stateHandle.lastRenderedTurn,
        reinforcementCount: Math.min(
          DEFAULT_RECALL_STATE_REINFORCEMENT_CAP,
          stateHandle.reinforcementCount,
        ),
      });
    }

    for (const key of freshHandles.keys()) {
      delete nextSuppressedHandles[key];
    }

    for (const [key, freshHandle] of freshHandles) {
      const stateHandle = nextHandles.get(key);

      if (stateHandle === undefined) {
        continue;
      }

      nextHandles.set(key, {
        ...stateHandle,
        handle: freshHandle,
        lastSeenTurn: context.turn,
        lastRenderedTurn: renderedHandleKeys.has(key) ? context.turn : stateHandle.lastRenderedTurn,
        expiresAfterTurn: context.turn + ttlTurns,
        reinforcementCount: Math.min(
          DEFAULT_RECALL_STATE_REINFORCEMENT_CAP,
          effectiveRecallStateReinforcementCount(stateHandle, context.turn) + 1,
        ),
      });
    }

    const newFreshHandles = [...freshHandles].filter(([key]) => !nextHandles.has(key));
    const admittedNewHandles = rankFreshAdmissions(newFreshHandles, selectedHandlePriorities).slice(
      0,
      this.maxNewHandlesPerTurn(),
    );

    for (const [key, handle] of admittedNewHandles) {
      nextHandles.set(
        key,
        createRecallStateHandle({
          handle,
          turn: context.turn,
          ttlTurns,
          rendered: renderedHandleKeys.has(key),
        }),
      );
    }

    const warmSuppressionTurns = this.warmSuppressionTurns();

    if (warmSuppressionTurns > 0) {
      for (const key of renderedHandleKeys) {
        if (freshHandles.has(key) || !warmRecallHandles.has(key) || !nextHandles.has(key)) {
          continue;
        }

        nextSuppressedHandles[key] = context.turn + warmSuppressionTurns;
      }
    }

    const activeHandles = capRecallStateHandles(
      [...nextHandles.values()],
      this.maxActiveHandles(),
      context.turn,
      selectedHandlePriorities,
    ).sort((left, right) => compareRecallStateRetentionPriority(left, right, context.turn));

    return {
      scopeKey: context.scopeKey,
      activeHandles,
      suppressedHandles: nextSuppressedHandles,
      lastRefreshTurn: context.turn,
      updatedAt: input.nowMs,
      ttlTurns,
    };
  }

  private warmSuppressionTurns(): number {
    return normalizeRecallStateBound(
      this.options.recallStateWarmSuppressionTurns,
      DEFAULT_RECALL_STATE_WARM_SUPPRESSION_TURNS,
    );
  }

  private maxActiveHandles(): number {
    return normalizeRecallStateBound(
      this.options.recallStateMaxActiveHandles,
      DEFAULT_RECALL_STATE_MAX_ACTIVE_HANDLES,
    );
  }

  private maxNewHandlesPerTurn(): number {
    return normalizeRecallStateBound(
      this.options.recallStateMaxNewHandlesPerTurn,
      DEFAULT_RECALL_STATE_MAX_NEW_HANDLES_PER_TURN,
    );
  }

  private maxWarmEvidenceRendered(): number {
    return normalizeRecallStateBound(
      this.options.recallStateMaxWarmEvidenceRendered,
      DEFAULT_RECALL_STATE_MAX_WARM_EVIDENCE_RENDERED,
    );
  }

  private async buildRecallIntents(
    query: string,
    options: RetrievalExecutionOptions,
  ): Promise<RecallIntent[]> {
    const intents: RecallIntent[] = [
      {
        id: "recall_raw_text_0",
        kind: "raw_text",
        query,
        terms: [],
        priority: 100,
        source: "raw-user-message",
      },
    ];
    const expansion = await this.tryExpandRecall(query, options);

    intents.push(...expansion.facetIntents);

    const knownTerms = dedupeTermInputs([
      ...expansion.namedTerms.map((term) => ({ term, source: "llm-expansion" as const })),
      ...(options.entityTerms ?? []).map((term) => ({
        term,
        source: "perception-entities" as const,
      })),
      ...(options.audienceTerms ?? []).map((term) => ({
        term,
        source: "audience-aliases" as const,
      })),
    ]);

    for (const [index, item] of knownTerms.entries()) {
      intents.push({
        id: `recall_known_term_${index}`,
        kind: "known_term",
        query: item.term,
        terms: [item.term],
        priority: 90,
        source: item.source,
      });
    }

    const timeIntentRange = resolveTimeSignals(options).scoringRange;

    if (timeIntentRange !== null) {
      intents.push({
        id: "recall_time_0",
        kind: "time",
        query: options.temporalCue?.label ?? "time range",
        terms: [],
        timeRange: timeIntentRange,
        strictTime: options.strictTimeRange === true,
        priority: 70,
        source: "temporal-cue",
      });
    }

    intents.push({
      id: "recall_recent_0",
      kind: "recent",
      query: "recent memory",
      terms: [],
      priority: 10,
      source: "recency",
    });

    return intents;
  }

  private async tryExpandRecall(
    query: string,
    options: RetrievalExecutionOptions,
  ): Promise<ExpansionOutcome> {
    if (this.options.llmClient === undefined) {
      return {
        succeeded: false,
        facetIntents: [],
        namedTerms: [],
      };
    }

    try {
      const expansion = await expandRecall({
        llmClient: this.options.llmClient,
        model: this.options.recallExpansionModel ?? DEFAULT_RECALL_EXPANSION_MODEL,
        userMessage: query,
        tracer: this.tracer,
        turnId: options.traceTurnId,
        sessionId: options.sessionId,
      });
      const namedTerms = dedupeStrings(expansion.named_terms);
      const facetIntents = expansion.facets.map((facet, index): RecallIntent => {
        const kind: RecallIntentKind = facet.kind;

        return {
          id: `recall_${kind}_${index}`,
          kind,
          query: facet.query,
          terms: [],
          priority: 60 + facet.priority * 20,
          source: "llm-expansion",
        };
      });

      return {
        succeeded: true,
        facetIntents,
        namedTerms,
      };
    } catch (error) {
      if (this.tracer.enabled && options.traceTurnId !== undefined) {
        this.tracer.emit("retrieval.degraded", {
          turnId: options.traceTurnId,
          session_id: options.sessionId,
          subsystem: "recall_expansion",
          reason: error instanceof Error ? error.message : String(error),
        });
      }

      return {
        succeeded: false,
        facetIntents: [],
        namedTerms: [],
      };
    }
  }

  private async collectEpisodicEvidenceCandidates(
    intents: readonly RecallIntent[],
    options: RetrievalExecutionOptions,
    scoringFeatures: RetrievalScoringFeatures,
    nowMs: number,
    limit: number,
    mode: RetrievalExecutionMode,
  ): Promise<EpisodeEvidenceCandidate[]> {
    const rawCandidates = (
      await Promise.all(
        intents.map((intent) =>
          this.collectEpisodicCandidatesForDisclosureModeIntent(intent, options, limit, mode),
        ),
      )
    ).flat();
    const participantEntityIds = this.resolveParticipantEntityIds(
      rawCandidates,
      rankingAudienceEntityId(options),
    );

    return rawCandidates.map((entry) => {
      const score = this.scoreEpisodeCandidateForIntent(
        entry,
        options,
        scoringFeatures,
        nowMs,
        participantEntityIds,
      );

      return {
        ...entry,
        score,
      };
    });
  }

  private emitIntentCandidateTrace(
    intents: readonly RecallIntent[],
    episodeCandidates: readonly EpisodeEvidenceCandidate[],
    options: RetrievalExecutionOptions,
  ): void {
    if (!this.tracer.enabled || options.traceTurnId === undefined) {
      return;
    }

    const candidatesByIntent = new Map<string, EpisodeEvidenceCandidate[]>();

    for (const candidate of episodeCandidates) {
      const candidates = candidatesByIntent.get(candidate.intent.id) ?? [];
      if (candidates.length === 0) {
        candidatesByIntent.set(candidate.intent.id, candidates);
      }
      candidates.push(candidate);
    }

    for (const intent of intents) {
      const candidates = candidatesByIntent.get(intent.id) ?? [];

      this.tracer.emit("retrieval.intent_candidates", {
        turnId: options.traceTurnId,
        session_id: options.sessionId,
        intent_id: intent.id,
        intent_kind: intent.kind,
        intent_source: intent.source,
        intent_priority: intent.priority,
        candidate_count: candidates.length,
        candidates: candidates.map((candidate) => ({
          episode_id: candidate.candidate.episode.id,
          score: candidate.score.score,
          vector_score: candidate.candidate.similarity,
        })),
        ...(this.tracer.includePayloads
          ? {
              intent_query: intent.query,
              matched_terms_by_candidate: candidates.map((candidate) => ({
                episode_id: candidate.candidate.episode.id,
                matched_terms: [...candidate.matchedTerms],
              })),
              candidate_texts: candidates.map((candidate) => ({
                episode_id: candidate.candidate.episode.id,
                title: candidate.candidate.episode.title,
              })),
            }
          : {}),
      });
    }
  }

  private async collectEpisodicCandidatesForDisclosureModeIntent(
    intent: RecallIntent,
    options: RetrievalExecutionOptions,
    limit: number,
    mode: RetrievalExecutionMode,
  ): Promise<RawEpisodeEvidenceCandidate[]> {
    const vectorBudget = Math.max(limit * 2, 12);
    const indexedBudget = Math.max(limit * 2, 8);
    const recentBudget = Math.max(limit, 4);

    if (intent.kind === "raw_text" || intent.kind === "topic" || intent.kind === "relationship") {
      const intentVector = await this.options.embeddingClient.embed(intent.query);
      const candidates =
        mode === "cognition"
          ? await this.options.episodicRepository.recallByVectorForCognition(intentVector, {
              ...episodeCognitionRecallOptions(options),
              limit: vectorBudget,
            })
          : await this.options.episodicRepository.searchByVectorForDisclosure(intentVector, {
              ...episodeSearchOptions(options),
              limit: vectorBudget,
            });

      return candidates.map((candidate) => ({
        intent,
        candidate,
        matchedTerms: [],
      }));
    }

    if (intent.kind === "known_term") {
      const [indexed, lexical] =
        mode === "cognition"
          ? await Promise.all([
              this.options.episodicRepository.recallByParticipantsOrTagsForCognition(intent.terms, {
                limit: indexedBudget,
              }),
              this.lexicalFusionEnabled
                ? this.options.episodicRepository.recallByLexicalTermsForCognition(intent.terms, {
                    limit: indexedBudget,
                  })
                : Promise.resolve([]),
            ])
          : await Promise.all([
              this.options.episodicRepository.searchByParticipantsOrTagsForDisclosure(
                intent.terms,
                {
                  ...episodeVisibilityOptions(options),
                  limit: indexedBudget,
                },
              ),
              this.lexicalFusionEnabled
                ? this.options.episodicRepository.searchByLexicalTermsForDisclosure(intent.terms, {
                    ...episodeVisibilityOptions(options),
                    limit: indexedBudget,
                  })
                : Promise.resolve([]),
            ]);

      return mergeRawEpisodeCandidates(
        [...indexed, ...lexical].map((candidate) => ({
          intent,
          candidate,
          matchedTerms: [...intent.terms],
        })),
      );
    }

    if (intent.kind === "time" && intent.timeRange !== undefined) {
      const candidates =
        mode === "cognition"
          ? await this.options.episodicRepository.recallByTimeRangeForCognition(intent.timeRange, {
              limit: indexedBudget,
            })
          : await this.options.episodicRepository.searchByTimeRangeForDisclosure(intent.timeRange, {
              ...episodeVisibilityOptions(options),
              limit: indexedBudget,
            });

      return candidates.map((candidate) => ({
        intent,
        candidate,
        matchedTerms: [],
      }));
    }

    if (intent.kind === "recent") {
      const recentLimit = Math.max(1, Math.ceil(recentBudget / 2));
      const heatLimit = Math.max(1, recentBudget - recentLimit);
      const [recent, hottest] =
        mode === "cognition"
          ? await Promise.all([
              this.options.episodicRepository.listRecentForCognition({
                limit: recentLimit,
              }),
              this.options.episodicRepository.listHottestForCognition({
                limit: heatLimit,
              }),
            ])
          : await Promise.all([
              this.options.episodicRepository.listRecentForDisclosure({
                ...episodeVisibilityOptions(options),
                limit: recentLimit,
              }),
              this.options.episodicRepository.listHottestForDisclosure({
                ...episodeVisibilityOptions(options),
                limit: heatLimit,
              }),
            ]);

      return mergeRawEpisodeCandidates([
        ...recent.map((candidate) => ({
          intent,
          candidate,
          matchedTerms: [],
        })),
        ...hottest.map((candidate) => ({
          intent,
          candidate,
          matchedTerms: [],
        })),
      ]);
    }

    return [];
  }

  private scoreEpisodeCandidateForIntent(
    entry: RawEpisodeEvidenceCandidate,
    options: RetrievalExecutionOptions,
    scoringFeatures: RetrievalScoringFeatures,
    nowMs: number,
    participantEntityIds: ParticipantEntityResolutionLookup | undefined,
  ): EpisodeScoreDetails {
    const intentTimeRange = entry.intent.kind === "time" ? (entry.intent.timeRange ?? null) : null;
    const score = scoreCandidate(
      entry.candidate,
      {
        ...options,
        audienceEntityId: rankingAudienceEntityId(options),
        scoringFeatures,
        entityTerms: entry.intent.kind === "known_term" ? entry.intent.terms : [],
        ...(participantEntityIds === undefined ? {} : { participantEntityIds }),
      },
      nowMs,
      intentTimeRange,
      this.scoringDefaults(),
    );
    const exactBoost = entry.intent.kind === "known_term" ? KNOWN_TERM_INTENT_SCORE_BOOST : 0;
    const recencyBoost = entry.intent.kind === "recent" ? RECENT_INTENT_SCORE_BOOST : 0;
    const rawScore = score.score + exactBoost + recencyBoost;

    return {
      ...score,
      score: clamp(rawScore, 0, 1),
      rawScore,
    };
  }

  private async collectSemanticRetrievals(
    intents: readonly RecallIntent[],
    options: RetrievalExecutionOptions,
    mode: RetrievalExecutionMode,
  ): Promise<SemanticEvidenceCandidate[]> {
    const relevantIntents = intents.filter((intent) => isSemanticIntentKind(intent.kind));

    const results = await mapWithConcurrency(
      relevantIntents,
      RETRIEVAL_FANOUT_CONCURRENCY,
      async (intent): Promise<SemanticEvidenceCandidate | null> => {
        try {
          const intentVector = await this.options.embeddingClient.embed(intent.query);
          const resolveSemanticContextForMode =
            mode === "disclosure"
              ? resolveSemanticContextForDisclosure
              : resolveSemanticContextForCognition;
          const semantic = await resolveSemanticContextForMode(
            intent.query,
            {
              ...options,
              queryVector: intentVector,
              exactTerms: intent.kind === "known_term" ? intent.terms : [],
              underReviewMultiplier:
                options.underReviewMultiplier ?? this.options.semanticUnderReviewMultiplier,
              statusMultipliers:
                options.statusMultipliers ?? this.options.semanticStatusMultipliers,
              overfetchMultiplier:
                options.semanticOverfetchMultiplier ?? this.options.semanticOverfetchMultiplier,
            },
            {
              embeddingClient: this.options.embeddingClient,
              episodicRepository: this.options.episodicRepository,
              semanticNodeRepository: this.options.semanticNodeRepository,
              semanticGraph: this.options.semanticGraph,
              reviewQueueRepository: this.options.reviewQueueRepository,
            },
          );

          return {
            intent,
            semantic,
          };
        } catch (error) {
          this.emitRetrievalDegraded(options, "semantic", error);
          return null;
        }
      },
    );

    return results.filter((item): item is SemanticEvidenceCandidate => item !== null);
  }

  private async collectOpenQuestions(
    intents: readonly RecallIntent[],
    semantic: ResolvedSemanticRetrieval,
    options: RetrievalExecutionOptions,
  ): Promise<OpenQuestionEvidenceCandidate[]> {
    const shouldInclude = options.includeOpenQuestions === true;
    const relevantIntents = intents.filter(
      (intent) =>
        intent.kind === "open_question" || (shouldInclude && isSemanticIntentKind(intent.kind)),
    );
    const byId = new Map<string, OpenQuestionEvidenceCandidate>();

    const results = await mapWithConcurrency(
      relevantIntents,
      RETRIEVAL_FANOUT_CONCURRENCY,
      async (intent): Promise<OpenQuestionEvidenceCandidate[]> => {
        try {
          const questions = await this.retrieveOpenQuestionsForQuery(intent.query, {
            relatedSemanticNodeIds: semantic.matchedNodeIds,
            limit: options.openQuestionsLimit,
            traceTurnId: options.traceTurnId,
            sessionId: options.sessionId,
          });

          return questions.map((question) => ({
            intent,
            question,
            score: question.urgency + intent.priority / 100,
          }));
        } catch (error) {
          this.emitRetrievalDegraded(options, "open_questions", error);
          return [];
        }
      },
    );

    for (const item of results.flat()) {
      const current = byId.get(item.question.id);

      if (current === undefined || item.score > current.score) {
        byId.set(item.question.id, item);
      }
    }

    return [...byId.values()].sort(
      (left, right) =>
        right.score - left.score ||
        right.question.urgency - left.question.urgency ||
        right.question.last_touched - left.question.last_touched,
    );
  }

  private async collectImagePerceptionEvidenceWithDisclosureMode(
    intents: readonly RecallIntent[],
    options: RetrievalExecutionOptions,
    mode: RetrievalExecutionMode,
  ): Promise<ImagePerceptionEvidenceCandidate[]> {
    if (this.options.imagePerceptionRepository === undefined) {
      return [];
    }

    const byId = new Map<string, ImagePerceptionEvidenceCandidate>();
    const relevantIntents = intents.filter((intent) => isSemanticIntentKind(intent.kind));
    const currentTurnAttachmentIds = new Set(options.currentTurnAttachmentIds ?? []);
    const results = await mapWithConcurrency(
      relevantIntents,
      RETRIEVAL_FANOUT_CONCURRENCY,
      async (intent): Promise<ImagePerceptionEvidenceCandidate[]> => {
        try {
          const vector = await this.options.embeddingClient.embed(intent.query);
          const imageRecallLimit = Math.max(1, options.limit ?? 5);
          const imageRecallInput = {
            vector,
            limit: imageRecallLimit + currentTurnAttachmentIds.size,
          };
          const hits =
            mode === "cognition"
              ? await this.options.imagePerceptionRepository!.recallForCognition(imageRecallInput)
              : await this.options.imagePerceptionRepository!.searchForDisclosure({
                  ...imageRecallInput,
                  audienceEntityId: options.audienceEntityId ?? null,
                  crossAudience: options.crossAudience,
                });

          return hits
            .filter((hit) => !currentTurnAttachmentIds.has(hit.record.attachment_id))
            .slice(0, imageRecallLimit)
            .map((hit) => ({
              intent,
              hit,
            }));
        } catch (error) {
          this.emitRetrievalDegraded(options, "image_perception", error);
          return [];
        }
      },
    );

    for (const item of results.flat()) {
      const current = byId.get(item.hit.record.perception_id);

      if (current === undefined || item.hit.similarity > current.hit.similarity) {
        byId.set(item.hit.record.perception_id, item);
      }
    }

    return [...byId.values()]
      .sort((left, right) => right.hit.similarity - left.hit.similarity)
      .slice(0, this.options.maxRetrievedImageRefs ?? 8);
  }

  private emitRetrievalDegraded(
    options: RetrievalExecutionOptions,
    subsystem: string,
    error: unknown,
  ): void {
    if (this.tracer.enabled && options.traceTurnId !== undefined) {
      this.tracer.emit("retrieval.degraded", {
        turnId: options.traceTurnId,
        session_id: options.sessionId,
        subsystem,
        reason: error instanceof Error ? error.message : String(error),
      });
    }
  }

  private async collectCommitmentEvidence(
    intents: readonly RecallIntent[],
    options: RetrievalExecutionOptions,
  ): Promise<EvidenceItem[]> {
    if (this.options.commitmentRepository === undefined) {
      return [];
    }

    const relevantIntents = intents.filter(
      (intent) => intent.kind === "commitment" || intent.kind === "known_term",
    );

    if (relevantIntents.length === 0) {
      return [];
    }

    const activeCommitments = this.options.commitmentRepository.list({
      activeOnly: true,
      nowMs: this.clock.now(),
    });

    if (activeCommitments.length === 0) {
      return [];
    }

    let intentVectors: Float32Array[];
    let commitmentVectors: Float32Array[];
    try {
      intentVectors = await this.options.embeddingClient.embedBatch(
        relevantIntents.map((intent) => intent.query),
      );
      commitmentVectors = await this.options.embeddingClient.embedBatch(
        activeCommitments.map((commitment) => commitment.directive),
      );
    } catch (error) {
      this.emitRetrievalDegraded(options, "commitments", error);
      return [];
    }
    const threshold =
      this.options.commitmentEvidenceSimilarityThreshold ??
      DEFAULT_COMMITMENT_EVIDENCE_SIMILARITY_THRESHOLD;
    const evidence: EvidenceItem[] = [];

    for (const [intentIndex, intent] of relevantIntents.entries()) {
      const intentVector = intentVectors[intentIndex];

      if (intentVector === undefined) {
        continue;
      }

      for (const [commitmentIndex, commitment] of activeCommitments.entries()) {
        const commitmentVector = commitmentVectors[commitmentIndex];

        if (commitmentVector === undefined) {
          continue;
        }

        const similarity = cosineSimilarity(intentVector, commitmentVector);

        if (similarity >= threshold) {
          evidence.push(commitmentToEvidence(commitment, intent, similarity));
        }
      }
    }

    return evidence;
  }

  private collectCurrentSessionRecentRawStreamEvidence(
    intents: readonly RecallIntent[],
    options: RetrievalExecutionOptions,
  ): EvidenceItem[] {
    const recentIntent = intents.find((intent) => intent.kind === "recent");

    if (recentIntent === undefined) {
      return [];
    }

    const adapter = new RawStreamAdapter({
      dataDir: this.options.dataDir,
      entryIndex: this.options.entryIndex,
    });

    // This lane is not durable/cross-session memory recall. It is a
    // current-session transcript recency bridge used to keep the live turn
    // coherent before episodic consolidation/citation hydration catches up.
    // Cross-session source-stream recall stays in streamEntriesToEvidence()
    // via retrieved episodes and carries episode-derived disclosure labels.
    return adapter
      .recent({
        limit: 3,
        ...(options.sessionId === undefined ? {} : { sessionId: options.sessionId }),
      })
      .map((entry) => streamEntryToEvidence(entry, recentIntent, "recent_raw_stream"));
  }

  async resolveSourceEntries(ids: readonly StreamEntryId[]): Promise<Map<string, StreamEntry>> {
    const citationResolver = this.createCitationResolver();
    return citationResolver.resolveCitationEntries(ids);
  }

  async getEpisode(
    id: Episode["id"],
    options: RetrievalGetEpisodeOptions = {},
  ): Promise<RetrievedEpisode | null> {
    const episode = await this.options.episodicRepository.get(id);

    if (episode === null) {
      return null;
    }

    if (!isEpisodeVisibleToRetrievalCapability(episode, options)) {
      return null;
    }

    const stats = this.options.episodicRepository.getStats(id);

    if (stats === null) {
      throw new StorageError(`Missing episode stats for ${id}`, {
        code: "EPISODE_STATS_MISSING",
      });
    }

    const nowMs = this.clock.now();
    const candidate: EpisodeSearchCandidate = {
      episode,
      stats,
      similarity: 1,
    };
    const scored = scoreCandidate(candidate, {}, nowMs, null, this.scoringDefaults());
    const citationResolver = this.createCitationResolver();
    const citationEntries = await citationResolver.resolveCitationEntries(
      episode.source_stream_ids,
    );
    const citationChain = citationResolver.resolveCitationChainFromMap(
      episode.source_stream_ids,
      citationEntries,
    );

    return buildRetrievedEpisode(
      candidate,
      {
        ...scored,
        score: 1,
        rawScore: 1,
      },
      citationChain,
    );
  }

  private scoringDefaults(): EpisodeScoreDefaults {
    const defaults: EpisodeScoreDefaults = {
      scoreWeights: this.scoreWeights,
    };

    if (this.decayOptions !== undefined) {
      defaults.decayOptions = this.decayOptions;
    }

    return defaults;
  }

  private createCitationResolver(): CitationResolver {
    const options: CitationResolverOptions = {
      dataDir: this.options.dataDir,
      tracer: this.tracer,
    };

    if (this.options.entryIndex !== undefined) {
      options.entryIndex = this.options.entryIndex;
    }

    return new CitationResolver(options);
  }

  private resolveParticipantEntityIds(
    candidates: readonly { candidate: EpisodeSearchCandidate }[],
    audienceEntityId: EntityId | null | undefined,
  ): ParticipantEntityResolutionLookup | undefined {
    if (
      audienceEntityId === null ||
      audienceEntityId === undefined ||
      this.options.entityRepository === undefined
    ) {
      return undefined;
    }

    const participantEntityIds = new Map<string, EntityId | null>();

    for (const entry of candidates) {
      for (const participant of entry.candidate.episode.participants) {
        const key = participantEntityResolutionKey(participant);

        if (key.length === 0 || participantEntityIds.has(key)) {
          continue;
        }

        participantEntityIds.set(
          key,
          parseEpisodeParticipantEntityIdTerm(participant) ??
            this.options.entityRepository.findByName(participant),
        );
      }
    }

    return participantEntityIds;
  }
}

function summarizeRetrievalOptions(options: RetrievalExecutionOptions): JsonValue {
  return {
    limit: options.limit ?? null,
    strictTimeRange: options.strictTimeRange ?? false,
    includeOpenQuestions: options.includeOpenQuestions ?? false,
    temporalCue: summarizeTemporalCue(options.temporalCue ?? null),
    attentionWeights: options.attentionWeights ?? null,
    goalCount: options.goalDescriptions?.length ?? 0,
    primaryGoalSelected: options.primaryGoalDescription !== undefined,
    activeValueCount: options.activeValues?.length ?? 0,
    audienceTermCount: options.audienceTerms?.length ?? 0,
    entityTerms: options.entityTerms === undefined ? [] : [...options.entityTerms],
    graphWalkDepth: options.graphWalkDepth ?? null,
    maxGraphNodes: options.maxGraphNodes ?? null,
    asOf: options.asOf ?? null,
  };
}

function summarizeTemporalCue(cue: TemporalCue | null): JsonValue {
  if (cue === null) {
    return null;
  }

  return {
    ...(cue.sinceTs === undefined ? {} : { sinceTs: cue.sinceTs }),
    ...(cue.untilTs === undefined ? {} : { untilTs: cue.untilTs }),
    ...(cue.label === undefined ? {} : { label: cue.label }),
  };
}

function countSemanticHits(semantic: RetrievedSemantic): number {
  return (
    semantic.matched_nodes.length +
    semantic.support_hits.length +
    semantic.causal_hits.length +
    semantic.contradiction_hits.length +
    semantic.category_hits.length
  );
}

function recallStateScopeKey(
  options: RetrievalExecutionOptions,
  mode: RetrievalExecutionMode,
): string {
  if (mode === "cognition") {
    return SELF_RECALL_SCOPE;
  }

  return options.audienceEntityId ?? options.sessionId ?? DEFAULT_SESSION_ID;
}

function resolveRecallStateTurn(state: RecallState, requestedTurn: number | undefined): number {
  const nextStateTurn = state.lastRefreshTurn + 1;

  if (requestedTurn === undefined) {
    return nextStateTurn;
  }

  return Math.max(nextStateTurn, Math.max(0, Math.floor(requestedTurn)));
}

function isRecallHandleSuppressed(state: RecallState, key: string, turn: number): boolean {
  return (state.suppressedHandles[key] ?? -1) >= turn;
}

function selectWarmRecallCandidates(
  activeHandles: readonly RecallStateHandle[],
  state: RecallState,
  turn: number,
  limit: number,
): WarmRecallCandidate[] {
  if (limit <= 0) {
    return [];
  }

  const candidates: WarmRecallCandidate[] = [];

  for (const stateHandle of activeHandles) {
    const handle = normalizeRecallEvidenceHandle(stateHandle.handle);
    const key = recallEvidenceHandleKey(handle);

    if (stateHandle.expiresAfterTurn < turn) {
      continue;
    }

    if (isRecallHandleSuppressed(state, key, turn)) {
      continue;
    }

    candidates.push({ key, handle, stateHandle });
  }

  return candidates
    .sort((left, right) => compareWarmRecallCandidates(left, right, turn))
    .slice(0, limit);
}

function collectEvidenceHandles(
  evidence: readonly EvidenceItem[],
): Map<string, RecallEvidenceHandle> {
  const handles = new Map<string, RecallEvidenceHandle>();

  for (const item of evidence) {
    const handle = deriveRecallEvidenceHandle(item);

    if (handle === null) {
      continue;
    }

    const normalized = normalizeRecallEvidenceHandle(handle);
    const key = recallEvidenceHandleKey(normalized);

    if (!handles.has(key)) {
      handles.set(key, normalized);
    }
  }

  return handles;
}

function compareWarmRecallCandidates(
  left: WarmRecallCandidate,
  right: WarmRecallCandidate,
  turn: number,
): number {
  return (
    right.stateHandle.lastSeenTurn - left.stateHandle.lastSeenTurn ||
    right.stateHandle.firstSeenTurn - left.stateHandle.firstSeenTurn ||
    effectiveRecallStateReinforcementCount(right.stateHandle, turn) -
      effectiveRecallStateReinforcementCount(left.stateHandle, turn) ||
    compareNullableTurnAscending(
      left.stateHandle.lastRenderedTurn,
      right.stateHandle.lastRenderedTurn,
    ) ||
    compareStableText(left.key, right.key)
  );
}

function compareNullableTurnAscending(left: number | null, right: number | null): number {
  return (left ?? -1) - (right ?? -1);
}

function createRecallStateHandle(input: {
  handle: RecallEvidenceHandle;
  turn: number;
  ttlTurns: number;
  rendered: boolean;
}): RecallStateHandle {
  return {
    handle: input.handle,
    firstSeenTurn: input.turn,
    lastSeenTurn: input.turn,
    lastRenderedTurn: input.rendered ? input.turn : null,
    expiresAfterTurn: input.turn + input.ttlTurns,
    reinforcementCount: 1,
  };
}

function rankFreshAdmissions(
  handles: readonly [string, RecallEvidenceHandle][],
  selectedHandlePriorities: ReadonlyMap<string, number>,
): [string, RecallEvidenceHandle][] {
  return [...handles].sort(
    (left, right) =>
      compareSelectedHandlePriority(left[0], right[0], selectedHandlePriorities) ||
      sourceRetentionRank(right[1]) - sourceRetentionRank(left[1]) ||
      compareStableText(left[0], right[0]),
  );
}

function compareRecallStateRetentionPriority(
  left: RecallStateHandle,
  right: RecallStateHandle,
  turn: number,
): number {
  return (
    right.lastSeenTurn - left.lastSeenTurn ||
    right.expiresAfterTurn - left.expiresAfterTurn ||
    sourceRetentionRank(right.handle) - sourceRetentionRank(left.handle) ||
    right.firstSeenTurn - left.firstSeenTurn ||
    effectiveRecallStateReinforcementCount(right, turn) -
      effectiveRecallStateReinforcementCount(left, turn) ||
    compareStableText(recallEvidenceHandleKey(left.handle), recallEvidenceHandleKey(right.handle))
  );
}

function sourceRetentionRank(handle: RecallEvidenceHandle): number {
  if (handle.source === "episode") {
    return RECALL_HANDLE_EPISODE_RETENTION_RANK;
  }

  if (handle.source === "raw_stream") {
    return handle.parentEpisodeId === undefined
      ? RECALL_HANDLE_ORPHAN_RAW_STREAM_RETENTION_RANK
      : RECALL_HANDLE_PARENTED_RAW_STREAM_RETENTION_RANK;
  }

  if (handle.source === "commitment") {
    return RECALL_HANDLE_COMMITMENT_RETENTION_RANK;
  }

  if (handle.source === "open_question") {
    return RECALL_HANDLE_OPEN_QUESTION_RETENTION_RANK;
  }

  return RECALL_HANDLE_SECONDARY_RETENTION_RANK;
}

function compareStableText(left: string, right: string): number {
  if (left < right) {
    return -1;
  }

  if (left > right) {
    return 1;
  }

  return 0;
}

function compareSelectedHandlePriority(
  leftKey: string,
  rightKey: string,
  priorities: ReadonlyMap<string, number>,
): number {
  const leftPriority = priorities.get(leftKey);
  const rightPriority = priorities.get(rightKey);

  if (leftPriority === undefined) {
    return rightPriority === undefined ? 0 : 1;
  }

  return rightPriority === undefined ? -1 : leftPriority - rightPriority;
}

function capRecallStateHandles(
  handles: readonly RecallStateHandle[],
  limit: number,
  turn: number,
  selectedHandlePriorities: ReadonlyMap<string, number>,
): RecallStateHandle[] {
  if (limit <= 0) {
    return [];
  }

  if (handles.length <= limit) {
    return [...handles];
  }

  return [...handles]
    .sort(
      (left, right) =>
        compareSelectedHandlePriority(
          recallEvidenceHandleKey(left.handle),
          recallEvidenceHandleKey(right.handle),
          selectedHandlePriorities,
        ) || compareRecallStateRetentionPriority(left, right, turn),
    )
    .slice(0, limit);
}

function pruneSuppressedRecallHandles(
  suppressedHandles: RecallState["suppressedHandles"],
  turn: number,
): Record<string, number> {
  return Object.fromEntries(
    Object.entries(suppressedHandles).filter(([, expiresAfterTurn]) => expiresAfterTurn >= turn),
  );
}

function normalizeRecallStateBound(value: number | undefined, fallback: number): number {
  const raw = value ?? fallback;

  if (!Number.isFinite(raw)) {
    return fallback;
  }

  return Math.max(0, Math.floor(raw));
}

function warmRecallScore(stateHandle: RecallStateHandle): number {
  return clamp(
    WARM_RECALL_BASE_SCORE +
      Math.min(
        WARM_RECALL_MAX_REINFORCEMENT_BONUS,
        Math.min(DEFAULT_RECALL_STATE_REINFORCEMENT_CAP, stateHandle.reinforcementCount) *
          WARM_RECALL_REINFORCEMENT_STEP,
      ),
    0,
    WARM_RECALL_SCORE_CAP,
  );
}

function episodeVisibilityOptions(options: RetrievalExecutionOptions): EpisodeSearchOptions {
  return {
    audienceEntityId: options.audienceEntityId,
    crossAudience: options.crossAudience,
  };
}

function isEpisodeVisibleToRetrievalCapability(
  episode: Episode,
  options: EpisodeSearchOptions,
): boolean {
  return isEpisodeVisibleToCapability(episode, resolveViewerCapability(options));
}

function episodeSearchOptions(options: RetrievalExecutionOptions): EpisodeSearchOptions {
  return {
    ...episodeVisibilityOptions(options),
    minSimilarity: options.minSimilarity,
    tagFilter: options.tagFilter,
    tierFilter: options.tierFilter,
  };
}

function episodeCognitionRecallOptions(
  options: RetrievalExecutionOptions,
): EpisodeCognitionRecallOptions {
  return {
    minSimilarity: options.minSimilarity,
    tagFilter: options.tagFilter,
    tierFilter: options.tierFilter,
    timeRange: options.timeRange,
  };
}

function rankingAudienceEntityId(options: RetrievalExecutionOptions): EntityId | null | undefined {
  return options.rankingAudienceEntityId ?? options.audienceEntityId;
}

function normalizeTermInput(value: string): string {
  return value.trim();
}

function normalizeTermKey(value: string): string {
  return normalizeTermInput(value).toLowerCase();
}

function dedupeStrings(values: readonly string[]): string[] {
  return dedupeTermInputs(values.map((term) => ({ term, source: "llm-expansion" as const }))).map(
    (item) => item.term,
  );
}

function termSourcePrecedence(source: RecallIntent["source"]): number {
  if (source === "llm-expansion") {
    return 3;
  }

  if (source === "perception-entities") {
    return 2;
  }

  if (source === "audience-aliases") {
    return 1;
  }

  return 0;
}

function dedupeTermInputs<T extends { term: string; source: RecallIntent["source"] }>(
  values: readonly T[],
): T[] {
  const byKey = new Map<string, T>();

  for (const value of values) {
    const term = normalizeTermInput(value.term);

    if (term.length === 0) {
      continue;
    }

    const key = normalizeTermKey(term);

    const existing = byKey.get(key);

    if (
      existing === undefined ||
      termSourcePrecedence(value.source) > termSourcePrecedence(existing.source)
    ) {
      byKey.set(key, {
        ...value,
        term,
      });
    }
  }

  return [...byKey.values()];
}

function mergeRawEpisodeCandidates(
  candidates: readonly RawEpisodeEvidenceCandidate[],
): RawEpisodeEvidenceCandidate[] {
  const byId = new Map<EpisodeId, RawEpisodeEvidenceCandidate>();

  for (const candidate of candidates) {
    const current = byId.get(candidate.candidate.episode.id);

    if (
      current === undefined ||
      candidate.candidate.similarity > current.candidate.similarity ||
      (candidate.candidate.similarity === current.candidate.similarity &&
        candidate.candidate.episode.updated_at > current.candidate.episode.updated_at)
    ) {
      byId.set(candidate.candidate.episode.id, candidate);
    }
  }

  return [...byId.values()];
}

function isSemanticIntentKind(kind: RecallIntentKind): boolean {
  return (
    kind === "raw_text" ||
    kind === "topic" ||
    kind === "relationship" ||
    kind === "known_term" ||
    kind === "commitment" ||
    kind === "open_question"
  );
}

function mergeSemanticRetrievals(
  retrievals: readonly ResolvedSemanticRetrieval[],
): ResolvedSemanticRetrieval {
  const supports = new Map<string, SemanticNode>();
  const contradicts = new Map<string, SemanticNode>();
  const categories = new Map<string, SemanticNode>();
  const matchedNodes = new Map<string, ResolvedSemanticRetrieval["matchedNodes"][number]>();
  const supportHits = new Map<string, ResolvedSemanticRetrieval["supportHits"][number]>();
  const causalHits = new Map<string, ResolvedSemanticRetrieval["causalHits"][number]>();
  const contradictionHits = new Map<
    string,
    ResolvedSemanticRetrieval["contradictionHits"][number]
  >();
  const categoryHits = new Map<string, ResolvedSemanticRetrieval["categoryHits"][number]>();

  for (const retrieval of retrievals) {
    for (const node of retrieval.context.supports) {
      supports.set(node.id, node);
    }

    for (const node of retrieval.context.contradicts) {
      contradicts.set(node.id, node);
    }

    for (const node of retrieval.context.categories) {
      categories.set(node.id, node);
    }

    for (const node of retrieval.matchedNodes) {
      const current = matchedNodes.get(node.id);

      if (current === undefined || (node.retrieval_score ?? 0) > (current.retrieval_score ?? 0)) {
        matchedNodes.set(node.id, node);
      }
    }

    for (const hit of retrieval.supportHits) {
      supportHits.set(semanticHitKey(hit), hit);
    }

    for (const hit of retrieval.causalHits) {
      causalHits.set(semanticHitKey(hit), hit);
    }

    for (const hit of retrieval.contradictionHits) {
      contradictionHits.set(semanticHitKey(hit), hit);
    }

    for (const hit of retrieval.categoryHits) {
      categoryHits.set(semanticHitKey(hit), hit);
    }
  }

  const sortedMatchedNodes = [...matchedNodes.values()].sort(
    (left, right) =>
      (right.retrieval_score ?? 0) - (left.retrieval_score ?? 0) ||
      (right.base_retrieval_score ?? 0) - (left.base_retrieval_score ?? 0) ||
      right.updated_at - left.updated_at ||
      left.id.localeCompare(right.id),
  );

  return {
    context: {
      supports: [...supports.values()],
      contradicts: [...contradicts.values()],
      categories: [...categories.values()],
    },
    contradictionPresent: contradictionHits.size > 0 || contradicts.size > 0,
    matchedNodeIds: sortedMatchedNodes.map((node) => node.id),
    matchedNodes: sortedMatchedNodes,
    supportHits: [...supportHits.values()],
    causalHits: [...causalHits.values()],
    contradictionHits: [...contradictionHits.values()],
    categoryHits: [...categoryHits.values()],
    asOf: retrievals.find((retrieval) => retrieval.asOf !== undefined)?.asOf,
  };
}

function semanticHitKey(hit: ResolvedSemanticRetrieval["supportHits"][number]): string {
  return [hit.root_node_id, hit.node.id, ...hit.edgePath.map((edge) => edge.id)].join("|");
}

function episodeCandidateToEvidence(item: EpisodeEvidenceCandidate): EvidenceItem {
  const episode = item.candidate.episode;

  return {
    id: `evidence_episode_${episode.id}_${item.intent.id}`,
    source: "episode",
    text: `${episode.title}: ${episode.narrative}`,
    provenance: {
      episodeId: episode.id,
      streamIds: [...episode.source_stream_ids],
    },
    recallIntentId: item.intent.id,
    matchedTerms: [...item.matchedTerms],
    score: item.score.score,
    rawScore: item.score.rawScore,
    scoreBreakdown: {
      vector: item.candidate.similarity,
      salience: item.score.decayedSalience,
      recency: computeRecencyEvidenceScore(episode.updated_at),
      exactTerm: item.intent.kind === "known_term" ? item.score.entityRelevance : undefined,
    },
    disclosureLabel: memoryDisclosureLabelFromEpisodeAccess(episode),
  };
}

function semanticRetrievalToEvidence(
  semantic: ResolvedSemanticRetrieval,
  intent: RecallIntent,
): EvidenceItem[] {
  const nodeEvidence = semantic.matchedNodes.map(
    (node): EvidenceItem => ({
      id: `evidence_semantic_node_${node.id}_${intent.id}`,
      source: "semantic_node",
      text: `${node.label}: ${node.description}`,
      provenance: {
        nodeId: node.id,
      },
      recallIntentId: intent.id,
      matchedTerms: intent.kind === "known_term" ? [...intent.terms] : [],
      score: clamp(node.retrieval_score ?? node.base_retrieval_score ?? 0.5, 0, 1),
      scoreBreakdown: {
        vector: node.base_retrieval_score,
        exactTerm: intent.kind === "known_term" ? 1 : undefined,
      },
      source_episode_ids: [...node.source_episode_ids],
      ...(node.partial_source_visibility === true
        ? {
            partial_source_visibility: true,
            source_visibility_fraction: node.source_visibility_fraction,
          }
        : {}),
      disclosureLabel: node.disclosureLabel ?? unknownMemoryDisclosureLabel(),
    }),
  );
  const edgeEvidence = [
    ...semantic.supportHits,
    ...semantic.causalHits,
    ...semantic.contradictionHits,
    ...semantic.categoryHits,
  ].map((hit): EvidenceItem => {
    const edge = hit.edgePath.at(-1);
    const edgeId = edge?.id;
    const hasPartialSourceVisibility =
      hit.node.partial_source_visibility === true ||
      hit.edgePath.some((pathEdge) => pathEdge.partial_source_visibility === true);
    const sourceVisibilityFractions = [
      hit.node.partial_source_visibility === true ? hit.node.source_visibility_fraction : undefined,
      ...hit.edgePath.map((pathEdge) =>
        pathEdge.partial_source_visibility === true
          ? pathEdge.source_visibility_fraction
          : undefined,
      ),
    ].filter((fraction): fraction is number => typeof fraction === "number");
    const sourceVisibilityFraction =
      sourceVisibilityFractions.length === 0 ? undefined : Math.min(...sourceVisibilityFractions);
    const semanticEdgeSourceEpisodeIds = [
      ...new Set([
        ...hit.node.source_episode_ids,
        ...hit.edgePath.flatMap((pathEdge) => pathEdge.evidence_episode_ids),
      ]),
    ];
    const semanticEdgeDisclosureLabel = combineMemoryDisclosureLabels([
      hit.node.disclosureLabel ?? combineMemoryDisclosureLabels([]),
      ...hit.edgePath.map(
        (pathEdge) => pathEdge.disclosureLabel ?? combineMemoryDisclosureLabels([]),
      ),
    ]);

    return {
      id: `evidence_semantic_edge_${edgeId ?? hit.node.id}_${intent.id}`,
      source: "semantic_edge",
      text: `${hit.node.label}: ${hit.node.description}`,
      provenance: {
        ...(edgeId === undefined ? {} : { edgeId }),
        nodeId: hit.node.id,
      },
      recallIntentId: intent.id,
      matchedTerms: [],
      score: clamp(
        averageEdgeConfidence(hit.edgePath) * (hit.node.status_retrieval_multiplier ?? 1),
        0,
        1,
      ),
      scoreBreakdown: {
        provenance: hit.edgePath.length > 0 ? 1 : 0,
      },
      source_episode_ids:
        semanticEdgeSourceEpisodeIds.length === 0
          ? [...hit.node.source_episode_ids]
          : semanticEdgeSourceEpisodeIds,
      ...(hasPartialSourceVisibility
        ? {
            partial_source_visibility: true,
            ...(sourceVisibilityFraction === undefined
              ? {}
              : { source_visibility_fraction: sourceVisibilityFraction }),
          }
        : {}),
      disclosureLabel: semanticEdgeDisclosureLabel,
    };
  });

  return [...nodeEvidence, ...edgeEvidence];
}

function averageEdgeConfidence(
  edgePath: ResolvedSemanticRetrieval["supportHits"][number]["edgePath"],
) {
  if (edgePath.length === 0) {
    return SEMANTIC_EVIDENCE_EMPTY_EDGE_PATH_CONFIDENCE;
  }

  return clamp(edgePath.reduce((sum, edge) => sum + edge.confidence, 0) / edgePath.length, 0, 1);
}

function openQuestionToEvidence(
  question: OpenQuestion,
  intent: RecallIntent,
  score: number,
): EvidenceItem {
  return {
    id: `evidence_open_question_${question.id}_${intent.id}`,
    source: "open_question",
    text: question.question,
    provenance: {
      openQuestionId: question.id,
    },
    recallIntentId: intent.id,
    matchedTerms: [],
    score: clamp(score, 0, 1),
    scoreBreakdown: {
      salience: question.urgency,
    },
    disclosureLabel: openQuestionMemoryDisclosureLabel(question),
  };
}

function imagePerceptionToEvidence(
  item: ImagePerceptionEvidenceCandidate,
  nowMs: number,
): EvidenceItem {
  const record = item.hit.record;
  // originAge is the only part of this evidence item that varies between turns:
  // caption, scene, kind, terms and provenance all derive from the immutable
  // perception row, so a re-attached image is otherwise byte-identical every
  // time it is recalled. The age is recomputed here against the turn's nowMs
  // (the rehydration path routes through this function for exactly that
  // reason), so it rolls over at the upload's time of day rather than at
  // midnight, and it can differ between two turns inside one calendar day.
  // Do not memoize the returned item per record: that would silently freeze
  // the one field on a remembered image that still carries information.
  const originAge = formatRelativeAge(record.created_at, nowMs);
  const originFrame = `[remembered image -- not sent in this message; first shared ${originAge}]`;
  const imageLabel = `Image: remembered user-uploaded ${record.image_kind}`;
  const disclosureLabel = imagePerceptionMemoryDisclosureLabel(record);

  return {
    id: `evidence_image_perception_${record.perception_id}_${item.intent.id}`,
    source: "image_perception",
    text: [
      `${imageLabel}.`,
      `Caption: ${record.caption}`,
      `Scene: ${record.scene}`,
      record.search_terms.length === 0 ? null : `Search terms: ${record.search_terms.join("; ")}`,
      record.visible_text.length === 0 ? null : `Visible text: ${record.visible_text.join("; ")}`,
      record.possible_user_relevant_details.length === 0
        ? null
        : `Possible relevant details: ${record.possible_user_relevant_details.join("; ")}`,
    ]
      .filter((part): part is string => part !== null)
      .join("\n"),
    provenance: {
      imagePerceptionId: record.perception_id,
      attachmentId: record.attachment_id,
      streamIds:
        record.stream_entry_id === null
          ? [record.parent_entry_id]
          : [record.parent_entry_id, record.stream_entry_id],
    },
    recallIntentId: item.intent.id,
    matchedTerms: item.intent.kind === "known_term" ? [...item.intent.terms] : [],
    score: clamp(item.hit.similarity, 0, 1),
    scoreBreakdown: {
      vector: item.hit.similarity,
      provenance: IMAGE_PERCEPTION_PROVENANCE_SCORE,
    },
    imageAttachmentId: record.attachment_id,
    imageLabel,
    imageOriginFrame: originFrame,
    disclosureLabel,
    citationType:
      record.stream_entry_id === null ? "parent_user_message" : "generated_perception_text",
  };
}

function streamEntriesToEvidence(
  entries: ReadonlyMap<string, StreamEntry>,
  episodeCandidates: readonly EpisodeEvidenceCandidate[],
): EvidenceItem[] {
  const intentByStreamId = new Map<string, RecallIntent>();
  const parentEpisodeIdByStreamId = new Map<string, EpisodeId>();
  const disclosureLabelByStreamId = new Map<string, MemoryDisclosureLabel>();

  for (const candidate of episodeCandidates) {
    const disclosureLabel = memoryDisclosureLabelFromEpisodeAccess(candidate.candidate.episode);

    for (const streamId of candidate.candidate.episode.source_stream_ids) {
      intentByStreamId.set(streamId, candidate.intent);
      parentEpisodeIdByStreamId.set(streamId, candidate.candidate.episode.id);
      disclosureLabelByStreamId.set(streamId, disclosureLabel);
    }
  }

  return [...entries.values()]
    .filter((entry) => intentByStreamId.has(entry.id))
    .map((entry) =>
      streamEntryToEvidence(entry, intentByStreamId.get(entry.id)!, "raw_stream", {
        parentEpisodeId: parentEpisodeIdByStreamId.get(entry.id),
        disclosureLabel: disclosureLabelByStreamId.get(entry.id),
      }),
    );
}

function streamEntryToEvidence(
  entry: StreamEntry,
  intent: RecallIntent,
  source: "raw_stream" | "recent_raw_stream" = "raw_stream",
  options: {
    parentEpisodeId?: EpisodeId;
    disclosureLabel?: MemoryDisclosureLabel;
  } = {},
): EvidenceItem {
  return {
    id: `evidence_raw_stream_${entry.id}_${intent.id}`,
    source,
    text: streamEntryContentToText(entry),
    provenance: {
      streamIds: [entry.id],
      ...(options.parentEpisodeId === undefined
        ? {}
        : { parentEpisodeId: options.parentEpisodeId }),
    },
    recallIntentId: intent.id,
    matchedTerms: [],
    score:
      intent.kind === "recent"
        ? RECENT_RAW_STREAM_EVIDENCE_SCORE
        : DEFAULT_RAW_STREAM_EVIDENCE_SCORE,
    scoreBreakdown: {
      provenance: 1,
      recency: intent.kind === "recent" ? 1 : undefined,
    },
    disclosureLabel: options.disclosureLabel ?? unknownMemoryDisclosureLabel(),
  };
}

function streamEntryContentToText(entry: StreamEntry): string {
  if (typeof entry.content === "string") {
    return entry.content;
  }

  return JSON.stringify(entry.content ?? null);
}

function commitmentToEvidence(
  commitment: CommitmentRecord,
  intent: RecallIntent,
  similarity: number,
): EvidenceItem {
  const vector = clamp(similarity, 0, 1);

  return {
    id: `evidence_commitment_${commitment.id}_${intent.id}`,
    source: "commitment",
    text: `${commitment.type}: ${commitment.directive}`,
    provenance: {
      commitmentId: commitment.id,
    },
    recallIntentId: intent.id,
    matchedTerms: [],
    score: clamp(commitment.priority / 10 + vector * 0.4, 0, 1),
    scoreBreakdown: {
      vector,
    },
    disclosureLabel: commitmentMemoryDisclosureLabel(commitment),
    commitment_enforcement_class: effectiveCommitmentEnforcementClass(commitment),
    commitment_critical_domain: effectiveCommitmentCriticalDomain(commitment),
    commitment_directive_chars: commitment.directive.length,
  };
}

function computeRecencyEvidenceScore(updatedAt: number): number {
  return Number.isFinite(updatedAt) ? 0.1 : 0;
}
