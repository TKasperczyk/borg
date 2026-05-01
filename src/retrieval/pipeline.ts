import type { AttentionWeights, TemporalCue } from "../cognition/types.js";
import type { EmbeddingClient } from "../embeddings/index.js";
import type { LLMClient } from "../llm/index.js";
import type {
  CommitmentRecord,
  CommitmentRepository,
  EntityRepository,
} from "../memory/commitments/index.js";
import { isEpisodeVisibleToAudience } from "../memory/episodic/index.js";
import type { EpisodicRepository } from "../memory/episodic/repository.js";
import type {
  Episode,
  EpisodeSearchCandidate,
  EpisodeSearchOptions,
} from "../memory/episodic/types.js";
import type { DecayOptions } from "../memory/episodic/decay.js";
import type { OpenQuestion, OpenQuestionsRepository, ValueRecord } from "../memory/self/index.js";
import type { SemanticGraph } from "../memory/semantic/graph.js";
import type { SemanticNodeRepository } from "../memory/semantic/repository.js";
import type { ReviewQueueRepository } from "../memory/semantic/review-queue.js";
import type { SemanticNode } from "../memory/semantic/types.js";
import type { SocialProfile } from "../memory/social/index.js";
import type { StreamEntry, StreamEntryIndexRepository } from "../stream/index.js";
import { NOOP_TRACER, type TurnTracer } from "../cognition/tracing/tracer.js";
import { SystemClock, type Clock } from "../util/clock.js";
import { StorageError } from "../util/errors.js";
import type { EntityId, EpisodeId } from "../util/ids.js";
import type { JsonValue } from "../util/json-value.js";

import { CitationResolver, type CitationResolverOptions } from "./citations.js";
import { assembleRetrievedContext, type RetrievedContext } from "./context-assembly.js";
import { cosineSimilarity } from "./embedding-similarity.js";
import { rankEvidenceItems } from "./evidence-pool.js";
import {
  projectEpisodes,
  projectOpenQuestions,
  projectSemantic,
  type EpisodeProjectionSource,
} from "./evidence-projections.js";
import { retrieveOpenQuestionsForQuery as retrieveOpenQuestionsForQueryFromRepository } from "./open-questions.js";
import { RawStreamAdapter } from "./raw-stream-adapter.js";
import { DEFAULT_RECALL_EXPANSION_MODEL, expandRecall } from "./recall-expansion.js";
import type { EvidenceItem, EvidencePool, RecallIntent, RecallIntentKind } from "./recall-types.js";
import {
  buildRetrievedEpisode,
  clamp,
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
  resolveSemanticContext,
  toRetrievedSemantic,
  type ResolvedSemanticRetrieval,
  type RetrievedSemantic,
} from "./semantic-retrieval.js";
import { resolveTimeSignals } from "./time-signals.js";

export type { RetrievedContext } from "./context-assembly.js";
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
  semanticGraph?: SemanticGraph;
  reviewQueueRepository?: Pick<ReviewQueueRepository, "listOpenBeliefRevisionsByTarget">;
  openQuestionsRepository?: OpenQuestionsRepository;
  entityRepository?: Pick<EntityRepository, "findByName">;
  commitmentRepository?: Pick<CommitmentRepository, "list">;
  clock?: Clock;
  tracer?: TurnTracer;
  scoreWeights?: ScoreWeights;
  mmrLambda?: number;
  decayOptions?: Omit<DecayOptions, "nowMs">;
  semanticUnderReviewMultiplier?: number;
  commitmentEvidenceSimilarityThreshold?: number;
};

export type RetrievalSearchOptions = EpisodeSearchOptions & {
  limit?: number;
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
  includeOpenQuestions?: boolean;
  openQuestionsLimit?: number;
  moodState?: RetrievalMoodState | null;
  audienceProfile?: SocialProfile | null;
  audienceTerms?: readonly string[];
  entityTerms?: readonly string[];
  traceTurnId?: string;
};

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
};

type SemanticEvidenceCandidate = {
  intent: RecallIntent;
  semantic: ResolvedSemanticRetrieval;
};

type OpenQuestionEvidenceCandidate = {
  intent: RecallIntent;
  question: OpenQuestion;
  score: number;
};

const DEFAULT_COMMITMENT_EVIDENCE_SIMILARITY_THRESHOLD = 0.3;
const RETRIEVAL_FANOUT_CONCURRENCY = 5;

export class RetrievalPipeline {
  private readonly clock: Clock;
  private readonly scoreWeights: ScoreWeights;
  private readonly mmrLambda: number;
  private readonly decayOptions?: Omit<DecayOptions, "nowMs">;
  private readonly tracer: TurnTracer;

  constructor(private readonly options: RetrievalPipelineOptions) {
    this.clock = options.clock ?? new SystemClock();
    this.tracer = options.tracer ?? NOOP_TRACER;
    this.scoreWeights = options.scoreWeights ?? {
      similarity: 0.7,
      salience: 0.3,
    };
    this.mmrLambda = options.mmrLambda ?? 0.7;
    this.decayOptions = options.decayOptions;
  }

  retrieveOpenQuestionsForQuery(
    query: string,
    options: {
      relatedSemanticNodeIds?: readonly SemanticNode["id"][];
      audienceEntityId?: EntityId | null;
      limit?: number;
      queryVector?: Float32Array;
      traceTurnId?: string;
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
                this.tracer.emit("retrieval_degraded", {
                  turnId: options.traceTurnId!,
                  subsystem: "open_questions",
                  reason: error instanceof Error ? `${reason}: ${error.message}` : reason,
                });
              }
            : undefined,
      },
    );
  }

  async searchWithContext(
    query: string,
    options: RetrievalSearchOptions = {},
  ): Promise<RetrievedContext> {
    if (this.tracer.enabled && options.traceTurnId !== undefined) {
      this.tracer.emit("retrieval_started", {
        turnId: options.traceTurnId,
        query,
        options: summarizeRetrievalOptions(options),
      });
    }

    const nowMs = this.clock.now();
    const limit = Math.max(1, options.limit ?? 5);
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
          this.tracer.emit("retrieval_degraded", {
            turnId: options.traceTurnId,
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
    );
    const citationResolver = this.createCitationResolver();
    const citationEntries = await citationResolver.resolveCitationEntries(
      episodeCandidates.flatMap((item) => item.candidate.episode.source_stream_ids),
    );
    const semanticRetrievals = await this.collectSemanticRetrievals(intents, options);
    const semantic = mergeSemanticRetrievals(semanticRetrievals.map((item) => item.semantic));
    const openQuestions = await this.collectOpenQuestions(intents, semantic, options);
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
    const commitmentEvidence = await this.collectCommitmentEvidence(intents, options);
    const rawStreamEvidence = [
      ...streamEntriesToEvidence(citationEntries, episodeCandidates),
      ...this.collectRecentRawStreamEvidence(intents),
    ];
    const evidencePool: EvidencePool = {
      intents,
      items: rankEvidenceItems([
        ...episodeEvidence,
        ...semanticEvidence,
        ...openQuestionEvidence,
        ...commitmentEvidence,
        ...rawStreamEvidence,
      ]),
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
    });
    const semanticProjection = projectSemantic(evidencePool, toRetrievedSemantic(semantic));
    const openQuestionProjection = projectOpenQuestions(
      evidencePool,
      new Map(
        openQuestionEvidenceSources.map(({ evidence, item }) => [evidence.id, item.question]),
      ),
    );

    for (const result of episodeProjection.episodes) {
      this.options.episodicRepository.recordRetrieval(result.episode.id, nowMs, result.score);
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
      expectedCount: limit,
    });

    if (this.tracer.enabled && options.traceTurnId !== undefined) {
      this.tracer.emit("retrieval_completed", {
        turnId: options.traceTurnId,
        episodeCount: context.episodes.length,
        semanticHits: countSemanticHits(context.semantic),
        asOf: options.asOf ?? null,
        confidence: context.confidence,
      });
    }

    return context;
  }

  async search(query: string, options: RetrievalSearchOptions = {}): Promise<RetrievedEpisode[]> {
    const result = await this.searchWithContext(query, options);
    return result.episodes;
  }

  private async buildRecallIntents(
    query: string,
    options: RetrievalSearchOptions,
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
    options: RetrievalSearchOptions,
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
      if (
        this.tracer.enabled &&
        options.traceTurnId !== undefined &&
        !isUnscriptedFakeRecallExpansion(error)
      ) {
        this.tracer.emit("retrieval_degraded", {
          turnId: options.traceTurnId,
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
    options: RetrievalSearchOptions,
    scoringFeatures: RetrievalScoringFeatures,
    nowMs: number,
    limit: number,
  ): Promise<EpisodeEvidenceCandidate[]> {
    const rawCandidates = (
      await Promise.all(
        intents.map((intent) => this.collectEpisodicCandidatesForIntent(intent, options, limit)),
      )
    ).flat();
    const participantEntityIds = this.resolveParticipantEntityIds(
      rawCandidates,
      options.audienceEntityId,
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

  private async collectEpisodicCandidatesForIntent(
    intent: RecallIntent,
    options: RetrievalSearchOptions,
    limit: number,
  ): Promise<RawEpisodeEvidenceCandidate[]> {
    const vectorBudget = Math.max(limit * 2, 12);
    const indexedBudget = Math.max(limit * 2, 8);
    const recentBudget = Math.max(limit, 4);

    if (intent.kind === "raw_text" || intent.kind === "topic" || intent.kind === "relationship") {
      const intentVector = await this.options.embeddingClient.embed(intent.query);
      const candidates = await this.options.episodicRepository.searchByVector(intentVector, {
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
      const candidates = await this.options.episodicRepository.searchByParticipantsOrTags(
        intent.terms,
        {
          ...episodeVisibilityOptions(options),
          limit: indexedBudget,
        },
      );

      return candidates.map((candidate) => ({
        intent,
        candidate,
        matchedTerms: [...intent.terms],
      }));
    }

    if (intent.kind === "time" && intent.timeRange !== undefined) {
      const candidates = await this.options.episodicRepository.searchByTimeRange(intent.timeRange, {
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
      const [recent, hottest] = await Promise.all([
        this.options.episodicRepository.listRecent({
          ...episodeVisibilityOptions(options),
          limit: recentLimit,
        }),
        this.options.episodicRepository.listHottest({
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
    options: RetrievalSearchOptions,
    scoringFeatures: RetrievalScoringFeatures,
    nowMs: number,
    participantEntityIds: ParticipantEntityResolutionLookup | undefined,
  ): EpisodeScoreDetails {
    const intentTimeRange = entry.intent.kind === "time" ? (entry.intent.timeRange ?? null) : null;
    const score = scoreCandidate(
      entry.candidate,
      {
        ...options,
        scoringFeatures,
        entityTerms: entry.intent.kind === "known_term" ? entry.intent.terms : [],
        ...(participantEntityIds === undefined ? {} : { participantEntityIds }),
      },
      nowMs,
      intentTimeRange,
      this.scoringDefaults(),
    );
    const exactBoost = entry.intent.kind === "known_term" ? 0.25 : 0;
    const recencyBoost = entry.intent.kind === "recent" ? 0.05 : 0;

    return {
      ...score,
      score: clamp(score.score + exactBoost + recencyBoost, 0, 1),
    };
  }

  private async collectSemanticRetrievals(
    intents: readonly RecallIntent[],
    options: RetrievalSearchOptions,
  ): Promise<SemanticEvidenceCandidate[]> {
    const relevantIntents = intents.filter((intent) => isSemanticIntentKind(intent.kind));

    const results = await mapWithConcurrency(
      relevantIntents,
      RETRIEVAL_FANOUT_CONCURRENCY,
      async (intent): Promise<SemanticEvidenceCandidate | null> => {
        try {
          const intentVector = await this.options.embeddingClient.embed(intent.query);
          const semantic = await resolveSemanticContext(
            intent.query,
            {
              ...options,
              queryVector: intentVector,
              exactTerms: intent.kind === "known_term" ? intent.terms : [],
              underReviewMultiplier:
                options.underReviewMultiplier ?? this.options.semanticUnderReviewMultiplier,
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
    options: RetrievalSearchOptions,
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
            audienceEntityId: options.audienceEntityId ?? null,
            limit: options.openQuestionsLimit,
            traceTurnId: options.traceTurnId,
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

  private emitRetrievalDegraded(
    options: RetrievalSearchOptions,
    subsystem: string,
    error: unknown,
  ): void {
    if (this.tracer.enabled && options.traceTurnId !== undefined) {
      this.tracer.emit("retrieval_degraded", {
        turnId: options.traceTurnId,
        subsystem,
        reason: error instanceof Error ? error.message : String(error),
      });
    }
  }

  private async collectCommitmentEvidence(
    intents: readonly RecallIntent[],
    options: RetrievalSearchOptions,
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
      audience: options.audienceEntityId ?? null,
      nowMs: this.clock.now(),
    });

    if (activeCommitments.length === 0) {
      return [];
    }

    const intentVectors = await this.options.embeddingClient.embedBatch(
      relevantIntents.map((intent) => intent.query),
    );
    const commitmentVectors = await this.options.embeddingClient.embedBatch(
      activeCommitments.map((commitment) => commitment.directive),
    );
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

  private collectRecentRawStreamEvidence(intents: readonly RecallIntent[]): EvidenceItem[] {
    const recentIntent = intents.find((intent) => intent.kind === "recent");

    if (recentIntent === undefined) {
      return [];
    }

    const adapter = new RawStreamAdapter({
      dataDir: this.options.dataDir,
      entryIndex: this.options.entryIndex,
    });

    return adapter
      .recent({ limit: 3 })
      .map((entry) => streamEntryToEvidence(entry, recentIntent, "recent_raw_stream"));
  }

  async getEpisode(
    id: Episode["id"],
    options: RetrievalGetEpisodeOptions = {},
  ): Promise<RetrievedEpisode | null> {
    const episode = await this.options.episodicRepository.get(id);

    if (episode === null) {
      return null;
    }

    if (
      !isEpisodeVisibleToAudience(episode, options.audienceEntityId, {
        crossAudience: options.crossAudience,
      })
    ) {
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
      scored.decayedSalience,
      scored.heat,
      scored.goalRelevance,
      scored.valueAlignment,
      scored.timeRelevance,
      scored.moodBoost,
      scored.socialRelevance,
      scored.entityRelevance,
      scored.suppressionPenalty,
      1,
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

        participantEntityIds.set(key, this.options.entityRepository.findByName(participant));
      }
    }

    return participantEntityIds;
  }
}

function summarizeRetrievalOptions(options: RetrievalSearchOptions): JsonValue {
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

function isUnscriptedFakeRecallExpansion(error: unknown): boolean {
  return (
    error instanceof Error &&
    error.message === "FakeLLMClient has no scripted recall expansion response available"
  );
}

function episodeVisibilityOptions(options: RetrievalSearchOptions): EpisodeSearchOptions {
  return {
    audienceEntityId: options.audienceEntityId,
    crossAudience: options.crossAudience,
    globalIdentitySelfAudienceEntityId: options.globalIdentitySelfAudienceEntityId,
  };
}

function episodeSearchOptions(options: RetrievalSearchOptions): EpisodeSearchOptions {
  return {
    ...episodeVisibilityOptions(options),
    minSimilarity: options.minSimilarity,
    tagFilter: options.tagFilter,
    tierFilter: options.tierFilter,
  };
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

async function mapWithConcurrency<T, U>(
  items: readonly T[],
  limit: number,
  mapper: (item: T, index: number) => Promise<U>,
): Promise<U[]> {
  const normalizedLimit = Math.max(1, Math.floor(limit));
  const results: U[] = [];

  for (let start = 0; start < items.length; start += normalizedLimit) {
    const batch = items.slice(start, start + normalizedLimit);
    results.push(
      ...(await Promise.all(batch.map((item, index) => mapper(item, start + index)))),
    );
  }

  return results;
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
    scoreBreakdown: {
      vector: item.candidate.similarity,
      salience: item.score.decayedSalience,
      recency: computeRecencyEvidenceScore(episode.updated_at),
      exactTerm: item.intent.kind === "known_term" ? item.score.entityRelevance : undefined,
    },
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
      score: averageEdgeConfidence(hit.edgePath),
      scoreBreakdown: {
        provenance: hit.edgePath.length > 0 ? 1 : 0,
      },
    };
  });

  return [...nodeEvidence, ...edgeEvidence];
}

function averageEdgeConfidence(
  edgePath: ResolvedSemanticRetrieval["supportHits"][number]["edgePath"],
) {
  if (edgePath.length === 0) {
    return 0.3;
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
  };
}

function streamEntriesToEvidence(
  entries: ReadonlyMap<string, StreamEntry>,
  episodeCandidates: readonly EpisodeEvidenceCandidate[],
): EvidenceItem[] {
  const intentByStreamId = new Map<string, RecallIntent>();

  for (const candidate of episodeCandidates) {
    for (const streamId of candidate.candidate.episode.source_stream_ids) {
      intentByStreamId.set(streamId, candidate.intent);
    }
  }

  return [...entries.values()]
    .filter((entry) => intentByStreamId.has(entry.id))
    .map((entry) => streamEntryToEvidence(entry, intentByStreamId.get(entry.id)!));
}

function streamEntryToEvidence(
  entry: StreamEntry,
  intent: RecallIntent,
  source: "raw_stream" | "recent_raw_stream" = "raw_stream",
): EvidenceItem {
  return {
    id: `evidence_raw_stream_${entry.id}_${intent.id}`,
    source,
    text: streamEntryContentToText(entry),
    provenance: {
      streamIds: [entry.id],
    },
    recallIntentId: intent.id,
    matchedTerms: [],
    score: intent.kind === "recent" ? 0.2 : 1,
    scoreBreakdown: {
      provenance: 1,
      recency: intent.kind === "recent" ? 1 : undefined,
    },
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
  };
}

function computeRecencyEvidenceScore(updatedAt: number): number {
  return Number.isFinite(updatedAt) ? 0.1 : 0;
}
