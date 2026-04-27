import type { AttentionWeights, TemporalCue } from "../cognition/types.js";
import type { EmbeddingClient } from "../embeddings/index.js";
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
import type { SemanticNode } from "../memory/semantic/types.js";
import type { SocialProfile } from "../memory/social/index.js";
import type { StreamEntryIndexRepository } from "../stream/index.js";
import { NOOP_TRACER, type TurnTracer } from "../cognition/tracing/tracer.js";
import { SystemClock, type Clock } from "../util/clock.js";
import { StorageError } from "../util/errors.js";
import type { EntityId } from "../util/ids.js";
import type { JsonValue } from "../util/json-value.js";

import { CitationResolver, type CitationResolverOptions } from "./citations.js";
import { assembleRetrievedContext, type RetrievedContext } from "./context-assembly.js";
import {
  generateEpisodicCandidates,
  mergeCandidates as mergeEpisodeCandidates,
  tagCandidates as tagEpisodeCandidates,
  type EpisodeCandidateSource,
  type MergedEpisodeCandidate,
} from "./episodic-candidates.js";
import { applyMmr } from "./mmr.js";
import { retrieveOpenQuestionsForQuery as retrieveOpenQuestionsForQueryFromRepository } from "./open-questions.js";
import {
  buildRetrievedEpisode,
  clamp,
  scoreCandidate,
  type EpisodeScoreDefaults,
  type RetrievalMoodState,
  type RetrievedEpisode,
  type ScoreWeights,
  type SuppressionLookup,
} from "./scoring.js";
import {
  resolveSemanticContext,
  toRetrievedSemantic,
  type RetrievedSemantic,
  type RetrievedSemanticHit,
} from "./semantic-retrieval.js";
import { resolveTimeSignals } from "./time-signals.js";

export type { RetrievedContext } from "./context-assembly.js";
export type { RetrievedEpisode } from "./scoring.js";
export type { RetrievedSemantic, RetrievedSemanticHit } from "./semantic-retrieval.js";

export type RetrievalPipelineOptions = {
  embeddingClient: EmbeddingClient;
  episodicRepository: EpisodicRepository;
  dataDir: string;
  entryIndex?: StreamEntryIndexRepository;
  semanticNodeRepository?: SemanticNodeRepository;
  semanticGraph?: SemanticGraph;
  openQuestionsRepository?: OpenQuestionsRepository;
  clock?: Clock;
  tracer?: TurnTracer;
  scoreWeights?: ScoreWeights;
  mmrLambda?: number;
  decayOptions?: Omit<DecayOptions, "nowMs">;
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
  temporalCue?: TemporalCue | null;
  strictTimeRange?: boolean;
  suppressionSet?: SuppressionLookup;
  graphWalkDepth?: number;
  maxGraphNodes?: number;
  asOf?: number;
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
    } = {},
  ): OpenQuestion[] {
    return retrieveOpenQuestionsForQueryFromRepository(
      this.options.openQuestionsRepository,
      query,
      options,
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
    const queryVector = await this.options.embeddingClient.embed(query);
    const timeSignals = resolveTimeSignals(options);
    const candidates = await generateEpisodicCandidates({
      repository: this.options.episodicRepository,
      query,
      queryVector,
      options,
      limit,
      timeSignals,
    });
    const scored = candidates.map((entry) => {
      const candidate = entry.candidate;
      const score = scoreCandidate(
        candidate,
        options,
        nowMs,
        timeSignals.scoringRange,
        this.scoringDefaults(),
      );

      return {
        ...entry,
        candidate,
        ...score,
      };
    });
    const preMmrLimit = Math.max(limit * 4, 24);
    const trimmed = [...scored]
      .sort(
        (left, right) =>
          right.score - left.score ||
          right.candidate.similarity - left.candidate.similarity ||
          right.candidate.episode.updated_at - left.candidate.episode.updated_at,
      )
      .slice(0, preMmrLimit);
    const selected = applyMmr(
      trimmed.map((item) => ({
        item,
        vector: item.candidate.episode.embedding,
        relevanceScore: item.score,
      })),
      {
        limit,
        lambda: options.mmrLambda ?? this.mmrLambda,
      },
    );
    const citationResolver = this.createCitationResolver();
    const citationEntries = await citationResolver.resolveCitationEntries(
      selected.flatMap((choice) => choice.item.candidate.episode.source_stream_ids),
    );
    const semantic = await resolveSemanticContext(query, options, {
      embeddingClient: this.options.embeddingClient,
      episodicRepository: this.options.episodicRepository,
      semanticNodeRepository: this.options.semanticNodeRepository,
      semanticGraph: this.options.semanticGraph,
    });
    const openQuestions =
      options.includeOpenQuestions === true
        ? this.retrieveOpenQuestionsForQuery(query, {
            relatedSemanticNodeIds: semantic.matchedNodeIds,
            audienceEntityId: options.audienceEntityId ?? null,
            limit: options.openQuestionsLimit,
          })
        : [];
    const results: RetrievedEpisode[] = [];

    for (const choice of selected) {
      const item = choice.item;
      const citationChain = citationResolver.resolveCitationChainFromMap(
        item.candidate.episode.source_stream_ids,
        citationEntries,
        options.traceTurnId,
      );
      const result = buildRetrievedEpisode(
        item.candidate,
        item.decayedSalience,
        item.heat,
        item.goalRelevance,
        item.valueAlignment,
        item.timeRelevance,
        item.moodBoost,
        item.socialRelevance,
        item.entityRelevance,
        item.suppressionPenalty,
        clamp(item.score, 0, 1),
        citationChain,
      );

      this.options.episodicRepository.recordRetrieval(
        item.candidate.episode.id,
        nowMs,
        result.score,
      );
      results.push(result);
    }

    const context = assembleRetrievedContext({
      episodes: results,
      semantic: toRetrievedSemantic(semantic),
      openQuestions,
      contradictionPresent: semantic.contradictionPresent,
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

  private tagCandidates(
    source: EpisodeCandidateSource,
    candidates: readonly EpisodeSearchCandidate[],
  ): MergedEpisodeCandidate[] {
    return tagEpisodeCandidates(source, candidates);
  }

  private mergeCandidates(
    candidateSets: readonly MergedEpisodeCandidate[][],
  ): MergedEpisodeCandidate[] {
    return mergeEpisodeCandidates(candidateSets);
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
