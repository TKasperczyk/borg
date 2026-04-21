import { existsSync, readdirSync } from "node:fs";
import { basename } from "node:path";

import { computeGoalRelevance } from "../cognition/attention/goal-relevance.js";
import { extractEntitiesHeuristically } from "../cognition/perception/entity-extractor.js";
import type { AttentionWeights, TemporalCue } from "../cognition/types.js";
import type { EmbeddingClient } from "../embeddings/index.js";
import type { OpenQuestion, OpenQuestionsRepository } from "../memory/self/index.js";
import { SemanticGraph } from "../memory/semantic/graph.js";
import type { SemanticNodeRepository } from "../memory/semantic/repository.js";
import type { SemanticContext, SemanticNode, SemanticRelation } from "../memory/semantic/types.js";
import { StreamReader, getStreamDirectory, type StreamEntry } from "../stream/index.js";
import { SystemClock, type Clock } from "../util/clock.js";
import { StorageError } from "../util/errors.js";
import { tokenizeText } from "../util/text/tokenize.js";
import {
  DEFAULT_SESSION_ID,
  parseSessionId,
  type SessionId,
  type StreamEntryId,
} from "../util/ids.js";
import { applyEpisodeDecay, type DecayOptions } from "../memory/episodic/decay.js";
import { computeEpisodeHeat } from "../memory/episodic/heat.js";
import {
  type Episode,
  type EpisodeSearchCandidate,
  type EpisodeSearchOptions,
  type EpisodeStats,
} from "../memory/episodic/types.js";
import { EpisodicRepository } from "../memory/episodic/repository.js";

import { applyMmr } from "./mmr.js";

type SuppressionLookup = {
  isSuppressed(id: string): boolean;
};

export type RetrievedEpisode = {
  episode: Episode;
  score: number;
  scoreBreakdown: {
    similarity: number;
    decayedSalience: number;
    heat: number;
    goalRelevance: number;
    timeRelevance: number;
    suppressionPenalty: number;
  };
  citationChain: StreamEntry[];
  semantic_context: SemanticContext;
};

export type RetrievalSearchResult = {
  episodes: RetrievedEpisode[];
  contradiction_present: boolean;
  open_questions_context: OpenQuestion[];
};

export type RetrievalPipelineOptions = {
  embeddingClient: EmbeddingClient;
  episodicRepository: EpisodicRepository;
  dataDir: string;
  semanticNodeRepository?: SemanticNodeRepository;
  semanticGraph?: SemanticGraph;
  openQuestionsRepository?: OpenQuestionsRepository;
  clock?: Clock;
  scoreWeights?: {
    similarity: number;
    salience: number;
  };
  mmrLambda?: number;
  decayOptions?: Omit<DecayOptions, "nowMs">;
};

function clamp(value: number, min: number, max: number): number {
  return Math.min(max, Math.max(min, value));
}

function parseSessionIdFromFilename(filename: string): SessionId | null {
  if (!filename.endsWith(".jsonl")) {
    return null;
  }

  const sessionName = basename(filename, ".jsonl");

  try {
    return parseSessionId(sessionName);
  } catch {
    return null;
  }
}

function defaultDecayOptions(nowMs: number): DecayOptions {
  return {
    nowMs,
    baseHalfLifeHours: 24 * 7,
    halfLifeByTier: {
      T1: 24 * 3,
      T2: 24 * 7,
      T3: 24 * 14,
      T4: 24 * 30,
    },
  };
}

function buildResult(
  candidate: EpisodeSearchCandidate,
  decayedSalience: number,
  heat: number,
  goalRelevance: number,
  timeRelevance: number,
  suppressionPenalty: number,
  score: number,
  citationChain: StreamEntry[],
): RetrievedEpisode {
  return {
    episode: candidate.episode,
    score,
    scoreBreakdown: {
      similarity: candidate.similarity,
      decayedSalience,
      heat,
      goalRelevance,
      timeRelevance,
      suppressionPenalty,
    },
    citationChain,
    semantic_context: {
      supports: [],
      contradicts: [],
      categories: [],
    },
  };
}

export type RetrievalSearchOptions = EpisodeSearchOptions & {
  limit?: number;
  mmrLambda?: number;
  scoreWeights?: {
    similarity: number;
    salience: number;
  };
  decayOptions?: Omit<DecayOptions, "nowMs">;
  attentionWeights?: AttentionWeights;
  goalDescriptions?: readonly string[];
  temporalCue?: TemporalCue | null;
  suppressionSet?: SuppressionLookup;
  graphWalkDepth?: number;
  maxGraphNodes?: number;
  includeOpenQuestions?: boolean;
  openQuestionsLimit?: number;
};

function normalizeHeat(heat: number): number {
  return clamp(heat / 20, 0, 1);
}

function computeTimeRelevance(
  episode: Episode,
  temporalCue: TemporalCue | null | undefined,
): number {
  if (temporalCue === null || temporalCue === undefined) {
    return 0;
  }

  const sinceTs = temporalCue.sinceTs ?? Number.NEGATIVE_INFINITY;
  const untilTs = temporalCue.untilTs ?? Number.POSITIVE_INFINITY;
  return episode.start_time <= untilTs && episode.end_time >= sinceTs ? 1 : 0;
}

export class RetrievalPipeline {
  private readonly clock: Clock;
  private readonly scoreWeights: {
    similarity: number;
    salience: number;
  };
  private readonly mmrLambda: number;
  private readonly decayOptions?: Omit<DecayOptions, "nowMs">;

  constructor(private readonly options: RetrievalPipelineOptions) {
    this.clock = options.clock ?? new SystemClock();
    this.scoreWeights = options.scoreWeights ?? {
      similarity: 0.7,
      salience: 0.3,
    };
    this.mmrLambda = options.mmrLambda ?? 0.7;
    this.decayOptions = options.decayOptions;
  }

  private listSessionIds(): SessionId[] {
    const streamDir = getStreamDirectory(this.options.dataDir);

    if (!existsSync(streamDir)) {
      return [DEFAULT_SESSION_ID];
    }

    const sessionIds = readdirSync(streamDir)
      .map((filename) => parseSessionIdFromFilename(filename))
      .filter((value): value is SessionId => value !== null);

    return sessionIds.length === 0 ? [DEFAULT_SESSION_ID] : sessionIds;
  }

  private async resolveCitationEntries(
    sourceStreamIds: readonly StreamEntryId[],
  ): Promise<Map<string, StreamEntry>> {
    const entries = new Map<string, StreamEntry>();
    const pendingIds = new Set<string>(sourceStreamIds);

    if (pendingIds.size === 0) {
      return entries;
    }

    const sessionIds = this.listSessionIds();

    for (const sessionId of sessionIds) {
      const reader = new StreamReader({
        dataDir: this.options.dataDir,
        sessionId,
      });

      for await (const entry of reader.iterate()) {
        if (!pendingIds.has(entry.id)) {
          continue;
        }

        entries.set(entry.id, entry);
        pendingIds.delete(entry.id);

        if (pendingIds.size === 0) {
          break;
        }
      }

      if (pendingIds.size === 0) {
        break;
      }
    }

    return entries;
  }

  private resolveCitationChainFromMap(
    sourceStreamIds: readonly StreamEntryId[],
    entries: ReadonlyMap<string, StreamEntry>,
  ): StreamEntry[] {
    return sourceStreamIds
      .map((sourceId) => entries.get(sourceId))
      .filter((entry): entry is StreamEntry => entry !== undefined);
  }

  private async resolveSemanticContext(
    query: string,
    options: RetrievalSearchOptions,
  ): Promise<{
    context: SemanticContext;
    contradictionPresent: boolean;
    matchedNodeIds: SemanticNode["id"][];
  }> {
    if (
      this.options.semanticNodeRepository === undefined ||
      this.options.semanticGraph === undefined
    ) {
      return {
        context: {
          supports: [],
          contradicts: [],
          categories: [],
        },
        contradictionPresent: false,
        matchedNodeIds: [],
      };
    }

    const labels = [...extractEntitiesHeuristically(query), ...query.split(/[,\n]+/)]
      .map((value) => value.trim())
      .filter((value) => value.length > 0);
    const matchedNodes: SemanticNode[] = [];

    for (const label of labels) {
      const matches = await this.options.semanticNodeRepository.findByLabelOrAlias(label, 3);
      matchedNodes.push(...matches);
    }

    if (matchedNodes.length === 0) {
      const queryVector = await this.options.embeddingClient.embed(query);
      const byVector = await this.options.semanticNodeRepository.searchByVector(queryVector, {
        limit: 3,
        includeArchived: false,
      });
      matchedNodes.push(...byVector.map((item) => item.node));
    }

    const uniqueNodes = new Map(matchedNodes.map((node) => [node.id, node]));
    const supports = new Map<string, SemanticNode>();
    const contradicts = new Map<string, SemanticNode>();
    const categories = new Map<string, SemanticNode>();
    const walkDepth = options.graphWalkDepth ?? 2;
    const maxGraphNodes = options.maxGraphNodes ?? 16;

    for (const node of uniqueNodes.values()) {
      const supportNeighbors = await this.options.semanticGraph.walk(node.id, {
        relations: ["supports"],
        depth: walkDepth,
        maxNodes: maxGraphNodes,
      });
      const contradictionNeighbors = await this.options.semanticGraph.walk(node.id, {
        relations: ["contradicts"],
        depth: walkDepth,
        maxNodes: maxGraphNodes,
      });
      const categoryNeighbors = await this.options.semanticGraph.walk(node.id, {
        relations: ["is_a"],
        depth: walkDepth,
        maxNodes: maxGraphNodes,
      });

      for (const item of supportNeighbors) {
        supports.set(item.node.id, item.node);
      }

      for (const item of contradictionNeighbors) {
        contradicts.set(item.node.id, item.node);
      }

      for (const item of categoryNeighbors) {
        categories.set(item.node.id, item.node);
      }
    }

    return {
      context: {
        supports: [...supports.values()],
        contradicts: [...contradicts.values()],
        categories: [...categories.values()],
      },
      contradictionPresent: contradicts.size > 0,
      matchedNodeIds: [...uniqueNodes.keys()],
    };
  }

  retrieveOpenQuestionsForQuery(
    query: string,
    options: {
      relatedSemanticNodeIds?: readonly SemanticNode["id"][];
      limit?: number;
    } = {},
  ): OpenQuestion[] {
    if (this.options.openQuestionsRepository === undefined) {
      return [];
    }

    const queryTokens = tokenizeText(query);
    const relatedNodeIds = new Set(options.relatedSemanticNodeIds ?? []);
    const limit = Math.max(1, options.limit ?? 3);
    const candidates = this.options.openQuestionsRepository.list({
      status: "open",
      limit: 100,
    });
    const scored = candidates
      .map((question) => {
        const questionTokens = tokenizeText(question.question);
        let overlap = 0;

        for (const token of queryTokens) {
          if (questionTokens.has(token)) {
            overlap += 1;
          }
        }

        const unionSize = new Set([...queryTokens, ...questionTokens]).size;
        const tokenScore = unionSize === 0 ? 0 : overlap / unionSize;
        const relatedScore = question.related_semantic_node_ids.some((id) => relatedNodeIds.has(id))
          ? 0.35
          : 0;
        const score =
          tokenScore === 0 && relatedScore === 0
            ? 0
            : tokenScore + relatedScore + question.urgency * 0.15;

        return {
          question,
          score,
        };
      })
      .filter((item) => item.score > 0)
      .sort(
        (left, right) =>
          right.score - left.score ||
          right.question.urgency - left.question.urgency ||
          right.question.last_touched - left.question.last_touched,
      );

    return scored.slice(0, limit).map((item) => item.question);
  }

  private scoreCandidate(
    candidate: EpisodeSearchCandidate,
    searchOptions: RetrievalSearchOptions,
    nowMs: number,
  ): {
    decayedSalience: number;
    heat: number;
    goalRelevance: number;
    timeRelevance: number;
    suppressionPenalty: number;
    score: number;
  } {
    const decay = applyEpisodeDecay(
      candidate.episode,
      candidate.stats,
      searchOptions.decayOptions === undefined
        ? this.decayOptions === undefined
          ? defaultDecayOptions(nowMs)
          : { ...this.decayOptions, nowMs }
        : { ...searchOptions.decayOptions, nowMs },
    );
    const heat = computeEpisodeHeat(candidate.episode, candidate.stats, nowMs);
    const goalRelevance = computeGoalRelevance(
      searchOptions.goalDescriptions ?? [],
      candidate.episode,
    );
    const timeRelevance = computeTimeRelevance(candidate.episode, searchOptions.temporalCue);
    const suppressionPenalty =
      searchOptions.suppressionSet?.isSuppressed(candidate.episode.id) === true ? 1 : 0;

    if (searchOptions.attentionWeights !== undefined) {
      const weights = searchOptions.attentionWeights;
      const semanticScore =
        weights.semantic * candidate.similarity + (1 - weights.semantic) * decay.decayedSalience;

      return {
        decayedSalience: decay.decayedSalience,
        heat,
        goalRelevance,
        timeRelevance,
        suppressionPenalty,
        score:
          semanticScore +
          weights.goal_relevance * goalRelevance +
          weights.time * timeRelevance +
          weights.heat * normalizeHeat(heat) -
          weights.suppression_penalty * suppressionPenalty,
      };
    }

    const weights = searchOptions.scoreWeights ?? this.scoreWeights;

    return {
      decayedSalience: decay.decayedSalience,
      heat,
      goalRelevance,
      timeRelevance,
      suppressionPenalty,
      score: weights.similarity * candidate.similarity + weights.salience * decay.decayedSalience,
    };
  }

  async searchWithContext(
    query: string,
    options: RetrievalSearchOptions = {},
  ): Promise<RetrievalSearchResult> {
    const nowMs = this.clock.now();
    const limit = Math.max(1, options.limit ?? 5);
    const queryVector = await this.options.embeddingClient.embed(query);
    const candidates = await this.options.episodicRepository.searchByVector(queryVector, {
      ...options,
      limit: Math.max(limit * 3, limit),
    });
    const scored = candidates.map((candidate) => {
      const score = this.scoreCandidate(candidate, options, nowMs);

      return {
        candidate,
        ...score,
      };
    });
    const selected = applyMmr(
      scored.map((item) => ({
        item,
        vector: item.candidate.episode.embedding,
        relevanceScore: item.score,
      })),
      {
        limit,
        lambda: options.mmrLambda ?? this.mmrLambda,
      },
    );
    const citationEntries = await this.resolveCitationEntries(
      selected.flatMap((choice) => choice.item.candidate.episode.source_stream_ids),
    );
    const semantic = await this.resolveSemanticContext(query, options);
    const openQuestionsContext =
      options.includeOpenQuestions === true
        ? this.retrieveOpenQuestionsForQuery(query, {
            relatedSemanticNodeIds: semantic.matchedNodeIds,
            limit: options.openQuestionsLimit,
          })
        : [];
    const results: RetrievedEpisode[] = [];

    for (const choice of selected) {
      const item = choice.item;
      const citationChain = this.resolveCitationChainFromMap(
        item.candidate.episode.source_stream_ids,
        citationEntries,
      );
      const result = buildResult(
        item.candidate,
        item.decayedSalience,
        item.heat,
        item.goalRelevance,
        item.timeRelevance,
        item.suppressionPenalty,
        clamp(item.score, 0, 1),
        citationChain,
      );
      result.semantic_context = semantic.context;

      this.options.episodicRepository.recordRetrieval(
        item.candidate.episode.id,
        nowMs,
        result.score,
      );
      results.push(result);
    }

    return {
      episodes: results,
      contradiction_present: semantic.contradictionPresent,
      open_questions_context: openQuestionsContext,
    };
  }

  async search(query: string, options: RetrievalSearchOptions = {}): Promise<RetrievedEpisode[]> {
    const result = await this.searchWithContext(query, options);
    return result.episodes;
  }

  async getEpisode(id: Episode["id"]): Promise<RetrievedEpisode | null> {
    const episode = await this.options.episodicRepository.get(id);

    if (episode === null) {
      return null;
    }

    const stats = this.options.episodicRepository.getStats(id);

    if (stats === null) {
      throw new StorageError(`Missing episode stats for ${id}`, {
        code: "EPISODE_STATS_MISSING",
      });
    }

    const nowMs = this.clock.now();
    const candidate = {
      episode,
      stats,
      similarity: 1,
    };
    const scored = this.scoreCandidate(candidate, {}, nowMs);
    const citationEntries = await this.resolveCitationEntries(episode.source_stream_ids);
    const citationChain = this.resolveCitationChainFromMap(
      episode.source_stream_ids,
      citationEntries,
    );

    return buildResult(
      candidate,
      scored.decayedSalience,
      scored.heat,
      scored.goalRelevance,
      scored.timeRelevance,
      scored.suppressionPenalty,
      1,
      citationChain,
    );
  }
}
