import { existsSync, readdirSync } from "node:fs";
import { basename } from "node:path";

import { computeGoalRelevance } from "../cognition/attention/goal-relevance.js";
import { computeValueAlignment } from "../cognition/attention/value-alignment.js";
import { extractEntitiesHeuristically } from "../cognition/perception/entity-extractor.js";
import type { AttentionWeights, TemporalCue } from "../cognition/types.js";
import type { EmbeddingClient } from "../embeddings/index.js";
import type { MoodState } from "../memory/affective/index.js";
import type { OpenQuestion, OpenQuestionsRepository, ValueRecord } from "../memory/self/index.js";
import { SemanticGraph } from "../memory/semantic/graph.js";
import type { SemanticNodeRepository } from "../memory/semantic/repository.js";
import type {
  SemanticContext,
  SemanticNode,
  SemanticRelation,
  SemanticWalkStep,
} from "../memory/semantic/types.js";
import type { SocialProfile } from "../memory/social/index.js";
import { StreamReader, getStreamDirectory, type StreamEntry } from "../stream/index.js";
import { SystemClock, type Clock } from "../util/clock.js";
import { StorageError } from "../util/errors.js";
import { tokenizeText } from "../util/text/tokenize.js";
import {
  DEFAULT_SESSION_ID,
  parseSessionId,
  type EntityId,
  type SessionId,
  type StreamEntryId,
} from "../util/ids.js";
import { applyEpisodeDecay, type DecayOptions } from "../memory/episodic/decay.js";
import { computeEpisodeHeat } from "../memory/episodic/heat.js";
import { isEpisodeVisibleToAudience } from "../memory/episodic/index.js";
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

type ResolvedTimeRange = {
  start: number;
  end: number;
};

export type RetrievedEpisode = {
  episode: Episode;
  score: number;
  scoreBreakdown: {
    similarity: number;
    decayedSalience: number;
    heat: number;
    goalRelevance: number;
    valueAlignment: number;
    timeRelevance: number;
    moodBoost: number;
    socialRelevance: number;
    entityRelevance: number;
    suppressionPenalty: number;
  };
  citationChain: StreamEntry[];
};

/**
 * Semantic-band retrieval output: matched root nodes for the query plus
 * nodes reached by graph walks. Lifted to the top level of RetrievedContext
 * so the semantic lane is addressable independently of episode retrieval
 * (pre-Phase-C it was nested inside each RetrievedEpisode with the same
 * value duplicated across episodes).
 */
export type RetrievedSemantic = SemanticContext & {
  matched_node_ids: SemanticNode["id"][];
  matched_nodes: SemanticNode[];
  support_hits: RetrievedSemanticHit[];
  contradiction_hits: RetrievedSemanticHit[];
  category_hits: RetrievedSemanticHit[];
};

export type RetrievedSemanticHit = {
  root_node_id: SemanticNode["id"];
  node: SemanticNode;
  edgePath: SemanticWalkStep["edgePath"];
};

/**
 * Unified per-turn retrieval context. Each band contributes an independent
 * section; none piggyback on another. Commitments and selected skill are
 * retrieved by the turn orchestrator (they depend on audience/entity
 * resolution the pipeline doesn't own) and are joined downstream.
 */
export type RetrievedContext = {
  episodes: RetrievedEpisode[];
  semantic: RetrievedSemantic;
  open_questions: OpenQuestion[];
  contradiction_present: boolean;
};

type EpisodeCandidateSource = "vector" | "temporal" | "audience" | "entity" | "recent" | "heat";

type MergedEpisodeCandidate = {
  candidate: EpisodeSearchCandidate;
  sources: Set<EpisodeCandidateSource>;
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
  valueAlignment: number,
  timeRelevance: number,
  moodBoost: number,
  socialRelevance: number,
  entityRelevance: number,
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
      valueAlignment,
      timeRelevance,
      moodBoost,
      socialRelevance,
      entityRelevance,
      suppressionPenalty,
    },
    citationChain,
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
  activeValues?: readonly ValueRecord[];
  temporalCue?: TemporalCue | null;
  strictTimeRange?: boolean;
  suppressionSet?: SuppressionLookup;
  graphWalkDepth?: number;
  maxGraphNodes?: number;
  includeOpenQuestions?: boolean;
  openQuestionsLimit?: number;
  moodState?: MoodState | null;
  audienceProfile?: SocialProfile | null;
  audienceTerms?: readonly string[];
  entityTerms?: readonly string[];
};

export type RetrievalGetEpisodeOptions = {
  audienceEntityId?: EntityId | null;
  crossAudience?: boolean;
};

function normalizeHeat(heat: number): number {
  return clamp(heat / 20, 0, 1);
}

function normalizeAttentionWeights(weights: AttentionWeights): AttentionWeights {
  return {
    ...weights,
    value_alignment:
      Number.isFinite((weights as Partial<AttentionWeights>).value_alignment) &&
      (weights as Partial<AttentionWeights>).value_alignment !== undefined
        ? weights.value_alignment
        : 0,
  };
}

function resolveTemporalCueTimeRange(
  temporalCue: TemporalCue | null | undefined,
): ResolvedTimeRange | null {
  if (temporalCue === null || temporalCue === undefined) {
    return null;
  }

  return {
    start: temporalCue.sinceTs ?? Number.NEGATIVE_INFINITY,
    end: temporalCue.untilTs ?? Number.POSITIVE_INFINITY,
  };
}

function resolveTimeSignals(
  options: Pick<RetrievalSearchOptions, "timeRange" | "temporalCue" | "strictTimeRange">,
): {
  scoringRange: ResolvedTimeRange | null;
  strictFilterRange: ResolvedTimeRange | null;
} {
  const temporalCueRange = resolveTemporalCueTimeRange(options.temporalCue);
  const explicitRange = options.timeRange ?? null;
  const effectiveRange = explicitRange ?? temporalCueRange;

  return {
    scoringRange: effectiveRange,
    strictFilterRange: options.strictTimeRange === true ? effectiveRange : null,
  };
}

function overlapsTimeRange(
  episode: Episode,
  range: ResolvedTimeRange,
): boolean {
  return episode.start_time <= range.end && episode.end_time >= range.start;
}

function computeTimeRelevance(
  episode: Episode,
  timeRange: ResolvedTimeRange | null,
): number {
  if (timeRange === null) {
    return 0;
  }

  return overlapsTimeRange(episode, timeRange) ? 1 : 0;
}

function normalizeTerm(value: string): string {
  return value.trim().toLowerCase();
}

function computeMoodBoost(episode: Episode, moodState: MoodState | null | undefined): number {
  if (
    moodState === null ||
    moodState === undefined ||
    Math.abs(moodState.valence) + Math.abs(moodState.arousal) <= 0.3 ||
    episode.emotional_arc === null
  ) {
    return 0;
  }

  const episodeValence =
    (episode.emotional_arc.start.valence +
      episode.emotional_arc.peak.valence +
      episode.emotional_arc.end.valence) /
    3;
  const episodeArousal =
    (episode.emotional_arc.start.arousal +
      episode.emotional_arc.peak.arousal +
      episode.emotional_arc.end.arousal) /
    3;

  return (
    (1 - Math.abs(moodState.valence - episodeValence) / 2) *
    (1 - Math.abs(moodState.arousal - episodeArousal) / 2)
  );
}

function computeSocialRelevance(
  episode: Episode,
  audienceTerms: readonly string[] | undefined,
  audienceProfile: SocialProfile | null | undefined,
): number {
  if (audienceTerms === undefined || audienceTerms.length === 0) {
    return 0;
  }

  const normalizedTerms = new Set(audienceTerms.map((term) => normalizeTerm(term)));
  const includesAudience = episode.participants.some((participant) =>
    normalizedTerms.has(normalizeTerm(participant)),
  );

  if (!includesAudience) {
    return 0;
  }

  return audienceProfile !== null && audienceProfile !== undefined && audienceProfile.trust > 0.7
    ? 0.25
    : 0.2;
}

function computeEntityRelevance(
  episode: Episode,
  entityTerms: readonly string[] | undefined,
): number {
  if (entityTerms === undefined || entityTerms.length === 0) {
    return 0;
  }

  const normalizedTerms = new Set(
    entityTerms.map((term) => normalizeTerm(term)).filter((term) => term.length > 0),
  );

  if (normalizedTerms.size === 0) {
    return 0;
  }

  return [...episode.participants, ...episode.tags].some((value) =>
    normalizedTerms.has(normalizeTerm(value)),
  )
    ? 1
    : 0;
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

  private async resolveVisibleEpisodeIds(
    episodeIds: readonly Episode["id"][],
    visibility: RetrievalGetEpisodeOptions,
  ): Promise<Set<string> | null> {
    if (visibility.crossAudience === true) {
      return null;
    }

    const uniqueEpisodeIds = [...new Set(episodeIds)];

    if (uniqueEpisodeIds.length === 0) {
      return new Set<string>();
    }

    const episodes = await this.options.episodicRepository.getMany(uniqueEpisodeIds);

    return new Set(
      episodes
        .filter((episode) =>
          isEpisodeVisibleToAudience(episode, visibility.audienceEntityId, {
            crossAudience: visibility.crossAudience,
          }),
        )
        .map((episode) => episode.id),
    );
  }

  private isSemanticNodeVisible(
    node: SemanticNode,
    visibleEpisodeIds: ReadonlySet<string> | null,
  ): boolean {
    return (
      visibleEpisodeIds === null ||
      node.source_episode_ids.every((episodeId) => visibleEpisodeIds.has(episodeId))
    );
  }

  private isSemanticWalkStepVisible(
    step: SemanticWalkStep,
    visibleEpisodeIds: ReadonlySet<string> | null,
  ): boolean {
    return (
      this.isSemanticNodeVisible(step.node, visibleEpisodeIds) &&
      (visibleEpisodeIds === null ||
        step.edgePath.every((edge) =>
          edge.evidence_episode_ids.every((episodeId) => visibleEpisodeIds.has(episodeId)),
        ))
    );
  }

  private async resolveSemanticContext(
    query: string,
    options: RetrievalSearchOptions,
  ): Promise<{
    context: SemanticContext;
    contradictionPresent: boolean;
    matchedNodeIds: SemanticNode["id"][];
    matchedNodes: SemanticNode[];
    supportHits: RetrievedSemanticHit[];
    contradictionHits: RetrievedSemanticHit[];
    categoryHits: RetrievedSemanticHit[];
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
        matchedNodes: [],
        supportHits: [],
        contradictionHits: [],
        categoryHits: [],
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

    const matchedNodeCandidates = [...new Map(matchedNodes.map((node) => [node.id, node])).values()];
    const matchedNodeVisibility = await this.resolveVisibleEpisodeIds(
      matchedNodeCandidates.flatMap((node) => node.source_episode_ids),
      options,
    );
    const uniqueNodes = new Map(
      matchedNodeCandidates
        .filter((node) => this.isSemanticNodeVisible(node, matchedNodeVisibility))
        .map((node) => [node.id, node] as const),
    );
    const supports = new Map<string, SemanticNode>();
    const contradicts = new Map<string, SemanticNode>();
    const categories = new Map<string, SemanticNode>();
    const walkDepth = options.graphWalkDepth ?? 2;
    const maxGraphNodes = options.maxGraphNodes ?? 16;
    const supportNeighbors: Array<{ rootNodeId: SemanticNode["id"]; step: SemanticWalkStep }> = [];
    const contradictionNeighbors: Array<{ rootNodeId: SemanticNode["id"]; step: SemanticWalkStep }> = [];
    const categoryNeighbors: Array<{ rootNodeId: SemanticNode["id"]; step: SemanticWalkStep }> = [];
    const supportHits: RetrievedSemanticHit[] = [];
    const contradictionHits: RetrievedSemanticHit[] = [];
    const categoryHits: RetrievedSemanticHit[] = [];

    for (const node of uniqueNodes.values()) {
      const walkedSupports = await this.options.semanticGraph.walk(node.id, {
        relations: ["supports"],
        direction: "both",
        depth: walkDepth,
        maxNodes: maxGraphNodes,
      });
      const walkedContradictions = await this.options.semanticGraph.walk(node.id, {
        relations: ["contradicts"],
        direction: "both",
        depth: walkDepth,
        maxNodes: maxGraphNodes,
      });
      const walkedCategories = await this.options.semanticGraph.walk(node.id, {
        relations: ["is_a"],
        direction: "out",
        depth: walkDepth,
        maxNodes: maxGraphNodes,
      });

      supportNeighbors.push(...walkedSupports.map((step) => ({ rootNodeId: node.id, step })));
      contradictionNeighbors.push(
        ...walkedContradictions.map((step) => ({ rootNodeId: node.id, step })),
      );
      categoryNeighbors.push(...walkedCategories.map((step) => ({ rootNodeId: node.id, step })));
    }

    const semanticVisibility = await this.resolveVisibleEpisodeIds(
      [
        ...[...uniqueNodes.values()].flatMap((node) => node.source_episode_ids),
        ...supportNeighbors.flatMap(({ step }) => [
          ...step.node.source_episode_ids,
          ...step.edgePath.flatMap((edge) => edge.evidence_episode_ids),
        ]),
        ...contradictionNeighbors.flatMap(({ step }) => [
          ...step.node.source_episode_ids,
          ...step.edgePath.flatMap((edge) => edge.evidence_episode_ids),
        ]),
        ...categoryNeighbors.flatMap(({ step }) => [
          ...step.node.source_episode_ids,
          ...step.edgePath.flatMap((edge) => edge.evidence_episode_ids),
        ]),
      ],
      options,
    );

    const visibleMatchedNodes = [...uniqueNodes.values()].filter((node) =>
      this.isSemanticNodeVisible(node, semanticVisibility),
    );

    for (const item of supportNeighbors) {
      if (!this.isSemanticWalkStepVisible(item.step, semanticVisibility)) {
        continue;
      }

      supports.set(item.step.node.id, item.step.node);
      supportHits.push({
        root_node_id: item.rootNodeId,
        node: item.step.node,
        edgePath: item.step.edgePath,
      });
    }

    for (const item of contradictionNeighbors) {
      if (!this.isSemanticWalkStepVisible(item.step, semanticVisibility)) {
        continue;
      }

      contradicts.set(item.step.node.id, item.step.node);
      contradictionHits.push({
        root_node_id: item.rootNodeId,
        node: item.step.node,
        edgePath: item.step.edgePath,
      });
    }

    for (const item of categoryNeighbors) {
      if (!this.isSemanticWalkStepVisible(item.step, semanticVisibility)) {
        continue;
      }

      categories.set(item.step.node.id, item.step.node);
      categoryHits.push({
        root_node_id: item.rootNodeId,
        node: item.step.node,
        edgePath: item.step.edgePath,
      });
    }

    return {
      context: {
        supports: [...supports.values()],
        contradicts: [...contradicts.values()],
        categories: [...categories.values()],
      },
      contradictionPresent: contradicts.size > 0,
      matchedNodeIds: visibleMatchedNodes.map((node) => node.id),
      matchedNodes: visibleMatchedNodes,
      supportHits,
      contradictionHits,
      categoryHits,
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

  private tagCandidates(
    source: EpisodeCandidateSource,
    candidates: readonly EpisodeSearchCandidate[],
  ): MergedEpisodeCandidate[] {
    return candidates.map((candidate) => ({
      candidate,
      sources: new Set([source]),
    }));
  }

  private mergeCandidates(
    candidateSets: readonly MergedEpisodeCandidate[][],
  ): MergedEpisodeCandidate[] {
    const merged = new Map<string, MergedEpisodeCandidate>();

    for (const candidateSet of candidateSets) {
      for (const entry of candidateSet) {
        const existing = merged.get(entry.candidate.episode.id);

        if (existing === undefined) {
          merged.set(entry.candidate.episode.id, {
            candidate: {
              ...entry.candidate,
            },
            sources: new Set(entry.sources),
          });
          continue;
        }

        existing.candidate = {
          ...existing.candidate,
          similarity: Math.max(existing.candidate.similarity, entry.candidate.similarity),
          stats: entry.candidate.stats,
        };

        for (const source of entry.sources) {
          existing.sources.add(source);
        }
      }
    }

    return [...merged.values()];
  }

  private async generateAudienceCandidates(
    audienceEntityId: EntityId,
    limit: number,
    visibleEpisodes: Promise<readonly Episode[]>,
  ): Promise<MergedEpisodeCandidate[]> {
    const recentLimit = Math.max(1, Math.ceil(limit / 2));
    const heatLimit = Math.max(1, limit - recentLimit);
    const sharedVisibleEpisodes = await visibleEpisodes;
    const [recent, hottest] = await Promise.all([
      this.options.episodicRepository.listByAudience(audienceEntityId, {
        limit: recentLimit,
        orderBy: "recent",
        visibleEpisodes: sharedVisibleEpisodes,
      }),
      this.options.episodicRepository.listByAudience(audienceEntityId, {
        limit: heatLimit,
        orderBy: "heat",
        visibleEpisodes: sharedVisibleEpisodes,
      }),
    ]);

    return this.mergeCandidates([
      this.tagCandidates("audience", recent),
      this.tagCandidates("audience", hottest),
    ]);
  }

  private async generateRecentAndHeatCandidates(
    options: RetrievalSearchOptions,
    limit: number,
    visibleEpisodes: Promise<readonly Episode[]>,
  ): Promise<MergedEpisodeCandidate[]> {
    const recentLimit = Math.max(1, Math.ceil(limit / 2));
    const heatLimit = Math.max(1, limit - recentLimit);
    const sharedVisibleEpisodes = await visibleEpisodes;
    const [recent, hottest] = await Promise.all([
      this.options.episodicRepository.listRecent({
        limit: recentLimit,
        audienceEntityId: options.audienceEntityId,
        crossAudience: options.crossAudience,
        visibleEpisodes: sharedVisibleEpisodes,
      }),
      this.options.episodicRepository.listHottest({
        limit: heatLimit,
        audienceEntityId: options.audienceEntityId,
        crossAudience: options.crossAudience,
        visibleEpisodes: sharedVisibleEpisodes,
      }),
    ]);

    return this.mergeCandidates([
      this.tagCandidates("recent", recent),
      this.tagCandidates("heat", hottest),
    ]);
  }

  private scoreCandidate(
    candidate: EpisodeSearchCandidate,
    searchOptions: RetrievalSearchOptions,
    nowMs: number,
    scoringTimeRange: ResolvedTimeRange | null,
  ): {
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
    const valueAlignment = computeValueAlignment(
      searchOptions.activeValues ?? [],
      candidate.episode,
    );
    const timeRelevance = computeTimeRelevance(candidate.episode, scoringTimeRange);
    const moodBoost = computeMoodBoost(candidate.episode, searchOptions.moodState);
    const socialRelevance = computeSocialRelevance(
      candidate.episode,
      searchOptions.audienceTerms,
      searchOptions.audienceProfile,
    );
    const entityRelevance = computeEntityRelevance(candidate.episode, searchOptions.entityTerms);
    const suppressionPenalty =
      searchOptions.suppressionSet?.isSuppressed(candidate.episode.id) === true ? 1 : 0;

    if (searchOptions.attentionWeights !== undefined) {
      const weights = normalizeAttentionWeights(searchOptions.attentionWeights);
      const semanticScore =
        weights.semantic * candidate.similarity + (1 - weights.semantic) * decay.decayedSalience;

      return {
        decayedSalience: decay.decayedSalience,
        heat,
        goalRelevance,
        valueAlignment,
        timeRelevance,
        moodBoost,
        socialRelevance,
        entityRelevance,
        suppressionPenalty,
        score:
          semanticScore +
          weights.goal_relevance * goalRelevance +
          weights.value_alignment * valueAlignment +
          weights.mood * moodBoost +
          weights.social * socialRelevance +
          weights.entity * entityRelevance +
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
      valueAlignment,
      timeRelevance,
      moodBoost,
      socialRelevance,
      entityRelevance,
      suppressionPenalty,
      score:
        weights.similarity * candidate.similarity +
        weights.salience * decay.decayedSalience +
        valueAlignment * 0.15 +
        entityRelevance * 0.15,
    };
  }

  async searchWithContext(
    query: string,
    options: RetrievalSearchOptions = {},
  ): Promise<RetrievedContext> {
    const nowMs = this.clock.now();
    const limit = Math.max(1, options.limit ?? 5);
    const queryVector = await this.options.embeddingClient.embed(query);
    const vectorBudget = Math.max(limit * 2, 12);
    const temporalBudget = Math.max(limit * 2, 8);
    const audienceBudget = Math.max(limit * 2, 8);
    const entityBudget = Math.max(limit * 2, 8);
    const recentHeatBudget = Math.max(limit, 4);
    const timeSignals = resolveTimeSignals(options);
    const visibleEpisodes = this.options.episodicRepository.listVisibleEpisodes({
      audienceEntityId: options.audienceEntityId,
      crossAudience: options.crossAudience,
    });
    const vectorSearchOptions: EpisodeSearchOptions = {
      ...options,
      timeRange: timeSignals.strictFilterRange ?? undefined,
    };
    const generatorCalls: Array<Promise<MergedEpisodeCandidate[]>> = [
      this.options.episodicRepository
        .searchByVector(queryVector, {
          ...vectorSearchOptions,
          limit: vectorBudget,
        })
        .then((candidates) => this.tagCandidates("vector", candidates)),
      this.generateRecentAndHeatCandidates(options, recentHeatBudget, visibleEpisodes),
    ];

    if (timeSignals.scoringRange !== null) {
      generatorCalls.push(
        this.options.episodicRepository
          .searchByTimeRange(timeSignals.scoringRange, {
            limit: temporalBudget,
            audienceEntityId: options.audienceEntityId,
            crossAudience: options.crossAudience,
          })
          .then((candidates) => this.tagCandidates("temporal", candidates)),
      );
    }

    if (
      options.audienceEntityId !== null &&
      options.audienceEntityId !== undefined &&
      options.crossAudience !== true
    ) {
      generatorCalls.push(
        this.generateAudienceCandidates(options.audienceEntityId, audienceBudget, visibleEpisodes),
      );
    }

    if (options.entityTerms !== undefined && options.entityTerms.length > 0) {
      generatorCalls.push(
        this.options.episodicRepository
          .searchByParticipantsOrTags(options.entityTerms, {
            limit: entityBudget,
            audienceEntityId: options.audienceEntityId,
            crossAudience: options.crossAudience,
            visibleEpisodes: await visibleEpisodes,
          })
          .then((candidates) => this.tagCandidates("entity", candidates)),
      );
    }

    const candidates = this.mergeCandidates(await Promise.all(generatorCalls)).filter((entry) =>
      timeSignals.strictFilterRange === null
        ? true
        : overlapsTimeRange(entry.candidate.episode, timeSignals.strictFilterRange),
    );
    if (timeSignals.strictFilterRange !== null && candidates.length === 0) {
      console.warn("Strict time filter returned 0 retrieval candidates.", {
        query,
        timeRange: timeSignals.strictFilterRange,
      });
    }
    const scored = candidates.map((entry) => {
      const candidate = entry.candidate;
      const score = this.scoreCandidate(candidate, options, nowMs, timeSignals.scoringRange);

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
    const citationEntries = await this.resolveCitationEntries(
      selected.flatMap((choice) => choice.item.candidate.episode.source_stream_ids),
    );
    const semantic = await this.resolveSemanticContext(query, options);
    const openQuestions =
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

    return {
      episodes: results,
      semantic: {
        supports: semantic.context.supports,
        contradicts: semantic.context.contradicts,
        categories: semantic.context.categories,
        matched_node_ids: semantic.matchedNodeIds,
        matched_nodes: semantic.matchedNodes,
        support_hits: semantic.supportHits,
        contradiction_hits: semantic.contradictionHits,
        category_hits: semantic.categoryHits,
      },
      open_questions: openQuestions,
      contradiction_present: semantic.contradictionPresent,
    };
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
    const candidate = {
      episode,
      stats,
      similarity: 1,
    };
    const scored = this.scoreCandidate(candidate, {}, nowMs, null);
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
}
