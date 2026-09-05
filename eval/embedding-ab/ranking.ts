import { cosineSimilarity } from "../../src/retrieval/embedding-similarity.js";

import type { CrossModelRank, EpisodeDocument, RankMetrics, RankedEpisode } from "./types.js";

export type EpisodeVector = {
  episode: EpisodeDocument;
  vector: Float32Array;
};

export function commonEpisodeIds(
  episodeIds: readonly string[],
  vectorsByModel: readonly ReadonlyMap<string, Float32Array>[],
): string[] {
  if (vectorsByModel.length === 0) {
    return [];
  }

  return episodeIds.filter((episodeId) =>
    vectorsByModel.every((vectors) => vectors.has(episodeId)),
  );
}

export function rankEpisodes(
  queryVector: Float32Array,
  candidates: readonly EpisodeVector[],
): RankedEpisode[] {
  return candidates
    .map(({ episode, vector }) => ({
      rank: 0,
      episode_id: episode.id,
      title: episode.title,
      cosine_similarity: cosineSimilarity(queryVector, vector),
    }))
    .sort(
      (left, right) =>
        right.cosine_similarity - left.cosine_similarity ||
        left.episode_id.localeCompare(right.episode_id),
    )
    .map((candidate, index) => ({ ...candidate, rank: index + 1 }));
}

export function summarizeRanks(ranks: readonly (number | null)[]): RankMetrics {
  if (ranks.length === 0) {
    return {
      question_count: 0,
      ranked_source_count: 0,
      recall_at_1: null,
      recall_at_3: null,
      recall_at_10: null,
      mrr: null,
    };
  }

  const ranked = ranks.filter((rank): rank is number => rank !== null);
  const recallAt = (limit: number): number =>
    ranks.reduce<number>((hits, rank) => hits + (rank !== null && rank <= limit ? 1 : 0), 0) /
    ranks.length;

  return {
    question_count: ranks.length,
    ranked_source_count: ranked.length,
    recall_at_1: recallAt(1),
    recall_at_3: recallAt(3),
    recall_at_10: recallAt(10),
    mrr:
      ranks.reduce<number>((sum, rank) => sum + (rank === null ? 0 : 1 / rank), 0) / ranks.length,
  };
}

export function topKOverlap(
  left: readonly RankedEpisode[],
  right: readonly RankedEpisode[],
  k: number,
): { count: number; denominator: number; ratio: number | null } {
  const leftIds = new Set(left.slice(0, k).map((candidate) => candidate.episode_id));
  const rightTop = right.slice(0, k);
  const count = rightTop.reduce(
    (overlap, candidate) => overlap + (leftIds.has(candidate.episode_id) ? 1 : 0),
    0,
  );
  const denominator = Math.min(k, left.length, right.length);

  return {
    count,
    denominator,
    ratio: denominator === 0 ? null : count / denominator,
  };
}

export function ranksOfSourceTopKInTarget(
  source: readonly RankedEpisode[],
  target: readonly RankedEpisode[],
  k: number,
): CrossModelRank[] {
  const targetRanks = new Map(target.map((candidate) => [candidate.episode_id, candidate.rank]));

  return source.slice(0, k).map((candidate) => ({
    episode_id: candidate.episode_id,
    source_rank: candidate.rank,
    target_rank: targetRanks.get(candidate.episode_id) ?? null,
  }));
}
