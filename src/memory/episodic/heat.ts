import type { Episode, EpisodeStats } from "./types.js";

const RECENCY_HALF_LIFE_MS = 7 * 24 * 60 * 60 * 1000;

export function computeEpisodeHeat(episode: Episode, stats: EpisodeStats, nowMs: number): number {
  const referenceTimestamp = stats.last_retrieved ?? episode.updated_at;
  const elapsedMs = Math.max(0, nowMs - referenceTimestamp);
  const recencyScore =
    referenceTimestamp <= 0 ? 0 : Math.pow(0.5, elapsedMs / RECENCY_HALF_LIFE_MS);

  return stats.retrieval_count + 2 * (stats.win_rate * 10) + 0.5 * (recencyScore * 10);
}
