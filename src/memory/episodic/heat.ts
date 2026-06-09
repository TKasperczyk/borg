import type { Episode, EpisodeStats } from "./types.js";

// Tunes the recency decay half-life used by episodic heat.
const RECENCY_HALF_LIFE_MS = 7 * 24 * 60 * 60 * 1000;

// Tunes the maximum retrieval-count contribution to episodic heat.
export const RETRIEVAL_HEAT_CAP = 40;

// Tunes the exponential base for heat recency decay.
const EPISODE_HEAT_RECENCY_DECAY_BASE = 0.5;

// Tunes the win-rate contribution to episodic heat.
const EPISODE_HEAT_WIN_RATE_WEIGHT = 2;

// Tunes the win-rate scale before applying the heat weight.
const EPISODE_HEAT_WIN_RATE_SCALE = 10;

// Tunes the recency contribution to episodic heat.
const EPISODE_HEAT_RECENCY_WEIGHT = 0.5;

// Tunes the recency scale before applying the heat weight.
const EPISODE_HEAT_RECENCY_SCALE = 10;

// Tunes the default multiplier for episode heat when no stats multiplier exists.
const DEFAULT_EPISODE_HEAT_MULTIPLIER = 1;

export function computeEpisodeHeatForTimestamp(
  updatedAt: number,
  stats: EpisodeStats,
  nowMs: number,
): number {
  const referenceTimestamp = stats.last_retrieved ?? updatedAt;
  const elapsedMs = Math.max(0, nowMs - referenceTimestamp);
  const recencyScore =
    referenceTimestamp <= 0
      ? 0
      : Math.pow(EPISODE_HEAT_RECENCY_DECAY_BASE, elapsedMs / RECENCY_HALF_LIFE_MS);
  const heatMultiplier = stats.heat_multiplier ?? DEFAULT_EPISODE_HEAT_MULTIPLIER;
  const retrievalHeat = Math.min(stats.retrieval_count, RETRIEVAL_HEAT_CAP);

  return (
    (retrievalHeat +
      EPISODE_HEAT_WIN_RATE_WEIGHT * (stats.win_rate * EPISODE_HEAT_WIN_RATE_SCALE) +
      EPISODE_HEAT_RECENCY_WEIGHT * (recencyScore * EPISODE_HEAT_RECENCY_SCALE)) *
    heatMultiplier
  );
}

export function computeEpisodeHeat(episode: Episode, stats: EpisodeStats, nowMs: number): number {
  return computeEpisodeHeatForTimestamp(episode.updated_at, stats, nowMs);
}
