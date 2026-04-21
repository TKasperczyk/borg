import type { Episode, EpisodeStats, EpisodeTier } from "./types.js";

export type DecayOptions = {
  nowMs: number;
  baseHalfLifeHours: number;
  halfLifeByTier?: Partial<Record<EpisodeTier, number>>;
};

export type DecayResult = {
  decayedSalience: number;
  effectiveHalfLifeHours: number;
};

function clamp(value: number, min: number, max: number): number {
  return Math.min(max, Math.max(min, value));
}

function getTierHalfLifeHours(stats: EpisodeStats, options: DecayOptions): number {
  const tierHalfLife = options.halfLifeByTier?.[stats.tier];
  return tierHalfLife ?? options.baseHalfLifeHours;
}

function getModulatedHalfLifeHours(stats: EpisodeStats, options: DecayOptions): number {
  let halfLifeHours = getTierHalfLifeHours(stats, options);

  if (stats.retrieval_count === 0) {
    halfLifeHours *= 0.5;
  }

  if (stats.use_count >= 3 && stats.win_rate >= 0.7) {
    halfLifeHours *= 2;
  }

  return Math.max(0.01, halfLifeHours);
}

function getReferenceTimestamp(episode: Episode, stats: EpisodeStats): number {
  return Math.max(
    episode.updated_at,
    stats.promoted_at,
    stats.last_retrieved ?? 0,
    stats.last_decayed_at ?? 0,
  );
}

export function applyEpisodeDecay(
  episode: Episode,
  stats: EpisodeStats,
  options: DecayOptions,
): DecayResult {
  const effectiveHalfLifeHours = getModulatedHalfLifeHours(stats, options);
  const referenceTimestamp = getReferenceTimestamp(episode, stats);
  const elapsedHours = Math.max(0, options.nowMs - referenceTimestamp) / 3_600_000;
  const decayedSalience = clamp(
    episode.significance * Math.pow(0.5, elapsedHours / effectiveHalfLifeHours),
    0,
    1,
  );

  return {
    decayedSalience,
    effectiveHalfLifeHours,
  };
}
