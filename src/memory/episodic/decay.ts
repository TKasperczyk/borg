import type { Episode, EpisodeStats, EpisodeTier } from "./types.js";
import { clamp, halfLifeDecay } from "../../util/math.js";

export type DecayOptions = {
  nowMs: number;
  baseHalfLifeHours: number;
  halfLifeByTier?: Partial<Record<EpisodeTier, number>>;
};

export type DecayResult = {
  decayedSalience: number;
  effectiveHalfLifeHours: number;
};

// DELIBERATE: episodic 'heat' is a persisted column that gates DB-level candidate PRE-selection (listHottest in src/retrieval/pipeline.ts), not just a ranking weight -- it changes which episodes are even considered. It is one of several distinct aging timers (heat, shared-state salience/lifecycle-aging, warm-recall TTL, working-memory suppression TTL) that look like duplicated decay but are NOT consolidatable: they age different stores for different consumers on two incompatible time bases -- wall-clock exponential half-life vs integer turn-count TTL. The shared half-life arithmetic is already centralized in util/math.halfLifeDecay; the timers themselves are deliberately separate (Tier-3 review).

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
    episode.significance * halfLifeDecay(elapsedHours, effectiveHalfLifeHours),
    0,
    1,
  );

  return {
    decayedSalience,
    effectiveHalfLifeHours,
  };
}
