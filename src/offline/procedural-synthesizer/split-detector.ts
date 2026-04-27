import type { SkillContextStatsRecord, SkillRecord } from "../../memory/procedural/index.js";

const DAY_MS = 24 * 60 * 60 * 1_000;

export type SkillSplitBucket = {
  stats: SkillContextStatsRecord;
  posterior_mean: number;
};

export type SkillSplitCandidate = {
  skill: SkillRecord;
  buckets: SkillSplitBucket[];
  min_posterior_mean: number;
  max_posterior_mean: number;
  divergence: number;
};

export type DetectDivergentSkillSplitsInput = {
  skills: readonly SkillRecord[];
  contextStatsBySkillId: ReadonlyMap<SkillRecord["id"], readonly SkillContextStatsRecord[]>;
  nowMs: number;
  minContextAttemptsForSplit: number;
  minDivergenceForSplit: number;
  splitCooldownDays: number;
  splitClaimStaleSec: number;
};

function posteriorMean(stats: SkillContextStatsRecord): number {
  return stats.alpha / (stats.alpha + stats.beta);
}

function isInCooldown(
  skill: SkillRecord,
  nowMs: number,
  cooldownMs: number,
  claimStaleMs: number,
): boolean {
  if (skill.splitting_at !== null && nowMs - skill.splitting_at < claimStaleMs) {
    return true;
  }

  if (
    skill.last_split_attempt_at !== undefined &&
    skill.last_split_attempt_at !== null &&
    nowMs - skill.last_split_attempt_at < cooldownMs
  ) {
    return true;
  }

  return skill.superseded_at !== null && nowMs - skill.superseded_at < cooldownMs;
}

export function detectDivergentSkillSplits(
  input: DetectDivergentSkillSplitsInput,
): SkillSplitCandidate[] {
  const cooldownMs = input.splitCooldownDays * DAY_MS;
  const claimStaleMs = input.splitClaimStaleSec * 1_000;
  const candidates: SkillSplitCandidate[] = [];

  for (const skill of input.skills) {
    if (
      skill.status !== "active" ||
      skill.superseded_by.length > 0 ||
      isInCooldown(skill, input.nowMs, cooldownMs, claimStaleMs)
    ) {
      continue;
    }

    const buckets = (input.contextStatsBySkillId.get(skill.id) ?? [])
      .filter((stats) => stats.attempts >= input.minContextAttemptsForSplit)
      .map((stats) => ({
        stats,
        posterior_mean: posteriorMean(stats),
      }))
      .sort((left, right) => left.stats.context_key.localeCompare(right.stats.context_key));

    if (buckets.length < 2) {
      continue;
    }

    const means = buckets.map((bucket) => bucket.posterior_mean);
    const minPosteriorMean = Math.min(...means);
    const maxPosteriorMean = Math.max(...means);
    const divergence = maxPosteriorMean - minPosteriorMean;

    if (divergence < input.minDivergenceForSplit) {
      continue;
    }

    candidates.push({
      skill,
      buckets,
      min_posterior_mean: minPosteriorMean,
      max_posterior_mean: maxPosteriorMean,
      divergence,
    });
  }

  return candidates.sort(
    (left, right) =>
      right.divergence - left.divergence || left.skill.updated_at - right.skill.updated_at,
  );
}
