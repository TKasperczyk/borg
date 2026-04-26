/* Episodic scoring and result construction for retrieval. */
import { computeGoalRelevance } from "../cognition/attention/goal-relevance.js";
import { computeValueAlignment } from "../cognition/attention/value-alignment.js";
import type { AttentionWeights } from "../cognition/types.js";
import type { MoodState } from "../memory/affective/index.js";
import { applyEpisodeDecay, type DecayOptions } from "../memory/episodic/decay.js";
import { computeEpisodeHeat } from "../memory/episodic/heat.js";
import type { Episode, EpisodeSearchCandidate } from "../memory/episodic/types.js";
import type { ValueRecord } from "../memory/self/index.js";
import type { SocialProfile } from "../memory/social/index.js";
import type { StreamEntry } from "../stream/index.js";

import { computeTimeRelevance, type ResolvedTimeRange } from "./time-signals.js";

export type SuppressionLookup = {
  isSuppressed(id: string): boolean;
};

export type ScoreWeights = {
  similarity: number;
  salience: number;
};

export type RetrievalMoodState = Pick<MoodState, "valence" | "arousal">;

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

export type EpisodeScoringOptions = {
  scoreWeights?: ScoreWeights;
  decayOptions?: Omit<DecayOptions, "nowMs">;
  attentionWeights?: AttentionWeights;
  goalDescriptions?: readonly string[];
  activeValues?: readonly ValueRecord[];
  moodState?: RetrievalMoodState | null;
  audienceProfile?: SocialProfile | null;
  audienceTerms?: readonly string[];
  entityTerms?: readonly string[];
  suppressionSet?: SuppressionLookup;
};

export type EpisodeScoreDefaults = {
  scoreWeights: ScoreWeights;
  decayOptions?: Omit<DecayOptions, "nowMs">;
};

export type EpisodeScore = {
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

export function clamp(value: number, min: number, max: number): number {
  return Math.min(max, Math.max(min, value));
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

function normalizeTerm(value: string): string {
  return value.trim().toLowerCase();
}

function computeMoodBoost(
  episode: Episode,
  moodState: RetrievalMoodState | null | undefined,
): number {
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

export function scoreCandidate(
  candidate: EpisodeSearchCandidate,
  searchOptions: EpisodeScoringOptions,
  nowMs: number,
  scoringTimeRange: ResolvedTimeRange | null,
  defaults: EpisodeScoreDefaults,
): EpisodeScore {
  const decay = applyEpisodeDecay(
    candidate.episode,
    candidate.stats,
    searchOptions.decayOptions === undefined
      ? defaults.decayOptions === undefined
        ? defaultDecayOptions(nowMs)
        : { ...defaults.decayOptions, nowMs }
      : { ...searchOptions.decayOptions, nowMs },
  );
  const heat = computeEpisodeHeat(candidate.episode, candidate.stats, nowMs);
  const goalRelevance = computeGoalRelevance(
    searchOptions.goalDescriptions ?? [],
    candidate.episode,
  );
  const valueAlignment = computeValueAlignment(searchOptions.activeValues ?? [], candidate.episode);
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

  const weights = searchOptions.scoreWeights ?? defaults.scoreWeights;

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

export function buildRetrievedEpisode(
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
