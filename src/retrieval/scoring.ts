/* Episodic scoring and result construction for retrieval. */
import { computeGoalRelevance, computeValueAlignment } from "./attention-relevance.js";
import type { AttentionWeights } from "../contracts/cognitive-contracts.js";
import { MOOD_ACTIVITY_THRESHOLD, type MoodState } from "../memory/affective/index.js";
import { applyEpisodeDecay, type DecayOptions } from "../memory/episodic/decay.js";
import { computeEpisodeHeat, RETRIEVAL_HEAT_CAP } from "../memory/episodic/heat.js";
import type { Episode, EpisodeSearchCandidate } from "../memory/episodic/types.js";
import type { ValueRecord } from "../memory/self/index.js";
import type { SocialProfile } from "../memory/social/index.js";
import type { StreamEntry } from "../stream/index.js";
import type { EntityId } from "../util/ids.js";
import { clamp } from "../util/math.js";

import { computeTimeRelevance, type ResolvedTimeRange } from "./time-signals.js";
import type { RetrievalScoringFeatures } from "./scoring-features.js";
import {
  memoryDisclosureLabelFromEpisodeAccess,
  type MemoryDisclosureLabel,
} from "./recall-context.js";

export { clamp } from "../util/math.js";

export type SuppressionLookup = {
  isSuppressed(id: string): boolean;
};

export type ParticipantEntityResolutionLookup = ReadonlyMap<string, EntityId | null>;

export type ScoreWeights = {
  similarity: number;
  salience: number;
};

// Tunes the legacy/base retrieval blend between vector similarity and decayed salience.
export const DEFAULT_EPISODE_SCORE_WEIGHTS: ScoreWeights = {
  similarity: 0.7,
  salience: 0.3,
};

// Tunes how much active values can lift base-path episode scores.
const BASE_VALUE_ALIGNMENT_WEIGHT = 0.15;

// Tunes how much exact entity mentions can lift base-path episode scores.
const BASE_ENTITY_RELEVANCE_WEIGHT = 0.15;

// Tunes the trust cutoff for the higher social relevance bonus.
const SOCIAL_RELEVANCE_HIGH_TRUST_THRESHOLD = 0.7;

// Tunes social relevance when the audience is a trusted participant in the episode.
const SOCIAL_RELEVANCE_HIGH_TRUST_SCORE = 0.25;

// Tunes social relevance when the audience is a participant without high trust.
const SOCIAL_RELEVANCE_DEFAULT_SCORE = 0.2;

export type RetrievalMoodState = Pick<MoodState, "valence" | "arousal">;

export type RetrievedEpisode = {
  episode: Episode;
  disclosureLabel?: MemoryDisclosureLabel;
  score: number;
  // Pre-clamp fused score (formula + intent boosts, un-clamped). Diagnostic
  // export only: `score` remains the ranking key and the [0,1] contract.
  rawScore: number;
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
  primaryGoalDescription?: string;
  activeValues?: readonly ValueRecord[];
  scoringFeatures?: RetrievalScoringFeatures;
  moodState?: RetrievalMoodState | null;
  audienceEntityId?: EntityId | null;
  audienceProfile?: SocialProfile | null;
  audienceTerms?: readonly string[];
  participantEntityIds?: ParticipantEntityResolutionLookup;
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
  // Un-clamped fusion value; equals `score` until a caller clamps `score`.
  rawScore: number;
};

type EpisodeScoreFormulaSignals = Omit<EpisodeScore, "score" | "rawScore"> & {
  similarity: number;
};

type EpisodeScoreFormulaWeights = {
  similarity: number;
  salience: number;
  goalRelevance: number;
  valueAlignment: number;
  mood: number;
  social: number;
  entity: number;
  time: number;
  heat: number;
  suppressionPenalty: number;
};

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
  return clamp(heat / RETRIEVAL_HEAT_CAP, 0, 1);
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

// Used by live turns via attention modes; preserves semantic as independent similarity/salience weights.
function attentionScoreFormulaWeights(weights: AttentionWeights): EpisodeScoreFormulaWeights {
  const normalized = normalizeAttentionWeights(weights);

  return {
    similarity: normalized.semantic,
    salience: 1 - normalized.semantic,
    goalRelevance: normalized.goal_relevance,
    valueAlignment: normalized.value_alignment,
    mood: normalized.mood,
    social: normalized.social,
    entity: normalized.entity,
    time: normalized.time,
    heat: normalized.heat,
    suppressionPenalty: normalized.suppression_penalty,
  };
}

// Used by non-turn facade/disclosure/export search; preserves independent scoreWeights and implicit base bonuses.
function baseScoreFormulaWeights(weights: ScoreWeights): EpisodeScoreFormulaWeights {
  return {
    similarity: weights.similarity,
    salience: weights.salience,
    goalRelevance: 0,
    valueAlignment: BASE_VALUE_ALIGNMENT_WEIGHT,
    mood: 0,
    social: 0,
    entity: BASE_ENTITY_RELEVANCE_WEIGHT,
    time: 0,
    heat: 0,
    suppressionPenalty: 0,
  };
}

function computeEpisodeScoreFormula(
  signals: EpisodeScoreFormulaSignals,
  weights: EpisodeScoreFormulaWeights,
): number {
  return (
    weights.similarity * signals.similarity +
    weights.salience * signals.decayedSalience +
    weights.goalRelevance * signals.goalRelevance +
    weights.valueAlignment * signals.valueAlignment +
    weights.mood * signals.moodBoost +
    weights.social * signals.socialRelevance +
    weights.entity * signals.entityRelevance +
    weights.time * signals.timeRelevance +
    weights.heat * normalizeHeat(signals.heat) -
    weights.suppressionPenalty * signals.suppressionPenalty
  );
}

function normalizeTerm(value: string): string {
  return value.trim().toLowerCase();
}

export function participantEntityResolutionKey(value: string): string {
  return normalizeTerm(value);
}

function computeMoodBoost(
  episode: Episode,
  moodState: RetrievalMoodState | null | undefined,
): number {
  if (
    moodState === null ||
    moodState === undefined ||
    Math.abs(moodState.valence) + Math.abs(moodState.arousal) <= MOOD_ACTIVITY_THRESHOLD ||
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
  audienceEntityId: EntityId | null | undefined,
  participantEntityIds: ParticipantEntityResolutionLookup | undefined,
): number {
  const fallbackParticipants: string[] = [];

  if (
    audienceEntityId !== null &&
    audienceEntityId !== undefined &&
    participantEntityIds !== undefined
  ) {
    for (const participant of episode.participants) {
      const resolvedParticipantEntityId = participantEntityIds.get(
        participantEntityResolutionKey(participant),
      );

      if (resolvedParticipantEntityId === audienceEntityId) {
        return audienceProfile !== null &&
          audienceProfile !== undefined &&
          audienceProfile.trust > SOCIAL_RELEVANCE_HIGH_TRUST_THRESHOLD
          ? SOCIAL_RELEVANCE_HIGH_TRUST_SCORE
          : SOCIAL_RELEVANCE_DEFAULT_SCORE;
      }

      if (resolvedParticipantEntityId === null || resolvedParticipantEntityId === undefined) {
        fallbackParticipants.push(participant);
      }
    }
  } else {
    fallbackParticipants.push(...episode.participants);
  }

  const normalizedTerms = new Set(
    (audienceTerms ?? []).map((term) => normalizeTerm(term)).filter((term) => term.length > 0),
  );

  if (normalizedTerms.size === 0) {
    return 0;
  }

  const includesAudience = fallbackParticipants.some((participant) =>
    normalizedTerms.has(normalizeTerm(participant)),
  );

  if (!includesAudience) {
    return 0;
  }

  return audienceProfile !== null &&
    audienceProfile !== undefined &&
    audienceProfile.trust > SOCIAL_RELEVANCE_HIGH_TRUST_THRESHOLD
    ? SOCIAL_RELEVANCE_HIGH_TRUST_SCORE
    : SOCIAL_RELEVANCE_DEFAULT_SCORE;
}

function computeExactEntityMentionBonus(
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
  const goalRelevance =
    searchOptions.scoringFeatures === undefined
      ? 0
      : computeGoalRelevance({
          episodeEmbedding: candidate.episode.embedding,
          goalVectors: searchOptions.scoringFeatures.goalVectors,
          primaryGoalVector: searchOptions.scoringFeatures.primaryGoalVector,
        });
  const valueAlignment =
    searchOptions.scoringFeatures === undefined
      ? 0
      : computeValueAlignment({
          episodeEmbedding: candidate.episode.embedding,
          valueVectors: searchOptions.scoringFeatures.valueVectors,
        });
  const timeRelevance = computeTimeRelevance(candidate.episode, scoringTimeRange);
  const moodBoost = computeMoodBoost(candidate.episode, searchOptions.moodState);
  const socialRelevance = computeSocialRelevance(
    candidate.episode,
    searchOptions.audienceTerms,
    searchOptions.audienceProfile,
    searchOptions.audienceEntityId,
    searchOptions.participantEntityIds,
  );
  // This is deliberately only a verbatim mention bonus. Cross-language semantic relevance is
  // handled by vector candidate retrieval and embedding-backed goal/value scoring.
  const entityRelevance = computeExactEntityMentionBonus(
    candidate.episode,
    searchOptions.entityTerms,
  );
  const suppressionPenalty =
    searchOptions.suppressionSet?.isSuppressed(candidate.episode.id) === true ? 1 : 0;
  const weights =
    searchOptions.attentionWeights === undefined
      ? baseScoreFormulaWeights(searchOptions.scoreWeights ?? defaults.scoreWeights)
      : attentionScoreFormulaWeights(searchOptions.attentionWeights);
  const signals = {
    similarity: candidate.similarity,
    decayedSalience: decay.decayedSalience,
    heat,
    goalRelevance,
    valueAlignment,
    timeRelevance,
    moodBoost,
    socialRelevance,
    entityRelevance,
    suppressionPenalty,
  };
  const fused = computeEpisodeScoreFormula(signals, weights);

  return {
    decayedSalience: signals.decayedSalience,
    heat: signals.heat,
    goalRelevance: signals.goalRelevance,
    valueAlignment: signals.valueAlignment,
    timeRelevance: signals.timeRelevance,
    moodBoost: signals.moodBoost,
    socialRelevance: signals.socialRelevance,
    entityRelevance: signals.entityRelevance,
    suppressionPenalty: signals.suppressionPenalty,
    score: fused,
    rawScore: fused,
  };
}

export function buildRetrievedEpisode(
  candidate: EpisodeSearchCandidate,
  score: EpisodeScore,
  citationChain: StreamEntry[],
): RetrievedEpisode {
  return {
    episode: candidate.episode,
    disclosureLabel: memoryDisclosureLabelFromEpisodeAccess(candidate.episode),
    score: score.score,
    rawScore: score.rawScore,
    scoreBreakdown: {
      similarity: candidate.similarity,
      decayedSalience: score.decayedSalience,
      heat: score.heat,
      goalRelevance: score.goalRelevance,
      valueAlignment: score.valueAlignment,
      timeRelevance: score.timeRelevance,
      moodBoost: score.moodBoost,
      socialRelevance: score.socialRelevance,
      entityRelevance: score.entityRelevance,
      suppressionPenalty: score.suppressionPenalty,
    },
    citationChain,
  };
}
