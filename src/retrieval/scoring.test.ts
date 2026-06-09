import { describe, expect, it } from "vitest";

import { COGNITIVE_MODES, type AttentionWeights } from "../contracts/cognitive-contracts.js";
import type { EpisodeSearchCandidate, EpisodeStats } from "../memory/episodic/types.js";
import type { SocialProfile } from "../memory/social/index.js";
import { createEpisodeFixture } from "../offline/test-support.js";
import type { EntityId } from "../util/ids.js";
import { computeWeights } from "../cognition/attention/index.js";
import { RETRIEVAL_HEAT_CAP } from "../memory/episodic/heat.js";
import type { RetrievalScoringFeatures } from "./scoring-features.js";
import {
  DEFAULT_EPISODE_SCORE_WEIGHTS,
  scoreCandidate,
  type EpisodeScore,
  type EpisodeScoreDefaults,
  type ScoreWeights,
} from "./scoring.js";

const NOW_MS = 1_000_000;
const AUDIENCE_ID = "ent_aaaaaaaaaaaaaaaa" as EntityId;
const BASE_VALUE_ALIGNMENT_WEIGHT = 0.15;
const BASE_ENTITY_RELEVANCE_WEIGHT = 0.15;

const DEFAULTS: EpisodeScoreDefaults = {
  scoreWeights: { ...DEFAULT_EPISODE_SCORE_WEIGHTS },
};

const HIGH_TRUST_PROFILE: SocialProfile = {
  entity_id: AUDIENCE_ID,
  record_version: 1,
  trust: 0.82,
  attachment: 0.3,
  communication_style: null,
  shared_history_summary: null,
  last_interaction_at: NOW_MS - 10_000,
  interaction_count: 4,
  commitment_count: 0,
  sentiment_history: [],
  notes: null,
  created_at: NOW_MS - 200_000,
  updated_at: NOW_MS - 10_000,
};

const SCORING_FEATURES: RetrievalScoringFeatures = {
  goalVectors: [Float32Array.from([1, 0, 0, 0])],
  primaryGoalVector: Float32Array.from([1, 0, 0, 0]),
  valueVectors: [Float32Array.from([0.5, 0.5, 0, 0])],
};

function makeCandidate(similarity = 0.42): EpisodeSearchCandidate {
  const episode = createEpisodeFixture(
    {
      title: "Atlas planning sync",
      narrative: "Alice and Borg planned the Atlas rollout.",
      participants: ["Alice"],
      tags: ["Atlas", "planning"],
      significance: 0.64,
      start_time: NOW_MS - 30_000,
      end_time: NOW_MS - 20_000,
      created_at: NOW_MS - 60_000,
      updated_at: NOW_MS - 10_000,
      emotional_arc: {
        start: { valence: -0.2, arousal: 0.5 },
        peak: { valence: -0.5, arousal: 0.8 },
        end: { valence: -0.1, arousal: 0.4 },
        dominant_emotion: "curiosity",
      },
    },
    [1, 0, 0, 0],
  );
  const stats: EpisodeStats = {
    episode_id: episode.id,
    retrieval_count: 7,
    use_count: 3,
    last_retrieved: NOW_MS - 2_000,
    win_rate: 0.62,
    tier: "T2",
    promoted_at: NOW_MS - 50_000,
    promoted_from: null,
    gist: null,
    gist_generated_at: null,
    last_decayed_at: null,
    heat_multiplier: 1.15,
    valence_mean: -0.2,
    archived: false,
  };

  return {
    episode,
    stats,
    similarity,
  };
}

function commonSearchOptions(candidate: EpisodeSearchCandidate) {
  return {
    scoringFeatures: SCORING_FEATURES,
    moodState: { valence: -0.4, arousal: 0.7 },
    audienceEntityId: AUDIENCE_ID,
    audienceProfile: HIGH_TRUST_PROFILE,
    audienceTerms: ["Alice"],
    participantEntityIds: new Map([["alice", AUDIENCE_ID]]),
    entityTerms: ["Atlas"],
    suppressionSet: {
      isSuppressed: (id: string) => id === candidate.episode.id,
    },
  };
}

function expectedBaseScore(
  candidate: EpisodeSearchCandidate,
  score: EpisodeScore,
  weights: ScoreWeights,
): number {
  return (
    weights.similarity * candidate.similarity +
    weights.salience * score.decayedSalience +
    score.valueAlignment * BASE_VALUE_ALIGNMENT_WEIGHT +
    score.entityRelevance * BASE_ENTITY_RELEVANCE_WEIGHT
  );
}

function normalizeHeatForTest(heat: number): number {
  return Math.min(1, Math.max(0, heat / RETRIEVAL_HEAT_CAP));
}

function expectedAttentionScore(
  candidate: EpisodeSearchCandidate,
  score: EpisodeScore,
  weights: AttentionWeights,
): number {
  const semanticScore =
    weights.semantic * candidate.similarity + (1 - weights.semantic) * score.decayedSalience;

  return (
    semanticScore +
    weights.goal_relevance * score.goalRelevance +
    weights.value_alignment * score.valueAlignment +
    weights.mood * score.moodBoost +
    weights.social * score.socialRelevance +
    weights.entity * score.entityRelevance +
    weights.time * score.timeRelevance +
    weights.heat * normalizeHeatForTest(score.heat) -
    weights.suppression_penalty * score.suppressionPenalty
  );
}

describe("scoreCandidate equivalence", () => {
  it("matches the current default base-path arithmetic exactly", () => {
    const candidate = makeCandidate();
    const score = scoreCandidate(
      candidate,
      commonSearchOptions(candidate),
      NOW_MS,
      { start: NOW_MS - 40_000, end: NOW_MS - 15_000 },
      DEFAULTS,
    );

    expect(score.score).toBe(expectedBaseScore(candidate, score, DEFAULT_EPISODE_SCORE_WEIGHTS));
  });

  it.each([
    { similarity: 0.4, salience: 0.9 },
    { similarity: 1.1, salience: 0.05 },
    { similarity: 0.2, salience: 0.6 },
  ] satisfies ScoreWeights[])(
    "matches custom independent base scoreWeights exactly: %j",
    (weights) => {
      const candidate = makeCandidate(0.37);
      const score = scoreCandidate(
        candidate,
        {
          ...commonSearchOptions(candidate),
          scoreWeights: weights,
        },
        NOW_MS,
        { start: NOW_MS - 40_000, end: NOW_MS - 15_000 },
        DEFAULTS,
      );

      expect(score.score).toBe(expectedBaseScore(candidate, score, weights));
    },
  );

  it.each(COGNITIVE_MODES)("matches attention-path arithmetic exactly for %s mode", (mode) => {
    const candidate = makeCandidate(0.58);
    const attentionWeights = computeWeights(mode, {
      currentGoals: [{} as never],
      hasActiveValues: true,
      hasTemporalCue: true,
      moodActive: true,
      audienceTrust: HIGH_TRUST_PROFILE.trust,
    });
    const score = scoreCandidate(
      candidate,
      {
        ...commonSearchOptions(candidate),
        attentionWeights,
      },
      NOW_MS,
      { start: NOW_MS - 40_000, end: NOW_MS - 15_000 },
      DEFAULTS,
    );

    expect(score.score).toBe(expectedAttentionScore(candidate, score, attentionWeights));
  });

  it("matches literal facade-style attentionWeights exactly", () => {
    const candidate = makeCandidate(0.49);
    const attentionWeights: AttentionWeights = {
      semantic: 0.35,
      goal_relevance: 0.1,
      value_alignment: 0,
      mood: 0,
      time: 0.2,
      social: 0.15,
      entity: 0.2,
      heat: 0.45,
      suppression_penalty: 0.5,
    };
    const score = scoreCandidate(
      candidate,
      {
        ...commonSearchOptions(candidate),
        attentionWeights,
      },
      NOW_MS,
      { start: NOW_MS - 40_000, end: NOW_MS - 15_000 },
      DEFAULTS,
    );

    expect(score.score).toBe(expectedAttentionScore(candidate, score, attentionWeights));
  });
});
