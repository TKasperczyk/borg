// Aggregates per-result retrieval signals into a single answer-confidence number.
//
// The raw `score` on RetrievedEpisode blends similarity, salience, heat, goal/value
// alignment, mood, social/entity relevance, time, and a suppression penalty -- all
// weighted together. That score is good for *ranking* results but doesn't represent
// epistemic confidence in the retrieved evidence. This module derives a separate
// signal from source-strength, coverage, source diversity, and the contradiction flag
// from the semantic walk, so S1/S2 routing and uncertainty surfacing can key off it.

import type { RetrievedEpisode } from "./scoring.js";

export type RetrievalConfidence = {
  overall: number;
  evidenceStrength: number;
  coverage: number;
  sourceDiversity: number;
  contradictionPresent: boolean;
  sampleSize: number;
};

export type ComputeRetrievalConfidenceInput = {
  episodes: readonly RetrievedEpisode[];
  contradictionPresent: boolean;
  expectedCount?: number;
  topN?: number;
};

const DEFAULT_EXPECTED_COUNT = 5;
const DEFAULT_TOP_N = 5;
const CONTRADICTION_PENALTY = 0.7;

function clamp01(value: number): number {
  if (Number.isNaN(value)) {
    return 0;
  }

  return Math.min(1, Math.max(0, value));
}

export function computeRetrievalConfidence(
  input: ComputeRetrievalConfidenceInput,
): RetrievalConfidence {
  const topN = input.topN ?? DEFAULT_TOP_N;
  const expectedCount = input.expectedCount ?? DEFAULT_EXPECTED_COUNT;
  const episodes = input.episodes;

  if (episodes.length === 0) {
    return {
      overall: 0,
      evidenceStrength: 0,
      coverage: 0,
      sourceDiversity: 0,
      contradictionPresent: input.contradictionPresent,
      sampleSize: 0,
    };
  }

  // Evidence strength: mean decayed salience of the top-N results. Decayed
  // salience reflects how well-established the memory is (source-strength
  // adjusted for how much it has been reinforced and how recent it is),
  // which is closer to epistemic confidence than the blended score.
  const topEpisodes = episodes.slice(0, Math.min(topN, episodes.length));
  const salienceSum = topEpisodes.reduce(
    (sum, episode) => sum + clamp01(episode.scoreBreakdown.decayedSalience),
    0,
  );
  const evidenceStrength = clamp01(salienceSum / topEpisodes.length);

  // Coverage: did we find enough evidence to answer confidently.
  const coverage = clamp01(episodes.length / Math.max(1, expectedCount));

  // Source diversity: distinct participant sets across the top-N. Episodes
  // that involve the same participants are more likely to be one conversation
  // viewed multiple ways; episodes with different participants are genuinely
  // independent evidence. Normalizes against the top-N count.
  const participantSignatures = new Set<string>();

  for (const result of topEpisodes) {
    const signature = [...result.episode.participants].sort().join("|");
    participantSignatures.add(signature);
  }

  const sourceDiversity =
    topEpisodes.length === 0 ? 0 : clamp01(participantSignatures.size / topEpisodes.length);

  // Multiplicative gate: evidenceStrength is the base ceiling. Coverage and
  // diversity can modulate *downward* from that ceiling but cannot lift weak
  // evidence above it -- many weak matches from many participants still add
  // up to low epistemic confidence, not high. The modulation factor ranges
  // from 0.7 (no coverage, no diversity) to 1.0 (full coverage + diversity).
  // Contradiction multiplicatively penalizes the final number further.
  const modulation = 0.7 + 0.2 * coverage + 0.1 * sourceDiversity;
  const rawOverall = evidenceStrength * modulation;
  const contradictionFactor = input.contradictionPresent ? CONTRADICTION_PENALTY : 1;

  return {
    overall: clamp01(rawOverall * contradictionFactor),
    evidenceStrength,
    coverage,
    sourceDiversity,
    contradictionPresent: input.contradictionPresent,
    sampleSize: episodes.length,
  };
}
