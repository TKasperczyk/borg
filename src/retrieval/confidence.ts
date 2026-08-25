// Aggregates per-result retrieval signals into a single answer-confidence number.
//
// The raw `score` on RetrievedEpisode blends similarity, salience, heat, goal/value
// alignment, mood, social/entity relevance, time, and a suppression penalty -- all
// weighted together. That score is good for *ranking* results but doesn't represent
// epistemic confidence in the retrieved evidence. This module derives a separate
// signal from source-strength, coverage, source diversity, and the contradiction flag
// from the semantic walk, so S1/S2 routing and uncertainty surfacing can key off it.

import type { RetrievedEpisode } from "./scoring.js";
import type { SemanticEdge, SemanticNode } from "../memory/semantic/index.js";

export type RetrievalConfidence = {
  overall: number;
  evidenceStrength: number;
  coverage: number;
  sourceDiversity: number;
  contradictionPresent: boolean;
  sampleSize: number;
  // The denominators the two ratios were actually divided by. Neither is
  // `sampleSize`, and neither is a constant, so a 1.00 on either line is only
  // readable next to the figure it saturated against. Carried on the result so
  // the render can print the fraction instead of just the quotient.
  coverageExpected: number;
  diversitySources: number;
  diversitySampleSize: number;
  // The two addends `evidenceStrength` is the clamped sum of. Their sum is not
  // bounded by 1 before the clamp, so a 1.00 on that line is more often a
  // ceiling hit than a measurement, and the quotient alone cannot say which.
  // Carried so the render can show the components and mark the clamp.
  evidenceEpisodeStrength: number;
  evidenceSemanticStrength: number;
};

export type ComputeRetrievalConfidenceInput = {
  episodes: readonly RetrievedEpisode[];
  contradictionPresent: boolean;
  contradictionEdges?: readonly Pick<SemanticEdge, "valid_from" | "valid_to">[];
  semanticEvidence?: {
    matched_nodes: readonly (Pick<SemanticNode, "id" | "confidence" | "source_episode_ids"> & {
      partial_source_visibility?: boolean;
      source_visibility_fraction?: number;
    })[];
    support_hits: readonly {
      root_node_id: SemanticNode["id"];
      edgePath: readonly Pick<SemanticEdge, "evidence_episode_ids">[];
    }[];
    causal_hits?: readonly {
      root_node_id: SemanticNode["id"];
      edgePath: readonly Pick<SemanticEdge, "evidence_episode_ids">[];
    }[];
  };
  nowMs: number;
  asOf?: number;
  expectedCount?: number;
  topN?: number;
};

// Tunes the expected evidence count for full retrieval coverage.
const DEFAULT_EXPECTED_COUNT = 5;

// Tunes how many top ranked episodes contribute to confidence strength.
const DEFAULT_TOP_N = 5;

// Tunes the multiplicative confidence retained when valid contradiction exists.
const CONTRADICTION_PENALTY = 0.7;

// Tunes the minimum semantic node confidence admitted into confidence support.
const SEMANTIC_CONFIDENCE_THRESHOLD = 0.6;

// Tunes how strongly semantic support can raise evidence strength. This is also
// the semantic addend's hard ceiling: the addend is this scale times a sigmoid,
// and the sigmoid saturates once a handful of supported matches exist, so the
// addend sits on the ceiling on most turns rather than measuring anything. It is
// exported so the render can print the addend against the bound it is pinned to
// -- a bare `sem=0.30` reads as a measurement and cannot say it is the maximum.
export const SEMANTIC_EVIDENCE_STRENGTH_SCALE = 0.3;

// Tunes the confidence modulation floor before coverage and diversity are applied.
const CONFIDENCE_MODULATION_BASE = 0.7;

// The episode addend's hard ceiling. It is a mean of clamped saliences, so
// `clamp01` is what bounds it, and this is the value `clamp01` saturates at --
// exported for the same reason as the semantic scale above, and defined as the
// clamp's own ceiling so the printed bound cannot drift from the computation.
// A bare `ep=0.72` beside three terms that all print against a bound argues by
// silence that this one has none.
export const EPISODE_EVIDENCE_STRENGTH_BOUND = 1;

// Tunes coverage's contribution to confidence modulation.
const CONFIDENCE_COVERAGE_MODULATION_WEIGHT = 0.2;

// Tunes source diversity's contribution to confidence modulation.
const CONFIDENCE_DIVERSITY_MODULATION_WEIGHT = 0.1;

function clamp01(value: number): number {
  if (Number.isNaN(value)) {
    return 0;
  }

  return Math.min(1, Math.max(0, value));
}

function isEdgeValidAt(edge: Pick<SemanticEdge, "valid_from" | "valid_to">, asOf: number): boolean {
  return edge.valid_from <= asOf && (edge.valid_to === null || edge.valid_to > asOf);
}

function sigmoid(value: number): number {
  return 1 / (1 + Math.exp(-value));
}

function hasEpisodeOverlap(left: readonly string[], right: ReadonlySet<string>): boolean {
  return left.some((value) => right.has(value));
}

function semanticConfidenceContribution(
  node: Pick<SemanticNode, "confidence"> & {
    partial_source_visibility?: boolean;
    source_visibility_fraction?: number;
  },
): number {
  const visibleSourceFraction =
    node.partial_source_visibility === true ? clamp01(node.source_visibility_fraction ?? 1) : 1;

  return clamp01(node.confidence) * visibleSourceFraction;
}

function computeSemanticEvidence(input: ComputeRetrievalConfidenceInput): {
  strength: number;
  count: number;
  sourceSignatures: string[];
} {
  const semanticEvidence = input.semanticEvidence;
  const positiveHits =
    semanticEvidence === undefined
      ? []
      : [...semanticEvidence.support_hits, ...(semanticEvidence.causal_hits ?? [])];

  if (semanticEvidence === undefined || positiveHits.length === 0) {
    return {
      strength: 0,
      count: 0,
      sourceSignatures: [],
    };
  }

  const retrievedEpisodeIds = new Set(input.episodes.map((episode) => episode.episode.id));
  const positiveHitCountByRoot = new Map<string, number>();

  for (const hit of positiveHits) {
    if (
      hit.edgePath.some((edge) => hasEpisodeOverlap(edge.evidence_episode_ids, retrievedEpisodeIds))
    ) {
      continue;
    }

    positiveHitCountByRoot.set(
      hit.root_node_id,
      (positiveHitCountByRoot.get(hit.root_node_id) ?? 0) + 1,
    );
  }

  const supportedMatches = semanticEvidence.matched_nodes.filter(
    (node) =>
      node.confidence >= SEMANTIC_CONFIDENCE_THRESHOLD &&
      !hasEpisodeOverlap(node.source_episode_ids, retrievedEpisodeIds) &&
      (positiveHitCountByRoot.get(node.id) ?? 0) > 0,
  );

  if (supportedMatches.length === 0) {
    return {
      strength: 0,
      count: 0,
      sourceSignatures: [],
    };
  }

  const meanConfidence =
    supportedMatches.reduce((sum, node) => sum + semanticConfidenceContribution(node), 0) /
    supportedMatches.length;
  const positiveHitCount = supportedMatches.reduce(
    (sum, node) => sum + (positiveHitCountByRoot.get(node.id) ?? 0),
    0,
  );

  return {
    strength: clamp01(
      SEMANTIC_EVIDENCE_STRENGTH_SCALE * sigmoid(meanConfidence * positiveHitCount),
    ),
    count: supportedMatches.length,
    sourceSignatures: supportedMatches.map((node) => [...node.source_episode_ids].sort().join("|")),
  };
}

export function computeRetrievalConfidence(
  input: ComputeRetrievalConfidenceInput,
): RetrievalConfidence {
  const topN = input.topN ?? DEFAULT_TOP_N;
  const expectedCount = input.expectedCount ?? DEFAULT_EXPECTED_COUNT;
  const episodes = input.episodes;
  const contradictionPresent =
    input.contradictionEdges === undefined
      ? input.contradictionPresent
      : input.contradictionPresent &&
        input.contradictionEdges.some((edge) => isEdgeValidAt(edge, input.asOf ?? input.nowMs));

  const semanticEvidence = computeSemanticEvidence(input);

  if (episodes.length === 0 && semanticEvidence.count === 0) {
    return {
      overall: 0,
      evidenceStrength: 0,
      coverage: 0,
      sourceDiversity: 0,
      contradictionPresent,
      sampleSize: 0,
      coverageExpected: expectedCount,
      diversitySources: 0,
      diversitySampleSize: 0,
      evidenceEpisodeStrength: 0,
      evidenceSemanticStrength: 0,
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
  const episodeEvidenceStrength =
    topEpisodes.length === 0 ? 0 : clamp01(salienceSum / topEpisodes.length);
  // Both addends are already clamped to [0,1] individually, so their sum can
  // reach 1.30 and the outer clamp silently discards the overshoot. The
  // semantic half is bounded by SEMANTIC_EVIDENCE_STRENGTH_SCALE and sits near
  // its own ceiling whenever any supported match exists, so a well-established
  // episode set alone is usually enough to pin the line at 1.00. Same failure
  // as coverage below, one field up: the printed quotient cannot distinguish a
  // measured 1.00 from a clamped one, which is why both addends ship out.
  const evidenceStrength = clamp01(episodeEvidenceStrength + semanticEvidence.strength);

  // Coverage: did we find enough evidence to answer confidently.
  //
  // Read the two halves of this fraction before reading the number. The
  // numerator is the same expression as the reported `sampleSize` below. The
  // denominator is not a constant and not the sample: callers pass the
  // retrieval limit (1/4/5/6 by cognitive mode), and the episode list has
  // already been capped at exactly that limit upstream, with the semantic count
  // added on top of the cap. So on any turn that fills its episode budget and
  // matches a single semantic node, coverage is pinned at 1.00 -- not because
  // evidence was broad but because the numerator can exceed a denominator it
  // was never bounded by. Pinned at its ceiling, it stops discriminating and
  // its modulation term below becomes a constant. Only the zero-evidence early
  // return can drive it to 0. That is why `coverageExpected` ships with it: a
  // saturated ratio and a measured one print identically on their own.
  const coverage = clamp01((episodes.length + semanticEvidence.count) / Math.max(1, expectedCount));

  // Source diversity: distinct participant sets across the top-N. Episodes
  // that involve the same participants are more likely to be one conversation
  // viewed multiple ways; episodes with different participants are genuinely
  // independent evidence. Normalizes against the top-N count.
  const participantSignatures = new Set<string>();

  for (const result of topEpisodes) {
    const signature = [...result.episode.participants].sort().join("|");
    participantSignatures.add(signature);
  }

  for (const signature of semanticEvidence.sourceSignatures) {
    participantSignatures.add(`semantic:${signature}`);
  }

  // Note the denominator: `topEpisodes.length`, capped at topN, not the
  // `episodes.length` used by the reported `sampleSize` below. Once retrieval
  // returns more than topN episodes the two diverge, and diversity stops being
  // readable against the sample figure that ships beside it -- a shrinking
  // sample does not narrow this denominator at all while it stays above topN.
  // The other half of the reading: in a two-participant session every episode
  // signature collapses to the same string, so episodes contribute exactly 1 to
  // the numerator no matter how many there are, and every further point of
  // diversity comes from distinct `semantic:` source sets. A 1.00 there is not
  // evidence of breadth across conversations; it is the small-denominator case.
  // Both halves ship on the result as `diversitySources` / `diversitySampleSize`
  // so the render can show the fraction rather than only its quotient.
  const diversitySampleSize = topEpisodes.length + semanticEvidence.count;
  const sourceDiversity =
    diversitySampleSize === 0 ? 0 : clamp01(participantSignatures.size / diversitySampleSize);

  // Multiplicative gate: evidenceStrength is the base ceiling. Coverage and
  // diversity can modulate *downward* from that ceiling but cannot lift weak
  // evidence above it -- many weak matches from many participants still add
  // up to low epistemic confidence, not high. The modulation factor ranges
  // from 0.7 (no coverage, no diversity) to 1.0 (full coverage + diversity).
  // Contradiction multiplicatively penalizes the final number further.
  const modulation =
    CONFIDENCE_MODULATION_BASE +
    CONFIDENCE_COVERAGE_MODULATION_WEIGHT * coverage +
    CONFIDENCE_DIVERSITY_MODULATION_WEIGHT * sourceDiversity;
  const rawOverall = evidenceStrength * modulation;
  const contradictionFactor = contradictionPresent ? CONTRADICTION_PENALTY : 1;

  return {
    overall: clamp01(rawOverall * contradictionFactor),
    evidenceStrength,
    coverage,
    sourceDiversity,
    contradictionPresent,
    sampleSize: episodes.length + semanticEvidence.count,
    coverageExpected: Math.max(1, expectedCount),
    diversitySources: participantSignatures.size,
    diversitySampleSize,
    evidenceEpisodeStrength: episodeEvidenceStrength,
    evidenceSemanticStrength: semanticEvidence.strength,
  };
}
