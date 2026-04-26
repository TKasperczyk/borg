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
};

export type ComputeRetrievalConfidenceInput = {
  episodes: readonly RetrievedEpisode[];
  contradictionPresent: boolean;
  contradictionEdges?: readonly Pick<SemanticEdge, "valid_from" | "valid_to">[];
  semanticEvidence?: {
    matched_nodes: readonly Pick<SemanticNode, "id" | "confidence" | "source_episode_ids">[];
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

const DEFAULT_EXPECTED_COUNT = 5;
const DEFAULT_TOP_N = 5;
const CONTRADICTION_PENALTY = 0.7;
const SEMANTIC_CONFIDENCE_THRESHOLD = 0.6;

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
    supportedMatches.reduce((sum, node) => sum + clamp01(node.confidence), 0) /
    supportedMatches.length;
  const positiveHitCount = supportedMatches.reduce(
    (sum, node) => sum + (positiveHitCountByRoot.get(node.id) ?? 0),
    0,
  );

  return {
    strength: clamp01(0.3 * sigmoid(meanConfidence * positiveHitCount)),
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
  const evidenceStrength = clamp01(episodeEvidenceStrength + semanticEvidence.strength);

  // Coverage: did we find enough evidence to answer confidently.
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

  const diversitySampleSize = topEpisodes.length + semanticEvidence.count;
  const sourceDiversity =
    diversitySampleSize === 0 ? 0 : clamp01(participantSignatures.size / diversitySampleSize);

  // Multiplicative gate: evidenceStrength is the base ceiling. Coverage and
  // diversity can modulate *downward* from that ceiling but cannot lift weak
  // evidence above it -- many weak matches from many participants still add
  // up to low epistemic confidence, not high. The modulation factor ranges
  // from 0.7 (no coverage, no diversity) to 1.0 (full coverage + diversity).
  // Contradiction multiplicatively penalizes the final number further.
  const modulation = 0.7 + 0.2 * coverage + 0.1 * sourceDiversity;
  const rawOverall = evidenceStrength * modulation;
  const contradictionFactor = contradictionPresent ? CONTRADICTION_PENALTY : 1;

  return {
    overall: clamp01(rawOverall * contradictionFactor),
    evidenceStrength,
    coverage,
    sourceDiversity,
    contradictionPresent,
    sampleSize: episodes.length + semanticEvidence.count,
  };
}
