/* Compatibility projections from the ranked EvidencePool. */
import type { OpenQuestion } from "../memory/self/index.js";
import type { EpisodeSearchCandidate } from "../memory/episodic/types.js";
import type { SemanticNode } from "../memory/semantic/types.js";
import type { StreamEntry } from "../stream/index.js";
import { RetrievalError } from "../util/errors.js";
import type { SemanticEdgeId, SemanticNodeId } from "../util/ids.js";

import { applyMmr } from "./mmr.js";
import type { EvidenceItem, EvidencePool } from "./recall-types.js";
import {
  buildRetrievedEpisode,
  clamp,
  type EpisodeScore,
  type RetrievedEpisode,
} from "./scoring.js";
import type { RetrievedSemantic, RetrievedSemanticHit } from "./semantic-retrieval.js";

export type EpisodeProjectionSource = {
  candidate: EpisodeSearchCandidate;
  score: EpisodeScore;
  citationChain: () => StreamEntry[];
};

export type EpisodeProjection = {
  episodes: RetrievedEpisode[];
  selectedEvidence: EvidenceItem[];
};

export function projectEpisodes(
  pool: EvidencePool,
  sourcesByEvidenceId: ReadonlyMap<string, EpisodeProjectionSource>,
  options: {
    limit: number;
    mmrLambda: number;
  },
): EpisodeProjection {
  const candidates = pool.items
    .filter((item) => item.source === "episode")
    .map((evidence) => {
      if (evidence.provenance?.episodeId === undefined) {
        throw missingProjectionSource(
          `Episode evidence ${evidence.id} is missing episode provenance`,
        );
      }

      const source = sourcesByEvidenceId.get(evidence.id);

      if (source === undefined) {
        throw missingProjectionSource(
          `Episode evidence ${evidence.id} references ${evidence.provenance!.episodeId}, but no projection source was hydrated`,
        );
      }

      return {
        evidence,
        source,
      };
    });

  const selected = applyMmr(
    candidates.map((item) => ({
      item,
      vector: item.source.candidate.episode.embedding,
      relevanceScore: item.source.score.score,
    })),
    {
      limit: options.limit,
      lambda: options.mmrLambda,
    },
  ).map((choice) => choice.item);

  return {
    episodes: selected.map(({ source }) =>
      buildRetrievedEpisode(
        source.candidate,
        source.score.decayedSalience,
        source.score.heat,
        source.score.goalRelevance,
        source.score.valueAlignment,
        source.score.timeRelevance,
        source.score.moodBoost,
        source.score.socialRelevance,
        source.score.entityRelevance,
        source.score.suppressionPenalty,
        clamp(source.score.score, 0, 1),
        source.citationChain(),
      ),
    ),
    selectedEvidence: selected.map((item) => item.evidence),
  };
}

export function projectSemantic(
  pool: EvidencePool,
  semantic: RetrievedSemantic,
): RetrievedSemantic {
  const nodeOrder = new Map<SemanticNodeId, number>();
  const edgeOrder = new Map<string, number>();

  pool.items.forEach((item, index) => {
    if (item.source === "semantic_node") {
      if (item.provenance?.nodeId === undefined) {
        throw missingProjectionSource(
          `Semantic-node evidence ${item.id} is missing node provenance`,
        );
      }

      setFirstOrder(nodeOrder, item.provenance.nodeId, index);
    }

    if (item.source === "semantic_edge") {
      const key = semanticEdgeEvidenceKey(item);

      if (key === null) {
        throw missingProjectionSource(
          `Semantic-edge evidence ${item.id} is missing edge or node provenance`,
        );
      }

      setFirstOrder(edgeOrder, key, index);
    }
  });

  const matchedNodes = orderByEvidence(
    semantic.matched_nodes.filter((node) => nodeOrder.has(node.id)),
    (node) => nodeOrder.get(node.id),
  );
  assertSemanticNodesHydrated(nodeOrder, semantic.matched_nodes);
  const supportHits = orderSemanticHitsByEvidence(
    semantic.support_hits.filter((hit) => semanticHitHasEvidence(hit, edgeOrder)),
    edgeOrder,
  );
  const causalHits = orderSemanticHitsByEvidence(
    semantic.causal_hits.filter((hit) => semanticHitHasEvidence(hit, edgeOrder)),
    edgeOrder,
  );
  const contradictionHits = orderSemanticHitsByEvidence(
    semantic.contradiction_hits.filter((hit) => semanticHitHasEvidence(hit, edgeOrder)),
    edgeOrder,
  );
  const categoryHits = orderSemanticHitsByEvidence(
    semantic.category_hits.filter((hit) => semanticHitHasEvidence(hit, edgeOrder)),
    edgeOrder,
  );
  assertSemanticEdgesHydrated(edgeOrder, [
    ...semantic.support_hits,
    ...semantic.causal_hits,
    ...semantic.contradiction_hits,
    ...semantic.category_hits,
  ]);

  return {
    as_of: semantic.as_of,
    supports: projectContextNodes(semantic.supports, supportHits),
    contradicts: projectContextNodes(semantic.contradicts, contradictionHits),
    categories: projectContextNodes(semantic.categories, categoryHits),
    matched_node_ids: matchedNodes.map((node) => node.id),
    matched_nodes: matchedNodes,
    support_hits: supportHits,
    causal_hits: causalHits,
    contradiction_hits: contradictionHits,
    category_hits: categoryHits,
  };
}

export function projectOpenQuestions(
  pool: EvidencePool,
  questionsByEvidenceId: ReadonlyMap<string, OpenQuestion>,
): OpenQuestion[] {
  return pool.items
    .filter((item) => item.source === "open_question")
    .map((item) => {
      if (item.provenance?.openQuestionId === undefined) {
        throw missingProjectionSource(
          `Open-question evidence ${item.id} is missing open-question provenance`,
        );
      }

      const question = questionsByEvidenceId.get(item.id);

      if (question === undefined) {
        throw missingProjectionSource(
          `Open-question evidence ${item.id} references ${item.provenance!.openQuestionId}, but no question was hydrated`,
        );
      }

      return question;
    });
}

function missingProjectionSource(message: string): RetrievalError {
  return new RetrievalError(message, {
    code: "BORG_RETRIEVAL_PROJECTION_INVARIANT",
  });
}

function projectContextNodes(
  nodes: readonly SemanticNode[],
  hits: readonly RetrievedSemanticHit[],
): SemanticNode[] {
  const hitNodeIds = new Set(hits.map((hit) => hit.node.id));

  return nodes.filter((node) => hitNodeIds.has(node.id));
}

function orderSemanticHitsByEvidence(
  hits: readonly RetrievedSemanticHit[],
  edgeOrder: ReadonlyMap<string, number>,
): RetrievedSemanticHit[] {
  return orderByEvidence(hits, (hit) => semanticHitEvidenceOrder(hit, edgeOrder));
}

function orderByEvidence<T>(items: readonly T[], order: (item: T) => number | undefined): T[] {
  return [...items].sort((left, right) => (order(left) ?? Infinity) - (order(right) ?? Infinity));
}

function assertSemanticNodesHydrated(
  nodeOrder: ReadonlyMap<SemanticNodeId, number>,
  nodes: readonly SemanticNode[],
): void {
  const hydratedNodeIds = new Set(nodes.map((node) => node.id));

  for (const nodeId of nodeOrder.keys()) {
    if (!hydratedNodeIds.has(nodeId)) {
      throw missingProjectionSource(
        `Semantic-node evidence references ${nodeId}, but no semantic node was hydrated`,
      );
    }
  }
}

function assertSemanticEdgesHydrated(
  edgeOrder: ReadonlyMap<string, number>,
  hits: readonly RetrievedSemanticHit[],
): void {
  const hydratedKeys = new Set(hits.map((hit) => semanticHitEvidenceKey(hit)));

  for (const key of edgeOrder.keys()) {
    if (hydratedKeys.has(key)) {
      continue;
    }

    throw missingProjectionSource(
      `Semantic-edge evidence references ${describeSemanticEdgeKey(key)}, but no semantic hit was hydrated`,
    );
  }
}

function semanticHitEvidenceKey(hit: RetrievedSemanticHit): string {
  const edge = hit.edgePath.at(-1);

  if (edge !== undefined) {
    return semanticEdgeKey(edge.id);
  }

  return semanticEdgeNodeFallbackKey(hit.node.id);
}

function describeSemanticEdgeKey(key: string): string {
  if (key.startsWith("edge:")) {
    return key.slice("edge:".length);
  }

  if (key.startsWith("node:")) {
    return key.slice("node:".length);
  }

  return key;
}

function setFirstOrder<T>(order: Map<T, number>, key: T, index: number): void {
  if (!order.has(key)) {
    order.set(key, index);
  }
}

function semanticHitHasEvidence(
  hit: RetrievedSemanticHit,
  edgeOrder: ReadonlyMap<string, number>,
): boolean {
  return semanticHitEvidenceOrder(hit, edgeOrder) !== undefined;
}

function semanticHitEvidenceOrder(
  hit: RetrievedSemanticHit,
  edgeOrder: ReadonlyMap<string, number>,
): number | undefined {
  const edge = hit.edgePath.at(-1);

  if (edge !== undefined) {
    return edgeOrder.get(semanticEdgeKey(edge.id));
  }

  return edgeOrder.get(semanticEdgeNodeFallbackKey(hit.node.id));
}

function semanticEdgeEvidenceKey(item: EvidenceItem): string | null {
  if (item.provenance?.edgeId !== undefined) {
    return semanticEdgeKey(item.provenance.edgeId);
  }

  if (item.provenance?.nodeId !== undefined) {
    return semanticEdgeNodeFallbackKey(item.provenance.nodeId);
  }

  return null;
}

function semanticEdgeKey(edgeId: SemanticEdgeId): string {
  return `edge:${edgeId}`;
}

function semanticEdgeNodeFallbackKey(nodeId: SemanticNodeId): string {
  return `node:${nodeId}`;
}
