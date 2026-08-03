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

type EpisodeProjectionCandidate = {
  evidence: EvidenceItem;
  source: EpisodeProjectionSource;
};

// The lexical lane reserves two tail positions by default. The enforcement
// helper still caps opt-in reservation at three positions.
export const EXACT_TERM_RESERVED_SLOTS = 2;
const MAX_EXACT_TERM_RESERVED_SLOTS = 3;

export function projectEpisodes(
  pool: EvidencePool,
  sourcesByEvidenceId: ReadonlyMap<string, EpisodeProjectionSource>,
  options: {
    limit: number;
    mmrLambda: number;
    exactTermReservedSlots?: number;
  },
): EpisodeProjection {
  const candidates = dedupeEpisodeProjectionCandidates(
    pool.items
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
      }),
  );

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

  const exactTermReservedSlots = options.exactTermReservedSlots ?? 0;

  if (exactTermReservedSlots > 0) {
    enforceExactTermReservedSlots(selected, candidates, exactTermReservedSlots);
  }

  return {
    episodes: selected.map(({ source }) =>
      buildRetrievedEpisode(
        source.candidate,
        {
          ...source.score,
          score: clamp(source.score.score, 0, 1),
        },
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

function dedupeEpisodeProjectionCandidates(
  candidates: readonly EpisodeProjectionCandidate[],
): EpisodeProjectionCandidate[] {
  const byEpisodeId = new Map<string, EpisodeProjectionCandidate>();

  for (const candidate of candidates) {
    const episodeId = candidate.evidence.provenance?.episodeId;

    if (episodeId === undefined) {
      continue;
    }

    const current = byEpisodeId.get(episodeId);

    if (current === undefined) {
      byEpisodeId.set(episodeId, candidate);
      continue;
    }

    const representative =
      candidate.source.score.score > current.source.score.score ? candidate : current;
    byEpisodeId.set(episodeId, {
      ...representative,
      evidence: {
        ...representative.evidence,
        matchedTerms: [
          ...new Set([...current.evidence.matchedTerms, ...candidate.evidence.matchedTerms]),
        ],
      },
    });
  }

  return [...byEpisodeId.values()];
}

/**
 * Post-MMR guarantee for exact-term candidates. It may replace only tail
 * entries, so the original top-1 is immutable, and it can reserve at most
 * three final positions. Candidates and selections are deduplicated by
 * episode id before replacement.
 */
function enforceExactTermReservedSlots(
  selected: EpisodeProjectionCandidate[],
  candidates: readonly EpisodeProjectionCandidate[],
  requestedSlots: number,
): void {
  const reservedSlots = Math.min(
    MAX_EXACT_TERM_RESERVED_SLOTS,
    Math.max(0, Math.floor(requestedSlots)),
  );

  if (reservedSlots === 0 || selected.length <= 1) {
    return;
  }

  const exactEpisodeIds = new Set(
    candidates
      .filter((candidate) => candidate.evidence.matchedTerms.length > 0)
      .map((candidate) => candidate.evidence.provenance?.episodeId)
      .filter((episodeId) => episodeId !== undefined),
  );

  if (exactEpisodeIds.size === 0) {
    return;
  }

  const isExactSelection = (candidate: EpisodeProjectionCandidate): boolean => {
    const episodeId = candidate.evidence.provenance?.episodeId;
    return episodeId !== undefined && exactEpisodeIds.has(episodeId);
  };
  const selectedEpisodeIds = new Set(
    selected
      .map((candidate) => candidate.evidence.provenance?.episodeId)
      .filter((episodeId) => episodeId !== undefined),
  );
  const replacementByEpisodeId = new Map<string, EpisodeProjectionCandidate>();

  for (const candidate of candidates) {
    if (candidate.evidence.matchedTerms.length === 0) {
      continue;
    }

    const episodeId = candidate.evidence.provenance?.episodeId;

    if (episodeId === undefined || selectedEpisodeIds.has(episodeId)) {
      continue;
    }

    const current = replacementByEpisodeId.get(episodeId);

    if (current === undefined || candidate.source.score.score > current.source.score.score) {
      replacementByEpisodeId.set(episodeId, candidate);
    }
  }

  const replacements = [...replacementByEpisodeId.values()].sort(
    (left, right) =>
      right.source.score.score - left.source.score.score ||
      right.evidence.id.localeCompare(left.evidence.id),
  );
  let exactSelectedCount = selected.filter(isExactSelection).length;
  const target = Math.min(reservedSlots, exactSelectedCount + replacements.length);
  const selectedReplacements = replacements.slice(0, target - exactSelectedCount);

  for (
    let index = selected.length - 1;
    index > 0 && exactSelectedCount < target && selectedReplacements.length > 0;
    index -= 1
  ) {
    const current = selected[index];

    if (current === undefined || isExactSelection(current)) {
      continue;
    }

    // Fill from the tail with the lowest-ranked chosen replacement so the
    // highest-ranked exact candidate remains first among inserted entries.
    const replacement = selectedReplacements.pop();

    if (replacement === undefined) {
      break;
    }

    selected[index] = replacement;
    exactSelectedCount += 1;
  }
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
