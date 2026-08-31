import { describe, expect, it } from "vitest";

import type { SemanticEdge, SemanticNode } from "../memory/semantic/index.js";
import { episodeIdHelpers, semanticEdgeIdHelpers, semanticNodeIdHelpers } from "../util/ids.js";

import {
  assembleRetrievedContext,
  countRetrievedContradictionRelations,
} from "./context-assembly.js";
import type { RetrievedSemantic, RetrievedSemanticHit } from "./semantic-retrieval.js";

const NOW = 1_700_000_000_000;

const NODE_A = semanticNodeIdHelpers.create();
const NODE_B = semanticNodeIdHelpers.create();
const NODE_C = semanticNodeIdHelpers.create();
const NODE_D = semanticNodeIdHelpers.create();
const EDGE_AB = semanticEdgeIdHelpers.create();
const EDGE_CD = semanticEdgeIdHelpers.create();
const EPISODE = episodeIdHelpers.create();

function node(id: SemanticNode["id"]): SemanticNode {
  return {
    id,
    kind: "proposition",
    label: `label ${id}`,
    description: `description of ${id}`,
    domain: null,
    aliases: [],
    observation_metadata: null,
    confidence: 0.9,
    source_episode_ids: [EPISODE],
    created_at: NOW,
    updated_at: NOW,
    last_verified_at: NOW,
    embedding: new Float32Array([1, 0, 0]),
    archived: false,
    superseded_by: null,
    status: "active",
    corrected_by: null,
    superseded_at: null,
  };
}

function contradictsEdge(
  id: SemanticEdge["id"],
  fromNodeId: SemanticNode["id"],
  toNodeId: SemanticNode["id"],
): SemanticEdge {
  return {
    id,
    from_node_id: fromNodeId,
    to_node_id: toNodeId,
    relation: "contradicts",
    confidence: 0.8,
    evidence_episode_ids: [EPISODE],
    created_at: NOW,
    last_verified_at: NOW,
    valid_from: NOW,
    valid_to: null,
    invalidated_at: null,
    invalidated_by_edge_id: null,
    invalidated_by_review_id: null,
    invalidated_by_process: null,
    invalidated_reason: null,
  };
}

function hit(
  rootNodeId: SemanticNode["id"],
  reachedNodeId: SemanticNode["id"],
  edge: SemanticEdge,
): RetrievedSemanticHit {
  return { root_node_id: rootNodeId, node: node(reachedNodeId), edgePath: [edge] };
}

function semanticWith(contradictionHits: RetrievedSemanticHit[]): RetrievedSemantic {
  return {
    supports: [],
    contradicts: contradictionHits.map((entry) => entry.node),
    categories: [],
    matched_node_ids: [],
    matched_nodes: [],
    support_hits: [],
    causal_hits: [],
    contradiction_hits: contradictionHits,
    category_hits: [],
  };
}

function routedContradictionCount(semantic: RetrievedSemantic): number {
  return assembleRetrievedContext({
    episodes: [],
    semantic,
    openQuestions: [],
    evidence: [],
    recallIntents: [],
    contradictionPresent: semantic.contradiction_hits.length > 0,
    nowMs: NOW,
  }).contradictionRouting.contradictions.length;
}

describe("countRetrievedContradictionRelations", () => {
  // The contradicts walk runs in both directions, so a relation whose two nodes were
  // both matched produces two traversals. The evidence ledger reports traversals and
  // the deliberation contradiction line reports relations; if the two counts are ever
  // derived independently they will silently disagree, which is exactly the divergence
  // this helper exists to keep legible.
  it("collapses a relation reached from both of its nodes into one contradiction", () => {
    const edge = contradictsEdge(EDGE_AB, NODE_A, NODE_B);
    const semantic = semanticWith([hit(NODE_A, NODE_B, edge), hit(NODE_B, NODE_A, edge)]);

    expect(semantic.contradiction_hits).toHaveLength(2);
    expect(countRetrievedContradictionRelations(semantic)).toBe(1);
  });

  it("counts distinct relations separately", () => {
    const semantic = semanticWith([
      hit(NODE_A, NODE_B, contradictsEdge(EDGE_AB, NODE_A, NODE_B)),
      hit(NODE_C, NODE_D, contradictsEdge(EDGE_CD, NODE_C, NODE_D)),
    ]);

    expect(countRetrievedContradictionRelations(semantic)).toBe(2);
  });

  it("agrees with the routing the deliberation contradiction line reports", () => {
    const shared = contradictsEdge(EDGE_AB, NODE_A, NODE_B);
    const cases = [
      semanticWith([]),
      semanticWith([hit(NODE_A, NODE_B, shared)]),
      semanticWith([hit(NODE_A, NODE_B, shared), hit(NODE_B, NODE_A, shared)]),
      semanticWith([
        hit(NODE_A, NODE_B, shared),
        hit(NODE_B, NODE_A, shared),
        hit(NODE_C, NODE_D, contradictsEdge(EDGE_CD, NODE_C, NODE_D)),
      ]),
    ];

    for (const semantic of cases) {
      expect(countRetrievedContradictionRelations(semantic)).toBe(
        routedContradictionCount(semantic),
      );
    }
  });
});
