import { describe, expect, it } from "vitest";

import type { SemanticEdge, SemanticNode } from "../../../memory/semantic/index.js";
import type { RetrievalConfidence, RetrievedSemantic } from "../../../retrieval/index.js";
import { ManualClock } from "../../../util/clock.js";
import {
  summarizeRetrievalConfidence,
  summarizeRetrievedEpisodes,
  summarizeSemanticContext,
} from "./retrieval.js";

function makeRetrievalConfidence(
  overrides: Partial<RetrievalConfidence> = {},
): RetrievalConfidence {
  return {
    overall: overrides.overall ?? 0,
    evidenceStrength: overrides.evidenceStrength ?? 0,
    coverage: overrides.coverage ?? 0,
    sourceDiversity: overrides.sourceDiversity ?? 0,
    contradictionPresent: overrides.contradictionPresent ?? false,
    sampleSize: overrides.sampleSize ?? 0,
  };
}

function makeNode(overrides: Partial<SemanticNode> = {}): SemanticNode {
  return {
    id: overrides.id ?? ("semn_aaaaaaaaaaaaaaaa" as SemanticNode["id"]),
    kind: overrides.kind ?? "proposition",
    label: overrides.label ?? "Atlas claim",
    description: overrides.description ?? "A claim about Atlas deployment state.",
    domain: overrides.domain ?? null,
    aliases: overrides.aliases ?? [],
    confidence: overrides.confidence ?? 0.7,
    source_episode_ids:
      overrides.source_episode_ids ??
      (["ep_aaaaaaaaaaaaaaaa"] as SemanticNode["source_episode_ids"]),
    created_at: overrides.created_at ?? 0,
    updated_at: overrides.updated_at ?? 0,
    last_verified_at: overrides.last_verified_at ?? 0,
    embedding: overrides.embedding ?? Float32Array.from([1, 0, 0, 0]),
    archived: overrides.archived ?? false,
    superseded_by: overrides.superseded_by ?? null,
  };
}

function makeClosedEdge(overrides: Partial<SemanticEdge> = {}): SemanticEdge {
  return {
    id: overrides.id ?? ("seme_aaaaaaaaaaaaaaaa" as SemanticEdge["id"]),
    from_node_id:
      overrides.from_node_id ?? ("semn_aaaaaaaaaaaaaaaa" as SemanticEdge["from_node_id"]),
    to_node_id: overrides.to_node_id ?? ("semn_bbbbbbbbbbbbbbbb" as SemanticEdge["to_node_id"]),
    relation: overrides.relation ?? "supports",
    confidence: overrides.confidence ?? 0.7,
    evidence_episode_ids:
      overrides.evidence_episode_ids ??
      (["ep_aaaaaaaaaaaaaaaa"] as SemanticEdge["evidence_episode_ids"]),
    created_at: overrides.created_at ?? Date.UTC(2024, 0, 1),
    last_verified_at: overrides.last_verified_at ?? Date.UTC(2024, 0, 1),
    valid_from: overrides.valid_from ?? Date.UTC(2024, 0, 1),
    valid_to: overrides.valid_to ?? Date.UTC(2024, 0, 10),
    invalidated_at: overrides.invalidated_at ?? Date.UTC(2024, 0, 12),
    invalidated_by_edge_id: overrides.invalidated_by_edge_id ?? null,
    invalidated_by_review_id: overrides.invalidated_by_review_id ?? null,
    invalidated_by_process: overrides.invalidated_by_process ?? "manual",
    invalidated_reason: overrides.invalidated_reason ?? "superseded",
  };
}

describe("retrieval confidence prompt rendering", () => {
  it("renders an explicit no-evidence policy when confidence has zero samples", () => {
    const summary = summarizeRetrievalConfidence(makeRetrievalConfidence());

    expect(summary).not.toBeNull();
    expect(summary).toContain("overall=0.00");
    expect(summary).toContain("samples=0");
    expect(summary).toContain("No relevant memory was retrieved for this turn.");
    expect(summary).toContain("tool.openQuestions.create");
    expect(summary).toContain("Do not fabricate specifics");
  });

  it("adds grounding policy only when non-empty confidence is low", () => {
    const low = summarizeRetrievalConfidence(
      makeRetrievalConfidence({ overall: 0.2, evidenceStrength: 0.2, sampleSize: 1 }),
    );
    const healthy = summarizeRetrievalConfidence(
      makeRetrievalConfidence({ overall: 0.8, evidenceStrength: 0.8, sampleSize: 3 }),
    );

    expect(low).toContain("must not over-claim");
    expect(low).toContain("tool.openQuestions.create");
    expect(healthy).not.toContain("tool.openQuestions.create");
    expect(healthy).not.toContain("Policy:");
  });

  it("renders an empty retrieved-episodes placeholder", () => {
    const summary = summarizeRetrievedEpisodes("Retrieved context", []);

    expect(summary).toBe("No episodes retrieved for this turn.");
  });
});

describe("semantic retrieval prompt rendering", () => {
  it("tags closed path edges for historical as-of context", () => {
    const root = makeNode({
      id: "semn_aaaaaaaaaaaaaaaa" as SemanticNode["id"],
      kind: "entity",
      label: "Atlas",
      description: "Atlas deployment service.",
    });
    const support = makeNode({
      id: "semn_bbbbbbbbbbbbbbbb" as SemanticNode["id"],
      label: "Rerun install",
      description: "Rerun pnpm install before deploying Atlas.",
    });
    const edge = makeClosedEdge({
      from_node_id: root.id,
      to_node_id: support.id,
    });
    const summary = summarizeSemanticContext(
      {
        as_of: Date.UTC(2024, 0, 5),
        matched_node_ids: [root.id],
        matched_nodes: [root],
        supports: [support],
        contradicts: [],
        categories: [],
        support_hits: [
          {
            root_node_id: root.id,
            node: support,
            edgePath: [edge],
          },
        ],
        causal_hits: [],
        contradiction_hits: [],
        category_hits: [],
      } satisfies RetrievedSemantic,
      1_000,
    );

    expect(summary).toContain("[valid 2024-01-01..2024-01-10, closed 2024-01-12]");
  });

  it("does not render closed path edges in current mode and marks historical direct matches", () => {
    const root = makeNode({
      id: "semn_aaaaaaaaaaaaaaaa" as SemanticNode["id"],
      kind: "entity",
      label: "Atlas",
      description: "Atlas deployment service.",
    });
    const support = makeNode({
      id: "semn_bbbbbbbbbbbbbbbb" as SemanticNode["id"],
      label: "Rerun install",
      description: "Rerun pnpm install before deploying Atlas.",
    });
    const historical = {
      ...makeNode({
        id: "semn_cccccccccccccccc" as SemanticNode["id"],
        label: "Closed Atlas proposition",
        description: "A proposition whose support is no longer current.",
      }),
      historical: true,
    };
    const summary = summarizeSemanticContext(
      {
        matched_node_ids: [root.id, historical.id],
        matched_nodes: [root, historical],
        supports: [support],
        contradicts: [],
        categories: [],
        support_hits: [
          {
            root_node_id: root.id,
            node: support,
            edgePath: [
              makeClosedEdge({
                from_node_id: root.id,
                to_node_id: support.id,
              }),
            ],
          },
        ],
        causal_hits: [],
        contradiction_hits: [],
        category_hits: [],
      } satisfies RetrievedSemantic,
      1_000,
    );

    expect(summary).toContain("Closed Atlas proposition [historical]");
    expect(summary).not.toContain("-[supports");
    expect(summary).not.toContain("[valid 2024-01-01..2024-01-10");
  });

  it("uses injected current time when filtering current-mode closed edges", () => {
    const clock = new ManualClock(Date.UTC(2024, 0, 5));
    const root = makeNode({
      id: "semn_aaaaaaaaaaaaaaaa" as SemanticNode["id"],
      kind: "entity",
      label: "Atlas",
      description: "Atlas deployment service.",
    });
    const support = makeNode({
      id: "semn_bbbbbbbbbbbbbbbb" as SemanticNode["id"],
      label: "Rerun install",
      description: "Rerun pnpm install before deploying Atlas.",
    });
    const edge = makeClosedEdge({
      from_node_id: root.id,
      to_node_id: support.id,
      valid_to: Date.UTC(2024, 0, 10),
    });
    const retrievedSemantic = {
      matched_node_ids: [root.id],
      matched_nodes: [root],
      supports: [support],
      contradicts: [],
      categories: [],
      support_hits: [
        {
          root_node_id: root.id,
          node: support,
          edgePath: [edge],
        },
      ],
      causal_hits: [],
      contradiction_hits: [],
      category_hits: [],
    } satisfies RetrievedSemantic;

    const beforeClose = summarizeSemanticContext(retrievedSemantic, 1_000, clock.now());
    clock.set(Date.UTC(2024, 0, 11));
    const afterClose = summarizeSemanticContext(retrievedSemantic, 1_000, clock.now());

    expect(beforeClose).toContain("-[supports");
    expect(afterClose).not.toContain("-[supports");
  });

  it("renders causal semantic hits in a separate bucket", () => {
    const root = makeNode({
      id: "semn_aaaaaaaaaaaaaaaa" as SemanticNode["id"],
      kind: "entity",
      label: "Atlas",
      description: "Atlas deployment service.",
    });
    const effect = makeNode({
      id: "semn_bbbbbbbbbbbbbbbb" as SemanticNode["id"],
      label: "Rollback pressure",
      description: "Atlas rollback pressure rises after failed deploys.",
    });
    const edge = makeClosedEdge({
      from_node_id: root.id,
      to_node_id: effect.id,
      relation: "causes",
      valid_to: Date.UTC(2099, 0, 1),
    });
    const summary = summarizeSemanticContext(
      {
        matched_node_ids: [root.id],
        matched_nodes: [root],
        supports: [],
        contradicts: [],
        categories: [],
        support_hits: [],
        causal_hits: [
          {
            root_node_id: root.id,
            node: effect,
            edgePath: [edge],
          },
        ],
        contradiction_hits: [],
        category_hits: [],
      } satisfies RetrievedSemantic,
      1_000,
      Date.UTC(2024, 0, 5),
    );

    expect(summary).toContain("causal:");
    expect(summary).toContain("-[causes");
  });

  it("labels under-review direct semantic matches", () => {
    const underReview = {
      ...makeNode({
        label: "Atlas claim under review",
      }),
      under_review: {
        review_id: 1,
        reason: "Supporting semantic edge was invalidated; target needs re-evaluation",
        reason_code: "support_chain_collapsed",
        invalidated_edge_id: "seme_aaaaaaaaaaaaaaaa",
      },
    } satisfies RetrievedSemantic["matched_nodes"][number];
    const summary = summarizeSemanticContext(
      {
        matched_node_ids: [underReview.id],
        matched_nodes: [underReview],
        supports: [],
        contradicts: [],
        categories: [],
        support_hits: [],
        causal_hits: [],
        contradiction_hits: [],
        category_hits: [],
      } satisfies RetrievedSemantic,
      1_000,
    );

    expect(summary).toContain("[under re-evaluation: support_chain_collapsed]");
    expect(summary).toContain("Atlas claim under review");
  });

  it("does not label nodes without an open under-review marker", () => {
    const closedReviewNode = makeNode({
      label: "Closed review claim",
    });
    const summary = summarizeSemanticContext(
      {
        matched_node_ids: [closedReviewNode.id],
        matched_nodes: [closedReviewNode],
        supports: [],
        contradicts: [],
        categories: [],
        support_hits: [],
        causal_hits: [],
        contradiction_hits: [],
        category_hits: [],
      } satisfies RetrievedSemantic,
      1_000,
    );

    expect(summary).toContain("Closed review claim");
    expect(summary).not.toContain("[under re-evaluation:");
  });

  it("labels multiple under-review semantic nodes inline", () => {
    const first = {
      ...makeNode({
        id: "semn_bbbbbbbbbbbbbbbb" as SemanticNode["id"],
        label: "First weak claim",
      }),
      under_review: {
        review_id: 1,
        reason: "First support was invalidated",
        reason_code: "evidence_invalidated",
        invalidated_edge_id: "seme_bbbbbbbbbbbbbbbb",
      },
    } satisfies RetrievedSemantic["matched_nodes"][number];
    const second = {
      ...makeNode({
        id: "semn_cccccccccccccccc" as SemanticNode["id"],
        label: "Second weak claim",
      }),
      under_review: {
        review_id: 2,
        reason: "Second support was invalidated",
        reason_code: "support_chain_collapsed",
        invalidated_edge_id: "seme_cccccccccccccccc",
      },
    } satisfies RetrievedSemantic["matched_nodes"][number];
    const summary = summarizeSemanticContext(
      {
        matched_node_ids: [first.id, second.id],
        matched_nodes: [first, second],
        supports: [],
        contradicts: [],
        categories: [],
        support_hits: [],
        causal_hits: [],
        contradiction_hits: [],
        category_hits: [],
      } satisfies RetrievedSemantic,
      1_000,
    );

    expect(summary?.match(/\[under re-evaluation:/g)).toHaveLength(2);
    expect(summary).toContain("First weak claim");
    expect(summary).toContain("Second weak claim");
  });
});
