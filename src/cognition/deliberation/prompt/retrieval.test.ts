import { describe, expect, it } from "vitest";

import type { SemanticEdge, SemanticNode } from "../../../memory/semantic/index.js";
import type { RetrievedSemantic } from "../../../retrieval/index.js";
import { ManualClock } from "../../../util/clock.js";
import { summarizeSemanticContext } from "./retrieval.js";

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
      contradiction_hits: [],
      category_hits: [],
    } satisfies RetrievedSemantic;

    const beforeClose = summarizeSemanticContext(retrievedSemantic, 1_000, clock.now());
    clock.set(Date.UTC(2024, 0, 11));
    const afterClose = summarizeSemanticContext(retrievedSemantic, 1_000, clock.now());

    expect(beforeClose).toContain("-[supports");
    expect(afterClose).not.toContain("-[supports");
  });
});
