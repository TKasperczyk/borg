import { screen } from "@testing-library/react";
import { describe, expect, it } from "vitest";

import { renderWithInspector } from "../test/inspector";
import { SemanticEdgeDetail } from "./SemanticEdgeDetail";

describe("SemanticEdgeDetail", () => {
  it("renders semantic edge content and provenance fields", () => {
    renderWithInspector(
      <SemanticEdgeDetail
        nodes={[
          {
            id: "semn_from00000000",
            kind: "proposition",
            label: "Source node",
            description: "Source description.",
            domain: "project",
            aliases: [],
            confidence: 0.9,
            status: "active",
            source_episode_ids: [],
            source_count: 0,
            created_at: 1,
            updated_at: 2,
          },
          {
            id: "semn_to000000000",
            kind: "proposition",
            label: "Target node",
            description: "Target description.",
            domain: "project",
            aliases: [],
            confidence: 0.8,
            status: "active",
            source_episode_ids: [],
            source_count: 0,
            created_at: 1,
            updated_at: 2,
          },
        ]}
        edge={{
          id: "seme_detail000000",
          from_node_id: "semn_from00000000",
          to_node_id: "semn_to000000000",
          relation: "contradicts",
          confidence: 0.76,
          evidence_episode_ids: ["ep_source000000000"],
          source_count: 1,
          valid_from: 1,
          valid_to: null,
          invalidated_at: null,
          invalidated_by_edge_id: null,
          invalidated_by_review_id: null,
          invalidated_by_process: null,
          invalidated_reason: null,
        }}
      />,
    );

    expect(screen.getByText("contradicts")).toBeInTheDocument();
    expect(screen.getByText("confidence 0.76")).toBeInTheDocument();
    expect(screen.getByText(/Source node/)).toBeInTheDocument();
    expect(screen.getByText(/Target node/)).toBeInTheDocument();
    expect(screen.getByText("seme_detail000000")).toBeInTheDocument();
    expect(screen.getByText("semn_from00000000")).toBeInTheDocument();
    expect(screen.getByText("semn_to000000000")).toBeInTheDocument();
    expect(screen.getByText("open")).toBeInTheDocument();
    expect(screen.getByText("ep_source000000000")).toBeInTheDocument();
  });
});
