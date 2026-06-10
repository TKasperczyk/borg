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
          disclosure_class: "operator_private",
          disclosure_label: {
            disclosure_class: "operator_private",
            origin_audience_entity_ids: ["ent_bbbbbbbbbbbbbbbb"],
            private_to_entity_ids: ["ent_bbbbbbbbbbbbbbbb"],
            public_to_entity_ids: [],
          },
          origin_audience_refs: [
            { value: "ent_bbbbbbbbbbbbbbbb", id: "ent_bbbbbbbbbbbbbbbb", label: "Operator" },
          ],
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
    expect(screen.getByText("operator")).toHaveClass("tag", "warn");
    expect(screen.getByText("Operator")).toBeInTheDocument();
    expect(screen.getByText("confidence 0.76")).toBeInTheDocument();
    expect(screen.getByText(/Source node/)).toBeInTheDocument();
    expect(screen.getByText(/Target node/)).toBeInTheDocument();
    expect(screen.getByRole("button", { name: "jump to seme_detail000000" })).toBeInTheDocument();
    expect(screen.getByRole("button", { name: "jump to semn_from00000000" })).toBeInTheDocument();
    expect(screen.getByRole("button", { name: "jump to semn_to000000000" })).toBeInTheDocument();
    expect(screen.getByText("open")).toBeInTheDocument();
    expect(screen.getByRole("button", { name: "jump to ep_source000000000" })).toBeInTheDocument();
  });

  it("fails closed to an unknown chip when semantic edge disclosure fields are absent", () => {
    renderWithInspector(
      <SemanticEdgeDetail
        nodes={[]}
        edge={{
          id: "seme_unknown000000",
          from_node_id: "semn_from00000000",
          to_node_id: "semn_to000000000",
          relation: "supports",
          confidence: 0.5,
          evidence_episode_ids: [],
          source_count: 0,
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

    expect(screen.getByText("unknown")).toHaveClass("tag", "solid");
  });
});
