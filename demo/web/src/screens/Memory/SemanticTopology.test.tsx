import { fireEvent, render, screen, waitFor, within } from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import type { SemanticGraphResponse } from "../../api/types";
import { SemanticTopology, edgeClass } from "./SemanticTopology";

function graphResponse(): SemanticGraphResponse {
  return {
    nodes: [
      { id: "semn_alpha00000000", label: "Sol", status: "active", kind: "entity", edge_count: 3 },
      { id: "semn_beta000000000", label: "Sol", status: "active", kind: "entity", edge_count: 2 },
      {
        id: "semn_gamma00000000",
        label: "Semantic memory",
        status: "contested",
        kind: "concept",
        edge_count: 2,
      },
      {
        id: "semn_delta00000000",
        label: "Support claim",
        status: "contradicted",
        kind: "proposition",
        edge_count: 1,
      },
    ],
    edges: [
      {
        id: "seme_instance00000",
        source: "semn_alpha00000000",
        target: "semn_gamma00000000",
        type: "instance_of",
        weight: 0.8,
      },
      {
        id: "seme_part000000000",
        source: "semn_beta000000000",
        target: "semn_gamma00000000",
        type: "part_of",
        weight: 0.7,
      },
      {
        id: "seme_related00000",
        source: "semn_gamma00000000",
        target: "semn_delta00000000",
        type: "related_to",
        weight: 0.6,
      },
    ],
    total_nodes: 4,
    total_edges: 3,
    rendered: { nodes: 4, edges: 3 },
  };
}

describe("SemanticTopology", () => {
  beforeEach(() => {
    vi.spyOn(window, "requestAnimationFrame").mockImplementation((callback) =>
      window.setTimeout(() => callback(performance.now()), 16),
    );
    vi.spyOn(window, "cancelAnimationFrame").mockImplementation((id) => {
      window.clearTimeout(id);
    });
  });

  afterEach(() => {
    vi.restoreAllMocks();
  });

  it("uses distinct classes for semantic relations that the old graph bucketed as other", async () => {
    const { container } = render(
      <SemanticTopology graph={graphResponse()} selectedId={null} onSelectNode={vi.fn()} />,
    );

    expect(edgeClass("instance_of")).toBe("semantic-topology-edge-instance-of");
    expect(edgeClass("part_of")).toBe("semantic-topology-edge-part-of");
    expect(edgeClass("related_to")).toBe("semantic-topology-edge-related-to");

    await waitFor(() => {
      expect(container.querySelector(".semantic-topology-edge-instance-of")).toBeInTheDocument();
      expect(container.querySelector(".semantic-topology-edge-part-of")).toBeInTheDocument();
      expect(container.querySelector(".semantic-topology-edge-related-to")).toBeInTheDocument();
    });
  });

  it("selects nodes by click and keyboard and highlights the selected node", async () => {
    const onSelectNode = vi.fn();
    const { container, rerender } = render(
      <SemanticTopology
        graph={graphResponse()}
        selectedId="semn_beta000000000"
        onSelectNode={onSelectNode}
      />,
    );

    await waitFor(() => {
      expect(container.querySelectorAll(".semantic-topology-node")).toHaveLength(4);
    });
    const selected = container.querySelector<SVGGElement>('[data-node-id="semn_beta000000000"]');
    expect(selected).not.toBeNull();
    expect(selected).toHaveClass("selected");

    fireEvent.click(within(container).getByRole("button", { name: /Semantic memory/ }));
    expect(onSelectNode).toHaveBeenCalledWith("semn_gamma00000000");

    fireEvent.keyDown(within(container).getByRole("button", { name: /Support claim/ }), {
      key: "Enter",
    });
    expect(onSelectNode).toHaveBeenCalledWith("semn_delta00000000");

    rerender(
      <SemanticTopology
        graph={graphResponse()}
        selectedId="semn_delta00000000"
        onSelectNode={onSelectNode}
      />,
    );
    await waitFor(() => {
      expect(container.querySelector('[data-node-id="semn_delta00000000"]')).toHaveClass(
        "selected",
      );
    });
  });

  it("disambiguates duplicate labels with ordinal badges and a cluster legend", async () => {
    const { container } = render(
      <SemanticTopology graph={graphResponse()} selectedId={null} onSelectNode={vi.fn()} />,
    );

    expect(await screen.findByText("Sol x2")).toBeInTheDocument();
    await waitFor(() => {
      expect(container.querySelectorAll(".semantic-topology-node-badge")).toHaveLength(2);
    });
    expect(container.textContent).toContain("Sol #1");
    expect(container.textContent).toContain("Sol #2");
  });
});
