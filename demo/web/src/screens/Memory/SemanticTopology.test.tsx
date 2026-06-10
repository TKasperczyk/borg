import { act, fireEvent, screen, waitFor, within } from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import type { SemanticGraphResponse } from "../../api/types";
import { renderWithInspector } from "../../test/inspector";
import { SemanticTopology, edgeClass } from "./SemanticTopology";

function graphResponse(): SemanticGraphResponse {
  return {
    nodes: [
      {
        id: "semn_alpha00000000",
        label: "ent_abcdefghijklmnop",
        display_label: "Sol",
        status: "active",
        kind: "entity",
        edge_count: 3,
      },
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

function flowGraphResponse(): SemanticGraphResponse {
  const graph = graphResponse();
  return {
    ...graph,
    edges: [
      ...graph.edges,
      {
        id: "seme_supports00000",
        source: "semn_alpha00000000",
        target: "semn_delta00000000",
        type: "supports",
        weight: 0.9,
      },
    ],
    total_edges: graph.total_edges + 1,
    rendered: { ...graph.rendered, edges: graph.rendered.edges + 1 },
  };
}

function clusteredGraphResponse(): SemanticGraphResponse {
  const nodeCount = 36;
  const edgeCounts = Array.from({ length: nodeCount }, () => 0);
  const edges: SemanticGraphResponse["edges"] = [];
  const nodeId = (index: number) => `semn_cluster_${String(index).padStart(4, "0")}`;
  const addEdge = (sourceIndex: number, targetIndex: number) => {
    const source = sourceIndex % nodeCount;
    const target = targetIndex % nodeCount;
    const edgeIndex = edges.length;
    edgeCounts[source] = (edgeCounts[source] ?? 0) + 1;
    edgeCounts[target] = (edgeCounts[target] ?? 0) + 1;
    edges.push({
      id: `seme_cluster_${String(edgeIndex).padStart(4, "0")}`,
      source: nodeId(source),
      target: nodeId(target),
      type: edgeIndex % 3 === 0 ? "supports" : edgeIndex % 3 === 1 ? "part_of" : "related_to",
      weight: 0.5 + (edgeIndex % 5) * 0.08,
    });
  };

  for (let clusterStart = 0; clusterStart < nodeCount; clusterStart += 12) {
    for (let offset = 0; offset < 12; offset += 1) {
      addEdge(clusterStart + offset, clusterStart + ((offset + 1) % 12));
    }
    for (let offset = 0; offset < 6; offset += 1) {
      addEdge(clusterStart + offset, clusterStart + offset + 6);
    }
  }

  for (let index = 0; index < nodeCount; index += 3) {
    addEdge(index, index + 12);
  }

  return {
    nodes: Array.from({ length: nodeCount }, (_, index) => ({
      id: nodeId(index),
      label: `Cluster node ${index}`,
      status: "active",
      kind: index % 7 === 0 ? "proposition" : index % 5 === 0 ? "concept" : "entity",
      edge_count: edgeCounts[index] ?? 0,
    })),
    edges,
    total_nodes: nodeCount,
    total_edges: edges.length,
    rendered: { nodes: nodeCount, edges: edges.length },
  };
}

function installReducedMotion(matches: boolean) {
  vi.stubGlobal(
    "matchMedia",
    vi.fn(() => ({
      addEventListener: vi.fn(),
      addListener: vi.fn(),
      dispatchEvent: vi.fn(),
      matches,
      media: "(prefers-reduced-motion: reduce)",
      onchange: null,
      removeEventListener: vi.fn(),
      removeListener: vi.fn(),
    })),
  );
}

function renderedNodePositions(
  container: HTMLElement,
): Array<{ id: string; radius: number; x: number; y: number }> {
  return [...container.querySelectorAll<SVGGElement>(".semantic-topology-node")].map((node) => {
    const transform = node.getAttribute("transform") ?? "";
    const match = /^translate\((-?\d+(?:\.\d+)?) (-?\d+(?:\.\d+)?)\)$/.exec(transform);

    if (match === null) {
      throw new Error(`Unexpected topology node transform: ${transform}`);
    }

    return {
      id: node.dataset.nodeId ?? "",
      radius: Number(node.dataset.radius),
      x: Number(match[1]),
      y: Number(match[2]),
    };
  });
}

function minimumNodeGap(
  nodes: readonly { id: string; radius: number; x: number; y: number }[],
): number {
  let minGap = Number.POSITIVE_INFINITY;

  for (let leftIndex = 0; leftIndex < nodes.length; leftIndex += 1) {
    for (let rightIndex = leftIndex + 1; rightIndex < nodes.length; rightIndex += 1) {
      const left = nodes[leftIndex]!;
      const right = nodes[rightIndex]!;
      const gap = Math.hypot(right.x - left.x, right.y - left.y) - (left.radius + right.radius);
      minGap = Math.min(minGap, gap);
    }
  }

  return minGap;
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
    vi.unstubAllGlobals();
  });

  it("uses distinct classes for semantic relations that the old graph bucketed as other", async () => {
    const { container } = renderWithInspector(
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

  it("keeps edge flow off at rest and gates dash motion to selected or hovered connected edges", async () => {
    const graph = flowGraphResponse();
    const { container, rerender } = renderWithInspector(
      <SemanticTopology graph={graph} selectedId={null} onSelectNode={vi.fn()} />,
    );

    await waitFor(() => {
      expect(container.querySelector('[data-edge-id="seme_supports00000"]')).toBeInTheDocument();
    });

    const supports = container.querySelector<SVGLineElement>('[data-edge-id="seme_supports00000"]');
    const related = container.querySelector<SVGLineElement>('[data-edge-id="seme_related00000"]');
    expect(edgeClass("supports")).toBe("semantic-topology-edge-supports");
    expect(edgeClass("causes")).toBe("semantic-topology-edge-causes");
    expect(supports).not.toHaveClass("semantic-topology-edge-flow");

    rerender(
      <SemanticTopology graph={graph} selectedId="semn_alpha00000000" onSelectNode={vi.fn()} />,
    );
    await waitFor(() => {
      expect(supports).toHaveClass("semantic-topology-edge-flow");
    });
    expect(related).not.toHaveClass("semantic-topology-edge-flow");

    rerender(<SemanticTopology graph={graph} selectedId={null} onSelectNode={vi.fn()} />);
    await waitFor(() => {
      expect(supports).not.toHaveClass("semantic-topology-edge-flow");
    });

    const delta = container.querySelector<SVGGElement>('[data-node-id="semn_delta00000000"]');
    expect(delta).not.toBeNull();
    fireEvent.pointerEnter(delta as SVGGElement);
    expect(supports).toHaveClass("semantic-topology-edge-flow");

    fireEvent.pointerLeave(delta as SVGGElement);
    expect(supports).not.toHaveClass("semantic-topology-edge-flow");
  });

  it("stops scheduling animation frames once the force simulation converges", () => {
    const callbacks = new Map<number, FrameRequestCallback>();
    let nextFrame = 1;
    const requestFrame = vi
      .spyOn(window, "requestAnimationFrame")
      .mockImplementation((callback) => {
        const frame = nextFrame;
        nextFrame += 1;
        callbacks.set(frame, callback);
        return frame;
      });
    vi.spyOn(window, "cancelAnimationFrame").mockImplementation((frame) => {
      callbacks.delete(frame);
    });

    const { container } = renderWithInspector(
      <SemanticTopology
        graph={clusteredGraphResponse()}
        selectedId={null}
        onSelectNode={vi.fn()}
      />,
    );

    let now = performance.now();
    let stoppedOnFrame = false;
    let requestCountBeforeStoppedFrame = 0;
    let requestCountAfterStoppedFrame = 0;

    act(() => {
      for (let frameCount = 0; frameCount < 800 && callbacks.size > 0; frameCount += 1) {
        const next = callbacks.entries().next().value;
        if (next === undefined) {
          return;
        }
        const [frame, callback] = next;
        callbacks.delete(frame);
        now += 16.67;
        const requestCountBeforeFrame = requestFrame.mock.calls.length;
        callback(now);

        if (callbacks.size === 0) {
          stoppedOnFrame = true;
          requestCountBeforeStoppedFrame = requestCountBeforeFrame;
          requestCountAfterStoppedFrame = requestFrame.mock.calls.length;
        }
      }
    });

    const renderedNodes = renderedNodePositions(container);

    expect(stoppedOnFrame).toBe(true);
    expect(callbacks.size).toBe(0);
    expect(requestFrame.mock.calls.length).toBeLessThan(800);
    expect(requestCountAfterStoppedFrame).toBe(requestCountBeforeStoppedFrame);
    expect(minimumNodeGap(renderedNodes)).toBeGreaterThanOrEqual(-0.01);
  });

  it("cancels a pending topology animation frame on unmount", () => {
    const callbacks = new Map<number, FrameRequestCallback>();
    let nextFrame = 1;
    vi.spyOn(window, "requestAnimationFrame").mockImplementation((callback) => {
      const frame = nextFrame;
      nextFrame += 1;
      callbacks.set(frame, callback);
      return frame;
    });
    const cancelFrame = vi.spyOn(window, "cancelAnimationFrame").mockImplementation((frame) => {
      callbacks.delete(frame);
    });

    const { unmount } = renderWithInspector(
      <SemanticTopology graph={graphResponse()} selectedId={null} onSelectNode={vi.fn()} />,
    );
    const pendingFrame = callbacks.keys().next().value;
    expect(pendingFrame).toBeDefined();

    unmount();

    expect(cancelFrame).toHaveBeenCalledWith(pendingFrame);
    expect(callbacks.size).toBe(0);
  });

  it("renders a static topology without requestAnimationFrame under reduced motion", () => {
    installReducedMotion(true);
    const requestFrame = vi.spyOn(window, "requestAnimationFrame").mockImplementation(() => 1);

    const { container } = renderWithInspector(
      <SemanticTopology graph={graphResponse()} selectedId={null} onSelectNode={vi.fn()} />,
    );

    expect(container.querySelector(".semantic-topology-node")).toBeInTheDocument();
    expect(requestFrame).not.toHaveBeenCalled();
  });

  it("selects nodes by click and keyboard and highlights the selected node", async () => {
    const onSelectNode = vi.fn();
    const { container, rerender } = renderWithInspector(
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
    const { container } = renderWithInspector(
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
