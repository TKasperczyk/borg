import type { SemanticGraphEdge, SemanticGraphNode } from "../../api/types";
import { edgeStyleForType, layoutGraph, nodeStatusColor } from "./graph";

const nodes: SemanticGraphNode[] = [
  {
    id: "n1",
    label: "operator",
    display_label: "operator",
    status: "active",
    kind: "entity",
    edge_count: 2,
  },
  {
    id: "n2",
    label: "deadline",
    display_label: "deadline",
    status: "contested",
    kind: "proposition",
    edge_count: 1,
  },
  {
    id: "n3",
    label: "private notes",
    display_label: "private notes",
    status: "quarantined",
    kind: "concept",
    edge_count: 1,
  },
];

const edges: SemanticGraphEdge[] = [
  { id: "e1", source: "n1", target: "n2", type: "supports", weight: 0.8 },
  { id: "e2", source: "n2", target: "n3", type: "contradicts", weight: 0.4 },
];

describe("mind graph helpers", () => {
  it("lays out nodes deterministically within the viewbox", () => {
    const first = layoutGraph(nodes, edges);
    const second = layoutGraph(nodes, edges);

    expect(first).toEqual(second);
    for (const node of first.nodes) {
      expect(Number.isFinite(node.x)).toBe(true);
      expect(Number.isFinite(node.y)).toBe(true);
      expect(node.x).toBeGreaterThanOrEqual(0);
      expect(node.x).toBeLessThanOrEqual(618);
      expect(node.y).toBeGreaterThanOrEqual(0);
      expect(node.y).toBeLessThanOrEqual(312);
    }
  });

  it("maps node status and real edge relation enums to visual styles", () => {
    expect(nodeStatusColor("active")).toBe("var(--ac)");
    expect(nodeStatusColor("contested")).toBe("#C9A227");
    expect(nodeStatusColor("contradicted")).toBe("oklch(0.62 0.19 25)");
    expect(nodeStatusColor("quarantined")).toBe("#5C5A52");

    expect(edgeStyleForType("contradicts")).toMatchObject({ strokeDasharray: "6 4" });
    expect(edgeStyleForType("supports")).toMatchObject({ strokeDasharray: "none" });
  });
});
