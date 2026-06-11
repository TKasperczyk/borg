import type {
  SemanticGraphEdge,
  SemanticGraphEdgeType,
  SemanticGraphNode,
  SemanticGraphNodeStatus,
} from "../../api/types";

export type LayoutNode = SemanticGraphNode & {
  x: number;
  y: number;
  r: number;
};

export type LayoutEdge = SemanticGraphEdge & {
  x1: number;
  y1: number;
  x2: number;
  y2: number;
};

export type GraphLayout = {
  nodes: LayoutNode[];
  edges: LayoutEdge[];
};

export type EdgeVisualStyle = {
  stroke: string;
  strokeWidth: number;
  strokeDasharray: string;
};

const STATUS_COLORS: Record<SemanticGraphNodeStatus, string> = {
  active: "var(--ac)",
  contested: "#C9A227",
  contradicted: "oklch(0.62 0.19 25)",
  quarantined: "#5C5A52",
};

export function nodeStatusColor(status: SemanticGraphNodeStatus): string {
  return STATUS_COLORS[status];
}

export function edgeStyleForType(type: SemanticGraphEdgeType): EdgeVisualStyle {
  if (type === "contradicts") {
    return {
      stroke: "oklch(0.62 0.19 25 / 0.72)",
      strokeWidth: 1.6,
      strokeDasharray: "6 4",
    };
  }

  return {
    stroke: "rgba(150, 148, 138, 0.28)",
    strokeWidth: 1.1,
    strokeDasharray: "none",
  };
}

function clamp(value: number, min: number, max: number): number {
  return Math.min(max, Math.max(min, value));
}

export function layoutGraph(
  nodes: readonly SemanticGraphNode[],
  edges: readonly SemanticGraphEdge[],
  width = 618,
  height = 312,
): GraphLayout {
  const cx = width / 2;
  const cy = height / 2;
  const radiusX = Math.max(64, width * 0.38);
  const radiusY = Math.max(48, height * 0.32);
  const count = Math.max(1, nodes.length);
  const placed = nodes.map<LayoutNode>((node, index) => {
    const ring = index % 3;
    const angle = (index / count) * Math.PI * 2 + ring * 0.41;
    const radialScale = 0.56 + ring * 0.2;
    const edgeBoost = Math.min(36, node.edge_count * 2.5);
    const x = cx + Math.cos(angle) * (radiusX * radialScale + edgeBoost);
    const y = cy + Math.sin(angle) * (radiusY * radialScale + edgeBoost * 0.35);

    return {
      ...node,
      x: clamp(x, 28, width - 28),
      y: clamp(y, 28, height - 38),
      r: node.kind === "entity" ? 7 : 4.5,
    };
  });

  for (let pass = 0; pass < 3; pass += 1) {
    for (let leftIndex = 0; leftIndex < placed.length; leftIndex += 1) {
      for (let rightIndex = leftIndex + 1; rightIndex < placed.length; rightIndex += 1) {
        const left = placed[leftIndex]!;
        const right = placed[rightIndex]!;
        const dx = right.x - left.x;
        const dy = right.y - left.y;
        const dist = Math.max(1, Math.hypot(dx, dy));
        const minDist = left.r + right.r + 22;
        if (dist >= minDist) {
          continue;
        }

        const push = (minDist - dist) / 2;
        const ux = dx / dist;
        const uy = dy / dist;
        left.x = clamp(left.x - ux * push, 18, width - 18);
        left.y = clamp(left.y - uy * push, 18, height - 28);
        right.x = clamp(right.x + ux * push, 18, width - 18);
        right.y = clamp(right.y + uy * push, 18, height - 28);
      }
    }
  }

  const byId = new Map(placed.map((node) => [node.id, node]));
  const laidEdges = edges.flatMap<LayoutEdge>((edge) => {
    const source = byId.get(edge.source);
    const target = byId.get(edge.target);
    if (source === undefined || target === undefined) {
      return [];
    }

    return [{ ...edge, x1: source.x, y1: source.y, x2: target.x, y2: target.y }];
  });

  return { nodes: placed, edges: laidEdges };
}
