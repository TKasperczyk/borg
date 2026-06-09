import { useEffect, useMemo, useRef } from "react";
import type { SemanticRelation } from "borg";

import type {
  SemanticGraphEdge,
  SemanticGraphNode,
  SemanticGraphNodeStatus,
  SemanticGraphResponse,
} from "../../api/types";
import { IdRef } from "../../components/Inspector/IdRef";
import { Tag } from "../../components/Tag";
import { shortId } from "../screen-utils";

const VIEWBOX_WIDTH = 1000;
const VIEWBOX_HEIGHT = 640;
const CENTER_X = VIEWBOX_WIDTH / 2;
const CENTER_Y = VIEWBOX_HEIGHT / 2;
const NODE_MIN_RADIUS = 4;
const NODE_MAX_RADIUS = 14;
const INITIAL_RING_RADIUS = 230;
const SETTLE_MS = 4_200;
const SVG_NS = "http://www.w3.org/2000/svg";

export const SEMANTIC_TOPOLOGY_EDGE_CLASSES = {
  is_a: "semantic-topology-edge-is-a",
  part_of: "semantic-topology-edge-part-of",
  causes: "semantic-topology-edge-causes semantic-topology-edge-flow",
  prevents: "semantic-topology-edge-prevents",
  supports: "semantic-topology-edge-supports semantic-topology-edge-flow",
  contradicts: "semantic-topology-edge-contradicts",
  related_to: "semantic-topology-edge-related-to",
  instance_of: "semantic-topology-edge-instance-of",
} satisfies Record<SemanticRelation, string>;

export function edgeClass(type: SemanticRelation): string {
  return SEMANTIC_TOPOLOGY_EDGE_CLASSES[type];
}

type DuplicateInfo = {
  count: number;
  ordinal: number;
};

type DuplicateCluster = {
  label: string;
  count: number;
};

type SimNode = SemanticGraphNode & {
  duplicate: DuplicateInfo;
  index: number;
  radius: number;
  x: number;
  y: number;
  vx: number;
  vy: number;
};

type SimEdge = SemanticGraphEdge & {
  sourceNode: SimNode;
  targetNode: SimNode;
};

type MountedNode = {
  node: SimNode;
  group: SVGGElement;
};

type MountedEdge = {
  edge: SimEdge;
  line: SVGLineElement;
};

export type SemanticTopologyProps = {
  graph: SemanticGraphResponse;
  selectedId: string | null;
  onSelectNode: (nodeId: string) => void;
};

function nodeRadius(edgeCount: number): number {
  return Math.min(NODE_MAX_RADIUS, Math.max(NODE_MIN_RADIUS, 4 + Math.sqrt(edgeCount) * 3.2));
}

function edgeWidth(weight: number | undefined): number {
  const normalized = Math.min(1, Math.max(0, weight ?? 0.5));
  return 0.6 + normalized * 0.6;
}

function statusColor(status: SemanticGraphNodeStatus): string {
  switch (status) {
    case "active":
      return "var(--acc)";
    case "contested":
      return "var(--warn)";
    case "contradicted":
      return "var(--bad)";
    case "quarantined":
      return "var(--text-faint)";
  }
}

function compactLabel(label: string): string {
  return label.length > 28 ? `${label.slice(0, 25)}...` : label;
}

function createSvgElement<K extends keyof SVGElementTagNameMap>(
  tagName: K,
): SVGElementTagNameMap[K] {
  return document.createElementNS(SVG_NS, tagName);
}

function clearSvg(svg: SVGSVGElement): void {
  while (svg.firstChild !== null) {
    svg.removeChild(svg.firstChild);
  }
}

function duplicateInfoForNodes(nodes: readonly SemanticGraphNode[]): Map<string, DuplicateInfo> {
  const counts = new Map<string, number>();
  const seen = new Map<string, number>();
  const info = new Map<string, DuplicateInfo>();

  for (const node of nodes) {
    counts.set(node.label, (counts.get(node.label) ?? 0) + 1);
  }

  for (const node of nodes) {
    const count = counts.get(node.label) ?? 1;
    const ordinal = (seen.get(node.label) ?? 0) + 1;
    seen.set(node.label, ordinal);
    info.set(node.id, { count, ordinal });
  }

  return info;
}

function duplicateClustersForNodes(nodes: readonly SemanticGraphNode[]): DuplicateCluster[] {
  const counts = new Map<string, number>();

  for (const node of nodes) {
    counts.set(node.label, (counts.get(node.label) ?? 0) + 1);
  }

  return [...counts.entries()]
    .filter(([, count]) => count > 1)
    .map(([label, count]) => ({ label, count }))
    .sort((left, right) => left.label.localeCompare(right.label));
}

function createSimNodes(
  data: SemanticGraphResponse,
  duplicateInfo: ReadonlyMap<string, DuplicateInfo>,
): SimNode[] {
  const total = Math.max(1, data.nodes.length);

  return data.nodes.map((node, index) => {
    const ring = INITIAL_RING_RADIUS + ((index % 5) - 2) * 22;
    const angle = (index / total) * Math.PI * 2;

    return {
      ...node,
      duplicate: duplicateInfo.get(node.id) ?? { count: 1, ordinal: 1 },
      index,
      radius: nodeRadius(node.edge_count),
      x: CENTER_X + Math.cos(angle) * ring,
      y: CENTER_Y + Math.sin(angle) * ring * 0.72,
      vx: 0,
      vy: 0,
    };
  });
}

function createSimEdges(data: SemanticGraphResponse, nodes: readonly SimNode[]): SimEdge[] {
  const byId = new Map(nodes.map((node) => [node.id, node]));

  return data.edges.flatMap((edge) => {
    const sourceNode = byId.get(edge.source);
    const targetNode = byId.get(edge.target);

    if (sourceNode === undefined || targetNode === undefined) {
      return [];
    }

    return [{ ...edge, sourceNode, targetNode }];
  });
}

function nodeDisplayLabel(node: SimNode): string {
  const label = compactLabel(node.label);
  return node.duplicate.count > 1 ? `${label} #${node.duplicate.ordinal}` : label;
}

function nodeAccessibleLabel(node: SimNode): string {
  const duplicate =
    node.duplicate.count > 1
      ? ` duplicate ${node.duplicate.ordinal} of ${node.duplicate.count}`
      : "";
  return `${node.label}${duplicate}, ${node.kind ?? "node"}, ${node.status}, ${node.edge_count} edges, ${shortId(node.id)}`;
}

function appendNodeShape(group: SVGGElement, node: SimNode, color: string): void {
  const bodyClass = "semantic-topology-node-body";

  if (node.kind === "proposition") {
    const body = createSvgElement("polygon");
    const r = node.radius;
    body.setAttribute("points", `0,${-r} ${r},0 0,${r} ${-r},0`);
    body.setAttribute("class", `${bodyClass} semantic-topology-node-body-diamond`);
    body.setAttribute("fill", color);
    body.setAttribute("stroke", color);
    group.appendChild(body);
    return;
  }

  if (node.kind === "concept") {
    const body = createSvgElement("rect");
    const size = node.radius * 1.8;
    body.setAttribute("x", String(-size / 2));
    body.setAttribute("y", String(-size / 2));
    body.setAttribute("width", String(size));
    body.setAttribute("height", String(size));
    body.setAttribute("rx", "3");
    body.setAttribute("class", `${bodyClass} semantic-topology-node-body-square`);
    body.setAttribute("fill", color);
    body.setAttribute("stroke", color);
    group.appendChild(body);
    return;
  }

  const body = createSvgElement("circle");
  body.setAttribute("r", String(node.radius));
  body.setAttribute("class", `${bodyClass} semantic-topology-node-body-circle`);
  body.setAttribute("fill", color);
  body.setAttribute("stroke", color);
  group.appendChild(body);
}

function appendMountedNode(
  layer: SVGGElement,
  node: SimNode,
  onSelectNode: (nodeId: string) => void,
): MountedNode {
  const color = statusColor(node.status);
  const group = createSvgElement("g");
  group.setAttribute(
    "class",
    `semantic-topology-node semantic-topology-node-status-${node.status}`,
  );
  group.setAttribute("data-node-id", node.id);
  group.setAttribute("tabindex", "0");
  group.setAttribute("focusable", "true");
  group.setAttribute("role", "button");
  group.setAttribute("aria-label", nodeAccessibleLabel(node));
  group.setAttribute("aria-pressed", "false");

  const activate = () => onSelectNode(node.id);
  group.addEventListener("click", activate);
  group.addEventListener("keydown", (event) => {
    if (event.key === "Enter" || event.key === " ") {
      event.preventDefault();
      activate();
    }
  });

  const glow = createSvgElement("circle");
  glow.setAttribute("class", "semantic-topology-node-glow");
  glow.setAttribute("r", String(node.radius * 2.15));
  glow.setAttribute("fill", color);
  group.appendChild(glow);

  const hoverHalo = createSvgElement("circle");
  hoverHalo.setAttribute("class", "semantic-topology-node-hover-halo");
  hoverHalo.setAttribute("r", String(node.radius * 2.7));
  hoverHalo.setAttribute("stroke", color);
  group.appendChild(hoverHalo);

  appendNodeShape(group, node, color);

  const inner = createSvgElement("circle");
  inner.setAttribute("class", "semantic-topology-node-inner");
  inner.setAttribute("r", String(Math.max(2, node.radius * 0.38)));
  inner.setAttribute("fill", color);
  group.appendChild(inner);

  if (node.duplicate.count > 1) {
    const badge = createSvgElement("text");
    badge.setAttribute("class", "semantic-topology-node-badge");
    badge.setAttribute("x", "0");
    badge.setAttribute("y", String(-node.radius - 7));
    badge.setAttribute("text-anchor", "middle");
    badge.textContent = `#${node.duplicate.ordinal}`;
    group.appendChild(badge);
  }

  const label = createSvgElement("text");
  label.setAttribute(
    "class",
    node.radius > 7 || node.duplicate.count > 1
      ? "semantic-topology-node-label is-visible"
      : "semantic-topology-node-label",
  );
  label.setAttribute("x", "0");
  label.setAttribute("y", String(node.radius + 13));
  label.setAttribute("text-anchor", "middle");
  label.textContent = nodeDisplayLabel(node);
  group.appendChild(label);

  const title = createSvgElement("title");
  title.textContent = `${node.label} [${shortId(node.id)}] - ${node.edge_count} edges`;
  group.appendChild(title);

  layer.appendChild(group);
  return { node, group };
}

function appendMountedEdge(layer: SVGGElement, edge: SimEdge): MountedEdge {
  const line = createSvgElement("line");
  line.setAttribute("class", `semantic-topology-edge ${edgeClass(edge.type)}`);
  line.setAttribute("stroke-width", edgeWidth(edge.weight).toFixed(2));
  line.setAttribute("data-edge-id", edge.id);
  layer.appendChild(line);

  return { edge, line };
}

function updateEdgeLine(mounted: MountedEdge): void {
  const { sourceNode, targetNode } = mounted.edge;
  const dx = targetNode.x - sourceNode.x;
  const dy = targetNode.y - sourceNode.y;
  const distance = Math.hypot(dx, dy) || 1;
  const ux = dx / distance;
  const uy = dy / distance;

  mounted.line.setAttribute("x1", (sourceNode.x + ux * sourceNode.radius).toFixed(2));
  mounted.line.setAttribute("y1", (sourceNode.y + uy * sourceNode.radius).toFixed(2));
  mounted.line.setAttribute("x2", (targetNode.x - ux * targetNode.radius).toFixed(2));
  mounted.line.setAttribute("y2", (targetNode.y - uy * targetNode.radius).toFixed(2));
}

function updateNodeGroup(mounted: MountedNode): void {
  mounted.group.setAttribute(
    "transform",
    `translate(${mounted.node.x.toFixed(2)} ${mounted.node.y.toFixed(2)})`,
  );
}

function applyLinkForce(edges: readonly SimEdge[], alpha: number, dt: number): void {
  for (const edge of edges) {
    const source = edge.sourceNode;
    const target = edge.targetNode;
    const dx = target.x - source.x;
    const dy = target.y - source.y;
    const distance = Math.hypot(dx, dy) || 1;
    const desired =
      56 + source.radius + target.radius + Math.min(34, source.edge_count + target.edge_count);
    const force = (distance - desired) * 0.01 * alpha * dt;
    const fx = (dx / distance) * force;
    const fy = (dy / distance) * force;

    source.vx += fx;
    source.vy += fy;
    target.vx -= fx;
    target.vy -= fy;
  }
}

function applyChargeAndCollision(nodes: readonly SimNode[], alpha: number, dt: number): void {
  for (let leftIndex = 0; leftIndex < nodes.length; leftIndex += 1) {
    for (let rightIndex = leftIndex + 1; rightIndex < nodes.length; rightIndex += 1) {
      const left = nodes[leftIndex]!;
      const right = nodes[rightIndex]!;
      let dx = right.x - left.x;
      let dy = right.y - left.y;
      let distance = Math.hypot(dx, dy);

      if (distance < 0.01) {
        dx = 0.01 * (right.index + 1);
        dy = 0.01 * (left.index + 1);
        distance = Math.hypot(dx, dy);
      }

      const ux = dx / distance;
      const uy = dy / distance;
      const charge = (145 * alpha * dt) / Math.max(65, distance);

      left.vx -= ux * charge;
      left.vy -= uy * charge;
      right.vx += ux * charge;
      right.vy += uy * charge;

      const collisionDistance = (left.radius + right.radius) * 1.2;

      if (distance < collisionDistance) {
        const overlap = (collisionDistance - distance) * 0.055 * alpha * dt;
        left.vx -= ux * overlap;
        left.vy -= uy * overlap;
        right.vx += ux * overlap;
        right.vy += uy * overlap;
      }
    }
  }
}

function applyCenterAndDrift(
  nodes: readonly SimNode[],
  alpha: number,
  dt: number,
  elapsed: number,
): void {
  for (const node of nodes) {
    const drift = elapsed > SETTLE_MS ? Math.sin(elapsed / 1300 + node.index * 1.7) * 0.018 : 0;

    node.vx += (CENTER_X - node.x) * 0.0022 * alpha * dt;
    node.vy += (CENTER_Y - node.y) * 0.0022 * alpha * dt;
    node.vx += Math.cos(node.index * 2.11) * drift;
    node.vy += Math.sin(node.index * 1.73) * drift;
  }
}

function integrateNodes(nodes: readonly SimNode[], dt: number): void {
  const margin = 28;

  for (const node of nodes) {
    node.vx *= 0.86;
    node.vy *= 0.86;
    node.x += node.vx * dt;
    node.y += node.vy * dt;

    if (node.x < margin) {
      node.x = margin;
      node.vx *= -0.25;
    } else if (node.x > VIEWBOX_WIDTH - margin) {
      node.x = VIEWBOX_WIDTH - margin;
      node.vx *= -0.25;
    }

    if (node.y < margin) {
      node.y = margin;
      node.vy *= -0.25;
    } else if (node.y > VIEWBOX_HEIGHT - margin) {
      node.y = VIEWBOX_HEIGHT - margin;
      node.vy *= -0.25;
    }
  }
}

function syncSelectedNodes(svg: SVGSVGElement, selectedId: string | null): void {
  const nodes = svg.querySelectorAll<SVGGElement>(".semantic-topology-node");

  for (const node of nodes) {
    const active = node.dataset.nodeId === selectedId;
    node.classList.toggle("selected", active);
    node.setAttribute("aria-pressed", active ? "true" : "false");
  }
}

function mountGraph(
  svg: SVGSVGElement,
  data: SemanticGraphResponse,
  duplicateInfo: ReadonlyMap<string, DuplicateInfo>,
  onSelectNode: (nodeId: string) => void,
): () => void {
  clearSvg(svg);
  svg.setAttribute("viewBox", `0 0 ${VIEWBOX_WIDTH} ${VIEWBOX_HEIGHT}`);
  svg.setAttribute("preserveAspectRatio", "xMidYMid meet");

  const edgeLayer = createSvgElement("g");
  edgeLayer.setAttribute("class", "semantic-topology-edge-layer");
  const nodeLayer = createSvgElement("g");
  nodeLayer.setAttribute("class", "semantic-topology-node-layer");
  svg.appendChild(edgeLayer);
  svg.appendChild(nodeLayer);

  const nodes = createSimNodes(data, duplicateInfo);
  const edges = createSimEdges(data, nodes);
  const mountedEdges = edges.map((edge) => appendMountedEdge(edgeLayer, edge));
  const mountedNodes = nodes.map((node) => appendMountedNode(nodeLayer, node, onSelectNode));
  const startedAt = performance.now();
  let alpha = 1;
  let previousTick = startedAt;
  let frame = 0;
  let stopped = false;

  const update = () => {
    for (const mounted of mountedEdges) {
      updateEdgeLine(mounted);
    }
    for (const mounted of mountedNodes) {
      updateNodeGroup(mounted);
    }
  };

  const tick = (now: number) => {
    if (stopped) {
      return;
    }

    const elapsed = now - startedAt;
    const dt = Math.min(2, Math.max(0.5, (now - previousTick) / 16.67));
    const targetAlpha = elapsed > SETTLE_MS ? 0.05 : 0.012;
    previousTick = now;
    alpha += (targetAlpha - alpha) * 0.024 * dt;

    applyLinkForce(edges, alpha, dt);
    applyChargeAndCollision(nodes, alpha, dt);
    applyCenterAndDrift(nodes, alpha, dt, elapsed);
    integrateNodes(nodes, dt);
    update();

    frame = window.requestAnimationFrame(tick);
  };

  update();
  frame = window.requestAnimationFrame(tick);

  return () => {
    stopped = true;
    window.cancelAnimationFrame(frame);
    clearSvg(svg);
  };
}

export function SemanticTopology({ graph, selectedId, onSelectNode }: SemanticTopologyProps) {
  const svgRef = useRef<SVGSVGElement | null>(null);
  const onSelectRef = useRef(onSelectNode);
  const duplicateInfo = useMemo(() => duplicateInfoForNodes(graph.nodes), [graph.nodes]);
  const duplicateClusters = useMemo(() => duplicateClustersForNodes(graph.nodes), [graph.nodes]);

  useEffect(() => {
    onSelectRef.current = onSelectNode;
  }, [onSelectNode]);

  useEffect(() => {
    if (svgRef.current === null || graph.nodes.length === 0) {
      return undefined;
    }

    return mountGraph(svgRef.current, graph, duplicateInfo, (nodeId) =>
      onSelectRef.current(nodeId),
    );
  }, [duplicateInfo, graph]);

  useEffect(() => {
    if (svgRef.current !== null) {
      syncSelectedNodes(svgRef.current, selectedId);
    }
  }, [graph, selectedId]);

  if (graph.nodes.length === 0) {
    return (
      <div className="semantic-topology">
        <div className="semantic-topology-canvas">
          <div className="semantic-topology-empty">
            <div className="semantic-topology-empty-mark">0</div>
            <div className="semantic-topology-empty-title">semantic graph empty</div>
            <div className="semantic-topology-empty-sub">
              no entities or relations recorded for this audience yet
            </div>
          </div>
        </div>
      </div>
    );
  }

  return (
    <div className="semantic-topology">
      <div className="semantic-topology-canvas">
        <div className="semantic-topology-overlay">
          <Tag>
            {graph.rendered.nodes.toLocaleString()} nodes / {graph.rendered.edges.toLocaleString()}{" "}
            edges
          </Tag>
          <Tag kind="info">
            showing {graph.rendered.nodes.toLocaleString()} of {graph.total_nodes.toLocaleString()}
          </Tag>
          {selectedId === null ? null : (
            <Tag kind="acc">
              selected <IdRef id={selectedId} type="semantic_node" label={shortId(selectedId)} />
            </Tag>
          )}
        </div>
        <svg
          ref={svgRef}
          className="semantic-topology-svg"
          data-testid="semantic-topology-svg"
          role="img"
          aria-label="semantic topology graph"
        />
      </div>
      {duplicateClusters.length > 0 ? (
        <div className="semantic-topology-clusters" aria-label="duplicate label clusters">
          <span className="semantic-topology-clusters-label">duplicate labels</span>
          {duplicateClusters.map((cluster) => (
            <Tag key={cluster.label} kind="info">
              {cluster.label} x{cluster.count}
            </Tag>
          ))}
        </div>
      ) : null}
    </div>
  );
}
