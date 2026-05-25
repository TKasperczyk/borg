import { useMemo, useState } from "react";

import { Tag } from "../../components/Tag";
import { GRAPH_EDGES, GRAPH_NODES, SELECTED_NODE, type GraphNode } from "./mock-data";

const VB_W = 1000;
const VB_H = 540;

function radius(node: GraphNode): number {
  return 18 + Math.round(node.hot * 8);
}

function NodeLabel({ x, y, text, selected }: { x: number; y: number; text: string; selected: boolean }) {
  const padX = 7;
  const width = text.length * 8 + padX * 2;
  const height = 17;
  return (
    <g pointerEvents="none">
      <rect
        x={x - width / 2}
        y={y - height / 2}
        width={width}
        height={height}
        fill="var(--bg-0)"
        stroke={selected ? "var(--acc-dim)" : "var(--line-soft)"}
        strokeWidth="0.8"
      />
      <text x={x} y={y + 4} fill={selected ? "var(--acc)" : "var(--text-dim)"} fontSize="13" textAnchor="middle" style={{ fontFamily: "var(--mono)" }}>
        {text}
      </text>
    </g>
  );
}

function EdgeLabel({ x, y, text, color, angle }: { x: number; y: number; text: string; color: string; angle: number }) {
  let rotate = angle;
  if (rotate > 90) {
    rotate -= 180;
  }
  if (rotate < -90) {
    rotate += 180;
  }
  const width = text.length * 7 + 12;
  return (
    <g transform={`translate(${x}, ${y}) rotate(${rotate})`} pointerEvents="none">
      <rect x={-width / 2} y={-7} width={width} height={14} fill="var(--bg-0)" />
      <text x={0} y={4} fill={color} fontSize="11" textAnchor="middle" style={{ fontFamily: "var(--mono)" }}>
        {text}
      </text>
    </g>
  );
}

function nodeShape(node: GraphNode, r: number, fill: string, stroke: string, strokeWidth: number) {
  if (node.kind === "proposition") {
    return (
      <polygon
        points={`${node.x},${node.y - r} ${node.x + r},${node.y} ${node.x},${node.y + r} ${node.x - r},${node.y}`}
        fill={fill}
        stroke={stroke}
        strokeWidth={strokeWidth}
      />
    );
  }
  if (node.kind === "concept") {
    const size = r * 1.8;
    return (
      <rect
        x={node.x - size / 2}
        y={node.y - size / 2}
        width={size}
        height={size}
        rx="3"
        fill={fill}
        stroke={stroke}
        strokeWidth={strokeWidth}
      />
    );
  }
  return <circle cx={node.x} cy={node.y} r={r} fill={fill} stroke={stroke} strokeWidth={strokeWidth} />;
}

export function GraphScreen() {
  const [selectedId, setSelectedId] = useState(SELECTED_NODE);
  const [hoverEdge, setHoverEdge] = useState<number | null>(null);
  const [walkDepth, setWalkDepth] = useState(2);
  const nodeById = useMemo(() => new Map(GRAPH_NODES.map((node) => [node.id, node])), []);
  const selectedNode = nodeById.get(selectedId);
  const reachable = useMemo(() => {
    const adjacency = new Map<string, string[]>();
    for (const edge of GRAPH_EDGES) {
      adjacency.set(edge.from, [...(adjacency.get(edge.from) ?? []), edge.to]);
      adjacency.set(edge.to, [...(adjacency.get(edge.to) ?? []), edge.from]);
    }
    const seen = new Set([selectedId]);
    let frontier = [selectedId];
    for (let index = 0; index < walkDepth; index += 1) {
      const next: string[] = [];
      for (const id of frontier) {
        for (const adjacent of adjacency.get(id) ?? []) {
          if (!seen.has(adjacent)) {
            seen.add(adjacent);
            next.push(adjacent);
          }
        }
      }
      frontier = next;
    }
    return seen;
  }, [selectedId, walkDepth]);
  const incident = GRAPH_EDGES.filter((edge) => edge.from === selectedId || edge.to === selectedId);

  return (
    <div className="graph-screen">
      <div className="graph-canvas">
        <svg viewBox={`0 0 ${VB_W} ${VB_H}`} preserveAspectRatio="xMidYMid meet">
          <defs>
            <pattern id="grid" width="40" height="40" patternUnits="userSpaceOnUse">
              <path d="M 40 0 L 0 0 0 40" fill="none" stroke="var(--line-soft)" strokeWidth="0.4" />
            </pattern>
            <pattern id="grid-major" width="200" height="200" patternUnits="userSpaceOnUse">
              <path d="M 200 0 L 0 0 0 200" fill="none" stroke="var(--line)" strokeWidth="0.5" />
            </pattern>
            <marker id="arrow-dim" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="8" markerHeight="8" orient="auto-start-reverse">
              <path d="M 0 0 L 10 5 L 0 10 z" fill="var(--text-faint)" />
            </marker>
            <marker id="arrow-acc" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="8" markerHeight="8" orient="auto-start-reverse">
              <path d="M 0 0 L 10 5 L 0 10 z" fill="var(--acc)" />
            </marker>
            <marker id="arrow-bad" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="8" markerHeight="8" orient="auto-start-reverse">
              <path d="M 0 0 L 10 5 L 0 10 z" fill="var(--bad)" />
            </marker>
            <radialGradient id="halo" cx="50%" cy="50%" r="50%">
              <stop offset="0%" stopColor="oklch(0.84 0.155 142 / 0.35)" />
              <stop offset="60%" stopColor="oklch(0.84 0.155 142 / 0.08)" />
              <stop offset="100%" stopColor="oklch(0.84 0.155 142 / 0)" />
            </radialGradient>
          </defs>
          <rect width={VB_W} height={VB_H} fill="url(#grid)" />
          <rect width={VB_W} height={VB_H} fill="url(#grid-major)" />
          {selectedNode === undefined ? null : (
            <circle cx={selectedNode.x} cy={selectedNode.y} r={radius(selectedNode) * 3.2} fill="url(#halo)" pointerEvents="none" />
          )}
          {GRAPH_EDGES.map((edge, index) => {
            const from = nodeById.get(edge.from);
            const to = nodeById.get(edge.to);
            if (from === undefined || to === undefined) {
              return null;
            }
            const dx = to.x - from.x;
            const dy = to.y - from.y;
            const dist = Math.sqrt(dx * dx + dy * dy) || 1;
            const ux = dx / dist;
            const uy = dy / dist;
            const endpoint = {
              x1: from.x + ux * radius(from),
              y1: from.y + uy * radius(from),
              x2: to.x - ux * (radius(to) + 6),
              y2: to.y - uy * (radius(to) + 6),
              mx: (from.x + to.x) / 2,
              my: (from.y + to.y) / 2,
              angle: Math.atan2(dy, dx) * (180 / Math.PI)
            };
            const isIncident = edge.from === selectedId || edge.to === selectedId;
            const isReachable = reachable.has(edge.from) && reachable.has(edge.to);
            const contradicted = edge.state === "contradicted";
            const color = isIncident ? (contradicted ? "var(--bad)" : "var(--acc)") : isReachable ? "var(--text-mute)" : "var(--line)";
            const marker = isIncident ? (contradicted ? "url(#arrow-bad)" : "url(#arrow-acc)") : "url(#arrow-dim)";
            return (
              <g key={`${edge.from}-${edge.to}-${edge.rel}`} style={{ opacity: isIncident || isReachable ? 1 : 0.4 }} onMouseEnter={() => setHoverEdge(index)} onMouseLeave={() => setHoverEdge(null)}>
                <line
                  x1={endpoint.x1}
                  y1={endpoint.y1}
                  x2={endpoint.x2}
                  y2={endpoint.y2}
                  stroke={color}
                  strokeWidth={isIncident ? 1.8 : 1.2}
                  strokeDasharray={contradicted ? "5 4" : "none"}
                  markerEnd={marker}
                  pointerEvents="stroke"
                  style={{ cursor: "pointer" }}
                />
                {isIncident || hoverEdge === index ? (
                  <EdgeLabel x={endpoint.mx} y={endpoint.my} angle={endpoint.angle} text={contradicted ? `${edge.rel} x` : edge.rel} color={color} />
                ) : null}
              </g>
            );
          })}
          {GRAPH_NODES.map((node) => {
            const selected = node.id === selectedId;
            const isReachable = reachable.has(node.id);
            const r = radius(node);
            const stroke = selected ? "var(--acc)" : isReachable ? (node.kind === "entity" ? "var(--info)" : node.kind === "proposition" ? "var(--bad)" : "var(--text)") : "var(--text-mute)";
            return (
              <g key={node.id} style={{ cursor: "pointer", opacity: isReachable ? 1 : 0.45 }} onClick={() => setSelectedId(node.id)}>
                {node.hot > 0.55 ? <circle cx={node.x} cy={node.y} r={r + 4} fill="none" stroke={stroke} strokeWidth="0.5" opacity="0.4" /> : null}
                {nodeShape(node, r, selected ? "oklch(0.84 0.155 142 / 0.12)" : "var(--bg-0)", stroke, selected ? 2.2 : 1.5)}
                {node.kind === "entity" ? (
                  <text x={node.x} y={node.y + 5} fill={selected ? "var(--acc)" : stroke} fontSize="16" textAnchor="middle" style={{ fontFamily: "var(--mono)", fontWeight: 600, pointerEvents: "none" }}>
                    {node.label.slice(0, 1)}
                  </text>
                ) : null}
                <NodeLabel x={node.x} y={node.y + r + 18} text={node.label} selected={selected} />
              </g>
            );
          })}
        </svg>
        <div className="graph-overlay">
          <span><span className="dim">nodes</span> <span className="tab-num">{GRAPH_NODES.length}</span></span>
          <span className="sep">|</span>
          <span><span className="dim">edges</span> <span className="tab-num">{GRAPH_EDGES.length}</span></span>
          <span className="sep">|</span>
          <span><span className="dim">audience</span> <span className="acc">mock</span></span>
          <span className="sep">|</span>
          <span><span className="dim">walk</span></span>
          <span style={{ display: "flex", gap: 3 }}>
            {[1, 2, 3].map((depth) => (
              <span key={depth} onClick={() => setWalkDepth(depth)} className={`pill ${walkDepth === depth ? "on" : ""}`}>
                {depth}
              </span>
            ))}
          </span>
        </div>
      </div>
      <div className="graph-side">
        <div className="sect">
          <div className="h">selected</div>
          <div style={{ fontSize: 14, color: "var(--text)", marginBottom: 6, fontFamily: "var(--mono)" }}>{selectedNode?.label}</div>
          <div style={{ display: "flex", gap: 6, flexWrap: "wrap", marginBottom: 8 }}>
            <Tag kind={selectedNode?.kind === "entity" ? "info" : selectedNode?.kind === "proposition" ? "bad" : ""} dot>
              {selectedNode?.kind ?? "node"}
            </Tag>
            <Tag kind="acc" dot>active</Tag>
            <Tag>heat {(selectedNode?.hot ?? 0).toFixed(2)}</Tag>
          </div>
          <div className="dim" style={{ fontSize: 10.5 }}>[{selectedNode?.id}]</div>
        </div>
        <div className="sect">
          <div className="h">incident edges <span style={{ color: "var(--text-faint)", marginLeft: 6 }}>{incident.length}</span></div>
          {incident.map((edge) => {
            const other = nodeById.get(edge.from === selectedId ? edge.to : edge.from);
            const outgoing = edge.from === selectedId;
            return (
              <div key={`${edge.from}-${edge.to}-${edge.rel}`} style={{ padding: "6px 0", borderBottom: "1px solid var(--line-soft)", fontSize: 11.5, display: "flex", gap: 6, alignItems: "baseline" }}>
                <span className="dim" style={{ fontFamily: "var(--mono)", minWidth: 100 }}>{outgoing ? "--" : "<-"}{edge.rel}{outgoing ? "->" : "--"}</span>
                <span style={{ color: edge.state === "contradicted" ? "var(--bad)" : "var(--text-dim)", cursor: "pointer", flex: 1 }} onClick={() => other && setSelectedId(other.id)}>
                  {other?.label}
                </span>
                {edge.state === "contradicted" ? <Tag kind="bad">x</Tag> : null}
              </div>
            );
          })}
        </div>
        <div className="sect">
          <div className="h">walk</div>
          <div style={{ display: "flex", flexDirection: "column", gap: 6, fontSize: 11 }}>
            <div className="dim">depth {walkDepth} · {reachable.size} nodes reachable</div>
            <button className="btn sm" disabled title="mocked for v1">open in retrieval trace</button>
            <button className="btn sm ghost" disabled title="mocked for v1">export subgraph</button>
          </div>
        </div>
        <div className="sect" style={{ borderBottom: "none" }}>
          <div className="h">extraction trail</div>
          <pre style={{ margin: 0, fontSize: 10.5, color: "var(--text-mute)", whiteSpace: "pre-wrap", fontFamily: "var(--mono)" }}>
{`mocked_for:     P2 demo
real_wiring:     deferred
source_eps:      7
support_edges:   4
contradict:      1`}
          </pre>
        </div>
      </div>
    </div>
  );
}
