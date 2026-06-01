import { useMemo, type CSSProperties, type ReactNode } from "react";

import type { TurnPhaseName, TurnTerminalOutcome } from "../../api/types";
import type { PhaseState } from "../../hooks/use-turn-stream";
import { ParticleField, type ParticleTarget } from "./ParticleField";

// Synaptic flow chart -- each phase rendered as a small "ganglion" with a
// halo, dendrite stubs and a state-coloured nucleus. Edges are smooth bezier
// traces with arrowheads; active edges march and emit pulse particles.
//
// Layout is a zigzag/snake: tier 1 flows left -> right, tier 2 right -> left,
// tier 3 left -> right. This eliminates the old wrap-around path -- cross-tier
// descents become short S-curves on the right (gen -> retrieval) and on the
// left (delib -> final via the S1/S2 fork). Tier labels are editorial chips in
// the top-left of each band so they never collide with the first node of a row.
// A gravitational-lens particle cloud sits behind the SVG and bends toward the
// active node.

export type FlowChartProps = {
  phases: readonly PhaseState[];
  activeTurnId: string | null;
  tokenTextByPhase: Map<string, string>;
  detailByPhase: Map<string, string[]>;
  terminalOutcome: TurnTerminalOutcome | null;
  delibPath: "system_1" | "system_2" | null;
  finalAttempt: number;
  particleEnabled?: boolean;
  particleDensity?: number;
};

// Layout (viewBox 1200 x 660)
const VIEW_W = 1200;
const VIEW_H = 660;

const NODE_R = 24;
const GATE_R = 26;
const ENDPOINT_H = 34;

const TIER_Y = { 1: 125, 2: 320, 3: 510 } as const;
const BAND = { 1: [30, 225], 2: [225, 420], 3: [420, 615] } as const;

type TierId = 1 | 2 | 3;
const TIERS: ReadonlyArray<{ id: TierId; label: string; flow: string }> = [
  { id: 1, label: "perception", flow: "left -> right" },
  { id: 2, label: "context", flow: "right -> left" },
  { id: 3, label: "synthesis", flow: "left -> right" },
];

type EndpointNode = {
  id: string;
  kind: "endpoint";
  x: number;
  y: number;
  w: number;
  label: string;
  side: string;
};

type PhaseNode = {
  id: TurnPhaseName;
  kind: "phase" | "gate";
  x: number;
  y: number;
  label: string;
};

type LayoutNode = EndpointNode | PhaseNode;

const INPUT_NODE: EndpointNode = {
  id: "input",
  kind: "endpoint",
  x: 78,
  y: TIER_Y[1],
  w: 72,
  label: "input",
  side: "ENTRY",
};
const TERMINAL_NODE: EndpointNode = {
  id: "terminal",
  kind: "endpoint",
  x: 950,
  y: TIER_Y[3],
  w: 150,
  label: "terminal",
  side: "EXIT",
};

// Phase IDs are the canonical names from use-turn-stream's PHASES list. The
// design prototype shortened a few labels (e.g. "gen?"), but the ids must stay
// stable -- they are the live-data binding keys.
const PHASE_IDS = [
  "ingest",
  "audience",
  "perception",
  "frame",
  "extract",
  "closure_loop",
  "generation_gate",
  "retrieval",
  "ledger",
  "shared",
  "delib",
  "final",
  "guards",
  "persist",
  "reflect",
] as const satisfies ReadonlyArray<TurnPhaseName>;

const PHASES_LAYOUT: readonly PhaseNode[] = [
  // tier 1 -- left -> right
  { id: "ingest", kind: "phase", x: 200, y: TIER_Y[1], label: "ingest" },
  { id: "audience", kind: "phase", x: 330, y: TIER_Y[1], label: "audience" },
  { id: "perception", kind: "phase", x: 460, y: TIER_Y[1], label: "perception" },
  { id: "frame", kind: "phase", x: 595, y: TIER_Y[1], label: "frame" },
  { id: "extract", kind: "phase", x: 730, y: TIER_Y[1], label: "extract" },
  { id: "closure_loop", kind: "gate", x: 865, y: TIER_Y[1], label: "closure?" },
  { id: "generation_gate", kind: "gate", x: 1000, y: TIER_Y[1], label: "gen?" },
  // tier 2 -- right -> left
  { id: "retrieval", kind: "phase", x: 1000, y: TIER_Y[2], label: "retrieval" },
  { id: "ledger", kind: "phase", x: 800, y: TIER_Y[2], label: "ev. ledger" },
  { id: "shared", kind: "phase", x: 600, y: TIER_Y[2], label: "shared" },
  { id: "delib", kind: "phase", x: 400, y: TIER_Y[2], label: "deliberation" },
  // tier 3 -- left -> right
  { id: "final", kind: "phase", x: 200, y: TIER_Y[3], label: "finalizer" },
  { id: "guards", kind: "gate", x: 370, y: TIER_Y[3], label: "guards?" },
  { id: "persist", kind: "phase", x: 540, y: TIER_Y[3], label: "persist" },
  { id: "reflect", kind: "phase", x: 720, y: TIER_Y[3], label: "reflection" },
];

const NODES: Record<string, LayoutNode> = (() => {
  const map: Record<string, LayoutNode> = {
    input: INPUT_NODE,
    terminal: TERMINAL_NODE,
  };
  for (const p of PHASES_LAYOUT) map[p.id] = p;
  return map;
})();
const PHASE_LABELS: Record<string, string> = Object.fromEntries(
  PHASES_LAYOUT.map((phase) => [phase.id, phase.label]),
);

// Same edge topology as upstream -- the routing changes, not the graph. The
// delib -> final transition is the S1/S2 fork, drawn separately.
const SPINE_EDGES: ReadonlyArray<readonly [string, string]> = [
  ["input", "ingest"],
  ["ingest", "audience"],
  ["audience", "perception"],
  ["perception", "frame"],
  ["frame", "extract"],
  ["extract", "closure_loop"],
  ["closure_loop", "generation_gate"],
  ["generation_gate", "retrieval"], // tier 1 -> tier 2 (right-side descent)
  ["retrieval", "ledger"],
  ["ledger", "shared"],
  ["shared", "delib"],
  ["final", "guards"],
  ["guards", "persist"],
  ["persist", "reflect"],
  ["reflect", "terminal"],
];

// -----------------------------------------------------------------------------
// Geometry helpers
// -----------------------------------------------------------------------------

function nodeRadius(n: LayoutNode): number {
  if (n.kind === "endpoint") return n.w / 2;
  if (n.kind === "gate") return GATE_R;
  return NODE_R;
}
function rightAnchor(n: LayoutNode): { x: number; y: number } {
  if (n.kind === "endpoint") return { x: n.x + n.w / 2, y: n.y };
  return { x: n.x + nodeRadius(n) + 2, y: n.y };
}
function leftAnchor(n: LayoutNode): { x: number; y: number } {
  if (n.kind === "endpoint") return { x: n.x - n.w / 2, y: n.y };
  return { x: n.x - nodeRadius(n) - 2, y: n.y };
}
function bottomAnchor(n: LayoutNode): { x: number; y: number } {
  if (n.kind === "endpoint") return { x: n.x, y: n.y + ENDPOINT_H / 2 };
  return { x: n.x, y: n.y + nodeRadius(n) + 2 };
}
function topAnchor(n: LayoutNode): { x: number; y: number } {
  if (n.kind === "endpoint") return { x: n.x, y: n.y - ENDPOINT_H / 2 };
  return { x: n.x, y: n.y - nodeRadius(n) - 2 };
}

function spinePath(fromId: string, toId: string): string {
  const from = NODES[fromId]!;
  const to = NODES[toId]!;
  const dy = to.y - from.y;
  if (Math.abs(dy) < 5) {
    // intra-tier -- horizontal bezier, honouring the row's flow direction
    const rtl = to.x < from.x;
    const a = rtl ? leftAnchor(from) : rightAnchor(from);
    const b = rtl ? rightAnchor(to) : leftAnchor(to);
    const span = Math.abs(b.x - a.x);
    const cx = span * 0.42;
    if (rtl) {
      return `M ${a.x} ${a.y} C ${a.x - cx} ${a.y}, ${b.x + cx} ${b.y}, ${b.x} ${b.y}`;
    }
    return `M ${a.x} ${a.y} C ${a.x + cx} ${a.y}, ${b.x - cx} ${b.y}, ${b.x} ${b.y}`;
  }
  // cross-tier descent -- vertical with a subtle outward bow
  const a = bottomAnchor(from);
  const b = topAnchor(to);
  const mid = (a.y + b.y) / 2;
  const bow = from.x > VIEW_W / 2 ? 26 : -26;
  return `M ${a.x} ${a.y} C ${a.x + bow} ${mid}, ${b.x + bow} ${mid}, ${b.x} ${b.y}`;
}

// S1/S2 fork: both originate at delib's left anchor and end at final's top.
// Two clean parallel cubic beziers that mirror across the direct diagonal --
// S1 a tighter inner curve ("fast path"), S2 a wider outer sweep ("plan").
function forkPath(which: "s1" | "s2"): string {
  const a = leftAnchor(NODES.delib!);
  const b = topAnchor(NODES.final!);
  const dx = b.x - a.x;
  const dy = b.y - a.y;
  const len = Math.hypot(dx, dy) || 1;
  // Unit perpendicular to (a->b), rotated 90 CCW.
  const px = -dy / len;
  const py = dx / len;

  if (which === "s1") {
    const off = -34;
    const cp1x = a.x + dx * 0.33 + px * off;
    const cp1y = a.y + dy * 0.33 + py * off;
    const cp2x = a.x + dx * 0.67 + px * off;
    const cp2y = a.y + dy * 0.67 + py * off;
    return `M ${a.x} ${a.y} C ${cp1x.toFixed(1)} ${cp1y.toFixed(1)}, ${cp2x.toFixed(1)} ${cp2y.toFixed(1)}, ${b.x} ${b.y}`;
  }
  const off = 58;
  const cp1x = a.x + dx * 0.3 + px * off;
  const cp1y = a.y + dy * 0.3 + py * off;
  const cp2x = a.x + dx * 0.7 + px * off;
  const cp2y = a.y + dy * 0.7 + py * off;
  return `M ${a.x} ${a.y} C ${cp1x.toFixed(1)} ${cp1y.toFixed(1)}, ${cp2x.toFixed(1)} ${cp2y.toFixed(1)}, ${b.x} ${b.y}`;
}

// Regen arc: guards bottom -> final bottom, looping UNDER tier 3.
function regenPath(): string {
  const g = bottomAnchor(NODES.guards!);
  const f = bottomAnchor(NODES.final!);
  const peakY = 612;
  return `M ${g.x} ${g.y} C ${g.x} ${peakY}, ${f.x} ${peakY}, ${f.x} ${f.y}`;
}

// Thorn drop -- tier 1 thorns go UP into the top margin; the tier 3 thorn is a
// sidecar out to the lower-right (the clear zone between guards and persist).
function thornDrop(nodeId: string, dir: "up" | "side"): string {
  const n = NODES[nodeId]!;
  if (dir === "up") {
    const from = topAnchor(n);
    return `M ${from.x} ${from.y} L ${from.x} ${from.y - 32}`;
  }
  const ax = n.x + Math.cos(Math.PI / 6) * GATE_R;
  const ay = n.y + Math.sin(Math.PI / 6) * GATE_R;
  const px = n.x + 40;
  const py = n.y + 45;
  return `M ${ax} ${ay} Q ${ax + 6} ${ay + 14}, ${px} ${py}`;
}

// -----------------------------------------------------------------------------
// Phase-state plumbing (unchanged contract with use-turn-stream)
// -----------------------------------------------------------------------------

type PhaseStatus = PhaseState["status"];

type PhasesRecord = Record<
  string,
  { name?: string; status: PhaseStatus; sub?: string; durationMs?: number }
>;

function buildPhaseRecord(phases: readonly PhaseState[]): PhasesRecord {
  const out: PhasesRecord = {};
  for (const phase of phases) {
    out[phase.id] = {
      name: phase.name,
      status: phase.status,
      sub: phase.sub,
      durationMs: phase.durationMs,
    };
  }
  return out;
}

function tokenKey(turnId: string, phaseId: string): string {
  return `${turnId}:${phaseId}`;
}

function deriveActiveStreamPhase(
  phases: PhasesRecord,
  activeTurnId: string | null,
  tokenTextByPhase: ReadonlyMap<string, string>,
  detailByPhase: ReadonlyMap<string, readonly string[]>,
): TurnPhaseName | null {
  if (phases.final?.status === "running") return "final";
  if (phases.delib?.status === "running") return "delib";
  for (const phase of PHASE_IDS) {
    if (phases[phase]?.status === "running") return phase;
  }
  if (activeTurnId !== null) {
    for (let index = PHASE_IDS.length - 1; index >= 0; index -= 1) {
      const phase = PHASE_IDS[index]!;
      const key = tokenKey(activeTurnId, phase);
      if (
        (tokenTextByPhase.get(key) ?? "").length > 0 ||
        (detailByPhase.get(key)?.length ?? 0) > 0
      ) {
        return phase;
      }
    }
  }
  return null;
}

// -----------------------------------------------------------------------------
// Node + endpoint components
// -----------------------------------------------------------------------------

function Ganglion({
  node,
  status,
  sub,
  duration,
}: {
  node: PhaseNode;
  status: PhaseStatus;
  sub?: string;
  duration?: number;
}) {
  const { x, y, label, kind } = node;
  const isGate = kind === "gate";
  const r = nodeRadius(node);

  const hex = useMemo(() => {
    const pts: string[] = [];
    for (let i = 0; i < 6; i++) {
      const angle = (Math.PI / 3) * i - Math.PI / 2;
      pts.push(`${(Math.cos(angle) * r).toFixed(2)},${(Math.sin(angle) * r).toFixed(2)}`);
    }
    return pts.join(" ");
  }, [r]);

  return (
    <g
      className={`fc-node fc-node-${status}`}
      data-status={status}
      data-testid={`phase-${node.id}`}
      transform={`translate(${x} ${y})`}
    >
      <circle className="fc-node-halo" r={r + 8} />
      <circle className="fc-pulse-ring" r={r} />
      <circle className="fc-pulse-ring b" r={r} />
      <circle className="fc-pulse-ring c" r={r} />

      {!isGate ? (
        <g className="fc-dendrite-grp">
          <line className="fc-dendrite" x1={-r - 1} y1={0} x2={-r - 7} y2={-3} />
          <line className="fc-dendrite" x1={-r - 1} y1={0} x2={-r - 7} y2={3} />
          <line className="fc-dendrite" x1={r + 1} y1={0} x2={r + 7} y2={-3} />
          <line className="fc-dendrite" x1={r + 1} y1={0} x2={r + 7} y2={3} />
        </g>
      ) : null}

      {isGate ? (
        <polygon className="fc-node-body" points={hex} />
      ) : (
        <circle className="fc-node-body" r={r} />
      )}
      <circle className="fc-node-core" r={4.5} />

      <text className="fc-node-label" y={r + 16}>
        {label}
      </text>
      {status === "done" && duration !== undefined ? (
        <text className="fc-node-time" y={r + 28}>
          {Math.round(duration)}ms
        </text>
      ) : null}
      {(status === "running" || status === "fail") && sub ? (
        <text className="fc-node-sub" y={r + 28}>
          {sub.length > 22 ? sub.slice(0, 21) + "…" : sub}
        </text>
      ) : null}
    </g>
  );
}

function Endpoint({
  node,
  outcome,
}: {
  node: EndpointNode;
  outcome: TurnTerminalOutcome | null;
}) {
  const { x, y, w, label, id, side } = node;
  const h = ENDPOINT_H;
  const r = h / 2;
  const isTerminal = id === "terminal";
  const display = isTerminal
    ? outcome === null
      ? "waiting"
      : outcome.replace(/_/g, " ")
    : label;
  return (
    <g
      className={`fc-endpoint ${isTerminal ? "terminal" : "input"}`}
      data-outcome={outcome ?? ""}
      transform={`translate(${x - w / 2} ${y - h / 2})`}
    >
      <rect className="fc-endpoint-body" x={0} y={0} width={w} height={h} rx={r} ry={r} />
      <text className="fc-endpoint-label" x={w / 2} y={h / 2 + 3.5}>
        {display}
      </text>
      <text className="fc-endpoint-side" x={w / 2} y={-7}>
        {side}
      </text>
    </g>
  );
}

// Thorn pills are content-sized so any label fits inside the box. The label is
// uppercase + letter-spaced 8.5px mono, so per-char advance is ~6.1px. The icon
// occupies a fixed left zone; the label is left-anchored just past it.
const THORN_CHAR_W = 6.1;
const THORN_ICON_X = 10;
const THORN_LABEL_X = 18;
const THORN_PAD_R = 9;
function thornPillW(label: string): number {
  return Math.round(THORN_LABEL_X + label.length * THORN_CHAR_W + THORN_PAD_R);
}

function ThornUp({
  nodeId,
  label,
  active,
}: {
  nodeId: string;
  label: string;
  active: boolean;
}) {
  const n = NODES[nodeId]!;
  const top = topAnchor(n);
  const pillY = top.y - 50;
  const pillW = thornPillW(label);
  const pillH = 20;
  return (
    <g className={`fc-thorn ${active ? "active" : ""}`}>
      <path d={thornDrop(nodeId, "up")} className={`fc-edge branch ${active ? "fire" : ""}`} />
      <g transform={`translate(${n.x - pillW / 2} ${pillY - pillH / 2})`}>
        <rect className="fc-thorn-shape" x={0} y={0} width={pillW} height={pillH} rx={2} />
        <text className="fc-thorn-icon" x={THORN_ICON_X} y={pillH / 2 + 3}>
          {"↯"}
        </text>
        <text className="fc-thorn-label" x={THORN_LABEL_X} y={pillH / 2 + 3}>
          {label}
        </text>
      </g>
    </g>
  );
}

function ThornSidecar({
  nodeId,
  label,
  active,
}: {
  nodeId: string;
  label: string;
  active: boolean;
}) {
  const n = NODES[nodeId]!;
  const pillW = thornPillW(label);
  const pillH = 19;
  const pillX = n.x + 40;
  const pillY = n.y + 45 - pillH / 2;
  return (
    <g className={`fc-thorn ${active ? "active" : ""}`}>
      <path d={thornDrop(nodeId, "side")} className={`fc-edge branch ${active ? "fire" : ""}`} />
      <g transform={`translate(${pillX} ${pillY})`}>
        <rect className="fc-thorn-shape" x={0} y={0} width={pillW} height={pillH} rx={2} />
        <text className="fc-thorn-icon" x={THORN_ICON_X} y={pillH / 2 + 3}>
          {"↯"}
        </text>
        <text className="fc-thorn-label" x={THORN_LABEL_X} y={pillH / 2 + 3}>
          {label}
        </text>
      </g>
    </g>
  );
}

function TierHeader({
  tier,
  band,
}: {
  tier: { id: TierId; label: string; flow: string };
  band: readonly [number, number];
}) {
  const x = 18;
  const y = band[0] + 18;
  return (
    <g transform={`translate(${x} ${y})`}>
      <rect className="fc-tier-chip-bg" x={0} y={-13} width={28} height={18} rx={2} />
      <text className="fc-tier-chip-num" x={14} y={0} textAnchor="middle">
        {String(tier.id).padStart(2, "0")}
      </text>
      <text className="fc-tier-chip-label" x={38} y={0}>
        {tier.label}
      </text>
      <text className="fc-tier-chip-meta" x={38} y={14}>
        {tier.flow}
      </text>
    </g>
  );
}

function Edge({
  d,
  state,
  descent,
}: {
  d: string;
  state: "queue" | "active" | "done";
  descent?: boolean;
}) {
  const cls = `fc-edge ${descent ? "descent" : ""} ${state === "active" ? "active" : state === "done" ? "done" : "dim"}`;
  const pulseStyle = { offsetPath: `path("${d}")` } as CSSProperties;
  const marker =
    state === "active"
      ? "url(#arrow-active)"
      : state === "done"
        ? "url(#arrow-done)"
        : "url(#arrow-queue)";
  return (
    <g className={`fc-edge-wrap ${state === "active" ? "active" : ""}`}>
      <path d={d} className={cls} markerEnd={marker} />
      {state === "active" ? (
        <>
          <circle className="fc-pulse a" r={2.5} style={pulseStyle} />
          <circle className="fc-pulse b" r={2.5} style={pulseStyle} />
        </>
      ) : null}
    </g>
  );
}

function ActiveStream({
  phases,
  tokenText,
  detailLines,
  delibPath,
  finalAttempt,
  activeStreamPhase,
}: {
  phases: PhasesRecord;
  tokenText: string;
  detailLines: readonly string[];
  delibPath: "system_1" | "system_2" | null;
  finalAttempt: number;
  activeStreamPhase: TurnPhaseName | null;
}) {
  const phase = activeStreamPhase;
  const status: PhaseStatus | "idle" = phase ? phases[phase]?.status ?? "queue" : "idle";
  const isRunning = status === "running";
  const phaseName = phase ? phases[phase]?.name ?? PHASE_LABELS[phase] ?? phase : null;
  const tokenPhase = phase === "delib" || phase === "final";
  const delibMeta =
    delibPath === "system_2"
      ? "S2 · plan"
      : delibPath === "system_1"
        ? "S1 · fast"
        : "path pending";
  // Detail lines are stored and rendered oldest-to-newest so the latest phase
  // update lands at the bottom, matching token stream reading order.
  const detailText = detailLines.join("\n");

  const streamPending = "stream open…";
  const body = !phase
    ? "waiting for a running phase"
    : tokenPhase
      ? tokenText.length > 0
        ? tokenText
        : streamPending
      : detailText.length > 0
        ? detailText
        : streamPending;

  return (
    <div className={`flow-active-stream ${!phase ? "idle" : ""}`} data-status={status}>
      <div className="flow-active-head">
        <span className="pin">
          active stream
          {phaseName ? (
            <>
              {" · "}
              <strong>{phaseName}</strong>
            </>
          ) : null}
        </span>
        <span className="meta">
          {phase === "final" ? (
            <>
              <span className={`chip ${finalAttempt > 1 ? "warn" : "acc"}`}>
                attempt {finalAttempt}
              </span>
              {delibPath ? (
                <span className="chip">{delibPath === "system_2" ? "via S2" : "via S1"}</span>
              ) : null}
            </>
          ) : phase === "delib" ? (
            <span className={`chip ${finalAttempt > 1 ? "warn" : "acc"}`}>{delibMeta}</span>
          ) : phase ? (
            <span
              className={`chip ${
                status === "running" ? "acc" : status === "fail" ? "warn" : ""
              }`.trim()}
            >
              {status}
            </span>
          ) : null}
        </span>
      </div>
      <pre
        className={`flow-active-body ${!phase ? "empty" : ""} ${status === "done" ? "muted" : ""}`}
      >
        {body}
        {isRunning ? <span className="caret"></span> : null}
      </pre>
    </div>
  );
}

// -----------------------------------------------------------------------------
// Main
// -----------------------------------------------------------------------------

export function FlowChart({
  phases,
  activeTurnId,
  tokenTextByPhase,
  detailByPhase,
  terminalOutcome,
  delibPath,
  finalAttempt,
  particleEnabled = true,
  particleDensity = 320,
}: FlowChartProps) {
  const phasesRecord = useMemo(() => buildPhaseRecord(phases), [phases]);
  const activeStreamPhase = useMemo(
    () => deriveActiveStreamPhase(phasesRecord, activeTurnId, tokenTextByPhase, detailByPhase),
    [phasesRecord, activeTurnId, tokenTextByPhase, detailByPhase],
  );
  const streamTokenText =
    activeTurnId === null || activeStreamPhase === null
      ? ""
      : tokenTextByPhase.get(tokenKey(activeTurnId, activeStreamPhase)) ?? "";
  const streamDetailLines =
    activeTurnId === null || activeStreamPhase === null
      ? []
      : detailByPhase.get(tokenKey(activeTurnId, activeStreamPhase)) ?? [];

  const phStatus = (id: string): PhaseStatus => phasesRecord[id]?.status ?? "queue";

  const closureSuppressed = terminalOutcome === "suppressed_closure";
  const gateSuppressed = terminalOutcome === "suppressed_generation_gate";
  const guardsSuppressed = terminalOutcome === "suppressed_action";

  const edgeState = (fromId: string, toId: string): "queue" | "active" | "done" => {
    const from = phStatus(fromId);
    const to = phStatus(toId);
    if (from === "queue") return "queue";
    if (to === "running" || from === "running") return "active";
    return "done";
  };

  // gen -> retrieval descent: stays dim until the gate actually completes (so
  // a running/suppressed/failed gate never renders the hop as done), goes live
  // while retrieval runs, then done.
  const wrapState: "queue" | "active" | "done" = (() => {
    const gateDone = phStatus("generation_gate") === "done";
    const retrievalActive = phStatus("retrieval") === "running";
    if (!gateDone) return "queue";
    if (retrievalActive) return "active";
    return "done";
  })();

  const delibDone = phStatus("delib") === "done";
  const finalRun = phStatus("final") === "running";
  const finalDone = phStatus("final") === "done";

  const forkActive = (lane: "s1" | "s2"): boolean => {
    if (!delibDone || delibPath === null) return false;
    if (delibPath === "system_1" && lane === "s1") return finalRun;
    if (delibPath === "system_2" && lane === "s2") return finalRun;
    return false;
  };
  const forkChosen = (lane: "s1" | "s2"): boolean | null => {
    if (delibPath === null) return null;
    return (
      (lane === "s1" && delibPath === "system_1") ||
      (lane === "s2" && delibPath === "system_2")
    );
  };

  const outcomeTone = (() => {
    if (terminalOutcome === null) return "idle";
    if (terminalOutcome === "reflected") return "";
    if (terminalOutcome === "aborted") return "warn";
    return "bad";
  })();
  const outcomeLabel =
    terminalOutcome === null ? "waiting" : terminalOutcome.replace(/_/g, " ");

  // Subtle background dots -- sparser than the original (every 50px).
  const dots = useMemo(() => {
    const out: ReactNode[] = [];
    for (let y = 50; y < VIEW_H - 30; y += 50) {
      for (let x = 60; x < VIEW_W - 30; x += 50) {
        out.push(<circle key={`${x}-${y}`} cx={x} cy={y} r={0.6} className="fc-bg-dots" />);
      }
    }
    return out;
  }, []);

  // Particle cloud focal point: first running node, else first failing node.
  const particleTarget = useMemo<ParticleTarget>(() => {
    for (const id of PHASE_IDS) {
      if (phasesRecord[id]?.status === "running") {
        const n = NODES[id]!;
        return { x: n.x, y: n.y };
      }
    }
    for (const id of PHASE_IDS) {
      if (phasesRecord[id]?.status === "fail") {
        const n = NODES[id]!;
        return { x: n.x, y: n.y };
      }
    }
    return null;
  }, [phasesRecord]);

  return (
    <div className="flow-shell">
      <div className="flow-topline">
        <div className="left">
          <span className="eyebrow">turn</span>
          <span className="turn-id">
            {activeTurnId === null ? (
              <span className="dim">idle</span>
            ) : (
              <span className="acc">{activeTurnId}</span>
            )}
          </span>
          <span className="eyebrow">outcome</span>
          <span className={`flow-topline-status ${outcomeTone}`.trim()}>{outcomeLabel}</span>
          {finalAttempt > 1 ? (
            <span className="flow-topline-status warn">regen · attempt {finalAttempt}</span>
          ) : null}
        </div>
        <div className="right">
          <div className="flow-legend" aria-label="phase legend">
            <span className="leg queue">queue</span>
            <span className="leg run">run</span>
            <span className="leg done">done</span>
            <span className="leg fail">fail</span>
          </div>
        </div>
      </div>

      <div className="fc-canvas">
        <ParticleField
          target={particleTarget}
          viewW={VIEW_W}
          viewH={VIEW_H}
          density={particleDensity}
          enabled={particleEnabled}
        />
        <svg
          className="fc-svg fc-style-synaptic"
          viewBox={`0 0 ${VIEW_W} ${VIEW_H}`}
          preserveAspectRatio="xMidYMid meet"
          role="img"
          aria-label="cognitive turn flow chart"
        >
          <defs>
            <radialGradient id="bg-glow-a" cx="50%" cy="50%" r="50%">
              <stop offset="0%" stopColor="oklch(0.84 0.155 142 / 0.07)" />
              <stop offset="100%" stopColor="oklch(0.84 0.155 142 / 0)" />
            </radialGradient>
            <radialGradient id="bg-glow-b" cx="50%" cy="50%" r="50%">
              <stop offset="0%" stopColor="oklch(0.78 0.115 232 / 0.05)" />
              <stop offset="100%" stopColor="oklch(0.78 0.115 232 / 0)" />
            </radialGradient>

            <marker
              id="arrow-active"
              viewBox="0 0 10 10"
              refX="8.5"
              refY="5"
              markerWidth="6"
              markerHeight="6"
              orient="auto-start-reverse"
            >
              <path d="M 1 1.5 L 9 5 L 1 8.5 z" fill="oklch(0.84 0.155 142)" />
            </marker>
            <marker
              id="arrow-done"
              viewBox="0 0 10 10"
              refX="8.5"
              refY="5"
              markerWidth="6"
              markerHeight="6"
              orient="auto-start-reverse"
            >
              <path d="M 1 1.5 L 9 5 L 1 8.5 z" fill="oklch(0.55 0.1 142 / 0.85)" />
            </marker>
            <marker
              id="arrow-queue"
              viewBox="0 0 10 10"
              refX="8.5"
              refY="5"
              markerWidth="6"
              markerHeight="6"
              orient="auto-start-reverse"
            >
              <path d="M 1 1.5 L 9 5 L 1 8.5 z" fill="oklch(0.37 0.006 80)" />
            </marker>
            <marker
              id="arrow-warn"
              viewBox="0 0 10 10"
              refX="8.5"
              refY="5"
              markerWidth="6"
              markerHeight="6"
              orient="auto-start-reverse"
            >
              <path d="M 1 1.5 L 9 5 L 1 8.5 z" fill="oklch(0.835 0.135 85)" />
            </marker>
            <marker
              id="arrow-warn-dim"
              viewBox="0 0 10 10"
              refX="8.5"
              refY="5"
              markerWidth="6"
              markerHeight="6"
              orient="auto-start-reverse"
            >
              <path d="M 1 1.5 L 9 5 L 1 8.5 z" fill="oklch(0.55 0.085 85)" />
            </marker>
          </defs>

          {/* Background bands */}
          {TIERS.map((t) => (
            <rect
              key={`band-${t.id}`}
              className={`fc-band ${t.id === 2 ? "alt" : ""}`.trim()}
              x={0}
              y={BAND[t.id][0]}
              width={VIEW_W}
              height={BAND[t.id][1] - BAND[t.id][0]}
            />
          ))}

          {/* Soft glows by tier */}
          <circle cx={550} cy={TIER_Y[1]} r={340} fill="url(#bg-glow-b)" />
          <circle cx={650} cy={TIER_Y[2]} r={380} fill="url(#bg-glow-a)" />
          <circle cx={500} cy={TIER_Y[3]} r={340} fill="url(#bg-glow-a)" />

          {/* Band dividers */}
          <line className="fc-band-divider" x1={0} y1={BAND[2][0]} x2={VIEW_W} y2={BAND[2][0]} />
          <line className="fc-band-divider" x1={0} y1={BAND[3][0]} x2={VIEW_W} y2={BAND[3][0]} />

          {dots}

          {/* Tier headers (top-left of each band) */}
          {TIERS.map((t) => (
            <TierHeader key={`tier-${t.id}`} tier={t} band={BAND[t.id]} />
          ))}

          {/* Spine edges */}
          {SPINE_EDGES.map(([fromId, toId]) => {
            const isDescent = NODES[fromId]!.y !== NODES[toId]!.y;
            const state =
              isDescent && fromId === "generation_gate" && toId === "retrieval"
                ? wrapState
                : edgeState(fromId, toId);
            return (
              <Edge
                key={`${fromId}-${toId}`}
                d={spinePath(fromId, toId)}
                state={state}
                descent={isDescent}
              />
            );
          })}

          {/* S1/S2 fork (delib -> final) */}
          {(() => {
            const s1Active = forkActive("s1") || (forkChosen("s1") === true && finalDone);
            const s2Active = forkActive("s2") || (forkChosen("s2") === true && finalDone);
            const s1Cls = `fc-fork-lane ${s1Active ? "active" : ""} ${forkChosen("s1") === true ? "chosen" : ""} ${forkChosen("s1") === false ? "unchosen" : ""}`;
            const s2Cls = `fc-fork-lane ${s2Active ? "active" : ""} ${forkChosen("s2") === true ? "chosen" : ""} ${forkChosen("s2") === false ? "unchosen" : ""}`;
            const laneMarker = (lane: "s1" | "s2") => {
              if (forkActive(lane)) return "url(#arrow-active)";
              if (forkChosen(lane) === true) return "url(#arrow-done)";
              return "url(#arrow-queue)";
            };
            return (
              <>
                <g className={s1Cls}>
                  <path d={forkPath("s1")} className="fc-fork-lane-path" markerEnd={laneMarker("s1")} />
                  <g transform="translate(305 421)">
                    <rect className="fc-fork-tag-bg" x={-30} y={-13} width={60} height={26} rx={3} />
                    <text className="fc-fork-tag" x={0} y={-2}>
                      S1
                    </text>
                    <text className="fc-fork-desc" x={0} y={9}>
                      fast
                    </text>
                  </g>
                </g>
                <g className={s2Cls}>
                  <path d={forkPath("s2")} className="fc-fork-lane-path" markerEnd={laneMarker("s2")} />
                  <g transform="translate(257 370)">
                    <rect className="fc-fork-tag-bg" x={-30} y={-13} width={60} height={26} rx={3} />
                    <text className="fc-fork-tag" x={0} y={-2}>
                      S2
                    </text>
                    <text className="fc-fork-desc" x={0} y={9}>
                      plan
                    </text>
                  </g>
                </g>
              </>
            );
          })()}

          {/* Regen arc */}
          {(() => {
            const active = finalAttempt > 1;
            const cls = `fc-edge regen ${active ? "fire" : ""}`;
            const marker = active ? "url(#arrow-warn)" : "url(#arrow-warn-dim)";
            return (
              <g className={`fc-regen-group ${active ? "active" : ""}`}>
                <path d={regenPath()} className={cls} markerEnd={marker} />
                <g transform={`translate(${(NODES.guards!.x + NODES.final!.x) / 2} 624)`}>
                  <rect className="fc-regen-label-bg" x={-38} y={-9} width={76} height={16} rx={2} />
                  <text className={`fc-regen-label ${active ? "active" : ""}`} x={0} y={3}>
                    regen ↻
                  </text>
                </g>
              </g>
            );
          })()}

          {/* Suppression thorns */}
          <ThornUp nodeId="closure_loop" label="suppress closure" active={closureSuppressed} />
          <ThornUp nodeId="generation_gate" label="suppress gen" active={gateSuppressed} />
          <ThornSidecar nodeId="guards" label="guards trip" active={guardsSuppressed} />

          {/* Endpoints + nodes */}
          <Endpoint node={INPUT_NODE} outcome={null} />
          {PHASE_IDS.map((id) => {
            const n = NODES[id];
            if (!n || n.kind === "endpoint") return null;
            return (
              <Ganglion
                key={id}
                node={n}
                status={phStatus(id)}
                sub={phasesRecord[id]?.sub}
                duration={phasesRecord[id]?.durationMs}
              />
            );
          })}
          <Endpoint node={TERMINAL_NODE} outcome={terminalOutcome} />
        </svg>
      </div>

      <ActiveStream
        phases={phasesRecord}
        tokenText={streamTokenText}
        detailLines={streamDetailLines}
        delibPath={delibPath}
        finalAttempt={finalAttempt}
        activeStreamPhase={activeStreamPhase}
      />
    </div>
  );
}
