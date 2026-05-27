import { useMemo, type CSSProperties, type ReactNode } from "react";

import type { TurnTerminalOutcome } from "../../api/types";
import type { PhaseState } from "../../hooks/use-turn-stream";

// Synaptic flow chart -- each phase rendered as a small "ganglion"
// with a halo, dendrite stubs and a state-coloured nucleus. Edges are
// smooth bezier traces; active edges march and emit pulse particles.
// Decision gates are hexagons. The S1/S2 fork shows two parallel
// paths bifurcating from delib and converging at final; the regen arc
// loops back over tier 3.

export type FlowChartProps = {
  phases: readonly PhaseState[];
  activeTurnId: string | null;
  tokenTextByPhase: Map<string, string>;
  terminalOutcome: TurnTerminalOutcome | null;
  delibPath: "system_1" | "system_2" | null;
  finalAttempt: number;
};

// Layout (viewBox 1200 x 600)
const NODE_R = 22;
const GATE_R = 26;
const ENDPOINT_H = 32;

const TIER_Y = { 1: 85, 2: 295, 3: 500 } as const;

type TierDef = { id: string; y: number; label: string };
const TIERS: readonly TierDef[] = [
  { id: "perception", y: TIER_Y[1], label: "PERCEPTION" },
  { id: "context", y: TIER_Y[2], label: "CONTEXT" },
  { id: "synthesis", y: TIER_Y[3], label: "SYNTHESIS" },
];

type EndpointNode = {
  id: string;
  x: number;
  y: number;
  w: number;
  label: string;
  kind: "endpoint";
};

type PhaseNode = {
  id: string;
  x: number;
  y: number;
  label: string;
  kind: "phase" | "gate";
};

type LayoutNode = EndpointNode | PhaseNode;

const INPUT_NODE: EndpointNode = {
  id: "input",
  x: 60,
  y: TIER_Y[1],
  w: 78,
  label: "input",
  kind: "endpoint",
};
const TERMINAL_NODE: EndpointNode = {
  id: "terminal",
  x: 1020,
  y: TIER_Y[3],
  w: 130,
  label: "terminal",
  kind: "endpoint",
};

const PHASES_LAYOUT: readonly PhaseNode[] = [
  { id: "ingest", x: 195, y: TIER_Y[1], label: "ingest", kind: "phase" },
  { id: "audience", x: 330, y: TIER_Y[1], label: "audience", kind: "phase" },
  { id: "perception", x: 465, y: TIER_Y[1], label: "perception", kind: "phase" },
  { id: "frame", x: 600, y: TIER_Y[1], label: "frame gate", kind: "phase" },
  { id: "extract", x: 735, y: TIER_Y[1], label: "extraction", kind: "phase" },
  { id: "closure_loop", x: 870, y: TIER_Y[1], label: "closure?", kind: "gate" },
  { id: "generation_gate", x: 1005, y: TIER_Y[1], label: "gen gate?", kind: "gate" },
  { id: "retrieval", x: 195, y: TIER_Y[2], label: "retrieval", kind: "phase" },
  { id: "ledger", x: 425, y: TIER_Y[2], label: "ev. ledger", kind: "phase" },
  { id: "shared", x: 655, y: TIER_Y[2], label: "shared state", kind: "phase" },
  { id: "delib", x: 885, y: TIER_Y[2], label: "deliberation", kind: "phase" },
  { id: "final", x: 330, y: TIER_Y[3], label: "finalizer", kind: "phase" },
  { id: "guards", x: 500, y: TIER_Y[3], label: "guards?", kind: "gate" },
  { id: "persist", x: 670, y: TIER_Y[3], label: "persist", kind: "phase" },
  { id: "reflect", x: 840, y: TIER_Y[3], label: "reflection", kind: "phase" },
];

const NODE_BY_ID: Record<string, LayoutNode> = (() => {
  const map: Record<string, LayoutNode> = {
    input: INPUT_NODE,
    terminal: TERMINAL_NODE,
  };
  for (const p of PHASES_LAYOUT) map[p.id] = p;
  return map;
})();

const SPINE_EDGES: ReadonlyArray<readonly [string, string]> = [
  ["input", "ingest"],
  ["ingest", "audience"],
  ["audience", "perception"],
  ["perception", "frame"],
  ["frame", "extract"],
  ["extract", "closure_loop"],
  ["closure_loop", "generation_gate"],
  ["retrieval", "ledger"],
  ["ledger", "shared"],
  ["shared", "delib"],
  ["final", "guards"],
  ["guards", "persist"],
  ["persist", "reflect"],
  ["reflect", "terminal"],
];

function nodeRight(node: LayoutNode): { x: number; y: number } {
  if (node.kind === "endpoint") return { x: node.x + node.w / 2, y: node.y };
  if (node.kind === "gate") return { x: node.x + GATE_R + 2, y: node.y };
  return { x: node.x + NODE_R + 2, y: node.y };
}
function nodeLeft(node: LayoutNode): { x: number; y: number } {
  if (node.kind === "endpoint") return { x: node.x - node.w / 2, y: node.y };
  if (node.kind === "gate") return { x: node.x - GATE_R - 2, y: node.y };
  return { x: node.x - NODE_R - 2, y: node.y };
}
function nodeBottom(node: LayoutNode): { x: number; y: number } {
  if (node.kind === "endpoint") return { x: node.x, y: node.y + ENDPOINT_H / 2 };
  if (node.kind === "gate") return { x: node.x, y: node.y + GATE_R + 2 };
  return { x: node.x, y: node.y + NODE_R + 2 };
}
function nodeTop(node: LayoutNode): { x: number; y: number } {
  if (node.kind === "endpoint") return { x: node.x, y: node.y - ENDPOINT_H / 2 };
  if (node.kind === "gate") return { x: node.x, y: node.y - GATE_R - 2 };
  return { x: node.x, y: node.y - NODE_R - 2 };
}

function spinePath(from: LayoutNode, to: LayoutNode): string {
  const a = nodeRight(from);
  const b = nodeLeft(to);
  const dx = (b.x - a.x) * 0.45;
  return `M ${a.x} ${a.y} C ${a.x + dx} ${a.y}, ${b.x - dx} ${b.y}, ${b.x} ${b.y}`;
}

function wrapAroundPath(): string {
  const from = nodeRight(NODE_BY_ID.generation_gate!);
  const to = nodeTop(NODE_BY_ID.retrieval!);
  return `M ${from.x} ${from.y} C 1180 ${from.y}, 1180 190, 1090 190 L 280 190 C 60 190, 60 ${to.y - 30}, ${to.x} ${to.y}`;
}

function forkPath(which: "s1" | "s2"): string {
  const a = nodeBottom(NODE_BY_ID.delib!);
  const b = nodeTop(NODE_BY_ID.final!);
  const midY = (a.y + b.y) / 2;
  if (which === "s1") {
    return `M ${a.x} ${a.y} C ${a.x} ${midY}, ${b.x + 60} ${midY}, ${b.x} ${b.y}`;
  }
  return `M ${a.x} ${a.y} C ${a.x + 60} ${a.y + 30}, ${a.x + 60} ${midY + 30}, ${a.x - 80} ${midY + 30} C ${b.x - 40} ${midY + 30}, ${b.x} ${midY + 50}, ${b.x} ${b.y}`;
}

function regenPath(): string {
  const a = nodeTop(NODE_BY_ID.guards!);
  const b = nodeTop(NODE_BY_ID.final!);
  const peakY = a.y - 38;
  return `M ${a.x} ${a.y} C ${a.x} ${peakY}, ${b.x} ${peakY}, ${b.x} ${b.y}`;
}

function thornDrop(gateId: string, thornY: number): string {
  const node = NODE_BY_ID[gateId]!;
  const from = nodeBottom(node);
  return `M ${from.x} ${from.y} V ${thornY}`;
}

type PhaseStatus = PhaseState["status"];

type PhasesRecord = Record<
  string,
  { status: PhaseStatus; sub?: string; durationMs?: number }
>;

function buildPhaseRecord(phases: readonly PhaseState[]): PhasesRecord {
  const out: PhasesRecord = {};
  for (const phase of phases) {
    out[phase.id] = {
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
): "delib" | "final" | null {
  if (phases.final?.status === "running") return "final";
  if (phases.delib?.status === "running") return "delib";
  if (activeTurnId !== null) {
    if ((tokenTextByPhase.get(tokenKey(activeTurnId, "final")) ?? "").length > 0) return "final";
    if ((tokenTextByPhase.get(tokenKey(activeTurnId, "delib")) ?? "").length > 0) return "delib";
  }
  return null;
}

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
  const r = isGate ? GATE_R : NODE_R;

  const hex = useMemo(() => {
    const pts: string[] = [];
    for (let i = 0; i < 6; i++) {
      const angle = (Math.PI / 3) * i - Math.PI / 2;
      pts.push(`${(Math.cos(angle) * r).toFixed(2)},${(Math.sin(angle) * r).toFixed(2)}`);
    }
    return pts.join(" ");
  }, [r]);

  const dendrites = !isGate ? (
    <g className="fc-dendrite-grp">
      <line className="fc-dendrite" x1={-r - 1} y1={0} x2={-r - 6} y2={-3} />
      <line className="fc-dendrite" x1={-r - 1} y1={0} x2={-r - 6} y2={3} />
      <line className="fc-dendrite" x1={r + 1} y1={0} x2={r + 6} y2={-3} />
      <line className="fc-dendrite" x1={r + 1} y1={0} x2={r + 6} y2={3} />
    </g>
  ) : null;

  return (
    <g
      className={`fc-node fc-node-${status}`}
      data-status={status}
      data-testid={`phase-${node.id}`}
      transform={`translate(${x} ${y})`}
    >
      <circle className="fc-node-halo" r={r + 9} />
      <circle className="fc-pulse-ring" r={r} />
      <circle className="fc-pulse-ring b" r={r} />
      <circle className="fc-pulse-ring c" r={r} />

      {dendrites}

      {isGate ? (
        <polygon className="fc-node-body" points={hex} />
      ) : (
        <circle className="fc-node-body" r={r} />
      )}
      <circle className="fc-node-core" r={5} />

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
          {sub}
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
  const { x, y, w, label, id } = node;
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
      className={`fc-endpoint ${isTerminal ? "terminal" : ""}`}
      data-outcome={outcome ?? ""}
      transform={`translate(${x - w / 2} ${y - h / 2})`}
    >
      <rect className="fc-endpoint-body" x={0} y={0} width={w} height={h} rx={r} ry={r} />
      <text className="fc-endpoint-label" x={w / 2} y={h / 2 + 3.5}>
        {display}
      </text>
    </g>
  );
}

function Thorn({
  gateId,
  x,
  y,
  label,
  active,
}: {
  gateId: string;
  x: number;
  y: number;
  label: string;
  active: boolean;
}) {
  return (
    <g className={`fc-thorn ${active ? "active" : ""}`}>
      <path d={thornDrop(gateId, y)} className={`fc-edge branch ${active ? "fire" : ""}`} />
      <g transform={`translate(${x - 38} ${y})`}>
        <rect className="fc-thorn-shape" x={0} y={0} width={76} height={20} rx={2} />
        <text className="fc-thorn-label" x={38} y={14}>
          {label}
        </text>
      </g>
    </g>
  );
}

function ActiveStream({
  phases,
  tokenText,
  delibPath,
  finalAttempt,
  activeStreamPhase,
}: {
  phases: PhasesRecord;
  tokenText: string;
  delibPath: "system_1" | "system_2" | null;
  finalAttempt: number;
  activeStreamPhase: "delib" | "final" | null;
}) {
  const phase = activeStreamPhase;
  const status: PhaseStatus | "idle" = phase
    ? phases[phase]?.status ?? "queue"
    : "idle";
  const isRunning = status === "running";
  const phaseName =
    phase === "final" ? "finalizer" : phase === "delib" ? "deliberation" : null;

  const meta = (() => {
    if (!phase) return "no streaming phase";
    if (phase === "delib") {
      if (delibPath === null) return "path pending";
      return delibPath === "system_2" ? "S2 · EmitTurnPlan" : "S1 · fast path";
    }
    return `attempt ${finalAttempt}`;
  })();

  const body = !phase
    ? "waiting for delib or final to produce tokens"
    : tokenText.length > 0
      ? tokenText
      : "stream open…";

  return (
    <div className={`flow-active-stream ${status}`} data-status={status}>
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
        <span className="dim">{meta}</span>
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

export function FlowChart({
  phases,
  activeTurnId,
  tokenTextByPhase,
  terminalOutcome,
  delibPath,
  finalAttempt,
}: FlowChartProps) {
  const phasesRecord = useMemo(() => buildPhaseRecord(phases), [phases]);
  const activeStreamPhase = useMemo(
    () => deriveActiveStreamPhase(phasesRecord, activeTurnId, tokenTextByPhase),
    [phasesRecord, activeTurnId, tokenTextByPhase],
  );
  const streamTokenText =
    activeTurnId === null || activeStreamPhase === null
      ? ""
      : tokenTextByPhase.get(tokenKey(activeTurnId, activeStreamPhase)) ?? "";

  const phStatus = (id: string): PhaseStatus =>
    phasesRecord[id]?.status ?? "queue";

  const closureSuppressed = terminalOutcome === "suppressed_closure";
  const gateSuppressed = terminalOutcome === "suppressed_generation_gate";
  const guardsSuppressed = terminalOutcome === "suppressed_action";

  const edgeState = (fromId: string, toId: string): "queue" | "active" | "done" => {
    const from = phStatus(fromId);
    const to = phStatus(toId);
    if (from === "queue") return "queue";
    if (to === "running") return "active";
    if (from === "running") return "active";
    if (from === "done" && to === "done") return "done";
    return "done";
  };

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
    if (!delibDone) return false;
    if (delibPath === null) return false;
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

  const dots = useMemo(() => {
    const out: ReactNode[] = [];
    for (let y = 30; y < 600; y += 30) {
      for (let x = 30; x < 1200; x += 30) {
        out.push(
          <circle key={`${x}-${y}`} cx={x} cy={y} r={0.7} className="fc-bg-dots" />,
        );
      }
    }
    return out;
  }, []);

  const outcomeTone = (() => {
    if (terminalOutcome === null) return "idle";
    if (terminalOutcome === "reflected") return "";
    if (terminalOutcome === "aborted") return "warn";
    return "bad";
  })();
  const outcomeLabel =
    terminalOutcome === null ? "waiting" : terminalOutcome.replace(/_/g, " ");

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
          <span className={`flow-topline-status ${outcomeTone}`.trim()}>
            {outcomeLabel}
          </span>
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
        <svg
          className="fc-svg fc-style-synaptic"
          viewBox="0 0 1200 600"
          preserveAspectRatio="xMidYMid meet"
          role="img"
          aria-label="cognitive turn flow chart"
        >
          <defs>
            <radialGradient id="bg-glow-1" cx="50%" cy="50%" r="50%">
              <stop offset="0%" stopColor="oklch(0.84 0.155 142 / 0.12)" />
              <stop offset="100%" stopColor="oklch(0.84 0.155 142 / 0)" />
            </radialGradient>
            <radialGradient id="bg-glow-2" cx="50%" cy="50%" r="50%">
              <stop offset="0%" stopColor="oklch(0.78 0.115 232 / 0.08)" />
              <stop offset="100%" stopColor="oklch(0.78 0.115 232 / 0)" />
            </radialGradient>
            <radialGradient id="bg-glow-3" cx="50%" cy="50%" r="50%">
              <stop offset="0%" stopColor="oklch(0.74 0.135 305 / 0.06)" />
              <stop offset="100%" stopColor="oklch(0.74 0.135 305 / 0)" />
            </radialGradient>
          </defs>

          <circle cx={465} cy={85} r={280} fill="url(#bg-glow-2)" />
          <circle cx={655} cy={295} r={340} fill="url(#bg-glow-1)" />
          <circle cx={500} cy={500} r={300} fill="url(#bg-glow-1)" />
          <circle cx={1100} cy={300} r={220} fill="url(#bg-glow-3)" />

          <line x1={50} y1={195} x2={1150} y2={195} className="fc-membrane" />
          <line x1={50} y1={400} x2={1150} y2={400} className="fc-membrane" />

          {dots}

          {TIERS.map((t) => (
            <g key={t.id} transform={`translate(18 ${t.y})`}>
              <line className="fc-tier-bar" x1={0} y1={-28} x2={0} y2={28} />
              <text className="fc-tier-label" x={8} y={-8}>
                {t.label}
              </text>
            </g>
          ))}

          {SPINE_EDGES.map(([fromId, toId]) => {
            const from = NODE_BY_ID[fromId]!;
            const to = NODE_BY_ID[toId]!;
            const state = edgeState(fromId, toId);
            const d = spinePath(from, to);
            const cls = `fc-edge ${state === "active" ? "active" : state === "done" ? "done" : "dim"}`;
            const pulseStyle = { offsetPath: `path("${d}")` } as CSSProperties;
            return (
              <g
                key={`${fromId}-${toId}`}
                className={`fc-edge-wrap ${state === "active" ? "active" : ""}`}
              >
                <path d={d} className={cls} />
                {state === "active" ? (
                  <>
                    <circle className="fc-pulse a" r={2.2} style={pulseStyle} />
                    <circle className="fc-pulse b" r={2.2} style={pulseStyle} />
                  </>
                ) : null}
              </g>
            );
          })}

          {(() => {
            const d = wrapAroundPath();
            const cls = `fc-edge ${wrapState === "active" ? "active" : wrapState === "done" ? "done" : "dim"}`;
            const pulseStyle = { offsetPath: `path("${d}")` } as CSSProperties;
            return (
              <g className={`fc-edge-wrap ${wrapState === "active" ? "active" : ""}`}>
                <path d={d} className={cls} />
                {wrapState === "active" ? (
                  <>
                    <circle className="fc-pulse a" r={2.2} style={pulseStyle} />
                    <circle className="fc-pulse b" r={2.2} style={pulseStyle} />
                  </>
                ) : null}
              </g>
            );
          })()}

          {(() => {
            const s1 = forkPath("s1");
            const s2 = forkPath("s2");
            const s1Chosen = forkChosen("s1");
            const s2Chosen = forkChosen("s2");
            const s1Active = forkActive("s1") || (s1Chosen === true && finalDone);
            const s2Active = forkActive("s2") || (s2Chosen === true && finalDone);
            const s1Cls = `fc-fork-lane ${s1Active ? "active" : ""} ${s1Chosen === false ? "unchosen" : ""}`;
            const s2Cls = `fc-fork-lane ${s2Active ? "active" : ""} ${s2Chosen === false ? "unchosen" : ""}`;
            const labelY1 = 380;
            const labelY2 = 430;
            return (
              <>
                <g className={s1Cls}>
                  <path d={s1} className="fc-fork-lane-path" />
                  <text className="fc-fork-tag" x={600} y={labelY1 - 4}>
                    S1
                  </text>
                  <text className="fc-fork-desc" x={600} y={labelY1 + 8}>
                    fast
                  </text>
                </g>
                <g className={s2Cls}>
                  <path d={s2} className="fc-fork-lane-path" />
                  <text className="fc-fork-tag" x={780} y={labelY2 - 4}>
                    S2
                  </text>
                  <text className="fc-fork-desc" x={780} y={labelY2 + 8}>
                    plan
                  </text>
                </g>
              </>
            );
          })()}

          {(() => {
            const d = regenPath();
            const active = finalAttempt > 1;
            const cls = `fc-edge regen ${active ? "fire" : ""}`;
            return (
              <g>
                <path d={d} className={cls} />
                <text
                  className={`fc-regen-label ${active ? "active" : ""}`}
                  x={(NODE_BY_ID.guards!.x + NODE_BY_ID.final!.x) / 2}
                  y={NODE_BY_ID.guards!.y - 50}
                >
                  regen ↻
                </text>
              </g>
            );
          })()}

          <Thorn
            gateId="closure_loop"
            x={870}
            y={170}
            label="closure suppress"
            active={closureSuppressed}
          />
          <Thorn
            gateId="generation_gate"
            x={1005}
            y={170}
            label="gate suppress"
            active={gateSuppressed}
          />
          <Thorn
            gateId="guards"
            x={500}
            y={570}
            label="guards trip"
            active={guardsSuppressed}
          />

          <Endpoint node={INPUT_NODE} outcome={null} />
          {PHASES_LAYOUT.map((p) => (
            <Ganglion
              key={p.id}
              node={p}
              status={phStatus(p.id)}
              sub={phasesRecord[p.id]?.sub}
              duration={phasesRecord[p.id]?.durationMs}
            />
          ))}
          <Endpoint node={TERMINAL_NODE} outcome={terminalOutcome} />
        </svg>
      </div>

      <ActiveStream
        phases={phasesRecord}
        tokenText={streamTokenText}
        delibPath={delibPath}
        finalAttempt={finalAttempt}
        activeStreamPhase={activeStreamPhase}
      />
    </div>
  );
}
