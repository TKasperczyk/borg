import type { TurnTerminalOutcome } from "../../api/types";
import type { PhaseState } from "../../hooks/use-turn-stream";

export type FlowChartProps = {
  phases: readonly PhaseState[];
  activeTurnId: string | null;
  tokenTextByPhase: Map<string, string>;
  terminalOutcome: TurnTerminalOutcome | null;
  delibPath: "system_1" | "system_2" | null;
  finalAttempt: number;
};

const STREAM_PHASES = new Set(["delib", "final"]);

function tokenKey(turnId: string, phase: string): string {
  return `${turnId}:${phase}`;
}

function phaseTokenText(
  phaseId: string,
  activeTurnId: string | null,
  tokenTextByPhase: Map<string, string>,
): string {
  if (activeTurnId === null) return "";
  return tokenTextByPhase.get(tokenKey(activeTurnId, phaseId)) ?? "";
}

function terminalLabel(outcome: TurnTerminalOutcome | null): string {
  return outcome === null ? "waiting" : outcome;
}

/* SVG geometry — viewBox 1000x520. The chart is a true 2D flow:
   - Row 1 (y=40): input + 7 sequential pre-retrieval phases flowing
     left-to-right. closure_loop and generation_gate are decision diamonds.
   - Both decision diamonds branch DOWN to small dashed suppression-terminal
     boxes at y=125.
   - After generation_gate the flow turns around: a curved path drops down
     to y=215 and continues left into retrieval.
   - Row 2 (y=215): retrieval → ledger → shared → delib (4 phases).
   - Below delib (y=300): S1/S2 fork — two mini-pills branching from delib
     and merging back up at final.
   - Row 3 (y=380): final → guards (diamond) → persist → reflect → terminal.
   - guards branches DOWN to a guards_trip suppression terminal at y=465.
   - A curved regen path loops from guards back up to final's left edge,
     drawn above row 3, lighting up only when finalAttempt > 1.
*/

type NodeStatus = PhaseState["status"];

type SpineNode = {
  id: string;
  label: string;
  x: number;
  y: number;
  w: number;
  h: number;
  shape: "rect" | "diamond" | "round";
  state: NodeStatus | "marker";
  sub?: string;
  duration?: number;
};

const NODE_HEIGHT = 38;
const NODE_HEIGHT_DECISION = 44;

function rectPath(x: number, y: number, w: number, h: number, r = 4): string {
  return `M${x + r} ${y} h${w - 2 * r} a${r} ${r} 0 0 1 ${r} ${r} v${h - 2 * r} a${r} ${r} 0 0 1 -${r} ${r} h-${w - 2 * r} a${r} ${r} 0 0 1 -${r} -${r} v-${h - 2 * r} a${r} ${r} 0 0 1 ${r} -${r} z`;
}

function diamondPath(x: number, y: number, w: number, h: number): string {
  const cx = x + w / 2;
  const cy = y + h / 2;
  return `M${cx} ${y} L${x + w} ${cy} L${cx} ${y + h} L${x} ${cy} Z`;
}

function statusClass(state: NodeStatus | "marker"): string {
  if (state === "marker") return "marker";
  return state;
}

function findPhase(phases: readonly PhaseState[], id: string): PhaseState | undefined {
  return phases.find((p) => p.id === id);
}

function nodeFromPhase(
  phases: readonly PhaseState[],
  id: string,
  x: number,
  y: number,
  w: number,
  opts: { shape?: SpineNode["shape"]; label?: string } = {},
): SpineNode {
  const phase = findPhase(phases, id);
  const shape = opts.shape ?? "rect";
  return {
    id,
    label: opts.label ?? phase?.name ?? id,
    x,
    y,
    w,
    h: shape === "diamond" ? NODE_HEIGHT_DECISION : NODE_HEIGHT,
    shape,
    state: phase?.status ?? "queue",
    sub: phase?.sub,
    duration: phase?.durationMs,
  };
}

function PhaseNode({ node }: { node: SpineNode }) {
  const path =
    node.shape === "diamond"
      ? diamondPath(node.x, node.y, node.w, node.h)
      : rectPath(node.x, node.y, node.w, node.h, node.shape === "round" ? 18 : 4);
  return (
    <g
      className={`fc-node fc-node-${statusClass(node.state)}`}
      data-testid={`phase-${node.id}`}
    >
      <path d={path} className="fc-node-shape" />
      <text x={node.x + node.w / 2} y={node.y + node.h / 2 + 0.5} className="fc-node-label">
        {node.label}
      </text>
      {node.duration !== undefined && node.state === "done" ? (
        <text
          x={node.x + node.w / 2}
          y={node.y + node.h + 11}
          className="fc-node-time"
        >
          {Math.round(node.duration)}ms
        </text>
      ) : null}
    </g>
  );
}

function BranchTerminal({
  x,
  y,
  w,
  label,
  active,
  variant = "suppress",
}: {
  x: number;
  y: number;
  w: number;
  label: string;
  active: boolean;
  variant?: "suppress" | "regen";
}) {
  return (
    <g className={`fc-branch fc-branch-${variant}${active ? " active" : ""}`}>
      <path d={rectPath(x, y, w, 26, 3)} className="fc-branch-shape" />
      <text x={x + w / 2} y={y + 17} className="fc-branch-label">
        {label}
      </text>
    </g>
  );
}

function ForkLane({
  x,
  y,
  w,
  h,
  tag,
  desc,
  active,
  unchosen,
}: {
  x: number;
  y: number;
  w: number;
  h: number;
  tag: string;
  desc: string;
  active: boolean;
  unchosen: boolean;
}) {
  return (
    <g
      className={`fc-fork-lane${active ? " active" : ""}${unchosen ? " unchosen" : ""}`}
    >
      <path d={rectPath(x, y, w, h, 4)} className="fc-fork-shape" />
      <text x={x + w / 2} y={y + 14} className="fc-fork-tag">
        {tag}
      </text>
      <text x={x + w / 2} y={y + 26} className="fc-fork-desc">
        {desc}
      </text>
    </g>
  );
}

function Arrow({
  d,
  active = false,
  dashed = false,
  marker = "arrow",
}: {
  d: string;
  active?: boolean;
  dashed?: boolean;
  marker?: "arrow" | "arrow-bad" | "arrow-warn";
}) {
  return (
    <path
      d={d}
      className={`fc-arrow${active ? " active" : ""}${dashed ? " dashed" : ""}`}
      markerEnd={`url(#fc-${marker})`}
      fill="none"
    />
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
  // Spine node positions (viewBox 1000x520)
  const ROW1_Y = 40;
  const ROW2_Y = 215;
  const ROW3_Y = 380;
  const BRANCH_Y = 125;
  const GUARDS_BRANCH_Y = 465;
  const FORK_Y = 290;

  const input: SpineNode = {
    id: "input",
    label: "input",
    x: 14,
    y: ROW1_Y,
    w: 78,
    h: NODE_HEIGHT,
    shape: "round",
    state: "marker",
  };

  const row1 = [
    input,
    nodeFromPhase(phases, "ingest", 110, ROW1_Y, 100),
    nodeFromPhase(phases, "audience", 228, ROW1_Y, 110),
    nodeFromPhase(phases, "perception", 356, ROW1_Y, 110),
    nodeFromPhase(phases, "frame", 484, ROW1_Y, 110, { label: "frame gate" }),
    nodeFromPhase(phases, "extract", 612, ROW1_Y, 110, { label: "extraction" }),
    nodeFromPhase(phases, "closure_loop", 740, ROW1_Y - 3, 110, {
      shape: "diamond",
      label: "closure?",
    }),
    nodeFromPhase(phases, "generation_gate", 868, ROW1_Y - 3, 118, {
      shape: "diamond",
      label: "gen gate?",
    }),
  ];

  const row2 = [
    nodeFromPhase(phases, "retrieval", 14, ROW2_Y, 110),
    nodeFromPhase(phases, "ledger", 142, ROW2_Y, 110, { label: "ev. ledger" }),
    nodeFromPhase(phases, "shared", 270, ROW2_Y, 110, { label: "shared state" }),
    nodeFromPhase(phases, "delib", 398, ROW2_Y, 110, { label: "deliberation" }),
  ];

  const row3 = [
    nodeFromPhase(phases, "final", 332, ROW3_Y, 110, { label: "finalizer" }),
    nodeFromPhase(phases, "guards", 460, ROW3_Y - 3, 110, {
      shape: "diamond",
      label: "guards?",
    }),
    nodeFromPhase(phases, "persist", 588, ROW3_Y, 110),
    nodeFromPhase(phases, "reflect", 716, ROW3_Y, 110, { label: "reflection" }),
  ];

  const terminal: SpineNode = {
    id: "terminal",
    label: terminalLabel(terminalOutcome),
    x: 844,
    y: ROW3_Y,
    w: 140,
    h: NODE_HEIGHT,
    shape: "round",
    state: terminalOutcome === null ? "queue" : "done",
  };

  const closureNode = row1[6]!;
  const gateNode = row1[7]!;
  const guardsNode = row3[1]!;
  const finalNode = row3[0]!;
  const delibNode = row2[3]!;

  const closureTouched =
    findPhase(phases, "closure_loop")?.status !== "queue";
  const gateTouched =
    findPhase(phases, "generation_gate")?.status !== "queue";
  const guardsTouched = findPhase(phases, "guards")?.status !== "queue";
  const delibTouched = findPhase(phases, "delib")?.status !== "queue";

  const closureSuppressed = terminalOutcome === "suppressed_closure";
  const gateSuppressed = terminalOutcome === "suppressed_generation_gate";
  const guardsSuppressed = terminalOutcome === "suppressed_action";

  // Turn-around path: gen_gate → retrieval. Exits gen_gate's RIGHT vertex
  // so it doesn't collide with the gate_suppress terminal hanging below.
  // Routes along the right edge of the canvas, across the row gap (y=185,
  // below the suppress terminals at y=125-151), and back to retrieval.
  const gateRightX = gateNode.x + gateNode.w;
  const gateRightY = gateNode.y + gateNode.h / 2;
  const retrievalEntryX = row2[0]!.x + row2[0]!.w / 2;
  const retrievalEntryY = row2[0]!.y - 4;
  const TURN_LANE_X = 994;
  const TURN_LANE_Y = 185;
  const turnAroundPath = `M${gateRightX} ${gateRightY} H${TURN_LANE_X} V${TURN_LANE_Y} H${retrievalEntryX} V${retrievalEntryY}`;

  // delib → final: down past the fork pills, then left to final's top.
  // The horizontal segment sits just above row 3 (y=365) so the regen arc
  // below has room to live in the same strip without colliding with it.
  const delibCenterX = delibNode.x + delibNode.w / 2;
  const finalCenterX = finalNode.x + finalNode.w / 2;
  const FORK_TO_FINAL_Y = 365;
  const delibToFinal = `M${delibCenterX} ${delibNode.y + delibNode.h} V${FORK_TO_FINAL_Y} H${finalCenterX} V${finalNode.y - 4}`;

  // Regen loop: small arc from guards' TOP vertex back to final's TOP vertex.
  // The S1/S2 fork pills occupy y=290-324 in the gap between row 2 and row 3,
  // so the arc has to live in the strip BETWEEN the fork bottom and row 3
  // (y range ~325-378). Peak at y=348 keeps the arc inside that strip and
  // clear of the delib→final horizontal segment at y=365.
  const guardsCenterX = guardsNode.x + guardsNode.w / 2;
  const guardsTopY = guardsNode.y;
  const finalTopY = finalNode.y;
  const ARC_PEAK_Y = 348;
  const regenLoop = `M${guardsCenterX} ${guardsTopY} C${guardsCenterX} ${ARC_PEAK_Y}, ${finalCenterX} ${ARC_PEAK_Y}, ${finalCenterX} ${finalTopY - 4}`;
  const regenLabelX = (guardsCenterX + finalCenterX) / 2;
  const regenLabelY = ARC_PEAK_Y - 6;

  // Active stream
  const finalRunning = findPhase(phases, "final")?.status === "running";
  const delibRunning = findPhase(phases, "delib")?.status === "running";
  let activeStream: { phaseId: string; phaseName: string; status: NodeStatus; text: string } | null = null;
  if (finalRunning) {
    activeStream = {
      phaseId: "final",
      phaseName: "finalizer",
      status: "running",
      text: phaseTokenText("final", activeTurnId, tokenTextByPhase),
    };
  } else if (delibRunning) {
    activeStream = {
      phaseId: "delib",
      phaseName: "deliberation",
      status: "running",
      text: phaseTokenText("delib", activeTurnId, tokenTextByPhase),
    };
  } else {
    const lastWithText = [...phases]
      .reverse()
      .find(
        (p) =>
          STREAM_PHASES.has(p.id) &&
          phaseTokenText(p.id, activeTurnId, tokenTextByPhase).length > 0,
      );
    if (lastWithText) {
      activeStream = {
        phaseId: lastWithText.id,
        phaseName: lastWithText.name,
        status: lastWithText.status,
        text: phaseTokenText(lastWithText.id, activeTurnId, tokenTextByPhase),
      };
    }
  }

  const activeMeta =
    activeStream === null
      ? "no streaming phase"
      : activeStream.phaseId === "delib"
        ? delibPath === null
          ? "path pending"
          : delibPath === "system_2"
            ? "S2 · EmitTurnPlan"
            : "S1 · fast path"
        : `attempt ${finalAttempt}`;

  const outcomeLabel = terminalLabel(terminalOutcome);
  const outcomeTone = (() => {
    if (terminalOutcome === null) return "idle";
    if (terminalOutcome === "reflected") return "";
    if (terminalOutcome === "aborted") return "warn";
    return "bad";
  })();

  return (
    <div className="flow-shell">
      <div className="flow-topline">
        <div className="left">
          <span className="eyebrow">turn</span>
          <span className="turn-id">{activeTurnId ?? "idle"}</span>
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
          className="fc-svg"
          viewBox="0 0 1000 520"
          preserveAspectRatio="xMidYMid meet"
          role="img"
          aria-label="cognitive turn flow chart"
        >
          <defs>
            <marker
              id="fc-arrow"
              viewBox="0 0 10 10"
              refX="9"
              refY="5"
              markerWidth="6"
              markerHeight="6"
              orient="auto-start-reverse"
            >
              <path d="M0 0 L10 5 L0 10 z" className="fc-arrowhead" />
            </marker>
            <marker
              id="fc-arrow-bad"
              viewBox="0 0 10 10"
              refX="9"
              refY="5"
              markerWidth="6"
              markerHeight="6"
              orient="auto-start-reverse"
            >
              <path d="M0 0 L10 5 L0 10 z" className="fc-arrowhead bad" />
            </marker>
            <marker
              id="fc-arrow-warn"
              viewBox="0 0 10 10"
              refX="9"
              refY="5"
              markerWidth="6"
              markerHeight="6"
              orient="auto-start-reverse"
            >
              <path d="M0 0 L10 5 L0 10 z" className="fc-arrowhead warn" />
            </marker>
          </defs>

          {/* Row 1 spine arrows */}
          {row1.slice(0, -1).map((node, i) => {
            const next = row1[i + 1]!;
            const fromX = node.x + node.w;
            const fromY = node.y + node.h / 2;
            const toX = next.x;
            return <Arrow key={`r1-${i}`} d={`M${fromX} ${fromY} H${toX - 1}`} />;
          })}

          {/* Decision branches DOWN to suppression terminals */}
          <Arrow
            d={`M${closureNode.x + closureNode.w / 2} ${closureNode.y + closureNode.h} V${BRANCH_Y - 1}`}
            active={closureSuppressed}
            dashed
            marker={closureSuppressed ? "arrow-bad" : "arrow"}
          />
          <BranchTerminal
            x={closureNode.x + (closureNode.w - 110) / 2}
            y={BRANCH_Y}
            w={110}
            label="closure suppress"
            active={closureSuppressed}
          />

          <Arrow
            d={`M${gateNode.x + gateNode.w / 2} ${gateNode.y + gateNode.h} V${BRANCH_Y - 1}`}
            active={gateSuppressed}
            dashed
            marker={gateSuppressed ? "arrow-bad" : "arrow"}
          />
          <BranchTerminal
            x={gateNode.x + (gateNode.w - 110) / 2}
            y={BRANCH_Y}
            w={110}
            label="gate suppress"
            active={gateSuppressed}
          />

          {/* Turn-around: gen_gate → retrieval (curve down + left + down) */}
          <Arrow d={turnAroundPath} />

          {/* Row 2 spine arrows */}
          {row2.slice(0, -1).map((node, i) => {
            const next = row2[i + 1]!;
            return (
              <Arrow
                key={`r2-${i}`}
                d={`M${node.x + node.w} ${node.y + node.h / 2} H${next.x - 1}`}
              />
            );
          })}

          {/* delib → fork → final */}
          <Arrow d={delibToFinal} />

          {/* S1 / S2 fork mini-pills below delib (sit between row2 and row3) */}
          <ForkLane
            x={delibNode.x - 8}
            y={FORK_Y}
            w={50}
            h={34}
            tag="S1"
            desc="fast"
            active={delibTouched && delibPath === "system_1"}
            unchosen={delibTouched && delibPath !== null && delibPath !== "system_1"}
          />
          <ForkLane
            x={delibNode.x + delibNode.w - 42}
            y={FORK_Y}
            w={50}
            h={34}
            tag="S2"
            desc="plan"
            active={delibTouched && delibPath === "system_2"}
            unchosen={delibTouched && delibPath !== null && delibPath !== "system_2"}
          />

          {/* Row 3 spine arrows */}
          {row3.slice(0, -1).map((node, i) => {
            const next = row3[i + 1]!;
            return (
              <Arrow
                key={`r3-${i}`}
                d={`M${node.x + node.w} ${node.y + node.h / 2} H${next.x - 1}`}
              />
            );
          })}
          <Arrow
            d={`M${row3[row3.length - 1]!.x + row3[row3.length - 1]!.w} ${row3[row3.length - 1]!.y + row3[row3.length - 1]!.h / 2} H${terminal.x - 1}`}
          />

          {/* Guards → guards_trip suppression terminal */}
          <Arrow
            d={`M${guardsNode.x + guardsNode.w / 2} ${guardsNode.y + guardsNode.h} V${GUARDS_BRANCH_Y - 1}`}
            active={guardsSuppressed}
            dashed
            marker={guardsSuppressed ? "arrow-bad" : "arrow"}
          />
          <BranchTerminal
            x={guardsNode.x + (guardsNode.w - 110) / 2}
            y={GUARDS_BRANCH_Y}
            w={110}
            label="guards trip"
            active={guardsSuppressed}
          />

          {/* Regen loop: guards → final */}
          <Arrow
            d={regenLoop}
            active={finalAttempt > 1}
            dashed
            marker={finalAttempt > 1 ? "arrow-warn" : "arrow"}
          />
          <text
            x={regenLabelX}
            y={regenLabelY}
            className={`fc-regen-label${finalAttempt > 1 ? " active" : ""}`}
          >
            regen ↻
          </text>

          {/* Nodes (drawn on top of arrows) */}
          {row1.map((n) => (
            <PhaseNode key={n.id} node={n} />
          ))}
          {row2.map((n) => (
            <PhaseNode key={n.id} node={n} />
          ))}
          {row3.map((n) => (
            <PhaseNode key={n.id} node={n} />
          ))}
          <PhaseNode node={terminal} />
        </svg>
      </div>

      <div className={`flow-active-stream ${activeStream?.status ?? "idle"}`}>
        <div className="flow-active-head">
          <span>
            active stream
            {activeStream === null ? null : (
              <>
                {" · "}
                <strong>{activeStream.phaseName}</strong>
              </>
            )}
          </span>
          <span className="dim">{activeMeta}</span>
        </div>
        <pre
          className={`flow-active-body ${activeStream === null ? "empty" : ""}${
            activeStream?.status === "done" ? " muted" : ""
          }`}
        >
          {activeStream === null
            ? "waiting for delib or final to produce tokens"
            : activeStream.text.length > 0
              ? activeStream.text
              : "stream open..."}
        </pre>
      </div>
    </div>
  );
}
