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

function phaseGlyph(status: PhaseState["status"]): string {
  if (status === "running") {
    return "●";
  }
  if (status === "done") {
    return "✓";
  }
  if (status === "fail") {
    return "×";
  }
  return "—";
}

function tokenKey(turnId: string, phase: string): string {
  return `${turnId}:${phase}`;
}

function phaseTokenText(
  phase: PhaseState,
  activeTurnId: string | null,
  tokenTextByPhase: Map<string, string>,
): string {
  if (activeTurnId === null) {
    return "";
  }

  return tokenTextByPhase.get(tokenKey(activeTurnId, phase.id)) ?? "";
}

function branchClass(active: boolean): string {
  return `flow-branch${active ? " active" : ""}`;
}

function terminalLabel(outcome: TurnTerminalOutcome | null): string {
  return outcome === null ? "waiting" : outcome;
}

// True once a phase has fired anything (running or terminal); used to gate the
// S1/S2 lanes so they don't visually 'choose' until delib actually starts.
function isPhaseTouched(status: PhaseState["status"]): boolean {
  return status !== "queue";
}

function PhaseNode({
  phase,
  activeTurnId,
  tokenTextByPhase,
  delibPath,
  finalAttempt,
}: {
  phase: PhaseState;
  activeTurnId: string | null;
  tokenTextByPhase: Map<string, string>;
  delibPath: "system_1" | "system_2" | null;
  finalAttempt: number;
}) {
  const tokenText = phaseTokenText(phase, activeTurnId, tokenTextByPhase);
  const showTokenBlock =
    STREAM_PHASES.has(phase.id) && (phase.status === "running" || tokenText.length > 0);
  const touched = isPhaseTouched(phase.status);

  // delib renders two horizontal lanes (S1 / S2). Once delib is touched and we
  // know the path, the matching lane lights up; the other stays dim.
  const renderDelibLanes = phase.id === "delib";
  const renderFinalAttempt = phase.id === "final" && finalAttempt > 1;

  return (
    <div className={`flow-phase ${phase.status}`} data-testid={`phase-${phase.id}`}>
      <div className="flow-phase-head">
        <div className="flow-status" aria-hidden="true">
          {phaseGlyph(phase.status)}
        </div>
        <div className="flow-copy">
          <div className="flow-name">
            {phase.name}
            {renderFinalAttempt ? (
              <span
                className="flow-attempt-badge"
                title="finalizer re-invoked after a commitment-guard regeneration"
              >
                attempt {finalAttempt}
              </span>
            ) : null}
          </div>
          <div className="flow-sub">{phase.sub === "waiting" ? "queued" : phase.sub}</div>
        </div>
        <div className="flow-time">
          {phase.durationMs === undefined ? "—" : `${Math.round(phase.durationMs)}ms`}
        </div>
      </div>
      {renderDelibLanes ? (
        <div className="flow-lanes">
          <div
            className={`flow-lane${touched && delibPath === "system_1" ? " active" : ""}${
              touched && delibPath !== null && delibPath !== "system_1" ? " unchosen" : ""
            }`}
          >
            <span className="flow-lane-tag">S1</span>
            <span className="flow-lane-desc">fast path · ledger sufficient</span>
          </div>
          <div
            className={`flow-lane${touched && delibPath === "system_2" ? " active" : ""}${
              touched && delibPath !== null && delibPath !== "system_2" ? " unchosen" : ""
            }`}
          >
            <span className="flow-lane-tag">S2</span>
            <span className="flow-lane-desc">EmitTurnPlan · reasoning before answer</span>
          </div>
        </div>
      ) : null}
      {showTokenBlock ? (
        <pre className={`flow-token ${phase.status === "done" ? "muted" : ""}`}>
          {tokenText.length > 0 ? tokenText : "stream open..."}
        </pre>
      ) : null}
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
  const gateSuppressed = terminalOutcome === "suppressed_generation_gate";
  const guardsSuppressed = terminalOutcome === "suppressed_action";
  const closureSuppressed = terminalOutcome === "suppressed_closure";

  return (
    <div className="flow-shell">
      <div className="flow-topline">
        <span>turn {activeTurnId ?? "idle"}</span>
        <span className="flow-topline-status">{terminalLabel(terminalOutcome)}</span>
      </div>

      <div className="flow-canvas">
        <div className="flow-input">input message</div>
        <div className="flow-arrow">↓</div>

        {phases.map((phase) => (
          <div key={phase.id} className="flow-step">
            <PhaseNode
              phase={phase}
              activeTurnId={activeTurnId}
              tokenTextByPhase={tokenTextByPhase}
              delibPath={delibPath}
              finalAttempt={finalAttempt}
            />

            {phase.id === "closure_loop" ? (
              <div className={branchClass(closureSuppressed)}>
                <span className="flow-branch-line">╺╴</span>
                <span>closure-loop suppression terminal</span>
              </div>
            ) : null}

            {phase.id === "generation_gate" ? (
              <div className={branchClass(gateSuppressed)}>
                <span className="flow-branch-line">╺╴</span>
                <span>generation-gate suppression terminal</span>
              </div>
            ) : null}

            {phase.id === "guards" ? (
              <div className={branchClass(guardsSuppressed)}>
                <span className="flow-branch-line">╺╴</span>
                <span>guards-trip terminal</span>
              </div>
            ) : null}

            {phase.id !== phases[phases.length - 1]?.id ? (
              <div className="flow-arrow">↓</div>
            ) : null}
          </div>
        ))}

        <div className="flow-arrow">↓</div>
        <div className={`flow-terminal${terminalOutcome === null ? "" : " active"}`}>
          <span>terminal</span>
          <span>{terminalLabel(terminalOutcome)}</span>
        </div>
      </div>
    </div>
  );
}
