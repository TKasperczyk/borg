import type { TurnTerminalOutcome } from "../../api/types";
import type { PhaseState } from "../../hooks/use-turn-stream";

export type FlowChartProps = {
  phases: readonly PhaseState[];
  activeTurnId: string | null;
  tokenTextByPhase: Map<string, string>;
  terminalOutcome: TurnTerminalOutcome | null;
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

function PhaseNode({
  phase,
  activeTurnId,
  tokenTextByPhase,
}: {
  phase: PhaseState;
  activeTurnId: string | null;
  tokenTextByPhase: Map<string, string>;
}) {
  const tokenText = phaseTokenText(phase, activeTurnId, tokenTextByPhase);
  const showTokenBlock =
    STREAM_PHASES.has(phase.id) && (phase.status === "running" || tokenText.length > 0);

  return (
    <div className={`flow-phase ${phase.status}`} data-testid={`phase-${phase.id}`}>
      <div className="flow-phase-head">
        <div className="flow-status" aria-hidden="true">
          {phaseGlyph(phase.status)}
        </div>
        <div className="flow-copy">
          <div className="flow-name">{phase.name}</div>
          <div className="flow-sub">{phase.sub === "waiting" ? "queued" : phase.sub}</div>
        </div>
        <div className="flow-time">
          {phase.durationMs === undefined ? "—" : `${Math.round(phase.durationMs)}ms`}
        </div>
      </div>
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
            />

            {phase.id === "frame" ? (
              <div className={branchClass(closureSuppressed)}>
                <span className="flow-branch-line">╺╴</span>
                <span>anomaly/closure terminal</span>
              </div>
            ) : null}

            {phase.id === "delib" ? (
              <div className="flow-gate">
                <div className="flow-arrow">↓</div>
                <div className={`flow-gate-node${gateSuppressed ? " active" : ""}`}>
                  generation gate
                </div>
                <div className={branchClass(gateSuppressed)}>
                  <span className="flow-branch-line">╺╴</span>
                  <span>suppression terminal</span>
                </div>
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
