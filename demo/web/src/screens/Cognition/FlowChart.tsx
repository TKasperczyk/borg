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
  if (status === "running") return "●";
  if (status === "done") return "✓";
  if (status === "fail") return "×";
  return "—";
}

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

function isTouched(status: PhaseState["status"]): boolean {
  return status !== "queue";
}

function terminalLabel(outcome: TurnTerminalOutcome | null): string {
  return outcome === null ? "waiting" : outcome;
}

function suppressionFor(
  phaseId: string,
  outcome: TurnTerminalOutcome | null,
): { label: string; active: boolean } | null {
  if (phaseId === "closure_loop") {
    return { label: "→ closure suppress", active: outcome === "suppressed_closure" };
  }
  if (phaseId === "generation_gate") {
    return { label: "→ gate suppress", active: outcome === "suppressed_generation_gate" };
  }
  if (phaseId === "guards") {
    return { label: "→ guards trip", active: outcome === "suppressed_action" };
  }
  return null;
}

function PhasePill({
  phase,
  delibPath,
  finalAttempt,
  terminalOutcome,
}: {
  phase: PhaseState;
  delibPath: "system_1" | "system_2" | null;
  finalAttempt: number;
  terminalOutcome: TurnTerminalOutcome | null;
}) {
  const touched = isTouched(phase.status);
  const suppression = suppressionFor(phase.id, terminalOutcome);

  return (
    <div
      className={`flow-pill ${phase.status}${touched ? " touched" : ""}`}
      data-testid={`phase-${phase.id}`}
    >
      <div className="flow-pill-head">
        <span className="flow-pill-glyph" aria-hidden="true">
          {phaseGlyph(phase.status)}
        </span>
        <span className="flow-pill-name">{phase.name}</span>
        <span className="flow-pill-time">
          {phase.durationMs === undefined ? "" : `${Math.round(phase.durationMs)}ms`}
        </span>
      </div>

      {phase.id === "delib" ? (
        <div className="flow-pill-lanes">
          <span
            className={`flow-mini-lane${touched && delibPath === "system_1" ? " active" : ""}${
              touched && delibPath !== null && delibPath !== "system_1" ? " unchosen" : ""
            }`}
            title="System 1: ledger sufficient, no LLM planning"
          >
            S1
          </span>
          <span
            className={`flow-mini-lane${touched && delibPath === "system_2" ? " active" : ""}${
              touched && delibPath !== null && delibPath !== "system_2" ? " unchosen" : ""
            }`}
            title="System 2: EmitTurnPlan before answer"
          >
            S2
          </span>
        </div>
      ) : phase.id === "final" && finalAttempt > 1 ? (
        <div className="flow-pill-sub">
          <span className="flow-attempt-badge" title="finalizer re-invoked after a guard trip">
            attempt {finalAttempt}
          </span>
        </div>
      ) : (
        <div className="flow-pill-sub">
          <span className="flow-pill-sub-text">{phase.sub === "waiting" ? "queued" : phase.sub}</span>
        </div>
      )}

      {suppression !== null ? (
        <div className={`flow-pill-branch${suppression.active ? " active" : ""}`}>
          {suppression.label}
        </div>
      ) : null}
    </div>
  );
}

function ActiveStreamPane({
  phases,
  activeTurnId,
  tokenTextByPhase,
  delibPath,
  finalAttempt,
}: {
  phases: readonly PhaseState[];
  activeTurnId: string | null;
  tokenTextByPhase: Map<string, string>;
  delibPath: "system_1" | "system_2" | null;
  finalAttempt: number;
}) {
  // Pick the streaming phase to show — preference: final if running, then delib
  // if running, else the most recent streaming phase that has any text.
  const finalPhase = phases.find((p) => p.id === "final");
  const delibPhase = phases.find((p) => p.id === "delib");

  let activePhase: PhaseState | undefined;
  if (finalPhase?.status === "running") {
    activePhase = finalPhase;
  } else if (delibPhase?.status === "running") {
    activePhase = delibPhase;
  } else {
    activePhase = phases
      .filter(
        (p) =>
          STREAM_PHASES.has(p.id) &&
          phaseTokenText(p.id, activeTurnId, tokenTextByPhase).length > 0,
      )
      .pop();
  }

  if (activePhase === undefined) {
    return (
      <div className="flow-active-stream idle">
        <div className="flow-active-head">
          <span>active stream</span>
          <span className="dim">no streaming phase</span>
        </div>
        <div className="flow-active-body empty">
          waiting for delib or final to produce tokens
        </div>
      </div>
    );
  }

  const text = phaseTokenText(activePhase.id, activeTurnId, tokenTextByPhase);
  const status = activePhase.status;
  const meta =
    activePhase.id === "delib"
      ? delibPath === null
        ? "path pending"
        : delibPath === "system_2"
          ? "S2 · EmitTurnPlan"
          : "S1 · fast path"
      : `attempt ${finalAttempt}`;

  return (
    <div className={`flow-active-stream ${status}`}>
      <div className="flow-active-head">
        <span>
          active stream · <strong>{activePhase.name}</strong>
        </span>
        <span className="dim">{meta}</span>
      </div>
      <pre className={`flow-active-body ${status === "done" ? "muted" : ""}`}>
        {text.length > 0 ? text : "stream open..."}
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
  return (
    <div className="flow-shell">
      <div className="flow-topline">
        <span>turn {activeTurnId ?? "idle"}</span>
        <span className="flow-topline-status">{terminalLabel(terminalOutcome)}</span>
      </div>

      <div className="flow-pipeline-wrap">
        <div className="flow-pipeline">
          <div className="flow-pill input-pill" data-testid="phase-input">
            <div className="flow-pill-head">
              <span className="flow-pill-glyph" aria-hidden="true">
                ▸
              </span>
              <span className="flow-pill-name">input</span>
            </div>
          </div>

          {phases.map((phase) => (
            <PhasePill
              key={phase.id}
              phase={phase}
              delibPath={delibPath}
              finalAttempt={finalAttempt}
              terminalOutcome={terminalOutcome}
            />
          ))}

          <div
            className={`flow-pill terminal-pill${terminalOutcome === null ? "" : " active touched"}`}
            data-testid="phase-terminal"
          >
            <div className="flow-pill-head">
              <span className="flow-pill-glyph" aria-hidden="true">
                ⊙
              </span>
              <span className="flow-pill-name">terminal</span>
            </div>
            <div className="flow-pill-sub">
              <span className="flow-pill-sub-text">{terminalLabel(terminalOutcome)}</span>
            </div>
          </div>
        </div>
      </div>

      <ActiveStreamPane
        phases={phases}
        activeTurnId={activeTurnId}
        tokenTextByPhase={tokenTextByPhase}
        delibPath={delibPath}
        finalAttempt={finalAttempt}
      />
    </div>
  );
}
