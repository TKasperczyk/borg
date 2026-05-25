import type { PhaseState } from "../../hooks/use-turn-stream";

export type PhasesViewProps = {
  phases: readonly PhaseState[];
  activeTurnId: string | null;
};

function phaseGlyph(status: PhaseState["status"]): string {
  if (status === "running") {
    return "●";
  }
  if (status === "done") {
    return "✓";
  }
  if (status === "fail") {
    return "✕";
  }
  return "○";
}

function phaseDetail(phase: PhaseState): { substrate: string; detail: string; degraded: string } {
  return {
    substrate: "turn trace",
    detail: phase.sub === "waiting" ? "pending" : phase.sub,
    degraded: phase.status === "fail" ? "yes" : "no",
  };
}

export function PhasesView({ phases, activeTurnId }: PhasesViewProps) {
  const activeIdx = phases.findIndex((phase) => phase.status === "running");
  const elapsed = phases.reduce((sum, phase) => sum + (phase.durationMs ?? 0), 0);

  return (
    <div>
      <div style={{ padding: "14px 14px 6px 14px", borderBottom: "1px solid var(--line-soft)" }}>
        <div
          style={{
            display: "flex",
            justifyContent: "space-between",
            marginBottom: 6,
            fontSize: 10.5,
            color: "var(--text-mute)",
            textTransform: "uppercase",
            letterSpacing: "0.08em",
          }}
        >
          <span>turn {activeTurnId ?? "idle"}</span>
          <span>live trace</span>
        </div>
        <div style={{ display: "flex", gap: 10, fontSize: 10.5, color: "var(--text-mute)" }}>
          <span>
            phases{" "}
            <span className="acc">{phases.filter((phase) => phase.status === "done").length}</span>/
            {phases.length}
          </span>
          <span className="dim">·</span>
          <span>
            elapsed{" "}
            <span className="tab-num" style={{ color: "var(--text)" }}>
              {Math.round(elapsed)}ms
            </span>
          </span>
        </div>
      </div>
      <div className="phases">
        {phases.map((phase, index) => {
          const detail = phaseDetail(phase);
          return (
            <div key={phase.id}>
              <div className={`phase ${phase.status}`}>
                <div className="glyph">{phaseGlyph(phase.status)}</div>
                <div>
                  <div className="name">{phase.name}</div>
                  <div className="sub">{phase.sub}</div>
                </div>
                <div className="timing">
                  {phase.status === "done" || phase.status === "running" || phase.status === "fail"
                    ? `${Math.round(phase.durationMs ?? 0)}ms`
                    : "—"}
                </div>
              </div>
              {index === activeIdx ? (
                <div className="phase-detail">
                  <div className="row">
                    <span className="k">substrate</span>
                    <span className="v">{detail.substrate}</span>
                  </div>
                  <div className="row">
                    <span className="k">detail</span>
                    <span className="v">{detail.detail}</span>
                  </div>
                  <div className="row">
                    <span className="k">degraded</span>
                    <span className="v">{detail.degraded}</span>
                  </div>
                </div>
              ) : null}
            </div>
          );
        })}
      </div>
    </div>
  );
}
