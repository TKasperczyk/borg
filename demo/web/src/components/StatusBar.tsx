import type { MaintenanceTickFrame, StateSnapshot, WsState } from "../api/types";
import { wsLabel, wsToneClass } from "./Topbar";

export type StatusBarProps = {
  state: StateSnapshot | null;
  wsState: WsState;
};

export function moodLabel(state: StateSnapshot | null): string {
  if (state === null) {
    return "neutral";
  }

  const valence = state.current_mood.valence.toFixed(2);
  const arousal = state.current_mood.arousal.toFixed(2);
  return `v ${valence} · a ${arousal}`;
}

export function countValue(count: number | undefined): string {
  return count === undefined ? "—" : count.toString();
}

export function maintenanceTickLabel(
  frame: MaintenanceTickFrame | null | undefined,
): string | null {
  if (frame === null || frame === undefined) {
    return null;
  }

  return `${frame.cadence} ${frame.processes.length}p ${frame.changes}chg`;
}

export function maintenanceTickTone(frame: MaintenanceTickFrame): "ok" | "warn" | "bad" {
  if (frame.status === "error" || frame.errors > 0) {
    return "bad";
  }

  return frame.changed ? "ok" : "warn";
}

function embeddingLabel(state: StateSnapshot | null): string {
  const embedding = state?.runtime?.embedding;
  if (embedding === undefined || embedding.model === null) {
    return "—";
  }
  if (embedding.dims === null) {
    return embedding.model;
  }
  return `${embedding.model} · ${embedding.dims}d`;
}

export function StatusBar({ state, wsState }: StatusBarProps) {
  return (
    <div className="statusbar">
      <span className="seg">
        <span className={wsState === "live" ? "live-dot" : `dot ${wsToneClass(wsState)}`}></span>
        <span className="k">ws</span>
        <span className={`v ${wsToneClass(wsState)}`}>{wsLabel(wsState)}</span>
      </span>
      <span className="seg">
        <span className="k">ver</span>
        <span className="v">{state?.version ?? "—"}</span>
      </span>
      <span className="seg grow"></span>
      <span className="seg">
        <span className="k">model</span>
        <span className="v">{state?.runtime?.model ?? "—"}</span>
      </span>
      <span className="seg">
        <span className="k">emb</span>
        <span className="v">{embeddingLabel(state)}</span>
      </span>
    </div>
  );
}
