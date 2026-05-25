import type { StateSnapshot } from "../api/types";

export type StatusBarProps = {
  state: StateSnapshot | null;
  lastPhase?: string;
};

function moodLabel(state: StateSnapshot | null): string {
  if (state === null) {
    return "neutral";
  }

  const valence = state.current_mood.valence.toFixed(2);
  const arousal = state.current_mood.arousal.toFixed(2);
  return `v ${valence} · a ${arousal}`;
}

export function StatusBar({ state, lastPhase }: StatusBarProps) {
  return (
    <div className="statusbar">
      <span>
        <span className="ok">●</span> borg/main
      </span>
      <span className="sep">│</span>
      <span>
        dream <span className="dim">{state === null ? "—" : `${state.counts.dream_audit_rows} audit`}</span>
      </span>
      <span className="sep">│</span>
      <span>
        mood <span className="dim">{moodLabel(state)}</span>
      </span>
      <span className="sep">│</span>
      <span>
        review <span className="dim">{state?.counts.open_qs ?? "—"}</span>
      </span>
      <span className="sep">│</span>
      <span>
        commit <span className="dim">{state?.counts.commitments ?? "—"}</span>
      </span>
      <span className="sep">│</span>
      <span>
        last <span className="dim">{lastPhase ?? "idle"}</span>
      </span>
      <span className="grow"></span>
      <span className="dim">opus-4.7</span>
      <span className="sep">·</span>
      <span className="dim">qwen3-8b · 4096d</span>
    </div>
  );
}
