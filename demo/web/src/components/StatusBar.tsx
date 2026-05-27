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

function countValue(count: number | undefined): string {
  return count === undefined ? "—" : count.toString();
}

export function StatusBar({ state, lastPhase }: StatusBarProps) {
  return (
    <div className="statusbar">
      <span className="seg">
        <span className="v ok">●</span>
        <span className="k">branch</span>
        <span className="v">borg/main</span>
      </span>
      <span className="seg">
        <span className="k">dream</span>
        <span className="v">
          {state === null ? "—" : `${state.counts.dream_audit_rows} audit`}
        </span>
      </span>
      <span className="seg">
        <span className="k">mood</span>
        <span className="v">{moodLabel(state)}</span>
      </span>
      <span className="seg">
        <span className="k">review</span>
        <span className="v">{countValue(state?.counts.open_qs)}</span>
      </span>
      <span className="seg">
        <span className="k">commit</span>
        <span className="v">{countValue(state?.counts.commitments)}</span>
      </span>
      <span className="seg">
        <span className="k">last</span>
        <span className="v">{lastPhase ?? "idle"}</span>
      </span>
      <span className="seg grow"></span>
      <span className="seg">
        <span className="k">model</span>
        <span className="v">opus-4.7</span>
      </span>
      <span className="seg">
        <span className="k">emb</span>
        <span className="v">qwen3-8b · 4096d</span>
      </span>
    </div>
  );
}
