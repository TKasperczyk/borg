import type { WsState } from "../api/types";
import { ResetButton } from "./ResetButton";

export type TopbarProps = {
  session_id: string;
  audience: string;
  turns: number;
  ws_state: WsState;
  now: string;
  route?: string;
};

export function wsLabel(state: WsState): string {
  if (state === "live") {
    return "live";
  }
  if (state === "reconnecting") {
    return "reconnect";
  }
  return "down";
}

export function wsToneClass(state: WsState): string {
  if (state === "live") {
    return "acc";
  }
  if (state === "reconnecting") {
    return "warn";
  }
  return "bad";
}

export function formatTurns(turns: number): string {
  return turns.toString().padStart(3, "0");
}

export function Topbar({
  session_id,
  audience,
  turns,
  ws_state,
  now,
  route = "cognition",
}: TopbarProps) {
  return (
    <div className="topbar">
      <div className="topbar-brand">
        <span>
          <span className="accent">[</span>borg<span className="accent">]</span>
        </span>
      </div>
      <div className="topbar-crumb">
        <span className="seg">console</span>
        <span className="sep">›</span>
        <span className="seg">{session_id}</span>
        <span className="sep">›</span>
        <span className="seg here">{route}</span>
      </div>
      <div className="topbar-spacer"></div>
      <div className="topbar-pills">
        <div className="topbar-pill">
          <span className="k">audience</span>
          <span className="v">{audience}</span>
        </div>
        <div className="topbar-pill">
          <span className="k">turn</span>
          <span className="v">{formatTurns(turns)}</span>
        </div>
        <div className="topbar-pill">
          <span className="k">ws</span>
          <span className={`v ${wsToneClass(ws_state)}`}>{wsLabel(ws_state)}</span>
        </div>
        <div className="topbar-pill">
          <span className="k">utc</span>
          <span className="v">{now}</span>
        </div>
        <span className="topbar-live" aria-hidden="true">
          <span className={ws_state === "live" ? "live-dot" : "dot warn"}></span>
        </span>
      </div>
      <div className="topbar-end">
        <ResetButton />
      </div>
    </div>
  );
}
