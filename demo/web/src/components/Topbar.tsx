import { apiBase } from "../api/client";
import type { WsState } from "../api/types";

export type TopbarProps = {
  session_id: string;
  audience: string;
  turns: number;
  ws_state: WsState;
  now: string;
};

function wsLabel(state: WsState): string {
  if (state === "live") {
    return "live";
  }
  if (state === "reconnecting") {
    return "reconnect";
  }
  return "down";
}

export function Topbar({ session_id, audience, turns, ws_state, now }: TopbarProps) {
  return (
    <div className="topbar">
      <div className="topbar-group">
        <span className="topbar-key">session</span>
        <span className="topbar-val">{session_id}</span>
        <span className="topbar-sep">·</span>
        <span className="topbar-key">audience</span>
        <span className="topbar-val acc">{audience}</span>
        <span className="topbar-sep">·</span>
        <span className="topbar-key">turns</span>
        <span className="topbar-val tab-num">{turns}</span>
      </div>
      <div className="topbar-group">
        <span className={ws_state === "live" ? "live-dot" : "dot warn"}></span>
        <span className={ws_state === "live" ? "acc upper" : "warn upper"} style={{ fontSize: "10.5px" }}>
          {wsLabel(ws_state)}
        </span>
        <span className="topbar-sep">·</span>
        <span className="topbar-key">api</span>
        <span className="topbar-val">{apiBase().replace(/^https?:\/\//, "")}</span>
      </div>
      <div className="topbar-group flex"></div>
      <div className="topbar-group end">
        <span className="kbd">⌘K</span>
        <span className="topbar-key">command</span>
        <span className="topbar-sep">·</span>
        <span className="topbar-val tab-num">{now}</span>
      </div>
    </div>
  );
}
