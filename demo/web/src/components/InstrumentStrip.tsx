import type { EntityRecord, SessionRecord, StateSnapshot, WsState } from "../api/types";
import type { RouteId } from "../routes";
import { ResetButton } from "./ResetButton";
import { countValue, moodLabel } from "./StatusBar";
import { formatTurns, wsLabel, wsToneClass } from "./Topbar";

export type InstrumentStripProps = {
  sessionId: string;
  activeSession: SessionRecord | null;
  audience: string;
  creator: EntityRecord | null;
  state: StateSnapshot | null;
  wsState: WsState;
  now: string;
  route: RouteId;
};

function moodGlyph(state: StateSnapshot | null): string {
  if (state === null) {
    return "·";
  }
  if (state.current_mood.valence > 0.15) {
    return "+";
  }
  if (state.current_mood.valence < -0.15) {
    return "-";
  }
  return "·";
}

export function InstrumentStrip({
  sessionId,
  activeSession,
  audience,
  creator,
  state,
  wsState,
  now,
  route,
}: InstrumentStripProps) {
  return (
    <div className="topbar instrument-strip">
      <div className="topbar-brand">
        <span>
          <span className="accent">[</span>borg<span className="accent">]</span>
        </span>
      </div>
      <div className="topbar-crumb">
        <span className="seg">console</span>
        <span className="sep">›</span>
        <span className="seg">{sessionId}</span>
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
          <span className="k">role</span>
          <span className="v">{activeSession?.audience_role ?? "—"}</span>
        </div>
        <div className="topbar-pill">
          <span className="k">policy</span>
          <span className="v">{activeSession?.participation_policy ?? "—"}</span>
        </div>
        <div className="topbar-pill">
          <span className="k">creator</span>
          <span className="v">{creator?.canonical_name ?? "unset"}</span>
        </div>
        <div className="topbar-pill">
          <span className="k">ws</span>
          <span className={`v ${wsToneClass(wsState)}`}>{wsLabel(wsState)}</span>
        </div>
        <div className="topbar-pill">
          <span className="k">mood</span>
          <span className="v">
            <span className="instrument-mood-glyph">{moodGlyph(state)}</span> {moodLabel(state)}
          </span>
        </div>
        <div className="topbar-pill">
          <span className="k">counts</span>
          <span className="v">
            r {countValue(state?.counts.open_reviews)} · q {countValue(state?.counts.open_qs)} · c{" "}
            {countValue(state?.counts.commitments)}
          </span>
        </div>
        <div className="topbar-pill">
          <span className="k">turn</span>
          <span className="v">{formatTurns(state?.counts.turns ?? 0)}</span>
        </div>
        <div className="topbar-pill">
          <span className="k">ver</span>
          <span className="v">{state?.version ?? "—"}</span>
        </div>
        <div className="topbar-pill">
          <span className="k">utc</span>
          <span className="v">{now}</span>
        </div>
        <span className="topbar-live" aria-hidden="true">
          <span className={wsState === "live" ? "live-dot" : "dot warn"}></span>
        </span>
      </div>
      <div className="topbar-end">
        <ResetButton />
      </div>
    </div>
  );
}
