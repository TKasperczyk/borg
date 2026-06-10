import type { EntityRecord, SessionRecord, StateSnapshot } from "../api/types";
import type { DreamActivity } from "../hooks/use-live-cache";
import type { AudienceDisplayIdentity } from "../lib/audience-identity";
import type { RouteId } from "../routes";
import { IdChip } from "./Inspector/IdChip";
import { MiniOrrery } from "./orrery/MiniOrrery";
import { moodLabel } from "./StatusBar";
import { formatTurns } from "./Topbar";

export type InstrumentStripProps = {
  sessionId: string;
  activeSession: SessionRecord | null;
  audienceDisplay: AudienceDisplayIdentity;
  creator: EntityRecord | null;
  state: StateSnapshot | null;
  dreamActivity: DreamActivity | null;
  route: RouteId;
  onOpenPalette?: () => void;
  onOpenHelp?: () => void;
};

function SessionCrumb({
  sessionId,
  activeSession,
}: Pick<InstrumentStripProps, "sessionId" | "activeSession">) {
  return (
    <span className="seg session-crumb">
      <span>{activeSession?.label ?? "unknown session"}</span>
      <IdChip id={sessionId} type="session" />
    </span>
  );
}

function AudiencePillValue({
  sessionId,
  activeSession,
  audienceDisplay,
}: Pick<InstrumentStripProps, "sessionId" | "activeSession" | "audienceDisplay">) {
  const fallbackId = audienceDisplay.fallbackId ?? sessionId;
  const entityId = audienceDisplay.entityId;

  return (
    <span className="identity-inline">
      <span>{audienceDisplay.label ?? "unknown"}</span>
      {entityId !== null ? (
        <IdChip id={entityId} type="entity" />
      ) : (
        <IdChip id={fallbackId} type={activeSession === null ? "session" : null} />
      )}
    </span>
  );
}

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
  audienceDisplay,
  creator,
  state,
  dreamActivity,
  route,
  onOpenPalette,
  onOpenHelp,
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
        <SessionCrumb sessionId={sessionId} activeSession={activeSession} />
        <span className="sep">›</span>
        <span className="seg here">{route}</span>
      </div>
      <div className="topbar-spacer"></div>
      <div className="topbar-pills">
        <div className="topbar-pill">
          <span className="k">audience</span>
          <span className="v">
            <AudiencePillValue
              sessionId={sessionId}
              activeSession={activeSession}
              audienceDisplay={audienceDisplay}
            />
          </span>
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
        <MiniOrrery dreamRunning={dreamActivity !== null} />
        <div className="topbar-pill">
          <span className="k">mood</span>
          <span className="v">
            <span className="instrument-mood-glyph">{moodGlyph(state)}</span> {moodLabel(state)}
          </span>
        </div>
        <div className="topbar-pill">
          <span className="k">turn</span>
          <span className="v">{formatTurns(state?.counts.turns ?? 0)}</span>
        </div>
        {onOpenPalette === undefined ? null : (
          <button
            type="button"
            className="topbar-shortcut-chip"
            onClick={onOpenPalette}
            aria-label="open command palette"
          >
            ctrl+K
          </button>
        )}
        {onOpenHelp === undefined ? null : (
          <button
            type="button"
            className="topbar-shortcut-chip icon"
            onClick={onOpenHelp}
            aria-label="open shortcut legend"
          >
            ?
          </button>
        )}
      </div>
      <div className="topbar-end"></div>
    </div>
  );
}
