import type { SessionRecord } from "../api/types";

export type SessionsSidebarProps = {
  sessions: readonly SessionRecord[];
  activeSessionId: string;
  onSelect: (sessionId: string) => void;
};

function relativeTime(ts: number): string {
  const diffMs = Date.now() - ts;
  if (!Number.isFinite(diffMs) || diffMs < 0) {
    return "now";
  }

  const diffSeconds = Math.floor(diffMs / 1_000);
  if (diffSeconds < 60) {
    return "now";
  }

  const diffMinutes = Math.floor(diffSeconds / 60);
  if (diffMinutes < 60) {
    return `${diffMinutes}m`;
  }

  const diffHours = Math.floor(diffMinutes / 60);
  if (diffHours < 24) {
    return `${diffHours}h`;
  }

  return `${Math.floor(diffHours / 24)}d`;
}

function sourceLabel(session: SessionRecord): string {
  if (session.source_type === "demo") {
    return session.conversation_kind;
  }
  return session.source_type;
}

export function SessionsSidebar({
  sessions,
  activeSessionId,
  onSelect,
}: SessionsSidebarProps) {
  return (
    <aside className="sessions-sidebar" aria-label="sessions">
      <div className="sessions-head">
        <span>sessions</span>
        <span className="count">{sessions.length}</span>
      </div>
      <div className="sessions-list">
        {sessions.map((session) => {
          const active = session.session_id === activeSessionId;
          return (
            <button
              key={session.session_id}
              type="button"
              className={`session-row ${active ? "active" : ""}`}
              onClick={() => onSelect(session.session_id)}
            >
              <span className="session-row-top">
                <span className="session-source">{sourceLabel(session)}</span>
                <span className="session-time">{relativeTime(session.last_activity_at)}</span>
              </span>
              <span className="session-label">{session.label}</span>
              <span className="session-meta">
                <span>{session.audience_label}</span>
                <span>{session.message_count.toLocaleString()} msg</span>
                {session.message_count === 0 ? <span className="session-new">new</span> : null}
              </span>
            </button>
          );
        })}
      </div>
    </aside>
  );
}
