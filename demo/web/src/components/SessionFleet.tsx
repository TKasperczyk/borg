import { useState, type FormEvent } from "react";

import type { EntityRecord, SessionRecord } from "../api/types";

export type SessionFleetProps = {
  sessions: readonly SessionRecord[];
  activeSessionId: string;
  onSelect: (sessionId: string) => void;
  creator?: EntityRecord | null;
  operatorChatError?: string | null;
  onOpenOperatorChat?: () => Promise<void> | void;
  onSetCreatorByName?: (name: string) => Promise<void> | void;
};

type GroupKey = "today" | "yesterday" | "earlier";

const GROUP_ORDER: readonly GroupKey[] = ["today", "yesterday", "earlier"];

const MS_PER_DAY = 24 * 60 * 60 * 1_000;

function startOfDay(ts: number): number {
  const date = new Date(ts);
  date.setHours(0, 0, 0, 0);
  return date.getTime();
}

function groupKey(ts: number): GroupKey {
  if (!Number.isFinite(ts) || ts <= 0) {
    return "earlier";
  }
  const todayStart = startOfDay(Date.now());
  const tsStart = startOfDay(ts);
  if (tsStart === todayStart) {
    return "today";
  }
  if (tsStart === todayStart - MS_PER_DAY) {
    return "yesterday";
  }
  return "earlier";
}

export function relativeTime(ts: number): string {
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

  const diffDays = Math.floor(diffHours / 24);
  if (diffDays < 7) {
    return `${diffDays}d`;
  }

  const diffWeeks = Math.floor(diffDays / 7);
  if (diffWeeks < 4) {
    return `${diffWeeks}w`;
  }

  return `${Math.floor(diffDays / 30)}mo`;
}

export function sourceLabel(session: SessionRecord): string {
  if (session.source_type === "demo") {
    return session.conversation_kind;
  }
  return session.source_type;
}

export function previewLine(session: SessionRecord): string {
  const audience = session.audience_label;
  const kind = sourceLabel(session);
  if (audience.length === 0) {
    return kind;
  }
  if (kind === audience) {
    return audience;
  }
  return `${audience} · ${kind}`;
}

export function shortId(sessionId: string): string {
  if (sessionId.length <= 8) {
    return sessionId;
  }
  return sessionId.slice(0, 8);
}

export function SessionFleet({
  sessions,
  activeSessionId,
  onSelect,
  creator,
  operatorChatError,
  onOpenOperatorChat,
  onSetCreatorByName,
}: SessionFleetProps) {
  const [collapsed, setCollapsed] = useState(false);
  const [creatorName, setCreatorName] = useState(creator?.canonical_name ?? "");
  const [creatorBusy, setCreatorBusy] = useState(false);
  const grouped = new Map<GroupKey, SessionRecord[]>();
  for (const key of GROUP_ORDER) {
    grouped.set(key, []);
  }
  for (const session of sessions) {
    grouped.get(groupKey(session.last_activity_at))!.push(session);
  }

  const submitCreator = async (event: FormEvent<HTMLFormElement>) => {
    event.preventDefault();
    const name = creatorName.trim();
    if (name.length === 0 || onSetCreatorByName === undefined) {
      return;
    }
    setCreatorBusy(true);
    try {
      await onSetCreatorByName(name);
    } finally {
      setCreatorBusy(false);
    }
  };

  return (
    <aside
      className={`sessions-sidebar session-fleet ${collapsed ? "collapsed" : ""}`}
      aria-label="sessions"
    >
      <div className="sessions-head">
        <span className="title">sessions</span>
        <span className="count">{sessions.length}</span>
        <button
          type="button"
          className="session-fleet-toggle"
          aria-label={collapsed ? "expand sessions" : "collapse sessions"}
          aria-expanded={!collapsed}
          onClick={() => setCollapsed((value) => !value)}
        >
          {collapsed ? "›" : "‹"}
        </button>
      </div>
      {collapsed ? null : (
        <>
          <div className="sessions-presets">
            <button type="button" className="operator-chat-button" onClick={onOpenOperatorChat}>
              operator chat
            </button>
          </div>
          <div className="sessions-list">
            {GROUP_ORDER.map((key) => {
              const rows = grouped.get(key) ?? [];
              if (rows.length === 0) {
                return null;
              }
              return (
                <div key={key}>
                  <div className="session-group-label">{key}</div>
                  {rows.map((session) => {
                    const active = session.session_id === activeSessionId;
                    return (
                      <button
                        key={session.session_id}
                        type="button"
                        className={`session-row ${active ? "active" : ""}`}
                        onClick={() => onSelect(session.session_id)}
                      >
                        <span className="session-row-top">
                          <span className="session-label">{session.label}</span>
                          <span className="session-time">
                            {relativeTime(session.last_activity_at)}
                          </span>
                        </span>
                        <span className="session-preview">{previewLine(session)}</span>
                        <span className="session-foot">
                          <span className={`dot ${active ? "alive" : ""}`}></span>
                          <span>{shortId(session.session_id)}</span>
                          <span className="sep">·</span>
                          <span>{session.message_count.toLocaleString()} msg</span>
                          {session.message_count === 0 ? (
                            <span className="session-new">new</span>
                          ) : null}
                        </span>
                      </button>
                    );
                  })}
                </div>
              );
            })}
          </div>
          <div className="sessions-creator-admin">
            {operatorChatError === null || operatorChatError === undefined ? null : (
              <div className="sessions-creator-error">{operatorChatError}</div>
            )}
            <form onSubmit={submitCreator}>
              <input
                value={creatorName}
                onChange={(event) => setCreatorName(event.target.value)}
                placeholder={creator?.canonical_name ?? "creator name"}
                aria-label="creator name"
              />
              <button type="submit" disabled={creatorBusy || creatorName.trim().length === 0}>
                mark creator
              </button>
            </form>
          </div>
        </>
      )}
    </aside>
  );
}
