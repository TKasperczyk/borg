import type { ReactNode } from "react";

import type {
  StreamEntry,
  TurnHistoryRow,
} from "../../api/types";
import { dayLabel, hm } from "../../format/time";

export type ThreadItem =
  | { type: "day"; id: string; label: string }
  | { type: "user"; id: string; who: string; time: string; text: string }
  | { type: "agent"; id: string; time: string; meta: string | null; text: string }
  | {
      type: "silence";
      id: string;
      time: string;
      reason: string | null;
      mechanics: string;
    }
  | {
      type: "suppressed";
      id: string;
      time: string;
      className: string | null;
      reason: string | null;
      draftPreview: string | null;
      mechanics: string;
    }
  | { type: "observed"; id: string; time: string; reason: string | null; mechanics: string }
  | { type: "dream"; id: string; text: string };

export function contentReason(content: unknown): string | null {
  if (typeof content !== "object" || content === null || Array.isArray(content)) {
    return null;
  }

  const reason = (content as { reason?: unknown }).reason;
  return typeof reason === "string" && reason.length > 0 ? reason : null;
}

function contentString(content: unknown): string | null {
  return typeof content === "string" && content.length > 0 ? content : null;
}

function displayText(entry: StreamEntry): string | null {
  return contentString(entry.display_content) ?? contentString(entry.content);
}

function contentField(content: unknown, key: "draftPreview" | "draft_preview"): string | null {
  if (typeof content !== "object" || content === null || Array.isArray(content)) {
    return null;
  }

  const value = (content as Record<string, unknown>)[key];
  return typeof value === "string" && value.length > 0 ? value : null;
}

function dayKey(timestamp: number): string {
  const date = new Date(timestamp);
  return `${date.getFullYear()}-${date.getMonth()}-${date.getDate()}`;
}

function mapTurns(turns: readonly TurnHistoryRow[]): Map<string, TurnHistoryRow> {
  return new Map(turns.map((turn) => [turn.turn_id, turn]));
}

export function isDeliberateSilence(
  _entry: Pick<StreamEntry, "content" | "turn_id">,
  turn: TurnHistoryRow | undefined,
): boolean {
  return turn?.outcome === "deliberate-silence";
}

export function threadItemsFromEntries(
  entries: readonly StreamEntry[],
  turns: readonly TurnHistoryRow[],
  draftPreviewsByTurn: Readonly<Record<string, string>> = {},
): ThreadItem[] {
  const turnsById = mapTurns(turns);
  const items: ThreadItem[] = [];
  let previousDay: string | null = null;

  for (const entry of [...entries].sort((left, right) => left.timestamp - right.timestamp)) {
    const rendered = threadItemFromEntry(entry, turnsById, draftPreviewsByTurn);
    if (rendered === null) {
      continue;
    }

    const currentDay = dayKey(entry.timestamp);
    if (currentDay !== previousDay) {
      previousDay = currentDay;
      items.push({
        type: "day",
        id: `day:${currentDay}`,
        label: dayLabel(new Date(entry.timestamp)),
      });
    }

    items.push(rendered);
  }

  return items;
}

function threadItemFromEntry(
  entry: StreamEntry,
  turnsById: ReadonlyMap<string, TurnHistoryRow>,
  draftPreviewsByTurn: Readonly<Record<string, string>>,
): ThreadItem | null {
  const time = hm(new Date(entry.timestamp));

  if (entry.kind === "user_msg") {
    const text = displayText(entry);
    if (text === null) {
      return null;
    }

    return {
      type: "user",
      id: entry.id,
      who: (entry.sender_label ?? "USER").toUpperCase(),
      time,
      text,
    };
  }

  if (entry.kind === "agent_msg") {
    const text = displayText(entry);
    if (text === null) {
      return null;
    }

    return {
      type: "agent",
      id: entry.id,
      time,
      meta: time,
      text,
    };
  }

  if (entry.kind === "agent_suppressed") {
    const turn = entry.turn_id === undefined ? undefined : turnsById.get(entry.turn_id);
    const reason = turn?.suppression_reason ?? null;
    const mechanics = `persisted as agent_suppressed${reason === null ? "" : ` (${reason})`}`;
    if (isDeliberateSilence(entry, turn)) {
      return {
        type: "silence",
        id: entry.id,
        time,
        reason,
        mechanics,
      };
    }

    const livePreview = entry.turn_id === undefined ? undefined : draftPreviewsByTurn[entry.turn_id];
    return {
      type: "suppressed",
      id: entry.id,
      time,
      className: turn?.outcome ?? null,
      reason,
      draftPreview:
        livePreview ??
        contentField(entry.content, "draftPreview") ??
        contentField(entry.content, "draft_preview"),
      mechanics,
    };
  }

  if (entry.kind === "agent_observed") {
    return {
      type: "observed",
      id: entry.id,
      time,
      reason: contentReason(entry.content),
      mechanics: "persisted as agent_observed",
    };
  }

  if (entry.kind === "dream_report") {
    return {
      type: "dream",
      id: entry.id,
      text: `— ${dayLabel(new Date(entry.timestamp))} · dream report · ${time} —`,
    };
  }

  return null;
}

export function ThreadArtifactList({ items }: { items: readonly ThreadItem[] }) {
  return (
    <>
      {items.map((item) => (
        <ThreadArtifact key={item.id} item={item} />
      ))}
    </>
  );
}

export function ThreadArtifact({ item }: { item: ThreadItem }) {
  if (item.type === "day") {
    return <div className="thread-day">— {item.label} —</div>;
  }
  if (item.type === "user") {
    return (
      <article className="thread-artifact thread-user">
        <ArtifactMeta>
          {item.who} · {item.time}
        </ArtifactMeta>
        <div className="artifact-body">{item.text}</div>
      </article>
    );
  }
  if (item.type === "agent") {
    return (
      <article className="thread-artifact thread-agent">
        <ArtifactMeta>
          <span className="entity-label">ENTITY</span>
          {item.meta === null ? null : <span>{item.meta}</span>}
        </ArtifactMeta>
        <div className="artifact-body pretty-wrap">{item.text}</div>
      </article>
    );
  }
  if (item.type === "silence") {
    return (
      <article className="thread-special thread-silence">
        <div className="special-head">
          <span className="special-glyph">∅</span>
          <strong>NO OUTPUT — DELIBERATE SILENCE</strong>
          <span className="special-time">{item.time}</span>
        </div>
        {item.reason === null ? null : <div className="special-reason italic">{item.reason}</div>}
        <div className="special-mechanics">{item.mechanics}</div>
      </article>
    );
  }
  if (item.type === "suppressed") {
    return (
      <article className="thread-special thread-suppressed">
        <div className="special-head">
          <span className="suppressed-glyph">✕</span>
          <strong>{item.className === null ? "SUPPRESSED" : `SUPPRESSED — ${item.className}`}</strong>
          <span className="special-time">{item.time}</span>
        </div>
        {item.reason === null ? null : <div className="special-reason">{item.reason}</div>}
        {item.draftPreview === null ? null : (
          <div className="draft-preview">{item.draftPreview}</div>
        )}
        <div className="special-mechanics">{item.mechanics}</div>
      </article>
    );
  }
  if (item.type === "observed") {
    return (
      <article className="thread-artifact thread-observed">
        <div className="observed-head">
          <span>◎ OBSERVED</span>
          <span>present, not replying · {item.time}</span>
        </div>
        {item.reason === null ? null : <div className="observed-reason">{item.reason}</div>}
        <div className="special-mechanics">{item.mechanics}</div>
      </article>
    );
  }

  return <div className="thread-day">{item.text}</div>;
}

function ArtifactMeta({ children }: { children: ReactNode }) {
  return <div className="artifact-meta">{children}</div>;
}
