import { useState, type ReactNode } from "react";

import type {
  StreamEntry,
  TurnHistoryRow,
} from "../../api/types";
import { dayLabel, hm, relativeDay } from "../../format/time";

export type ThreadItem =
  | { type: "day"; id: string; label: string }
  | { type: "user"; id: string; who: string; time: string; text: string }
  | { type: "agent"; id: string; time: string; meta: string | null; text: string }
  | {
      type: "silence";
      id: string;
      time: string;
      reason: string | null;
      primaryNoOutputReason: string | null;
      noOutputCategories: readonly string[];
      planRationale: string | null;
      planText: string | null;
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

type PlanThought = {
  text: string;
  rationale: string | null;
};

const PRIMARY_NO_OUTPUT_REASON_LABELS: Readonly<Record<string, string>> = {
  closure: "closure",
  low_value_echo: "low-value echo",
  other: "other",
  user_to_user: "user-to-user",
  when_borg_addressed: "when Borg addressed",
};

const NO_OUTPUT_CATEGORY_LABELS: Readonly<Record<string, string>> = {
  with_open_question: "open question pending",
};

const PLAN_RATIONALE_END_DELIMITERS = [
  " ; verify:",
  " ; tensions:",
  " ; voice:",
  " ; emission:",
  " ; intents:",
] as const;

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

function contentRecord(content: unknown): Record<string, unknown> | null {
  if (typeof content !== "object" || content === null || Array.isArray(content)) {
    return null;
  }

  return content as Record<string, unknown>;
}

function displayText(entry: StreamEntry): string | null {
  return contentString(entry.display_content) ?? contentString(entry.content);
}

function contentField(content: unknown, key: "draftPreview" | "draft_preview"): string | null {
  const record = contentRecord(content);
  if (record === null) {
    return null;
  }

  const value = record[key];
  return typeof value === "string" && value.length > 0 ? value : null;
}

function contentStringArrayField(content: unknown, key: string): string[] {
  const record = contentRecord(content);
  if (record === null) {
    return [];
  }

  const value = record[key];
  if (!Array.isArray(value)) {
    return [];
  }

  return value.filter((item): item is string => typeof item === "string" && item.length > 0);
}

function contentPrimaryNoOutputReason(content: unknown): string | null {
  const record = contentRecord(content);
  if (record === null) {
    return null;
  }

  const value = record.primary_no_output_reason;
  return typeof value === "string" && value.length > 0 ? value : null;
}

function labelCode(code: string, labels: Readonly<Record<string, string>>): string {
  return labels[code] ?? code.replaceAll("_", " ");
}

function extractPlanRationale(content: string): string | null {
  const marker = "uncertainty:";
  const start = content.indexOf(marker);
  if (start < 0) {
    return null;
  }

  const rest = content.slice(start + marker.length).trim();
  let end: number | null = null;
  for (const delimiter of PLAN_RATIONALE_END_DELIMITERS) {
    const delimiterIndex = rest.indexOf(delimiter);
    if (delimiterIndex >= 0 && (end === null || delimiterIndex < end)) {
      end = delimiterIndex;
    }
  }

  const rationale = (end === null ? rest : rest.slice(0, end)).trim();
  return rationale.length > 0 ? rationale : null;
}

function planThoughtFromEntry(entry: StreamEntry): PlanThought | null {
  if (entry.kind !== "thought" || entry.turn_id === undefined) {
    return null;
  }

  const text = contentString(entry.content);
  if (text === null || !text.startsWith("plan:")) {
    return null;
  }

  return {
    text,
    rationale: extractPlanRationale(text),
  };
}

function mapPlanThoughts(entries: readonly StreamEntry[]): Map<string, PlanThought> {
  const thoughtsByTurn = new Map<string, PlanThought>();
  for (const entry of [...entries].sort((left, right) => left.timestamp - right.timestamp)) {
    if (entry.turn_id === undefined) {
      continue;
    }

    const thought = planThoughtFromEntry(entry);
    if (thought !== null) {
      thoughtsByTurn.set(entry.turn_id, thought);
    }
  }

  return thoughtsByTurn;
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
  const planThoughtsByTurn = mapPlanThoughts(entries);
  const items: ThreadItem[] = [];
  let previousDay: string | null = null;

  for (const entry of [...entries].sort((left, right) => left.timestamp - right.timestamp)) {
    const rendered = threadItemFromEntry(entry, turnsById, draftPreviewsByTurn, planThoughtsByTurn);
    if (rendered === null) {
      continue;
    }

    const currentDay = dayKey(entry.timestamp);
    if (currentDay !== previousDay) {
      previousDay = currentDay;
      items.push({
        type: "day",
        id: `day:${currentDay}`,
        label: relativeDay(new Date(entry.timestamp)),
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
  planThoughtsByTurn: ReadonlyMap<string, PlanThought>,
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
      const planThought =
        entry.turn_id === undefined ? undefined : planThoughtsByTurn.get(entry.turn_id);
      const silenceReason = contentReason(entry.content) ?? reason;
      const silenceMechanics = `persisted as agent_suppressed${
        silenceReason === null ? "" : ` (${silenceReason})`
      }`;
      return {
        type: "silence",
        id: entry.id,
        time,
        reason: silenceReason,
        primaryNoOutputReason: contentPrimaryNoOutputReason(entry.content),
        noOutputCategories: contentStringArrayField(entry.content, "no_output_categories"),
        planRationale: planThought?.rationale ?? null,
        planText: planThought?.text ?? null,
        mechanics: silenceMechanics,
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
    return <SilenceArtifact item={item} />;
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

function hasSilenceDetail(item: Extract<ThreadItem, { type: "silence" }>): boolean {
  return (
    item.reason !== null ||
    item.primaryNoOutputReason !== null ||
    item.noOutputCategories.length > 0 ||
    item.planRationale !== null ||
    item.planText !== null
  );
}

function SilenceArtifact({ item }: { item: Extract<ThreadItem, { type: "silence" }> }) {
  const [expanded, setExpanded] = useState(false);
  const detailId = `${item.id}:detail`;
  const expandable = hasSilenceDetail(item);

  if (!expandable) {
    return (
      <article className="thread-special thread-silence">
        <div className="special-head">
          <span className="special-glyph">∅</span>
          <strong>NO OUTPUT — DELIBERATE SILENCE</strong>
          <span className="special-time">{item.time}</span>
        </div>
        <div className="special-mechanics">{item.mechanics}</div>
      </article>
    );
  }

  return (
    <article className="thread-special thread-silence">
      <button
        aria-controls={detailId}
        aria-expanded={expanded}
        className="special-head special-toggle"
        onClick={() => setExpanded((current) => !current)}
        type="button"
      >
        <span className="special-caret" aria-hidden="true">
          {expanded ? "▾" : "▸"}
        </span>
        <span className="special-glyph">∅</span>
        <strong>NO OUTPUT — DELIBERATE SILENCE</strong>
        <span className="special-time">{item.time}</span>
      </button>
      {expanded ? (
        <dl className="special-detail" id={detailId}>
          {item.planRationale === null ? null : (
            <>
              <dt>WHY</dt>
              <dd>{item.planRationale}</dd>
            </>
          )}
          {item.planText === null ? null : (
            <>
              <dt>DID</dt>
              <dd className="special-detail-muted">thought: {item.planText}</dd>
            </>
          )}
          {item.reason === null ? null : (
            <>
              <dt>CLASS</dt>
              <dd>{item.reason}</dd>
            </>
          )}
          {item.primaryNoOutputReason === null ? null : (
            <>
              <dt>PRIMARY</dt>
              <dd>
                {labelCode(item.primaryNoOutputReason, PRIMARY_NO_OUTPUT_REASON_LABELS)}
              </dd>
            </>
          )}
          {item.noOutputCategories.length === 0 ? null : (
            <>
              <dt>CATEGORIES</dt>
              <dd>
                {item.noOutputCategories
                  .map((category) => labelCode(category, NO_OUTPUT_CATEGORY_LABELS))
                  .join(", ")}
              </dd>
            </>
          )}
          <dt>STREAM</dt>
          <dd className="special-detail-muted">{item.mechanics}</dd>
        </dl>
      ) : null}
    </article>
  );
}

function ArtifactMeta({ children }: { children: ReactNode }) {
  return <div className="artifact-meta">{children}</div>;
}
