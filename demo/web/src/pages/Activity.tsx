import { useMemo, useState } from "react";

import {
  ApiError,
  fetchActivity,
  fetchAutonomyState,
  fetchJournal,
  fetchStream,
} from "../api/client";
import type {
  ActivityOrigin,
  ActivityResponse,
  ActivityRow,
  AutonomyStateResponse,
  JournalEntry,
  StreamEntry,
} from "../api/types";
import { useQuery } from "../api/useQuery";
import { dayLabel, hm, hms, humanMs } from "../format/time";
import { useLive } from "../live/useLive";
import { outcomeDisplayForTurnHistory, type OutcomeTone } from "./chat/outcome";
import { PHASE_LABELS } from "./chat/turnPhase";

type OriginFilter = "all" | ActivityOrigin;

const DIGEST_KEYS: Array<{ key: keyof ActivityResponse["digest"]; label: string }> = [
  { key: "turns", label: "turns" },
  { key: "autonomous_wakes", label: "autonomous wakes" },
  { key: "emissions", label: "emissions" },
  { key: "silences", label: "silences" },
  { key: "journal_notes", label: "journal notes" },
  { key: "dream_changes", label: "dream changes" },
];

function formatError(error: unknown): string {
  if (error instanceof ApiError) {
    return `${error.status} ${error.message}`;
  }
  return error instanceof Error ? error.message : String(error);
}

function todayDayString(): string {
  const date = new Date();
  return `${date.getFullYear()}-${String(date.getMonth() + 1).padStart(2, "0")}-${String(
    date.getDate(),
  ).padStart(2, "0")}`;
}

function dateFromDay(day: string): Date {
  const [yearRaw, monthRaw, dayRaw] = day.split("-");
  return new Date(Number(yearRaw), Number(monthRaw) - 1, Number(dayRaw));
}

function dayTabLabel(day: string): string {
  const label = dayLabel(dateFromDay(day));
  return day === todayDayString() ? `TODAY / ${label}` : label;
}

function journalTimeLabel(date: Date): string {
  const day = `${date.getFullYear()}-${String(date.getMonth() + 1).padStart(2, "0")}-${String(
    date.getDate(),
  ).padStart(2, "0")}`;
  return day === todayDayString() ? hm(date) : `${dayLabel(date)} · ${hm(date)}`;
}

function hmsClock(date: Date): string {
  return `${hm(date)}:${String(date.getSeconds()).padStart(2, "0")}`;
}

function shortId(id: string): string {
  return id.length <= 8 ? id : id.slice(0, 8);
}

function outcomeClass(tone: OutcomeTone | "dream"): string {
  if (tone === "dream") {
    return "activity-outcome-dream";
  }
  if (tone === "ok") {
    return "activity-outcome-ok";
  }
  if (tone === "dim" || tone === "idle") {
    return "activity-outcome-dim";
  }
  return "activity-outcome-red";
}

function activityOutcome(row: ActivityRow): { text: string; tone: OutcomeTone | "dream" } {
  if (row.kind === "dream") {
    return {
      text:
        row.dream.errors > 0
          ? `${row.dream.changes} changes / ${row.dream.errors} errors`
          : `${row.dream.changes} changes`,
      tone: "dream",
    };
  }

  const display = outcomeDisplayForTurnHistory(row.outcome);
  return { text: display.text, tone: display.tone };
}

function rowTitle(row: ActivityRow): string {
  if (row.excerpt !== null && row.excerpt.trim().length > 0) {
    return row.excerpt;
  }
  if (row.kind === "dream") {
    return "dream report";
  }
  return row.suppression_reason ?? "turn recorded";
}

function OriginChip({ origin }: { origin: ActivityOrigin }) {
  return <span className={`activity-origin activity-origin-${origin}`}>{origin}</span>;
}

function DigestStrip({
  digest,
  live,
}: {
  digest: ActivityResponse["digest"];
  live: boolean;
}) {
  return (
    <div className="activity-digest">
      {DIGEST_KEYS.map((item) => (
        <div className="activity-digest-item" key={item.key}>
          <b>{digest[item.key]}</b>
          <span>{item.label}</span>
        </div>
      ))}
      {live ? <div className="activity-digest-live">live -- updates as turns land</div> : null}
    </div>
  );
}

function PhaseBar({ turnId }: { turnId: string }) {
  const live = useLive();
  const cached = useMemo(
    () => live.getTurnPhaseDurations(turnId),
    [live, live.phaseCacheVersion, turnId],
  );

  if (cached === null || cached.totalMs <= 0) {
    return null;
  }

  return (
    <div className="activity-phase-wrap" aria-label={`Observed phases for ${turnId}`}>
      <div className="activity-phase-bar">
        {cached.phases.map((phase) => (
          <div
            className={phase.phase === "delib" ? "activity-phase activity-phase-delib" : "activity-phase"}
            key={phase.phase}
            style={{ flexGrow: Math.max(1, phase.durationMs) }}
            title={`${PHASE_LABELS[phase.phase]} ${humanMs(phase.durationMs)}`}
          />
        ))}
      </div>
      <div className="activity-phase-legend">
        {cached.phases.map((phase) => (
          <span key={phase.phase}>
            {PHASE_LABELS[phase.phase]} {humanMs(phase.durationMs)}
          </span>
        ))}
      </div>
    </div>
  );
}

function didEntryLabel(entry: StreamEntry): string | null {
  if (entry.kind === "agent_msg" && typeof entry.content === "string") {
    return `answered: ${entry.content}`;
  }
  if (
    (entry.kind === "agent_suppressed" || entry.kind === "agent_observed") &&
    typeof entry.content === "object" &&
    entry.content !== null &&
    "reason" in entry.content &&
    typeof entry.content.reason === "string"
  ) {
    return `${entry.kind}: ${entry.content.reason}`;
  }
  if (
    entry.kind === "internal_event" &&
    typeof entry.content === "object" &&
    entry.content !== null &&
    "kind" in entry.content &&
    typeof entry.content.kind === "string"
  ) {
    return `internal: ${entry.content.kind}`;
  }
  if (entry.kind === "thought" && typeof entry.content === "string") {
    return `thought: ${entry.content}`;
  }
  return null;
}

function ActivityDetails({
  row,
  journalEntries,
}: {
  row: ActivityRow;
  journalEntries: readonly JournalEntry[];
}) {
  const detail = useQuery(`activity:detail:${row.session_id}:${row.turn_id ?? row.id}`, () =>
    row.turn_id === null
      ? Promise.resolve({ entries: [], next_cursor: null })
      : fetchStream(row.session_id, 200),
  );
  const turnEntries =
    row.turn_id === null
      ? []
      : (detail.data?.entries ?? []).filter((entry) => entry.turn_id === row.turn_id);
  const didRows = turnEntries
    .map((entry) => ({ entry, label: didEntryLabel(entry) }))
    .filter((item): item is { entry: StreamEntry; label: string } => item.label !== null);
  const journal = row.turn_id === null
    ? []
    : journalEntries.filter((entry) => entry.source_turn_id === row.turn_id);

  return (
    <div className="activity-detail">
      {row.turn_id === null ? null : <PhaseBar turnId={row.turn_id} />}
      {detail.loading && row.turn_id !== null ? (
        <div className="activity-detail-empty">loading turn details...</div>
      ) : null}
      {!detail.loading && didRows.length > 0 ? (
        <div className="activity-did-list">
          <div className="activity-detail-label">DID</div>
          {didRows.map(({ entry, label }) => (
            <div className="activity-did-row" key={entry.id}>
              <time>{hms(new Date(entry.timestamp))}</time>
              <span>{label}</span>
            </div>
          ))}
        </div>
      ) : null}
      {!detail.loading && didRows.length === 0 && journal.length === 0 ? (
        <div className="activity-detail-empty">no additional persisted detail</div>
      ) : null}
      {journal.map((entry) => (
        <blockquote className="activity-journal-note" key={entry.id}>
          <time>{hm(new Date(entry.updated_at))}</time>
          <span>{entry.text}</span>
        </blockquote>
      ))}
    </div>
  );
}

function ActivityRowView({
  row,
  expanded,
  onToggle,
  journalEntries,
}: {
  row: ActivityRow;
  expanded: boolean;
  onToggle: () => void;
  journalEntries: readonly JournalEntry[];
}) {
  const outcome = activityOutcome(row);

  return (
    <article className={row.origin === "autonomous" ? "activity-row activity-row-auto" : "activity-row"}>
      <button className="activity-row-main" type="button" onClick={onToggle}>
        <time className="activity-row-time">{hm(new Date(row.started_at))}</time>
        <div className="activity-row-body">
          <div className="activity-row-top">
            <OriginChip origin={row.origin} />
            {row.trigger === null ? null : <span className="activity-trigger">{row.trigger}</span>}
            <span className={`activity-outcome ${outcomeClass(outcome.tone)}`}>{outcome.text}</span>
            <span className="activity-duration">{humanMs(row.duration_ms)}</span>
          </div>
          <div className={expanded ? "activity-title activity-title-open" : "activity-title"}>
            {rowTitle(row)}
          </div>
          <div className="activity-session">{row.session_label ?? row.session_id}</div>
        </div>
      </button>
      {expanded ? <ActivityDetails row={row} journalEntries={journalEntries} /> : null}
    </article>
  );
}

function WakeSources({ autonomy }: { autonomy: AutonomyStateResponse }) {
  return (
    <section className="activity-rail-section">
      <div className="activity-rail-head">
        <span>WAKE SOURCES</span>
        <b>{autonomy.scheduler.enabled ? "enabled" : "disabled"}</b>
      </div>
      {autonomy.scheduler.next_tick_at === null ? null : (
        <div className="wake-source-summary">
          next evaluation {hmsClock(new Date(autonomy.scheduler.next_tick_at))}
        </div>
      )}
      <div className="wake-source-list">
        {autonomy.wake_sources.map((source) => {
          const stateClass = source.enabled ? "wake-dot wake-dot-on" : "wake-dot wake-dot-off";
          const stateLabel = source.enabled ? "enabled" : "disabled";
          const scheduleLabel =
            source.wake_source_type === "condition"
              ? "event-driven"
              : source.next_due_at === null || source.next_due_at === undefined
                ? "nothing scheduled"
                : `next ${journalTimeLabel(new Date(source.next_due_at))}`;

          return (
            <div className="wake-source-row" key={source.name}>
              <span aria-label={`${source.name} ${stateLabel}`} className={stateClass} />
              <div>
                <b>{source.name}</b>
                <span>
                  {scheduleLabel}
                  {source.last_fired === null
                    ? " / no recent fire"
                    : ` / last ${hm(new Date(source.last_fired))}`}
                  {source.wake_count > 0 ? ` / ${source.wake_count} recent` : ""}
                </span>
              </div>
            </div>
          );
        })}
      </div>
      {autonomy.wake_budget === null ? null : (
        <div className="wake-budget">
          <div>
            budget {autonomy.wake_budget.used}/{autonomy.wake_budget.limit}
          </div>
          <div className="wake-budget-track">
            <span
              style={{
                width: `${Math.min(
                  100,
                  (autonomy.wake_budget.used / Math.max(1, autonomy.wake_budget.limit)) * 100,
                )}%`,
              }}
            />
          </div>
        </div>
      )}
    </section>
  );
}

function ScheduledWakes({ autonomy }: { autonomy: AutonomyStateResponse }) {
  if (autonomy.self_scheduled_wakes.length === 0) {
    return null;
  }

  return (
    <section className="activity-rail-section">
      <div className="activity-rail-head">
        <span>SELF-SCHEDULED WAKES</span>
      </div>
      {autonomy.self_scheduled_wakes.map((wake) => {
        const isCancelled = wake.status === "cancelled";
        return (
          <div className="scheduled-wake-row" key={wake.id}>
            <time>{hm(new Date(wake.due_at))}</time>
            <span>{wake.note}</span>
            {isCancelled ? <b>CANCELED</b> : null}
          </div>
        );
      })}
    </section>
  );
}

function TrainOfThought({ entries }: { entries: readonly JournalEntry[] }) {
  if (entries.length === 0) {
    return null;
  }

  return (
    <section className="activity-rail-section">
      <div className="activity-rail-head">
        <span>TRAIN OF THOUGHT</span>
      </div>
      {entries.map((entry) => (
        <div className="thought-row" key={entry.id}>
          <time>{journalTimeLabel(new Date(entry.updated_at))}</time>
          {entry.source_turn_id === null ? (
            <span>journal</span>
          ) : (
            <span title={entry.source_turn_id}>{shortId(entry.source_turn_id)}</span>
          )}
          <i>{entry.text}</i>
        </div>
      ))}
    </section>
  );
}

export function ActivityPage() {
  const [selectedDay, setSelectedDay] = useState<string | undefined>(undefined);
  const [originFilter, setOriginFilter] = useState<OriginFilter>("all");
  const [expandedRow, setExpandedRow] = useState<string | null>(null);
  const activity = useQuery(`activity:${selectedDay ?? "today"}`, () => fetchActivity(selectedDay));
  const autonomy = useQuery("autonomy", fetchAutonomyState);
  const activeDay = activity.data?.day ?? selectedDay ?? todayDayString();
  const journal = useQuery("journal", () => fetchJournal(10));
  const rows = useMemo(() => {
    const source = activity.data?.rows ?? [];
    return originFilter === "all" ? source : source.filter((row) => row.origin === originFilter);
  }, [activity.data?.rows, originFilter]);
  const journalEntries = journal.data?.entries ?? [];
  const dayTabs = activity.data?.days.length === 0 ? [activeDay] : activity.data?.days ?? [activeDay];

  return (
    <main className="page">
      <header className="page-header activity-header">
        <div>
          <span className="page-title">ACTIVITY</span>
          <span className="page-subtitle">
            what the entity did -- every turn, wake, and dream, without asking it
          </span>
        </div>
        <div className="activity-tabs">
          {dayTabs.map((day) => (
            <button
              className={day === activeDay ? "activity-tab activity-tab-active" : "activity-tab"}
              key={day}
              onClick={() => setSelectedDay(day)}
              type="button"
            >
              {dayTabLabel(day)}
            </button>
          ))}
          {(["all", "autonomous", "user", "dream"] as const).map((origin) => (
            <button
              className={originFilter === origin ? "activity-tab activity-tab-active" : "activity-tab"}
              key={origin}
              onClick={() => setOriginFilter(origin)}
              type="button"
            >
              {origin.toUpperCase()}
            </button>
          ))}
        </div>
      </header>
      {activity.data === undefined ? (
        <div className="activity-empty">loading activity...</div>
      ) : (
        <DigestStrip digest={activity.data.digest} live={activeDay === todayDayString()} />
      )}
      {activity.data?.truncated === true ? (
        <div className="activity-truncated">showing latest 200 rows for this day</div>
      ) : null}
      {activity.error === undefined ? null : (
        <div className="activity-error">{formatError(activity.error)}</div>
      )}
      <div className="activity-layout">
        <section className="activity-timeline">
          {rows.length === 0 && !activity.loading ? (
            <div className="activity-empty">no activity recorded for this day</div>
          ) : null}
          {rows.map((row) => (
            <ActivityRowView
              expanded={expandedRow === row.id}
              journalEntries={journalEntries}
              key={row.id}
              onToggle={() => setExpandedRow((current) => (current === row.id ? null : row.id))}
              row={row}
            />
          ))}
        </section>
        <aside className="activity-rail">
          {autonomy.data === undefined || autonomy.error !== undefined ? null : (
            <>
              <WakeSources autonomy={autonomy.data} />
              <ScheduledWakes autonomy={autonomy.data} />
            </>
          )}
          <TrainOfThought entries={journalEntries} />
        </aside>
      </div>
    </main>
  );
}
