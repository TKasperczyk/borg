import type { TurnHistoryOutcomeClass, TurnHistoryRow } from "../../api/types";
import { Tag, type TagKind } from "../../components/Tag";
import {
  STREAM_OUTCOME_DESCRIPTORS,
  type StreamOutcomeClass,
} from "../../lib/stream-outcomes";
import { formatTime } from "../../lib/stream-utils";
import { fieldLabel, shortId } from "../screen-utils";

type TurnOutcomeFilter = "all" | "suppressed_failed" | "emitted" | "observed";

const TURN_OUTCOME_FILTERS: ReadonlyArray<{ id: TurnOutcomeFilter; label: string }> = [
  { id: "all", label: "all" },
  { id: "suppressed_failed", label: "suppressed/failed" },
  { id: "emitted", label: "emitted" },
  { id: "observed", label: "observed" },
];

const SUPPRESSED_OR_FAILED_OUTCOMES = new Set<TurnHistoryOutcomeClass>([
  "deliberate-silence",
  "emission-failed",
  "guard-blocked",
  "failed",
  "unknown",
]);

function isSuppressedOrFailedOutcome(outcome: TurnHistoryOutcomeClass): boolean {
  return SUPPRESSED_OR_FAILED_OUTCOMES.has(outcome);
}

function rowMatchesOutcomeFilter(row: TurnHistoryRow, filter: TurnOutcomeFilter): boolean {
  if (filter === "all") {
    return true;
  }
  if (filter === "suppressed_failed") {
    return isSuppressedOrFailedOutcome(row.outcome);
  }
  return row.outcome === filter;
}

function historyOutcomeLabel(row: TurnHistoryRow): string {
  if (row.outcome === "emitted") {
    return "emitted";
  }
  if (row.outcome === "failed") {
    return "failed";
  }

  return STREAM_OUTCOME_DESCRIPTORS[row.outcome as StreamOutcomeClass]?.label ?? "unknown";
}

function historyOutcomeKind(row: TurnHistoryRow): TagKind {
  if (row.outcome === "emitted") {
    return "acc";
  }
  if (row.outcome === "failed") {
    return "bad";
  }

  return STREAM_OUTCOME_DESCRIPTORS[row.outcome as StreamOutcomeClass]?.tagKind ?? "";
}

export function TurnHistoryRail({
  rows,
  loading,
  error,
  selectedTurnId,
  filter,
  onFilterChange,
  onSelectTurn,
  onLive,
}: {
  rows: readonly TurnHistoryRow[];
  loading: boolean;
  error: Error | null;
  selectedTurnId: string | null;
  filter: TurnOutcomeFilter;
  onFilterChange: (filter: TurnOutcomeFilter) => void;
  onSelectTurn: (turnId: string) => void;
  onLive: () => void;
}) {
  const visibleRows = rows.filter((row) => rowMatchesOutcomeFilter(row, filter));

  return (
    <aside className="turn-history" aria-label="Invocation history">
      <div className="turn-history-head">
        <span className="title">invocations</span>
        <button
          className={`turn-history-live ${selectedTurnId === null ? "active" : ""}`.trim()}
          type="button"
          onClick={onLive}
        >
          live
        </button>
      </div>
      <div className="turn-history-filters" aria-label="Outcome filter">
        {TURN_OUTCOME_FILTERS.map((item) => (
          <button
            key={item.id}
            type="button"
            className={`turn-history-filter ${filter === item.id ? "active" : ""}`.trim()}
            onClick={() => onFilterChange(item.id)}
          >
            {item.label}
          </button>
        ))}
      </div>
      <div className="turn-history-list">
        {loading && rows.length === 0 ? <div className="turn-history-empty">loading</div> : null}
        {error !== null && rows.length === 0 ? (
          <div className="turn-history-empty">{error.message}</div>
        ) : null}
        {!loading && error === null && visibleRows.length === 0 ? (
          <div className="turn-history-empty">no turns</div>
        ) : null}
        {visibleRows.map((row) => (
          <button
            key={row.turn_id}
            type="button"
            className={`turn-history-row ${selectedTurnId === row.turn_id ? "active" : ""}`.trim()}
            onClick={() => onSelectTurn(row.turn_id)}
          >
            <div className="turn-history-row-top">
              <Tag kind={historyOutcomeKind(row)} dot>
                {historyOutcomeLabel(row)}
              </Tag>
              <span className="turn-history-time">{formatTime(row.started_at)}</span>
            </div>
            <div className="turn-history-turn">{shortId(row.turn_id)}</div>
            <div className="turn-history-meta">
              <span>{row.audience ?? "no audience"}</span>
              {row.suppression_reason === null ? null : (
                <span>reason {fieldLabel(row.suppression_reason)}</span>
              )}
            </div>
          </button>
        ))}
      </div>
    </aside>
  );
}
