import { STALE_OPEN_ROW_DAYS } from "../../memory/actions/activity.js";
import type {
  ActionThreadOmissionCounts,
  EvidenceLedgerEntry,
  EvidenceLedgerSection,
  EvidenceLedgerGlobalAccounting,
  EvidenceLedgerOmissionCounts,
} from "./types.js";

// Includes the rendered entry envelope, not just its text. Oversize counts move to
// the global line (or trace); they are never truncated into inaccurate accounting.
export const SECTION_ACCOUNTING_MAX_CHARS = 1_024;

export function mergeOmissionCounts(
  ...parts: readonly (EvidenceLedgerOmissionCounts | undefined)[]
): EvidenceLedgerOmissionCounts {
  const result: EvidenceLedgerOmissionCounts = {};
  for (const part of parts)
    for (const [key, value] of Object.entries(part ?? {})) result[key] = (result[key] ?? 0) + value;
  return result;
}

export function renderGlobalAccounting(
  accounting: EvidenceLedgerGlobalAccounting | undefined,
): string | null {
  return accounting === undefined
    ? null
    : `Ledger compaction counts: ${JSON.stringify(accounting)}`;
}

export function renderActionThreadOmissionCounts(
  input: ActionThreadOmissionCounts,
  removedAfterAssembly = 0,
): string {
  const recordsBelowDrawFloor =
    input.sourceRecordTotal === null
      ? "unknown_count_source_total_unavailable"
      : `${Math.max(0, input.sourceRecordTotal - input.consideredRecordCount)}`;
  const sourceRecordTotalField =
    input.sourceRecordTotal === null
      ? "source_record_total=unavailable"
      : `source_record_total=${input.sourceRecordTotal}; records_below_draw_floor is that total minus records_considered, not a separate count`;
  return [
    `Action threads omitted from this section: threads=${input.omittedThreadCount}, records=${input.omittedRecordCount}.`,
    ...(input.staleOpenOmitted === undefined
      ? []
      : [
          `${input.staleOpenOmitted} older open records (no activity for over ${STALE_OPEN_ROW_DAYS} days) not shown${input.staleOpenCountScope === "draw" ? " within this draw; store total unavailable" : " across the store"}. This count overlaps the omitted, below-draw and compaction populations.`,
        ]),
    `Not counted above: salience_dropped_threads=${input.salienceDroppedThreadCount}, records_below_draw_floor=${recordsBelowDrawFloor} (threads_built=${input.threadsBuiltCount} = rendered ${input.renderedThreadCount} + omitted ${input.omittedThreadCount} + dropped ${input.salienceDroppedThreadCount}${removedAfterAssembly === 0 ? "" : ` + removed_after_assembly ${removedAfterAssembly}`}; render_slots=${input.renderLimit}; records_considered=${input.consideredRecordCount} records, source_record_limit=${input.sourceRecordLimit}, ${sourceRecordTotalField}).`,
    ...(removedAfterAssembly === 0
      ? []
      : [
          "removed_after_assembly is already counted by ledger compaction/folding; it is not an additional omission population.",
        ]),
  ].join("\n");
}

/** Recompute from retained handles, never from the wording of a summary. */
export function reconcileOmissionAccounting(section: EvidenceLedgerSection): EvidenceLedgerSection {
  const shownIds = new Set(section.entries.map((entry) => entry.id));
  return {
    ...section,
    entries: section.entries.map((entry) => {
      const action = entry.omission_accounting?.action_threads;
      return action === undefined
        ? entry
        : {
            ...entry,
            omission_accounting: {
              ...entry.omission_accounting,
              action_threads: {
                ...action,
                shown_entry_ids: action.assembled_entries
                  .filter((row) => shownIds.has(row.id))
                  .map((row) => row.id),
              },
            },
          };
    }),
  };
}

export function omissionCounts(entry: EvidenceLedgerEntry): EvidenceLedgerOmissionCounts {
  if (entry.omission_accounting === undefined) return {};
  const action = entry.omission_accounting.action_threads;
  if (action === undefined) return entry.omission_accounting.counts ?? {};
  const shownIds = new Set(action.shown_entry_ids ?? action.assembled_entries.map((row) => row.id));
  const removed = action.assembled_entries.filter((row) => !shownIds.has(row.id));
  const counts = action.counts;
  return {
    ...entry.omission_accounting.counts,
    threads_built: counts.threadsBuiltCount,
    rendered_threads: action.assembled_entries.length - removed.length,
    draw_omitted_threads: counts.omittedThreadCount,
    draw_omitted_records: counts.omittedRecordCount,
    salience_dropped_threads: counts.salienceDroppedThreadCount,
    later_cut_threads: removed.length,
    records_considered: counts.consideredRecordCount,
    render_slots: counts.renderLimit,
    source_record_limit: counts.sourceRecordLimit,
    ...(counts.sourceRecordTotal === null
      ? {}
      : {
          source_record_total: counts.sourceRecordTotal,
          records_below_draw_floor: Math.max(
            0,
            counts.sourceRecordTotal - counts.consideredRecordCount,
          ),
        }),
    ...(counts.staleOpenOmitted === undefined
      ? {}
      : {
          [counts.staleOpenCountScope === "draw"
            ? "older_open_draw_not_shown"
            : "older_open_store_not_shown"]:
            counts.staleOpenOmitted + removed.filter((row) => row.stale_open).length,
          stale_open_days: STALE_OPEN_ROW_DAYS,
        }),
  };
}

export function renderOmissionAccounting(entry: EvidenceLedgerEntry): string | undefined {
  return entry.omission_accounting === undefined
    ? undefined
    : `Omission counts: ${JSON.stringify(omissionCounts(entry))}`;
}
