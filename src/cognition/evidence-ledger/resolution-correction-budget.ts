import type { ResolutionCorrectionPresentation } from "../../memory/self/resolution-corrections.js";
import type { EvidenceLedgerEntry } from "./types.js";

type CorrectionRows = NonNullable<ResolutionCorrectionPresentation["resolution_corrections"]>;

/** These structural fields come from the shared correction formatter. */
export function ledgerCorrectionRows(entry: EvidenceLedgerEntry): CorrectionRows {
  const rows = entry.state_metadata?.resolution_corrections;
  return Array.isArray(rows) ? (rows as CorrectionRows) : [];
}

export function ledgerCorrectionCount(entry: EvidenceLedgerEntry): number {
  const count = entry.state_metadata?.resolution_correction_count;
  return typeof count === "number" ? count : ledgerCorrectionRows(entry).length;
}

export function withLedgerCorrectionRows(
  entry: EvidenceLedgerEntry,
  rows: CorrectionRows,
): EvidenceLedgerEntry {
  const total = ledgerCorrectionCount(entry);
  if (total === 0) return entry;
  const { resolution_corrections_omitted: _previousOmission, ...metadata } =
    entry.state_metadata ?? {};
  const omitted = total - rows.length;
  return {
    ...entry,
    state_metadata: {
      ...metadata,
      resolution_correction_count: total,
      resolution_corrections: rows,
      ...(omitted === 0
        ? {}
        : {
            resolution_corrections_omitted: { count: omitted, total, reason: "budget" },
          }),
    },
  };
}
