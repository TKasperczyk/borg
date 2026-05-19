import type { EvidenceLedgerCompactionTraceSummary } from "../../evidence-ledger/index.js";

export function evidenceLedgerCompactionChanged(
  summary: EvidenceLedgerCompactionTraceSummary,
): boolean {
  return (
    summary.dedupedEntryCount > 0 ||
    summary.droppedSections.length > 0 ||
    summary.postCapTokens < summary.preCapTokens ||
    Object.values(summary.omittedEntryCountsBySection).some((count) => count > 0)
  );
}
