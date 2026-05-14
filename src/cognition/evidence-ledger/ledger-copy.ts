import { estimatePromptTokens } from "../../util/token-estimate.js";
import { renderEvidenceLedger } from "./finalizer-ledger.js";
import type { EvidenceLedger, EvidenceLedgerSection } from "./types.js";

export function estimateEvidenceLedgerTokens(ledger: EvidenceLedger): number {
  return estimatePromptTokens(
    renderEvidenceLedger({
      ...ledger,
      estimatedTokens: 0,
    }) ?? "",
  );
}

export function cloneLedgerWithSections(
  ledger: EvidenceLedger,
  sections: readonly EvidenceLedgerSection[],
): EvidenceLedger {
  const next = {
    ...ledger,
    sections: sections.map((section) => ({
      ...section,
      entries: section.entries.map((entry) => ({
        ...entry,
        citations: entry.citations === undefined ? undefined : [...entry.citations],
        state_metadata:
          entry.state_metadata === undefined ? undefined : { ...entry.state_metadata },
      })),
    })),
  };

  return {
    ...next,
    estimatedTokens: estimateEvidenceLedgerTokens(next),
  };
}
