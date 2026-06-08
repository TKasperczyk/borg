import { EVIDENCE_LEDGER_SECTION_DEFINITIONS, type EvidenceLedgerSectionId } from "./types.js";

export function allSectionIds(): EvidenceLedgerSectionId[] {
  return EVIDENCE_LEDGER_SECTION_DEFINITIONS.map((section) => section.id);
}

export function emptySectionCountRecord(): Record<EvidenceLedgerSectionId, number> {
  return Object.fromEntries(allSectionIds().map((sectionId) => [sectionId, 0])) as Record<
    EvidenceLedgerSectionId,
    number
  >;
}
