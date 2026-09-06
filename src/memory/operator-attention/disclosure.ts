import {
  combineMemoryDisclosureLabels,
  memoryDisclosureLabelFromMetadata,
  unknownMemoryDisclosureLabel,
  type MemoryDisclosureLabel,
  type MemoryDisclosureLabelMetadata,
} from "../common/disclosure-label.js";
import type { OperatorAttentionIndexRow, OperatorAttentionRecord } from "./types.js";

/** Attach disclosure to every model-facing row, including rows from older captures. */
export function operatorAttentionPromptRow(
  record: OperatorAttentionRecord & {
    disclosure_label?: MemoryDisclosureLabel | MemoryDisclosureLabelMetadata | null;
  },
): OperatorAttentionIndexRow {
  // This band's provenance is structural: operator-side metadata, authored by
  // the filer. The filer is not necessarily an operator, and this envelope does
  // not name operator recipients. Leave disclosure authorization IDs unknown;
  // internal cognition remains global regardless of those IDs.
  const filingLabel: MemoryDisclosureLabel = {
    disclosureClass: "operator_private",
    originAudienceEntityIds: [record.filer_entity_id],
    privateToEntityIds: [],
    publicToEntityIds: [],
  };
  const disclosureLabel =
    record.disclosure_label === undefined
      ? filingLabel
      : combineMemoryDisclosureLabels([
          memoryDisclosureLabelFromMetadata(record.disclosure_label) ??
            unknownMemoryDisclosureLabel(),
          filingLabel,
        ]);

  return {
    record_key: record.record_key,
    filed_at: record.filed_at,
    filer_entity_id: record.filer_entity_id,
    subject: record.subject,
    disclosure_label: disclosureLabel,
  };
}
