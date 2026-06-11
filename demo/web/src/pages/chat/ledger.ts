import type { EvidenceLedger, EvidenceLedgerEntry } from "../../api/types";

export type LedgerChip = {
  key: string;
  value: number;
};

export type LedgerSummary = {
  chips: LedgerChip[];
  disclosureCount: number;
  totalEntries: number;
};

const SOURCE_LABELS: Record<string, string> = {
  current_user_message: "USER",
  current_session_stream: "CUR",
  prior_session_stream: "PRIOR",
  episode: "EPI",
  semantic_node: "SEM",
  semantic_edge: "SEM",
  action_record: "ACT",
  relational_slot: "REL",
  commitment: "COM",
  shared_state: "SHARED",
  image_attachment: "IMG",
  assistant_stream: "ASST",
  system_metadata: "SYS",
};

function chipLabel(entry: EvidenceLedgerEntry, sectionId: string): string {
  if (entry.source_type !== undefined) {
    return SOURCE_LABELS[entry.source_type] ?? entry.source_type.toUpperCase();
  }

  return sectionId.toUpperCase();
}

function hasDisclosureMetadata(value: unknown): boolean {
  if (typeof value !== "object" || value === null) {
    return false;
  }

  if (Array.isArray(value)) {
    return value.some(hasDisclosureMetadata);
  }

  const record = value as Record<string, unknown>;
  if (
    record.disclosure_label !== undefined ||
    record.disclosure_class !== undefined ||
    record.privacy_level !== undefined
  ) {
    return true;
  }

  return Object.values(record).some(hasDisclosureMetadata);
}

function entryHasDisclosure(entry: EvidenceLedgerEntry): boolean {
  return hasDisclosureMetadata(entry.state_metadata);
}

export function summarizeLedger(ledger: EvidenceLedger | null | undefined): LedgerSummary | null {
  if (ledger === null || ledger === undefined) {
    return null;
  }

  const counts = new Map<string, number>();
  let totalEntries = 0;
  let disclosureCount = 0;

  for (const section of ledger.sections ?? []) {
    for (const entry of section.entries ?? []) {
      totalEntries += 1;
      const label = chipLabel(entry, section.id);
      counts.set(label, (counts.get(label) ?? 0) + 1);
      if (entryHasDisclosure(entry)) {
        disclosureCount += 1;
      }
    }
  }

  if (ledger.imageAttachments !== undefined && ledger.imageAttachments.length > 0) {
    counts.set("IMG", (counts.get("IMG") ?? 0) + ledger.imageAttachments.length);
  }

  return {
    chips: [...counts.entries()]
      .map(([key, value]) => ({ key, value }))
      .sort((left, right) => left.key.localeCompare(right.key)),
    disclosureCount,
    totalEntries,
  };
}
