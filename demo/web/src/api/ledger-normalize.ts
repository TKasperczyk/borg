import type {
  EvidenceLedger,
  EvidenceLedgerEntry,
  LedgerResponse,
  MemoryDisclosureClass,
  MemoryDisclosureLabelMetadata,
} from "./types";

const MEMORY_DISCLOSURE_CLASSES = new Set<MemoryDisclosureClass>([
  "public",
  "relationship_private",
  "operator_private",
  "self_private",
  "sensitive",
  "unknown",
]);

function isRecord(value: unknown): value is Record<string, unknown> {
  return value !== null && typeof value === "object" && !Array.isArray(value);
}

function stringArray(value: unknown): string[] {
  return Array.isArray(value)
    ? value.filter((item): item is string => typeof item === "string")
    : [];
}

function memoryDisclosureClass(value: unknown): MemoryDisclosureClass | null {
  return typeof value === "string" && MEMORY_DISCLOSURE_CLASSES.has(value as MemoryDisclosureClass)
    ? (value as MemoryDisclosureClass)
    : null;
}

function memoryDisclosureLabelMetadata(value: unknown): MemoryDisclosureLabelMetadata | undefined {
  if (!isRecord(value)) {
    return undefined;
  }

  const disclosureClass = memoryDisclosureClass(value.disclosure_class);
  if (disclosureClass === null) {
    return undefined;
  }

  return {
    disclosure_class: disclosureClass,
    origin_audience_entity_ids: stringArray(value.origin_audience_entity_ids),
    private_to_entity_ids: stringArray(value.private_to_entity_ids),
    public_to_entity_ids: stringArray(value.public_to_entity_ids),
  };
}

export function normalizeEvidenceLedgerEntry(entry: EvidenceLedgerEntry): EvidenceLedgerEntry {
  if (!isRecord(entry.state_metadata)) {
    return entry;
  }

  const disclosureLabel = memoryDisclosureLabelMetadata(entry.state_metadata.disclosure_label);
  const disclosureNote =
    typeof entry.state_metadata.disclosure_note === "string"
      ? entry.state_metadata.disclosure_note
      : undefined;
  const currentAudienceEntityId =
    typeof entry.state_metadata.current_audience_entity_id === "string" ||
    entry.state_metadata.current_audience_entity_id === null
      ? entry.state_metadata.current_audience_entity_id
      : undefined;

  if (
    disclosureLabel === undefined &&
    disclosureNote === undefined &&
    currentAudienceEntityId === undefined
  ) {
    return entry;
  }

  return {
    ...entry,
    ...(disclosureLabel === undefined ? {} : { disclosure_label: disclosureLabel }),
    ...(disclosureNote === undefined ? {} : { disclosure_note: disclosureNote }),
    ...(currentAudienceEntityId === undefined
      ? {}
      : { current_audience_entity_id: currentAudienceEntityId }),
  };
}

export function normalizeEvidenceLedger(ledger: EvidenceLedger): EvidenceLedger {
  return {
    ...ledger,
    sections: ledger.sections.map((section) => ({
      ...section,
      entries: section.entries.map((entry) => normalizeEvidenceLedgerEntry(entry)),
    })),
  };
}

export function normalizeLedgerResponse(response: LedgerResponse): LedgerResponse {
  return {
    ...response,
    ledger: normalizeEvidenceLedger(response.ledger),
  };
}
