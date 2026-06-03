import {
  EVIDENCE_LEDGER_SECTION_DEFINITIONS,
  type EvidenceLedgerEntry,
  type EvidenceLedgerSection,
  type EvidenceLedgerSectionId,
} from "./types.js";

export const CURRENT_USER_TRUST_RANK = 100;
export const TRANSCRIPT_TRUST_RANK = 95;
export const CROSS_SESSION_ACTIVITY_TRUST_RANK = 84;
export const COMMITMENT_TRUST_RANK = 82;
export const DISCOURSE_TRUST_RANK = 80;
export const QUARANTINE_TRUST_RANK = 78;
export const ACTION_TRUST_RANK = 72;
export const SLOT_TRUST_RANK = 70;
export const RAW_STREAM_TRUST_RANK = 68;
export const EPISODE_TRUST_RANK = 52;
export const SEMANTIC_TRUST_RANK = 42;
export const OPEN_QUESTION_TRUST_RANK = 38;
export const WARM_RECALL_TRUST_RANK = 34;
// Image perception is a recall bridge to the original attachment, not a
// primary source; keep it visibly below durable semantic memory.
export const IMAGE_PERCEPTION_TRUST_RANK = 10;
export const RELATIONAL_SLOT_LEDGER_LIMIT = 64;

const PRIOR_SESSION_TRUST_RANK_CAP = 30;
const PRIOR_SESSION_DIRECT_SECTION_IDS = new Set<EvidenceLedgerSectionId>([
  "prior_session_memory",
]);

export type SectionBucket = {
  entries: EvidenceLedgerEntry[];
  seenEntryIds: Set<string>;
};

export type SectionBuckets = Map<EvidenceLedgerSectionId, SectionBucket>;

export function createSectionBuckets(): SectionBuckets {
  return new Map(
    EVIDENCE_LEDGER_SECTION_DEFINITIONS.map((section) => [
      section.id,
      {
        entries: [],
        seenEntryIds: new Set<string>(),
      },
    ]),
  );
}

function sectionBucket(
  sections: SectionBuckets,
  sectionId: EvidenceLedgerSectionId,
): SectionBucket {
  const bucket = sections.get(sectionId);

  if (bucket !== undefined) {
    return bucket;
  }

  const next: SectionBucket = {
    entries: [],
    seenEntryIds: new Set<string>(),
  };
  sections.set(sectionId, next);
  return next;
}

function sectionEntries(
  sections: SectionBuckets,
  sectionId: EvidenceLedgerSectionId,
): EvidenceLedgerEntry[] {
  return sectionBucket(sections, sectionId).entries;
}

export function finalSections(sections: SectionBuckets): EvidenceLedgerSection[] {
  return EVIDENCE_LEDGER_SECTION_DEFINITIONS.flatMap((definition) => {
    const entries = sectionEntries(sections, definition.id);

    if ("optional" in definition && definition.optional === true && entries.length === 0) {
      return [];
    }

    return [
      {
        id: definition.id,
        label: definition.label,
        entries,
      },
    ];
  });
}

export function addEntry(
  sections: SectionBuckets,
  sectionId: EvidenceLedgerSectionId,
  entry: EvidenceLedgerEntry,
): void {
  const targetSectionId =
    entry.session_scope === "prior_session" && !PRIOR_SESSION_DIRECT_SECTION_IDS.has(sectionId)
      ? "prior_session_memory"
      : sectionId;
  const targetBucket = sectionBucket(sections, targetSectionId);

  if (targetBucket.seenEntryIds.has(entry.id)) {
    return;
  }

  targetBucket.seenEntryIds.add(entry.id);
  targetBucket.entries.push(entry);
}

export function cappedTrustRank(entry: EvidenceLedgerEntry): EvidenceLedgerEntry {
  if (entry.session_scope !== "prior_session") {
    return entry;
  }

  return {
    ...entry,
    trust_rank: Math.min(entry.trust_rank, PRIOR_SESSION_TRUST_RANK_CAP),
  };
}
