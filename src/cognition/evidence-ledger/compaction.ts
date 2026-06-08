import { estimatePromptTokens } from "../../util/token-estimate.js";
import { coercePositiveIntegerOrFallback } from "../../util/math.js";
import { allSectionIds, emptySectionCountRecord } from "./budget.js";
import { estimateEvidenceLedgerTokens, cloneLedgerWithSections } from "./ledger-copy.js";
import { dedupeEvidenceLedgerByProvenance } from "./provenance-dedupe.js";
import { renderSection } from "./section-rendering.js";
import type {
  EvidenceLedger,
  EvidenceLedgerEntry,
  EvidenceLedgerSection,
  EvidenceLedgerSectionId,
} from "./types.js";

type FullEvidenceLedgerSectionOptions = {
  maxEntries: number;
  maxTokens: number;
};

export type EvidenceLedgerCompactionOptions = {
  targetTokens?: number;
  hardCapTokens?: number;
  maxEntryTextTokens?: number;
  sectionOptions?: Partial<
    Record<EvidenceLedgerSectionId, Partial<FullEvidenceLedgerSectionOptions>>
  >;
};

export type EvidenceLedgerCompactionTraceSummary = {
  preDedupeTokens: number;
  postDedupeTokens: number;
  preCapTokens: number;
  postSectionCapTokens: number;
  postCapTokens: number;
  dedupedEntryCount: number;
  omittedEntryCountsBySection: Record<EvidenceLedgerSectionId, number>;
  droppedSections: EvidenceLedgerSectionId[];
  targetTokens: number;
  hardCapTokens: number;
};

export type CompactedEvidenceLedger = {
  ledger: EvidenceLedger;
  traceSummary: EvidenceLedgerCompactionTraceSummary;
};

const DEFAULT_FULL_LEDGER_TARGET_TOKENS = 60_000;
const DEFAULT_FULL_LEDGER_HARD_CAP_TOKENS = 100_000;
const DEFAULT_FULL_LEDGER_ENTRY_TEXT_TOKEN_CAP = 1_200;

const DEFAULT_FULL_LEDGER_SECTION_OPTIONS = {
  current_user_message: {
    maxEntries: 1,
    maxTokens: 1_200,
  },
  current_session_transcript: {
    maxEntries: 96,
    maxTokens: 8_000,
  },
  current_session_attribution_sidebar: {
    maxEntries: 16,
    maxTokens: 600,
  },
  attribution_matrix: {
    maxEntries: 24,
    maxTokens: 900,
  },
  closure_discourse_state: {
    maxEntries: 16,
    maxTokens: 800,
  },
  contradictions_quarantines: {
    maxEntries: 32,
    maxTokens: 2_500,
  },
  action_states: {
    maxEntries: 32,
    maxTokens: 5_000,
  },
  group_channel_memory: {
    maxEntries: 48,
    maxTokens: 3_000,
  },
  shared_state_recall: {
    maxEntries: 24,
    maxTokens: 3_000,
  },
  retrieved_raw_stream_evidence: {
    maxEntries: 80,
    maxTokens: 7_000,
  },
  retrieved_memory_evidence: {
    maxEntries: 80,
    maxTokens: 7_000,
  },
  episodes: {
    maxEntries: 48,
    maxTokens: 5_500,
  },
  semantic_graph: {
    maxEntries: 80,
    maxTokens: 5_500,
  },
  open_questions: {
    maxEntries: 32,
    maxTokens: 2_500,
  },
  prior_session_memory: {
    maxEntries: 48,
    maxTokens: 4_000,
  },
  autobiographical_recall: {
    maxEntries: 48,
    maxTokens: 5_000,
  },
} as const satisfies Record<EvidenceLedgerSectionId, FullEvidenceLedgerSectionOptions>;

// Compaction priority: sections are trimmed lowest-trust first (rank 0) to highest.
// `satisfies Record<EvidenceLedgerSectionId, number>` forces every section to be ranked, so a
// newly added section can never silently escape trust-ordered compaction.
const LOWEST_TRUST_SECTION_COMPACTION_PRIORITY = {
  prior_session_memory: 0,
  semantic_graph: 1,
  episodes: 2,
  retrieved_memory_evidence: 3,
  autobiographical_recall: 4,
  open_questions: 5,
  shared_state_recall: 6,
  current_session_attribution_sidebar: 7,
  group_channel_memory: 8,
  attribution_matrix: 9,
  action_states: 10,
  contradictions_quarantines: 11,
  closure_discourse_state: 12,
  retrieved_raw_stream_evidence: 13,
  current_session_transcript: 14,
  current_user_message: 15,
} as const satisfies Record<EvidenceLedgerSectionId, number>;

const LOWEST_TRUST_SECTION_ORDER: readonly EvidenceLedgerSectionId[] = [...allSectionIds()].sort(
  (left, right) =>
    LOWEST_TRUST_SECTION_COMPACTION_PRIORITY[left] -
    LOWEST_TRUST_SECTION_COMPACTION_PRIORITY[right],
);

type FullLedgerSectionRetentionPolicy = "head" | "tail";

const TAIL_PRESERVING_FULL_LEDGER_SECTIONS = new Set<EvidenceLedgerSectionId>([
  "current_session_transcript",
]);

function truncateTextForFullEvidenceLedger(
  value: string | undefined,
  maxTokens: number,
): string | undefined {
  if (value === undefined) {
    return undefined;
  }

  const maxChars = Math.max(80, maxTokens * 4);

  if (value.length <= maxChars) {
    return value;
  }

  const omission = `\n[evidence ledger entry truncated ${value.length - maxChars} chars]`;
  const bodyLimit = Math.max(0, maxChars - omission.length);

  return `${value.slice(0, bodyLimit).trimEnd()}${omission}`;
}

function compactFullLedgerEntry(
  entry: EvidenceLedgerEntry,
  maxEntryTextTokens: number,
): EvidenceLedgerEntry {
  return {
    ...entry,
    text: truncateTextForFullEvidenceLedger(entry.text, maxEntryTextTokens),
    value: truncateTextForFullEvidenceLedger(entry.value, Math.max(80, maxEntryTextTokens / 4)),
  };
}

type FullLedgerSectionState = {
  section: EvidenceLedgerSection;
  omittedCount: number;
  dropped: boolean;
  retentionPolicy: FullLedgerSectionRetentionPolicy;
};

function fullLedgerOmittedEntry(
  section: EvidenceLedgerSection,
  omittedCount: number,
  retentionPolicy: FullLedgerSectionRetentionPolicy,
): EvidenceLedgerEntry {
  const omittedKind = retentionPolicy === "tail" ? "older" : "lower-priority";

  return {
    id: `evidence_ledger_omitted:${section.id}`,
    source_type: "system_metadata",
    session_scope: "global",
    actor: "system",
    trust_rank: 0,
    state: "omitted",
    text: `Evidence ledger omitted ${omittedCount} ${omittedKind} entries from ${section.id} to stay within the finalizer ledger budget.`,
    taint: "none",
  };
}

function fullLedgerDroppedSectionEntry(
  section: EvidenceLedgerSection,
  omittedCount: number,
): EvidenceLedgerEntry {
  return {
    id: `evidence_ledger_dropped_section:${section.id}`,
    source_type: "system_metadata",
    session_scope: "global",
    actor: "system",
    trust_rank: 0,
    state: "omitted",
    text: `Evidence ledger dropped all entries from ${section.id} to stay within the global hard cap: entries=${omittedCount}.`,
    taint: "none",
  };
}

function materializeFullLedgerSectionState(state: FullLedgerSectionState): EvidenceLedgerSection {
  if (state.dropped) {
    return {
      ...state.section,
      entries:
        state.omittedCount <= 0
          ? []
          : [fullLedgerDroppedSectionEntry(state.section, state.omittedCount)],
    };
  }

  return {
    ...state.section,
    entries:
      state.omittedCount <= 0
        ? state.section.entries
        : [
            ...state.section.entries,
            fullLedgerOmittedEntry(state.section, state.omittedCount, state.retentionPolicy),
          ],
  };
}

function materializeFullLedgerStates(
  ledger: EvidenceLedger,
  states: readonly FullLedgerSectionState[],
): EvidenceLedger {
  return cloneLedgerWithSections(ledger, states.map(materializeFullLedgerSectionState));
}

function totalFullLedgerPromptTokens(
  ledger: EvidenceLedger,
  states: readonly FullLedgerSectionState[],
): number {
  return estimateEvidenceLedgerTokens(materializeFullLedgerStates(ledger, states));
}

function fullLedgerSectionOptions(
  sectionId: EvidenceLedgerSectionId,
  options: EvidenceLedgerCompactionOptions,
): FullEvidenceLedgerSectionOptions {
  const defaults = DEFAULT_FULL_LEDGER_SECTION_OPTIONS[sectionId];
  const overrides = options.sectionOptions?.[sectionId];

  return {
    maxEntries: coercePositiveIntegerOrFallback(overrides?.maxEntries, defaults.maxEntries),
    maxTokens: coercePositiveIntegerOrFallback(overrides?.maxTokens, defaults.maxTokens),
  };
}

function fullLedgerSectionRetentionPolicy(
  sectionId: EvidenceLedgerSectionId,
): FullLedgerSectionRetentionPolicy {
  return TAIL_PRESERVING_FULL_LEDGER_SECTIONS.has(sectionId) ? "tail" : "head";
}

function capFullLedgerSection(input: {
  section: EvidenceLedgerSection;
  maxEntryTextTokens: number;
  options: FullEvidenceLedgerSectionOptions;
}): FullLedgerSectionState {
  const retentionPolicy = fullLedgerSectionRetentionPolicy(input.section.id);
  const entries =
    retentionPolicy === "tail"
      ? input.section.entries.slice(-input.options.maxEntries)
      : input.section.entries.slice(0, input.options.maxEntries);
  const compactedEntries = entries.map((entry) =>
    compactFullLedgerEntry(entry, input.maxEntryTextTokens),
  );
  let includedEntries: EvidenceLedgerEntry[] = [];
  let omittedCount = Math.max(0, input.section.entries.length - compactedEntries.length);

  if (retentionPolicy === "tail") {
    for (let index = compactedEntries.length - 1; index >= 0; index -= 1) {
      const entry = compactedEntries[index]!;
      const candidateEntries = [entry, ...includedEntries];
      const candidateSection = {
        ...input.section,
        entries: candidateEntries,
      };
      const rendered = renderSection({
        ...candidateSection,
        entries:
          omittedCount <= 0
            ? candidateEntries
            : [
                ...candidateEntries,
                fullLedgerOmittedEntry(candidateSection, omittedCount, retentionPolicy),
              ],
      });

      if (
        estimatePromptTokens(rendered) <= input.options.maxTokens ||
        includedEntries.length === 0
      ) {
        includedEntries = candidateEntries;
        continue;
      }

      omittedCount += index + 1;
      break;
    }

    return {
      section: {
        ...input.section,
        entries: includedEntries,
      },
      omittedCount,
      dropped: false,
      retentionPolicy,
    };
  }

  for (let index = 0; index < compactedEntries.length; index += 1) {
    const entry = compactedEntries[index]!;
    const candidateEntries = [...includedEntries, entry];
    const candidateSection = {
      ...input.section,
      entries: candidateEntries,
    };
    const rendered = renderSection({
      ...candidateSection,
      entries:
        omittedCount <= 0
          ? candidateEntries
          : [
              ...candidateEntries,
              fullLedgerOmittedEntry(candidateSection, omittedCount, retentionPolicy),
            ],
    });

    if (estimatePromptTokens(rendered) <= input.options.maxTokens || includedEntries.length === 0) {
      includedEntries = candidateEntries;
      continue;
    }

    omittedCount += compactedEntries.length - index;
    break;
  }

  return {
    section: {
      ...input.section,
      entries: includedEntries,
    },
    omittedCount,
    dropped: false,
    retentionPolicy,
  };
}

function compactFullLedgerSections(
  ledger: EvidenceLedger,
  options: EvidenceLedgerCompactionOptions,
): FullLedgerSectionState[] {
  const maxEntryTextTokens = coercePositiveIntegerOrFallback(
    options.maxEntryTextTokens,
    DEFAULT_FULL_LEDGER_ENTRY_TEXT_TOKEN_CAP,
  );

  return ledger.sections.map((section) =>
    capFullLedgerSection({
      section,
      maxEntryTextTokens,
      options: fullLedgerSectionOptions(section.id, options),
    }),
  );
}

function trimFullLedgerToTarget(
  ledger: EvidenceLedger,
  states: FullLedgerSectionState[],
  targetTokens: number,
): void {
  while (totalFullLedgerPromptTokens(ledger, states) > targetTokens) {
    const sectionId = LOWEST_TRUST_SECTION_ORDER.find((candidate) => {
      const state = states.find((section) => section.section.id === candidate);
      return state !== undefined && !state.dropped && state.section.entries.length > 0;
    });

    if (sectionId === undefined) {
      break;
    }

    const state = states.find((section) => section.section.id === sectionId)!;
    state.section = {
      ...state.section,
      entries:
        state.retentionPolicy === "tail"
          ? state.section.entries.slice(1)
          : state.section.entries.slice(0, -1),
    };
    state.omittedCount += 1;
  }
}

function dropFullLedgerSectionsToHardCap(
  ledger: EvidenceLedger,
  states: FullLedgerSectionState[],
  hardCapTokens: number,
): EvidenceLedgerSectionId[] {
  const droppedSections: EvidenceLedgerSectionId[] = [];

  while (totalFullLedgerPromptTokens(ledger, states) > hardCapTokens) {
    const sectionId = LOWEST_TRUST_SECTION_ORDER.find((candidate) => {
      const state = states.find((section) => section.section.id === candidate);
      return (
        state !== undefined &&
        !state.dropped &&
        (state.section.entries.length > 0 || state.omittedCount > 0)
      );
    });

    if (sectionId === undefined) {
      break;
    }

    const state = states.find((section) => section.section.id === sectionId)!;
    state.omittedCount += state.section.entries.length;
    state.section = {
      ...state.section,
      entries: [],
    };
    state.dropped = true;
    droppedSections.push(sectionId);
  }

  return droppedSections;
}

export function compactEvidenceLedger(
  ledger: EvidenceLedger,
  options: EvidenceLedgerCompactionOptions = {},
): CompactedEvidenceLedger {
  const targetTokens = coercePositiveIntegerOrFallback(
    options.targetTokens,
    DEFAULT_FULL_LEDGER_TARGET_TOKENS,
  );
  const hardCapTokens = coercePositiveIntegerOrFallback(
    options.hardCapTokens,
    DEFAULT_FULL_LEDGER_HARD_CAP_TOKENS,
  );
  const preDedupeTokens = estimateEvidenceLedgerTokens(ledger);
  const deduped = dedupeEvidenceLedgerByProvenance(ledger);
  const postDedupeTokens = estimateEvidenceLedgerTokens(deduped.ledger);
  const states = compactFullLedgerSections(deduped.ledger, options);
  const preCapTokens = postDedupeTokens;
  const postSectionCapTokens = totalFullLedgerPromptTokens(deduped.ledger, states);
  let droppedSections: EvidenceLedgerSectionId[] = [];

  if (postSectionCapTokens > hardCapTokens) {
    droppedSections = dropFullLedgerSectionsToHardCap(deduped.ledger, states, hardCapTokens);
  } else if (postSectionCapTokens > targetTokens) {
    trimFullLedgerToTarget(deduped.ledger, states, targetTokens);
  }

  const compactedLedger = materializeFullLedgerStates(deduped.ledger, states);
  const postCapTokens = estimateEvidenceLedgerTokens(compactedLedger);
  const omittedEntryCountsBySection = emptySectionCountRecord();

  for (const state of states) {
    omittedEntryCountsBySection[state.section.id] = state.omittedCount;
  }

  return {
    ledger: {
      ...compactedLedger,
      estimatedTokens: postCapTokens,
    },
    traceSummary: {
      preDedupeTokens,
      postDedupeTokens,
      preCapTokens,
      postSectionCapTokens,
      postCapTokens,
      dedupedEntryCount: deduped.dedupedEntryCount,
      omittedEntryCountsBySection,
      droppedSections,
      targetTokens,
      hardCapTokens,
    },
  };
}
