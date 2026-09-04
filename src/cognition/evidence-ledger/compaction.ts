import { estimatePromptTokens } from "../../util/token-estimate.js";
import { coercePositiveIntegerOrFallback } from "../../util/math.js";
import { formatUtcDaySpanLabel } from "../../util/utc-day.js";
import { isRecentLivedExperienceSpineKind } from "../../memory/activity/index.js";
import { selfPrivateMemoryDisclosureLabel } from "../../memory/common/disclosure-label.js";
import { allSectionIds, emptySectionCountRecord } from "./budget.js";
import { estimateEvidenceLedgerTokens, cloneLedgerWithSections } from "./ledger-copy.js";
import { dedupeEvidenceLedgerByProvenance } from "./provenance-dedupe.js";
import { reconcileAutobiographicalRecallCapMetadata } from "./autobiographical-recall-cap-metadata.js";
import { renderSection } from "./section-rendering.js";
import {
  appendMemoryDisclosureState,
  appendMemoryDisclosureStateMetadata,
} from "./entry-metadata.js";
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
  recent_lived_experience: {
    maxEntries: 96,
    maxTokens: 5_000,
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
  recent_lived_experience: 1,
  semantic_graph: 2,
  episodes: 3,
  retrieved_memory_evidence: 4,
  autobiographical_recall: 5,
  open_questions: 6,
  shared_state_recall: 7,
  current_session_attribution_sidebar: 8,
  group_channel_memory: 9,
  attribution_matrix: 10,
  action_states: 11,
  contradictions_quarantines: 12,
  closure_discourse_state: 13,
  retrieved_raw_stream_evidence: 14,
  current_session_transcript: 15,
  current_user_message: 16,
} as const satisfies Record<EvidenceLedgerSectionId, number>;

const LOWEST_TRUST_SECTION_ORDER: readonly EvidenceLedgerSectionId[] = [...allSectionIds()].sort(
  (left, right) =>
    LOWEST_TRUST_SECTION_COMPACTION_PRIORITY[left] -
    LOWEST_TRUST_SECTION_COMPACTION_PRIORITY[right],
);

type FullLedgerSectionRetentionPolicy = "head" | "tail" | "spine";

type OmittedEntrySpan = {
  firstOccurredAt: number;
  lastOccurredAt: number;
};

const RECENT_LIVED_EXPERIENCE_BREADCRUMB_DISCLOSURE_LABEL = selfPrivateMemoryDisclosureLabel();

const TAIL_PRESERVING_FULL_LEDGER_SECTIONS = new Set<EvidenceLedgerSectionId>([
  "current_session_transcript",
]);

// Sections whose entries carry a lifecycle status as the leading token of `state`. For these a bare
// omitted count cannot answer "was anything of this kind cut?": retention keeps a prefix of assembly
// order, so the omitted tail can be entirely one status, and a reader checking a status that
// rendered nothing cannot tell an empty draw from a budget cut. The composition is counted off the
// `state` field itself, so it stays true for whatever statuses the section's rows actually carry.
const STATUS_COMPOSED_OMISSION_SECTIONS = new Set<EvidenceLedgerSectionId>(["open_questions"]);

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

type OmittedStateTally = Readonly<Record<string, number>>;

type FullLedgerSectionState = {
  section: EvidenceLedgerSection;
  omittedCount: number;
  omittedSpan?: OmittedEntrySpan;
  omittedStates?: OmittedStateTally;
  dropped: boolean;
  retentionPolicy: FullLedgerSectionRetentionPolicy;
};

function entryOccurredAt(entry: EvidenceLedgerEntry): number | null {
  const value = entry.state_metadata?.occurred_at;

  return typeof value === "number" && Number.isFinite(value) ? value : null;
}

function omittedSpanForEntries(entries: readonly EvidenceLedgerEntry[]): OmittedEntrySpan | null {
  const occurredAts = entries
    .map(entryOccurredAt)
    .filter((value): value is number => value !== null);

  if (occurredAts.length === 0) {
    return null;
  }

  return {
    firstOccurredAt: Math.min(...occurredAts),
    lastOccurredAt: Math.max(...occurredAts),
  };
}

// `state` is a space-joined list of `key=value` parts, optionally led by a bare lifecycle token. Only
// the bare token names a status; a `key=value` head (`score=0.42`) is per-row metadata and would mint
// one tally bucket per distinct value, so it is skipped rather than counted.
function entryStateHead(entry: EvidenceLedgerEntry): string | null {
  if (typeof entry.state !== "string") {
    return null;
  }

  const [head] = entry.state.trim().split(/\s+/, 1);

  return head === undefined || head.length === 0 || head.includes("=") ? null : head;
}

function omittedStateTallyForEntries(
  sectionId: EvidenceLedgerSectionId,
  entries: readonly EvidenceLedgerEntry[],
): OmittedStateTally | null {
  if (!STATUS_COMPOSED_OMISSION_SECTIONS.has(sectionId)) {
    return null;
  }

  const tally: Record<string, number> = {};

  for (const entry of entries) {
    const head = entryStateHead(entry);

    if (head !== null) {
      tally[head] = (tally[head] ?? 0) + 1;
    }
  }

  // An empty tally is a result, not an absence: for an opted-in section it says no status-bearing row
  // was cut. Collapsing it to null would render the same as "this section does not report composition".
  return tally;
}

function mergeOmittedStateTallies(
  left: OmittedStateTally | undefined,
  right: OmittedStateTally | null,
): OmittedStateTally | undefined {
  if (right === null) {
    return left;
  }

  if (left === undefined) {
    return right;
  }

  const merged: Record<string, number> = { ...left };

  for (const [state, count] of Object.entries(right)) {
    merged[state] = (merged[state] ?? 0) + count;
  }

  return merged;
}

function formatOmittedStateTally(tally: OmittedStateTally | undefined): string {
  if (tally === undefined) {
    return "";
  }

  const parts = Object.entries(tally)
    .sort(([left], [right]) => left.localeCompare(right))
    .map(([state, count]) => `${state} ${count}`);

  return parts.length === 0
    ? " Omitted by state: none of the omitted entries carried a status, so no status had rows cut here."
    : ` Omitted by state: ${parts.join(", ")}. A status absent from that list had no rows cut here, so its absence from the section above is an empty draw rather than a budget cut.`;
}

function mergeOmittedSpans(
  left: OmittedEntrySpan | undefined,
  right: OmittedEntrySpan | null,
): OmittedEntrySpan | undefined {
  if (right === null) {
    return left;
  }

  if (left === undefined) {
    return right;
  }

  return {
    firstOccurredAt: Math.min(left.firstOccurredAt, right.firstOccurredAt),
    lastOccurredAt: Math.max(left.lastOccurredAt, right.lastOccurredAt),
  };
}

function recentLivedExperienceKind(entry: EvidenceLedgerEntry): string | null {
  const value = entry.state_metadata?.lived_experience_kind;

  return typeof value === "string" && value.length > 0 ? value : null;
}

function isRecentLivedExperienceSpineEntry(entry: EvidenceLedgerEntry): boolean {
  return isRecentLivedExperienceSpineKind(recentLivedExperienceKind(entry));
}

function compareEntriesByOccurredAt(left: EvidenceLedgerEntry, right: EvidenceLedgerEntry): number {
  const leftOccurredAt = entryOccurredAt(left) ?? 0;
  const rightOccurredAt = entryOccurredAt(right) ?? 0;

  return leftOccurredAt - rightOccurredAt || left.id.localeCompare(right.id);
}

function recentLivedExperienceBreadcrumbDisclosure(
  section: EvidenceLedgerSection,
  stateMetadata: Record<string, unknown>,
): Pick<EvidenceLedgerEntry, "state" | "state_metadata"> {
  if (section.id !== "recent_lived_experience") {
    return {
      state: "omitted",
    };
  }

  return {
    state: appendMemoryDisclosureState({
      state: "omitted",
      disclosureLabel: RECENT_LIVED_EXPERIENCE_BREADCRUMB_DISCLOSURE_LABEL,
    }),
    state_metadata: appendMemoryDisclosureStateMetadata({
      stateMetadata,
      disclosureLabel: RECENT_LIVED_EXPERIENCE_BREADCRUMB_DISCLOSURE_LABEL,
    }),
  };
}

function fullLedgerOmittedEntry(
  section: EvidenceLedgerSection,
  omittedCount: number,
  retentionPolicy: FullLedgerSectionRetentionPolicy,
  omittedSpan?: OmittedEntrySpan,
  omittedStates?: OmittedStateTally,
): EvidenceLedgerEntry {
  const omittedKind = retentionPolicy === "tail" ? "older" : "lower-priority";
  const text =
    section.id === "recent_lived_experience"
      ? omittedSpan === undefined
        ? `Recent lived experience omitted ${omittedCount} detail entries; retained dated spine entries carry the elapsed span.`
        : `Recent lived experience omitted ${omittedCount} detail entries from ${formatUtcDaySpanLabel(
            omittedSpan.firstOccurredAt,
            omittedSpan.lastOccurredAt,
          )}; omitted detail is summarized in the dated spine above.`
      : `Evidence ledger omitted ${omittedCount} ${omittedKind} entries from ${section.id} to stay within the finalizer ledger budget.${formatOmittedStateTally(
          omittedStates,
        )}`;
  const disclosure = recentLivedExperienceBreadcrumbDisclosure(section, {
    breadcrumb_kind: "recent_lived_experience_omission",
    omitted_entry_count: omittedCount,
    ...(omittedSpan === undefined
      ? {}
      : {
          first_omitted_occurred_at: omittedSpan.firstOccurredAt,
          last_omitted_occurred_at: omittedSpan.lastOccurredAt,
        }),
  });

  return {
    id: `evidence_ledger_omitted:${section.id}`,
    source_type: "system_metadata",
    session_scope: "global",
    actor: "system",
    trust_rank: 0,
    ...disclosure,
    text,
    taint: "none",
  };
}

function fullLedgerDroppedSectionEntry(
  section: EvidenceLedgerSection,
  omittedCount: number,
  omittedStates?: OmittedStateTally,
): EvidenceLedgerEntry {
  const disclosure = recentLivedExperienceBreadcrumbDisclosure(section, {
    breadcrumb_kind: "recent_lived_experience_dropped_section",
    omitted_entry_count: omittedCount,
  });

  return {
    id: `evidence_ledger_dropped_section:${section.id}`,
    source_type: "system_metadata",
    session_scope: "global",
    actor: "system",
    trust_rank: 0,
    ...disclosure,
    text: `Evidence ledger dropped all entries from ${section.id} to stay within the global hard cap: entries=${omittedCount}.${formatOmittedStateTally(
      omittedStates,
    )}`,
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
          : [
              fullLedgerDroppedSectionEntry(
                state.section,
                state.omittedCount,
                state.omittedStates,
              ),
            ],
    };
  }

  return {
    ...state.section,
    entries:
      state.omittedCount <= 0
        ? state.section.entries
        : [
            ...state.section.entries,
            fullLedgerOmittedEntry(
              state.section,
              state.omittedCount,
              state.retentionPolicy,
              state.omittedSpan,
              state.omittedStates,
            ),
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
  if (sectionId === "recent_lived_experience") {
    return "spine";
  }

  return TAIL_PRESERVING_FULL_LEDGER_SECTIONS.has(sectionId) ? "tail" : "head";
}

function renderedSectionWithOmission(input: {
  section: EvidenceLedgerSection;
  entries: readonly EvidenceLedgerEntry[];
  omittedCount: number;
  retentionPolicy: FullLedgerSectionRetentionPolicy;
  omittedSpan?: OmittedEntrySpan;
}): string {
  return renderSection({
    ...input.section,
    entries:
      input.omittedCount <= 0
        ? [...input.entries]
        : [
            ...input.entries,
            fullLedgerOmittedEntry(
              input.section,
              input.omittedCount,
              input.retentionPolicy,
              input.omittedSpan,
            ),
          ],
  });
}

function capRecentLivedExperienceSection(input: {
  section: EvidenceLedgerSection;
  maxEntryTextTokens: number;
  options: FullEvidenceLedgerSectionOptions;
}): FullLedgerSectionState {
  const compactedEntries = input.section.entries
    .map((entry) => compactFullLedgerEntry(entry, input.maxEntryTextTokens))
    .sort(compareEntriesByOccurredAt);
  const spineEntries = compactedEntries.filter(isRecentLivedExperienceSpineEntry);
  const detailEntries = compactedEntries.filter(
    (entry) => !isRecentLivedExperienceSpineEntry(entry),
  );
  let includedSpineEntries = spineEntries;
  let includedDetailEntries: EvidenceLedgerEntry[] = [];
  let omittedCount = 0;
  let omittedSpan: OmittedEntrySpan | undefined;

  if (includedSpineEntries.length > input.options.maxEntries) {
    const omittedSpineEntries = includedSpineEntries.slice(
      0,
      includedSpineEntries.length - input.options.maxEntries,
    );

    omittedCount += omittedSpineEntries.length;
    omittedSpan = mergeOmittedSpans(omittedSpan, omittedSpanForEntries(omittedSpineEntries));
    includedSpineEntries = includedSpineEntries.slice(-input.options.maxEntries);
  }

  const detailCapacity = Math.max(0, input.options.maxEntries - includedSpineEntries.length);

  if (detailCapacity > 0) {
    includedDetailEntries = detailEntries.slice(-detailCapacity);
  }

  const omittedDetailEntries = detailEntries.slice(
    0,
    detailEntries.length - includedDetailEntries.length,
  );
  omittedCount += omittedDetailEntries.length;
  omittedSpan = mergeOmittedSpans(omittedSpan, omittedSpanForEntries(omittedDetailEntries));

  const includedEntries = [...includedSpineEntries, ...includedDetailEntries];

  while (includedEntries.length > 1) {
    const rendered = renderedSectionWithOmission({
      section: input.section,
      entries: includedEntries,
      omittedCount,
      retentionPolicy: "spine",
      omittedSpan,
    });

    if (estimatePromptTokens(rendered) <= input.options.maxTokens) {
      break;
    }

    const detailIndex = includedEntries.findLastIndex(
      (entry) => !isRecentLivedExperienceSpineEntry(entry),
    );
    const removeIndex = detailIndex >= 0 ? detailIndex : includedEntries.length - 1;
    const [removed] = includedEntries.splice(removeIndex, 1);

    if (removed !== undefined) {
      omittedCount += 1;
      omittedSpan = mergeOmittedSpans(omittedSpan, omittedSpanForEntries([removed]));
    }
  }

  return {
    section: {
      ...input.section,
      entries: includedEntries,
    },
    omittedCount,
    omittedSpan,
    dropped: false,
    retentionPolicy: "spine",
  };
}

function capFullLedgerSection(input: {
  section: EvidenceLedgerSection;
  maxEntryTextTokens: number;
  options: FullEvidenceLedgerSectionOptions;
}): FullLedgerSectionState {
  const retentionPolicy = fullLedgerSectionRetentionPolicy(input.section.id);

  if (retentionPolicy === "spine") {
    return capRecentLivedExperienceSection(input);
  }

  const entries =
    retentionPolicy === "tail"
      ? input.section.entries.slice(-input.options.maxEntries)
      : input.section.entries.slice(0, input.options.maxEntries);
  const entryCountCutEntries =
    retentionPolicy === "tail"
      ? input.section.entries.slice(0, Math.max(0, input.section.entries.length - entries.length))
      : input.section.entries.slice(entries.length);
  const compactedEntries = entries.map((entry) =>
    compactFullLedgerEntry(entry, input.maxEntryTextTokens),
  );
  let includedEntries: EvidenceLedgerEntry[] = [];
  let omittedCount = Math.max(0, input.section.entries.length - compactedEntries.length);
  let omittedStates = mergeOmittedStateTallies(
    undefined,
    omittedStateTallyForEntries(input.section.id, entryCountCutEntries),
  );

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
                fullLedgerOmittedEntry(
                  candidateSection,
                  omittedCount,
                  retentionPolicy,
                  undefined,
                  omittedStates,
                ),
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
      omittedStates = mergeOmittedStateTallies(
        omittedStates,
        omittedStateTallyForEntries(input.section.id, compactedEntries.slice(0, index + 1)),
      );
      break;
    }

    return {
      section: {
        ...input.section,
        entries: includedEntries,
      },
      omittedCount,
      ...(omittedStates === undefined ? {} : { omittedStates }),
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
              fullLedgerOmittedEntry(
                candidateSection,
                omittedCount,
                retentionPolicy,
                undefined,
                omittedStates,
              ),
            ],
    });

    if (estimatePromptTokens(rendered) <= input.options.maxTokens || includedEntries.length === 0) {
      includedEntries = candidateEntries;
      continue;
    }

    omittedCount += compactedEntries.length - index;
    omittedStates = mergeOmittedStateTallies(
      omittedStates,
      omittedStateTallyForEntries(input.section.id, compactedEntries.slice(index)),
    );
    break;
  }

  return {
    section: {
      ...input.section,
      entries: includedEntries,
    },
    omittedCount,
    ...(omittedStates === undefined ? {} : { omittedStates }),
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
    const recentDetailRemoveIndex = state.section.entries.findLastIndex(
      (entry) => !isRecentLivedExperienceSpineEntry(entry),
    );
    const removeIndex =
      state.retentionPolicy === "tail"
        ? 0
        : state.retentionPolicy === "spine"
          ? recentDetailRemoveIndex >= 0
            ? recentDetailRemoveIndex
            : state.section.entries.length - 1
          : state.section.entries.length - 1;
    const removedEntries =
      removeIndex < state.section.entries.length ? [state.section.entries[removeIndex]!] : [];

    state.section = {
      ...state.section,
      entries: state.section.entries.filter((_, index) => index !== removeIndex),
    };
    state.omittedCount += 1;
    state.omittedSpan = mergeOmittedSpans(state.omittedSpan, omittedSpanForEntries(removedEntries));
    state.omittedStates = mergeOmittedStateTallies(
      state.omittedStates,
      omittedStateTallyForEntries(state.section.id, removedEntries),
    );
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
    state.omittedSpan = mergeOmittedSpans(
      state.omittedSpan,
      omittedSpanForEntries(state.section.entries),
    );
    state.omittedStates = mergeOmittedStateTallies(
      state.omittedStates,
      omittedStateTallyForEntries(state.section.id, state.section.entries),
    );
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
  const dedupedLedger = reconcileAutobiographicalRecallCapMetadata({
    factLedger: ledger,
    renderedLedger: deduped.ledger,
  });
  const postDedupeTokens = estimateEvidenceLedgerTokens(dedupedLedger);
  const states = compactFullLedgerSections(dedupedLedger, options);
  const preCapTokens = postDedupeTokens;
  const postSectionCapTokens = totalFullLedgerPromptTokens(dedupedLedger, states);
  let droppedSections: EvidenceLedgerSectionId[] = [];

  if (postSectionCapTokens > hardCapTokens) {
    droppedSections = dropFullLedgerSectionsToHardCap(dedupedLedger, states, hardCapTokens);
  } else if (postSectionCapTokens > targetTokens) {
    trimFullLedgerToTarget(dedupedLedger, states, targetTokens);
  }

  const compactedLedger = reconcileAutobiographicalRecallCapMetadata({
    factLedger: dedupedLedger,
    renderedLedger: materializeFullLedgerStates(dedupedLedger, states),
  });
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
