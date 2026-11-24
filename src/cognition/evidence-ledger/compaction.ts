import { estimatePromptTokens } from "../../util/token-estimate.js";
import { coercePositiveIntegerOrFallback } from "../../util/math.js";
import { isRecentLivedExperienceSpineKind } from "../../memory/activity/index.js";
import { selfPrivateMemoryDisclosureLabel } from "../../memory/common/disclosure-label.js";
import { admitAnnotationRows } from "../../memory/common/evidence-prompt.js";
import { allSectionIds, emptySectionCountRecord } from "./budget.js";
import { estimateEvidenceLedgerTokens, cloneLedgerWithSections } from "./ledger-copy.js";
import {
  dedupeEvidenceLedgerByProvenance,
  foldedEntryKeysWithEvidenceOffPage,
  provenanceDedupeEntryKey,
  type ProvenanceDedupeGrouping,
} from "./provenance-dedupe.js";
import { reconcileAutobiographicalRecallCapMetadata } from "./autobiographical-recall-cap-metadata.js";
import { estimateEvidenceLedgerPromptTokens } from "./finalizer-ledger.js";
import { renderEntry, renderSection } from "./section-rendering.js";
import {
  ledgerAnnotationCount,
  ledgerAnnotations,
  withLedgerAnnotations,
} from "./record-annotations.js";
import {
  ledgerCorrectionCount,
  ledgerCorrectionRows,
  withLedgerCorrectionRows,
} from "./resolution-correction-budget.js";
import {
  reconcileOmissionAccounting,
  omissionCounts,
  mergeOmissionCounts,
  SECTION_ACCOUNTING_MAX_CHARS,
} from "./omission-accounting.js";
import {
  appendMemoryDisclosureState,
  appendMemoryDisclosureStateMetadata,
} from "./entry-metadata.js";
import type {
  EvidenceLedger,
  EvidenceLedgerEntry,
  EvidenceLedgerSection,
  EvidenceLedgerSectionId,
  EvidenceLedgerGlobalAccounting,
  EvidenceLedgerOmissionCounts,
  EvidenceLedgerOmissionCountsBySection,
  EvidenceLedgerBudgetOutcome,
} from "./types.js";

export type FullEvidenceLedgerSectionOptions = {
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
  accountingOmittedFromPage?: EvidenceLedgerOmissionCountsBySection;
  budgetOutcome: EvidenceLedgerBudgetOutcome;
};

export type CompactedEvidenceLedger = {
  ledger: EvidenceLedger;
  traceSummary: EvidenceLedgerCompactionTraceSummary;
};

const DEFAULT_FULL_LEDGER_TARGET_TOKENS = 60_000;
const DEFAULT_FULL_LEDGER_HARD_CAP_TOKENS = 100_000;
const DEFAULT_FULL_LEDGER_ENTRY_TEXT_TOKEN_CAP = 1_200;
export const REQUIRED_ENTRY_TEXT_STUB = "\n[truncated]";

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
  if (
    entry.omission_accounting !== undefined &&
    entry.omission_accounting.action_threads === undefined
  )
    return entry;
  return {
    ...entry,
    text: truncateTextForFullEvidenceLedger(entry.text, maxEntryTextTokens),
    value: truncateTextForFullEvidenceLedger(entry.value, Math.max(80, maxEntryTextTokens / 4)),
  };
}

type OmittedStateTally = Readonly<Record<string, number>>;

export type FullLedgerSectionState = {
  section: EvidenceLedgerSection;
  omittedCount: number;
  omittedSpan?: OmittedEntrySpan;
  omittedStates?: OmittedStateTally;
  dropped: boolean;
  retentionPolicy: FullLedgerSectionRetentionPolicy;
  sourceEntries?: readonly EvidenceLedgerEntry[];
  sourceAccounting?: readonly EvidenceLedgerEntry[];
  accountingMode?: "section" | "global";
  sectionTokenLimit?: number;
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
  _retentionPolicy: FullLedgerSectionRetentionPolicy,
  _omittedSpan?: OmittedEntrySpan,
  omittedStates?: OmittedStateTally,
): EvidenceLedgerEntry {
  return {
    id: `evidence_ledger_omitted:${section.id}`,
    source_type: "system_metadata",
    session_scope: "global",
    actor: "system",
    trust_rank: 0,
    ...recentLivedExperienceBreadcrumbDisclosure(section, {
      breadcrumb_kind: "recent_lived_experience_omission",
      omitted_entry_count: omittedCount,
    }),
    omission_accounting: {
      counts: {
        budget_entries: omittedCount,
        ...Object.fromEntries(
          Object.entries(omittedStates ?? {}).map(([state, count]) => [`state_${state}`, count]),
        ),
      },
    },
    taint: "none",
  };
}

export function sectionAccountingEntry(
  state: FullLedgerSectionState,
): EvidenceLedgerEntry | undefined {
  const retainedIds = new Set(state.section.entries.map((entry) => entry.id));
  const removedNotes = (state.sourceEntries ?? []).filter(
    (entry) => entry.omission_counts !== undefined && !retainedIds.has(entry.id),
  );
  const prior = reconcileOmissionAccounting({
    ...state.section,
    entries: [...state.section.entries, ...(state.sourceAccounting ?? [])],
  }).entries.filter((entry) => entry.omission_accounting !== undefined);
  const counts = mergeOmissionCounts(
    ...prior.map(omissionCounts),
    ...removedNotes.map((entry) => entry.omission_counts),
  );
  const annotationTotal = Math.max(
    counts.annotations_count ?? 0,
    (state.sourceEntries ?? []).reduce((sum, entry) => sum + ledgerAnnotationCount(entry), 0),
  );
  const annotationsOmitted =
    annotationTotal -
    state.section.entries.reduce((sum, entry) => sum + ledgerAnnotations(entry).length, 0);
  if (annotationsOmitted > 0) {
    counts.annotations_count = annotationTotal;
    counts.annotations_omitted = annotationsOmitted;
  } else {
    delete counts.annotations_count;
    delete counts.annotations_omitted;
  }
  // A second compaction carries the earlier population total; it must not add it again.
  const correctionTotal = Math.max(
    counts.resolution_correction_count ?? 0,
    (state.sourceEntries ?? []).reduce((sum, entry) => sum + ledgerCorrectionCount(entry), 0),
  );
  const correctionOmitted =
    correctionTotal -
    state.section.entries.reduce((sum, entry) => sum + ledgerCorrectionRows(entry).length, 0);
  if (correctionOmitted > 0) {
    counts.resolution_correction_count = correctionTotal;
    counts.resolution_corrections_omitted_count = correctionOmitted;
  } else {
    delete counts.resolution_correction_count;
    delete counts.resolution_corrections_omitted_count;
  }
  const reviewLabelTotal = Math.max(
    counts.review_label_count ?? 0,
    (state.sourceEntries ?? []).filter((entry) => entry.state_metadata?.review !== undefined)
      .length,
  );
  const reviewLabelsOmitted =
    reviewLabelTotal -
    state.section.entries.filter((entry) => entry.state_metadata?.review !== undefined).length;
  if (reviewLabelsOmitted > 0) {
    counts.review_label_count = reviewLabelTotal;
    counts.review_labels_omitted_count = reviewLabelsOmitted;
  } else {
    delete counts.review_label_count;
    delete counts.review_labels_omitted_count;
  }
  if (state.omittedCount === 0 && Object.keys(counts).length === 0) return undefined;
  const entry = fullLedgerOmittedEntry(
    state.section,
    state.omittedCount,
    state.retentionPolicy,
    state.omittedSpan,
    state.omittedStates,
  );
  const action = prior.find((entry) => entry.omission_accounting?.action_threads !== undefined);
  return {
    ...entry,
    id:
      action?.id ??
      (state.dropped ? `evidence_ledger_dropped_section:${state.section.id}` : entry.id),
    omission_accounting: {
      ...action?.omission_accounting,
      counts: mergeOmissionCounts(
        counts,
        entry.omission_accounting?.counts,
        removedNotes.length === 0 ? undefined : { budget_note_entries: removedNotes.length },
      ),
    },
  };
}

export function materializeFullLedgerSectionState(
  state: FullLedgerSectionState,
): EvidenceLedgerSection {
  const candidate = sectionAccountingEntry(state);
  const accounting =
    state.accountingMode === "global" ||
    (candidate !== undefined && renderEntry(candidate).length > SECTION_ACCOUNTING_MAX_CHARS)
      ? undefined
      : candidate;
  return reconcileOmissionAccounting({
    ...state.section,
    entries: [...state.section.entries, ...(accounting === undefined ? [] : [accounting])],
  });
}

function materializeFullLedgerStates(
  ledger: EvidenceLedger,
  states: readonly FullLedgerSectionState[],
  accounting?: EvidenceLedgerGlobalAccounting,
): EvidenceLedger {
  return cloneLedgerWithSections(
    { ...ledger, compaction_accounting: accounting },
    states.map(materializeFullLedgerSectionState),
  );
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
    includedSpineEntries =
      input.options.maxEntries === 0 ? [] : includedSpineEntries.slice(-input.options.maxEntries);
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

export function isRequiredLedgerEntry(
  _section: EvidenceLedgerSection,
  entry: EvidenceLedgerEntry,
): boolean {
  return entry.source_type === "current_user_message";
}

/** Measure the complete rendered surface, including metadata, section framing and tag escaping. */
function fitRequiredEntryText(
  states: readonly FullLedgerSectionState[],
  capTokens: number,
  measure: () => number,
): void {
  if (measure() <= capTokens) return;
  for (const state of [...states].reverse()) {
    for (let index = state.section.entries.length - 1; index >= 0; index--) {
      const entry = state.section.entries[index]!;
      const text = entry.text;
      if (
        !isRequiredLedgerEntry(state.section, entry) ||
        text === undefined ||
        text.length <= REQUIRED_ENTRY_TEXT_STUB.length
      )
        continue;
      const characters = Array.from(text);
      const retainPrefix = (length: number) => {
        state.section.entries[index] = {
          ...entry,
          text: characters.slice(0, length).join("") + REQUIRED_ENTRY_TEXT_STUB,
        };
      };
      retainPrefix(0);
      // Keep the handle and explicit stub even when their framing cannot fit; report that outcome.
      if (measure() > capTokens) continue;
      let lower = 0;
      let upper = characters.length - 1;
      while (lower < upper) {
        const middle = Math.ceil((lower + upper) / 2);
        retainPrefix(middle);
        if (measure() <= capTokens) lower = middle;
        else upper = middle - 1;
      }
      retainPrefix(lower);
      return;
    }
  }
}

export function ledgerBudgetOutcome(
  states: readonly FullLedgerSectionState[],
  pageTokens: number,
  pageCapTokens: number,
): EvidenceLedgerBudgetOutcome {
  const violations: Extract<EvidenceLedgerBudgetOutcome, { status: "over_budget" }>["violations"] =
    [];
  for (const state of states) {
    const tokens = estimatePromptTokens(renderSection(materializeFullLedgerSectionState(state)));
    if (state.sectionTokenLimit !== undefined && tokens > state.sectionTokenLimit) {
      violations.push({
        scope: "section",
        section_id: state.section.id,
        cap_tokens: state.sectionTokenLimit,
        rendered_tokens: tokens,
        required_entry_ids: state.section.entries
          .filter((entry) => isRequiredLedgerEntry(state.section, entry))
          .map((entry) => entry.id),
      });
    }
  }
  if (pageTokens > pageCapTokens)
    violations.push({
      scope: "page",
      cap_tokens: pageCapTokens,
      rendered_tokens: pageTokens,
      required_entry_ids: states.flatMap((state) =>
        state.section.entries
          .filter((entry) => isRequiredLedgerEntry(state.section, entry))
          .map((entry) => entry.id),
      ),
    });
  return violations.length === 0
    ? { status: "within_budget" }
    : { status: "over_budget", reason: "irreducible_framing", violations };
}

export function capFullLedgerSection(input: {
  section: EvidenceLedgerSection;
  maxEntryTextTokens: number;
  options: FullEvidenceLedgerSectionOptions;
  carriedAccounting?: EvidenceLedgerOmissionCounts;
}): FullLedgerSectionState {
  const accounting = input.section.entries.filter(
    (entry) => entry.omission_accounting !== undefined,
  );
  if (input.carriedAccounting !== undefined) {
    accounting.push({
      ...fullLedgerOmittedEntry(
        input.section,
        0,
        fullLedgerSectionRetentionPolicy(input.section.id),
      ),
      omission_accounting: { counts: input.carriedAccounting },
    });
  }
  const ordinary = input.section.entries.filter((entry) => entry.omission_accounting === undefined);
  const required = ordinary.filter((entry) => isRequiredLedgerEntry(input.section, entry));
  const state = capOrdinaryFullLedgerSection({
    ...input,
    section: {
      ...input.section,
      entries: ordinary
        .filter((entry) => !isRequiredLedgerEntry(input.section, entry))
        .map((entry) => withLedgerAnnotations(withLedgerCorrectionRows(entry, []), [])),
    },
    options: {
      ...input.options,
      maxEntries: Math.max(
        0,
        input.options.maxEntries - required.length - Number(accounting.length > 0),
      ),
    },
  });
  state.sourceEntries = ordinary;
  state.sourceAccounting = accounting;
  state.sectionTokenLimit = input.options.maxTokens;
  state.section.entries = [
    ...required.map((entry) =>
      withLedgerAnnotations(
        withLedgerCorrectionRows(compactFullLedgerEntry(entry, input.maxEntryTextTokens), []),
        [],
      ),
    ),
    ...state.section.entries,
  ];
  admitResolutionCorrections(state);
  const account = sectionAccountingEntry(state);
  if (
    input.section.omitted_from_prompt ||
    (account !== undefined && renderEntry(account).length > SECTION_ACCOUNTING_MAX_CHARS)
  )
    state.accountingMode = "global";
  fitSectionAccounting(state);
  admitResolutionCorrections(state);
  admitRecordAnnotations(state);
  return state;
}

function fitSectionAccounting(state: FullLedgerSectionState): void {
  const tokens = () =>
    estimatePromptTokens(renderSection(materializeFullLedgerSectionState(state)));
  const limit = state.sectionTokenLimit!;
  while (tokens() > limit) {
    if (removeLowestPriorityAnnotation(state)) continue;
    if (removeLowestPriorityCorrection(state)) continue;
    if (!removeLowestPriorityOrdinaryEntry(state)) break;
  }
  if (tokens() > limit) {
    state.accountingMode = "global";
    if (state.section.entries.length === 0) state.section.omitted_from_prompt = true;
  }
  fitRequiredEntryText([state], limit, tokens);
}

/** Annotation text is atomic; spend only the room left after the existing record compaction. */
function admitRecordAnnotations(state: FullLedgerSectionState): void {
  const sources = new Map((state.sourceEntries ?? []).map((entry) => [entry.id, entry]));
  const rows = state.section.entries.map((entry) =>
    ledgerAnnotations(sources.get(entry.id) ?? entry),
  );
  if (rows.every((annotations) => annotations.length === 0)) return;
  const base = state.section.entries.map((entry) => withLedgerAnnotations(entry, []));
  const fits = () =>
    estimatePromptTokens(renderSection(materializeFullLedgerSectionState(state))) <=
    state.sectionTokenLimit!;
  state.section.entries = base.map((entry, index) => withLedgerAnnotations(entry, rows[index]!));
  if (fits()) return;
  state.section.entries = [...base];
  for (let index = 0; index < base.length; index++) {
    admitAnnotationRows(rows[index]!, (admitted) => {
      const previous = state.section.entries[index]!;
      state.section.entries[index] = withLedgerAnnotations(base[index]!, admitted);
      if (fits()) return true;
      state.section.entries[index] = previous;
      return false;
    });
  }
}

function removeLowestPriorityAnnotation(state: FullLedgerSectionState): boolean {
  for (let index = state.section.entries.length - 1; index >= 0; index--) {
    const entry = state.section.entries[index]!;
    const rows = ledgerAnnotations(entry);
    if (rows.length === 0) continue;
    state.section.entries[index] = withLedgerAnnotations(entry, rows.slice(0, -1));
    return true;
  }
  return false;
}

/** Keep note envelopes first; only complete, labelled correction rows use the remaining space. */
function admitResolutionCorrections(state: FullLedgerSectionState): void {
  const sources = new Map((state.sourceEntries ?? []).map((entry) => [entry.id, entry]));
  const rows = state.section.entries.map((entry) =>
    [...ledgerCorrectionRows(sources.get(entry.id) ?? entry)].sort(
      (left, right) => right.seq - left.seq,
    ),
  );
  if (rows.every((corrections) => corrections.length === 0)) return;
  const base = state.section.entries.map((entry) => withLedgerCorrectionRows(entry, []));
  const fits = () =>
    estimatePromptTokens(renderSection(materializeFullLedgerSectionState(state))) <=
    state.sectionTokenLimit!;
  // Showing all rows can be cheaper than adding a separate omission entry for tiny rows.
  state.section.entries = base.map((entry, index) => withLedgerCorrectionRows(entry, rows[index]!));
  if (fits()) return;
  state.section.entries = [...base];
  for (let index = 0; index < base.length; index++) {
    for (let count = 1; count <= rows[index]!.length; count++) {
      const previous = state.section.entries[index]!;
      state.section.entries[index] = withLedgerCorrectionRows(
        base[index]!,
        rows[index]!.slice(0, count),
      );
      if (fits()) continue;
      state.section.entries[index] = previous;
      break;
    }
  }
}

function removeLowestPriorityCorrection(state: FullLedgerSectionState): boolean {
  for (let index = state.section.entries.length - 1; index >= 0; index--) {
    const entry = state.section.entries[index]!;
    const rows = ledgerCorrectionRows(entry);
    if (rows.length === 0) continue;
    state.section.entries[index] = withLedgerCorrectionRows(entry, rows.slice(0, -1));
    return true;
  }
  return false;
}

function capOrdinaryFullLedgerSection(input: {
  section: EvidenceLedgerSection;
  maxEntryTextTokens: number;
  options: FullEvidenceLedgerSectionOptions;
}): FullLedgerSectionState {
  const retentionPolicy = fullLedgerSectionRetentionPolicy(input.section.id);

  if (retentionPolicy === "spine") {
    return capRecentLivedExperienceSection(input);
  }

  const entries =
    input.options.maxEntries === 0
      ? []
      : retentionPolicy === "tail"
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
      carriedAccounting: ledger.compaction_accounting?.folded_sections[section.id],
    }),
  );
}

export function removeLowestPriorityOrdinaryEntry(state: FullLedgerSectionState): boolean {
  const ordinary = state.section.entries.filter(
    (entry) => !isRequiredLedgerEntry(state.section, entry),
  );
  const removed =
    state.retentionPolicy === "tail"
      ? ordinary[0]
      : state.retentionPolicy === "spine"
        ? (ordinary.findLast((entry) => !isRecentLivedExperienceSpineEntry(entry)) ??
          ordinary.at(-1))
        : ordinary.at(-1);
  if (removed === undefined) return false;
  state.section = {
    ...state.section,
    entries: state.section.entries.filter((entry) => entry !== removed),
  };
  state.omittedCount += 1;
  state.omittedSpan = mergeOmittedSpans(state.omittedSpan, omittedSpanForEntries([removed]));
  state.omittedStates = mergeOmittedStateTallies(
    state.omittedStates,
    omittedStateTallyForEntries(state.section.id, [removed]),
  );
  return true;
}

/** Required content, global counts, section counts, then ordinary entries. */
export function fitLedgerAccounting(input: {
  states: FullLedgerSectionState[];
  targetTokens: number;
  hardCapTokens: number;
  measure: (accounting: EvidenceLedgerGlobalAccounting | undefined) => number;
  sectionOrder?: readonly EvidenceLedgerSectionId[];
  dropSectionsAtHardCap?: boolean;
}): {
  accounting?: EvidenceLedgerGlobalAccounting;
  accountingOmittedFromPage: EvidenceLedgerOmissionCountsBySection;
} {
  const ordered = (input.sectionOrder ?? LOWEST_TRUST_SECTION_ORDER).flatMap((id) =>
    input.states.filter((state) => state.section.id === id),
  );
  const counts = () =>
    Object.fromEntries(
      input.states.flatMap((state) => {
        const entry = sectionAccountingEntry(state);
        return entry === undefined ? [] : [[state.section.id, omissionCounts(entry)]];
      }),
    ) as EvidenceLedgerOmissionCountsBySection;
  const global = (): EvidenceLedgerGlobalAccounting | undefined => {
    const all = counts();
    const folded = Object.fromEntries(
      input.states
        .filter((state) => {
          const entry = sectionAccountingEntry(state);
          return (
            entry !== undefined &&
            (state.accountingMode === "global" ||
              renderEntry(entry).length > SECTION_ACCOUNTING_MAX_CHARS)
          );
        })
        .map((state) => [state.section.id, all[state.section.id]!]),
    ) as EvidenceLedgerOmissionCountsBySection;
    const budgetEntries = Object.values(all).reduce(
      (total, count) => total + (count?.budget_entries ?? 0),
      0,
    );
    return budgetEntries === 0 && Object.keys(folded).length === 0
      ? undefined
      : { budget_entries: budgetEntries, folded_sections: folded };
  };
  const hardDrop = input.dropSectionsAtHardCap && input.measure(global()) > input.hardCapTokens;
  const limit = hardDrop ? input.hardCapTokens : Math.min(input.targetTokens, input.hardCapTokens);
  while (input.measure(global()) > limit) {
    const state = ordered.find((state) =>
      state.section.entries.some((entry) => !isRequiredLedgerEntry(state.section, entry)),
    );
    if (state === undefined) break;
    // A global cut can omit a correction without erasing the note or its correction counts.
    if (removeLowestPriorityAnnotation(state) || removeLowestPriorityCorrection(state)) {
      // The first omitted correction adds accounting framing, which also has to fit its section.
      fitSectionAccounting(state);
      continue;
    }
    if (hardDrop) {
      while (removeLowestPriorityOrdinaryEntry(state)) {
        /* whole section */
      }
      state.dropped = true;
    } else removeLowestPriorityOrdinaryEntry(state);
  }
  // Empty framing is lower priority than counts and required content too.
  for (const state of ordered) {
    if (input.measure(global()) <= input.hardCapTokens) break;
    state.accountingMode = "global";
    if (state.section.entries.length === 0) {
      state.section.omitted_from_prompt = true;
    }
  }
  if (input.measure(global()) <= input.hardCapTokens)
    return { accounting: global(), accountingOmittedFromPage: {} };
  // The page carries no omission count here; the trace carries it. Required entries retain
  // their handles and an explicit truncation stub; irreducible overflow is a budget outcome.
  for (const state of input.states) {
    state.accountingMode = "global";
    if (state.section.entries.length === 0) state.section.omitted_from_prompt = true;
  }
  fitRequiredEntryText(input.states, input.hardCapTokens, () => input.measure(undefined));
  return { accountingOmittedFromPage: counts() };
}

function dedupeAndCapFullLedger(
  ledger: EvidenceLedger,
  options: EvidenceLedgerCompactionOptions,
  groupingByEntryKey: ReadonlyMap<string, ProvenanceDedupeGrouping>,
) {
  const deduped = dedupeEvidenceLedgerByProvenance(ledger, { groupingByEntryKey });
  const dedupedLedger = reconcileAutobiographicalRecallCapMetadata({
    factLedger: ledger,
    renderedLedger: deduped.ledger,
  });

  return { deduped, dedupedLedger, states: compactFullLedgerSections(dedupedLedger, options) };
}

function cutFoldGroupEntryKeys(
  foldedEntryKeysByCanonicalKey: ReadonlyMap<string, readonly string[]>,
  keptEntryKeys: ReadonlySet<string>,
): string[] {
  return [...foldedEntryKeysByCanonicalKey].flatMap(([canonicalKey, foldedEntryKeys]) =>
    keptEntryKeys.has(canonicalKey) ? [] : [canonicalKey, ...foldedEntryKeys],
  );
}

function moveEntryKeysDown(
  groupingByEntryKey: Map<string, ProvenanceDedupeGrouping>,
  entryKeys: readonly string[],
): boolean {
  let moved = false;

  for (const key of entryKeys) {
    const grouping = groupingByEntryKey.get(key);

    if (grouping !== "section_object") {
      groupingByEntryKey.set(key, grouping === undefined ? "object" : "section_object");
      moved = true;
    }
  }

  return moved;
}

// Moves each row of a cut canonical's fold group one grouping down; once no cut moves a row, moves
// each row a kept canonical folded whose evidence is off the page instead. Reports whether any moved.
function moveCutFoldGroupsDown(
  ledger: EvidenceLedger,
  groupingByEntryKey: Map<string, ProvenanceDedupeGrouping>,
  pass: ReturnType<typeof dedupeAndCapFullLedger>,
): boolean {
  const keptEntryKeys = new Set(
    pass.states.flatMap((state) =>
      state.section.entries.map((entry) => provenanceDedupeEntryKey(state.section.id, entry.id)),
    ),
  );
  const { foldedEntryKeysByCanonicalKey } = pass.deduped;

  return (
    moveEntryKeysDown(
      groupingByEntryKey,
      cutFoldGroupEntryKeys(foldedEntryKeysByCanonicalKey, keptEntryKeys),
    ) ||
    moveEntryKeysDown(
      groupingByEntryKey,
      foldedEntryKeysWithEvidenceOffPage(ledger, foldedEntryKeysByCanonicalKey, keptEntryKeys),
    )
  );
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
  // A folded row reaches the page only through its canonical, and the fold runs before the section
  // caps, so a canonical the cap cuts would take its folded rows off the page with no count. A cut
  // canonical and the rows it folded group in the next pass only with rows rendering the same
  // object, and a group cut again at that grouping only with those rows in its own section: split
  // from them, a cut row's own duplicate would stand in that section in place of a kept row.
  // A cut at the last grouping takes one memory out of its own section, counted there once, and
  // moves nothing; every earlier cut moves rows down, so the passes end. A kept canonical's folded
  // row whose evidence the page does not show moves down the same way: it shares one handle with
  // the canonical, and the rest of what it says would leave the page under that row.
  const groupingByEntryKey = new Map<string, ProvenanceDedupeGrouping>();
  let pass = dedupeAndCapFullLedger(ledger, options, groupingByEntryKey);

  while (moveCutFoldGroupsDown(ledger, groupingByEntryKey, pass)) {
    pass = dedupeAndCapFullLedger(ledger, options, groupingByEntryKey);
  }

  const { deduped, dedupedLedger, states } = pass;
  const postDedupeTokens = estimateEvidenceLedgerTokens(dedupedLedger);
  const preCapTokens = postDedupeTokens;
  const postSectionCapTokens = estimateEvidenceLedgerTokens(
    materializeFullLedgerStates(dedupedLedger, states),
  );
  const fitted = fitLedgerAccounting({
    states,
    targetTokens,
    hardCapTokens,
    dropSectionsAtHardCap: true,
    measure: (accounting) => {
      const rendered = reconcileAutobiographicalRecallCapMetadata({
        factLedger: dedupedLedger,
        renderedLedger: materializeFullLedgerStates(dedupedLedger, states, accounting),
      });
      // Budget the header at the cap's digit width, so the final estimate cannot add tokens.
      return estimateEvidenceLedgerPromptTokens({ ...rendered, estimatedTokens: hardCapTokens });
    },
  });
  const droppedSections = states
    .filter((state) => state.dropped)
    .map((state) => state.section.id)
    .sort(
      (left, right) =>
        LOWEST_TRUST_SECTION_COMPACTION_PRIORITY[left] -
        LOWEST_TRUST_SECTION_COMPACTION_PRIORITY[right],
    );

  const compactedLedger = reconcileAutobiographicalRecallCapMetadata({
    factLedger: dedupedLedger,
    renderedLedger: materializeFullLedgerStates(dedupedLedger, states, fitted.accounting),
  });
  const postCapTokens = estimateEvidenceLedgerTokens(compactedLedger);
  const finalLedger = { ...compactedLedger, estimatedTokens: postCapTokens };
  const omittedEntryCountsBySection = emptySectionCountRecord();

  for (const state of states) {
    omittedEntryCountsBySection[state.section.id] = state.omittedCount;
  }

  return {
    ledger: finalLedger,
    traceSummary: {
      preDedupeTokens,
      postDedupeTokens,
      preCapTokens,
      postSectionCapTokens,
      postCapTokens,
      dedupedEntryCount: deduped.dedupedEntryCount,
      omittedEntryCountsBySection,
      droppedSections,
      accountingOmittedFromPage: fitted.accountingOmittedFromPage,
      budgetOutcome: ledgerBudgetOutcome(
        states,
        estimateEvidenceLedgerPromptTokens(finalLedger),
        hardCapTokens,
      ),
      targetTokens,
      hardCapTokens,
    },
  };
}
