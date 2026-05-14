import { estimatePromptTokens } from "../../util/token-estimate.js";
import type {
  DecisionArtifact,
  DecisionArtifactEntry,
  DecisionArtifactEntryKind,
} from "../../memory/decision-artifacts/index.js";
import { DECISION_ARTIFACT_ENTRY_KINDS } from "../../memory/decision-artifacts/index.js";
import { UNTRUSTED_DATA_PREAMBLE } from "../deliberation/constants.js";
import { renderTaggedPromptBlock } from "../deliberation/prompt/sections.js";
import type {
  EvidenceLedger,
  EvidenceLedgerEntry,
  EvidenceLedgerSection,
  EvidenceLedgerSectionId,
} from "./types.js";

const COMPACT_PLANNER_LEDGER_SECTION_IDS = [
  "current_user_message",
  "commitments_and_constraints",
  "closure_discourse_state",
  "contradictions_quarantines",
  "action_states",
  "group_channel_memory",
  "relational_slots",
] as const satisfies readonly EvidenceLedgerSectionId[];

const DEFAULT_COMPACT_PLANNER_TARGET_TOKENS = 8_000;
const DEFAULT_COMPACT_PLANNER_HARD_CAP_TOKENS = 15_000;
const DEFAULT_COMPACT_ENTRY_TEXT_TOKEN_CAP = 600;
const DEFAULT_DECISION_ARTIFACT_MAX_ENTRIES = 30;
const DEFAULT_DECISION_ARTIFACT_MAX_TOKENS = 3_000;
const DEFAULT_DECISION_ARTIFACT_RESERVED_SLOTS = {
  live: 8,
  invalidated: 3,
  pending: 3,
} as const satisfies Partial<Record<DecisionArtifactEntryKind, number>>;
const DEFAULT_DECISION_ARTIFACT_LOCKED_CAP = 14;
const DEFAULT_DECISION_ARTIFACT_PROMPT_SUMMARY_MAX_ENTRIES = {
  locked: 14,
  live: 8,
  pending: 6,
  invalidated: 4,
  tentative: 2,
} as const satisfies Record<DecisionArtifactEntryKind, number>;
const DEFAULT_DECISION_ARTIFACT_PROMPT_SUMMARY_TOKEN_BUDGET = 6_000;
const DEFAULT_DECISION_ARTIFACT_PROMPT_SUMMARY_ENTRY_TEXT_TOKENS = 1_000;
const DECISION_ARTIFACT_RESERVED_KINDS = [
  "live",
  "invalidated",
  "pending",
] as const satisfies readonly DecisionArtifactEntryKind[];
const DECISION_ARTIFACT_RENDER_FILL_ORDER = [
  "live",
  "pending",
  "invalidated",
  "locked",
  "tentative",
] as const satisfies readonly DecisionArtifactEntryKind[];
const DECISION_ARTIFACT_SINGLE_ENTRY_FLOOR_TOKENS = 200;
const DECISION_ARTIFACT_TEXT_TRUNCATION_MARKER = " ... [text truncated]";

const DEFAULT_COMPACT_SECTION_OPTIONS = {
  current_user_message: {
    maxEntries: 1,
    maxTokens: 1_200,
  },
  commitments_and_constraints: {
    maxEntries: 32,
    maxTokens: 2_400,
  },
  closure_discourse_state: {
    maxEntries: 8,
    maxTokens: 700,
  },
  contradictions_quarantines: {
    maxEntries: 16,
    maxTokens: 1_000,
  },
  action_states: {
    maxEntries: 12,
    maxTokens: 1_800,
  },
  group_channel_memory: {
    maxEntries: 24,
    maxTokens: 1_600,
  },
  relational_slots: {
    maxEntries: 24,
    maxTokens: 1_600,
  },
} as const satisfies Record<
  (typeof COMPACT_PLANNER_LEDGER_SECTION_IDS)[number],
  {
    maxEntries: number;
    maxTokens: number;
  }
>;

export type CompactPlannerLedgerOptions = {
  targetTokens?: number;
  hardCapTokens?: number;
  maxEntryTextTokens?: number;
  decisionArtifact?: DecisionArtifactRenderOptions;
};

export type CompactPlannerLedgerTraceSummary = {
  entryCountsBySection: Record<EvidenceLedgerSectionId, number>;
  omittedEntryCountsBySection: Record<EvidenceLedgerSectionId, number>;
  estimatedTokensBySection: Record<EvidenceLedgerSectionId, number>;
  decisionArtifactEntryCount: number;
  decisionArtifactRenderedTokens: number;
  decisionArtifactRenderedByKind: DecisionArtifactKindCounts;
  totalEstimatedTokens: number;
  targetTokens: number;
  hardCapTokens: number;
};

export type CompactPlannerLedgerPrompt = {
  promptSection: string | null;
  traceSummary: CompactPlannerLedgerTraceSummary;
};

export type DecisionStateArtifactRenderSummary = {
  totalEntryCount: number;
  activeEntryCount: number;
  renderedEntryCount: number;
  omittedEntryCount: number;
  estimatedTokens: number;
  renderedByKind: DecisionArtifactKindCounts;
  omittedByKind: DecisionArtifactKindCounts;
};

export type DecisionArtifactKindCounts = Record<DecisionArtifactEntryKind, number>;

export type DecisionArtifactRenderOptions = {
  maxEntries?: number;
  maxTokens?: number;
  reservedSlots?: Partial<Record<DecisionArtifactEntryKind, number>>;
  lockedMaxEntries?: number;
};

export type DecisionArtifactPromptSummaryOptions = {
  maxEntries?: Partial<Record<DecisionArtifactEntryKind, number>>;
  summaryTokenBudget?: number;
  maxEntryTextTokens?: number;
};

export type DecisionArtifactPromptSummaryEntry = {
  id: DecisionArtifactEntry["id"];
  text: string;
  owner_entity_id?: NonNullable<DecisionArtifactEntry["owner_entity_id"]>;
  last_updated_stream_entry_id:
    | DecisionArtifactEntry["last_updated_stream_entry_ids"][number]
    | null;
  canonicalizes_ids_count: number;
};

export type DecisionArtifactPromptSummarySupersededEntry = {
  id: DecisionArtifactEntry["id"];
  text: string;
  superseded_by_id: NonNullable<DecisionArtifactEntry["superseded_by_id"]>;
};

export type DecisionArtifactPromptSummary = {
  audience_entity_id: DecisionArtifact["audience_entity_id"];
  record_version: DecisionArtifact["record_version"];
  active_counts_by_kind: DecisionArtifactKindCounts;
  active_entries: Record<DecisionArtifactEntryKind, DecisionArtifactPromptSummaryEntry[]>;
  omitted_counts_by_kind: DecisionArtifactKindCounts;
  recent_superseded: DecisionArtifactPromptSummarySupersededEntry[];
};

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

const HIERARCHY_GUIDANCE = [
  "Current-session transcript is authoritative for what happened in this conversation.",
  "Prior-session memory must be attributed or hedged.",
  "Episodes and semantic graph are summaries; use source handles when making exact claims.",
  "Quarantined/contested/assistant-seeded values are not facts.",
].join("\n");

const COMPACT_PLANNER_LEDGER_GUIDANCE = [
  "CompactPlannerLedger: decision-relevant evidence slice for the S2 planner.",
  "Use these entries to check current-turn constraints before planning verification steps.",
  "Dialogue messages carry the conversational transcript; this compact ledger carries locked state, constraints, participant context, quarantines, and action threads.",
  "Quarantined/contested/assistant-seeded values are not facts.",
].join("\n");

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
  commitments_and_constraints: {
    maxEntries: 80,
    maxTokens: 5_000,
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
  relational_slots: {
    maxEntries: 48,
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
} as const satisfies Record<EvidenceLedgerSectionId, FullEvidenceLedgerSectionOptions>;

const CANONICAL_SECTION_PRIORITY = {
  current_user_message: 110,
  current_session_transcript: 100,
  commitments_and_constraints: 85,
  closure_discourse_state: 80,
  contradictions_quarantines: 78,
  action_states: 72,
  group_channel_memory: 70,
  relational_slots: 70,
  retrieved_raw_stream_evidence: 68,
  episodes: 52,
  retrieved_memory_evidence: 50,
  semantic_graph: 42,
  open_questions: 38,
  prior_session_memory: 30,
} as const satisfies Record<EvidenceLedgerSectionId, number>;

const LOWEST_TRUST_SECTION_ORDER = [
  "prior_session_memory",
  "semantic_graph",
  "episodes",
  "retrieved_memory_evidence",
  "open_questions",
  "relational_slots",
  "group_channel_memory",
  "action_states",
  "contradictions_quarantines",
  "closure_discourse_state",
  "commitments_and_constraints",
  "retrieved_raw_stream_evidence",
  "current_session_transcript",
  "current_user_message",
] as const satisfies readonly EvidenceLedgerSectionId[];

type FullLedgerSectionRetentionPolicy = "head" | "tail";

const TAIL_PRESERVING_FULL_LEDGER_SECTIONS = new Set<EvidenceLedgerSectionId>([
  "current_session_transcript",
]);

function renderEntry(entry: EvidenceLedgerEntry): string {
  const stateMetadata =
    entry.state_metadata === undefined ? undefined : JSON.stringify(entry.state_metadata);
  const metadata = [
    `id=${entry.id}`,
    `source_type=${entry.source_type}`,
    `scope=${entry.session_scope}`,
    `actor=${entry.actor}`,
    `trust_rank=${entry.trust_rank}`,
    entry.citations === undefined || entry.citations.length === 0
      ? null
      : `[citation: ${entry.citations.join(", ")}]`,
    entry.stream_index === undefined ? null : `stream_index=${entry.stream_index}`,
    entry.state === undefined ? null : `state=${entry.state}`,
    stateMetadata === undefined ? null : `state_metadata=${stateMetadata}`,
    entry.taint === undefined ? null : `taint=${entry.taint}`,
    entry.persistence_class === undefined ? null : `persistence_class=${entry.persistence_class}`,
    entry.via_retrieval === true ? "via_retrieval=true" : null,
  ].filter((part): part is string => part !== null);
  const body = [
    entry.value === undefined ? null : `  value: ${entry.value}`,
    entry.text === undefined ? null : `  text:\n${entry.text}`,
  ].filter((part): part is string => part !== null);

  return [`- ${metadata.join(" ")}`, ...body].join("\n");
}

function normalizePositiveInteger(value: number | undefined, fallback: number): number {
  return value === undefined || !Number.isFinite(value) || value <= 0
    ? fallback
    : Math.floor(value);
}

function allSectionIds(): EvidenceLedgerSectionId[] {
  return [
    "current_user_message",
    "current_session_transcript",
    "commitments_and_constraints",
    "closure_discourse_state",
    "contradictions_quarantines",
    "action_states",
    "group_channel_memory",
    "relational_slots",
    "retrieved_raw_stream_evidence",
    "retrieved_memory_evidence",
    "episodes",
    "semantic_graph",
    "open_questions",
    "prior_session_memory",
  ];
}

function emptySectionCountRecord(): Record<EvidenceLedgerSectionId, number> {
  return Object.fromEntries(allSectionIds().map((sectionId) => [sectionId, 0])) as Record<
    EvidenceLedgerSectionId,
    number
  >;
}

function cloneLedgerWithSections(
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

function estimateEvidenceLedgerTokens(ledger: EvidenceLedger): number {
  return estimatePromptTokens(
    renderEvidenceLedger({
      ...ledger,
      estimatedTokens: 0,
    }) ?? "",
  );
}

function metadataString(value: unknown): string | null {
  return typeof value === "string" && value.length > 0 ? value : null;
}

function metadataStringArray(metadata: Record<string, unknown> | undefined, key: string): string[] {
  const value = metadata?.[key];

  if (typeof value === "string" && value.length > 0) {
    return [value];
  }

  if (!Array.isArray(value)) {
    return [];
  }

  return value.filter((item): item is string => typeof item === "string" && item.length > 0);
}

function suffixAfterPrefix(value: string, prefix: string): string | null {
  return value.startsWith(prefix) ? value.slice(prefix.length) : null;
}

function suffixAfterLastColon(value: string): string | null {
  const index = value.lastIndexOf(":");
  return index < 0 || index >= value.length - 1 ? null : value.slice(index + 1);
}

function addHandle(handles: Set<string>, type: string, value: string | null): void {
  if (value === null || value.length === 0) {
    return;
  }

  handles.add(`${type}:${value}`);
}

function evidenceHandles(
  sectionId: EvidenceLedgerSectionId,
  entry: EvidenceLedgerEntry,
): Set<string> {
  const handles = new Set<string>();
  const metadata = entry.state_metadata;
  const currentSessionStreamId = suffixAfterPrefix(entry.id, "current_session_stream:");
  const episodeId = suffixAfterPrefix(entry.id, "episode:");
  const semanticNodeId = suffixAfterPrefix(entry.id, "semantic_node:");
  const semanticEdgeId = suffixAfterPrefix(entry.id, "semantic_edge:");
  const openQuestionId = suffixAfterPrefix(entry.id, "open_question:");
  const actionThreadId = suffixAfterPrefix(entry.id, "action_thread:");
  const groupActionId = suffixAfterPrefix(entry.id, "group_action:");

  addHandle(handles, "stream", currentSessionStreamId);
  addHandle(handles, "episode", episodeId);
  addHandle(handles, "semantic_node", semanticNodeId);
  addHandle(handles, "semantic_edge", semanticEdgeId);
  addHandle(handles, "open_question", openQuestionId);
  addHandle(handles, "action", actionThreadId);
  addHandle(handles, "action", groupActionId);

  if (
    entry.id.startsWith("commitment:") ||
    entry.id.startsWith("group_commitment:") ||
    entry.id.startsWith("participant_commitment:")
  ) {
    addHandle(handles, "commitment", suffixAfterLastColon(entry.id));
  }

  for (const key of ["stream_ids", "source_stream_ids", "evidence_stream_entry_ids"]) {
    for (const streamId of metadataStringArray(metadata, key)) {
      addHandle(handles, "stream", streamId);
    }
  }

  for (const key of ["episode_id", "parent_episode_id"]) {
    addHandle(handles, "episode", metadataString(metadata?.[key]));
  }

  for (const key of ["source_episode_ids", "evidence_episode_ids", "episode_ids"]) {
    for (const id of metadataStringArray(metadata, key)) {
      addHandle(handles, "episode", id);
    }
  }

  addHandle(handles, "semantic_node", metadataString(metadata?.["node_id"]));
  addHandle(handles, "semantic_edge", metadataString(metadata?.["edge_id"]));
  addHandle(handles, "commitment", metadataString(metadata?.["commitment_id"]));

  if (sectionId === "open_questions") {
    addHandle(handles, "open_question", metadataString(metadata?.["open_question_id"]));
  }

  for (const actionId of metadataStringArray(metadata, "record_ids")) {
    addHandle(handles, "action", actionId);
  }
  addHandle(handles, "action", metadataString(metadata?.["current_action_id"]));

  return handles;
}

function findParent(parents: Map<number, number>, id: number): number {
  let parent = parents.get(id) ?? id;

  while (parent !== (parents.get(parent) ?? parent)) {
    parent = parents.get(parent) ?? parent;
  }

  parents.set(id, parent);
  return parent;
}

function unionParents(parents: Map<number, number>, left: number, right: number): void {
  const leftRoot = findParent(parents, left);
  const rightRoot = findParent(parents, right);

  if (leftRoot !== rightRoot) {
    parents.set(rightRoot, leftRoot);
  }
}

type LedgerEntryRef = {
  sectionId: EvidenceLedgerSectionId;
  sectionIndex: number;
  entryIndex: number;
  entry: EvidenceLedgerEntry;
  handles: Set<string>;
};

function citationValue(handle: string): string {
  const index = handle.indexOf(":");
  return index < 0 ? handle : handle.slice(index + 1);
}

function citationRank(handle: string): number {
  if (handle.startsWith("episode:")) {
    return 0;
  }

  if (handle.startsWith("semantic_node:")) {
    return 1;
  }

  if (handle.startsWith("semantic_edge:")) {
    return 2;
  }

  if (handle.startsWith("stream:")) {
    return 3;
  }

  if (handle.startsWith("commitment:")) {
    return 4;
  }

  if (handle.startsWith("action:")) {
    return 5;
  }

  return 6;
}

function citationHandles(refs: readonly LedgerEntryRef[]): string[] {
  return [
    ...new Set(
      refs
        .flatMap((ref) => [...ref.handles])
        .sort((left, right) => {
          return citationRank(left) - citationRank(right) || left.localeCompare(right);
        }),
    ),
  ].map(citationValue);
}

function compareCanonicalRefs(left: LedgerEntryRef, right: LedgerEntryRef): number {
  return (
    CANONICAL_SECTION_PRIORITY[right.sectionId] - CANONICAL_SECTION_PRIORITY[left.sectionId] ||
    right.entry.trust_rank - left.entry.trust_rank ||
    (left.entry.stream_index ?? Number.MAX_SAFE_INTEGER) -
      (right.entry.stream_index ?? Number.MAX_SAFE_INTEGER) ||
    left.entry.id.localeCompare(right.entry.id)
  );
}

function dedupeEvidenceLedgerByProvenance(ledger: EvidenceLedger): {
  ledger: EvidenceLedger;
  dedupedEntryCount: number;
} {
  const refs: LedgerEntryRef[] = [];
  const parents = new Map<number, number>();
  const handleOwners = new Map<string, number>();

  for (const [sectionIndex, section] of ledger.sections.entries()) {
    for (const [entryIndex, entry] of section.entries.entries()) {
      const handles = evidenceHandles(section.id, entry);
      const refIndex = refs.length;
      refs.push({
        sectionId: section.id,
        sectionIndex,
        entryIndex,
        entry,
        handles,
      });
      parents.set(refIndex, refIndex);

      for (const handle of handles) {
        const owner = handleOwners.get(handle);
        if (owner === undefined) {
          handleOwners.set(handle, refIndex);
          continue;
        }
        unionParents(parents, owner, refIndex);
      }
    }
  }

  const groups = new Map<number, LedgerEntryRef[]>();

  for (const [index, ref] of refs.entries()) {
    const root = findParent(parents, index);
    groups.set(root, [...(groups.get(root) ?? []), ref]);
  }

  const canonicalByRef = new Map<LedgerEntryRef, EvidenceLedgerEntry>();
  const droppedRefs = new Set<LedgerEntryRef>();
  let dedupedEntryCount = 0;

  for (const group of groups.values()) {
    const sectionIds = new Set(group.map((ref) => ref.sectionId));

    if (group.length <= 1 || sectionIds.size <= 1) {
      continue;
    }

    const canonical = [...group].sort(compareCanonicalRefs)[0]!;
    const citations = citationHandles(group);
    canonicalByRef.set(canonical, {
      ...canonical.entry,
      citations: [...new Set([...(canonical.entry.citations ?? []), ...citations])],
    });

    for (const ref of group) {
      if (ref !== canonical) {
        droppedRefs.add(ref);
        dedupedEntryCount += 1;
      }
    }
  }

  const sections = ledger.sections.map((section, sectionIndex) => ({
    ...section,
    entries: section.entries.flatMap((entry, entryIndex) => {
      const ref = refs.find(
        (candidate) =>
          candidate.sectionIndex === sectionIndex && candidate.entryIndex === entryIndex,
      );

      if (ref === undefined || droppedRefs.has(ref)) {
        return [];
      }

      return [canonicalByRef.get(ref) ?? entry];
    }),
  }));

  return {
    ledger: cloneLedgerWithSections(ledger, sections),
    dedupedEntryCount,
  };
}

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
    maxEntries: normalizePositiveInteger(overrides?.maxEntries, defaults.maxEntries),
    maxTokens: normalizePositiveInteger(overrides?.maxTokens, defaults.maxTokens),
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
  const maxEntryTextTokens = normalizePositiveInteger(
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
  const targetTokens = normalizePositiveInteger(
    options.targetTokens,
    DEFAULT_FULL_LEDGER_TARGET_TOKENS,
  );
  const hardCapTokens = normalizePositiveInteger(
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

function truncateTextForCompactPlannerLedger(
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

  const omission = `\n[compact planner ledger truncated ${value.length - maxChars} chars]`;
  const bodyLimit = Math.max(0, maxChars - omission.length);

  return `${value.slice(0, bodyLimit).trimEnd()}${omission}`;
}

function compactEntry(entry: EvidenceLedgerEntry, maxEntryTextTokens: number): EvidenceLedgerEntry {
  return {
    ...entry,
    text: truncateTextForCompactPlannerLedger(entry.text, maxEntryTextTokens),
    value: truncateTextForCompactPlannerLedger(entry.value, Math.max(80, maxEntryTextTokens / 4)),
  };
}

function omittedEntry(section: EvidenceLedgerSection, omittedCount: number): EvidenceLedgerEntry {
  return {
    id: `compact_planner_ledger_omitted:${section.id}`,
    source_type: "system_metadata",
    session_scope: "global",
    actor: "system",
    trust_rank: 0,
    state: "omitted",
    text: `Compact planner ledger omitted ${omittedCount} older entries from ${section.id} to stay within the planner budget.`,
    taint: "none",
  };
}

function renderCompactSection(section: EvidenceLedgerSection, omittedCount: number): string {
  const entries =
    omittedCount <= 0 ? section.entries : [...section.entries, omittedEntry(section, omittedCount)];

  return renderSection({
    ...section,
    entries,
  });
}

function compactSection(input: { section: EvidenceLedgerSection; maxEntryTextTokens: number }): {
  section: EvidenceLedgerSection;
  omittedCount: number;
  estimatedTokens: number;
} {
  const options =
    DEFAULT_COMPACT_SECTION_OPTIONS[
      input.section.id as keyof typeof DEFAULT_COMPACT_SECTION_OPTIONS
    ];
  const maxEntries = options.maxEntries;
  const maxTokens = options.maxTokens;
  const entries = input.section.entries
    .slice(0, maxEntries)
    .map((entry) => compactEntry(entry, input.maxEntryTextTokens));
  let includedEntries: EvidenceLedgerEntry[] = [];
  let omittedCount = Math.max(0, input.section.entries.length - entries.length);

  for (let index = 0; index < entries.length; index += 1) {
    const entry = entries[index]!;
    const candidateEntries = [...includedEntries, entry];
    const candidateSection = {
      ...input.section,
      entries: candidateEntries,
    };
    const rendered = renderCompactSection(candidateSection, omittedCount);

    if (estimatePromptTokens(rendered) <= maxTokens || includedEntries.length === 0) {
      includedEntries = candidateEntries;
      continue;
    }

    omittedCount += entries.length - index;
    break;
  }

  const section = {
    ...input.section,
    entries: includedEntries,
  };

  return {
    section,
    omittedCount,
    estimatedTokens: estimatePromptTokens(renderCompactSection(section, omittedCount)),
  };
}

function totalCompactPromptTokens(
  sections: readonly CompactSectionResult[],
  decisionArtifact: DecisionArtifact | null | undefined,
  decisionArtifactOptions: DecisionArtifactRenderOptions | undefined,
): number {
  return estimatePromptTokens(
    renderCompactPlannerLedgerPromptSection(
      renderCompactPlannerLedgerContent(sections, decisionArtifact, decisionArtifactOptions),
    ) ?? "",
  );
}

type CompactSectionResult = {
  section: EvidenceLedgerSection;
  omittedCount: number;
};

function trimToTokenTarget(
  sections: CompactSectionResult[],
  targetTokens: number,
  decisionArtifact: DecisionArtifact | null | undefined,
  decisionArtifactOptions: DecisionArtifactRenderOptions | undefined,
): CompactSectionResult[] {
  while (
    totalCompactPromptTokens(sections, decisionArtifact, decisionArtifactOptions) > targetTokens
  ) {
    const trimIndex = [...sections]
      .reverse()
      .findIndex((section) => section.section.entries.length > 0);

    if (trimIndex < 0) {
      break;
    }

    const sectionIndex = sections.length - 1 - trimIndex;
    const section = sections[sectionIndex]!;
    section.section = {
      ...section.section,
      entries: section.section.entries.slice(0, -1),
    };
    section.omittedCount += 1;
  }

  return sections;
}

function emptyDecisionArtifactKindCounts(): DecisionArtifactKindCounts {
  return Object.fromEntries(
    DECISION_ARTIFACT_ENTRY_KINDS.map((kind) => [kind, 0]),
  ) as DecisionArtifactKindCounts;
}

function countDecisionArtifactEntriesByKind(
  entries: readonly DecisionArtifactEntry[],
): DecisionArtifactKindCounts {
  const counts = emptyDecisionArtifactKindCounts();

  for (const entry of entries) {
    counts[entry.kind] += 1;
  }

  return counts;
}

function subtractDecisionArtifactKindCounts(
  left: DecisionArtifactKindCounts,
  right: DecisionArtifactKindCounts,
): DecisionArtifactKindCounts {
  const counts = emptyDecisionArtifactKindCounts();

  for (const kind of DECISION_ARTIFACT_ENTRY_KINDS) {
    counts[kind] = Math.max(0, left[kind] - right[kind]);
  }

  return counts;
}

function activeDecisionArtifactEntries(artifact: DecisionArtifact | null | undefined) {
  return (artifact?.entries ?? []).filter((entry) => entry.superseded_by_id === null);
}

function compareDecisionArtifactEntriesByRecency(
  left: DecisionArtifactEntry,
  right: DecisionArtifactEntry,
): number {
  return (
    right.last_updated_at - left.last_updated_at ||
    left.rank - right.rank ||
    right.created_at - left.created_at ||
    left.id.localeCompare(right.id)
  );
}

function decisionArtifactPromptSummaryOptions(options: DecisionArtifactPromptSummaryOptions = {}): {
  maxEntries: Record<DecisionArtifactEntryKind, number>;
  summaryTokenBudget: number;
  maxEntryTextTokens: number;
} {
  const maxEntries: Record<DecisionArtifactEntryKind, number> = {
    ...DEFAULT_DECISION_ARTIFACT_PROMPT_SUMMARY_MAX_ENTRIES,
  };

  for (const kind of DECISION_ARTIFACT_ENTRY_KINDS) {
    const configured = options.maxEntries?.[kind];

    if (configured !== undefined && Number.isFinite(configured)) {
      maxEntries[kind] = Math.max(0, Math.floor(configured));
    }
  }

  return {
    maxEntries,
    summaryTokenBudget: normalizePositiveInteger(
      options.summaryTokenBudget,
      DEFAULT_DECISION_ARTIFACT_PROMPT_SUMMARY_TOKEN_BUDGET,
    ),
    maxEntryTextTokens: normalizePositiveInteger(
      options.maxEntryTextTokens,
      DEFAULT_DECISION_ARTIFACT_PROMPT_SUMMARY_ENTRY_TEXT_TOKENS,
    ),
  };
}

function emptyDecisionArtifactPromptSummaryEntries(): Record<
  DecisionArtifactEntryKind,
  DecisionArtifactPromptSummaryEntry[]
> {
  return {
    locked: [],
    live: [],
    pending: [],
    invalidated: [],
    tentative: [],
  };
}

function decisionArtifactCanonicalizesIdCount(entry: DecisionArtifactEntry): number {
  return (
    entry.canonicalizes.goal_ids.length +
    entry.canonicalizes.commitment_ids.length +
    entry.canonicalizes.action_ids.length +
    entry.canonicalizes.open_question_ids.length
  );
}

function lastUpdatedStreamEntryId(
  entry: DecisionArtifactEntry,
): DecisionArtifactPromptSummaryEntry["last_updated_stream_entry_id"] {
  return (
    entry.last_updated_stream_entry_ids[entry.last_updated_stream_entry_ids.length - 1] ?? null
  );
}

function toDecisionArtifactPromptSummaryEntry(
  entry: DecisionArtifactEntry,
  maxEntryTextTokens: number,
): DecisionArtifactPromptSummaryEntry {
  const text =
    estimatePromptTokens(entry.text) <= maxEntryTextTokens
      ? entry.text
      : truncateDecisionArtifactText(entry.text, maxEntryTextTokens);

  return {
    id: entry.id,
    text,
    ...(entry.owner_entity_id === null ? {} : { owner_entity_id: entry.owner_entity_id }),
    last_updated_stream_entry_id: lastUpdatedStreamEntryId(entry),
    canonicalizes_ids_count: decisionArtifactCanonicalizesIdCount(entry),
  };
}

function toDecisionArtifactPromptSummarySupersededEntry(
  entry: DecisionArtifactEntry,
): DecisionArtifactPromptSummarySupersededEntry | null {
  if (entry.superseded_by_id === null) {
    return null;
  }

  return {
    id: entry.id,
    text: entry.text,
    superseded_by_id: entry.superseded_by_id,
  };
}

function selectedDecisionArtifactPromptSummaryEntries(input: {
  activeEntries: readonly DecisionArtifactEntry[];
  maxEntries: Record<DecisionArtifactEntryKind, number>;
}): DecisionArtifactEntry[] {
  const totalMaxEntries = DECISION_ARTIFACT_ENTRY_KINDS.reduce(
    (sum, kind) => sum + input.maxEntries[kind],
    0,
  );
  const cappedActiveEntries = DECISION_ARTIFACT_ENTRY_KINDS.flatMap((kind) =>
    input.activeEntries
      .filter((entry) => entry.kind === kind)
      .sort(compareDecisionArtifactEntriesByRecency)
      .slice(0, input.maxEntries[kind]),
  );
  const selected = selectDecisionArtifactEntriesForRender({
    entries: cappedActiveEntries,
    maxEntries: totalMaxEntries,
    reservedSlots: {
      live: input.maxEntries.live,
      pending: input.maxEntries.pending,
      invalidated: input.maxEntries.invalidated,
    },
    lockedMaxEntries: input.maxEntries.locked,
  });
  const counts = emptyDecisionArtifactKindCounts();

  return selected.filter((entry) => {
    if (counts[entry.kind] >= input.maxEntries[entry.kind]) {
      return false;
    }

    counts[entry.kind] += 1;
    return true;
  });
}

function buildDecisionArtifactPromptSummaryFromEntries(input: {
  artifact: DecisionArtifact;
  activeEntries: readonly DecisionArtifactEntry[];
  selectedEntries: readonly DecisionArtifactEntry[];
  recentSuperseded: readonly DecisionArtifactPromptSummarySupersededEntry[];
  maxEntryTextTokens: number;
}): DecisionArtifactPromptSummary {
  const activeEntriesByKind = emptyDecisionArtifactPromptSummaryEntries();

  for (const entry of input.selectedEntries) {
    activeEntriesByKind[entry.kind].push(
      toDecisionArtifactPromptSummaryEntry(entry, input.maxEntryTextTokens),
    );
  }

  return {
    audience_entity_id: input.artifact.audience_entity_id,
    record_version: input.artifact.record_version,
    active_counts_by_kind: countDecisionArtifactEntriesByKind(input.activeEntries),
    active_entries: activeEntriesByKind,
    omitted_counts_by_kind: subtractDecisionArtifactKindCounts(
      countDecisionArtifactEntriesByKind(input.activeEntries),
      countDecisionArtifactEntriesByKind(input.selectedEntries),
    ),
    recent_superseded: [...input.recentSuperseded],
  };
}

function decisionArtifactPromptSummaryTokenEstimate(
  summary: DecisionArtifactPromptSummary,
): number {
  return estimatePromptTokens(JSON.stringify(summary));
}

function decisionArtifactPromptSummaryDropIndex(input: {
  entries: readonly DecisionArtifactEntry[];
  activeCounts: DecisionArtifactKindCounts;
}): number | null {
  const dropTentative = tokenDropIndexForKinds({
    entries: input.entries,
    kinds: ["tentative"],
    minimumForKind: () => 0,
  });

  if (dropTentative !== null) {
    return dropTentative;
  }

  const dropInvalidated = tokenDropIndexForKinds({
    entries: input.entries,
    kinds: ["invalidated"],
    minimumForKind: (kind) => onePerKindTokenDropFloor(kind, input.activeCounts),
  });

  if (dropInvalidated !== null) {
    return dropInvalidated;
  }

  return tokenDropIndexForKinds({
    entries: input.entries,
    kinds: ["live", "pending", "locked"],
    minimumForKind: (kind) => onePerKindTokenDropFloor(kind, input.activeCounts),
  });
}

export function buildDecisionArtifactPromptSummary(
  artifact: DecisionArtifact | null | undefined,
  options?: DecisionArtifactPromptSummaryOptions,
): DecisionArtifactPromptSummary | null {
  if (artifact === null || artifact === undefined) {
    return null;
  }

  const normalizedOptions = decisionArtifactPromptSummaryOptions(options);
  const activeEntries = activeDecisionArtifactEntries(artifact);
  let selectedEntries = selectedDecisionArtifactPromptSummaryEntries({
    activeEntries,
    maxEntries: normalizedOptions.maxEntries,
  });
  let recentSuperseded = artifact.entries
    .filter((entry) => entry.superseded_by_id !== null)
    .sort(compareDecisionArtifactEntriesByRecency)
    .slice(0, 5)
    .flatMap((entry) => {
      const summarized = toDecisionArtifactPromptSummarySupersededEntry(entry);

      return summarized === null ? [] : [summarized];
    });
  let summary = buildDecisionArtifactPromptSummaryFromEntries({
    artifact,
    activeEntries,
    selectedEntries,
    recentSuperseded,
    maxEntryTextTokens: normalizedOptions.maxEntryTextTokens,
  });

  while (
    decisionArtifactPromptSummaryTokenEstimate(summary) > normalizedOptions.summaryTokenBudget &&
    recentSuperseded.length > 0
  ) {
    recentSuperseded = recentSuperseded.slice(0, -1);
    summary = buildDecisionArtifactPromptSummaryFromEntries({
      artifact,
      activeEntries,
      selectedEntries,
      recentSuperseded,
      maxEntryTextTokens: normalizedOptions.maxEntryTextTokens,
    });
  }

  while (
    decisionArtifactPromptSummaryTokenEstimate(summary) > normalizedOptions.summaryTokenBudget &&
    selectedEntries.length > 0
  ) {
    const dropIndex = decisionArtifactPromptSummaryDropIndex({
      entries: selectedEntries,
      activeCounts: countDecisionArtifactEntriesByKind(activeEntries),
    });

    if (dropIndex === null) {
      break;
    }

    selectedEntries = [
      ...selectedEntries.slice(0, dropIndex),
      ...selectedEntries.slice(dropIndex + 1),
    ];
    summary = buildDecisionArtifactPromptSummaryFromEntries({
      artifact,
      activeEntries,
      selectedEntries,
      recentSuperseded,
      maxEntryTextTokens: normalizedOptions.maxEntryTextTokens,
    });
  }

  return summary;
}

function decisionArtifactRenderOptions(
  options: DecisionArtifactRenderOptions = {},
): Required<DecisionArtifactRenderOptions> {
  return {
    maxEntries: normalizePositiveInteger(options.maxEntries, DEFAULT_DECISION_ARTIFACT_MAX_ENTRIES),
    maxTokens: normalizePositiveInteger(options.maxTokens, DEFAULT_DECISION_ARTIFACT_MAX_TOKENS),
    reservedSlots: {
      ...DEFAULT_DECISION_ARTIFACT_RESERVED_SLOTS,
      ...(options.reservedSlots ?? {}),
    },
    lockedMaxEntries:
      options.lockedMaxEntries === undefined || !Number.isFinite(options.lockedMaxEntries)
        ? DEFAULT_DECISION_ARTIFACT_LOCKED_CAP
        : Math.max(0, Math.floor(options.lockedMaxEntries)),
  };
}

function selectDecisionArtifactEntriesForRender(input: {
  entries: readonly DecisionArtifactEntry[];
  maxEntries: number;
  reservedSlots: Partial<Record<DecisionArtifactEntryKind, number>>;
  lockedMaxEntries: number;
}): DecisionArtifactEntry[] {
  const byKind = new Map<DecisionArtifactEntryKind, DecisionArtifactEntry[]>();

  for (const kind of DECISION_ARTIFACT_ENTRY_KINDS) {
    byKind.set(
      kind,
      input.entries
        .filter((entry) => entry.kind === kind)
        .sort(compareDecisionArtifactEntriesByRecency),
    );
  }

  const selected: DecisionArtifactEntry[] = [];
  const selectedIds = new Set<DecisionArtifactEntry["id"]>();
  const selectedByKind = emptyDecisionArtifactKindCounts();

  const takeFromKind = (kind: DecisionArtifactEntryKind, limit: number): void => {
    if (limit <= 0 || selected.length >= input.maxEntries) {
      return;
    }

    const candidates = byKind.get(kind) ?? [];

    for (const candidate of candidates) {
      if (selected.length >= input.maxEntries || selectedByKind[kind] >= limit) {
        return;
      }

      if (kind === "locked" && selectedByKind.locked >= input.lockedMaxEntries) {
        return;
      }

      if (selectedIds.has(candidate.id)) {
        continue;
      }

      selected.push(candidate);
      selectedIds.add(candidate.id);
      selectedByKind[kind] += 1;
    }
  };

  for (const kind of DECISION_ARTIFACT_RESERVED_KINDS) {
    takeFromKind(kind, input.reservedSlots[kind] ?? 0);
  }

  for (const kind of DECISION_ARTIFACT_RENDER_FILL_ORDER) {
    const categoryLimit = kind === "locked" ? input.lockedMaxEntries : Number.POSITIVE_INFINITY;
    takeFromKind(kind, categoryLimit);
  }

  const orderByKind = new Map(
    DECISION_ARTIFACT_RENDER_FILL_ORDER.map((kind, index) => [kind, index]),
  );

  return selected.sort(
    (left, right) =>
      (orderByKind.get(left.kind) ?? Number.MAX_SAFE_INTEGER) -
        (orderByKind.get(right.kind) ?? Number.MAX_SAFE_INTEGER) ||
      compareDecisionArtifactEntriesByRecency(left, right),
  );
}

function decisionArtifactRenderedCounts(input: {
  activeEntries: readonly DecisionArtifactEntry[];
  renderedEntries: readonly DecisionArtifactEntry[];
}): {
  renderedByKind: DecisionArtifactKindCounts;
  omittedByKind: DecisionArtifactKindCounts;
  omittedEntryCount: number;
} {
  const activeByKind = countDecisionArtifactEntriesByKind(input.activeEntries);
  const renderedByKind = countDecisionArtifactEntriesByKind(input.renderedEntries);
  const omittedByKind = subtractDecisionArtifactKindCounts(activeByKind, renderedByKind);

  return {
    renderedByKind,
    omittedByKind,
    omittedEntryCount: Math.max(0, input.activeEntries.length - input.renderedEntries.length),
  };
}

function formatDecisionArtifactKindCounts(
  counts: DecisionArtifactKindCounts,
  options: { suffix?: string } = {},
): string {
  const parts = DECISION_ARTIFACT_ENTRY_KINDS.flatMap((kind) =>
    counts[kind] <= 0 ? [] : [`${counts[kind]} ${kind}${options.suffix ?? ""}`],
  );

  return parts.length === 0 ? "0 entries" : parts.join(", ");
}

function onePerKindTokenDropFloor(
  kind: DecisionArtifactEntryKind,
  activeCounts: DecisionArtifactKindCounts,
): number {
  if (activeCounts[kind] <= 0) {
    return 0;
  }

  if (kind === "tentative") {
    return 0;
  }

  return 1;
}

function reservedTokenDropMinimum(input: {
  kind: DecisionArtifactEntryKind;
  activeCounts: DecisionArtifactKindCounts;
  reservedSlots: Partial<Record<DecisionArtifactEntryKind, number>>;
}): number {
  const floor = onePerKindTokenDropFloor(input.kind, input.activeCounts);
  const reserved = input.reservedSlots[input.kind] ?? 0;

  if (reserved <= 0) {
    return floor;
  }

  return Math.max(floor, Math.min(input.activeCounts[input.kind], Math.floor(reserved)));
}

function latestDecisionArtifactDropIndex(
  entries: readonly DecisionArtifactEntry[],
  kind: DecisionArtifactEntryKind,
): number | null {
  for (let index = entries.length - 1; index >= 0; index -= 1) {
    if (entries[index]?.kind === kind) {
      return index;
    }
  }

  return null;
}

function tokenDropIndexForKinds(input: {
  entries: readonly DecisionArtifactEntry[];
  kinds: readonly DecisionArtifactEntryKind[];
  minimumForKind: (kind: DecisionArtifactEntryKind) => number;
}): number | null {
  const renderedCounts = countDecisionArtifactEntriesByKind(input.entries);
  let selectedKind: DecisionArtifactEntryKind | null = null;
  let selectedSurplus = 0;

  for (const kind of input.kinds) {
    const surplus = renderedCounts[kind] - input.minimumForKind(kind);

    if (surplus > selectedSurplus) {
      selectedKind = kind;
      selectedSurplus = surplus;
    }
  }

  return selectedKind === null
    ? null
    : latestDecisionArtifactDropIndex(input.entries, selectedKind);
}

function tokenDropIndex(input: {
  entries: readonly DecisionArtifactEntry[];
  activeCounts: DecisionArtifactKindCounts;
  reservedSlots: Partial<Record<DecisionArtifactEntryKind, number>>;
  lockedMaxEntries: number;
}): number {
  const dropTentative = tokenDropIndexForKinds({
    entries: input.entries,
    kinds: ["tentative"],
    minimumForKind: () => 0,
  });

  if (dropTentative !== null) {
    return dropTentative;
  }

  const dropLockedAboveCap = tokenDropIndexForKinds({
    entries: input.entries,
    kinds: ["locked"],
    minimumForKind: () => input.lockedMaxEntries,
  });

  if (dropLockedAboveCap !== null) {
    return dropLockedAboveCap;
  }

  const dropReservedAboveMinimum = tokenDropIndexForKinds({
    entries: input.entries,
    kinds: DECISION_ARTIFACT_RESERVED_KINDS,
    minimumForKind: (kind) =>
      reservedTokenDropMinimum({
        kind,
        activeCounts: input.activeCounts,
        reservedSlots: input.reservedSlots,
      }),
  });

  if (dropReservedAboveMinimum !== null) {
    return dropReservedAboveMinimum;
  }

  const dropLockedAboveFloor = tokenDropIndexForKinds({
    entries: input.entries,
    kinds: ["locked"],
    minimumForKind: (kind) => onePerKindTokenDropFloor(kind, input.activeCounts),
  });

  if (dropLockedAboveFloor !== null) {
    return dropLockedAboveFloor;
  }

  const dropReservedAboveFloor = tokenDropIndexForKinds({
    entries: input.entries,
    kinds: DECISION_ARTIFACT_RESERVED_KINDS,
    minimumForKind: (kind) => onePerKindTokenDropFloor(kind, input.activeCounts),
  });

  if (dropReservedAboveFloor !== null) {
    return dropReservedAboveFloor;
  }

  return (
    tokenDropIndexForKinds({
      entries: input.entries,
      kinds: DECISION_ARTIFACT_ENTRY_KINDS,
      minimumForKind: (kind) => onePerKindTokenDropFloor(kind, input.activeCounts),
    }) ?? Math.max(0, input.entries.length - 1)
  );
}

function renderDecisionArtifactEntry(entry: DecisionArtifactEntry): string {
  const owner = entry.owner_entity_id === null ? "owner=null" : `owner=${entry.owner_entity_id}`;
  const citations = `[citation: ${entry.provenance_stream_entry_ids.join(", ")}]`;

  return [
    `- kind=${entry.kind} id=${entry.id} ${owner} last_updated_at=${entry.last_updated_at} ${citations}`,
    `  text: ${entry.text}`,
  ].join("\n");
}

function renderDecisionArtifactContent(input: {
  artifact: DecisionArtifact;
  entries: readonly DecisionArtifactEntry[];
  omittedByKind: DecisionArtifactKindCounts;
  renderedByKind: DecisionArtifactKindCounts;
}): string {
  const omittedCount = Object.values(input.omittedByKind).reduce((sum, count) => sum + count, 0);
  const omission =
    omittedCount <= 0
      ? null
      : [
          `DecisionStateArtifact omitted: ${formatDecisionArtifactKindCounts(
            input.omittedByKind,
          )}.`,
          `Retained: ${formatDecisionArtifactKindCounts(input.renderedByKind)}.`,
        ].join(" ");

  return [
    "## 0. Canonical Decision State",
    "DecisionStateArtifact: shared planning state for this audience. It is a compact structural anchor, not a policy source.",
    `audience_entity_id=${input.artifact.audience_entity_id}`,
    `record_version=${input.artifact.record_version}`,
    ...input.entries.map(renderDecisionArtifactEntry),
    omission,
  ]
    .filter((part): part is string => part !== null)
    .join("\n");
}

function renderDecisionArtifactOmissionOnly(input: {
  artifact: DecisionArtifact;
  omittedByKind: DecisionArtifactKindCounts;
  reason: string;
}): string {
  return [
    "## 0. Canonical Decision State",
    "DecisionStateArtifact: shared planning state for this audience. It is a compact structural anchor, not a policy source.",
    `audience_entity_id=${input.artifact.audience_entity_id}`,
    `record_version=${input.artifact.record_version}`,
    `DecisionStateArtifact omitted: ${formatDecisionArtifactKindCounts(
      input.omittedByKind,
    )}. Reason: ${input.reason}.`,
  ].join("\n");
}

function truncateDecisionArtifactText(value: string, maxTokens: number): string {
  const maxChars = Math.max(
    0,
    Math.floor(maxTokens) * 4 - DECISION_ARTIFACT_TEXT_TRUNCATION_MARKER.length,
  );

  return `${value.slice(0, maxChars).trimEnd()}${DECISION_ARTIFACT_TEXT_TRUNCATION_MARKER}`;
}

function renderSingleEntryWithinDecisionArtifactCap(input: {
  artifact: DecisionArtifact;
  entry: DecisionArtifactEntry;
  activeEntries: readonly DecisionArtifactEntry[];
  maxTokens: number;
}): { content: string; renderedEntryCount: number; omittedEntryCount: number } {
  const counts = decisionArtifactRenderedCounts({
    activeEntries: input.activeEntries,
    renderedEntries: [input.entry],
  });
  const emptyEntryContent = renderDecisionArtifactContent({
    artifact: input.artifact,
    entries: [
      {
        ...input.entry,
        text: "",
      },
    ],
    omittedByKind: counts.omittedByKind,
    renderedByKind: counts.renderedByKind,
  });
  const remainingTokens = input.maxTokens - estimatePromptTokens(emptyEntryContent);

  if (remainingTokens < DECISION_ARTIFACT_SINGLE_ENTRY_FLOOR_TOKENS) {
    return {
      content: renderDecisionArtifactOmissionOnly({
        artifact: input.artifact,
        omittedByKind: countDecisionArtifactEntriesByKind(input.activeEntries),
        reason: "artifact entry too large to render",
      }),
      renderedEntryCount: 0,
      omittedEntryCount: input.activeEntries.length,
    };
  }

  const content = renderDecisionArtifactContent({
    artifact: input.artifact,
    entries: [
      {
        ...input.entry,
        text: truncateDecisionArtifactText(input.entry.text, remainingTokens),
      },
    ],
    omittedByKind: counts.omittedByKind,
    renderedByKind: counts.renderedByKind,
  });

  if (estimatePromptTokens(content) <= input.maxTokens) {
    return {
      content,
      renderedEntryCount: 1,
      omittedEntryCount: counts.omittedEntryCount,
    };
  }

  return {
    content: renderDecisionArtifactOmissionOnly({
      artifact: input.artifact,
      omittedByKind: countDecisionArtifactEntriesByKind(input.activeEntries),
      reason: "artifact entry too large to render",
    }),
    renderedEntryCount: 0,
    omittedEntryCount: input.activeEntries.length,
  };
}

function cappedDecisionArtifactRender(input: {
  artifact: DecisionArtifact;
  options?: DecisionArtifactRenderOptions;
}): { content: string | null; summary: DecisionStateArtifactRenderSummary } {
  const options = decisionArtifactRenderOptions(input.options);
  const activeEntries = activeDecisionArtifactEntries(input.artifact);

  if (activeEntries.length === 0) {
    return {
      content: null,
      summary: {
        totalEntryCount: input.artifact.entries.length,
        activeEntryCount: 0,
        renderedEntryCount: 0,
        omittedEntryCount: 0,
        estimatedTokens: 0,
        renderedByKind: emptyDecisionArtifactKindCounts(),
        omittedByKind: emptyDecisionArtifactKindCounts(),
      },
    };
  }

  const activeCounts = countDecisionArtifactEntriesByKind(activeEntries);
  let entries = selectDecisionArtifactEntriesForRender({
    entries: activeEntries,
    maxEntries: options.maxEntries,
    reservedSlots: options.reservedSlots,
    lockedMaxEntries: options.lockedMaxEntries,
  });
  let counts = decisionArtifactRenderedCounts({
    activeEntries,
    renderedEntries: entries,
  });
  let content = renderDecisionArtifactContent({
    artifact: input.artifact,
    entries,
    omittedByKind: counts.omittedByKind,
    renderedByKind: counts.renderedByKind,
  });

  while (estimatePromptTokens(content) > options.maxTokens && entries.length > 1) {
    const dropIndex = tokenDropIndex({
      entries,
      activeCounts,
      reservedSlots: options.reservedSlots,
      lockedMaxEntries: options.lockedMaxEntries,
    });
    entries = [...entries.slice(0, dropIndex), ...entries.slice(dropIndex + 1)];
    counts = decisionArtifactRenderedCounts({
      activeEntries,
      renderedEntries: entries,
    });
    content = renderDecisionArtifactContent({
      artifact: input.artifact,
      entries,
      omittedByKind: counts.omittedByKind,
      renderedByKind: counts.renderedByKind,
    });
  }

  if (estimatePromptTokens(content) > options.maxTokens && entries.length === 1) {
    const singleEntryRender = renderSingleEntryWithinDecisionArtifactCap({
      artifact: input.artifact,
      entry: entries[0]!,
      activeEntries,
      maxTokens: options.maxTokens,
    });

    content = singleEntryRender.content;
    entries = entries.slice(0, singleEntryRender.renderedEntryCount);
    counts = decisionArtifactRenderedCounts({
      activeEntries,
      renderedEntries: entries,
    });
  }

  return {
    content,
    summary: {
      totalEntryCount: input.artifact.entries.length,
      activeEntryCount: activeEntries.length,
      renderedEntryCount: entries.length,
      omittedEntryCount: counts.omittedEntryCount,
      estimatedTokens: estimatePromptTokens(content),
      renderedByKind: counts.renderedByKind,
      omittedByKind: counts.omittedByKind,
    },
  };
}

export function renderDecisionStateArtifact(
  artifact: DecisionArtifact | null | undefined,
  options?: DecisionArtifactRenderOptions,
): string | null {
  if (artifact === null || artifact === undefined) {
    return null;
  }

  return cappedDecisionArtifactRender({
    artifact,
    options,
  }).content;
}

export function summarizeDecisionStateArtifactRender(
  artifact: DecisionArtifact | null | undefined,
  options?: DecisionArtifactRenderOptions,
): DecisionStateArtifactRenderSummary {
  if (artifact === null || artifact === undefined) {
    return {
      totalEntryCount: 0,
      activeEntryCount: 0,
      renderedEntryCount: 0,
      omittedEntryCount: 0,
      estimatedTokens: 0,
      renderedByKind: emptyDecisionArtifactKindCounts(),
      omittedByKind: emptyDecisionArtifactKindCounts(),
    };
  }

  return cappedDecisionArtifactRender({
    artifact,
    options,
  }).summary;
}

function renderCompactPlannerLedgerContent(
  sections: readonly CompactSectionResult[],
  decisionArtifact: DecisionArtifact | null | undefined,
  decisionArtifactOptions: DecisionArtifactRenderOptions | undefined,
): string {
  return [
    renderDecisionStateArtifact(decisionArtifact, decisionArtifactOptions),
    COMPACT_PLANNER_LEDGER_GUIDANCE,
    ...sections.map((section) => renderCompactSection(section.section, section.omittedCount)),
  ]
    .filter((part): part is string => part !== null)
    .join("\n\n");
}

function renderCompactPlannerLedgerPromptSection(content: string): string | null {
  return renderTaggedPromptBlock(UNTRUSTED_DATA_PREAMBLE, [
    {
      tag: "borg_compact_planner_ledger",
      content,
    },
  ]);
}

function renderSection(section: EvidenceLedgerSection): string {
  if (section.entries.length === 0) {
    return [`## ${section.label}`, "No entries."].join("\n");
  }

  const sourceTypes = [...new Set(section.entries.map((entry) => entry.source_type))].join(", ");
  const scopes = [...new Set(section.entries.map((entry) => entry.session_scope))].join(", ");

  return [
    `## ${section.label}`,
    `source_types: ${sourceTypes}`,
    `scopes: ${scopes}`,
    ...section.entries.map((entry) => renderEntry(entry)),
  ].join("\n");
}

export function buildCompactPlannerLedgerPrompt(
  ledger: EvidenceLedger,
  options: CompactPlannerLedgerOptions = {},
): CompactPlannerLedgerPrompt {
  const targetTokens = normalizePositiveInteger(
    options.targetTokens,
    DEFAULT_COMPACT_PLANNER_TARGET_TOKENS,
  );
  const hardCapTokens = Math.max(
    targetTokens,
    normalizePositiveInteger(options.hardCapTokens, DEFAULT_COMPACT_PLANNER_HARD_CAP_TOKENS),
  );
  const maxEntryTextTokens = normalizePositiveInteger(
    options.maxEntryTextTokens,
    DEFAULT_COMPACT_ENTRY_TEXT_TOKEN_CAP,
  );
  const sectionsById = new Map(ledger.sections.map((section) => [section.id, section]));
  const compactSections = COMPACT_PLANNER_LEDGER_SECTION_IDS.map((sectionId) => {
    const section = sectionsById.get(sectionId);

    if (section === undefined) {
      return {
        section: {
          id: sectionId,
          label: sectionId,
          entries: [],
        },
        omittedCount: 0,
      };
    }

    const compacted = compactSection({
      section,
      maxEntryTextTokens,
    });

    return {
      section: compacted.section,
      omittedCount: compacted.omittedCount,
    };
  });
  const trimmedSections = trimToTokenTarget(
    compactSections,
    targetTokens,
    ledger.decisionArtifact,
    options.decisionArtifact,
  );
  const hardCappedSections = trimToTokenTarget(
    trimmedSections,
    hardCapTokens,
    ledger.decisionArtifact,
    options.decisionArtifact,
  );
  const content = renderCompactPlannerLedgerContent(
    hardCappedSections,
    ledger.decisionArtifact,
    options.decisionArtifact,
  );
  const promptSection = renderCompactPlannerLedgerPromptSection(content);
  const entryCountsBySection = emptySectionCountRecord();
  const omittedEntryCountsBySection = emptySectionCountRecord();
  const estimatedTokensBySection = emptySectionCountRecord();
  const decisionArtifactSummary = summarizeDecisionStateArtifactRender(
    ledger.decisionArtifact,
    options.decisionArtifact,
  );

  for (const section of hardCappedSections) {
    entryCountsBySection[section.section.id] = section.section.entries.length;
    omittedEntryCountsBySection[section.section.id] = section.omittedCount;
    estimatedTokensBySection[section.section.id] = estimatePromptTokens(
      renderCompactSection(section.section, section.omittedCount),
    );
  }

  return {
    promptSection,
    traceSummary: {
      entryCountsBySection,
      omittedEntryCountsBySection,
      estimatedTokensBySection,
      decisionArtifactEntryCount: decisionArtifactSummary.renderedEntryCount,
      decisionArtifactRenderedTokens: decisionArtifactSummary.estimatedTokens,
      decisionArtifactRenderedByKind: decisionArtifactSummary.renderedByKind,
      totalEstimatedTokens: estimatePromptTokens(promptSection ?? ""),
      targetTokens,
      hardCapTokens,
    },
  };
}

export function renderCompactPlannerLedger(
  ledger: EvidenceLedger,
  options: CompactPlannerLedgerOptions = {},
): string | null {
  return buildCompactPlannerLedgerPrompt(ledger, options).promptSection;
}

export function renderEvidenceLedger(
  ledger: EvidenceLedger,
  options: { decisionArtifact?: DecisionArtifactRenderOptions } = {},
): string | null {
  const transcriptStatus = ledger.transcriptIncluded
    ? ledger.transcriptCompacted
      ? "current_session_transcript=included compacted=true"
      : "current_session_transcript=included"
    : `current_session_transcript=omitted reason=${ledger.transcriptOmittedReason ?? "unknown"}`;
  const content = [
    "EvidenceLedger: prioritized evidence substrate for the final response.",
    HIERARCHY_GUIDANCE,
    transcriptStatus,
    `estimated_tokens=${ledger.estimatedTokens}`,
    renderDecisionStateArtifact(ledger.decisionArtifact, options.decisionArtifact),
    ...ledger.sections.map((section) => renderSection(section)),
  ]
    .filter((part): part is string => part !== null)
    .join("\n\n");

  return renderTaggedPromptBlock(UNTRUSTED_DATA_PREAMBLE, [
    {
      tag: "borg_evidence_ledger",
      content,
    },
  ]);
}

export function estimateEvidenceLedgerPromptTokens(
  ledger: EvidenceLedger,
  options: { decisionArtifact?: DecisionArtifactRenderOptions } = {},
): number {
  return estimatePromptTokens(renderEvidenceLedger(ledger, options) ?? "");
}
