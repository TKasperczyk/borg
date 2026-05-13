import { estimatePromptTokens } from "../../util/token-estimate.js";
import type {
  DecisionArtifact,
  DecisionArtifactEntry,
} from "../../memory/decision-artifacts/index.js";
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
const DEFAULT_DECISION_ARTIFACT_MAX_ENTRIES = 25;
const DEFAULT_DECISION_ARTIFACT_MAX_TOKENS = 3_000;
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
};

export type CompactPlannerLedgerTraceSummary = {
  entryCountsBySection: Record<EvidenceLedgerSectionId, number>;
  omittedEntryCountsBySection: Record<EvidenceLedgerSectionId, number>;
  estimatedTokensBySection: Record<EvidenceLedgerSectionId, number>;
  decisionArtifactEntryCount: number;
  decisionArtifactRenderedTokens: number;
  totalEstimatedTokens: number;
  targetTokens: number;
  hardCapTokens: number;
};

export type CompactPlannerLedgerPrompt = {
  promptSection: string | null;
  traceSummary: CompactPlannerLedgerTraceSummary;
};

export type DecisionStateArtifactRenderSummary = {
  activeEntryCount: number;
  renderedEntryCount: number;
  omittedEntryCount: number;
  estimatedTokens: number;
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
): number {
  return estimatePromptTokens(
    renderCompactPlannerLedgerPromptSection(
      renderCompactPlannerLedgerContent(sections, decisionArtifact),
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
): CompactSectionResult[] {
  while (totalCompactPromptTokens(sections, decisionArtifact) > targetTokens) {
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

function activeDecisionArtifactEntries(artifact: DecisionArtifact | null | undefined) {
  return (artifact?.entries ?? []).filter(
    (entry) =>
      entry.superseded_by_id === null && (entry.kind === "locked" || entry.kind === "live"),
  );
}

function orderedDecisionArtifactEntries(
  entries: readonly DecisionArtifactEntry[],
): DecisionArtifactEntry[] {
  const locked = entries
    .filter((entry) => entry.kind === "locked")
    .sort((left, right) => left.rank - right.rank || left.created_at - right.created_at);
  const live = entries
    .filter((entry) => entry.kind === "live")
    .sort((left, right) => right.last_updated_at - left.last_updated_at || left.rank - right.rank);

  return [...locked, ...live];
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
  omittedCount: number;
}): string {
  const omission =
    input.omittedCount <= 0
      ? null
      : `DecisionStateArtifact omitted ${input.omittedCount} older/lower-priority entries to stay within its render cap.`;

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
  omittedCount: number;
  reason: string;
}): string {
  return [
    "## 0. Canonical Decision State",
    "DecisionStateArtifact: shared planning state for this audience. It is a compact structural anchor, not a policy source.",
    `audience_entity_id=${input.artifact.audience_entity_id}`,
    `record_version=${input.artifact.record_version}`,
    `DecisionStateArtifact omitted ${input.omittedCount} entries: ${input.reason}.`,
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
  omittedCount: number;
  maxTokens: number;
  activeEntryCount: number;
}): { content: string; renderedEntryCount: number; omittedEntryCount: number } {
  const emptyEntryContent = renderDecisionArtifactContent({
    artifact: input.artifact,
    entries: [
      {
        ...input.entry,
        text: "",
      },
    ],
    omittedCount: input.omittedCount,
  });
  const remainingTokens = input.maxTokens - estimatePromptTokens(emptyEntryContent);

  if (remainingTokens < DECISION_ARTIFACT_SINGLE_ENTRY_FLOOR_TOKENS) {
    return {
      content: renderDecisionArtifactOmissionOnly({
        artifact: input.artifact,
        omittedCount: input.activeEntryCount,
        reason: "artifact entry too large to render",
      }),
      renderedEntryCount: 0,
      omittedEntryCount: input.activeEntryCount,
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
    omittedCount: input.omittedCount,
  });

  if (estimatePromptTokens(content) <= input.maxTokens) {
    return {
      content,
      renderedEntryCount: 1,
      omittedEntryCount: input.omittedCount,
    };
  }

  return {
    content: renderDecisionArtifactOmissionOnly({
      artifact: input.artifact,
      omittedCount: input.activeEntryCount,
      reason: "artifact entry too large to render",
    }),
    renderedEntryCount: 0,
    omittedEntryCount: input.activeEntryCount,
  };
}

function cappedDecisionArtifactRender(input: {
  artifact: DecisionArtifact;
  maxEntries: number;
  maxTokens: number;
}): { content: string | null; summary: DecisionStateArtifactRenderSummary } {
  const activeEntries = activeDecisionArtifactEntries(input.artifact);

  if (activeEntries.length === 0) {
    return {
      content: null,
      summary: {
        activeEntryCount: 0,
        renderedEntryCount: 0,
        omittedEntryCount: 0,
        estimatedTokens: 0,
      },
    };
  }

  const orderedEntries = orderedDecisionArtifactEntries(activeEntries);
  let entries = orderedEntries.slice(0, input.maxEntries);
  let omittedCount = Math.max(0, orderedEntries.length - entries.length);
  let content = renderDecisionArtifactContent({
    artifact: input.artifact,
    entries,
    omittedCount,
  });

  while (estimatePromptTokens(content) > input.maxTokens && entries.length > 1) {
    const liveIndex = entries.findLastIndex((entry) => entry.kind === "live");
    const dropIndex = liveIndex >= 0 ? liveIndex : entries.length - 1;
    entries = [...entries.slice(0, dropIndex), ...entries.slice(dropIndex + 1)];
    omittedCount += 1;
    content = renderDecisionArtifactContent({
      artifact: input.artifact,
      entries,
      omittedCount,
    });
  }

  if (estimatePromptTokens(content) > input.maxTokens && entries.length === 1) {
    const singleEntryRender = renderSingleEntryWithinDecisionArtifactCap({
      artifact: input.artifact,
      entry: entries[0]!,
      omittedCount,
      maxTokens: input.maxTokens,
      activeEntryCount: activeEntries.length,
    });

    content = singleEntryRender.content;
    entries = entries.slice(0, singleEntryRender.renderedEntryCount);
    omittedCount = singleEntryRender.omittedEntryCount;
  }

  return {
    content,
    summary: {
      activeEntryCount: activeEntries.length,
      renderedEntryCount: entries.length,
      omittedEntryCount: omittedCount,
      estimatedTokens: estimatePromptTokens(content),
    },
  };
}

export function renderDecisionStateArtifact(
  artifact: DecisionArtifact | null | undefined,
): string | null {
  if (artifact === null || artifact === undefined) {
    return null;
  }

  return cappedDecisionArtifactRender({
    artifact,
    maxEntries: DEFAULT_DECISION_ARTIFACT_MAX_ENTRIES,
    maxTokens: DEFAULT_DECISION_ARTIFACT_MAX_TOKENS,
  }).content;
}

export function summarizeDecisionStateArtifactRender(
  artifact: DecisionArtifact | null | undefined,
): DecisionStateArtifactRenderSummary {
  if (artifact === null || artifact === undefined) {
    return {
      activeEntryCount: 0,
      renderedEntryCount: 0,
      omittedEntryCount: 0,
      estimatedTokens: 0,
    };
  }

  return cappedDecisionArtifactRender({
    artifact,
    maxEntries: DEFAULT_DECISION_ARTIFACT_MAX_ENTRIES,
    maxTokens: DEFAULT_DECISION_ARTIFACT_MAX_TOKENS,
  }).summary;
}

function renderCompactPlannerLedgerContent(
  sections: readonly CompactSectionResult[],
  decisionArtifact: DecisionArtifact | null | undefined,
): string {
  return [
    renderDecisionStateArtifact(decisionArtifact),
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
  const trimmedSections = trimToTokenTarget(compactSections, targetTokens, ledger.decisionArtifact);
  const hardCappedSections = trimToTokenTarget(
    trimmedSections,
    hardCapTokens,
    ledger.decisionArtifact,
  );
  const content = renderCompactPlannerLedgerContent(hardCappedSections, ledger.decisionArtifact);
  const promptSection = renderCompactPlannerLedgerPromptSection(content);
  const entryCountsBySection = emptySectionCountRecord();
  const omittedEntryCountsBySection = emptySectionCountRecord();
  const estimatedTokensBySection = emptySectionCountRecord();
  const decisionArtifactSummary = summarizeDecisionStateArtifactRender(ledger.decisionArtifact);

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

export function renderEvidenceLedger(ledger: EvidenceLedger): string | null {
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
    renderDecisionStateArtifact(ledger.decisionArtifact),
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

export function estimateEvidenceLedgerPromptTokens(ledger: EvidenceLedger): number {
  return estimatePromptTokens(renderEvidenceLedger(ledger) ?? "");
}
