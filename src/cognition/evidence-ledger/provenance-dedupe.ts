import { cloneLedgerWithSections } from "./ledger-copy.js";
import type { EvidenceLedger, EvidenceLedgerEntry, EvidenceLedgerSectionId } from "./types.js";
import { DisjointSet } from "../../util/disjoint-set.js";

const CANONICAL_SECTION_PRIORITY = {
  current_user_message: 110,
  current_session_transcript: 100,
  current_session_attribution_sidebar: 96,
  attribution_matrix: 90,
  closure_discourse_state: 80,
  contradictions_quarantines: 78,
  action_states: 72,
  group_channel_memory: 70,
  retrieved_raw_stream_evidence: 68,
  shared_state_recall: 64,
  episodes: 52,
  retrieved_memory_evidence: 50,
  recent_lived_experience: 49,
  autobiographical_recall: 48,
  semantic_graph: 42,
  open_questions: 38,
  prior_session_memory: 30,
} as const satisfies Record<EvidenceLedgerSectionId, number>;

const PROVENANCE_DEDUPE_PROTECTED_SECTIONS = new Set<EvidenceLedgerSectionId>([
  "current_user_message",
  "current_session_transcript",
]);

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

type LedgerEntryRef = {
  sectionId: EvidenceLedgerSectionId;
  sectionIndex: number;
  entryIndex: number;
  entry: EvidenceLedgerEntry;
  handles: Set<string>;
};

function isProvenanceDedupeProtected(ref: LedgerEntryRef): boolean {
  return PROVENANCE_DEDUPE_PROTECTED_SECTIONS.has(ref.sectionId);
}

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

type RetrievedOpenQuestionMerge = {
  canonical: LedgerEntryRef;
  openQuestionId: string;
};

// Retrieval and lifecycle assembly intentionally contribute different facts about one question.
// Their shared machine handle makes this a merge even when both rows land in open_questions, where
// ordinary provenance groups remain independent.
function retrievedOpenQuestionId(ref: LedgerEntryRef): string | null {
  return ref.entry.via_retrieval === true
    ? metadataString(ref.entry.state_metadata?.["open_question_id"])
    : null;
}

function retrievedOpenQuestionMerge(
  group: readonly LedgerEntryRef[],
): RetrievedOpenQuestionMerge | null {
  for (const canonical of group) {
    const openQuestionId = suffixAfterPrefix(canonical.entry.id, "open_question:");

    if (openQuestionId === null) {
      continue;
    }

    if (
      group.length > 1 &&
      group.every((ref) => ref === canonical || retrievedOpenQuestionId(ref) === openQuestionId)
    ) {
      return { canonical, openQuestionId };
    }
  }

  return null;
}

function mergeRetrievedOpenQuestionEntries(
  group: readonly LedgerEntryRef[],
  merge: RetrievedOpenQuestionMerge,
): EvidenceLedgerEntry {
  const retrievedRefs = group.filter((ref) => ref !== merge.canonical);
  const retrievedFields = retrievedRefs.reduce<Partial<EvidenceLedgerEntry>>(
    (fields, ref) => ({ ...fields, ...ref.entry }),
    {},
  );
  const states = [merge.canonical, ...retrievedRefs]
    .map((ref) => ref.entry.state)
    .filter((state): state is string => state !== undefined && state.length > 0);
  const retrievalSources = [
    ...new Set(
      retrievedRefs
        .map((ref) => ref.entry.value)
        .filter((value): value is string => value !== undefined && value.length > 0),
    ),
  ];
  const stateMetadata = Object.assign(
    {},
    ...retrievedRefs.map((ref) => ref.entry.state_metadata ?? {}),
    merge.canonical.entry.state_metadata ?? {},
    {
      open_question_id: merge.openQuestionId,
      ...(retrievalSources.length === 0 ? {} : { retrieval_sources: retrievalSources }),
    },
  ) as Record<string, unknown>;
  const citations = [
    ...new Set([...group.flatMap((ref) => ref.entry.citations ?? []), ...citationHandles(group)]),
  ];

  return {
    ...retrievedFields,
    ...merge.canonical.entry,
    ...(states.length === 0 ? {} : { state: [...new Set(states)].join(" ") }),
    state_metadata: stateMetadata,
    citations,
    via_retrieval: true,
  };
}

export function dedupeEvidenceLedgerByProvenance(ledger: EvidenceLedger): {
  ledger: EvidenceLedger;
  dedupedEntryCount: number;
} {
  const refs: LedgerEntryRef[] = [];
  const parents = new DisjointSet<number>(() => -1);
  const protectedRefsByRoot = new Map<number, Set<number>>();
  const handleOwners = new Map<string, number>();

  const unionRefs = (left: number, right: number): boolean => {
    const leftRoot = parents.find(left);
    const rightRoot = parents.find(right);

    if (leftRoot === rightRoot) {
      return true;
    }

    const leftProtectedRefs = protectedRefsByRoot.get(leftRoot) ?? new Set<number>();
    const rightProtectedRefs = protectedRefsByRoot.get(rightRoot) ?? new Set<number>();

    if (leftProtectedRefs.size > 0 && rightProtectedRefs.size > 0) {
      return false;
    }

    parents.union(leftRoot, rightRoot);

    const root = parents.find(leftRoot);
    const mergedProtectedRefs = new Set([...leftProtectedRefs, ...rightProtectedRefs]);

    protectedRefsByRoot.delete(leftRoot);
    protectedRefsByRoot.delete(rightRoot);

    if (mergedProtectedRefs.size > 0) {
      protectedRefsByRoot.set(root, mergedProtectedRefs);
    }

    return true;
  };

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
      parents.add(refIndex);

      if (isProvenanceDedupeProtected(refs[refIndex]!)) {
        protectedRefsByRoot.set(refIndex, new Set([refIndex]));
      }

      for (const handle of handles) {
        const owner = handleOwners.get(handle);
        if (owner === undefined) {
          handleOwners.set(handle, refIndex);
          continue;
        }

        if (!unionRefs(owner, refIndex) && isProvenanceDedupeProtected(refs[refIndex]!)) {
          handleOwners.set(handle, refIndex);
        }
      }
    }
  }

  const groups = new Map<number, LedgerEntryRef[]>();

  for (const [index, ref] of refs.entries()) {
    const root = parents.find(index);
    groups.set(root, [...(groups.get(root) ?? []), ref]);
  }

  const canonicalByRef = new Map<LedgerEntryRef, EvidenceLedgerEntry>();
  const droppedRefs = new Set<LedgerEntryRef>();
  let dedupedEntryCount = 0;

  for (const group of groups.values()) {
    const sectionIds = new Set(group.map((ref) => ref.sectionId));
    const openQuestionMerge = retrievedOpenQuestionMerge(group);

    if (group.length <= 1 || (sectionIds.size <= 1 && openQuestionMerge === null)) {
      continue;
    }

    const canonical =
      openQuestionMerge?.canonical ??
      (() => {
        const protectedRefs = group.filter(isProvenanceDedupeProtected);
        return [...(protectedRefs.length > 0 ? protectedRefs : group)].sort(
          compareCanonicalRefs,
        )[0]!;
      })();
    canonicalByRef.set(
      canonical,
      openQuestionMerge === null
        ? {
            ...canonical.entry,
            citations: [
              ...new Set([...(canonical.entry.citations ?? []), ...citationHandles(group)]),
            ],
          }
        : mergeRetrievedOpenQuestionEntries(group, openQuestionMerge),
    );

    for (const ref of group) {
      if (ref !== canonical && !isProvenanceDedupeProtected(ref)) {
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
