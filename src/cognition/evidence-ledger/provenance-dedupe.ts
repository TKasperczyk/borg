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

const OBJECT_ID_PREFIXES = [
  ["current_session_stream:", "stream"],
  ["episode:", "episode"],
  ["semantic_node:", "semantic_node"],
  ["semantic_edge:", "semantic_edge"],
  ["open_question:", "open_question"],
] as const;

// A retrieved row's metadata names the object it renders next to the evidence that object rests on
// (an episode row also carries its source stream ids), so the most specific id present is the object.
const RETRIEVED_OBJECT_ID_KEYS = [
  ["open_question_id", "open_question"],
  ["commitment_id", "commitment"],
  ["image_perception_id", "image_perception"],
  ["edge_id", "semantic_edge"],
  ["node_id", "semantic_node"],
  ["episode_id", "episode"],
] as const;

// The one object a row renders, as opposed to the evidence it cites. Two rows with the same object
// show the same memory twice, so they fold even inside one section, where rows that merely share
// evidence stay apart. A row naming no single object has none and folds only by provenance.
function objectHandle(entry: EvidenceLedgerEntry): string | null {
  for (const [prefix, type] of OBJECT_ID_PREFIXES) {
    const value = suffixAfterPrefix(entry.id, prefix);

    if (value !== null && value.length > 0) {
      return `${type}:${value}`;
    }
  }

  if (
    entry.id.startsWith("commitment:") ||
    entry.id.startsWith("group_commitment:") ||
    entry.id.startsWith("participant_commitment:")
  ) {
    const commitmentId = suffixAfterLastColon(entry.id);
    return commitmentId === null ? null : `commitment:${commitmentId}`;
  }

  // A raw-stream row's text is the message itself; elsewhere a stream id is only evidence.
  if (entry.id.startsWith("retrieved_stream:")) {
    const streamIds = metadataStringArray(entry.state_metadata, "stream_ids");
    return streamIds.length === 1 ? `stream:${streamIds[0]!}` : null;
  }

  if (!entry.id.startsWith("retrieved_evidence:")) {
    return null;
  }

  for (const [key, type] of RETRIEVED_OBJECT_ID_KEYS) {
    const value = metadataString(entry.state_metadata?.[key]);

    if (value !== null) {
      return `${type}:${value}`;
    }
  }

  return null;
}

// How a row groups in one dedupe pass. By default it groups by every evidence handle it carries.
// "object" groups it only with rows rendering the same object; "section_object" only with those of
// them in its own section.
export type ProvenanceDedupeGrouping = "object" | "section_object";

type LedgerEntryRef = {
  sectionId: EvidenceLedgerSectionId;
  sectionIndex: number;
  entryIndex: number;
  entry: EvidenceLedgerEntry;
  handles: Set<string>;
  sourceHandles: Set<string>;
  object: string | null;
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
        .flatMap((ref) => [...ref.sourceHandles])
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

type FoldGroup = {
  group: LedgerEntryRef[];
  openQuestionMerge: RetrievedOpenQuestionMerge | null;
};

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

export function provenanceDedupeEntryKey(
  sectionId: EvidenceLedgerSectionId,
  entryId: string,
): string {
  return `${sectionId}\u0000${entryId}`;
}

function pushIndexed(index: Map<string, LedgerEntryRef[]>, key: string, ref: LedgerEntryRef): void {
  const refs = index.get(key);

  if (refs === undefined) {
    index.set(key, [ref]);
  } else {
    refs.push(ref);
  }
}

// Shared handles chain a group together, so two of its rows can share no evidence at all. A row
// folds only into a canonical that renders its object or shares one of its handles from another
// section; folded along the chain instead, its text would leave the page under a row it has nothing
// in common with. The rows that render one object fold together, where the first of them in
// canonical order would: split between two canonicals, a cut of one would put its rows back on the
// page beside the others. In canonical order, each row not yet placed becomes a canonical and folds
// every unplaced row that qualifies.
function directFoldGroups(group: readonly LedgerEntryRef[]): LedgerEntryRef[][] {
  const ordered = [...group].sort(
    (left, right) =>
      Number(isProvenanceDedupeProtected(right)) - Number(isProvenanceDedupeProtected(left)) ||
      compareCanonicalRefs(left, right),
  );
  const refsByHandle = new Map<string, LedgerEntryRef[]>();
  const refsByObject = new Map<string, LedgerEntryRef[]>();

  for (const ref of ordered) {
    for (const handle of ref.handles) {
      pushIndexed(refsByHandle, handle, ref);
    }

    if (ref.object !== null) {
      pushIndexed(refsByObject, ref.object, ref);
    }
  }

  const objectRows = (ref: LedgerEntryRef): LedgerEntryRef[] =>
    ref.object === null ? [ref] : (refsByObject.get(ref.object) ?? [ref]);
  const placed = new Set<LedgerEntryRef>();
  const foldGroups: LedgerEntryRef[][] = [];

  for (const canonical of ordered) {
    if (placed.has(canonical)) {
      continue;
    }

    placed.add(canonical);
    const folded = new Set<LedgerEntryRef>();
    const candidates = [
      ...objectRows(canonical),
      ...[...canonical.handles].flatMap((handle) =>
        (refsByHandle.get(handle) ?? [])
          .filter(
            (ref) =>
              ref.sectionId !== canonical.sectionId &&
              objectRows(ref)[0] === ref &&
              !isProvenanceDedupeProtected(ref),
          )
          .flatMap(objectRows),
      ),
    ];

    for (const ref of candidates) {
      if (!placed.has(ref) && !isProvenanceDedupeProtected(ref)) {
        folded.add(ref);
        placed.add(ref);
      }
    }

    if (folded.size > 0) {
      foldGroups.push(group.filter((ref) => ref === canonical || folded.has(ref)));
    }
  }

  return foldGroups;
}

// A row keyed in `groupingByEntryKey` groups as its value says; every other row groups by all its
// evidence handles. A keyed row naming no object stays in its section but neither absorbs nor is
// folded.
// `foldedEntryKeysByCanonicalKey` maps each canonical that folded at least one row into itself to
// the keys of the rows it folded.
export function dedupeEvidenceLedgerByProvenance(
  ledger: EvidenceLedger,
  options: { groupingByEntryKey?: ReadonlyMap<string, ProvenanceDedupeGrouping> } = {},
): {
  ledger: EvidenceLedger;
  dedupedEntryCount: number;
  foldedEntryKeysByCanonicalKey: Map<string, string[]>;
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
      const sourceHandles = evidenceHandles(section.id, entry);
      const object = objectHandle(entry);
      const grouping = options.groupingByEntryKey?.get(
        provenanceDedupeEntryKey(section.id, entry.id),
      );
      // Object handles live in their own namespaces, so a keyed row never joins a row that groups
      // another way.
      const handles =
        grouping === undefined
          ? sourceHandles
          : object === null
            ? new Set<string>()
            : new Set([
                grouping === "object"
                  ? `object:${object}`
                  : `section_object:${section.id}:${object}`,
              ]);
      const refIndex = refs.length;
      refs.push({
        sectionId: section.id,
        sectionIndex,
        entryIndex,
        entry,
        handles,
        sourceHandles,
        object,
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
  const droppedCountsBySectionIndex = new Map<number, number>();
  const foldedEntryKeysByCanonicalKey = new Map<string, string[]>();
  let dedupedEntryCount = 0;

  const foldGroups = [...groups.values()].flatMap((group): FoldGroup[] => {
    if (group.length <= 1) {
      return [];
    }

    const openQuestionMerge = retrievedOpenQuestionMerge(group);

    return openQuestionMerge !== null
      ? [{ group, openQuestionMerge }]
      : directFoldGroups(group).map((foldGroup) => ({
          group: foldGroup,
          openQuestionMerge: null,
        }));
  });

  for (const { group, openQuestionMerge } of foldGroups) {
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

    const foldedEntryKeys: string[] = [];

    for (const ref of group) {
      if (ref !== canonical && !isProvenanceDedupeProtected(ref)) {
        droppedRefs.add(ref);
        droppedCountsBySectionIndex.set(
          ref.sectionIndex,
          (droppedCountsBySectionIndex.get(ref.sectionIndex) ?? 0) + 1,
        );
        dedupedEntryCount += 1;
        foldedEntryKeys.push(provenanceDedupeEntryKey(ref.sectionId, ref.entry.id));
      }
    }

    if (foldedEntryKeys.length > 0) {
      const canonicalKey = provenanceDedupeEntryKey(canonical.sectionId, canonical.entry.id);
      foldedEntryKeysByCanonicalKey.set(canonicalKey, [
        ...(foldedEntryKeysByCanonicalKey.get(canonicalKey) ?? []),
        ...foldedEntryKeys,
      ]);
    }
  }

  // A section's framing counts are taken at assembly, upstream of this stage, and until now this
  // stage was the one reduction that reported nothing anywhere: the budget cut announces itself as
  // an evidence_ledger_omitted entry, and a fold left the shortfall between the population and the
  // page unaccountable from the page. Writing the per-section drop count back onto the framing the
  // section already carries closes that -- the two reductions subtract from rows_assembled to the
  // rows printed, and a residue in that subtraction is a stage neither of them names. Zero is
  // printed rather than omitted, so "nothing was folded" is a statement rather than an absence.
  const sections = ledger.sections.map((section, sectionIndex) => {
    const entries = section.entries.flatMap((entry, entryIndex) => {
      const ref = refs.find(
        (candidate) =>
          candidate.sectionIndex === sectionIndex && candidate.entryIndex === entryIndex,
      );

      if (ref === undefined || droppedRefs.has(ref)) {
        return [];
      }

      return [canonicalByRef.get(ref) ?? entry];
    });

    if (section.framing?.counts === undefined) {
      return { ...section, entries };
    }

    return {
      ...section,
      framing: {
        ...section.framing,
        counts: {
          ...section.framing.counts,
          folded_out_by_provenance: droppedCountsBySectionIndex.get(sectionIndex) ?? 0,
        },
      },
      entries,
    };
  });

  return {
    ledger: cloneLedgerWithSections(ledger, sections),
    dedupedEntryCount,
    foldedEntryKeysByCanonicalKey,
  };
}

// A row folded into a kept canonical reaches the page only as ids, so it is redundant only while
// the page shows the evidence it rests on: a handle of the canonical's own, or an object a kept row
// renders or folded rows render that rest on the page themselves. Lists the folded rows of kept
// canonicals resting on anything else, each with the folded rows that render its object. A row
// rendering the canonical's own object is that memory again and is never listed.
export function foldedEntryKeysWithEvidenceOffPage(
  ledger: EvidenceLedger,
  foldedEntryKeysByCanonicalKey: ReadonlyMap<string, readonly string[]>,
  keptEntryKeys: ReadonlySet<string>,
): string[] {
  type Row = { sourceHandles: Set<string>; object: string | null };
  const rowsByKey = new Map<string, Row>();

  for (const section of ledger.sections) {
    for (const entry of section.entries) {
      rowsByKey.set(provenanceDedupeEntryKey(section.id, entry.id), {
        sourceHandles: evidenceHandles(section.id, entry),
        object: objectHandle(entry),
      });
    }
  }

  const folds = [...foldedEntryKeysByCanonicalKey].flatMap(([canonicalKey, foldedEntryKeys]) => {
    const canonical = rowsByKey.get(canonicalKey);

    if (canonical === undefined || !keptEntryKeys.has(canonicalKey)) {
      return [];
    }

    return foldedEntryKeys.flatMap((key) => {
      const row = rowsByKey.get(key);
      return row === undefined ? [] : [{ key, row, canonical }];
    });
  });
  const onPage = new Set(
    [...keptEntryKeys].flatMap((key) => {
      const object = rowsByKey.get(key)?.object;
      return object === null || object === undefined ? [] : [object];
    }),
  );
  const restsOnPage = ({ row, canonical }: { row: Row; canonical: Row }) =>
    [...row.sourceHandles].every(
      (handle) => handle === row.object || canonical.sourceHandles.has(handle) || onPage.has(handle),
    );

  for (let grew = true; grew; ) {
    grew = false;

    for (const { row } of folds) {
      const { object } = row;

      if (
        object !== null &&
        !onPage.has(object) &&
        folds.every((other) => other.row.object !== object || restsOnPage(other))
      ) {
        onPage.add(object);
        grew = true;
      }
    }
  }

  const offPage = folds.filter(
    (fold) =>
      (fold.row.object === null || fold.row.object !== fold.canonical.object) &&
      !restsOnPage(fold),
  );
  const offPageKeys = new Set(offPage.map((fold) => fold.key));
  const offPageObjects = new Set(offPage.flatMap((fold) => fold.row.object ?? []));

  return folds.flatMap(({ key, row }) =>
    offPageKeys.has(key) || (row.object !== null && offPageObjects.has(row.object)) ? [key] : [],
  );
}
