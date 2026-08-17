import { estimatePromptTokens } from "../../util/token-estimate.js";
import {
  SHARED_STATE_ENTRY_KINDS,
  type SharedStateArtifact,
  type SharedStateEntry,
  type SharedStateEntryKind,
} from "../../memory/shared-state/index.js";
import {
  combineMemoryDisclosureLabels,
  memoryDisclosureInternalUseNote,
  renderMemoryDisclosureLabelFieldsForModel,
  renderMemoryDisclosureLabelForModel,
  type MemoryDisclosureLabel,
} from "../../retrieval/index.js";
import { coercePositiveIntegerOrFallback } from "../../util/math.js";
import { sharedStateMemoryDisclosureLabel } from "../../memory/common/disclosure-serializers.js";
import {
  activeSharedStateArtifactEntries,
  compareSharedStateArtifactEntriesByRecency,
  countSharedStateArtifactEntriesByKind,
  emptySharedStateKindCounts,
  selectSharedStateArtifactEntriesForRenderWithSummary,
  sharedStateEntryHasAnyOperationalCanonicalizer,
  sharedStateEntryHasCriticalCommitmentCanonicalizer,
  sharedStateEntryHasCurrentTurnUpdate,
  sharedStateEntryHasOperationalCanonicalizer,
  subtractSharedStateKindCounts,
  tokenDropIndex,
  type SharedStateKindCounts,
  type SharedStateRenderSalienceOptions,
} from "./selection.js";
import {
  countSharedStateEntriesByKey,
  sharedStateKeyBucket,
  topSharedStateEntryKeysByCount,
} from "./state-key.js";

// Tunes the default maximum number of shared-state entries rendered.
const DEFAULT_SHARED_STATE_MAX_ENTRIES = 40;

// Tunes the default token budget for rendered shared-state content. This budget covers the whole
// section, so the compact index of every active key competes with the expanded bodies for it, and
// the index cost grows with the active set while the entry cap does not. In a saturated audience
// that inverts: measured on a live 40-entry artifact the index alone took 3,514 of these 5,000
// tokens (~88/line over 40 lines), leaving 3-4 bodies rendered out of the 16-40 the selection pass
// had already chosen. The drop loops below, not the selection caps, are what bind there -- so a
// short expanded body count is evidence about this budget, not about `lockedMaxEntries`.
//
// Re-measured 2026-08-17 across all seven live audiences after the index disclosure hoist below:
// index cost fell by 19-962 tokens (largest registers most), expanded bodies rose from 2-4 to 4-5,
// and the maxTokens at which the single-entry floor first trips fell from 4,126-4,248 to 3,245-3,622
// on the three registers that reach it at all. The index is still the dominant term at 40 rows.
const DEFAULT_SHARED_STATE_MAX_TOKENS = 5_000;

// Tunes reserved render slots by shared-state entry kind.
const DEFAULT_SHARED_STATE_RESERVED_SLOTS = {
  live: 8,
  invalidated: 3,
} as const satisfies Partial<Record<SharedStateEntryKind, number>>;

// Tunes the maximum locked shared-state entries rendered.
const DEFAULT_SHARED_STATE_LOCKED_CAP = 14;

// Tunes newest-state-change entries protected during shared-state rendering.
const DEFAULT_NEWEST_STATE_CHANGE_RESERVED_SLOTS = 3;

// Tunes the minimum token floor needed to render one shared-state entry.
const SHARED_STATE_SINGLE_ENTRY_FLOOR_TOKENS = 200;

// Tunes the marker appended when shared-state text is truncated.
const SHARED_STATE_TEXT_TRUNCATION_MARKER = " ... [text truncated]";

// Tunes compact-index excerpt length for shared-state render summaries.
export const SHARED_STATE_COMPACT_INDEX_EXCERPT_CHAR_LIMIT = 80;

// Tunes when shared-state render summaries classify durable turn age as recent.
export const SHARED_STATE_RECENT_TURN_THRESHOLD = 5;

export type SharedStateArtifactRenderSummary = {
  totalEntryCount: number;
  activeEntryCount: number;
  renderedEntryCount: number;
  renderedEntryIds: SharedStateEntry["id"][];
  omittedEntryCount: number;
  estimatedTokens: number;
  newestReservedEntryCount: number;
  renderedByKind: SharedStateKindCounts;
  omittedByKind: SharedStateKindCounts;
  activeByKind: SharedStateKindCounts;
  activeEntriesByKey: Record<string, number>;
  topKeysByEntryCount: Record<string, number>;
  compactIndexEstimatedTokens: number;
  compactIndexLineCount: number;
  allActiveKeysIndexed: boolean;
  omittedLiveRecentOperational: number;
  omittedLiveRecentLowSalience: number;
  omittedLiveOld: number;
  omittedLiveUnknownAge: number;
  omittedLocked: number;
  omittedLockedRecent: number;
  omittedLockedOld: number;
  omittedLockedUnknownAge: number;
  omittedLockedWithActiveCriticalCommitment: number;
  omittedLockedWithOperationalCanonicalizer: number;
  omittedLockedIndexedOnly: number;
  omittedPending: number;
  omittedLowSalienceLive: number;
  omittedDormantLive: number;
};

type SharedStateRenderBudgetOptions = {
  maxEntries?: number;
  maxTokens?: number;
  reservedSlots?: Partial<Record<SharedStateEntryKind, number>>;
  lockedMaxEntries?: number;
  newestStateChangeReservedSlots?: number;
};

export type SharedStateRenderOptions = SharedStateRenderBudgetOptions &
  SharedStateRenderSalienceOptions & {
    currentTurnCounter?: number;
    lastUpdatedTurnByStreamEntryId?: Readonly<Record<string, number>>;
    recentTurnThreshold?: number;
  };

type NormalizedSharedStateRenderOptions = Required<SharedStateRenderBudgetOptions> &
  SharedStateRenderSalienceOptions & {
    currentTurnCounter?: number;
    lastUpdatedTurnByStreamEntryId: Readonly<Record<string, number>>;
    recentTurnThreshold: number;
  };

function sharedStateRenderOptions(
  options: SharedStateRenderOptions = {},
): NormalizedSharedStateRenderOptions {
  return {
    maxEntries: coercePositiveIntegerOrFallback(
      options.maxEntries,
      DEFAULT_SHARED_STATE_MAX_ENTRIES,
    ),
    maxTokens: coercePositiveIntegerOrFallback(options.maxTokens, DEFAULT_SHARED_STATE_MAX_TOKENS),
    reservedSlots: {
      ...DEFAULT_SHARED_STATE_RESERVED_SLOTS,
      ...(options.reservedSlots ?? {}),
    },
    lockedMaxEntries:
      options.lockedMaxEntries === undefined || !Number.isFinite(options.lockedMaxEntries)
        ? DEFAULT_SHARED_STATE_LOCKED_CAP
        : Math.max(0, Math.floor(options.lockedMaxEntries)),
    newestStateChangeReservedSlots:
      options.newestStateChangeReservedSlots === undefined ||
      !Number.isFinite(options.newestStateChangeReservedSlots)
        ? DEFAULT_NEWEST_STATE_CHANGE_RESERVED_SLOTS
        : Math.max(0, Math.floor(options.newestStateChangeReservedSlots)),
    currentUserStreamEntryId: options.currentUserStreamEntryId,
    ledgerStreamEntryIds: options.ledgerStreamEntryIds ?? [],
    activeOpenQuestionIds: options.activeOpenQuestionIds ?? [],
    activeActionIds: options.activeActionIds ?? [],
    activeGoalIds: options.activeGoalIds ?? [],
    activeCriticalCommitmentIds: options.activeCriticalCommitmentIds ?? [],
    activeOperationalCommitmentIds: options.activeOperationalCommitmentIds ?? [],
    recentlyRetrievedEntryIds: options.recentlyRetrievedEntryIds ?? [],
    currentTurnCounter: options.currentTurnCounter,
    lastUpdatedTurnByStreamEntryId: options.lastUpdatedTurnByStreamEntryId ?? {},
    recentTurnThreshold: coercePositiveIntegerOrFallback(
      options.recentTurnThreshold,
      SHARED_STATE_RECENT_TURN_THRESHOLD,
    ),
  };
}

function sharedStateRenderedCounts(input: {
  activeEntries: readonly SharedStateEntry[];
  renderedEntries: readonly SharedStateEntry[];
}): {
  renderedByKind: SharedStateKindCounts;
  omittedByKind: SharedStateKindCounts;
  omittedEntryCount: number;
} {
  const activeByKind = countSharedStateArtifactEntriesByKind(input.activeEntries);
  const renderedByKind = countSharedStateArtifactEntriesByKind(input.renderedEntries);
  const omittedByKind = subtractSharedStateKindCounts(activeByKind, renderedByKind);

  return {
    renderedByKind,
    omittedByKind,
    omittedEntryCount: Math.max(0, input.activeEntries.length - input.renderedEntries.length),
  };
}

function formatSharedStateKindCounts(
  counts: SharedStateKindCounts,
  options: { suffix?: string } = {},
): string {
  const parts = SHARED_STATE_ENTRY_KINDS.flatMap((kind) =>
    counts[kind] <= 0 ? [] : [`${counts[kind]} ${kind}${options.suffix ?? ""}`],
  );

  return parts.length === 0 ? "0 entries" : parts.join(", ");
}

// A supersede keeps the retracted body in the artifact and points it at its replacement, but
// the active set is exactly the rows whose superseded_by_id is null -- so the predecessor drops
// off this surface as completely as a prune deletes one. Without the backward pointer the
// successor is indistinguishable from an entry that replaced nothing, and a retraction that
// happened reads here as if it never did. The pointer is already on the record; only the render
// was dropping it.
function supersededPredecessorIdsBySuccessor(
  artifact: SharedStateArtifact,
): Map<string, string[]> {
  const bySuccessor = new Map<string, string[]>();

  for (const entry of artifact.entries) {
    if (entry.superseded_by_id === null) {
      continue;
    }

    bySuccessor.set(entry.superseded_by_id, [
      ...(bySuccessor.get(entry.superseded_by_id) ?? []),
      entry.id,
    ]);
  }

  return bySuccessor;
}

function renderSharedStateEntry(
  entry: SharedStateEntry,
  supersededPredecessorIds: ReadonlyMap<string, string[]>,
): string {
  const owner = entry.owner_entity_id === null ? "owner=null" : `owner=${entry.owner_entity_id}`;
  const stateKey = `state_key=${sharedStateKeyBucket(entry.state_key)}`;
  const citations = `[citation: ${entry.provenance_stream_entry_ids.join(", ")}]`;
  const disclosureLabel = sharedStateEntryDisclosureLabel(entry);
  const disclosure =
    disclosureLabel.disclosureClass === "public"
      ? ""
      : ` ${renderMemoryDisclosureLabelForModel(disclosureLabel)}`;

  // A body rewritten by a later update carries one stamp for the whole text, so
  // last_updated_at dates the newest sentence and nothing else. Surface created_at
  // only when the two differ: on those entries the stamp does not date the body.
  const created =
    entry.created_at === entry.last_updated_at ? "" : ` created_at=${entry.created_at}`;

  // Name what this entry retracted. Absent means it replaced nothing, so the field's absence
  // carries as much as its presence -- the same predicate created_at uses one field over.
  const predecessors = supersededPredecessorIds.get(entry.id) ?? [];
  const supersedes = predecessors.length === 0 ? "" : ` supersedes=${predecessors.join(",")}`;

  return [
    `- kind=${entry.kind} id=${entry.id} ${stateKey} ${owner}${created}${supersedes} last_updated_at=${entry.last_updated_at}${disclosure} ${citations}`,
    `  text: ${entry.text}`,
  ].join("\n");
}

function sharedStateEntryDisclosureLabel(entry: SharedStateEntry): MemoryDisclosureLabel {
  return sharedStateMemoryDisclosureLabel(entry);
}

function entriesGroupedByStateKey(entries: readonly SharedStateEntry[]): Array<{
  stateKey: string;
  entries: SharedStateEntry[];
}> {
  const groups = new Map<string, SharedStateEntry[]>();

  for (const entry of entries) {
    const key = sharedStateKeyBucket(entry.state_key);
    groups.set(key, [...(groups.get(key) ?? []), entry]);
  }

  return [...groups.entries()]
    .sort(([left], [right]) => left.localeCompare(right))
    .map(([stateKey, groupEntries]) => ({
      stateKey,
      entries: groupEntries,
    }));
}

function sharedStateCompactExcerpt(
  value: string,
  limit: number = SHARED_STATE_COMPACT_INDEX_EXCERPT_CHAR_LIMIT,
): string {
  return value.length <= limit ? value : `${value.slice(0, limit)}...`;
}

function latestSharedStateEntry(entries: readonly SharedStateEntry[]): SharedStateEntry {
  return [...entries].sort(compareSharedStateArtifactEntriesByRecency)[0]!;
}

type SharedStateCompactIndexRow = {
  stateKey: string;
  kinds: SharedStateEntryKind[];
  createdAt: number;
  lastUpdatedAt: number;
  activeCount: number;
  supersededCount: number;
  excerpt: string;
  disclosureLabel: MemoryDisclosureLabel;
  expanded: boolean;
};

function buildSharedStateCompactIndexRows(input: {
  activeEntries: readonly SharedStateEntry[];
  supersededPredecessorIds: ReadonlyMap<string, string[]>;
  expandedBuckets: ReadonlySet<string>;
}): SharedStateCompactIndexRow[] {
  return entriesGroupedByStateKey(input.activeEntries).map((group) => {
    const latestEntry = latestSharedStateEntry(group.entries);
    const kinds = SHARED_STATE_ENTRY_KINDS.filter((kind) =>
      group.entries.some((entry) => entry.kind === kind),
    );

    return {
      stateKey: group.stateKey,
      kinds,
      // The excerpt is the latest entry's body, so its created_at is what dates the
      // text on this line -- the same span the expanded entry line reports.
      createdAt: latestEntry.created_at,
      lastUpdatedAt: Math.max(...group.entries.map((entry) => entry.last_updated_at)),
      activeCount: group.entries.length,
      supersededCount: group.entries.reduce(
        (sum, entry) => sum + (input.supersededPredecessorIds.get(entry.id)?.length ?? 0),
        0,
      ),
      excerpt: sharedStateCompactExcerpt(latestEntry.text),
      disclosureLabel: combineMemoryDisclosureLabels(
        group.entries.map((entry) => sharedStateEntryDisclosureLabel(entry)),
      ),
      expanded: input.expandedBuckets.has(group.stateKey),
    };
  });
}

const SHARED_STATE_COMPACT_INDEX_FIELD_SEPARATOR = " | ";

function sharedStateIndexDisclosureDefaultLine(fields: string): string {
  return `  (disclosure label of every index row below that does not print its own: ${fields})`;
}

// What an index row prints for its own label, given whatever was hoisted above the index. With
// nothing hoisted this is the long-standing shape: fields on every non-public row, nothing on a
// public one. With a default hoisted, a row prints its fields exactly when they differ from that
// default -- including a public row, which has to say so rather than inherit a private default by
// silence.
function sharedStateIndexDisclosureFields(
  row: SharedStateCompactIndexRow,
  hoistedFields: string | null,
): string | null {
  const fields = renderMemoryDisclosureLabelFieldsForModel(row.disclosureLabel);

  if (hoistedFields === null) {
    return row.disclosureLabel.disclosureClass === "public" ? null : fields;
  }

  return fields === hoistedFields ? null : fields;
}

// The disclosure fields are the largest term on an index line and, in a register whose rows all came
// from the same audience, the same bytes over and over: measured on a live 40-row DM register
// (2026-08-17, record_version 105) 34 of 40 lines carried one 106-char string and the other 6 a
// 148-char variant, 4612 chars of the index's 15719 -- more than the state keys and more than the
// whole remaining headroom before the single-entry floor trips. So hoist the most-repeated field
// string to one line above the index and let the rows that differ print their own; a row saying
// nothing is a row that carries the hoisted label, which the line above states in those words.
// Nothing is dropped from any row -- this is the same compaction already applied to the constant
// internal-use note, one field group over.
//
// The gate is the arithmetic, not a tuned threshold: hoisting only happens where repeating the
// string costs more than stating it once plus what the exception rows then have to spend. Public
// rows print no fields today, so they are never the hoist candidate, and where a non-public default
// wins they must start printing their own class -- that added cost is charged to the gate below.
function hoistedSharedStateIndexDisclosureFields(
  rows: readonly SharedStateCompactIndexRow[],
): string | null {
  const countsByFields = new Map<string, number>();

  for (const row of rows) {
    if (row.disclosureLabel.disclosureClass === "public") {
      continue;
    }

    const fields = renderMemoryDisclosureLabelFieldsForModel(row.disclosureLabel);

    countsByFields.set(fields, (countsByFields.get(fields) ?? 0) + 1);
  }

  let candidate: { fields: string; count: number } | null = null;

  for (const [fields, count] of countsByFields) {
    const better =
      candidate === null ||
      count > candidate.count ||
      (count === candidate.count && fields.length > candidate.fields.length);

    if (better) {
      candidate = { fields, count };
    }
  }

  if (candidate === null) {
    return null;
  }

  const savedChars =
    candidate.count * (candidate.fields.length + SHARED_STATE_COMPACT_INDEX_FIELD_SEPARATOR.length);
  const addedChars = rows
    .filter((row) => row.disclosureLabel.disclosureClass === "public")
    .reduce(
      (sum, row) =>
        sum +
        renderMemoryDisclosureLabelFieldsForModel(row.disclosureLabel).length +
        SHARED_STATE_COMPACT_INDEX_FIELD_SEPARATOR.length,
      0,
    );

  return savedChars > sharedStateIndexDisclosureDefaultLine(candidate.fields).length + addedChars
    ? candidate.fields
    : null;
}

function renderSharedStateCompactIndexRows(rows: readonly SharedStateCompactIndexRow[]): string {
  const hoistedDisclosureFields = hoistedSharedStateIndexDisclosureFields(rows);
  const lines = rows.map((row) =>
    [
      `- ${row.stateKey}`,
      `kinds=${row.kinds.join(",")}`,
      // Same predicate as the expanded entry line: print the creation stamp only where it
      // differs, so its absence means "body never rewritten" on an index line too. Without
      // this the omitted rows -- the ones only ever seen through the index -- carry no
      // signal either way, and absence there would read as unknown rather than unrewritten.
      row.createdAt === row.lastUpdatedAt ? null : `created_at=${row.createdAt}`,
      `last_updated_at=${row.lastUpdatedAt}`,
      `active_count=${row.activeCount}`,
      // Only where something was retracted under this key, so an omitted row -- the only
      // thing ever said about most entries -- can still tell a replacement from a first draft.
      row.supersededCount === 0 ? null : `superseded_count=${row.supersededCount}`,
      sharedStateIndexDisclosureFields(row, hoistedDisclosureFields),
      `excerpt=${JSON.stringify(row.excerpt)}`,
      row.expanded ? "expanded" : "omitted",
    ]
      .filter((part): part is string => part !== null)
      .join(SHARED_STATE_COMPACT_INDEX_FIELD_SEPARATOR),
  );

  // The constant internal-use sentence is hoisted here rather than repeated per line. It stays a
  // separate line from the hoisted fields above: this one says what non-public means for the rows
  // that are non-public, which is true whether or not their fields happen to be repeated bytes.
  const disclosureNote = rows.some((row) => row.disclosureLabel.disclosureClass !== "public")
    ? [`  (rows below whose disclosure_class is not public: ${memoryDisclosureInternalUseNote()})`]
    : [];
  const disclosureDefault =
    hoistedDisclosureFields === null
      ? []
      : [sharedStateIndexDisclosureDefaultLine(hoistedDisclosureFields)];

  return [
    "SharedStateArtifact compact active-key index:",
    ...disclosureNote,
    ...disclosureDefault,
    ...lines,
  ].join("\n");
}

function renderSharedStateCompactIndex(input: {
  activeEntries: readonly SharedStateEntry[];
  supersededPredecessorIds: ReadonlyMap<string, string[]>;
  expandedBuckets: ReadonlySet<string>;
}): string {
  return renderSharedStateCompactIndexRows(buildSharedStateCompactIndexRows(input));
}

function allActiveSharedStateKeysIndexed(input: {
  activeEntries: readonly SharedStateEntry[];
  rows: readonly SharedStateCompactIndexRow[];
}): boolean {
  const activeKeys = new Set(
    entriesGroupedByStateKey(input.activeEntries).map((group) => group.stateKey),
  );
  const indexedKeys = new Set(input.rows.map((row) => row.stateKey));

  return (
    activeKeys.size === indexedKeys.size && [...activeKeys].every((key) => indexedKeys.has(key))
  );
}

// The evidence ledger renders the artifact read at turn start, while the shared-state compile that
// runs before the finalizer prompt is built can already have written a newer version to the store.
// The rendered `record_version` and entry set are therefore a snapshot with a name, not a live
// reading, and an unchanged `record_version` across turns is not evidence that no write landed.
const SHARED_STATE_SNAPSHOT_BASIS_LINE =
  "snapshot_basis=turn_start (this artifact and its record_version were read before this turn's shared-state compile, which may already have advanced both)";

// Omission here is this render's token budget spent over the entries the store already returned as
// active -- a different mechanism from the store's lifecycle cap, which is the one that deletes.
// Without saying so, `omitted` on an index row next to a store known to evict at a cap reads as
// "there is no room for this", and the entry's own body looks unreachable until something else
// leaves. Nothing has to leave: the omitted rows are still active and unchanged, and what promotes
// one into the rendered body is render salience (a current-turn update ranks first), not a free slot.
const SHARED_STATE_OMISSION_BASIS_LINE =
  "omission_basis=render_budget (omitted entries are still active and unchanged in the store; omission is this render's token budget over the active set, not the store's lifecycle cap, so expanding an omitted key costs render salience, not a stored slot)";

function renderSharedStateArtifactContent(input: {
  artifact: SharedStateArtifact;
  activeEntries: readonly SharedStateEntry[];
  entries: readonly SharedStateEntry[];
  omittedByKind: SharedStateKindCounts;
  renderedByKind: SharedStateKindCounts;
}): string {
  const omittedCount = Object.values(input.omittedByKind).reduce((sum, count) => sum + count, 0);
  const supersededPredecessorIds = supersededPredecessorIdsBySuccessor(input.artifact);
  const expandedBuckets = new Set(
    input.entries.map((entry) => sharedStateKeyBucket(entry.state_key)),
  );
  const omission =
    omittedCount <= 0
      ? null
      : [
          `SharedStateArtifact omitted: ${formatSharedStateKindCounts(input.omittedByKind)}.`,
          `Retained: ${formatSharedStateKindCounts(input.renderedByKind)}.`,
          SHARED_STATE_OMISSION_BASIS_LINE,
        ].join(" ");

  return [
    "## 0. Shared Audience State",
    "SharedStateArtifact: durable shared state for this audience. It is a compact structural anchor, not a policy source.",
    `audience_entity_id=${input.artifact.audience_entity_id}`,
    `record_version=${input.artifact.record_version}`,
    SHARED_STATE_SNAPSHOT_BASIS_LINE,
    renderSharedStateCompactIndex({
      activeEntries: input.activeEntries,
      supersededPredecessorIds,
      expandedBuckets,
    }),
    ...entriesGroupedByStateKey(input.entries).flatMap((group) => [
      `state_key_bucket=${group.stateKey}`,
      ...group.entries.map((groupEntry) =>
        renderSharedStateEntry(groupEntry, supersededPredecessorIds),
      ),
    ]),
    omission,
  ]
    .filter((part): part is string => part !== null)
    .join("\n");
}

// The zero-body render that carries a reason. Reached from exactly one place -- the single-entry cap
// below, after the drop loops have already narrowed to one entry that still does not fit -- so it is
// never "the budget bought no bodies at all". That outcome is the drop loop stopping at three or four
// bodies, which is a different code path with a different surface. What gates this one is
// SHARED_STATE_SINGLE_ENTRY_FLOOR_TOKENS measured against the compact index, not against the body
// pool: the index's token cost keeps growing with key length and per-row optional fields long after
// its line count has hit the active-entry cap, so a full index is not a bounded index. The other
// zero-body outcome renders through renderSharedStateArtifactContent with an empty entry list (see
// `remainingTokens <= 0` below); the two are told apart by the footer, which prints `Retained:` with
// zero counts there and `Reason:` here.
function renderSharedStateArtifactOmissionOnly(input: {
  artifact: SharedStateArtifact;
  activeEntries: readonly SharedStateEntry[];
  omittedByKind: SharedStateKindCounts;
  reason: string;
}): string {
  return [
    "## 0. Shared Audience State",
    "SharedStateArtifact: durable shared state for this audience. It is a compact structural anchor, not a policy source.",
    `audience_entity_id=${input.artifact.audience_entity_id}`,
    `record_version=${input.artifact.record_version}`,
    SHARED_STATE_SNAPSHOT_BASIS_LINE,
    renderSharedStateCompactIndex({
      activeEntries: input.activeEntries,
      supersededPredecessorIds: supersededPredecessorIdsBySuccessor(input.artifact),
      expandedBuckets: new Set(),
    }),
    `SharedStateArtifact omitted: ${formatSharedStateKindCounts(
      input.omittedByKind,
    )}. Reason: ${input.reason}. ${SHARED_STATE_OMISSION_BASIS_LINE}`,
  ].join("\n");
}

export function truncateSharedStateArtifactText(value: string, maxTokens: number): string {
  const maxChars = Math.max(
    0,
    Math.floor(maxTokens) * 4 - SHARED_STATE_TEXT_TRUNCATION_MARKER.length,
  );

  return `${value.slice(0, maxChars).trimEnd()}${SHARED_STATE_TEXT_TRUNCATION_MARKER}`;
}

function renderSingleEntryWithinSharedStateArtifactCap(input: {
  artifact: SharedStateArtifact;
  entry: SharedStateEntry;
  activeEntries: readonly SharedStateEntry[];
  maxTokens: number;
}): { content: string; renderedEntryCount: number; omittedEntryCount: number } {
  const counts = sharedStateRenderedCounts({
    activeEntries: input.activeEntries,
    renderedEntries: [input.entry],
  });
  const emptyEntryContent = renderSharedStateArtifactContent({
    artifact: input.artifact,
    activeEntries: input.activeEntries,
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

  if (remainingTokens < SHARED_STATE_SINGLE_ENTRY_FLOOR_TOKENS) {
    return {
      content: renderSharedStateArtifactOmissionOnly({
        artifact: input.artifact,
        activeEntries: input.activeEntries,
        omittedByKind: countSharedStateArtifactEntriesByKind(input.activeEntries),
        reason: "artifact entry too large to render",
      }),
      renderedEntryCount: 0,
      omittedEntryCount: input.activeEntries.length,
    };
  }

  const content = renderSharedStateArtifactContent({
    artifact: input.artifact,
    activeEntries: input.activeEntries,
    entries: [
      {
        ...input.entry,
        text: truncateSharedStateArtifactText(input.entry.text, remainingTokens),
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
    content: renderSharedStateArtifactOmissionOnly({
      artifact: input.artifact,
      activeEntries: input.activeEntries,
      omittedByKind: countSharedStateArtifactEntriesByKind(input.activeEntries),
      reason: "artifact entry too large to render",
    }),
    renderedEntryCount: 0,
    omittedEntryCount: input.activeEntries.length,
  };
}

function renderTruncatedEntriesWithinSharedStateArtifactCap(input: {
  artifact: SharedStateArtifact;
  entries: readonly SharedStateEntry[];
  activeEntries: readonly SharedStateEntry[];
  maxTokens: number;
}): { content: string; entries: SharedStateEntry[] } {
  const counts = sharedStateRenderedCounts({
    activeEntries: input.activeEntries,
    renderedEntries: input.entries,
  });
  const emptyEntryContent = renderSharedStateArtifactContent({
    artifact: input.artifact,
    activeEntries: input.activeEntries,
    entries: input.entries.map((entry) => ({
      ...entry,
      text: "",
    })),
    omittedByKind: counts.omittedByKind,
    renderedByKind: counts.renderedByKind,
  });
  const remainingTokens = input.maxTokens - estimatePromptTokens(emptyEntryContent);

  // The second zero-body outcome, and not the omission-only one above: this keeps the full content
  // shape with no entries, so it still prints `Retained:` (all zeros) and never a reason string.
  // Reaching it takes an index that exceeds the whole budget on its own -- one step past the floor
  // that trips the omission-only render.
  if (remainingTokens <= 0) {
    return {
      content: renderSharedStateArtifactContent({
        artifact: input.artifact,
        activeEntries: input.activeEntries,
        entries: [],
        omittedByKind: countSharedStateArtifactEntriesByKind(input.activeEntries),
        renderedByKind: emptySharedStateKindCounts(),
      }),
      entries: [],
    };
  }

  const entryTextTokens = Math.max(1, Math.floor(remainingTokens / input.entries.length));
  const truncatedEntries = input.entries.map((entry) => ({
    ...entry,
    text: truncateSharedStateArtifactText(entry.text, entryTextTokens),
  }));

  return {
    content: renderSharedStateArtifactContent({
      artifact: input.artifact,
      activeEntries: input.activeEntries,
      entries: truncatedEntries,
      omittedByKind: counts.omittedByKind,
      renderedByKind: counts.renderedByKind,
    }),
    entries: truncatedEntries,
  };
}

function sharedStateEntryLastUpdatedTurn(
  entry: SharedStateEntry,
  lastUpdatedTurnByStreamEntryId: Readonly<Record<string, number>>,
): number | null {
  if (entry.last_updated_turn_global !== null && Number.isFinite(entry.last_updated_turn_global)) {
    return entry.last_updated_turn_global;
  }

  let lastTurn: number | null = null;

  for (const streamEntryId of entry.last_updated_stream_entry_ids) {
    const turn = lastUpdatedTurnByStreamEntryId[streamEntryId];

    if (turn !== undefined && Number.isFinite(turn)) {
      lastTurn = lastTurn === null ? turn : Math.max(lastTurn, turn);
    }
  }

  return lastTurn;
}

function sharedStateEntryRecencyStatus(
  entry: SharedStateEntry,
  options: NormalizedSharedStateRenderOptions,
): "recent" | "old" | "unknown" {
  if (sharedStateEntryHasCurrentTurnUpdate(entry, options.currentUserStreamEntryId)) {
    return "recent";
  }

  if (options.currentTurnCounter === undefined) {
    return "unknown";
  }

  const lastUpdatedTurn = sharedStateEntryLastUpdatedTurn(
    entry,
    options.lastUpdatedTurnByStreamEntryId,
  );

  if (lastUpdatedTurn === null) {
    return "unknown";
  }

  return options.currentTurnCounter - lastUpdatedTurn <= options.recentTurnThreshold
    ? "recent"
    : "old";
}

function sharedStateEntryIsOperational(
  entry: SharedStateEntry,
  options: NormalizedSharedStateRenderOptions,
): boolean {
  return (
    sharedStateEntryHasCurrentTurnUpdate(entry, options.currentUserStreamEntryId) ||
    sharedStateEntryHasOperationalCanonicalizer(entry, options) ||
    sharedStateEntryHasAnyOperationalCanonicalizer(entry)
  );
}

function sharedStateOmissionSeverity(input: {
  activeEntries: readonly SharedStateEntry[];
  renderedEntries: readonly SharedStateEntry[];
  options: NormalizedSharedStateRenderOptions;
  indexedStateKeyBuckets: ReadonlySet<string>;
}): Pick<
  SharedStateArtifactRenderSummary,
  | "omittedLiveRecentOperational"
  | "omittedLiveRecentLowSalience"
  | "omittedLiveOld"
  | "omittedLiveUnknownAge"
  | "omittedLocked"
  | "omittedLockedRecent"
  | "omittedLockedOld"
  | "omittedLockedUnknownAge"
  | "omittedLockedWithActiveCriticalCommitment"
  | "omittedLockedWithOperationalCanonicalizer"
  | "omittedLockedIndexedOnly"
  | "omittedPending"
  | "omittedLowSalienceLive"
  | "omittedDormantLive"
> {
  const renderedIds = new Set(input.renderedEntries.map((entry) => entry.id));
  let omittedLiveRecentOperational = 0;
  let omittedLiveRecentLowSalience = 0;
  let omittedLiveOld = 0;
  let omittedLiveUnknownAge = 0;
  let omittedLocked = 0;
  let omittedLockedRecent = 0;
  let omittedLockedOld = 0;
  let omittedLockedUnknownAge = 0;
  let omittedLockedWithActiveCriticalCommitment = 0;
  let omittedLockedWithOperationalCanonicalizer = 0;
  let omittedLockedIndexedOnly = 0;
  let omittedPending = 0;
  let omittedLowSalienceLive = 0;
  let omittedDormantLive = 0;

  for (const entry of input.activeEntries) {
    if (renderedIds.has(entry.id)) {
      continue;
    }

    if (entry.kind === "locked") {
      omittedLocked += 1;
      const recency = sharedStateEntryRecencyStatus(entry, input.options);
      if (recency === "recent") {
        omittedLockedRecent += 1;
      } else if (recency === "old") {
        omittedLockedOld += 1;
      } else {
        omittedLockedUnknownAge += 1;
      }

      if (
        sharedStateEntryHasCriticalCommitmentCanonicalizer(
          entry,
          input.options.activeCriticalCommitmentIds,
        )
      ) {
        omittedLockedWithActiveCriticalCommitment += 1;
      }

      if (sharedStateEntryHasOperationalCanonicalizer(entry, input.options)) {
        omittedLockedWithOperationalCanonicalizer += 1;
      }

      if (input.indexedStateKeyBuckets.has(sharedStateKeyBucket(entry.state_key))) {
        omittedLockedIndexedOnly += 1;
      }
    }

    if (entry.kind === "pending") {
      omittedPending += 1;
    }

    if (entry.kind === "low_salience_live") {
      omittedLowSalienceLive += 1;
    }

    if (entry.kind === "dormant_live") {
      omittedDormantLive += 1;
    }

    if (entry.kind !== "live") {
      continue;
    }

    const recency = sharedStateEntryRecencyStatus(entry, input.options);

    if (recency === "old") {
      omittedLiveOld += 1;
      continue;
    }

    if (recency === "unknown") {
      omittedLiveUnknownAge += 1;
      continue;
    }

    if (sharedStateEntryIsOperational(entry, input.options)) {
      omittedLiveRecentOperational += 1;
    } else {
      omittedLiveRecentLowSalience += 1;
    }
  }

  return {
    omittedLiveRecentOperational,
    omittedLiveRecentLowSalience,
    omittedLiveOld,
    omittedLiveUnknownAge,
    omittedLocked,
    omittedLockedRecent,
    omittedLockedOld,
    omittedLockedUnknownAge,
    omittedLockedWithActiveCriticalCommitment,
    omittedLockedWithOperationalCanonicalizer,
    omittedLockedIndexedOnly,
    omittedPending,
    omittedLowSalienceLive,
    omittedDormantLive,
  };
}

function cappedSharedStateArtifactRender(input: {
  artifact: SharedStateArtifact;
  options?: SharedStateRenderOptions;
}): { content: string | null; summary: SharedStateArtifactRenderSummary } {
  const options = sharedStateRenderOptions(input.options);
  const activeEntries = activeSharedStateArtifactEntries(input.artifact);

  if (activeEntries.length === 0) {
    return {
      content: null,
      summary: {
        totalEntryCount: input.artifact.entries.length,
        activeEntryCount: 0,
        renderedEntryCount: 0,
        renderedEntryIds: [],
        omittedEntryCount: 0,
        estimatedTokens: 0,
        newestReservedEntryCount: 0,
        renderedByKind: emptySharedStateKindCounts(),
        omittedByKind: emptySharedStateKindCounts(),
        activeByKind: emptySharedStateKindCounts(),
        activeEntriesByKey: {},
        topKeysByEntryCount: {},
        compactIndexEstimatedTokens: 0,
        compactIndexLineCount: 0,
        allActiveKeysIndexed: true,
        omittedLiveRecentOperational: 0,
        omittedLiveRecentLowSalience: 0,
        omittedLiveOld: 0,
        omittedLiveUnknownAge: 0,
        omittedLocked: 0,
        omittedLockedRecent: 0,
        omittedLockedOld: 0,
        omittedLockedUnknownAge: 0,
        omittedLockedWithActiveCriticalCommitment: 0,
        omittedLockedWithOperationalCanonicalizer: 0,
        omittedLockedIndexedOnly: 0,
        omittedPending: 0,
        omittedLowSalienceLive: 0,
        omittedDormantLive: 0,
      },
    };
  }

  const activeCounts = countSharedStateArtifactEntriesByKind(activeEntries);
  const selection = selectSharedStateArtifactEntriesForRenderWithSummary({
    entries: activeEntries,
    maxEntries: options.maxEntries,
    reservedSlots: options.reservedSlots,
    lockedMaxEntries: options.lockedMaxEntries,
    newestStateChangeReservedSlots: options.newestStateChangeReservedSlots,
    salience: options,
  });
  const newestReservedIds = selection.newestReservedIds;
  let entries = selection.entries;
  let counts = sharedStateRenderedCounts({
    activeEntries,
    renderedEntries: entries,
  });
  let content = renderSharedStateArtifactContent({
    artifact: input.artifact,
    activeEntries,
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
      dropTiers: selection.dropTiers,
    });
    if (dropIndex === null) {
      break;
    }

    entries = [...entries.slice(0, dropIndex), ...entries.slice(dropIndex + 1)];
    counts = sharedStateRenderedCounts({
      activeEntries,
      renderedEntries: entries,
    });
    content = renderSharedStateArtifactContent({
      artifact: input.artifact,
      activeEntries,
      entries,
      omittedByKind: counts.omittedByKind,
      renderedByKind: counts.renderedByKind,
    });
  }

  if (estimatePromptTokens(content) > options.maxTokens && entries.length > 1) {
    const truncatedRender = renderTruncatedEntriesWithinSharedStateArtifactCap({
      artifact: input.artifact,
      entries,
      activeEntries,
      maxTokens: options.maxTokens,
    });

    content = truncatedRender.content;
    entries = truncatedRender.entries;
    counts = sharedStateRenderedCounts({
      activeEntries,
      renderedEntries: entries,
    });
  }

  while (estimatePromptTokens(content) > options.maxTokens && entries.length > 1) {
    entries = entries.slice(0, -1);
    const truncatedRender = renderTruncatedEntriesWithinSharedStateArtifactCap({
      artifact: input.artifact,
      entries,
      activeEntries,
      maxTokens: options.maxTokens,
    });

    content = truncatedRender.content;
    entries = truncatedRender.entries;
    counts = sharedStateRenderedCounts({
      activeEntries,
      renderedEntries: entries,
    });
  }

  if (estimatePromptTokens(content) > options.maxTokens && entries.length === 1) {
    const singleEntryRender = renderSingleEntryWithinSharedStateArtifactCap({
      artifact: input.artifact,
      entry: entries[0]!,
      activeEntries,
      maxTokens: options.maxTokens,
    });

    content = singleEntryRender.content;
    entries = entries.slice(0, singleEntryRender.renderedEntryCount);
    counts = sharedStateRenderedCounts({
      activeEntries,
      renderedEntries: entries,
    });
  }

  const expandedBuckets = new Set(entries.map((entry) => sharedStateKeyBucket(entry.state_key)));
  const compactIndexRows = buildSharedStateCompactIndexRows({
    activeEntries,
    supersededPredecessorIds: supersededPredecessorIdsBySuccessor(input.artifact),
    expandedBuckets,
  });
  const indexedStateKeyBuckets = new Set(compactIndexRows.map((row) => row.stateKey));

  return {
    content,
    summary: {
      totalEntryCount: input.artifact.entries.length,
      activeEntryCount: activeEntries.length,
      renderedEntryCount: entries.length,
      renderedEntryIds: entries.map((entry) => entry.id),
      omittedEntryCount: counts.omittedEntryCount,
      estimatedTokens: estimatePromptTokens(content),
      newestReservedEntryCount: entries.filter((entry) => newestReservedIds.has(entry.id)).length,
      renderedByKind: counts.renderedByKind,
      omittedByKind: counts.omittedByKind,
      activeByKind: activeCounts,
      activeEntriesByKey: countSharedStateEntriesByKey(activeEntries),
      topKeysByEntryCount: topSharedStateEntryKeysByCount(
        countSharedStateEntriesByKey(activeEntries),
        5,
      ),
      compactIndexEstimatedTokens: estimatePromptTokens(
        renderSharedStateCompactIndexRows(compactIndexRows),
      ),
      compactIndexLineCount: compactIndexRows.length,
      allActiveKeysIndexed: allActiveSharedStateKeysIndexed({
        activeEntries,
        rows: compactIndexRows,
      }),
      ...sharedStateOmissionSeverity({
        activeEntries,
        renderedEntries: entries,
        options,
        indexedStateKeyBuckets,
      }),
    },
  };
}

export function renderSharedStateArtifact(
  artifact: SharedStateArtifact | null | undefined,
  options?: SharedStateRenderOptions,
): string | null {
  if (artifact === null || artifact === undefined) {
    return null;
  }

  return cappedSharedStateArtifactRender({
    artifact,
    options,
  }).content;
}

export function summarizeSharedStateArtifactRender(
  artifact: SharedStateArtifact | null | undefined,
  options?: SharedStateRenderOptions,
): SharedStateArtifactRenderSummary {
  if (artifact === null || artifact === undefined) {
    return {
      totalEntryCount: 0,
      activeEntryCount: 0,
      renderedEntryCount: 0,
      renderedEntryIds: [],
      omittedEntryCount: 0,
      estimatedTokens: 0,
      newestReservedEntryCount: 0,
      renderedByKind: emptySharedStateKindCounts(),
      omittedByKind: emptySharedStateKindCounts(),
      activeByKind: emptySharedStateKindCounts(),
      activeEntriesByKey: {},
      topKeysByEntryCount: {},
      compactIndexEstimatedTokens: 0,
      compactIndexLineCount: 0,
      allActiveKeysIndexed: true,
      omittedLiveRecentOperational: 0,
      omittedLiveRecentLowSalience: 0,
      omittedLiveOld: 0,
      omittedLiveUnknownAge: 0,
      omittedLocked: 0,
      omittedLockedRecent: 0,
      omittedLockedOld: 0,
      omittedLockedUnknownAge: 0,
      omittedLockedWithActiveCriticalCommitment: 0,
      omittedLockedWithOperationalCanonicalizer: 0,
      omittedLockedIndexedOnly: 0,
      omittedPending: 0,
      omittedLowSalienceLive: 0,
      omittedDormantLive: 0,
    };
  }

  return cappedSharedStateArtifactRender({
    artifact,
    options,
  }).summary;
}
