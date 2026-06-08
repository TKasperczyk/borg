import { estimatePromptTokens } from "../../util/token-estimate.js";
import {
  SHARED_STATE_ENTRY_KINDS,
  type SharedStateArtifact,
  type SharedStateEntry,
  type SharedStateEntryKind,
} from "../../memory/decision-artifacts/index.js";
import {
  combineMemoryDisclosureLabels,
  renderMemoryDisclosureLabelForModel,
  type MemoryDisclosureLabel,
} from "../../retrieval/index.js";
import { coercePositiveIntegerOrFallback } from "../../util/math.js";
import { sharedStateMemoryDisclosureLabel } from "../disclosure-labels.js";
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

const DEFAULT_SHARED_STATE_MAX_ENTRIES = 40;
const DEFAULT_SHARED_STATE_MAX_TOKENS = 5_000;
const DEFAULT_SHARED_STATE_RESERVED_SLOTS = {
  live: 8,
  invalidated: 3,
} as const satisfies Partial<Record<SharedStateEntryKind, number>>;
const DEFAULT_SHARED_STATE_LOCKED_CAP = 14;
const DEFAULT_NEWEST_STATE_CHANGE_RESERVED_SLOTS = 3;
const SHARED_STATE_SINGLE_ENTRY_FLOOR_TOKENS = 200;
const SHARED_STATE_TEXT_TRUNCATION_MARKER = " ... [text truncated]";
export const SHARED_STATE_COMPACT_INDEX_EXCERPT_CHAR_LIMIT = 80;
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

function renderSharedStateEntry(entry: SharedStateEntry): string {
  const owner = entry.owner_entity_id === null ? "owner=null" : `owner=${entry.owner_entity_id}`;
  const stateKey = `state_key=${sharedStateKeyBucket(entry.state_key)}`;
  const citations = `[citation: ${entry.provenance_stream_entry_ids.join(", ")}]`;
  const disclosureLabel = sharedStateEntryDisclosureLabel(entry);
  const disclosure =
    disclosureLabel.disclosureClass === "public"
      ? ""
      : ` ${renderMemoryDisclosureLabelForModel(disclosureLabel)}`;

  return [
    `- kind=${entry.kind} id=${entry.id} ${stateKey} ${owner} last_updated_at=${entry.last_updated_at}${disclosure} ${citations}`,
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
  lastUpdatedAt: number;
  activeCount: number;
  excerpt: string;
  disclosureLabel: MemoryDisclosureLabel;
  expanded: boolean;
};

function buildSharedStateCompactIndexRows(input: {
  activeEntries: readonly SharedStateEntry[];
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
      lastUpdatedAt: Math.max(...group.entries.map((entry) => entry.last_updated_at)),
      activeCount: group.entries.length,
      excerpt: sharedStateCompactExcerpt(latestEntry.text),
      disclosureLabel: combineMemoryDisclosureLabels(
        group.entries.map((entry) => sharedStateEntryDisclosureLabel(entry)),
      ),
      expanded: input.expandedBuckets.has(group.stateKey),
    };
  });
}

function renderSharedStateCompactIndexRows(rows: readonly SharedStateCompactIndexRow[]): string {
  const lines = rows.map((row) =>
    [
      `- ${row.stateKey}`,
      `kinds=${row.kinds.join(",")}`,
      `last_updated_at=${row.lastUpdatedAt}`,
      `active_count=${row.activeCount}`,
      row.disclosureLabel.disclosureClass === "public"
        ? null
        : renderMemoryDisclosureLabelForModel(row.disclosureLabel),
      `excerpt=${JSON.stringify(row.excerpt)}`,
      row.expanded ? "expanded" : "omitted",
    ]
      .filter((part): part is string => part !== null)
      .join(" | "),
  );

  return ["SharedStateArtifact compact active-key index:", ...lines].join("\n");
}

function renderSharedStateCompactIndex(input: {
  activeEntries: readonly SharedStateEntry[];
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

function renderSharedStateArtifactContent(input: {
  artifact: SharedStateArtifact;
  activeEntries: readonly SharedStateEntry[];
  entries: readonly SharedStateEntry[];
  omittedByKind: SharedStateKindCounts;
  renderedByKind: SharedStateKindCounts;
}): string {
  const omittedCount = Object.values(input.omittedByKind).reduce((sum, count) => sum + count, 0);
  const expandedBuckets = new Set(
    input.entries.map((entry) => sharedStateKeyBucket(entry.state_key)),
  );
  const omission =
    omittedCount <= 0
      ? null
      : [
          `SharedStateArtifact omitted: ${formatSharedStateKindCounts(input.omittedByKind)}.`,
          `Retained: ${formatSharedStateKindCounts(input.renderedByKind)}.`,
        ].join(" ");

  return [
    "## 0. Shared Audience State",
    "SharedStateArtifact: durable shared state for this audience. It is a compact structural anchor, not a policy source.",
    `audience_entity_id=${input.artifact.audience_entity_id}`,
    `record_version=${input.artifact.record_version}`,
    renderSharedStateCompactIndex({
      activeEntries: input.activeEntries,
      expandedBuckets,
    }),
    ...entriesGroupedByStateKey(input.entries).flatMap((group) => [
      `state_key_bucket=${group.stateKey}`,
      ...group.entries.map(renderSharedStateEntry),
    ]),
    omission,
  ]
    .filter((part): part is string => part !== null)
    .join("\n");
}

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
    renderSharedStateCompactIndex({
      activeEntries: input.activeEntries,
      expandedBuckets: new Set(),
    }),
    `SharedStateArtifact omitted: ${formatSharedStateKindCounts(
      input.omittedByKind,
    )}. Reason: ${input.reason}.`,
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
