import { estimatePromptTokens } from "../../util/token-estimate.js";
import {
  SHARED_STATE_ENTRY_KINDS,
  type SharedStateArtifact,
  type SharedStateEntry,
  type SharedStateEntryKind,
} from "../../memory/decision-artifacts/index.js";
import { normalizePositiveInteger } from "../evidence-ledger/budget.js";
import {
  activeSharedStateArtifactEntries,
  countSharedStateArtifactEntriesByKind,
  emptySharedStateKindCounts,
  selectSharedStateArtifactEntriesForRenderWithSummary,
  subtractSharedStateKindCounts,
  tokenDropIndex,
  type SharedStateKindCounts,
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
  pending: 3,
} as const satisfies Partial<Record<SharedStateEntryKind, number>>;
const DEFAULT_SHARED_STATE_LOCKED_CAP = 14;
const DEFAULT_NEWEST_STATE_CHANGE_RESERVED_SLOTS = 3;
const SHARED_STATE_SINGLE_ENTRY_FLOOR_TOKENS = 200;
const SHARED_STATE_TEXT_TRUNCATION_MARKER = " ... [text truncated]";

export type SharedStateArtifactRenderSummary = {
  totalEntryCount: number;
  activeEntryCount: number;
  renderedEntryCount: number;
  omittedEntryCount: number;
  estimatedTokens: number;
  newestReservedEntryCount: number;
  renderedByKind: SharedStateKindCounts;
  omittedByKind: SharedStateKindCounts;
  activeEntriesByKey: Record<string, number>;
  topKeysByEntryCount: Record<string, number>;
};

export type SharedStateRenderOptions = {
  maxEntries?: number;
  maxTokens?: number;
  reservedSlots?: Partial<Record<SharedStateEntryKind, number>>;
  lockedMaxEntries?: number;
  newestStateChangeReservedSlots?: number;
};

function sharedStateRenderOptions(
  options: SharedStateRenderOptions = {},
): Required<SharedStateRenderOptions> {
  return {
    maxEntries: normalizePositiveInteger(options.maxEntries, DEFAULT_SHARED_STATE_MAX_ENTRIES),
    maxTokens: normalizePositiveInteger(options.maxTokens, DEFAULT_SHARED_STATE_MAX_TOKENS),
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

  return [
    `- kind=${entry.kind} id=${entry.id} ${stateKey} ${owner} last_updated_at=${entry.last_updated_at} ${citations}`,
    `  text: ${entry.text}`,
  ].join("\n");
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

function renderSharedStateArtifactContent(input: {
  artifact: SharedStateArtifact;
  entries: readonly SharedStateEntry[];
  omittedByKind: SharedStateKindCounts;
  renderedByKind: SharedStateKindCounts;
}): string {
  const omittedCount = Object.values(input.omittedByKind).reduce((sum, count) => sum + count, 0);
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
  omittedByKind: SharedStateKindCounts;
  reason: string;
}): string {
  return [
    "## 0. Shared Audience State",
    "SharedStateArtifact: durable shared state for this audience. It is a compact structural anchor, not a policy source.",
    `audience_entity_id=${input.artifact.audience_entity_id}`,
    `record_version=${input.artifact.record_version}`,
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
        omittedByKind: countSharedStateArtifactEntriesByKind(input.activeEntries),
        reason: "artifact entry too large to render",
      }),
      renderedEntryCount: 0,
      omittedEntryCount: input.activeEntries.length,
    };
  }

  const content = renderSharedStateArtifactContent({
    artifact: input.artifact,
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
      omittedByKind: countSharedStateArtifactEntriesByKind(input.activeEntries),
      reason: "artifact entry too large to render",
    }),
    renderedEntryCount: 0,
    omittedEntryCount: input.activeEntries.length,
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
        omittedEntryCount: 0,
        estimatedTokens: 0,
        newestReservedEntryCount: 0,
        renderedByKind: emptySharedStateKindCounts(),
        omittedByKind: emptySharedStateKindCounts(),
        activeEntriesByKey: {},
        topKeysByEntryCount: {},
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
  });
  const newestReservedIds = selection.newestReservedIds;
  let entries = selection.entries;
  let counts = sharedStateRenderedCounts({
    activeEntries,
    renderedEntries: entries,
  });
  let content = renderSharedStateArtifactContent({
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
    counts = sharedStateRenderedCounts({
      activeEntries,
      renderedEntries: entries,
    });
    content = renderSharedStateArtifactContent({
      artifact: input.artifact,
      entries,
      omittedByKind: counts.omittedByKind,
      renderedByKind: counts.renderedByKind,
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

  return {
    content,
    summary: {
      totalEntryCount: input.artifact.entries.length,
      activeEntryCount: activeEntries.length,
      renderedEntryCount: entries.length,
      omittedEntryCount: counts.omittedEntryCount,
      estimatedTokens: estimatePromptTokens(content),
      newestReservedEntryCount: entries.filter((entry) => newestReservedIds.has(entry.id)).length,
      renderedByKind: counts.renderedByKind,
      omittedByKind: counts.omittedByKind,
      activeEntriesByKey: countSharedStateEntriesByKey(activeEntries),
      topKeysByEntryCount: topSharedStateEntryKeysByCount(
        countSharedStateEntriesByKey(activeEntries),
        5,
      ),
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
      omittedEntryCount: 0,
      estimatedTokens: 0,
      newestReservedEntryCount: 0,
      renderedByKind: emptySharedStateKindCounts(),
      omittedByKind: emptySharedStateKindCounts(),
      activeEntriesByKey: {},
      topKeysByEntryCount: {},
    };
  }

  return cappedSharedStateArtifactRender({
    artifact,
    options,
  }).summary;
}
