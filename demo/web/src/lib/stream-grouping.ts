import type { StreamEntry } from "../api/types";

export const UNCLAIMED_STREAM_GROUP_ID = "__unclaimed_maintenance__";
export const UNCLAIMED_STREAM_GROUP_LABEL = "unclaimed / maintenance";
export const UNCLAIMED_STREAM_GROUP_ID_PREFIX = "unclaimed:";

export type StreamGroupStatus = "active" | "aborted" | "maintenance" | "mixed";

export type StreamTurnGroup = {
  id: string;
  turnId: string | null;
  label: string;
  entries: StreamEntry[];
  entryCount: number;
  startTimestamp: number;
  endTimestamp: number;
  maxEntryIndex?: number;
  status: StreamGroupStatus;
};

export type StreamStructuralFilterId =
  | "aborted"
  | "active"
  | "hasAttachment"
  | "hasTurnId"
  | "hasSourceMessageKey"
  | "compressed";

export type StreamStructuralFilterState = Partial<Record<StreamStructuralFilterId, boolean>>;

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}

export function streamEntryAttachmentId(entry: StreamEntry): string | undefined {
  if (!isRecord(entry.content)) {
    return undefined;
  }
  const value = entry.content.attachment_id;
  return typeof value === "string" && value.length > 0 ? value : undefined;
}

export function hasStreamAttachment(entry: StreamEntry): boolean {
  return entry.kind === "user_image_attachment" || streamEntryAttachmentId(entry) !== undefined;
}

export function hasTurnId(entry: StreamEntry): boolean {
  return entry.turn_id !== undefined;
}

export function hasSourceMessageKey(entry: StreamEntry): boolean {
  return entry.source_message_key !== undefined;
}

export function isCompressed(entry: StreamEntry): boolean {
  return entry.compressed === true;
}

export function isAbortedTurnEntry(entry: StreamEntry): boolean {
  return entry.turn_status === "aborted";
}

export function isActiveTurnEntry(entry: StreamEntry): boolean {
  return entry.turn_status !== "aborted";
}

export const STREAM_STRUCTURAL_FILTERS: Record<
  StreamStructuralFilterId,
  (entry: StreamEntry) => boolean
> = {
  aborted: isAbortedTurnEntry,
  active: isActiveTurnEntry,
  hasAttachment: hasStreamAttachment,
  hasTurnId,
  hasSourceMessageKey,
  compressed: isCompressed,
};

export function matchesStreamStructuralFilters(
  entry: StreamEntry,
  filters: StreamStructuralFilterState,
): boolean {
  return (Object.entries(filters) as [StreamStructuralFilterId, boolean | undefined][]).every(
    ([filterId, enabled]) => enabled !== true || STREAM_STRUCTURAL_FILTERS[filterId](entry),
  );
}

export function applyStreamStructuralFilters(
  entries: readonly StreamEntry[],
  filters: StreamStructuralFilterState,
): StreamEntry[] {
  return entries.filter((entry) => matchesStreamStructuralFilters(entry, filters));
}

export function compareStreamEntriesForTurnGroup(left: StreamEntry, right: StreamEntry): number {
  if (
    left.entry_index !== undefined &&
    right.entry_index !== undefined &&
    left.entry_index !== right.entry_index
  ) {
    return right.entry_index - left.entry_index;
  }

  if (left.timestamp !== right.timestamp) {
    return right.timestamp - left.timestamp;
  }

  return right.id.localeCompare(left.id);
}

export function mergeStreamEntriesForTurnGrouping(
  current: readonly StreamEntry[],
  incoming: readonly StreamEntry[],
): StreamEntry[] {
  const byId = new Map(current.map((entry) => [entry.id, entry]));

  for (const entry of incoming) {
    byId.set(entry.id, entry);
  }

  return [...byId.values()].sort(compareStreamEntriesForTurnGroup);
}

export function unclaimedStreamGroupId(entryId: string): string {
  return `${UNCLAIMED_STREAM_GROUP_ID_PREFIX}${entryId}`;
}

function groupStatus(turnId: string | null, entries: readonly StreamEntry[]): StreamGroupStatus {
  if (turnId === null) {
    return "maintenance";
  }

  const abortedCount = entries.filter(isAbortedTurnEntry).length;
  if (abortedCount === 0) {
    return "active";
  }
  if (abortedCount === entries.length) {
    return "aborted";
  }
  return "mixed";
}

function maxEntryIndex(entries: readonly StreamEntry[]): number | undefined {
  const indexes = entries.flatMap((entry) =>
    entry.entry_index === undefined ? [] : [entry.entry_index],
  );
  return indexes.length === 0 ? undefined : Math.max(...indexes);
}

function compareGroupsNewestFirst(left: StreamTurnGroup, right: StreamTurnGroup): number {
  if (
    left.maxEntryIndex !== undefined &&
    right.maxEntryIndex !== undefined &&
    left.maxEntryIndex !== right.maxEntryIndex
  ) {
    return right.maxEntryIndex - left.maxEntryIndex;
  }

  if (left.endTimestamp !== right.endTimestamp) {
    return right.endTimestamp - left.endTimestamp;
  }

  if (left.maxEntryIndex !== undefined && right.maxEntryIndex !== undefined) {
    return right.maxEntryIndex - left.maxEntryIndex;
  }

  return left.id.localeCompare(right.id);
}

export function groupStreamEntriesByTurn(entries: readonly StreamEntry[]): StreamTurnGroup[] {
  const groups = new Map<string, { turnId: string | null; entries: StreamEntry[] }>();

  for (const entry of entries) {
    const turnId = entry.turn_id ?? null;
    const id = turnId ?? unclaimedStreamGroupId(entry.id);
    const group = groups.get(id);
    if (group === undefined) {
      groups.set(id, { turnId, entries: [entry] });
    } else {
      group.entries.push(entry);
    }
  }

  return [...groups.entries()]
    .map(([id, group]): StreamTurnGroup => {
      const sortedEntries = [...group.entries].sort(compareStreamEntriesForTurnGroup);
      const timestamps = sortedEntries.map((entry) => entry.timestamp);
      const turnId = group.turnId;
      return {
        id,
        turnId,
        label: turnId ?? UNCLAIMED_STREAM_GROUP_LABEL,
        entries: sortedEntries,
        entryCount: sortedEntries.length,
        startTimestamp: Math.min(...timestamps),
        endTimestamp: Math.max(...timestamps),
        maxEntryIndex: maxEntryIndex(sortedEntries),
        status: groupStatus(turnId, sortedEntries),
      };
    })
    .sort(compareGroupsNewestFirst);
}
