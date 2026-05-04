import type { StreamEntry } from "./types.js";

export const ABORTED_TURN_EVENT = "aborted_turn";

export type AbortedTurnRefs = {
  turnIds: ReadonlySet<string>;
  streamEntryIds: ReadonlySet<string>;
};

function isRecord(value: unknown): value is Record<string, unknown> {
  return value !== null && typeof value === "object" && !Array.isArray(value);
}

function abortedEntryIds(content: Record<string, unknown>): string[] {
  const value = content.aborted_stream_entry_ids;

  if (!Array.isArray(value)) {
    return [];
  }

  return value.filter((item): item is string => typeof item === "string" && item.length > 0);
}

export function isAbortedTurnMarker(entry: StreamEntry): boolean {
  return (
    entry.kind === "internal_event" &&
    isRecord(entry.content) &&
    entry.content.event === ABORTED_TURN_EVENT
  );
}

export function collectAbortedTurnRefs(entries: readonly StreamEntry[]): AbortedTurnRefs {
  const turnIds = new Set<string>();
  const streamEntryIds = new Set<string>();

  for (const entry of entries) {
    if (!isAbortedTurnMarker(entry) || !isRecord(entry.content)) {
      continue;
    }

    const turnId = entry.content.turn_id;

    if (typeof turnId === "string" && turnId.length > 0) {
      turnIds.add(turnId);
    }

    for (const streamEntryId of abortedEntryIds(entry.content)) {
      streamEntryIds.add(streamEntryId);
    }
  }

  return {
    turnIds,
    streamEntryIds,
  };
}

export function streamEntryIsActive(entry: StreamEntry, refs: AbortedTurnRefs): boolean {
  if (isAbortedTurnMarker(entry)) {
    return false;
  }

  if (entry.turn_status === "aborted") {
    return false;
  }

  if (entry.turn_id !== undefined && refs.turnIds.has(entry.turn_id)) {
    return false;
  }

  return !refs.streamEntryIds.has(entry.id);
}

export function filterActiveStreamEntries(entries: readonly StreamEntry[]): StreamEntry[] {
  const refs = collectAbortedTurnRefs(entries);
  return entries.filter((entry) => streamEntryIsActive(entry, refs));
}
