import type { StreamEntry } from "./types.js";
import { isPlainRecord } from "../util/guards.js";
import { frameAnomalyAnnotationFromEntry } from "./frame-anomaly.js";

export const ABORTED_TURN_EVENT = "aborted_turn";
export const QUARANTINED_USER_ENTRY_EVENT = "quarantined_user_entry";

export type InactiveStreamEntryRefs = {
  turnIds: ReadonlySet<string>;
  streamEntryIds: ReadonlySet<string>;
};

function abortedEntryIds(content: Record<string, unknown>): string[] {
  const value = content.aborted_stream_entry_ids;

  if (!Array.isArray(value)) {
    return [];
  }

  return value.filter((item): item is string => typeof item === "string" && item.length > 0);
}

function citedStreamEntryIds(content: Record<string, unknown>): string[] {
  const value = content.cited_stream_entry_ids;

  if (!Array.isArray(value)) {
    return [];
  }

  return value.filter((item): item is string => typeof item === "string" && item.length > 0);
}

export function quarantinedUserEntrySourceIds(entry: StreamEntry): string[] {
  if (!isQuarantinedUserEntryMarker(entry) || !isPlainRecord(entry.content)) return [];
  const scalar = entry.content.source_stream_entry_id;
  return [
    ...new Set([
      ...(typeof scalar === "string" && scalar.length > 0 ? [scalar] : []),
      ...citedStreamEntryIds(entry.content),
    ]),
  ];
}

// The aborted-turn marker is the only record of *why* a turn died -- its content carries a
// `reason` string (in practice the provider error, e.g. "LLMError: Failed to complete Anthropic
// request") that exists nowhere else. It is also, by the predicate below, unconditionally
// inactive: `streamEntryIsActive` rejects the marker itself, every entry sharing its turn_id, and
// every entry it names. Measured on the live demo store (2026-08-18): 2137 aborted-turn markers
// across seven sessions, 2137 of them inactive -- the set has never had a member the entity could
// read. So a turn killed by a provider error is silent in a strictly stronger sense than a turn
// killed by a post-generation guard: the guard path writes an `agent_suppressed` entry that stays
// active and renders as "[system: prior turn suppressed -- reason: ...]" (recency/compiler.ts),
// while the abort path writes its reason to the one row that is defined as unreadable. Filtering
// the marker is correct -- it is what keeps half-generated aborted content out of cognition -- but
// it means the reason is discarded with it, and the entity experiences the outage as a turn that
// simply never happened. Surfacing the reason without un-filtering the turn would need a separate
// pass in the recency compiler, not a change here.
export function isAbortedTurnMarker(entry: StreamEntry): boolean {
  return (
    entry.kind === "internal_event" &&
    isPlainRecord(entry.content) &&
    entry.content.event === ABORTED_TURN_EVENT
  );
}

export function isQuarantinedUserEntryMarker(entry: StreamEntry): boolean {
  return (
    entry.kind === "internal_event" &&
    isPlainRecord(entry.content) &&
    entry.content.event === QUARANTINED_USER_ENTRY_EVENT
  );
}

// Exact source/marker pairs; restoring one citation never restores its siblings.
export function collectQuarantineRestorations(
  entries: readonly StreamEntry[],
): ReadonlyMap<string, ReadonlySet<string>> {
  const restorations = new Map<string, Set<string>>();
  for (const entry of entries) {
    const event = frameAnomalyAnnotationFromEntry(entry);
    for (const annotation of [
      ...(entry.frame_anomalies ?? []),
      ...(event === null ? [] : [event]),
    ]) {
      if (annotation.restores_quarantine_marker_ids.length === 0) continue;
      const markers = restorations.get(annotation.source_stream_entry_id) ?? new Set<string>();
      for (const id of annotation.restores_quarantine_marker_ids) markers.add(id);
      restorations.set(annotation.source_stream_entry_id, markers);
    }
  }

  return restorations;
}

export function collectInactiveStreamEntryRefs(
  entries: readonly StreamEntry[],
): InactiveStreamEntryRefs {
  const turnIds = new Set<string>();
  const streamEntryIds = new Set<string>();
  const restorations = collectQuarantineRestorations(entries);

  for (const entry of entries) {
    if (!isPlainRecord(entry.content)) {
      continue;
    }

    if (isAbortedTurnMarker(entry)) {
      const turnId = entry.content.turn_id;

      if (typeof turnId === "string" && turnId.length > 0) {
        turnIds.add(turnId);
      }

      for (const streamEntryId of abortedEntryIds(entry.content)) {
        streamEntryIds.add(streamEntryId);
      }

      continue;
    }

    // Non-peer quarantine still strikes its cited entries. An explicit restoration
    // cancels only this marker's contribution, never an abort or another strike.
    if (isQuarantinedUserEntryMarker(entry)) {
      for (const streamEntryId of quarantinedUserEntrySourceIds(entry)) {
        if (!restorations.get(streamEntryId)?.has(entry.id)) streamEntryIds.add(streamEntryId);
      }
    }
  }

  return {
    turnIds,
    streamEntryIds,
  };
}

export function streamEntryIsActive(entry: StreamEntry, refs: InactiveStreamEntryRefs): boolean {
  if (isAbortedTurnMarker(entry) || isQuarantinedUserEntryMarker(entry)) {
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
  const refs = collectInactiveStreamEntryRefs(entries);
  return entries.filter((entry) => streamEntryIsActive(entry, refs));
}
