import { filterActiveStreamEntries } from "./turn-status.js";
import type { StreamReader } from "./stream-reader.js";
import type { StreamEntry } from "./types.js";

export type TranscriptStreamEntryKind = Extract<
  StreamEntry["kind"],
  "user_msg" | "agent_msg" | "agent_suppressed"
>;

export type TranscriptStreamEntry = StreamEntry & {
  kind: TranscriptStreamEntryKind;
};

export function isTranscriptStreamEntry(entry: StreamEntry): entry is TranscriptStreamEntry {
  return (
    entry.kind === "user_msg" || entry.kind === "agent_msg" || entry.kind === "agent_suppressed"
  );
}

export async function loadSessionStreamEntries(reader: StreamReader): Promise<StreamEntry[]> {
  const entries: StreamEntry[] = [];

  for await (const entry of reader.iterate()) {
    entries.push(entry);
  }

  return entries;
}

export function activeSessionTranscriptEntries(
  entries: readonly StreamEntry[],
): TranscriptStreamEntry[] {
  return filterActiveStreamEntries(entries).filter(isTranscriptStreamEntry);
}

export async function loadActiveSessionTranscriptEntries(
  reader: StreamReader,
): Promise<TranscriptStreamEntry[]> {
  return activeSessionTranscriptEntries(await loadSessionStreamEntries(reader));
}
