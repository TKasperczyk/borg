import { closeSync, existsSync, fstatSync, openSync, readSync } from "node:fs";

import type { StreamEntryIndexRepository } from "./entry-index.js";
import { getSessionStreamPath } from "./path.js";
import { streamEntryReadSchema, type SessionId, type StreamEntry } from "./types.js";
import type { StreamReader } from "./stream-reader.js";
import type { StreamEntryId } from "../util/ids.js";

const INDEXED_STREAM_ENTRY_READ_CHUNK_SIZE_BYTES = 64 * 1024;
const NEWLINE_BYTE = 0x0a;

export type HydrateStreamEntriesByIdInput = {
  dataDir: string;
  sessionId: SessionId;
  streamEntryIds: readonly StreamEntryId[];
  entryIndex?: Pick<StreamEntryIndexRepository, "lookupMany">;
  createStreamReader?: (sessionId: SessionId) => StreamReader;
  fallbackSessionIds?: readonly SessionId[];
  budgetMs?: number;
  // Active state is an index fact. Requiring it also disables the reader fallback,
  // which cannot establish whether a stream-only entry has been quarantined.
  activeOnly?: boolean;
};

export function readStreamEntryAtOffset(input: {
  dataDir: string;
  sessionId: SessionId;
  byteOffset: number;
}): StreamEntry | null {
  const streamPath = getSessionStreamPath(input.dataDir, input.sessionId);

  if (!existsSync(streamPath)) {
    return null;
  }

  const fileDescriptor = openSync(streamPath, "r");
  const chunks: Buffer[] = [];

  try {
    const fileSize = fstatSync(fileDescriptor).size;

    if (input.byteOffset < 0 || input.byteOffset >= fileSize) {
      return null;
    }

    let position = input.byteOffset;

    while (position < fileSize) {
      const chunkSize = Math.min(INDEXED_STREAM_ENTRY_READ_CHUNK_SIZE_BYTES, fileSize - position);
      const chunk = Buffer.allocUnsafe(chunkSize);
      const bytesRead = readSync(fileDescriptor, chunk, 0, chunkSize, position);

      if (bytesRead <= 0) {
        break;
      }

      const chunkBytes = bytesRead === chunkSize ? chunk : chunk.subarray(0, bytesRead);
      const newlineIndex = chunkBytes.indexOf(NEWLINE_BYTE);

      if (newlineIndex === -1) {
        chunks.push(Buffer.from(chunkBytes));
        position += bytesRead;
        continue;
      }

      chunks.push(Buffer.from(chunkBytes.subarray(0, newlineIndex)));
      break;
    }
  } finally {
    closeSync(fileDescriptor);
  }

  if (chunks.length === 0) {
    return null;
  }

  const line = Buffer.concat(chunks).toString("utf8");

  if (line.trim() === "") {
    return null;
  }

  try {
    const raw = JSON.parse(line) as unknown;
    const parsed = streamEntryReadSchema.safeParse(raw);
    return parsed.success ? parsed.data : null;
  } catch {
    return null;
  }
}

export async function hydrateStreamEntriesById(
  input: HydrateStreamEntriesByIdInput,
): Promise<Map<StreamEntryId, StreamEntry>> {
  const deadlineAt =
    input.budgetMs === undefined ? null : Date.now() + Math.max(0, Math.floor(input.budgetMs));
  const hasBudget = () => deadlineAt === null || Date.now() < deadlineAt;
  const uniqueIds = [...new Set(input.streamEntryIds)];
  const pendingIds = new Set<StreamEntryId>(uniqueIds);
  const entries = new Map<StreamEntryId, StreamEntry>();

  if (pendingIds.size === 0 || !hasBudget()) {
    return entries;
  }

  if (input.entryIndex !== undefined) {
    const indexedEntries = input.entryIndex.lookupMany([...pendingIds]);

    for (const streamEntryId of [...pendingIds]) {
      if (!hasBudget()) {
        return entries;
      }
      const record = indexedEntries.get(streamEntryId);

      if (record === undefined) {
        continue;
      }

      if (input.activeOnly === true && !record.active) {
        pendingIds.delete(streamEntryId);
        continue;
      }

      const entry = readStreamEntryAtOffset({
        dataDir: input.dataDir,
        sessionId: record.session_id,
        byteOffset: record.byte_offset,
      });

      if (!hasBudget()) {
        return entries;
      }

      if (entry?.id !== streamEntryId) {
        continue;
      }

      entries.set(streamEntryId, entry);
      pendingIds.delete(streamEntryId);
    }
  }

  if (input.activeOnly === true) {
    return entries;
  }

  if (pendingIds.size === 0 || input.createStreamReader === undefined) {
    return entries;
  }

  const fallbackSessionIds = input.fallbackSessionIds ?? [input.sessionId];

  for (const sessionId of fallbackSessionIds) {
    for await (const entry of input.createStreamReader(sessionId).iterate()) {
      if (!hasBudget()) {
        return entries;
      }
      if (!pendingIds.has(entry.id)) {
        continue;
      }

      entries.set(entry.id, entry);
      pendingIds.delete(entry.id);

      if (pendingIds.size === 0) {
        return entries;
      }
    }
  }

  return entries;
}
