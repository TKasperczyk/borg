/* Citation stream lookup helpers for retrieval results. */
import { closeSync, existsSync, fstatSync, openSync, readSync, readdirSync } from "node:fs";
import { basename } from "node:path";

import {
  getSessionStreamPath,
  getStreamDirectory,
  StreamReader,
  streamEntrySchema,
  type StreamEntry,
  type StreamEntryIndexRepository,
} from "../stream/index.js";
import {
  DEFAULT_SESSION_ID,
  parseSessionId,
  type SessionId,
  type StreamEntryId,
} from "../util/ids.js";

const INDEXED_CITATION_READ_CHUNK_SIZE_BYTES = 64 * 1024;
const NEWLINE_BYTE = 0x0a;

export type CitationResolverOptions = {
  dataDir: string;
  entryIndex?: StreamEntryIndexRepository;
};

function parseSessionIdFromFilename(filename: string): SessionId | null {
  if (!filename.endsWith(".jsonl")) {
    return null;
  }

  const sessionName = basename(filename, ".jsonl");

  try {
    return parseSessionId(sessionName);
  } catch {
    return null;
  }
}

export class CitationResolver {
  constructor(private readonly options: CitationResolverOptions) {}

  listSessionIds(): SessionId[] {
    const streamDir = getStreamDirectory(this.options.dataDir);

    if (!existsSync(streamDir)) {
      return [DEFAULT_SESSION_ID];
    }

    const sessionIds = readdirSync(streamDir)
      .map((filename) => parseSessionIdFromFilename(filename))
      .filter((value): value is SessionId => value !== null);

    return sessionIds.length === 0 ? [DEFAULT_SESSION_ID] : sessionIds;
  }

  async resolveCitationEntries(
    sourceStreamIds: readonly StreamEntryId[],
  ): Promise<Map<string, StreamEntry>> {
    const entries = new Map<string, StreamEntry>();
    const pendingIds = new Set<string>(sourceStreamIds);

    if (pendingIds.size === 0) {
      return entries;
    }

    const sessionIds = this.listSessionIds();
    const existingSessionIds = new Set(sessionIds);

    if (this.options.entryIndex !== undefined) {
      const indexedEntries = this.options.entryIndex.lookupMany([...pendingIds]);

      for (const entryId of [...pendingIds]) {
        const record = indexedEntries.get(entryId);

        if (record === undefined) {
          console.warn("Citation index miss; falling back to stream scan.", {
            entryId,
          });
          continue;
        }

        if (!existingSessionIds.has(record.session_id)) {
          console.warn(
            "Citation index referenced a missing session; falling back to stream scan.",
            {
              entryId,
              sessionId: record.session_id,
            },
          );
          continue;
        }

        const entry = this.readCitationEntryAtOffset(record.session_id, record.byte_offset);

        if (entry === null) {
          console.warn("Citation index read failed; falling back to stream scan.", {
            entryId,
            sessionId: record.session_id,
            byteOffset: record.byte_offset,
          });
          continue;
        }

        if (entry.id !== entryId) {
          console.warn(
            "Citation index returned a mismatched stream entry; falling back to stream scan.",
            {
              entryId,
              foundEntryId: entry.id,
              sessionId: record.session_id,
              byteOffset: record.byte_offset,
            },
          );
          continue;
        }

        entries.set(entryId, entry);
        pendingIds.delete(entryId);
      }
    }

    if (pendingIds.size === 0) {
      return entries;
    }

    const scannedEntries = await this.scanCitationEntries(sessionIds, [...pendingIds]);

    for (const [entryId, entry] of scannedEntries) {
      entries.set(entryId, entry);
    }

    return entries;
  }

  resolveCitationChainFromMap(
    sourceStreamIds: readonly StreamEntryId[],
    entries: ReadonlyMap<string, StreamEntry>,
  ): StreamEntry[] {
    return sourceStreamIds
      .map((sourceId) => entries.get(sourceId))
      .filter((entry): entry is StreamEntry => entry !== undefined);
  }

  private async scanCitationEntries(
    sessionIds: readonly SessionId[],
    sourceStreamIds: readonly string[],
  ): Promise<Map<string, StreamEntry>> {
    const entries = new Map<string, StreamEntry>();
    const pendingIds = new Set<string>(sourceStreamIds);

    if (pendingIds.size === 0) {
      return entries;
    }

    for (const sessionId of sessionIds) {
      const reader = new StreamReader({
        dataDir: this.options.dataDir,
        sessionId,
      });

      for await (const entry of reader.iterate()) {
        if (!pendingIds.has(entry.id)) {
          continue;
        }

        entries.set(entry.id, entry);
        pendingIds.delete(entry.id);

        if (pendingIds.size === 0) {
          break;
        }
      }

      if (pendingIds.size === 0) {
        break;
      }
    }

    return entries;
  }

  private readCitationEntryAtOffset(sessionId: SessionId, byteOffset: number): StreamEntry | null {
    const streamPath = getSessionStreamPath(this.options.dataDir, sessionId);

    if (!existsSync(streamPath)) {
      return null;
    }

    const fileDescriptor = openSync(streamPath, "r");
    const chunks: Buffer[] = [];

    try {
      const fileSize = fstatSync(fileDescriptor).size;

      if (byteOffset < 0 || byteOffset >= fileSize) {
        return null;
      }

      let position = byteOffset;

      while (position < fileSize) {
        const chunkSize = Math.min(INDEXED_CITATION_READ_CHUNK_SIZE_BYTES, fileSize - position);
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
      const parsed = streamEntrySchema.safeParse(raw);
      return parsed.success ? parsed.data : null;
    } catch {
      return null;
    }
  }
}
