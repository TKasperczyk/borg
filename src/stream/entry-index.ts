import { closeSync, existsSync, fstatSync, openSync, readSync } from "node:fs";

import { type Migration, type SqliteDatabase } from "../storage/sqlite/index.js";

import { getSessionStreamPath } from "./path.js";
import { type SessionId, type StreamEntry, streamEntrySchema } from "./types.js";

type LoggerLike = Pick<Console, "error">;

const FORWARD_SCAN_CHUNK_SIZE_BYTES = 64 * 1024;
const NEWLINE_BYTE = 0x0a;

export const streamEntryIndexMigrations: Migration[] = [
  {
    id: 201,
    name: "create-stream-entry-index",
    up: `
      CREATE TABLE IF NOT EXISTS stream_entry_index (
        entry_id TEXT PRIMARY KEY,
        session_id TEXT NOT NULL,
        byte_offset INTEGER NOT NULL,
        timestamp INTEGER NOT NULL
      );
      CREATE INDEX IF NOT EXISTS idx_stream_entry_session
      ON stream_entry_index(session_id)
    `,
  },
];

export type StreamEntryIndexRecord = {
  entry_id: string;
  session_id: SessionId;
  byte_offset: number;
  timestamp: number;
};

type StreamEntryIndexRow = {
  entry_id: string;
  session_id: string;
  byte_offset: number;
  timestamp: number;
};

type SessionEntryCountRow = {
  entry_count: number;
};

export type StreamEntryIndexRepositoryOptions = {
  db: SqliteDatabase;
  dataDir: string;
  logger?: LoggerLike;
};

function parseIndexedStreamLine(
  line: string,
  streamPath: string,
  logger: LoggerLike,
): StreamEntry | null {
  if (line.trim() === "") {
    return null;
  }

  try {
    const raw = JSON.parse(line) as unknown;
    const parsed = streamEntrySchema.safeParse(raw);

    if (!parsed.success) {
      logger.error(`Skipping invalid stream line in ${streamPath}`);
      return null;
    }

    return parsed.data;
  } catch (error) {
    logger.error(`Skipping unreadable stream line in ${streamPath}`);
    logger.error(error instanceof Error ? error.message : String(error));
    return null;
  }
}

function recordFromRow(row: StreamEntryIndexRow): StreamEntryIndexRecord {
  return {
    entry_id: row.entry_id,
    session_id: row.session_id as SessionId,
    byte_offset: row.byte_offset,
    timestamp: row.timestamp,
  };
}

function forwardLineToString(
  carryChunks: readonly Buffer[],
  carryLength: number,
  lineSegment: Buffer,
): string {
  if (carryChunks.length === 0) {
    return lineSegment.toString("utf8");
  }

  if (lineSegment.length === 0) {
    return Buffer.concat(carryChunks, carryLength).toString("utf8");
  }

  return Buffer.concat([...carryChunks, lineSegment], carryLength + lineSegment.length).toString(
    "utf8",
  );
}

function scanForwardStreamEntries(
  fileDescriptor: number,
  fileSize: number,
  streamPath: string,
  logger: LoggerLike,
  onEntry: (entry: StreamEntry, byteOffset: number) => void,
): number {
  let position = 0;
  const carryChunks: Buffer[] = [];
  let carryLength = 0;
  let currentLineOffset = 0;
  let scannedEntries = 0;

  while (position < fileSize) {
    const chunkSize = Math.min(FORWARD_SCAN_CHUNK_SIZE_BYTES, fileSize - position);
    const chunk = Buffer.allocUnsafe(chunkSize);
    const bytesRead = readSync(fileDescriptor, chunk, 0, chunkSize, position);

    if (bytesRead <= 0) {
      break;
    }

    const chunkBytes = bytesRead === chunkSize ? chunk : chunk.subarray(0, bytesRead);
    let lineStart = 0;

    if (carryLength === 0) {
      currentLineOffset = position;
    }

    for (let index = 0; index < chunkBytes.length; index += 1) {
      if (chunkBytes[index] !== NEWLINE_BYTE) {
        continue;
      }

      const entry = parseIndexedStreamLine(
        forwardLineToString(carryChunks, carryLength, chunkBytes.subarray(lineStart, index)),
        streamPath,
        logger,
      );

      if (entry !== null) {
        scannedEntries += 1;
        onEntry(entry, currentLineOffset);
      }

      carryChunks.length = 0;
      carryLength = 0;
      lineStart = index + 1;
      currentLineOffset = position + lineStart;
    }

    if (lineStart < chunkBytes.length) {
      if (carryLength === 0) {
        currentLineOffset = position + lineStart;
      }

      const remainder = chunkBytes.subarray(lineStart);
      carryChunks.push(remainder);
      carryLength += remainder.length;
    }

    position += bytesRead;
  }

  if (carryLength > 0) {
    const entry = parseIndexedStreamLine(
      Buffer.concat(carryChunks, carryLength).toString("utf8"),
      streamPath,
      logger,
    );

    if (entry !== null) {
      scannedEntries += 1;
      onEntry(entry, currentLineOffset);
    }
  }

  return scannedEntries;
}

export class StreamEntryIndexRepository {
  private readonly db: SqliteDatabase;
  private readonly dataDir: string;
  private readonly logger: LoggerLike;

  constructor(options: StreamEntryIndexRepositoryOptions) {
    this.db = options.db;
    this.dataDir = options.dataDir;
    this.logger = options.logger ?? console;
  }

  record(entryId: string, sessionId: SessionId, byteOffset: number, timestamp: number): void {
    this.db
      .prepare(
        `INSERT INTO stream_entry_index (entry_id, session_id, byte_offset, timestamp)
         VALUES (?, ?, ?, ?)
         ON CONFLICT (entry_id) DO UPDATE SET
           session_id = excluded.session_id,
           byte_offset = excluded.byte_offset,
           timestamp = excluded.timestamp`,
      )
      .run(entryId, sessionId, byteOffset, timestamp);
  }

  lookup(entryId: string): StreamEntryIndexRecord | null {
    const row = this.db
      .prepare(
        `SELECT entry_id, session_id, byte_offset, timestamp
         FROM stream_entry_index
         WHERE entry_id = ?`,
      )
      .get(entryId) as StreamEntryIndexRow | undefined;

    return row === undefined ? null : recordFromRow(row);
  }

  lookupMany(entryIds: readonly string[]): Map<string, StreamEntryIndexRecord> {
    const uniqueIds = [...new Set(entryIds)];

    if (uniqueIds.length === 0) {
      return new Map();
    }

    const rows = this.db
      .prepare(
        `SELECT entry_id, session_id, byte_offset, timestamp
         FROM stream_entry_index
         WHERE entry_id IN (${uniqueIds.map(() => "?").join(", ")})`,
      )
      .all(...uniqueIds) as StreamEntryIndexRow[];

    return new Map(rows.map((row) => [row.entry_id, recordFromRow(row)]));
  }

  async backfillSession(sessionId: SessionId): Promise<{ inserted: number }> {
    const streamPath = getSessionStreamPath(this.dataDir, sessionId);

    if (!existsSync(streamPath)) {
      return { inserted: 0 };
    }

    const coverage = this.db
      .prepare(
        `SELECT COUNT(*) AS entry_count
         FROM stream_entry_index
         WHERE session_id = ?`,
      )
      .get(sessionId) as SessionEntryCountRow;
    const fileDescriptor = openSync(streamPath, "r");

    try {
      const fileSize = fstatSync(fileDescriptor).size;

      if (fileSize === 0) {
        return { inserted: 0 };
      }

      const fileEntryCount = scanForwardStreamEntries(
        fileDescriptor,
        fileSize,
        streamPath,
        this.logger,
        () => undefined,
      );

      if (coverage.entry_count === fileEntryCount) {
        return { inserted: 0 };
      }

      const insertMissing = this.db.transaction((): number => {
        const insert = this.db.prepare(
          `INSERT INTO stream_entry_index (entry_id, session_id, byte_offset, timestamp)
           VALUES (?, ?, ?, ?)
           ON CONFLICT (entry_id) DO NOTHING`,
        );
        let inserted = 0;

        scanForwardStreamEntries(
          fileDescriptor,
          fileSize,
          streamPath,
          this.logger,
          (entry, byteOffset) => {
            inserted += Number(
              insert.run(entry.id, sessionId, byteOffset, entry.timestamp).changes,
            );
          },
        );

        return inserted;
      });

      return {
        inserted: insertMissing(),
      };
    } finally {
      closeSync(fileDescriptor);
    }
  }
}
