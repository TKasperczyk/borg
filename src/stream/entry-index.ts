import { closeSync, existsSync, fstatSync, openSync, readSync } from "node:fs";

import { type Migration, type SqliteDatabase } from "../storage/sqlite/index.js";
import { tableExists, tableHasColumn } from "../storage/sqlite/migrations-utils.js";
import type { EntityId } from "../util/ids.js";

import { getSessionStreamPath } from "./path.js";
import {
  type SessionId,
  type StreamEntry,
  type StreamEntryKind,
  type StreamTurnStatus,
  streamEntrySchema,
} from "./types.js";
import { collectInactiveStreamEntryRefs, streamEntryIsActive } from "./turn-status.js";

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
        timestamp INTEGER NOT NULL,
        kind TEXT NULL,
        sender_entity_id TEXT NULL
      );
      CREATE INDEX IF NOT EXISTS idx_stream_entry_session
      ON stream_entry_index(session_id)
    `,
  },
  {
    id: 202,
    name: "add-stream-entry-sender-entity-id",
    up: (db) => {
      if (
        tableExists(db, "stream_entry_index") &&
        !tableHasColumn(db, "stream_entry_index", "sender_entity_id")
      ) {
        db.exec(`
          ALTER TABLE stream_entry_index
            ADD COLUMN sender_entity_id TEXT NULL;
        `);
      }
    },
  },
  {
    id: 203,
    name: "add-stream-entry-kind",
    up: (db) => {
      if (
        tableExists(db, "stream_entry_index") &&
        !tableHasColumn(db, "stream_entry_index", "kind")
      ) {
        db.exec(`
          ALTER TABLE stream_entry_index
            ADD COLUMN kind TEXT NULL;
        `);
      }

      db.exec(`
        CREATE INDEX IF NOT EXISTS idx_stream_entry_session_kind
        ON stream_entry_index(session_id, kind);
      `);
    },
  },
  {
    id: 204,
    name: "add-stream-entry-trust-facts",
    up: (db) => {
      if (
        tableExists(db, "stream_entry_index") &&
        !tableHasColumn(db, "stream_entry_index", "turn_id")
      ) {
        db.exec(`
          ALTER TABLE stream_entry_index
            ADD COLUMN turn_id TEXT NULL;
        `);
      }

      if (
        tableExists(db, "stream_entry_index") &&
        !tableHasColumn(db, "stream_entry_index", "turn_status")
      ) {
        db.exec(`
          ALTER TABLE stream_entry_index
            ADD COLUMN turn_status TEXT NULL;
        `);
      }

      if (
        tableExists(db, "stream_entry_index") &&
        !tableHasColumn(db, "stream_entry_index", "active")
      ) {
        db.exec(`
          ALTER TABLE stream_entry_index
            ADD COLUMN active INTEGER NOT NULL DEFAULT 1;
        `);
      }

      db.exec(`
        CREATE INDEX IF NOT EXISTS idx_stream_entry_active
        ON stream_entry_index(active);
      `);
    },
  },
];

export type StreamEntryIndexRecord = {
  entry_id: string;
  session_id: SessionId;
  byte_offset: number;
  timestamp: number;
  kind: StreamEntryKind | null;
  sender_entity_id: EntityId | null;
  turn_id: string | null;
  turn_status: StreamTurnStatus | null;
  active: boolean;
};

export type IndexedEntryFacts = Pick<
  StreamEntryIndexRecord,
  "entry_id" | "session_id" | "timestamp" | "kind" | "turn_id" | "turn_status" | "active"
>;

type StreamEntryIndexRow = {
  entry_id: string;
  session_id: string;
  byte_offset: number;
  timestamp: number;
  kind?: string | null;
  sender_entity_id: string | null;
  turn_id?: string | null;
  turn_status?: string | null;
  active?: number | null;
};

type SessionEntryCountRow = {
  entry_count: number;
};

type MissingKindCountRow = {
  missing_kind_count: number;
};

type MissingTrustFactsCountRow = {
  missing_trust_fact_count: number;
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
    kind: row.kind === null || row.kind === undefined ? null : (row.kind as StreamEntryKind),
    sender_entity_id:
      row.sender_entity_id === null || row.sender_entity_id === undefined
        ? null
        : (row.sender_entity_id as EntityId),
    turn_id: row.turn_id ?? null,
    turn_status:
      row.turn_status === null || row.turn_status === undefined
        ? null
        : (row.turn_status as StreamTurnStatus),
    active: row.active === null || row.active === undefined ? true : row.active !== 0,
  };
}

function factsFromRecord(record: StreamEntryIndexRecord): IndexedEntryFacts {
  return {
    entry_id: record.entry_id,
    session_id: record.session_id,
    timestamp: record.timestamp,
    kind: record.kind,
    turn_id: record.turn_id,
    turn_status: record.turn_status,
    active: record.active,
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

  record(
    entryId: string,
    sessionId: SessionId,
    byteOffset: number,
    timestamp: number,
    kind: StreamEntryKind | null = null,
    senderEntityId: EntityId | null = null,
    turnId: string | null = null,
    turnStatus: StreamTurnStatus | null = null,
    active = true,
  ): void {
    this.db
      .prepare(
        `INSERT INTO stream_entry_index (
           entry_id, session_id, byte_offset, timestamp, kind, sender_entity_id,
           turn_id, turn_status, active
         )
         VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
         ON CONFLICT (entry_id) DO UPDATE SET
           session_id = excluded.session_id,
           byte_offset = excluded.byte_offset,
           timestamp = excluded.timestamp,
           kind = excluded.kind,
           sender_entity_id = excluded.sender_entity_id,
           turn_id = excluded.turn_id,
           turn_status = excluded.turn_status,
           active = excluded.active`,
      )
      .run(
        entryId,
        sessionId,
        byteOffset,
        timestamp,
        kind,
        senderEntityId,
        turnId,
        turnStatus,
        active ? 1 : 0,
      );
  }

  recordEntry(entry: StreamEntry, byteOffset: number): void {
    const inactiveRefs = collectInactiveStreamEntryRefs([entry]);

    this.record(
      entry.id,
      entry.session_id,
      byteOffset,
      entry.timestamp,
      entry.kind,
      entry.sender_entity_id,
      entry.turn_id ?? null,
      entry.turn_status ?? "active",
      streamEntryIsActive(entry, inactiveRefs),
    );

    if (inactiveRefs.streamEntryIds.size > 0) {
      const ids = [...inactiveRefs.streamEntryIds];

      this.db
        .prepare(
          `UPDATE stream_entry_index
           SET active = 0
           WHERE entry_id IN (${ids.map(() => "?").join(", ")})`,
        )
        .run(...ids);
    }

    if (inactiveRefs.turnIds.size > 0) {
      const turnIds = [...inactiveRefs.turnIds];

      this.db
        .prepare(
          `UPDATE stream_entry_index
           SET active = 0
           WHERE session_id = ? AND turn_id IN (${turnIds.map(() => "?").join(", ")})`,
        )
        .run(entry.session_id, ...turnIds);
    }
  }

  lookup(entryId: string): StreamEntryIndexRecord | null {
    const row = this.db
      .prepare(
        `SELECT entry_id, session_id, byte_offset, timestamp
              , kind, sender_entity_id, turn_id, turn_status, active
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
              , kind, sender_entity_id, turn_id, turn_status, active
         FROM stream_entry_index
         WHERE entry_id IN (${uniqueIds.map(() => "?").join(", ")})`,
      )
      .all(...uniqueIds) as StreamEntryIndexRow[];

    return new Map(rows.map((row) => [row.entry_id, recordFromRow(row)]));
  }

  lookupEntriesById(entryIds: readonly string[]): Map<string, IndexedEntryFacts> {
    return new Map(
      [...this.lookupMany(entryIds)].map(([entryId, record]) => [entryId, factsFromRecord(record)]),
    );
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
    const missingKindCoverage = this.db
      .prepare(
        `SELECT COUNT(*) AS missing_kind_count
         FROM stream_entry_index
         WHERE session_id = ? AND kind IS NULL`,
      )
      .get(sessionId) as MissingKindCountRow;
    const missingTrustFactsCoverage = this.db
      .prepare(
        `SELECT COUNT(*) AS missing_trust_fact_count
         FROM stream_entry_index
         WHERE session_id = ? AND turn_status IS NULL`,
      )
      .get(sessionId) as MissingTrustFactsCountRow;
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

      if (
        coverage.entry_count === fileEntryCount &&
        missingKindCoverage.missing_kind_count === 0 &&
        missingTrustFactsCoverage.missing_trust_fact_count === 0
      ) {
        return { inserted: 0 };
      }

      const insertMissing = this.db.transaction((): number => {
        const insert = this.db.prepare(
          `INSERT INTO stream_entry_index (
             entry_id, session_id, byte_offset, timestamp, kind, sender_entity_id,
             turn_id, turn_status, active
           )
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
           ON CONFLICT (entry_id) DO UPDATE SET
             session_id = excluded.session_id,
             byte_offset = excluded.byte_offset,
             timestamp = excluded.timestamp,
             kind = excluded.kind,
             sender_entity_id = excluded.sender_entity_id,
             turn_id = excluded.turn_id,
             turn_status = excluded.turn_status,
             active = excluded.active
           WHERE stream_entry_index.session_id != excluded.session_id
              OR stream_entry_index.byte_offset != excluded.byte_offset
              OR stream_entry_index.timestamp != excluded.timestamp
              OR stream_entry_index.kind IS NOT excluded.kind
              OR stream_entry_index.sender_entity_id IS NOT excluded.sender_entity_id
              OR stream_entry_index.turn_id IS NOT excluded.turn_id
              OR stream_entry_index.turn_status IS NOT excluded.turn_status
              OR stream_entry_index.active IS NOT excluded.active`,
        );
        let inserted = 0;
        const scannedEntries: { entry: StreamEntry; byteOffset: number }[] = [];

        scanForwardStreamEntries(
          fileDescriptor,
          fileSize,
          streamPath,
          this.logger,
          (entry, byteOffset) => {
            scannedEntries.push({ entry, byteOffset });
          },
        );

        const inactiveRefs = collectInactiveStreamEntryRefs(
          scannedEntries.map((scanned) => scanned.entry),
        );

        for (const { entry, byteOffset } of scannedEntries) {
          inserted += Number(
            insert.run(
              entry.id,
              sessionId,
              byteOffset,
              entry.timestamp,
              entry.kind,
              entry.sender_entity_id,
              entry.turn_id ?? null,
              entry.turn_status ?? "active",
              streamEntryIsActive(entry, inactiveRefs) ? 1 : 0,
            ).changes,
          );
        }

        return inserted;
      });

      return {
        inserted: insertMissing(),
      };
    } finally {
      closeSync(fileDescriptor);
    }
  }

  countSessionEntriesByKind(input: {
    sessionId: SessionId;
    kind: StreamEntryKind;
    excludeEntryId?: string;
  }): number {
    const row =
      input.excludeEntryId === undefined
        ? (this.db
            .prepare(
              `SELECT COUNT(*) AS entry_count
               FROM stream_entry_index
               WHERE session_id = ? AND kind = ?`,
            )
            .get(input.sessionId, input.kind) as SessionEntryCountRow)
        : (this.db
            .prepare(
              `SELECT COUNT(*) AS entry_count
               FROM stream_entry_index
               WHERE session_id = ? AND kind = ? AND entry_id != ?`,
            )
            .get(input.sessionId, input.kind, input.excludeEntryId) as SessionEntryCountRow);

    return row.entry_count;
  }
}
