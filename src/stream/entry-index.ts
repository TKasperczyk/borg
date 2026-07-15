import { closeSync, existsSync, fstatSync, openSync, readSync } from "node:fs";

import { type Migration, type SqliteDatabase } from "../storage/sqlite/index.js";
import { tableExists, tableHasColumn } from "../storage/sqlite/migrations-utils.js";
import { streamEntryIdHelpers, type EntityId, type StreamEntryId } from "../util/ids.js";
import { serializeJsonValue } from "../util/json-value.js";

import { getSessionStreamPath } from "./path.js";
import {
  type SessionId,
  type StreamCursor,
  type StreamEntry,
  type StreamEntryKind,
  type StreamSourceMessageKey,
  type StreamTurnStatus,
  streamEntrySchema,
} from "./types.js";
import {
  collectInactiveStreamEntryRefs,
  isQuarantinedUserEntryMarker,
  streamEntryIsActive,
} from "./turn-status.js";

type LoggerLike = Pick<Console, "error" | "warn">;

const FORWARD_SCAN_CHUNK_SIZE_BYTES = 64 * 1024;
const NEWLINE_BYTE = 0x0a;

export const streamEntryIndexMigrations: Migration[] = [
  {
    id: 1,
    name: "stream_entry_index_baseline",
    up: (db) => {
      db.exec(`
        CREATE TABLE stream_entry_index (
          entry_id TEXT PRIMARY KEY,
          session_id TEXT NOT NULL,
          byte_offset INTEGER NOT NULL,
          timestamp INTEGER NOT NULL,
          kind TEXT NULL,
          sender_entity_id TEXT NULL,
          turn_id TEXT NULL,
          turn_status TEXT NULL,
          active INTEGER NOT NULL DEFAULT 1,
          receipt_pending INTEGER NOT NULL DEFAULT 0,
          entry_index INTEGER NULL,
          source_message_key_source_type TEXT NULL,
          source_message_key_source_external_id TEXT NULL,
          source_message_key_external_message_id TEXT NULL,
          response_to_kind TEXT NULL,
          response_to_from_cursor_ts INTEGER NULL,
          response_to_from_cursor_entry_id TEXT NULL,
          response_to_through_cursor_ts INTEGER NULL,
          response_to_through_cursor_entry_id TEXT NULL,
          response_to_source_entry_ids TEXT NULL,
          response_to_count INTEGER NULL
        );
        CREATE INDEX idx_stream_entry_active
        ON stream_entry_index(active);
        CREATE INDEX idx_stream_entry_session
      ON stream_entry_index(session_id)
    ;
        CREATE INDEX idx_stream_entry_session_entry_index
        ON stream_entry_index(session_id, entry_index);
        CREATE INDEX idx_stream_entry_session_kind
        ON stream_entry_index(session_id, kind);
        CREATE UNIQUE INDEX idx_stream_entry_source_message_key
        ON stream_entry_index(
          source_message_key_source_type,
          source_message_key_source_external_id,
          source_message_key_external_message_id
        )
        WHERE kind = 'user_msg'
          AND source_message_key_source_type IS NOT NULL
          AND source_message_key_source_external_id IS NOT NULL
          AND source_message_key_external_message_id IS NOT NULL;
        CREATE INDEX idx_stream_entry_response_to_through_cursor
        ON stream_entry_index(
          response_to_kind,
          response_to_through_cursor_ts,
          response_to_through_cursor_entry_id
        )
        WHERE response_to_through_cursor_entry_id IS NOT NULL;
        CREATE TABLE stream_quarantine_refs (
        marker_entry_id TEXT NOT NULL,
        marker_session_id TEXT NOT NULL,
        referenced_entry_id TEXT NOT NULL,
        timestamp INTEGER NOT NULL,
        PRIMARY KEY (marker_entry_id, referenced_entry_id)
      );
        CREATE INDEX idx_stream_quarantine_refs_referenced
      ON stream_quarantine_refs(referenced_entry_id);
        CREATE INDEX idx_stream_quarantine_refs_session
      ON stream_quarantine_refs(marker_session_id);
      `);
    },
  },
  {
    id: 2,
    name: "corrective_preference_ingestion_receipts",
    up: (db) => {
      db.exec(`
        CREATE TABLE corrective_preference_ingestion_receipts (
          source_entry_id TEXT PRIMARY KEY,
          session_id TEXT NOT NULL,
          status TEXT NOT NULL CHECK (status IN ('processed', 'retryable', 'dead_letter')),
          failure_count INTEGER NOT NULL DEFAULT 0,
          last_error TEXT NULL,
          updated_at INTEGER NOT NULL
        );
        CREATE INDEX idx_corrective_preference_receipts_session_status
        ON corrective_preference_ingestion_receipts(session_id, status);
      `);
    },
  },
];

export type CorrectivePreferenceIngestionReceiptStatus =
  | "processed"
  | "retryable"
  | "dead_letter";

export type CorrectivePreferenceIngestionReceipt = {
  source_entry_id: StreamEntryId;
  session_id: SessionId;
  status: CorrectivePreferenceIngestionReceiptStatus;
  failure_count: number;
  last_error: string | null;
  updated_at: number;
};

export type StreamEntryIndexRecord = {
  entry_id: string;
  session_id: SessionId;
  byte_offset: number;
  entry_index: number | null;
  timestamp: number;
  kind: StreamEntryKind | null;
  sender_entity_id: EntityId | null;
  turn_id: string | null;
  turn_status: StreamTurnStatus | null;
  active: boolean;
  receipt_pending: boolean;
  source_message_key_source_type: string | null;
  source_message_key_source_external_id: string | null;
  source_message_key_external_message_id: string | null;
  response_to_kind: string | null;
  response_to_from_cursor_ts: number | null;
  response_to_from_cursor_entry_id: StreamEntryId | null;
  response_to_through_cursor_ts: number | null;
  response_to_through_cursor_entry_id: StreamEntryId | null;
  response_to_source_entry_ids: string | null;
  response_to_count: number | null;
};

export type IndexedEntryFacts = Pick<
  StreamEntryIndexRecord,
  "entry_id" | "session_id" | "timestamp" | "kind" | "turn_id" | "turn_status" | "active"
>;

type StreamEntryIndexRow = {
  entry_id: string;
  session_id: string;
  byte_offset: number;
  entry_index?: number | null;
  timestamp: number;
  kind?: string | null;
  sender_entity_id: string | null;
  turn_id?: string | null;
  turn_status?: string | null;
  active?: number | null;
  receipt_pending?: number | null;
  source_message_key_source_type?: string | null;
  source_message_key_source_external_id?: string | null;
  source_message_key_external_message_id?: string | null;
  response_to_kind?: string | null;
  response_to_from_cursor_ts?: number | null;
  response_to_from_cursor_entry_id?: string | null;
  response_to_through_cursor_ts?: number | null;
  response_to_through_cursor_entry_id?: string | null;
  response_to_source_entry_ids?: string | null;
  response_to_count?: number | null;
};

type SessionEntryCountRow = {
  entry_count: number;
};

type MissingKindCountRow = {
  missing_kind_count: number;
};

type MissingKindSampleRow = {
  entry_id: string;
};

type NextEntryIndexRow = {
  next_entry_index: number;
};

type QuarantineRefRow = {
  referenced_entry_id: string;
};

type SessionIdRow = {
  session_id: string;
};

type CorrectivePreferenceIngestionReceiptRow = {
  source_entry_id: string;
  session_id: string;
  status: string;
  failure_count: number;
  last_error: string | null;
  updated_at: number;
};

export type StreamEntryIndexRepositoryOptions = {
  db: SqliteDatabase;
  dataDir: string;
  logger?: LoggerLike;
};

export type LegacyMissingKindRowsReport = {
  count: number;
  sampleEntryIds: string[];
};

export type LookupSessionStreamBacklogResponseStampsInput = {
  sessionId: SessionId;
  terminalKinds: readonly StreamEntryKind[];
};

export type LookupExactStreamBacklogResponseStampInput = {
  sessionId: SessionId;
  terminalKinds: readonly StreamEntryKind[];
  fromCursorExclusive: StreamCursor | null;
  throughCursorInclusive: StreamCursor;
  sourceEntryIds: readonly StreamEntryId[];
  count: number;
};

type StreamEntryIndexStampColumns = Pick<
  StreamEntryIndexRecord,
  | "source_message_key_source_type"
  | "source_message_key_source_external_id"
  | "source_message_key_external_message_id"
  | "response_to_kind"
  | "response_to_from_cursor_ts"
  | "response_to_from_cursor_entry_id"
  | "response_to_through_cursor_ts"
  | "response_to_through_cursor_entry_id"
  | "response_to_source_entry_ids"
  | "response_to_count"
>;

const EMPTY_STAMP_COLUMNS: StreamEntryIndexStampColumns = {
  source_message_key_source_type: null,
  source_message_key_source_external_id: null,
  source_message_key_external_message_id: null,
  response_to_kind: null,
  response_to_from_cursor_ts: null,
  response_to_from_cursor_entry_id: null,
  response_to_through_cursor_ts: null,
  response_to_through_cursor_entry_id: null,
  response_to_source_entry_ids: null,
  response_to_count: null,
};

function stampColumnsFromEntry(entry: StreamEntry): StreamEntryIndexStampColumns {
  const sourceMessageKey = entry.source_message_key;
  const responseTo = entry.response_to;

  return {
    source_message_key_source_type: sourceMessageKey?.source_type ?? null,
    source_message_key_source_external_id: sourceMessageKey?.source_external_id ?? null,
    source_message_key_external_message_id: sourceMessageKey?.external_message_id ?? null,
    response_to_kind: responseTo?.kind ?? null,
    response_to_from_cursor_ts: responseTo?.from_cursor_exclusive?.ts ?? null,
    response_to_from_cursor_entry_id: responseTo?.from_cursor_exclusive?.entryId ?? null,
    response_to_through_cursor_ts: responseTo?.through_cursor_inclusive.ts ?? null,
    response_to_through_cursor_entry_id: responseTo?.through_cursor_inclusive.entryId ?? null,
    response_to_source_entry_ids:
      responseTo === undefined ? null : serializeJsonValue(responseTo.source_entry_ids),
    response_to_count: responseTo?.count ?? null,
  };
}

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
    entry_index: row.entry_index ?? null,
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
    receipt_pending:
      row.receipt_pending === null || row.receipt_pending === undefined
        ? false
        : row.receipt_pending !== 0,
    source_message_key_source_type: row.source_message_key_source_type ?? null,
    source_message_key_source_external_id: row.source_message_key_source_external_id ?? null,
    source_message_key_external_message_id: row.source_message_key_external_message_id ?? null,
    response_to_kind: row.response_to_kind ?? null,
    response_to_from_cursor_ts: row.response_to_from_cursor_ts ?? null,
    response_to_from_cursor_entry_id:
      row.response_to_from_cursor_entry_id === null ||
      row.response_to_from_cursor_entry_id === undefined
        ? null
        : (row.response_to_from_cursor_entry_id as StreamEntryId),
    response_to_through_cursor_ts: row.response_to_through_cursor_ts ?? null,
    response_to_through_cursor_entry_id:
      row.response_to_through_cursor_entry_id === null ||
      row.response_to_through_cursor_entry_id === undefined
        ? null
        : (row.response_to_through_cursor_entry_id as StreamEntryId),
    response_to_source_entry_ids: row.response_to_source_entry_ids ?? null,
    response_to_count: row.response_to_count ?? null,
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

function collectQuarantinedSharedStateArtifactRefs(entry: StreamEntry): StreamEntryId[] {
  if (!isQuarantinedUserEntryMarker(entry)) {
    return [];
  }

  const content: Record<string, unknown> =
    entry.content !== null && typeof entry.content === "object" && !Array.isArray(entry.content)
      ? (entry.content as Record<string, unknown>)
      : {};
  const refs = new Set<StreamEntryId>();
  const addRef = (value: unknown): void => {
    if (typeof value === "string" && streamEntryIdHelpers.is(value)) {
      refs.add(value);
    }
  };

  addRef(content.source_stream_entry_id);

  if (Array.isArray(content.cited_stream_entry_ids)) {
    for (const item of content.cited_stream_entry_ids) {
      addRef(item);
    }
  }

  return [...refs];
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
  private readonly poisonedSessions = new Set<SessionId>();

  constructor(options: StreamEntryIndexRepositoryOptions) {
    this.db = options.db;
    this.dataDir = options.dataDir;
    this.logger = options.logger ?? console;
  }

  getCorrectivePreferenceIngestionReceipt(
    sourceEntryId: StreamEntryId,
  ): CorrectivePreferenceIngestionReceipt | null {
    const row = this.db
      .prepare(
        `SELECT source_entry_id, session_id, status, failure_count, last_error, updated_at
         FROM corrective_preference_ingestion_receipts
         WHERE source_entry_id = ?`,
      )
      .get(sourceEntryId) as CorrectivePreferenceIngestionReceiptRow | undefined;

    if (row === undefined) {
      return null;
    }

    return {
      source_entry_id: row.source_entry_id as StreamEntryId,
      session_id: row.session_id as SessionId,
      status: row.status as CorrectivePreferenceIngestionReceiptStatus,
      failure_count: row.failure_count,
      last_error: row.last_error,
      updated_at: row.updated_at,
    };
  }

  recordCorrectivePreferenceIngestionProcessed(input: {
    sourceEntryId: StreamEntryId;
    sessionId: SessionId;
    updatedAt: number;
  }): CorrectivePreferenceIngestionReceipt {
    this.db
      .prepare(
        `INSERT INTO corrective_preference_ingestion_receipts (
           source_entry_id, session_id, status, failure_count, last_error, updated_at
         ) VALUES (?, ?, 'processed', 0, NULL, ?)
         ON CONFLICT (source_entry_id) DO UPDATE SET
           session_id = excluded.session_id,
           status = 'processed',
           last_error = NULL,
           updated_at = excluded.updated_at
         WHERE corrective_preference_ingestion_receipts.status != 'dead_letter'`,
      )
      .run(input.sourceEntryId, input.sessionId, input.updatedAt);

    const receipt = this.getCorrectivePreferenceIngestionReceipt(input.sourceEntryId);

    if (receipt === null) {
      throw new Error("Failed to read corrective-preference ingestion receipt after write");
    }

    return receipt;
  }

  recordCorrectivePreferenceIngestionFailure(input: {
    sourceEntryId: StreamEntryId;
    sessionId: SessionId;
    error: string;
    updatedAt: number;
    maxFailures: number;
    deadLetterImmediately?: boolean;
  }): CorrectivePreferenceIngestionReceipt {
    const recordFailure = this.db.transaction(() => {
      const previous = this.getCorrectivePreferenceIngestionReceipt(input.sourceEntryId);

      if (previous?.status === "processed" || previous?.status === "dead_letter") {
        return previous;
      }

      const failureCount = (previous?.failure_count ?? 0) + 1;
      const status: CorrectivePreferenceIngestionReceiptStatus =
        input.deadLetterImmediately === true || failureCount >= input.maxFailures
          ? "dead_letter"
          : "retryable";

      this.db
        .prepare(
          `INSERT INTO corrective_preference_ingestion_receipts (
             source_entry_id, session_id, status, failure_count, last_error, updated_at
           ) VALUES (?, ?, ?, ?, ?, ?)
           ON CONFLICT (source_entry_id) DO UPDATE SET
             session_id = excluded.session_id,
             status = excluded.status,
             failure_count = excluded.failure_count,
             last_error = excluded.last_error,
             updated_at = excluded.updated_at`,
        )
        .run(
          input.sourceEntryId,
          input.sessionId,
          status,
          failureCount,
          input.error,
          input.updatedAt,
        );

      const receipt = this.getCorrectivePreferenceIngestionReceipt(input.sourceEntryId);

      if (receipt === null) {
        throw new Error("Failed to read corrective-preference ingestion receipt after failure");
      }

      return receipt;
    });

    return recordFailure();
  }

  record(
    entryId: string,
    sessionId: SessionId,
    byteOffset: number,
    entryIndex: number | null,
    timestamp: number,
    kind: StreamEntryKind | null = null,
    senderEntityId: EntityId | null = null,
    turnId: string | null = null,
    turnStatus: StreamTurnStatus | null = null,
    active = true,
    receiptPending = false,
    stampColumns: StreamEntryIndexStampColumns = EMPTY_STAMP_COLUMNS,
  ): void {
    this.db
      .prepare(
        `INSERT INTO stream_entry_index (
           entry_id, session_id, byte_offset, entry_index, timestamp, kind, sender_entity_id,
           turn_id, turn_status, active, receipt_pending,
           source_message_key_source_type, source_message_key_source_external_id,
           source_message_key_external_message_id, response_to_kind, response_to_from_cursor_ts,
           response_to_from_cursor_entry_id, response_to_through_cursor_ts,
           response_to_through_cursor_entry_id, response_to_source_entry_ids, response_to_count
         )
         VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
         ON CONFLICT (entry_id) DO UPDATE SET
           session_id = excluded.session_id,
           byte_offset = excluded.byte_offset,
           entry_index = excluded.entry_index,
           timestamp = excluded.timestamp,
           kind = excluded.kind,
           sender_entity_id = excluded.sender_entity_id,
           turn_id = excluded.turn_id,
           turn_status = excluded.turn_status,
           active = excluded.active,
           receipt_pending = excluded.receipt_pending,
           source_message_key_source_type = excluded.source_message_key_source_type,
           source_message_key_source_external_id = excluded.source_message_key_source_external_id,
           source_message_key_external_message_id = excluded.source_message_key_external_message_id,
           response_to_kind = excluded.response_to_kind,
           response_to_from_cursor_ts = excluded.response_to_from_cursor_ts,
           response_to_from_cursor_entry_id = excluded.response_to_from_cursor_entry_id,
           response_to_through_cursor_ts = excluded.response_to_through_cursor_ts,
           response_to_through_cursor_entry_id = excluded.response_to_through_cursor_entry_id,
           response_to_source_entry_ids = excluded.response_to_source_entry_ids,
           response_to_count = excluded.response_to_count`,
      )
      .run(
        entryId,
        sessionId,
        byteOffset,
        entryIndex,
        timestamp,
        kind,
        senderEntityId,
        turnId,
        turnStatus,
        active ? 1 : 0,
        receiptPending ? 1 : 0,
        stampColumns.source_message_key_source_type,
        stampColumns.source_message_key_source_external_id,
        stampColumns.source_message_key_external_message_id,
        stampColumns.response_to_kind,
        stampColumns.response_to_from_cursor_ts,
        stampColumns.response_to_from_cursor_entry_id,
        stampColumns.response_to_through_cursor_ts,
        stampColumns.response_to_through_cursor_entry_id,
        stampColumns.response_to_source_entry_ids,
        stampColumns.response_to_count,
      );
  }

  recordEntry(entry: StreamEntry, byteOffset: number): void {
    const inactiveRefs = collectInactiveStreamEntryRefs([entry]);

    this.record(
      entry.id,
      entry.session_id,
      byteOffset,
      entry.entry_index ?? null,
      entry.timestamp,
      entry.kind,
      entry.sender_entity_id,
      entry.turn_id ?? null,
      entry.turn_status ?? "active",
      streamEntryIsActive(entry, inactiveRefs),
      entry.receipt_pending === true,
      stampColumnsFromEntry(entry),
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

    this.recordQuarantineRefs(entry);
  }

  nextEntryIndex(sessionId: SessionId): number {
    const row = this.db
      .prepare(
        `SELECT MAX(COALESCE(MAX(entry_index) + 1, 0), COUNT(*)) AS next_entry_index
         FROM stream_entry_index
         WHERE session_id = ?`,
      )
      .get(sessionId) as NextEntryIndexRow;

    return row.next_entry_index;
  }

  isPoisoned(sessionId: SessionId): boolean {
    return this.poisonedSessions.has(sessionId);
  }

  markPoisoned(sessionId: SessionId): void {
    this.poisonedSessions.add(sessionId);
  }

  clearPoisoned(sessionId: SessionId): void {
    this.poisonedSessions.delete(sessionId);
  }

  setReceiptPending(entryId: StreamEntryId, pending: boolean): void {
    this.db
      .prepare(
        `UPDATE stream_entry_index
         SET receipt_pending = ?
         WHERE entry_id = ?`,
      )
      .run(pending ? 1 : 0, entryId);
  }

  private recordQuarantineRefs(entry: StreamEntry): void {
    const refs = collectQuarantinedSharedStateArtifactRefs(entry);

    if (refs.length === 0) {
      return;
    }

    const insert = this.db.prepare(
      `INSERT OR IGNORE INTO stream_quarantine_refs (
         marker_entry_id, marker_session_id, referenced_entry_id, timestamp
       )
       VALUES (?, ?, ?, ?)`,
    );

    const insertRefs = this.db.transaction(() => {
      for (const ref of refs) {
        insert.run(entry.id, entry.session_id, ref, entry.timestamp);
      }
    });

    insertRefs();
  }

  private refreshQuarantineRefsForSession(
    entries: readonly StreamEntry[],
    sessionId: SessionId,
  ): void {
    const deleteSessionRefs = this.db.prepare(
      `DELETE FROM stream_quarantine_refs
       WHERE marker_session_id = ?`,
    );
    const insert = this.db.prepare(
      `INSERT OR IGNORE INTO stream_quarantine_refs (
         marker_entry_id, marker_session_id, referenced_entry_id, timestamp
       )
       VALUES (?, ?, ?, ?)`,
    );

    const refresh = this.db.transaction(() => {
      deleteSessionRefs.run(sessionId);

      for (const entry of entries) {
        for (const ref of collectQuarantinedSharedStateArtifactRefs(entry)) {
          insert.run(entry.id, sessionId, ref, entry.timestamp);
        }
      }
    });

    refresh();
  }

  lookup(entryId: string): StreamEntryIndexRecord | null {
    const row = this.db
      .prepare(
        `SELECT entry_id, session_id, byte_offset, timestamp
              , entry_index, kind, sender_entity_id, turn_id, turn_status, active, receipt_pending
              , source_message_key_source_type, source_message_key_source_external_id
              , source_message_key_external_message_id, response_to_kind
              , response_to_from_cursor_ts, response_to_from_cursor_entry_id
              , response_to_through_cursor_ts, response_to_through_cursor_entry_id
              , response_to_source_entry_ids, response_to_count
         FROM stream_entry_index
         WHERE entry_id = ?`,
      )
      .get(entryId) as StreamEntryIndexRow | undefined;

    return row === undefined ? null : recordFromRow(row);
  }

  lookupBySourceMessageKey(key: StreamSourceMessageKey): StreamEntryIndexRecord | null {
    const row = this.db
      .prepare(
        `SELECT entry_id, session_id, byte_offset, timestamp
              , entry_index, kind, sender_entity_id, turn_id, turn_status, active, receipt_pending
              , source_message_key_source_type, source_message_key_source_external_id
              , source_message_key_external_message_id, response_to_kind
              , response_to_from_cursor_ts, response_to_from_cursor_entry_id
              , response_to_through_cursor_ts, response_to_through_cursor_entry_id
              , response_to_source_entry_ids, response_to_count
         FROM stream_entry_index
         WHERE source_message_key_source_type = ?
           AND source_message_key_source_external_id = ?
           AND source_message_key_external_message_id = ?
           AND kind = 'user_msg'`,
      )
      .get(key.source_type, key.source_external_id, key.external_message_id) as
      | StreamEntryIndexRow
      | undefined;

    if (row === undefined) {
      return null;
    }

    const record = recordFromRow(row);

    if (
      record.kind !== "user_msg" ||
      record.source_message_key_source_type !== key.source_type ||
      record.source_message_key_source_external_id !== key.source_external_id ||
      record.source_message_key_external_message_id !== key.external_message_id
    ) {
      return null;
    }

    return record;
  }

  listSessionIdsWithPendingResponseBacklog(): SessionId[] {
    const rows = this.db
      .prepare(
        `SELECT DISTINCT session_id
         FROM stream_entry_index
         WHERE kind = 'user_msg'
           AND turn_id IS NULL
         ORDER BY session_id ASC`,
      )
      .all() as SessionIdRow[];

    return rows.map((row) => row.session_id as SessionId);
  }

  lookupMany(entryIds: readonly string[]): Map<string, StreamEntryIndexRecord> {
    const uniqueIds = [...new Set(entryIds)];

    if (uniqueIds.length === 0) {
      return new Map();
    }

    const rows = this.db
      .prepare(
        `SELECT entry_id, session_id, byte_offset, timestamp
              , entry_index, kind, sender_entity_id, turn_id, turn_status, active, receipt_pending
              , source_message_key_source_type, source_message_key_source_external_id
              , source_message_key_external_message_id, response_to_kind
              , response_to_from_cursor_ts, response_to_from_cursor_entry_id
              , response_to_through_cursor_ts, response_to_through_cursor_entry_id
              , response_to_source_entry_ids, response_to_count
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

  lookupSessionEntriesByKind(input: {
    sessionId: SessionId;
    kind: StreamEntryKind;
  }): StreamEntryIndexRecord[] {
    const rows = this.db
      .prepare(
        `SELECT entry_id, session_id, byte_offset, timestamp
              , entry_index, kind, sender_entity_id, turn_id, turn_status, active, receipt_pending
              , source_message_key_source_type, source_message_key_source_external_id
              , source_message_key_external_message_id, response_to_kind
              , response_to_from_cursor_ts, response_to_from_cursor_entry_id
              , response_to_through_cursor_ts, response_to_through_cursor_entry_id
              , response_to_source_entry_ids, response_to_count
         FROM stream_entry_index
         WHERE session_id = ? AND kind = ?
         ORDER BY byte_offset ASC`,
      )
      .all(input.sessionId, input.kind) as StreamEntryIndexRow[];

    return rows.map(recordFromRow);
  }

  lookupSessionStreamBacklogResponseStamps(
    input: LookupSessionStreamBacklogResponseStampsInput,
  ): StreamEntryIndexRecord[] {
    const terminalKinds = [...new Set(input.terminalKinds)];

    if (terminalKinds.length === 0) {
      return [];
    }

    const rows = this.db
      .prepare(
        `SELECT entry_id, session_id, byte_offset, timestamp
              , entry_index, kind, sender_entity_id, turn_id, turn_status, active, receipt_pending
              , source_message_key_source_type, source_message_key_source_external_id
              , source_message_key_external_message_id, response_to_kind
              , response_to_from_cursor_ts, response_to_from_cursor_entry_id
              , response_to_through_cursor_ts, response_to_through_cursor_entry_id
              , response_to_source_entry_ids, response_to_count
         FROM stream_entry_index
         WHERE session_id = ?
           AND response_to_kind = 'stream_backlog'
           AND kind IN (${terminalKinds.map(() => "?").join(", ")})
         ORDER BY byte_offset ASC`,
      )
      .all(input.sessionId, ...terminalKinds) as StreamEntryIndexRow[];

    return rows.map(recordFromRow);
  }

  lookupExactStreamBacklogResponseStamp(
    input: LookupExactStreamBacklogResponseStampInput,
  ): StreamEntryIndexRecord | null {
    const terminalKinds = [...new Set(input.terminalKinds)];

    if (terminalKinds.length === 0) {
      return null;
    }

    const serializedSourceEntryIds = serializeJsonValue(input.sourceEntryIds);
    const terminalKindPlaceholders = terminalKinds.map(() => "?").join(", ");
    const baseSql = `SELECT entry_id, session_id, byte_offset, timestamp
              , entry_index, kind, sender_entity_id, turn_id, turn_status, active, receipt_pending
              , source_message_key_source_type, source_message_key_source_external_id
              , source_message_key_external_message_id, response_to_kind
              , response_to_from_cursor_ts, response_to_from_cursor_entry_id
              , response_to_through_cursor_ts, response_to_through_cursor_entry_id
              , response_to_source_entry_ids, response_to_count
         FROM stream_entry_index
         WHERE response_to_kind = 'stream_backlog'
           AND response_to_through_cursor_ts = ?
           AND response_to_through_cursor_entry_id = ?
           AND session_id = ?
           AND kind IN (${terminalKindPlaceholders})`;
    const tailSql = `AND response_to_source_entry_ids = ?
           AND response_to_count = ?
         ORDER BY byte_offset ASC
         LIMIT 1`;
    const row =
      input.fromCursorExclusive === null
        ? (this.db
            .prepare(
              `${baseSql}
           AND response_to_from_cursor_ts IS NULL
           AND response_to_from_cursor_entry_id IS NULL
           ${tailSql}`,
            )
            .get(
              input.throughCursorInclusive.ts,
              input.throughCursorInclusive.entryId,
              input.sessionId,
              ...terminalKinds,
              serializedSourceEntryIds,
              input.count,
            ) as StreamEntryIndexRow | undefined)
        : (this.db
            .prepare(
              `${baseSql}
           AND response_to_from_cursor_ts = ?
           AND response_to_from_cursor_entry_id = ?
           ${tailSql}`,
            )
            .get(
              input.throughCursorInclusive.ts,
              input.throughCursorInclusive.entryId,
              input.sessionId,
              ...terminalKinds,
              input.fromCursorExclusive.ts,
              input.fromCursorExclusive.entryId,
              serializedSourceEntryIds,
              input.count,
            ) as StreamEntryIndexRow | undefined);

    return row === undefined ? null : recordFromRow(row);
  }

  quarantinedSharedStateArtifactRefs(): ReadonlySet<StreamEntryId> {
    const rows = this.db
      .prepare(
        `SELECT DISTINCT referenced_entry_id
         FROM stream_quarantine_refs
         ORDER BY timestamp ASC, marker_entry_id ASC, referenced_entry_id ASC`,
      )
      .all() as QuarantineRefRow[];

    return new Set(rows.map((row) => row.referenced_entry_id as StreamEntryId));
  }

  async backfillSession(sessionId: SessionId): Promise<{ inserted: number }> {
    const streamPath = getSessionStreamPath(this.dataDir, sessionId);

    if (!existsSync(streamPath)) {
      this.clearPoisoned(sessionId);
      return { inserted: 0 };
    }

    const fileDescriptor = openSync(streamPath, "r");

    try {
      const fileSize = fstatSync(fileDescriptor).size;

      if (fileSize === 0) {
        this.clearPoisoned(sessionId);
        return { inserted: 0 };
      }

      const scannedEntries: { entry: StreamEntry; byteOffset: number; entryIndex: number }[] = [];
      scanForwardStreamEntries(
        fileDescriptor,
        fileSize,
        streamPath,
        this.logger,
        (entry, byteOffset) => {
          scannedEntries.push({
            entry,
            byteOffset,
            entryIndex: scannedEntries.length,
          });
        },
      );

      this.refreshQuarantineRefsForSession(
        scannedEntries.map((scanned) => scanned.entry),
        sessionId,
      );

      const insertMissing = this.db.transaction((): number => {
        const insert = this.db.prepare(
          `INSERT INTO stream_entry_index (
             entry_id, session_id, byte_offset, entry_index, timestamp, kind, sender_entity_id,
             turn_id, turn_status, active, receipt_pending,
             source_message_key_source_type, source_message_key_source_external_id,
             source_message_key_external_message_id, response_to_kind, response_to_from_cursor_ts,
             response_to_from_cursor_entry_id, response_to_through_cursor_ts,
             response_to_through_cursor_entry_id, response_to_source_entry_ids, response_to_count
           )
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
           ON CONFLICT (entry_id) DO UPDATE SET
             session_id = excluded.session_id,
             byte_offset = excluded.byte_offset,
             entry_index = excluded.entry_index,
             timestamp = excluded.timestamp,
             kind = excluded.kind,
             sender_entity_id = excluded.sender_entity_id,
             turn_id = excluded.turn_id,
             turn_status = excluded.turn_status,
             active = excluded.active,
             receipt_pending = CASE
               WHEN stream_entry_index.receipt_pending = 0 THEN 0
               ELSE excluded.receipt_pending
             END,
             source_message_key_source_type = excluded.source_message_key_source_type,
             source_message_key_source_external_id = excluded.source_message_key_source_external_id,
             source_message_key_external_message_id = excluded.source_message_key_external_message_id,
             response_to_kind = excluded.response_to_kind,
             response_to_from_cursor_ts = excluded.response_to_from_cursor_ts,
             response_to_from_cursor_entry_id = excluded.response_to_from_cursor_entry_id,
             response_to_through_cursor_ts = excluded.response_to_through_cursor_ts,
             response_to_through_cursor_entry_id = excluded.response_to_through_cursor_entry_id,
             response_to_source_entry_ids = excluded.response_to_source_entry_ids,
             response_to_count = excluded.response_to_count
           WHERE stream_entry_index.session_id != excluded.session_id
              OR stream_entry_index.byte_offset != excluded.byte_offset
              OR stream_entry_index.entry_index IS NOT excluded.entry_index
              OR stream_entry_index.timestamp != excluded.timestamp
              OR stream_entry_index.kind IS NOT excluded.kind
              OR stream_entry_index.sender_entity_id IS NOT excluded.sender_entity_id
              OR stream_entry_index.turn_id IS NOT excluded.turn_id
              OR stream_entry_index.turn_status IS NOT excluded.turn_status
              OR stream_entry_index.active IS NOT excluded.active
              OR stream_entry_index.receipt_pending IS NOT excluded.receipt_pending
              OR stream_entry_index.source_message_key_source_type IS NOT excluded.source_message_key_source_type
              OR stream_entry_index.source_message_key_source_external_id IS NOT excluded.source_message_key_source_external_id
              OR stream_entry_index.source_message_key_external_message_id IS NOT excluded.source_message_key_external_message_id
              OR stream_entry_index.response_to_kind IS NOT excluded.response_to_kind
              OR stream_entry_index.response_to_from_cursor_ts IS NOT excluded.response_to_from_cursor_ts
              OR stream_entry_index.response_to_from_cursor_entry_id IS NOT excluded.response_to_from_cursor_entry_id
              OR stream_entry_index.response_to_through_cursor_ts IS NOT excluded.response_to_through_cursor_ts
              OR stream_entry_index.response_to_through_cursor_entry_id IS NOT excluded.response_to_through_cursor_entry_id
              OR stream_entry_index.response_to_source_entry_ids IS NOT excluded.response_to_source_entry_ids
              OR stream_entry_index.response_to_count IS NOT excluded.response_to_count`,
        );
        let inserted = 0;

        const inactiveRefs = collectInactiveStreamEntryRefs(
          scannedEntries.map((scanned) => scanned.entry),
        );

        for (const { entry, byteOffset, entryIndex } of scannedEntries) {
          const stampColumns = stampColumnsFromEntry(entry);

          inserted += Number(
            insert.run(
              entry.id,
              sessionId,
              byteOffset,
              entryIndex,
              entry.timestamp,
              entry.kind,
              entry.sender_entity_id,
              entry.turn_id ?? null,
              entry.turn_status ?? "active",
              streamEntryIsActive(entry, inactiveRefs) ? 1 : 0,
              entry.receipt_pending === true ? 1 : 0,
              stampColumns.source_message_key_source_type,
              stampColumns.source_message_key_source_external_id,
              stampColumns.source_message_key_external_message_id,
              stampColumns.response_to_kind,
              stampColumns.response_to_from_cursor_ts,
              stampColumns.response_to_from_cursor_entry_id,
              stampColumns.response_to_through_cursor_ts,
              stampColumns.response_to_through_cursor_entry_id,
              stampColumns.response_to_source_entry_ids,
              stampColumns.response_to_count,
            ).changes,
          );
        }

        return inserted;
      });

      const inserted = insertMissing();

      this.clearPoisoned(sessionId);

      return {
        inserted,
      };
    } finally {
      closeSync(fileDescriptor);
    }
  }

  legacyRowsMissingKindReport(sampleLimit = 5): LegacyMissingKindRowsReport {
    const countRow = this.db
      .prepare(
        `SELECT COUNT(*) AS missing_kind_count
         FROM stream_entry_index
         WHERE kind IS NULL`,
      )
      .get() as MissingKindCountRow;
    const sampleRows = this.db
      .prepare(
        `SELECT entry_id
         FROM stream_entry_index
         WHERE kind IS NULL
         ORDER BY session_id ASC, byte_offset ASC, entry_id ASC
         LIMIT ?`,
      )
      .all(sampleLimit) as MissingKindSampleRow[];

    return {
      count: countRow.missing_kind_count,
      sampleEntryIds: sampleRows.map((row) => row.entry_id),
    };
  }

  warnLegacyRowsMissingKind(sampleLimit = 5): LegacyMissingKindRowsReport {
    const report = this.legacyRowsMissingKindReport(sampleLimit);

    if (report.count > 0) {
      this.logger.warn(
        `Stream entry index has ${report.count} legacy rows with kind IS NULL after startup backfill; sample_entry_ids=${report.sampleEntryIds.join(",")}`,
      );
    }

    return report;
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
