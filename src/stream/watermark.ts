import { SystemClock, type Clock } from "../util/clock.js";
import type { Migration, SqliteDatabase } from "../storage/sqlite/index.js";
import { tableHasColumn } from "../storage/sqlite/migrations-utils.js";
import type { SessionId } from "../util/ids.js";
import { StorageError } from "../util/errors.js";
import { assertJsonValue, serializeJsonValue, type JsonValue } from "../util/json-value.js";

export const streamWatermarkMigrations: Migration[] = [
  {
    id: 1,
    name: "stream_watermark_baseline",
    up: (db) => {
      db.exec(`
        CREATE TABLE stream_watermarks (
        process_name TEXT NOT NULL,
        session_id TEXT NOT NULL,
        last_ts INTEGER NOT NULL,
        last_entry_id TEXT NOT NULL,
        updated_at INTEGER NOT NULL,
        PRIMARY KEY (process_name, session_id)
      );
      `);
    },
  },
  {
    id: 2,
    name: "stream_watermark_metadata_json",
    up: (db) => {
      if (!tableHasColumn(db, "stream_watermarks", "metadata_json")) {
        db.exec("ALTER TABLE stream_watermarks ADD COLUMN metadata_json TEXT");
      }
    },
  },
];

export type StreamWatermark = {
  processName: string;
  sessionId: SessionId;
  lastTs: number;
  lastEntryId: string;
  updatedAt: number;
  metadata: JsonValue | null;
};

type WatermarkRow = {
  process_name: string;
  session_id: string;
  last_ts: number;
  last_entry_id: string | null;
  updated_at: number;
  metadata_json?: string | null;
};

export type StreamWatermarkRepositoryOptions = {
  db: SqliteDatabase;
  clock?: Clock;
};

function parseMetadata(value: string | null | undefined): JsonValue | null {
  if (value === null || value === undefined) {
    return null;
  }

  try {
    const parsed = JSON.parse(value) as unknown;
    assertJsonValue(parsed);
    return parsed;
  } catch (error) {
    throw new StorageError("Stream watermark metadata failed validation", {
      cause: error,
      code: "STREAM_WATERMARK_METADATA_INVALID",
    });
  }
}

/**
 * Tracks per-process, per-session high-water marks in the stream. Each
 * downstream consumer (episodic extraction, semantic extraction, etc.)
 * advances its own watermark as it processes entries, so it can resume from
 * the right point without re-reading the full session.
 */
export class StreamWatermarkRepository {
  private readonly db: SqliteDatabase;
  private readonly clock: Clock;

  constructor(options: StreamWatermarkRepositoryOptions) {
    this.db = options.db;
    this.clock = options.clock ?? new SystemClock();
  }

  runInTransaction<T>(callback: () => T): T {
    return this.db.transaction(callback)();
  }

  get(processName: string, sessionId: SessionId): StreamWatermark | null {
    const row = this.db
      .prepare(
        `SELECT process_name, session_id, last_ts, last_entry_id, updated_at, metadata_json
         FROM stream_watermarks
         WHERE process_name = ? AND session_id = ?`,
      )
      .get(processName, sessionId) as WatermarkRow | undefined;

    if (row === undefined) {
      return null;
    }

    if (
      row.last_entry_id === null ||
      typeof row.last_entry_id !== "string" ||
      row.last_entry_id.length === 0
    ) {
      throw new StorageError(
        `Stream watermark has invalid last_entry_id (${row.last_entry_id}); fail loudly per strict-cursor contract.`,
        {
          code: "STREAM_WATERMARK_INVALID_CURSOR",
        },
      );
    }

    return {
      processName: row.process_name,
      sessionId: row.session_id as SessionId,
      lastTs: row.last_ts,
      lastEntryId: row.last_entry_id,
      updatedAt: row.updated_at,
      metadata: parseMetadata(row.metadata_json),
    };
  }

  set(
    processName: string,
    sessionId: SessionId,
    input: { lastTs: number; lastEntryId: string; metadata?: JsonValue | null },
  ): StreamWatermark {
    const nowMs = this.clock.now();
    const metadataJson =
      input.metadata === undefined || input.metadata === null
        ? null
        : serializeJsonValue(input.metadata);
    this.db
      .prepare(
        `INSERT INTO stream_watermarks (
           process_name, session_id, last_ts, last_entry_id, updated_at, metadata_json
         )
         VALUES (?, ?, ?, ?, ?, ?)
         ON CONFLICT (process_name, session_id) DO UPDATE SET
           last_ts = excluded.last_ts,
           last_entry_id = excluded.last_entry_id,
           updated_at = excluded.updated_at,
           metadata_json = excluded.metadata_json`,
      )
      .run(processName, sessionId, input.lastTs, input.lastEntryId, nowMs, metadataJson);

    return {
      processName,
      sessionId,
      lastTs: input.lastTs,
      lastEntryId: input.lastEntryId,
      updatedAt: nowMs,
      metadata: input.metadata ?? null,
    };
  }

  reset(processName: string, sessionId: SessionId): void {
    this.db
      .prepare(`DELETE FROM stream_watermarks WHERE process_name = ? AND session_id = ?`)
      .run(processName, sessionId);
  }
}
