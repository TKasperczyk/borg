import { SystemClock, type Clock } from "../util/clock.js";
import type { Migration, SqliteDatabase } from "../storage/sqlite/index.js";
import type { SessionId } from "../util/ids.js";
import { StorageError } from "../util/errors.js";

export const streamWatermarkMigrations: Migration[] = [
  {
    id: 200,
    name: "create-stream-watermarks",
    up: `
      CREATE TABLE IF NOT EXISTS stream_watermarks (
        process_name TEXT NOT NULL,
        session_id TEXT NOT NULL,
        last_ts INTEGER NOT NULL,
        last_entry_id TEXT NOT NULL,
        updated_at INTEGER NOT NULL,
        PRIMARY KEY (process_name, session_id)
      )
    `,
  },
];

export type StreamWatermark = {
  processName: string;
  sessionId: SessionId;
  lastTs: number;
  lastEntryId: string;
  updatedAt: number;
};

type WatermarkRow = {
  process_name: string;
  session_id: string;
  last_ts: number;
  last_entry_id: string | null;
  updated_at: number;
};

export type StreamWatermarkRepositoryOptions = {
  db: SqliteDatabase;
  clock?: Clock;
};

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

  get(processName: string, sessionId: SessionId): StreamWatermark | null {
    const row = this.db
      .prepare(
        `SELECT process_name, session_id, last_ts, last_entry_id, updated_at
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
    };
  }

  set(
    processName: string,
    sessionId: SessionId,
    input: { lastTs: number; lastEntryId: string },
  ): StreamWatermark {
    const nowMs = this.clock.now();
    this.db
      .prepare(
        `INSERT INTO stream_watermarks (process_name, session_id, last_ts, last_entry_id, updated_at)
         VALUES (?, ?, ?, ?, ?)
         ON CONFLICT (process_name, session_id) DO UPDATE SET
           last_ts = excluded.last_ts,
           last_entry_id = excluded.last_entry_id,
           updated_at = excluded.updated_at`,
      )
      .run(processName, sessionId, input.lastTs, input.lastEntryId, nowMs);

    return {
      processName,
      sessionId,
      lastTs: input.lastTs,
      lastEntryId: input.lastEntryId,
      updatedAt: nowMs,
    };
  }

  reset(processName: string, sessionId: SessionId): void {
    this.db
      .prepare(`DELETE FROM stream_watermarks WHERE process_name = ? AND session_id = ?`)
      .run(processName, sessionId);
  }
}
