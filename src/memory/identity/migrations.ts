import type { Migration, SqliteDatabase } from "../../storage/sqlite/index.js";

function tableExists(db: SqliteDatabase, tableName: string): boolean {
  const row = db
    .prepare("SELECT name FROM sqlite_master WHERE type = 'table' AND name = ?")
    .get(tableName) as { name: string } | undefined;

  return row !== undefined;
}

function columnExists(db: SqliteDatabase, tableName: string, columnName: string): boolean {
  const rows = db.prepare(`PRAGMA table_info(${tableName})`).all() as Array<{ name: string }>;

  return rows.some((row) => row.name === columnName);
}

export const identityMigrations = [
  {
    id: 1,
    name: "identity_initial_schema",
    up: (db) => {
      db.exec(`
        CREATE TABLE identity_events (
          id INTEGER PRIMARY KEY AUTOINCREMENT,
          record_type TEXT NOT NULL,
          record_id TEXT NOT NULL,
          action TEXT NOT NULL,
          old_value_json TEXT,
          new_value_json TEXT,
          reason TEXT,
          provenance_kind TEXT NOT NULL CHECK (
            provenance_kind IN (
              'episodes',
              'manual',
              'system',
              'offline',
              'online',
              'online_reflector'
            )
          ),
          provenance_episode_ids TEXT NOT NULL DEFAULT '[]',
          provenance_stream_entry_ids TEXT NOT NULL DEFAULT '[]',
          provenance_process TEXT,
          review_item_id INTEGER,
          overwrite_without_review INTEGER NOT NULL DEFAULT 0,
          ts INTEGER NOT NULL
        );

        CREATE INDEX IF NOT EXISTS idx_identity_events_record_ts
          ON identity_events (record_type, record_id, ts DESC, id DESC);
        CREATE INDEX IF NOT EXISTS idx_identity_events_ts
          ON identity_events (ts DESC, id DESC);
      `);
    },
  },
  {
    id: 2,
    name: "identity_online_reflector_provenance",
    up: (db) => {
      if (!tableExists(db, "identity_events")) {
        return;
      }

      if (columnExists(db, "identity_events", "provenance_stream_entry_ids")) {
        return;
      }

      db.exec(`
        ALTER TABLE identity_events
          RENAME TO identity_events_before_online_reflector_provenance;

        CREATE TABLE identity_events (
          id INTEGER PRIMARY KEY AUTOINCREMENT,
          record_type TEXT NOT NULL,
          record_id TEXT NOT NULL,
          action TEXT NOT NULL,
          old_value_json TEXT,
          new_value_json TEXT,
          reason TEXT,
          provenance_kind TEXT NOT NULL CHECK (
            provenance_kind IN (
              'episodes',
              'manual',
              'system',
              'offline',
              'online',
              'online_reflector'
            )
          ),
          provenance_episode_ids TEXT NOT NULL DEFAULT '[]',
          provenance_stream_entry_ids TEXT NOT NULL DEFAULT '[]',
          provenance_process TEXT,
          review_item_id INTEGER,
          overwrite_without_review INTEGER NOT NULL DEFAULT 0,
          ts INTEGER NOT NULL
        );

        INSERT INTO identity_events (
          id, record_type, record_id, action, old_value_json, new_value_json, reason,
          provenance_kind, provenance_episode_ids, provenance_stream_entry_ids,
          provenance_process, review_item_id, overwrite_without_review, ts
        )
        SELECT
          id, record_type, record_id, action, old_value_json, new_value_json, reason,
          provenance_kind, provenance_episode_ids, '[]', provenance_process, review_item_id,
          overwrite_without_review, ts
        FROM identity_events_before_online_reflector_provenance;

        DROP TABLE identity_events_before_online_reflector_provenance;

        CREATE INDEX IF NOT EXISTS idx_identity_events_record_ts
          ON identity_events (record_type, record_id, ts DESC, id DESC);
        CREATE INDEX IF NOT EXISTS idx_identity_events_ts
          ON identity_events (ts DESC, id DESC);
      `);
    },
  },
] as const satisfies readonly Migration[];
