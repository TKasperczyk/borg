import type { Migration } from "../../storage/sqlite/index.js";

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
            provenance_kind IN ('episodes', 'manual', 'system', 'offline', 'online')
          ),
          provenance_episode_ids TEXT NOT NULL DEFAULT '[]',
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
] as const satisfies readonly Migration[];
