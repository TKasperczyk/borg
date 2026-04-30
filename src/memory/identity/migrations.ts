import type { Migration } from "../../storage/sqlite/index.js";
import { StorageError } from "../../util/errors.js";

export const identityMigrations = [
  {
    id: 250,
    name: "create-identity-events",
    up: `
      CREATE TABLE IF NOT EXISTS identity_events (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        record_type TEXT NOT NULL,
        record_id TEXT NOT NULL,
        action TEXT NOT NULL,
        old_value_json TEXT,
        new_value_json TEXT,
        reason TEXT,
        provenance_kind TEXT NOT NULL CHECK (
          provenance_kind IN ('episodes', 'manual', 'system', 'offline')
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
    `,
  },
  {
    id: 251,
    name: "allow-online-identity-event-provenance",
    up: (db) => {
      const oldRowCount = (
        db.prepare("SELECT COUNT(*) as count FROM identity_events").get() as { count: number }
      ).count;
      if (oldRowCount > 0) {
        throw new StorageError(
          `identity_events table has ${oldRowCount} rows. Migration 251 cannot evolve the schema while preserving data; wipe the local data dir and re-run.`,
          {
            code: "IDENTITY_EVENTS_MIGRATION_REQUIRES_FRESH_DATABASE",
          },
        );
      }

      db.exec(`
        CREATE TABLE identity_events__next (
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

        DROP TABLE identity_events;
        ALTER TABLE identity_events__next RENAME TO identity_events;

        CREATE INDEX IF NOT EXISTS idx_identity_events_record_ts
          ON identity_events (record_type, record_id, ts DESC, id DESC);
        CREATE INDEX IF NOT EXISTS idx_identity_events_ts
          ON identity_events (ts DESC, id DESC);
      `);
    },
  },
] as const satisfies readonly Migration[];
