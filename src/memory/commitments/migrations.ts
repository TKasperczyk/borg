import type { Migration } from "../../storage/sqlite/index.js";

export const commitmentMigrations = [
  {
    id: 1,
    name: "commitments_initial_schema",
    up: (db) => {
      db.exec(`
        CREATE TABLE entities (
          id TEXT PRIMARY KEY,
          canonical_name TEXT NOT NULL,
          aliases TEXT NOT NULL,
          created_at INTEGER NOT NULL
        );

        CREATE INDEX IF NOT EXISTS entities_name_idx
          ON entities(canonical_name);

        CREATE TABLE commitments (
          id TEXT PRIMARY KEY,
          type TEXT NOT NULL,
          directive TEXT NOT NULL,
          priority INTEGER NOT NULL,
          made_to_entity TEXT NULL,
          restricted_audience TEXT NULL,
          about_entity TEXT NULL,
          source_episode_ids TEXT NOT NULL,
          created_at INTEGER NOT NULL,
          expires_at INTEGER NULL,
          revoked_at INTEGER NULL,
          superseded_by TEXT NULL,
          provenance_kind TEXT,
          provenance_episode_ids TEXT,
          provenance_process TEXT,
          expired_at INTEGER,
          revoked_reason TEXT,
          revoke_provenance_kind TEXT,
          revoke_provenance_episode_ids TEXT,
          revoke_provenance_process TEXT
        );

        CREATE INDEX IF NOT EXISTS commitments_audience_idx
          ON commitments(restricted_audience);
        CREATE INDEX IF NOT EXISTS commitments_about_idx
          ON commitments(about_entity);
      `);
    },
  },
] as const satisfies readonly Migration[];
