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
  {
    id: 2,
    name: "commitment_source_stream_entry_ids",
    up: (db) => {
      db.exec(`
        ALTER TABLE commitments
          ADD COLUMN source_stream_entry_ids TEXT NULL;
      `);
    },
  },
  {
    id: 3,
    name: "commitment_directive_family",
    up: (db) => {
      db.exec(`
        ALTER TABLE commitments
          ADD COLUMN directive_family TEXT NULL;

        UPDATE commitments
        SET directive_family = 'uncategorized'
        WHERE directive_family IS NULL;

        ALTER TABLE commitments
          ADD COLUMN last_reinforced_at INTEGER NULL;

        UPDATE commitments
        SET last_reinforced_at = created_at
        WHERE last_reinforced_at IS NULL;

        CREATE INDEX IF NOT EXISTS commitments_directive_family_idx
          ON commitments(directive_family, restricted_audience, made_to_entity);
      `);
    },
  },
] as const satisfies readonly Migration[];
