import type { Migration } from "../../storage/sqlite/index.js";

export const commitmentMigrations: Migration[] = [
  {
    id: 140,
    name: "entities_and_commitments",
    up: `
      CREATE TABLE IF NOT EXISTS entities (
        id TEXT PRIMARY KEY,
        canonical_name TEXT NOT NULL,
        aliases TEXT NOT NULL,
        created_at INTEGER NOT NULL
      );

      CREATE INDEX IF NOT EXISTS entities_name_idx
        ON entities(canonical_name);

      CREATE TABLE IF NOT EXISTS commitments (
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
        superseded_by TEXT NULL
      );

      CREATE INDEX IF NOT EXISTS commitments_audience_idx
        ON commitments(restricted_audience);
      CREATE INDEX IF NOT EXISTS commitments_about_idx
        ON commitments(about_entity);
    `,
  },
];
