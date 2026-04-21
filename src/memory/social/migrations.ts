import type { Migration } from "../../storage/sqlite/index.js";

export const socialMigrations = [
  {
    id: 171,
    name: "create-social-profiles",
    up: `
      CREATE TABLE IF NOT EXISTS social_profiles (
        entity_id TEXT PRIMARY KEY,
        trust REAL NOT NULL DEFAULT 0.5,
        attachment REAL NOT NULL DEFAULT 0.0,
        communication_style TEXT,
        shared_history_summary TEXT,
        last_interaction_at INTEGER,
        interaction_count INTEGER NOT NULL DEFAULT 0,
        commitment_count INTEGER NOT NULL DEFAULT 0,
        sentiment_history TEXT NOT NULL DEFAULT '[]',
        notes TEXT,
        created_at INTEGER NOT NULL,
        updated_at INTEGER NOT NULL
      )
    `,
  },
] as const satisfies readonly Migration[];
