import type { Migration } from "../../storage/sqlite/index.js";

export const proceduralMigrations = [
  {
    id: 172,
    name: "create-skills-table",
    up: `
      CREATE TABLE IF NOT EXISTS skills (
        id TEXT PRIMARY KEY,
        applies_when TEXT NOT NULL,
        approach TEXT NOT NULL,
        alpha REAL NOT NULL,
        beta REAL NOT NULL,
        attempts INTEGER NOT NULL,
        successes INTEGER NOT NULL,
        failures INTEGER NOT NULL,
        alternatives TEXT NOT NULL,
        source_episode_ids TEXT NOT NULL,
        last_used INTEGER,
        last_successful INTEGER,
        created_at INTEGER NOT NULL,
        updated_at INTEGER NOT NULL
      );
      CREATE INDEX IF NOT EXISTS idx_skills_updated_at
        ON skills (updated_at DESC);
    `,
  },
] as const satisfies readonly Migration[];
