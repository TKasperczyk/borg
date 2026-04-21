import type { Migration } from "../../storage/sqlite/index.js";

export const selfMigrations = [
  {
    id: 110,
    name: "create-self-tables",
    up: `
      CREATE TABLE IF NOT EXISTS "values" (
        id TEXT PRIMARY KEY,
        label TEXT NOT NULL,
        description TEXT NOT NULL,
        priority REAL NOT NULL,
        created_at INTEGER NOT NULL,
        last_affirmed INTEGER
      );
      CREATE TABLE IF NOT EXISTS value_sources (
        value_id TEXT NOT NULL,
        episode_id TEXT NOT NULL,
        PRIMARY KEY (value_id, episode_id),
        FOREIGN KEY (value_id) REFERENCES "values"(id) ON DELETE CASCADE
      );
      CREATE TABLE IF NOT EXISTS goals (
        id TEXT PRIMARY KEY,
        description TEXT NOT NULL,
        priority REAL NOT NULL,
        parent_goal_id TEXT,
        status TEXT NOT NULL CHECK (status IN ('active', 'done', 'abandoned', 'blocked')),
        progress_notes TEXT,
        created_at INTEGER NOT NULL,
        target_at INTEGER,
        FOREIGN KEY (parent_goal_id) REFERENCES goals(id) ON DELETE SET NULL
      );
      CREATE TABLE IF NOT EXISTS traits (
        label TEXT PRIMARY KEY,
        strength REAL NOT NULL,
        last_reinforced INTEGER NOT NULL,
        last_decayed INTEGER
      )
    `,
  },
] as const satisfies readonly Migration[];
