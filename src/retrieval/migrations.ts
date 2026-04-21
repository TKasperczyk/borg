import type { Migration } from "../storage/sqlite/index.js";

export const retrievalMigrations = [
  {
    id: 120,
    name: "create-retrieval-log",
    up: `
      CREATE TABLE IF NOT EXISTS retrieval_log (
        episode_id TEXT NOT NULL,
        timestamp INTEGER NOT NULL,
        score REAL NOT NULL
      );
      CREATE INDEX IF NOT EXISTS retrieval_log_episode_idx
      ON retrieval_log (episode_id, timestamp DESC)
    `,
  },
] as const satisfies readonly Migration[];
