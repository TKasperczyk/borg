import type { Migration } from "../storage/sqlite/index.js";

export const retrievalMigrations = [
  {
    id: 1,
    name: "retrieval_initial_schema",
    up: (db) => {
      db.exec(`
        CREATE TABLE retrieval_log (
          episode_id TEXT NOT NULL,
          timestamp INTEGER NOT NULL,
          score REAL NOT NULL
        );

        CREATE INDEX IF NOT EXISTS retrieval_log_episode_idx
          ON retrieval_log (episode_id, timestamp DESC);
        CREATE INDEX IF NOT EXISTS retrieval_log_timestamp_idx
          ON retrieval_log (timestamp);
      `);
    },
  },
] as const satisfies readonly Migration[];
