import type { Migration } from "../storage/sqlite/index.js";

export const retrievalMigrations = [
  {
    id: 1,
    name: "retrieval_baseline",
    up: (db) => {
      db.exec(`
        CREATE TABLE retrieval_log (
          episode_id TEXT NOT NULL,
          timestamp INTEGER NOT NULL,
          score REAL NOT NULL
        );
        CREATE INDEX retrieval_log_episode_idx
          ON retrieval_log (episode_id, timestamp DESC);
        CREATE INDEX retrieval_log_timestamp_idx
          ON retrieval_log (timestamp);
        CREATE TABLE recall_state (
          scope_key TEXT PRIMARY KEY,
          state_json TEXT NOT NULL,
          updated_at INTEGER NOT NULL
        );
        CREATE INDEX recall_state_updated_at_idx
          ON recall_state (updated_at);
      `);
    },
  },
] as const satisfies readonly Migration[];
