import type { Migration } from "../storage/sqlite/index.js";

export const operatorAdviceMigrations = [
  {
    id: 1,
    name: "operator_advice_initial_schema",
    up: (db) => {
      db.exec(`
        CREATE TABLE operator_advice (
          id TEXT PRIMARY KEY,
          session_id TEXT NULL,
          audience_entity_id TEXT NULL,
          text TEXT NOT NULL,
          created_at INTEGER NOT NULL,
          expires_at INTEGER NULL,
          consumed_at INTEGER NULL,
          consumed_by_turn_id TEXT NULL,
          canceled_at INTEGER NULL
        );

        CREATE INDEX IF NOT EXISTS idx_operator_advice_pending
          ON operator_advice (consumed_at, canceled_at, created_at);
      `);
    },
  },
] as const satisfies readonly Migration[];
