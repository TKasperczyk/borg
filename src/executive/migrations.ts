import type { Migration } from "../storage/sqlite/index.js";

export const executiveMigrations = [
  {
    id: 1,
    name: "executive_initial_schema",
    up: (db) => {
      db.exec(`
        CREATE TABLE executive_steps (
          id TEXT PRIMARY KEY,
          goal_id TEXT NOT NULL,
          description TEXT NOT NULL,
          status TEXT NOT NULL CHECK (
            status IN ('queued', 'doing', 'done', 'blocked', 'abandoned')
          ),
          kind TEXT NOT NULL CHECK (
            kind IN ('think', 'ask_user', 'research', 'act', 'wait')
          ),
          due_at INTEGER,
          last_attempt_ts INTEGER,
          created_at INTEGER NOT NULL,
          updated_at INTEGER NOT NULL,
          provenance_kind TEXT NOT NULL CHECK (
            provenance_kind IN ('episodes', 'manual', 'system', 'offline', 'online')
          ),
          provenance_episode_ids TEXT NOT NULL,
          provenance_process TEXT,
          FOREIGN KEY (goal_id) REFERENCES goals(id) ON DELETE CASCADE
        );

        CREATE INDEX IF NOT EXISTS idx_executive_steps_goal_status
          ON executive_steps (goal_id, status, due_at, created_at, id);
      `);
    },
  },
] as const satisfies readonly Migration[];
