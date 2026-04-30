import type { Migration } from "../../storage/sqlite/index.js";

export const affectiveMigrations = [
  {
    id: 1,
    name: "affective_initial_schema",
    up: (db) => {
      db.exec(`
        CREATE TABLE mood_state (
          session_id TEXT PRIMARY KEY,
          valence REAL NOT NULL,
          arousal REAL NOT NULL,
          updated_at INTEGER NOT NULL,
          half_life_hours REAL NOT NULL,
          recent_triggers TEXT NOT NULL
        );

        CREATE TABLE mood_history (
          id INTEGER PRIMARY KEY AUTOINCREMENT,
          session_id TEXT NOT NULL,
          ts INTEGER NOT NULL,
          valence REAL NOT NULL,
          arousal REAL NOT NULL,
          trigger_episode_id TEXT,
          trigger_reason TEXT,
          provenance_kind TEXT,
          provenance_episode_ids TEXT,
          provenance_process TEXT
        );

        CREATE INDEX IF NOT EXISTS idx_mood_history_session_ts
          ON mood_history (session_id, ts DESC);
      `);
    },
  },
] as const satisfies readonly Migration[];
