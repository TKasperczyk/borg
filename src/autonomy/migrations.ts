import type { Migration } from "../storage/sqlite/index.js";

export const autonomyMigrations = [
  {
    id: 1,
    name: "autonomy_baseline",
    up: (db) => {
      db.exec(`
        CREATE TABLE autonomy_wakes (
          id TEXT PRIMARY KEY,
          ts INTEGER NOT NULL,
          trigger_name TEXT NOT NULL CHECK (
            trigger_name IN (
              'commitment_expiring',
              'open_question_dormant',
              'scheduled_reflection',
              'scheduled_wake',
              'goal_followup_due',
              'executive_focus_due',
              'commitment_revoked',
              'mood_valence_drop',
              'open_question_urgency_bump'
            )
          ),
          condition_name TEXT CHECK (
            condition_name IS NULL OR condition_name IN (
              'commitment_revoked',
              'mood_valence_drop',
              'open_question_urgency_bump'
            )
          ),
          session_id TEXT,
          wake_source_type TEXT NOT NULL CHECK (wake_source_type IN ('trigger', 'condition'))
        );
        CREATE INDEX idx_autonomy_wakes_ts
          ON autonomy_wakes (ts);
        CREATE TABLE scheduled_wakes (
          id TEXT PRIMARY KEY,
          fire_at INTEGER NOT NULL,
          note TEXT NOT NULL,
          origin_session_id TEXT,
          status TEXT NOT NULL DEFAULT 'pending' CHECK (
            status IN ('pending', 'fired', 'cancelled')
          ),
          created_at INTEGER NOT NULL,
          updated_at INTEGER NOT NULL,
          fired_at INTEGER,
          cancelled_at INTEGER
        );
        CREATE INDEX idx_scheduled_wakes_due
          ON scheduled_wakes (status, fire_at);
      `);
    },
  },
] as const satisfies readonly Migration[];
