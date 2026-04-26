import type { Migration } from "../storage/sqlite/index.js";

export const autonomyMigrations = [
  {
    id: 270,
    name: "create-autonomy-wakes",
    up: `
      CREATE TABLE IF NOT EXISTS autonomy_wakes (
        id TEXT PRIMARY KEY,
        ts INTEGER NOT NULL,
        trigger_name TEXT NOT NULL CHECK (
          trigger_name IN (
            'commitment_expiring',
            'open_question_dormant',
            'scheduled_reflection',
            'goal_followup_due',
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

      CREATE INDEX IF NOT EXISTS idx_autonomy_wakes_ts
        ON autonomy_wakes (ts);
    `,
  },
] as const satisfies readonly Migration[];
