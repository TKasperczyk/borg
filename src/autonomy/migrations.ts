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
  {
    id: 2,
    name: "autonomy_wakes_source_category",
    up: (db) => {
      db.exec(`
        ALTER TABLE autonomy_wakes
        ADD COLUMN source_category TEXT NOT NULL DEFAULT 'operational' CHECK (
          source_category IN ('contemplative', 'operational')
        );
      `);
    },
  },
  {
    id: 3,
    name: "autonomy_wakes_outcome",
    up: (db) => {
      db.exec(`
        ALTER TABLE autonomy_wakes
        ADD COLUMN outcome TEXT CHECK (
          outcome IS NULL OR outcome IN ('headway', 'silent', 'error', 'busy')
        );
      `);
    },
  },
  {
    id: 4,
    name: "autonomy_wakes_outcome_detail",
    up: (db) => {
      // The scheduler already formats the failure that ended a wake -- it writes
      // it into the stream as `autonomous_action.outcome_summary` -- and then
      // dropped it on the way to this table, so `outcome='error'` was a count
      // with its own discriminator computed and discarded one line earlier.
      // Nullable by construction: rows written before this column existed have
      // no detail and must stay distinguishable from rows whose outcome carries
      // none, which is why nothing backfills a placeholder here.
      db.exec(`
        ALTER TABLE autonomy_wakes
        ADD COLUMN outcome_detail TEXT;
      `);
    },
  },
  {
    id: 5,
    name: "autonomy_wakes_interrupted_outcome",
    up: (db) => {
      db.exec(`
        CREATE TABLE autonomy_wakes__next (
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
          wake_source_type TEXT NOT NULL CHECK (wake_source_type IN ('trigger', 'condition')),
          source_category TEXT NOT NULL DEFAULT 'operational' CHECK (
            source_category IN ('contemplative', 'operational')
          ),
          outcome TEXT CHECK (
            outcome IS NULL OR outcome IN ('headway', 'silent', 'error', 'busy', 'interrupted')
          ),
          outcome_detail TEXT
        );

        INSERT INTO autonomy_wakes__next (
          id, ts, trigger_name, condition_name, session_id, wake_source_type, source_category,
          outcome, outcome_detail
        )
        SELECT
          id, ts, trigger_name, condition_name, session_id, wake_source_type, source_category,
          outcome, outcome_detail
        FROM autonomy_wakes;

        DROP TABLE autonomy_wakes;
        ALTER TABLE autonomy_wakes__next RENAME TO autonomy_wakes;

        CREATE INDEX idx_autonomy_wakes_ts
          ON autonomy_wakes (ts);
      `);
    },
  },
  {
    id: 6,
    name: "autonomy_wakes_selected_goal",
    up: (db) => {
      db.exec(`
        ALTER TABLE autonomy_wakes
        ADD COLUMN selected_goal_id TEXT;
      `);
    },
  },
  {
    id: 7,
    name: "autonomy_wakes_headway_bases",
    up: (db) => {
      // Nullable with no backfill: older rows have only their legacy display
      // detail, which cannot be losslessly reconstructed into structural bases.
      db.exec(`
        ALTER TABLE autonomy_wakes
        ADD COLUMN headway_bases_json TEXT;
      `);
    },
  },
  {
    id: 8,
    name: "autonomy_wakes_execution_counts",
    up: (db) => {
      // Nullable with no backfill: the trace held these facts for older wakes,
      // but migrations must not parse trace files or invent zeroes for them.
      db.exec(`
        ALTER TABLE autonomy_wakes
        ADD COLUMN finalizer_rounds INTEGER CHECK (
          finalizer_rounds IS NULL OR (
            typeof(finalizer_rounds) = 'integer' AND finalizer_rounds >= 0
          )
        );

        ALTER TABLE autonomy_wakes
        ADD COLUMN stall_retries INTEGER CHECK (
          stall_retries IS NULL OR (
            typeof(stall_retries) = 'integer' AND stall_retries >= 0
          )
        );
      `);
    },
  },
] as const satisfies readonly Migration[];
