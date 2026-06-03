import type { Migration } from "../../storage/sqlite/index.js";

export const selfDecisionMigrations = [
  {
    id: 1,
    name: "self_decision_events_baseline",
    up: (db) => {
      db.exec(`
        CREATE TABLE self_decision_events (
          id TEXT PRIMARY KEY,
          occurred_at INTEGER NOT NULL,
          session_id TEXT NOT NULL,
          trigger_name TEXT NOT NULL,
          trigger_type TEXT NOT NULL CHECK (trigger_type IN ('trigger', 'condition')),
          source_event_id TEXT NOT NULL,
          fire_event_id TEXT NOT NULL,
          origin TEXT NOT NULL CHECK (origin IN ('autonomous')),
          decision_summary TEXT NOT NULL,
          turn_result_id TEXT NULL,
          source_stream_entry_ids TEXT NOT NULL,
          disclosure_class TEXT NOT NULL CHECK (disclosure_class IN ('self_private')),
          created_at INTEGER NOT NULL,
          updated_at INTEGER NOT NULL
        );
        -- source_event_id is provenance: some autonomy sources reuse dueEvent.id
        -- across genuine re-fires, so fire_event_id is the action-entry fire key.
        CREATE UNIQUE INDEX idx_self_decision_events_fire_event
        ON self_decision_events(fire_event_id);
        CREATE INDEX idx_self_decision_events_session_recent
        ON self_decision_events(session_id, occurred_at DESC);
      `);
    },
  },
] as const satisfies readonly Migration[];
