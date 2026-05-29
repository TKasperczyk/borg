import type { Migration } from "../../storage/sqlite/index.js";

export const actionMigrations = [
  {
    id: 1,
    name: "action_baseline",
    up: (db) => {
      db.exec(`
        CREATE TABLE action_records (
          id TEXT PRIMARY KEY,
          description TEXT NOT NULL,
          actor TEXT NOT NULL,
          audience_entity_id TEXT NULL,
          goal_id TEXT NULL,
          open_question_id TEXT NULL,
          state TEXT NOT NULL CHECK (
            state IN (
              'considering',
              'committed_to_do',
              'scheduled',
              'completed',
              'not_done',
              'expired',
              'archived',
              'unknown'
            )
          ),
          confidence REAL NOT NULL CHECK (confidence >= 0 AND confidence <= 1),
          provenance_episode_ids TEXT NOT NULL,
          provenance_stream_entry_ids TEXT NOT NULL,
          created_at INTEGER NOT NULL,
          updated_at INTEGER NOT NULL,
          considering_at INTEGER NULL,
          committed_at INTEGER NULL,
          scheduled_at INTEGER NULL,
          completed_at INTEGER NULL,
          not_done_at INTEGER NULL,
          expired_at INTEGER NULL,
          archived_at INTEGER NULL,
          unknown_at INTEGER NULL,
          canonicalized_by_artifact_entry_id TEXT NULL,
          session_scope TEXT NULL CHECK (
            session_scope IS NULL OR session_scope IN ('current_session', 'next_session')
          ),
          session_anchor_id TEXT NULL,
          last_referenced_at_ms INTEGER NULL,
          last_referenced_turn_counter INTEGER NULL
        , last_referenced_turn_global INTEGER NULL);
        CREATE INDEX action_records_actor_idx
          ON action_records(actor);
        CREATE INDEX action_records_audience_entity_idx
          ON action_records(audience_entity_id);
        CREATE INDEX action_records_goal_idx
          ON action_records(goal_id);
        CREATE INDEX action_records_last_referenced_turn_global_idx
            ON action_records(last_referenced_turn_global);
        CREATE INDEX action_records_last_referenced_turn_idx
          ON action_records(last_referenced_turn_counter);
        CREATE INDEX action_records_open_question_idx
          ON action_records(open_question_id);
        CREATE INDEX action_records_session_anchor_scope_idx
          ON action_records(session_anchor_id, session_scope);
        CREATE INDEX action_records_session_scope_idx
          ON action_records(session_scope);
        CREATE INDEX action_records_state_idx
          ON action_records(state);
        CREATE INDEX action_records_updated_idx
          ON action_records(updated_at DESC, id ASC);
        CREATE TABLE action_lifecycle_turn_counter (
          id TEXT PRIMARY KEY CHECK (id = 'global'),
          value INTEGER NOT NULL CHECK (value >= 0)
        );
      `);
    },
  },
] as const satisfies readonly Migration[];
