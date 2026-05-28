import type { Migration } from "../../storage/sqlite/index.js";

export const activityMigrations = [
  {
    id: 1,
    name: "activity_events_initial_schema",
    up: `
      CREATE TABLE activity_events (
        id TEXT PRIMARY KEY,
        kind TEXT NOT NULL CHECK (
          kind IN ('user_contact', 'borg_replied', 'turn_completed')
        ),
        occurred_at INTEGER NOT NULL,
        session_id TEXT NOT NULL,
        turn_id TEXT NULL,
        speaker_entity_id TEXT NULL,
        actor_entity_id TEXT NULL,
        audience_entity_id TEXT NULL,
        participant_entity_ids TEXT NOT NULL,
        source_stream_entry_ids TEXT NOT NULL,
        status TEXT NOT NULL CHECK (status IN ('active', 'inactive')),
        created_at INTEGER NOT NULL,
        updated_at INTEGER NOT NULL
      );

      CREATE INDEX IF NOT EXISTS idx_activity_events_recent
        ON activity_events(status, occurred_at DESC);

      CREATE INDEX IF NOT EXISTS idx_activity_events_session_recent
        ON activity_events(session_id, status, occurred_at DESC);

      CREATE INDEX IF NOT EXISTS idx_activity_events_turn
        ON activity_events(session_id, turn_id, status);

      CREATE UNIQUE INDEX IF NOT EXISTS idx_activity_events_kind_source
        ON activity_events(kind, source_stream_entry_ids);
    `,
  },
] as const satisfies readonly Migration[];
