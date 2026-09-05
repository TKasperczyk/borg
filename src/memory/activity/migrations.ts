import type { Migration } from "../../storage/sqlite/index.js";

export const activityMigrations = [
  {
    id: 1,
    name: "activity_baseline",
    up: (db) => {
      db.exec(`
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
        CREATE UNIQUE INDEX idx_activity_events_kind_source
        ON activity_events(kind, source_stream_entry_ids);
        CREATE INDEX idx_activity_events_recent
        ON activity_events(status, occurred_at DESC);
        CREATE INDEX idx_activity_events_session_recent
        ON activity_events(session_id, status, occurred_at DESC);
        CREATE INDEX idx_activity_events_turn
        ON activity_events(session_id, turn_id, status);
      `);
    },
  },
  {
    id: 2,
    name: "lived_experience_day_summaries",
    up: (db) => {
      db.exec(`
        CREATE TABLE IF NOT EXISTS lived_experience_day_summaries (
          id TEXT PRIMARY KEY,
          self_entity_id TEXT NOT NULL,
          utc_day TEXT NOT NULL,
          day_start_ms INTEGER NOT NULL,
          day_end_ms INTEGER NOT NULL,
          gist TEXT NOT NULL,
          salience REAL NOT NULL,
          counts_snapshot TEXT NOT NULL,
          source_episode_ids TEXT NOT NULL,
          source_stream_entry_ids TEXT NOT NULL,
          disclosure_label TEXT NOT NULL,
          provenance_kind TEXT NOT NULL,
          provenance_episode_ids TEXT NOT NULL,
          provenance_process TEXT NULL,
          source_run_id TEXT NULL,
          created_at INTEGER NOT NULL,
          updated_at INTEGER NOT NULL,
          UNIQUE(self_entity_id, utc_day)
        );
        CREATE INDEX IF NOT EXISTS idx_lived_experience_day_summaries_entity_day_start
          ON lived_experience_day_summaries(self_entity_id, day_start_ms DESC);
      `);
    },
  },
  {
    id: 3,
    name: "activity_audience_observation_indexes",
    up: (db) => {
      db.exec(`
        CREATE INDEX IF NOT EXISTS idx_activity_events_speaker_audience
          ON activity_events(speaker_entity_id, status, kind, audience_entity_id);
        CREATE INDEX IF NOT EXISTS idx_activity_events_audience_recent
          ON activity_events(audience_entity_id, status, occurred_at DESC);
      `);
    },
  },
] as const satisfies readonly Migration[];
