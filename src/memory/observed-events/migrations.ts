import type { Migration } from "../../storage/sqlite/index.js";

export const observedEventMigrations = [
  {
    id: 1,
    name: "observed_events_baseline",
    up: (db) => {
      db.exec(`
        CREATE TABLE observed_events (
          id TEXT PRIMARY KEY,
          occurred_at INTEGER NOT NULL,
          session_id TEXT NOT NULL,
          stance TEXT NOT NULL,
          taint TEXT NOT NULL,
          belief_effect TEXT NOT NULL,
          classification_kind TEXT NOT NULL,
          disclosure_class TEXT NOT NULL CHECK (disclosure_class IN ('social_observed', 'self_private')),
          interaction_text TEXT NOT NULL,
          recurrence_key TEXT NOT NULL,
          recurrence_count INTEGER NOT NULL DEFAULT 1,
          last_seen_at INTEGER NOT NULL,
          speaker_entity_id TEXT NULL,
          audience_entity_id TEXT NULL,
          source_entity_id TEXT NULL,
          source_stream_entry_ids TEXT NOT NULL,
          created_at INTEGER NOT NULL,
          updated_at INTEGER NOT NULL
        );
        CREATE UNIQUE INDEX idx_observed_events_recurrence
        ON observed_events(recurrence_key);
        CREATE INDEX idx_observed_events_session_recent
        ON observed_events(session_id, last_seen_at DESC);
        CREATE INDEX idx_observed_events_disclosure_recent
        ON observed_events(session_id, disclosure_class, last_seen_at DESC);
      `);
    },
  },
  {
    id: 2,
    name: "observed_events_fire_dedup_key",
    up: (db) => {
      db.exec(`
        ALTER TABLE observed_events ADD COLUMN fire_dedup_key TEXT NULL;
        CREATE UNIQUE INDEX idx_observed_events_fire_dedup
        ON observed_events(fire_dedup_key) WHERE fire_dedup_key IS NOT NULL;
      `);
    },
  },
  {
    id: 3,
    name: "observed_events_speaker_recent",
    up: (db) => {
      db.exec(`
        CREATE INDEX idx_observed_events_speaker_recent
        ON observed_events(speaker_entity_id, last_seen_at DESC);
      `);
    },
  },
  {
    id: 4,
    name: "observed_events_global_relevance",
    up: (db) => {
      db.exec(`
        CREATE INDEX idx_observed_events_global_recent
        ON observed_events(disclosure_class, last_seen_at DESC, id DESC);
        CREATE INDEX idx_observed_events_global_recurring
        ON observed_events(disclosure_class, recurrence_count DESC, last_seen_at DESC, id DESC);
      `);
    },
  },
] as const satisfies readonly Migration[];
