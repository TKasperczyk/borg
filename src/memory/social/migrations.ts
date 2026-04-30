import type { Migration } from "../../storage/sqlite/index.js";

export const socialMigrations = [
  {
    id: 171,
    name: "create-social-profiles",
    up: `
      CREATE TABLE IF NOT EXISTS social_profiles (
        entity_id TEXT PRIMARY KEY,
        trust REAL NOT NULL DEFAULT 0.5,
        attachment REAL NOT NULL DEFAULT 0.0,
        communication_style TEXT,
        shared_history_summary TEXT,
        last_interaction_at INTEGER,
        interaction_count INTEGER NOT NULL DEFAULT 0,
        commitment_count INTEGER NOT NULL DEFAULT 0,
        sentiment_history TEXT NOT NULL DEFAULT '[]',
        notes TEXT,
        created_at INTEGER NOT NULL,
        updated_at INTEGER NOT NULL
      )
    `,
  },
  {
    id: 240,
    name: "create-social-events",
    up: (db) => {
      db.exec(`
        CREATE TABLE IF NOT EXISTS social_events (
          id INTEGER PRIMARY KEY AUTOINCREMENT,
          entity_id TEXT NOT NULL,
          ts INTEGER NOT NULL,
          kind TEXT NOT NULL CHECK (
            kind IN ('interaction', 'trust_adjustment', 'baseline')
          ),
          provenance_kind TEXT NOT NULL CHECK (
            provenance_kind IN ('episodes', 'manual', 'system', 'offline')
          ),
          provenance_episode_ids TEXT NOT NULL DEFAULT '[]',
          provenance_process TEXT,
          trust_delta REAL NOT NULL DEFAULT 0,
          attachment_delta REAL NOT NULL DEFAULT 0,
          interaction_delta INTEGER NOT NULL DEFAULT 0,
          valence REAL
        );
        CREATE INDEX IF NOT EXISTS idx_social_events_entity_ts
          ON social_events (entity_id, ts DESC, id DESC);
      `);
    },
  },
  {
    id: 241,
    name: "allow-online-social-event-provenance",
    up: (db) => {
      db.exec(`
        CREATE TABLE social_events__next (
          id INTEGER PRIMARY KEY AUTOINCREMENT,
          entity_id TEXT NOT NULL,
          ts INTEGER NOT NULL,
          kind TEXT NOT NULL CHECK (
            kind IN ('interaction', 'trust_adjustment', 'baseline')
          ),
          provenance_kind TEXT NOT NULL CHECK (
            provenance_kind IN ('episodes', 'manual', 'system', 'offline', 'online')
          ),
          provenance_episode_ids TEXT NOT NULL DEFAULT '[]',
          provenance_process TEXT,
          trust_delta REAL NOT NULL DEFAULT 0,
          attachment_delta REAL NOT NULL DEFAULT 0,
          interaction_delta INTEGER NOT NULL DEFAULT 0,
          valence REAL
        );

        INSERT INTO social_events__next (
          id,
          entity_id,
          ts,
          kind,
          provenance_kind,
          provenance_episode_ids,
          provenance_process,
          trust_delta,
          attachment_delta,
          interaction_delta,
          valence
        )
        SELECT
          id,
          entity_id,
          ts,
          kind,
          provenance_kind,
          provenance_episode_ids,
          provenance_process,
          trust_delta,
          attachment_delta,
          interaction_delta,
          valence
        FROM social_events;

        DROP TABLE social_events;
        ALTER TABLE social_events__next RENAME TO social_events;

        CREATE INDEX IF NOT EXISTS idx_social_events_entity_ts
          ON social_events (entity_id, ts DESC, id DESC);
      `);
    },
  },
] as const satisfies readonly Migration[];
