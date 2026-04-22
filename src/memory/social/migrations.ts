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

      const legacyProfiles = db
        .prepare(
          `
            SELECT entity_id, trust, attachment, interaction_count, created_at, updated_at
            FROM social_profiles
            ORDER BY updated_at ASC, created_at ASC, entity_id ASC
          `,
        )
        .all() as Array<Record<string, unknown>>;
      const insertBaseline = db.prepare(
        `
          INSERT INTO social_events (
            entity_id, ts, kind, provenance_kind, provenance_episode_ids, provenance_process,
            trust_delta, attachment_delta, interaction_delta, valence
          ) VALUES (?, ?, 'baseline', 'system', '[]', NULL, ?, ?, ?, NULL)
        `,
      );

      for (const profile of legacyProfiles) {
        insertBaseline.run(
          profile.entity_id,
          Number(profile.updated_at ?? profile.created_at ?? 0),
          Number(profile.trust),
          Number(profile.attachment),
          Number(profile.interaction_count),
        );
      }
    },
  },
] as const satisfies readonly Migration[];
