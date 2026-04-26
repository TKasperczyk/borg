import type { Migration } from "../../storage/sqlite/index.js";
import { tableHasColumn } from "../../storage/sqlite/migrations-utils.js";

export const episodicMigrations = [
  {
    id: 100,
    name: "create-episode-stats",
    up: `
      CREATE TABLE IF NOT EXISTS episode_stats (
        episode_id TEXT PRIMARY KEY,
        retrieval_count INTEGER NOT NULL DEFAULT 0,
        use_count INTEGER NOT NULL DEFAULT 0,
        last_retrieved INTEGER,
        win_rate REAL NOT NULL DEFAULT 0,
        tier TEXT NOT NULL DEFAULT 'T1',
        promoted_at INTEGER NOT NULL,
        promoted_from TEXT,
        gist TEXT,
        gist_generated_at INTEGER,
        last_decayed_at INTEGER
      )
    `,
  },
  {
    id: 101,
    name: "add-episode-archived-flag",
    up: (db) => {
      if (!tableHasColumn(db, "episode_stats", "archived")) {
        db.exec("ALTER TABLE episode_stats ADD COLUMN archived INTEGER NOT NULL DEFAULT 0");
      }
    },
  },
  {
    id: 102,
    name: "add-episode-valence-mean",
    up: (db) => {
      if (!tableHasColumn(db, "episode_stats", "valence_mean")) {
        db.exec("ALTER TABLE episode_stats ADD COLUMN valence_mean REAL NOT NULL DEFAULT 0");
      }
    },
  },
  {
    id: 103,
    name: "add-episode-heat-multiplier",
    up: (db) => {
      if (!tableHasColumn(db, "episode_stats", "heat_multiplier")) {
        db.exec("ALTER TABLE episode_stats ADD COLUMN heat_multiplier REAL NOT NULL DEFAULT 1");
      }
    },
  },
  {
    id: 104,
    name: "create-episode-hot-path-indexes",
    up: `
      CREATE TABLE IF NOT EXISTS episode_index (
        episode_id TEXT PRIMARY KEY,
        audience_entity_id TEXT,
        shared INTEGER NOT NULL DEFAULT 1 CHECK (shared IN (0, 1)),
        start_time INTEGER NOT NULL,
        end_time INTEGER NOT NULL,
        created_at INTEGER NOT NULL,
        updated_at INTEGER NOT NULL,
        retrieval_count INTEGER NOT NULL DEFAULT 0,
        win_rate REAL NOT NULL DEFAULT 0,
        last_retrieved INTEGER,
        tier TEXT NOT NULL DEFAULT 'T1',
        archived INTEGER NOT NULL DEFAULT 0 CHECK (archived IN (0, 1)),
        heat_multiplier REAL NOT NULL DEFAULT 1,
        heat_score REAL NOT NULL DEFAULT 0,
        FOREIGN KEY (episode_id) REFERENCES episode_stats(episode_id) ON DELETE CASCADE
      );

      CREATE TABLE IF NOT EXISTS episode_participants (
        episode_id TEXT NOT NULL,
        term TEXT NOT NULL,
        value TEXT NOT NULL,
        PRIMARY KEY (episode_id, term, value),
        FOREIGN KEY (episode_id) REFERENCES episode_index(episode_id) ON DELETE CASCADE
      );

      CREATE TABLE IF NOT EXISTS episode_tags (
        episode_id TEXT NOT NULL,
        term TEXT NOT NULL,
        value TEXT NOT NULL,
        PRIMARY KEY (episode_id, term, value),
        FOREIGN KEY (episode_id) REFERENCES episode_index(episode_id) ON DELETE CASCADE
      );

      CREATE TABLE IF NOT EXISTS episode_index_metadata (
        key TEXT PRIMARY KEY,
        value TEXT NOT NULL
      );

      CREATE INDEX IF NOT EXISTS idx_episode_index_recent
        ON episode_index (updated_at DESC, episode_id DESC)
        WHERE archived = 0;
      CREATE INDEX IF NOT EXISTS idx_episode_index_audience_recent
        ON episode_index (audience_entity_id, updated_at DESC, episode_id DESC)
        WHERE archived = 0;
      CREATE INDEX IF NOT EXISTS idx_episode_index_shared_recent
        ON episode_index (shared, updated_at DESC, episode_id DESC)
        WHERE archived = 0;
      CREATE INDEX IF NOT EXISTS idx_episode_index_heat
        ON episode_index (heat_score DESC, updated_at DESC, episode_id DESC)
        WHERE archived = 0;
      CREATE INDEX IF NOT EXISTS idx_episode_index_audience_heat
        ON episode_index (audience_entity_id, heat_score DESC, updated_at DESC, episode_id DESC)
        WHERE archived = 0;
      CREATE INDEX IF NOT EXISTS idx_episode_index_shared_heat
        ON episode_index (shared, heat_score DESC, updated_at DESC, episode_id DESC)
        WHERE archived = 0;
      CREATE INDEX IF NOT EXISTS idx_episode_index_audience_retrieved
        ON episode_index (audience_entity_id, last_retrieved DESC, updated_at DESC, episode_id DESC)
        WHERE archived = 0;
      CREATE INDEX IF NOT EXISTS idx_episode_index_time_start
        ON episode_index (start_time, updated_at DESC, episode_id DESC)
        WHERE archived = 0;
      CREATE INDEX IF NOT EXISTS idx_episode_index_time_end
        ON episode_index (end_time, updated_at DESC, episode_id DESC)
        WHERE archived = 0;
      CREATE INDEX IF NOT EXISTS idx_episode_participants_term
        ON episode_participants (term, episode_id);
      CREATE INDEX IF NOT EXISTS idx_episode_tags_term
        ON episode_tags (term, episode_id);
    `,
  },
] as const satisfies readonly Migration[];
