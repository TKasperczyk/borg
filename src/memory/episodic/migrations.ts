import type { Migration } from "../../storage/sqlite/index.js";

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
] as const satisfies readonly Migration[];
