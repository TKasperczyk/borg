import type { Migration } from "../../storage/sqlite/index.js";

export const promptSurfaceHistoryMigrations = [
  {
    id: 1,
    name: "prompt_surface_history_baseline",
    up: (db) => {
      db.exec(`
        CREATE TABLE prompt_surface_snapshots (
          hash TEXT PRIMARY KEY,
          observed_at INTEGER NOT NULL,
          block_ids TEXT NOT NULL,
          surface_placements TEXT NOT NULL
        );

        CREATE INDEX idx_prompt_surface_snapshots_observed_at
        ON prompt_surface_snapshots(observed_at DESC, hash ASC);

        CREATE TABLE prompt_surface_changes (
          id INTEGER PRIMARY KEY AUTOINCREMENT,
          observed_at INTEGER NOT NULL,
          from_hash TEXT NULL,
          to_hash TEXT NOT NULL,
          added_block_ids TEXT NOT NULL,
          removed_block_ids TEXT NOT NULL,
          added_surface_placements TEXT NOT NULL,
          removed_surface_placements TEXT NOT NULL,
          UNIQUE(to_hash),
          FOREIGN KEY(from_hash) REFERENCES prompt_surface_snapshots(hash),
          FOREIGN KEY(to_hash) REFERENCES prompt_surface_snapshots(hash)
        );

        CREATE INDEX idx_prompt_surface_changes_observed_at
        ON prompt_surface_changes(observed_at DESC, id DESC);

        CREATE INDEX idx_prompt_surface_changes_from_hash
        ON prompt_surface_changes(from_hash);
      `);
    },
  },
] as const satisfies readonly Migration[];
