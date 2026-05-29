import type { Migration } from "../../storage/sqlite/index.js";

export const promptOverrideMigrations = [
  {
    id: 1,
    name: "prompt_override_baseline",
    up: (db) => {
      db.exec(`
        CREATE TABLE prompt_overrides (
          prompt_key TEXT PRIMARY KEY,
          override_text TEXT NOT NULL,
          updated_at INTEGER NOT NULL
        );
      `);
    },
  },
] as const satisfies readonly Migration[];
