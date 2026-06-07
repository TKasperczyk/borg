import type { Migration } from "../../storage/sqlite/index.js";

export const trainOfThoughtMigrations = [
  {
    id: 1,
    name: "train_of_thought_baseline",
    up: (db) => {
      db.exec(`
        CREATE TABLE train_of_thought (
          id INTEGER PRIMARY KEY CHECK (id = 1),
          self_entity_id TEXT NOT NULL,
          text TEXT NOT NULL,
          disclosure_class TEXT NOT NULL CHECK (disclosure_class IN ('self_private')),
          created_at INTEGER NOT NULL,
          updated_at INTEGER NOT NULL
        );
      `);
    },
  },
] as const satisfies readonly Migration[];
