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
  {
    id: 2,
    name: "train_of_thought_append_journal",
    up: (db) => {
      db.exec(`
        CREATE TABLE train_of_thought_journal_entries (
          id INTEGER PRIMARY KEY AUTOINCREMENT,
          self_entity_id TEXT NOT NULL,
          text TEXT NOT NULL,
          disclosure_class TEXT NOT NULL CHECK (disclosure_class IN ('self_private')),
          created_at INTEGER NOT NULL,
          updated_at INTEGER NOT NULL,
          source_turn_id TEXT,
          marker_stream_entry_id TEXT
        );

        CREATE INDEX idx_train_of_thought_journal_latest
          ON train_of_thought_journal_entries (updated_at DESC, id DESC);

        INSERT INTO train_of_thought_journal_entries (
          self_entity_id,
          text,
          disclosure_class,
          created_at,
          updated_at
        )
        SELECT
          self_entity_id,
          text,
          disclosure_class,
          created_at,
          updated_at
        FROM train_of_thought
        WHERE text <> '';

        DROP TABLE train_of_thought;
      `);
    },
  },
] as const satisfies readonly Migration[];
