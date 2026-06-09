import type { Migration } from "../../storage/sqlite/index.js";
import { tableExists } from "../../storage/sqlite/migrations-utils.js";

export const sharedStateMigrations = [
  {
    id: 1,
    name: "shared_state_baseline",
    up: (db) => {
      db.exec(`
        CREATE TABLE shared_state_artifacts (
        audience_entity_id TEXT PRIMARY KEY,
        record_version INTEGER NOT NULL DEFAULT 1,
        created_at INTEGER NOT NULL,
        updated_at INTEGER NOT NULL,
        last_compiled_at INTEGER NULL,
        last_compiled_stream_entry_id TEXT NULL
      );
        CREATE TABLE "shared_state_entries" (
          id TEXT PRIMARY KEY,
          audience_entity_id TEXT NOT NULL,
          state_key TEXT NULL,
          kind TEXT NOT NULL CHECK (
            kind IN (
              'locked',
              'live',
              'low_salience_live',
              'dormant_live',
              'tentative',
              'invalidated',
              'pending'
            )
          ),
          text TEXT NOT NULL,
          owner_entity_id TEXT NULL,
          provenance_stream_entry_ids TEXT NOT NULL,
          last_updated_stream_entry_ids TEXT NOT NULL,
          created_at INTEGER NOT NULL,
          last_updated_at INTEGER NOT NULL,
          superseded_by_id TEXT NULL,
          rank INTEGER NOT NULL DEFAULT 0,
          canonicalizes TEXT NOT NULL DEFAULT '{"goal_ids":[],"commitment_ids":[],"action_ids":[],"open_question_ids":[]}', last_updated_turn_global INTEGER NULL,
          FOREIGN KEY (audience_entity_id)
            REFERENCES shared_state_artifacts(audience_entity_id)
            ON DELETE CASCADE,
          FOREIGN KEY (superseded_by_id)
            REFERENCES "shared_state_entries"(id)
            ON DELETE RESTRICT
        );
        CREATE INDEX idx_shared_state_entries_audience_rank
          ON shared_state_entries(audience_entity_id, rank ASC, created_at ASC);
        CREATE INDEX idx_shared_state_entries_audience_state_key
          ON shared_state_entries(audience_entity_id, state_key);
        CREATE INDEX idx_shared_state_entries_kind
          ON shared_state_entries(kind);
        CREATE INDEX idx_shared_state_entries_superseded
          ON shared_state_entries(superseded_by_id);
      `);
    },
  },
  {
    id: 2,
    name: "shared_state_table_names",
    up: (db) => {
      db.exec(`
        DROP INDEX IF EXISTS idx_decision_artifact_entries_audience_rank;
        DROP INDEX IF EXISTS idx_decision_artifact_entries_audience_state_key;
        DROP INDEX IF EXISTS idx_decision_artifact_entries_kind;
        DROP INDEX IF EXISTS idx_decision_artifact_entries_superseded;
      `);

      if (
        tableExists(db, "decision_artifacts") &&
        !tableExists(db, "shared_state_artifacts")
      ) {
        db.exec("ALTER TABLE decision_artifacts RENAME TO shared_state_artifacts");
      }

      if (
        tableExists(db, "decision_artifact_entries") &&
        !tableExists(db, "shared_state_entries")
      ) {
        db.exec("ALTER TABLE decision_artifact_entries RENAME TO shared_state_entries");
      }

      if (tableExists(db, "shared_state_entries")) {
        db.exec(`
          CREATE INDEX IF NOT EXISTS idx_shared_state_entries_audience_rank
            ON shared_state_entries(audience_entity_id, rank ASC, created_at ASC);
          CREATE INDEX IF NOT EXISTS idx_shared_state_entries_audience_state_key
            ON shared_state_entries(audience_entity_id, state_key);
          CREATE INDEX IF NOT EXISTS idx_shared_state_entries_kind
            ON shared_state_entries(kind);
          CREATE INDEX IF NOT EXISTS idx_shared_state_entries_superseded
            ON shared_state_entries(superseded_by_id);
        `);
      }
    },
  },
] as const satisfies readonly Migration[];
