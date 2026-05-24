import type { Migration } from "../../storage/sqlite/index.js";
import { tableHasColumn, tableExists } from "../../storage/sqlite/migrations-utils.js";

export const sharedStateMigrations = [
  {
    id: 1,
    name: "decision_artifacts_initial_schema",
    up: `
      CREATE TABLE decision_artifacts (
        audience_entity_id TEXT PRIMARY KEY,
        record_version INTEGER NOT NULL DEFAULT 1,
        created_at INTEGER NOT NULL,
        updated_at INTEGER NOT NULL,
        last_compiled_at INTEGER NULL,
        last_compiled_stream_entry_id TEXT NULL
      );

      CREATE TABLE decision_artifact_entries (
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
        FOREIGN KEY (audience_entity_id)
          REFERENCES decision_artifacts(audience_entity_id)
          ON DELETE CASCADE,
        FOREIGN KEY (superseded_by_id)
          REFERENCES decision_artifact_entries(id)
          ON DELETE RESTRICT
      );

      CREATE INDEX IF NOT EXISTS idx_decision_artifact_entries_audience_rank
        ON decision_artifact_entries(audience_entity_id, rank ASC, created_at ASC);
      CREATE INDEX IF NOT EXISTS idx_decision_artifact_entries_kind
        ON decision_artifact_entries(kind);
      CREATE INDEX IF NOT EXISTS idx_decision_artifact_entries_superseded
        ON decision_artifact_entries(superseded_by_id);
      CREATE INDEX IF NOT EXISTS idx_decision_artifact_entries_audience_state_key
        ON decision_artifact_entries(audience_entity_id, state_key);
    `,
  },
  {
    id: 2,
    name: "decision_artifact_planning_state_canonicalizes",
    up: (db) => {
      if (
        !tableExists(db, "decision_artifact_entries") ||
        tableHasColumn(db, "decision_artifact_entries", "canonicalizes")
      ) {
        return;
      }

      db.exec(`
        ALTER TABLE decision_artifact_entries
          ADD COLUMN canonicalizes TEXT NOT NULL DEFAULT '{"goal_ids":[],"commitment_ids":[],"action_ids":[],"open_question_ids":[]}';
      `);
    },
  },
  {
    id: 3,
    name: "decision_artifact_entry_state_key",
    up: (db) => {
      if (
        !tableExists(db, "decision_artifact_entries") ||
        tableHasColumn(db, "decision_artifact_entries", "state_key")
      ) {
        return;
      }

      db.exec(`
        ALTER TABLE decision_artifact_entries
          ADD COLUMN state_key TEXT NULL;

        CREATE INDEX IF NOT EXISTS idx_decision_artifact_entries_audience_state_key
          ON decision_artifact_entries(audience_entity_id, state_key);
      `);
    },
  },
  {
    id: 4,
    name: "decision_artifact_live_lifecycle_kinds",
    up: (db) => {
      if (!tableExists(db, "decision_artifact_entries")) {
        return;
      }

      if (!tableHasColumn(db, "decision_artifact_entries", "state_key")) {
        db.exec(`
          ALTER TABLE decision_artifact_entries
            ADD COLUMN state_key TEXT NULL;
        `);
      }

      if (!tableHasColumn(db, "decision_artifact_entries", "canonicalizes")) {
        db.exec(`
          ALTER TABLE decision_artifact_entries
            ADD COLUMN canonicalizes TEXT NOT NULL DEFAULT '{"goal_ids":[],"commitment_ids":[],"action_ids":[],"open_question_ids":[]}';
        `);
      }

      db.exec(`
        CREATE TABLE decision_artifact_entries_next (
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
          canonicalizes TEXT NOT NULL DEFAULT '{"goal_ids":[],"commitment_ids":[],"action_ids":[],"open_question_ids":[]}',
          FOREIGN KEY (audience_entity_id)
            REFERENCES decision_artifacts(audience_entity_id)
            ON DELETE CASCADE,
          FOREIGN KEY (superseded_by_id)
            REFERENCES decision_artifact_entries_next(id)
            ON DELETE RESTRICT
        );

        INSERT INTO decision_artifact_entries_next (
          id, audience_entity_id, state_key, kind, text, owner_entity_id,
          provenance_stream_entry_ids, last_updated_stream_entry_ids,
          created_at, last_updated_at, superseded_by_id, rank,
          canonicalizes
        )
        SELECT
          id, audience_entity_id, state_key, kind, text, owner_entity_id,
          provenance_stream_entry_ids, last_updated_stream_entry_ids,
          created_at, last_updated_at, superseded_by_id, rank, canonicalizes
        FROM decision_artifact_entries;

        DROP TABLE decision_artifact_entries;
        ALTER TABLE decision_artifact_entries_next RENAME TO decision_artifact_entries;

        CREATE INDEX IF NOT EXISTS idx_decision_artifact_entries_audience_rank
          ON decision_artifact_entries(audience_entity_id, rank ASC, created_at ASC);
        CREATE INDEX IF NOT EXISTS idx_decision_artifact_entries_kind
          ON decision_artifact_entries(kind);
        CREATE INDEX IF NOT EXISTS idx_decision_artifact_entries_superseded
          ON decision_artifact_entries(superseded_by_id);
        CREATE INDEX IF NOT EXISTS idx_decision_artifact_entries_audience_state_key
          ON decision_artifact_entries(audience_entity_id, state_key);
      `);
    },
  },
  {
    id: 5,
    name: "decision_artifact_entries_last_updated_turn_global",
    up: (db) => {
      if (
        !tableExists(db, "decision_artifact_entries") ||
        tableHasColumn(db, "decision_artifact_entries", "last_updated_turn_global")
      ) {
        return;
      }

      db.exec(`
        ALTER TABLE decision_artifact_entries
          ADD COLUMN last_updated_turn_global INTEGER NULL;
      `);
    },
  },
] as const satisfies readonly Migration[];
