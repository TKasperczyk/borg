import { StorageError } from "../../util/errors.js";
import type { Migration, SqliteDatabase } from "../../storage/sqlite/index.js";
import { tableExists, tableHasColumn } from "../../storage/sqlite/migrations-utils.js";

function ensureActionRecordsCanBeCreated(db: SqliteDatabase): void {
  if (!tableExists(db, "action_records")) {
    return;
  }

  const row = db.prepare("SELECT COUNT(*) AS count FROM action_records").get() as
    | { count: number }
    | undefined;
  const count = Number(row?.count ?? 0);

  if (count > 0) {
    throw new StorageError("Existing action_records table is non-empty", {
      code: "ACTION_RECORDS_EXISTING_TABLE_NON_EMPTY",
    });
  }

  db.exec("DROP TABLE action_records");
}

export const actionMigrations = [
  {
    id: 1,
    name: "actions_initial_schema",
    up: (db) => {
      ensureActionRecordsCanBeCreated(db);

      db.exec(`
        CREATE TABLE action_records (
          id TEXT PRIMARY KEY,
          description TEXT NOT NULL,
          actor TEXT NOT NULL,
          audience_entity_id TEXT NULL,
          goal_id TEXT NULL,
          open_question_id TEXT NULL,
          state TEXT NOT NULL CHECK (
            state IN (
              'considering',
              'committed_to_do',
              'scheduled',
              'completed',
              'not_done',
              'expired',
              'archived',
              'unknown'
            )
          ),
          confidence REAL NOT NULL CHECK (confidence >= 0 AND confidence <= 1),
          provenance_episode_ids TEXT NOT NULL,
          provenance_stream_entry_ids TEXT NOT NULL,
          created_at INTEGER NOT NULL,
          updated_at INTEGER NOT NULL,
          considering_at INTEGER NULL,
          committed_at INTEGER NULL,
          scheduled_at INTEGER NULL,
          completed_at INTEGER NULL,
          not_done_at INTEGER NULL,
          expired_at INTEGER NULL,
          archived_at INTEGER NULL,
          unknown_at INTEGER NULL,
          canonicalized_by_artifact_entry_id TEXT NULL,
          session_scope TEXT NULL CHECK (
            session_scope IS NULL OR session_scope IN ('current_session', 'next_session')
          ),
          session_anchor_id TEXT NULL,
          last_referenced_at_ms INTEGER NULL,
          last_referenced_turn_counter INTEGER NULL,
          last_referenced_turn_global INTEGER NULL
        );

        CREATE INDEX IF NOT EXISTS action_records_state_idx
          ON action_records(state);
        CREATE INDEX IF NOT EXISTS action_records_actor_idx
          ON action_records(actor);
        CREATE INDEX IF NOT EXISTS action_records_audience_entity_idx
          ON action_records(audience_entity_id);
        CREATE INDEX IF NOT EXISTS action_records_goal_idx
          ON action_records(goal_id);
        CREATE INDEX IF NOT EXISTS action_records_open_question_idx
          ON action_records(open_question_id);
        CREATE INDEX IF NOT EXISTS action_records_updated_idx
          ON action_records(updated_at DESC, id ASC);
        CREATE INDEX IF NOT EXISTS action_records_session_scope_idx
          ON action_records(session_scope);
        CREATE INDEX IF NOT EXISTS action_records_session_anchor_scope_idx
          ON action_records(session_anchor_id, session_scope);
        CREATE INDEX IF NOT EXISTS action_records_last_referenced_turn_idx
          ON action_records(last_referenced_turn_counter);
        CREATE INDEX IF NOT EXISTS action_records_last_referenced_turn_global_idx
          ON action_records(last_referenced_turn_global);
        CREATE TABLE IF NOT EXISTS action_lifecycle_turn_counter (
          id TEXT PRIMARY KEY CHECK (id = 'global'),
          value INTEGER NOT NULL CHECK (value >= 0)
        );
        INSERT OR IGNORE INTO action_lifecycle_turn_counter (id, value)
          VALUES ('global', 0);
      `);
    },
  },
  {
    id: 2,
    name: "action_records_intent_links",
    up: (db) => {
      if (!tableExists(db, "action_records")) {
        return;
      }

      if (!tableHasColumn(db, "action_records", "goal_id")) {
        db.exec(`
          ALTER TABLE action_records
            ADD COLUMN goal_id TEXT NULL;
        `);
      }

      if (!tableHasColumn(db, "action_records", "open_question_id")) {
        db.exec(`
          ALTER TABLE action_records
            ADD COLUMN open_question_id TEXT NULL;
        `);
      }

      db.exec(`
        CREATE INDEX IF NOT EXISTS action_records_goal_idx
          ON action_records(goal_id);
        CREATE INDEX IF NOT EXISTS action_records_open_question_idx
          ON action_records(open_question_id);
      `);
    },
  },
  {
    id: 3,
    name: "action_records_artifact_backref",
    up: (db) => {
      if (!tableExists(db, "action_records")) {
        return;
      }

      if (!tableHasColumn(db, "action_records", "canonicalized_by_artifact_entry_id")) {
        db.exec(`
          ALTER TABLE action_records
            ADD COLUMN canonicalized_by_artifact_entry_id TEXT NULL;
        `);
      }
    },
  },
  {
    id: 4,
    name: "action_records_scope_and_aging",
    up: (db) => {
      if (!tableExists(db, "action_records")) {
        return;
      }

      if (
        tableHasColumn(db, "action_records", "session_scope") &&
        tableHasColumn(db, "action_records", "last_referenced_at_ms") &&
        tableHasColumn(db, "action_records", "last_referenced_turn_counter") &&
        tableHasColumn(db, "action_records", "expired_at") &&
        tableHasColumn(db, "action_records", "archived_at")
      ) {
        return;
      }

      db.exec(`
        CREATE TABLE action_records_next (
          id TEXT PRIMARY KEY,
          description TEXT NOT NULL,
          actor TEXT NOT NULL,
          audience_entity_id TEXT NULL,
          goal_id TEXT NULL,
          open_question_id TEXT NULL,
          state TEXT NOT NULL CHECK (
            state IN (
              'considering',
              'committed_to_do',
              'scheduled',
              'completed',
              'not_done',
              'expired',
              'archived',
              'unknown'
            )
          ),
          confidence REAL NOT NULL CHECK (confidence >= 0 AND confidence <= 1),
          provenance_episode_ids TEXT NOT NULL,
          provenance_stream_entry_ids TEXT NOT NULL,
          created_at INTEGER NOT NULL,
          updated_at INTEGER NOT NULL,
          considering_at INTEGER NULL,
          committed_at INTEGER NULL,
          scheduled_at INTEGER NULL,
          completed_at INTEGER NULL,
          not_done_at INTEGER NULL,
          expired_at INTEGER NULL,
          archived_at INTEGER NULL,
          unknown_at INTEGER NULL,
          canonicalized_by_artifact_entry_id TEXT NULL,
          session_scope TEXT NULL CHECK (
            session_scope IS NULL OR session_scope IN ('current_session', 'next_session')
          ),
          session_anchor_id TEXT NULL,
          last_referenced_at_ms INTEGER NULL,
          last_referenced_turn_counter INTEGER NULL,
          last_referenced_turn_global INTEGER NULL
        );

        INSERT INTO action_records_next (
          id, description, actor, audience_entity_id, goal_id, open_question_id, state, confidence,
          provenance_episode_ids, provenance_stream_entry_ids, created_at, updated_at,
          considering_at, committed_at, scheduled_at, completed_at, not_done_at, expired_at,
          archived_at, unknown_at, canonicalized_by_artifact_entry_id, session_scope,
          session_anchor_id, last_referenced_at_ms, last_referenced_turn_counter,
          last_referenced_turn_global
        )
        SELECT
          id,
          description,
          actor,
          audience_entity_id,
          goal_id,
          open_question_id,
          state,
          confidence,
          provenance_episode_ids,
          provenance_stream_entry_ids,
          created_at,
          updated_at,
          considering_at,
          committed_at,
          scheduled_at,
          completed_at,
          not_done_at,
          NULL,
          NULL,
          unknown_at,
          canonicalized_by_artifact_entry_id,
          NULL,
          NULL,
          updated_at,
          NULL,
          NULL
        FROM action_records;

        DROP TABLE action_records;
        ALTER TABLE action_records_next RENAME TO action_records;

        CREATE INDEX IF NOT EXISTS action_records_state_idx
          ON action_records(state);
        CREATE INDEX IF NOT EXISTS action_records_actor_idx
          ON action_records(actor);
        CREATE INDEX IF NOT EXISTS action_records_audience_entity_idx
          ON action_records(audience_entity_id);
        CREATE INDEX IF NOT EXISTS action_records_goal_idx
          ON action_records(goal_id);
        CREATE INDEX IF NOT EXISTS action_records_open_question_idx
          ON action_records(open_question_id);
        CREATE INDEX IF NOT EXISTS action_records_updated_idx
          ON action_records(updated_at DESC, id ASC);
        CREATE INDEX IF NOT EXISTS action_records_session_scope_idx
          ON action_records(session_scope);
        CREATE INDEX IF NOT EXISTS action_records_session_anchor_scope_idx
          ON action_records(session_anchor_id, session_scope);
        CREATE INDEX IF NOT EXISTS action_records_last_referenced_turn_idx
          ON action_records(last_referenced_turn_counter);
        CREATE INDEX IF NOT EXISTS action_records_last_referenced_turn_global_idx
          ON action_records(last_referenced_turn_global);
      `);
    },
  },
  {
    id: 5,
    name: "action_records_session_anchor",
    up: (db) => {
      if (!tableExists(db, "action_records")) {
        return;
      }

      if (!tableHasColumn(db, "action_records", "session_anchor_id")) {
        db.exec(`
          ALTER TABLE action_records
            ADD COLUMN session_anchor_id TEXT NULL;
        `);
      }

      db.exec(`
        CREATE INDEX IF NOT EXISTS action_records_session_anchor_scope_idx
          ON action_records(session_anchor_id, session_scope);
      `);
    },
  },
  {
    id: 6,
    name: "action_records_global_lifecycle_turn",
    up: (db) => {
      if (tableExists(db, "action_records")) {
        if (!tableHasColumn(db, "action_records", "last_referenced_turn_global")) {
          db.exec(`
            ALTER TABLE action_records
              ADD COLUMN last_referenced_turn_global INTEGER NULL;
          `);
        }

        const row = db
          .prepare(
            `
              SELECT COALESCE(MAX(last_referenced_turn_counter), 0) AS value
              FROM action_records
            `,
          )
          .get() as { value: number } | undefined;
        const migrationWatermark = Math.max(0, Math.floor(Number(row?.value ?? 0)));

        // Legacy turn counters were session-local. Preserve ordering going forward by treating
        // every legacy reference as touched at the migration boundary instead of copying
        // incomparable per-session values row-by-row.
        db.prepare(
          `
            UPDATE action_records
            SET last_referenced_turn_global = ?
            WHERE last_referenced_turn_global IS NULL
              AND last_referenced_turn_counter IS NOT NULL
          `,
        ).run(migrationWatermark);

        db.exec(`
          CREATE INDEX IF NOT EXISTS action_records_last_referenced_turn_global_idx
            ON action_records(last_referenced_turn_global);
        `);
      }

      db.exec(`
        CREATE TABLE IF NOT EXISTS action_lifecycle_turn_counter (
          id TEXT PRIMARY KEY CHECK (id = 'global'),
          value INTEGER NOT NULL CHECK (value >= 0)
        );
      `);

      if (tableExists(db, "action_records")) {
        db.exec(`
          INSERT OR IGNORE INTO action_lifecycle_turn_counter (id, value)
          SELECT
            'global',
            COALESCE(
              (SELECT MAX(last_referenced_turn_global) FROM action_records),
              0
            );
        `);
      } else {
        db.exec(`
          INSERT OR IGNORE INTO action_lifecycle_turn_counter (id, value)
            VALUES ('global', 0);
        `);
      }
    },
  },
] as const satisfies readonly Migration[];
