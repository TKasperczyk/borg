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
          unknown_at INTEGER NULL
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
] as const satisfies readonly Migration[];
