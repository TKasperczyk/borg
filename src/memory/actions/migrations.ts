import { StorageError } from "../../util/errors.js";
import type { Migration, SqliteDatabase } from "../../storage/sqlite/index.js";

function tableExists(db: SqliteDatabase, tableName: string): boolean {
  const row = db
    .prepare("SELECT name FROM sqlite_master WHERE type = 'table' AND name = ?")
    .get(tableName) as { name: string } | undefined;

  return row !== undefined;
}

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
        CREATE INDEX IF NOT EXISTS action_records_updated_idx
          ON action_records(updated_at DESC, id ASC);
      `);
    },
  },
] as const satisfies readonly Migration[];
