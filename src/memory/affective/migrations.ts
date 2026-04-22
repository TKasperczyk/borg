import type { Migration, SqliteDatabase } from "../../storage/sqlite/index.js";

function tableHasColumn(db: SqliteDatabase, table: string, column: string): boolean {
  const escapedTable = table.replaceAll('"', '""');
  const columns = db.prepare(`PRAGMA table_info("${escapedTable}")`).all() as Array<{
    name: string;
  }>;
  return columns.some((entry) => entry.name === column);
}

export const affectiveMigrations = [
  {
    id: 170,
    name: "create-mood-tables",
    up: `
      CREATE TABLE IF NOT EXISTS mood_state (
        session_id TEXT PRIMARY KEY,
        valence REAL NOT NULL,
        arousal REAL NOT NULL,
        updated_at INTEGER NOT NULL,
        half_life_hours REAL NOT NULL,
        recent_triggers TEXT NOT NULL
      );

      CREATE TABLE IF NOT EXISTS mood_history (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        session_id TEXT NOT NULL,
        ts INTEGER NOT NULL,
        valence REAL NOT NULL,
        arousal REAL NOT NULL,
        trigger_episode_id TEXT,
        trigger_reason TEXT
      );

      CREATE INDEX IF NOT EXISTS idx_mood_history_session_ts
        ON mood_history (session_id, ts DESC);
    `,
  },
  {
    id: 230,
    name: "add-mood-history-provenance",
    up: (db) => {
      if (!tableHasColumn(db, "mood_history", "provenance_kind")) {
        db.exec("ALTER TABLE mood_history ADD COLUMN provenance_kind TEXT");
      }
      if (!tableHasColumn(db, "mood_history", "provenance_episode_ids")) {
        db.exec("ALTER TABLE mood_history ADD COLUMN provenance_episode_ids TEXT");
      }
      if (!tableHasColumn(db, "mood_history", "provenance_process")) {
        db.exec("ALTER TABLE mood_history ADD COLUMN provenance_process TEXT");
      }

      const rows = db
        .prepare(
          `
            SELECT id, trigger_episode_id
            FROM mood_history
            ORDER BY ts ASC, id ASC
          `,
        )
        .all() as Array<Record<string, unknown>>;
      const update = db.prepare(
        `
          UPDATE mood_history
          SET provenance_kind = ?, provenance_episode_ids = ?, provenance_process = NULL
          WHERE id = ?
        `,
      );

      for (const row of rows) {
        const episodeId =
          row.trigger_episode_id === null || row.trigger_episode_id === undefined
            ? null
            : String(row.trigger_episode_id);
        update.run(
          episodeId === null ? "system" : "episodes",
          JSON.stringify(episodeId === null ? [] : [episodeId]),
          row.id,
        );
      }
    },
  },
] as const satisfies readonly Migration[];
