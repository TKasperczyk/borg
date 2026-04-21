import type { Migration } from "../../storage/sqlite/index.js";
import { parseEpisodeId, parseSemanticNodeId } from "../../util/ids.js";

import { buildOpenQuestionDedupeKey } from "./open-questions.js";

function parseStoredIdArray(value: unknown): string[] {
  if (typeof value !== "string") {
    return [];
  }

  try {
    const parsed = JSON.parse(value) as unknown;
    return Array.isArray(parsed)
      ? parsed.filter((item): item is string => typeof item === "string")
      : [];
  } catch {
    return [];
  }
}

export const selfMigrations = [
  {
    id: 110,
    name: "create-self-tables",
    up: `
      CREATE TABLE IF NOT EXISTS "values" (
        id TEXT PRIMARY KEY,
        label TEXT NOT NULL,
        description TEXT NOT NULL,
        priority REAL NOT NULL,
        created_at INTEGER NOT NULL,
        last_affirmed INTEGER
      );
      CREATE TABLE IF NOT EXISTS value_sources (
        value_id TEXT NOT NULL,
        episode_id TEXT NOT NULL,
        PRIMARY KEY (value_id, episode_id),
        FOREIGN KEY (value_id) REFERENCES "values"(id) ON DELETE CASCADE
      );
      CREATE TABLE IF NOT EXISTS goals (
        id TEXT PRIMARY KEY,
        description TEXT NOT NULL,
        priority REAL NOT NULL,
        parent_goal_id TEXT,
        status TEXT NOT NULL CHECK (status IN ('active', 'done', 'abandoned', 'blocked')),
        progress_notes TEXT,
        created_at INTEGER NOT NULL,
        target_at INTEGER,
        FOREIGN KEY (parent_goal_id) REFERENCES goals(id) ON DELETE SET NULL
      );
      CREATE TABLE IF NOT EXISTS traits (
        label TEXT PRIMARY KEY,
        strength REAL NOT NULL,
        last_reinforced INTEGER NOT NULL,
        last_decayed INTEGER
      )
    `,
  },
  {
    id: 111,
    name: "create-self-narrative-tables",
    up: `
      CREATE TABLE IF NOT EXISTS autobiographical_periods (
        id TEXT PRIMARY KEY,
        label TEXT NOT NULL UNIQUE,
        start_ts INTEGER NOT NULL,
        end_ts INTEGER,
        narrative TEXT NOT NULL,
        key_episode_ids TEXT NOT NULL,
        themes TEXT NOT NULL,
        created_at INTEGER NOT NULL,
        last_updated INTEGER NOT NULL
      );
      CREATE INDEX IF NOT EXISTS idx_autobiographical_periods_start
        ON autobiographical_periods (start_ts DESC);
      CREATE INDEX IF NOT EXISTS idx_autobiographical_periods_end
        ON autobiographical_periods (end_ts);

      CREATE TABLE IF NOT EXISTS growth_markers (
        id TEXT PRIMARY KEY,
        ts INTEGER NOT NULL,
        category TEXT NOT NULL CHECK (
          category IN ('skill', 'value', 'habit', 'relationship', 'understanding')
        ),
        what_changed TEXT NOT NULL,
        before_description TEXT,
        after_description TEXT,
        evidence_episode_ids TEXT NOT NULL,
        confidence REAL NOT NULL,
        source_process TEXT NOT NULL,
        created_at INTEGER NOT NULL
      );
      CREATE INDEX IF NOT EXISTS idx_growth_markers_ts
        ON growth_markers (ts DESC);
      CREATE INDEX IF NOT EXISTS idx_growth_markers_category
        ON growth_markers (category);

      CREATE TABLE IF NOT EXISTS open_questions (
        id TEXT PRIMARY KEY,
        question TEXT NOT NULL,
        urgency REAL NOT NULL,
        status TEXT NOT NULL CHECK (status IN ('open', 'resolved', 'abandoned')),
        related_episode_ids TEXT NOT NULL,
        related_semantic_node_ids TEXT NOT NULL,
        source TEXT NOT NULL CHECK (
          source IN ('user', 'reflection', 'contradiction', 'ruminator', 'overseer')
        ),
        created_at INTEGER NOT NULL,
        last_touched INTEGER NOT NULL,
        resolution_episode_id TEXT,
        resolution_note TEXT,
        resolved_at INTEGER,
        abandoned_reason TEXT,
        abandoned_at INTEGER
      );
      CREATE INDEX IF NOT EXISTS idx_open_questions_status_urgency
        ON open_questions (status, urgency DESC, last_touched DESC);
    `,
  },
  {
    id: 112,
    name: "add-open-question-dedupe-key",
    up: (db) => {
      const columns = db.prepare("PRAGMA table_info(open_questions)").all() as Array<{
        name: string;
      }>;

      if (!columns.some((column) => column.name === "dedupe_key")) {
        db.exec("ALTER TABLE open_questions ADD COLUMN dedupe_key TEXT");
      }

      const rows = db
        .prepare(
          `
            SELECT id, question, related_episode_ids, related_semantic_node_ids, created_at
            FROM open_questions
            ORDER BY created_at ASC, id ASC
          `,
        )
        .all() as Array<Record<string, unknown>>;
      const seenKeys = new Set<string>();
      const updateStatement = db.prepare("UPDATE open_questions SET dedupe_key = ? WHERE id = ?");

      for (const row of rows) {
        const baseKey = buildOpenQuestionDedupeKey({
          question: String(row.question ?? ""),
          relatedEpisodeIds: parseStoredIdArray(row.related_episode_ids).map((id) =>
            parseEpisodeId(id),
          ),
          relatedSemanticNodeIds: parseStoredIdArray(row.related_semantic_node_ids).map((id) =>
            parseSemanticNodeId(id),
          ),
        });
        const dedupeKey = seenKeys.has(baseKey) ? `${baseKey}|legacy:${String(row.id)}` : baseKey;
        seenKeys.add(dedupeKey);
        updateStatement.run(dedupeKey, row.id);
      }

      db.exec(`
        CREATE UNIQUE INDEX IF NOT EXISTS idx_open_questions_dedupe_key
          ON open_questions (dedupe_key)
          WHERE dedupe_key IS NOT NULL
      `);
    },
  },
  {
    id: 113,
    name: "allow-duplicate-autobiographical-labels",
    up: (db) => {
      const rows = db
        .prepare(
          `
            SELECT id, label, start_ts, end_ts, narrative, key_episode_ids, themes, created_at, last_updated
            FROM autobiographical_periods
            ORDER BY start_ts ASC, created_at ASC
          `,
        )
        .all() as Array<Record<string, unknown>>;
      const openRows = rows.filter((row) => row.end_ts === null || row.end_ts === undefined);
      const latestOpenId =
        openRows.length === 0 ? null : String(openRows[openRows.length - 1]?.id ?? "");

      db.exec(`
        DROP INDEX IF EXISTS idx_autobiographical_periods_start;
        DROP INDEX IF EXISTS idx_autobiographical_periods_end;
        DROP INDEX IF EXISTS autobiographical_single_open;

        ALTER TABLE autobiographical_periods RENAME TO autobiographical_periods_legacy;

        CREATE TABLE autobiographical_periods (
          id TEXT PRIMARY KEY,
          label TEXT NOT NULL,
          start_ts INTEGER NOT NULL,
          end_ts INTEGER,
          narrative TEXT NOT NULL,
          key_episode_ids TEXT NOT NULL,
          themes TEXT NOT NULL,
          created_at INTEGER NOT NULL,
          last_updated INTEGER NOT NULL
        );
      `);

      const insertStatement = db.prepare(
        `
          INSERT INTO autobiographical_periods (
            id, label, start_ts, end_ts, narrative, key_episode_ids, themes, created_at, last_updated
          ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        `,
      );

      for (const row of rows) {
        const startTs = Number(row.start_ts);
        const endTs =
          latestOpenId !== null &&
          String(row.id) !== latestOpenId &&
          (row.end_ts === null || row.end_ts === undefined)
            ? startTs
            : row.end_ts;

        insertStatement.run(
          row.id,
          row.label,
          startTs,
          endTs,
          row.narrative,
          row.key_episode_ids,
          row.themes,
          row.created_at,
          row.last_updated,
        );
      }

      db.exec(`
        DROP TABLE autobiographical_periods_legacy;

        CREATE INDEX IF NOT EXISTS idx_autobiographical_periods_start
          ON autobiographical_periods (start_ts DESC);
        CREATE INDEX IF NOT EXISTS idx_autobiographical_periods_end
          ON autobiographical_periods (end_ts);
        CREATE UNIQUE INDEX IF NOT EXISTS autobiographical_single_open
          ON autobiographical_periods (CASE WHEN end_ts IS NULL THEN 1 END)
          WHERE end_ts IS NULL;
      `);
    },
  },
] as const satisfies readonly Migration[];
