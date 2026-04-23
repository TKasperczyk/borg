import type { Migration, SqliteDatabase } from "../../storage/sqlite/index.js";
import { createTraitId, parseEpisodeId, parseSemanticNodeId } from "../../util/ids.js";

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

function tableHasColumn(db: SqliteDatabase, table: string, column: string): boolean {
  const escapedTable = table.replaceAll('"', '""');
  const columns = db.prepare(`PRAGMA table_info("${escapedTable}")`).all() as Array<{
    name: string;
  }>;
  return columns.some((entry) => entry.name === column);
}

function tableExists(db: SqliteDatabase, table: string): boolean {
  return (
    db
      .prepare(
        `
          SELECT 1
          FROM sqlite_master
          WHERE type = 'table' AND name = ?
          LIMIT 1
        `,
      )
      .get(table) !== undefined
  );
}

function getRecentDistinctEpisodeIds(
  rows: Array<{ ts: number; provenance_episode_ids: unknown }>,
  limit: number,
): string[] {
  const latestEpisodeTs = new Map<string, number>();

  for (const row of rows) {
    for (const episodeId of parseStoredIdArray(row.provenance_episode_ids)) {
      const ts = Number(row.ts);
      const currentTs = latestEpisodeTs.get(episodeId) ?? Number.NEGATIVE_INFINITY;
      if (ts > currentTs) {
        latestEpisodeTs.set(episodeId, ts);
      }
    }
  }

  return [...latestEpisodeTs.entries()]
    .sort((left, right) => right[1] - left[1] || left[0].localeCompare(right[0]))
    .slice(0, limit)
    .map(([episodeId]) => episodeId);
}

const CONFIDENCE_ALPHA = 2;
const CONFIDENCE_BETA = 1;

function computeEvidenceConfidence(supportCount: number, contradictionCount: number): number {
  return (CONFIDENCE_ALPHA + supportCount) /
    (CONFIDENCE_ALPHA + CONFIDENCE_BETA + supportCount + contradictionCount);
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
  {
    id: 220,
    name: "add-self-provenance",
    up: (db) => {
      if (!tableHasColumn(db, "values", "provenance_kind")) {
        db.exec("ALTER TABLE \"values\" ADD COLUMN provenance_kind TEXT");
      }
      if (!tableHasColumn(db, "values", "provenance_episode_ids")) {
        db.exec("ALTER TABLE \"values\" ADD COLUMN provenance_episode_ids TEXT");
      }
      if (!tableHasColumn(db, "values", "provenance_process")) {
        db.exec("ALTER TABLE \"values\" ADD COLUMN provenance_process TEXT");
      }
      if (!tableHasColumn(db, "goals", "provenance_kind")) {
        db.exec("ALTER TABLE goals ADD COLUMN provenance_kind TEXT");
      }
      if (!tableHasColumn(db, "goals", "provenance_episode_ids")) {
        db.exec("ALTER TABLE goals ADD COLUMN provenance_episode_ids TEXT");
      }
      if (!tableHasColumn(db, "goals", "provenance_process")) {
        db.exec("ALTER TABLE goals ADD COLUMN provenance_process TEXT");
      }
      if (!tableHasColumn(db, "traits", "id")) {
        db.exec("ALTER TABLE traits ADD COLUMN id TEXT");
      }
      if (!tableHasColumn(db, "traits", "provenance_kind")) {
        db.exec("ALTER TABLE traits ADD COLUMN provenance_kind TEXT");
      }
      if (!tableHasColumn(db, "traits", "provenance_episode_ids")) {
        db.exec("ALTER TABLE traits ADD COLUMN provenance_episode_ids TEXT");
      }
      if (!tableHasColumn(db, "traits", "provenance_process")) {
        db.exec("ALTER TABLE traits ADD COLUMN provenance_process TEXT");
      }
      if (!tableHasColumn(db, "autobiographical_periods", "provenance_kind")) {
        db.exec("ALTER TABLE autobiographical_periods ADD COLUMN provenance_kind TEXT");
      }
      if (!tableHasColumn(db, "autobiographical_periods", "provenance_episode_ids")) {
        db.exec("ALTER TABLE autobiographical_periods ADD COLUMN provenance_episode_ids TEXT");
      }
      if (!tableHasColumn(db, "autobiographical_periods", "provenance_process")) {
        db.exec("ALTER TABLE autobiographical_periods ADD COLUMN provenance_process TEXT");
      }
      if (!tableHasColumn(db, "open_questions", "provenance_kind")) {
        db.exec("ALTER TABLE open_questions ADD COLUMN provenance_kind TEXT");
      }
      if (!tableHasColumn(db, "open_questions", "provenance_episode_ids")) {
        db.exec("ALTER TABLE open_questions ADD COLUMN provenance_episode_ids TEXT");
      }
      if (!tableHasColumn(db, "open_questions", "provenance_process")) {
        db.exec("ALTER TABLE open_questions ADD COLUMN provenance_process TEXT");
      }

      db.exec(`
        CREATE TABLE IF NOT EXISTS trait_reinforcement_events (
          id INTEGER PRIMARY KEY AUTOINCREMENT,
          trait_id TEXT NOT NULL,
          delta REAL NOT NULL,
          ts INTEGER NOT NULL,
          provenance_kind TEXT NOT NULL CHECK (
            provenance_kind IN ('episodes', 'manual', 'system', 'offline')
          ),
          provenance_episode_ids TEXT NOT NULL DEFAULT '[]',
          provenance_process TEXT
        );
        CREATE INDEX IF NOT EXISTS idx_trait_reinforcement_events_trait_ts
          ON trait_reinforcement_events (trait_id, ts DESC, id DESC);
        CREATE UNIQUE INDEX IF NOT EXISTS idx_traits_id
          ON traits (id)
          WHERE id IS NOT NULL;
      `);

      const valueRows = db
        .prepare(
          `
            SELECT id
            FROM "values"
            ORDER BY created_at ASC, id ASC
          `,
        )
        .all() as Array<{ id: string }>;
      const valueSourcesStatement = db.prepare(
        `
          SELECT episode_id
          FROM value_sources
          WHERE value_id = ?
          ORDER BY episode_id ASC
        `,
      );
      const updateValueStatement = db.prepare(
        `
          UPDATE "values"
          SET provenance_kind = ?, provenance_episode_ids = ?, provenance_process = NULL
          WHERE id = ?
        `,
      );

      for (const row of valueRows) {
        const sources = (valueSourcesStatement.all(row.id) as Array<{ episode_id: string }>).map(
          (entry) => entry.episode_id,
        );
        updateValueStatement.run(
          sources.length > 0 ? "episodes" : "system",
          JSON.stringify(sources),
          row.id,
        );
      }

      db.exec(`
        UPDATE goals
        SET provenance_kind = COALESCE(provenance_kind, 'system'),
            provenance_episode_ids = COALESCE(provenance_episode_ids, '[]'),
            provenance_process = NULL;

        UPDATE autobiographical_periods
        SET provenance_kind = COALESCE(provenance_kind, 'system'),
            provenance_episode_ids = COALESCE(provenance_episode_ids, '[]'),
            provenance_process = NULL;
      `);

      const openQuestionRows = db
        .prepare(
          `
            SELECT id, related_episode_ids
            FROM open_questions
            ORDER BY created_at ASC, id ASC
          `,
        )
        .all() as Array<Record<string, unknown>>;
      const updateOpenQuestionStatement = db.prepare(
        `
          UPDATE open_questions
          SET provenance_kind = ?, provenance_episode_ids = ?, provenance_process = NULL
          WHERE id = ?
        `,
      );

      for (const row of openQuestionRows) {
        const relatedEpisodeIds = parseStoredIdArray(row.related_episode_ids).filter(
          (value) => value.length > 0,
        );
        updateOpenQuestionStatement.run(
          relatedEpisodeIds.length > 0 ? "episodes" : "system",
          JSON.stringify(relatedEpisodeIds),
          row.id,
        );
      }

      const traitRows = db
        .prepare(
          `
            SELECT label
            FROM traits
            ORDER BY label ASC
          `,
        )
        .all() as Array<{ label: string }>;
      const updateTraitStatement = db.prepare(
        `
          UPDATE traits
          SET id = COALESCE(id, ?),
              provenance_kind = COALESCE(provenance_kind, 'system'),
              provenance_episode_ids = COALESCE(provenance_episode_ids, '[]'),
              provenance_process = NULL
          WHERE label = ?
        `,
      );

      for (const row of traitRows) {
        updateTraitStatement.run(createTraitId(), row.label);
      }
    },
  },
  {
    id: 221,
    name: "add-self-identity-state",
    up: (db) => {
      if (!tableHasColumn(db, "values", "state")) {
        db.exec("ALTER TABLE \"values\" ADD COLUMN state TEXT");
      }
      if (!tableHasColumn(db, "values", "established_at")) {
        db.exec("ALTER TABLE \"values\" ADD COLUMN established_at INTEGER");
      }
      if (!tableHasColumn(db, "traits", "state")) {
        db.exec("ALTER TABLE traits ADD COLUMN state TEXT");
      }
      if (!tableHasColumn(db, "traits", "established_at")) {
        db.exec("ALTER TABLE traits ADD COLUMN established_at INTEGER");
      }

      db.exec(`
        CREATE TABLE IF NOT EXISTS value_reinforcement_events (
          id INTEGER PRIMARY KEY AUTOINCREMENT,
          value_id TEXT NOT NULL,
          ts INTEGER NOT NULL,
          provenance_kind TEXT NOT NULL CHECK (
            provenance_kind IN ('episodes', 'manual', 'system', 'offline')
          ),
          provenance_episode_ids TEXT NOT NULL DEFAULT '[]',
          provenance_process TEXT
        );
        CREATE INDEX IF NOT EXISTS idx_value_reinforcement_events_value_ts
          ON value_reinforcement_events (value_id, ts DESC, id DESC);
      `);

      const valueRows = db
        .prepare(
          `
            SELECT id, created_at, last_affirmed, provenance_kind
            FROM "values"
            ORDER BY created_at ASC, id ASC
          `,
        )
        .all() as Array<Record<string, unknown>>;
      const valueSourcesStatement = db.prepare(
        `
          SELECT episode_id
          FROM value_sources
          WHERE value_id = ?
          ORDER BY episode_id ASC
        `,
      );
      const existingValueEventCount = Number(
        (db.prepare("SELECT COUNT(*) AS count FROM value_reinforcement_events").get() as {
          count: number;
        }).count,
      );
      const insertValueEvent = db.prepare(
        `
          INSERT INTO value_reinforcement_events (
            value_id, ts, provenance_kind, provenance_episode_ids, provenance_process
          ) VALUES (?, ?, ?, ?, NULL)
        `,
      );
      const updateValueState = db.prepare(
        `
          UPDATE "values"
          SET state = ?, established_at = ?
          WHERE id = ?
        `,
      );

      if (existingValueEventCount === 0) {
        for (const row of valueRows) {
          const episodeIds = (valueSourcesStatement.all(row.id) as Array<{ episode_id: string }>).map(
            (entry) => entry.episode_id,
          );

          for (const episodeId of episodeIds) {
            insertValueEvent.run(
              row.id,
              Number(row.created_at),
              "episodes",
              JSON.stringify([episodeId]),
            );
          }
        }
      }

      for (const row of valueRows) {
        const storedKind = String(row.provenance_kind ?? "system");
        const sourceEpisodeIds = (valueSourcesStatement.all(row.id) as Array<{ episode_id: string }>).map(
          (entry) => entry.episode_id,
        );
        const distinctEpisodeCount = new Set(sourceEpisodeIds).size;
        const establishedAt =
          Number(row.last_affirmed ?? row.created_at ?? 0) || Number(row.created_at ?? 0);

        if (storedKind === "manual" || storedKind === "system") {
          updateValueState.run("established", establishedAt, row.id);
          continue;
        }

        updateValueState.run(
          distinctEpisodeCount >= 3 ? "established" : "candidate",
          distinctEpisodeCount >= 3 ? Number(row.created_at) : null,
          row.id,
        );
      }

      const traitRows = db
        .prepare(
          `
            SELECT id
            FROM traits
            ORDER BY label ASC
          `,
        )
        .all() as Array<{ id: string }>;
      const traitEventRowsStatement = db.prepare(
        `
          SELECT ts, provenance_kind, provenance_episode_ids
          FROM trait_reinforcement_events
          WHERE trait_id = ?
          ORDER BY ts ASC, id ASC
        `,
      );
      const updateTraitState = db.prepare(
        `
          UPDATE traits
          SET state = ?, established_at = ?
          WHERE id = ?
        `,
      );

      for (const row of traitRows) {
        const events = traitEventRowsStatement.all(row.id) as Array<Record<string, unknown>>;
        const seenEpisodes = new Set<string>();
        let establishedAt: number | null = null;

        for (const event of events) {
          if (event.provenance_kind !== "episodes") {
            continue;
          }

          for (const episodeId of parseStoredIdArray(event.provenance_episode_ids)) {
            seenEpisodes.add(episodeId);
          }

          if (seenEpisodes.size >= 5) {
            establishedAt = Number(event.ts);
            break;
          }
        }

        updateTraitState.run(
          establishedAt === null ? "candidate" : "established",
          establishedAt,
          row.id,
        );
      }
    },
  },
  {
    id: 260,
    name: "add-self-evidence-claims",
    up: (db) => {
      for (const table of ["values", "traits"] as const) {
        if (!tableHasColumn(db, table, "confidence")) {
          db.exec(`ALTER TABLE ${table === "values" ? '"values"' : table} ADD COLUMN confidence REAL`);
        }
        if (!tableHasColumn(db, table, "last_tested_at")) {
          db.exec(`ALTER TABLE ${table === "values" ? '"values"' : table} ADD COLUMN last_tested_at INTEGER`);
        }
        if (!tableHasColumn(db, table, "last_contradicted_at")) {
          db.exec(
            `ALTER TABLE ${table === "values" ? '"values"' : table} ADD COLUMN last_contradicted_at INTEGER`,
          );
        }
        if (!tableHasColumn(db, table, "support_count")) {
          db.exec(`ALTER TABLE ${table === "values" ? '"values"' : table} ADD COLUMN support_count INTEGER`);
        }
        if (!tableHasColumn(db, table, "contradiction_count")) {
          db.exec(
            `ALTER TABLE ${table === "values" ? '"values"' : table} ADD COLUMN contradiction_count INTEGER`,
          );
        }
        if (!tableHasColumn(db, table, "evidence_episode_ids")) {
          db.exec(
            `ALTER TABLE ${table === "values" ? '"values"' : table} ADD COLUMN evidence_episode_ids TEXT`,
          );
        }
      }

      db.exec(`
        CREATE TABLE IF NOT EXISTS value_contradiction_events (
          id INTEGER PRIMARY KEY AUTOINCREMENT,
          value_id TEXT NOT NULL,
          ts INTEGER NOT NULL,
          weight REAL NOT NULL DEFAULT 1,
          provenance_kind TEXT NOT NULL CHECK (
            provenance_kind IN ('episodes', 'manual', 'system', 'offline')
          ),
          provenance_episode_ids TEXT NOT NULL DEFAULT '[]',
          provenance_process TEXT
        );
        CREATE INDEX IF NOT EXISTS idx_value_contradiction_events_value_ts
          ON value_contradiction_events (value_id, ts DESC, id DESC);

        CREATE TABLE IF NOT EXISTS trait_contradiction_events (
          id INTEGER PRIMARY KEY AUTOINCREMENT,
          trait_id TEXT NOT NULL,
          ts INTEGER NOT NULL,
          weight REAL NOT NULL DEFAULT 1,
          provenance_kind TEXT NOT NULL CHECK (
            provenance_kind IN ('episodes', 'manual', 'system', 'offline')
          ),
          provenance_episode_ids TEXT NOT NULL DEFAULT '[]',
          provenance_process TEXT
        );
        CREATE INDEX IF NOT EXISTS idx_trait_contradiction_events_trait_ts
          ON trait_contradiction_events (trait_id, ts DESC, id DESC);
      `);

      const valueRows = db
        .prepare(
          `
            SELECT id, state
            FROM "values"
            ORDER BY created_at ASC, id ASC
          `,
        )
        .all() as Array<{ id: string; state: string | null }>;
      const valueSupportRows = db.prepare(
        `
          SELECT ts, provenance_episode_ids
          FROM value_reinforcement_events
          WHERE value_id = ? AND provenance_kind = 'episodes'
          ORDER BY ts DESC, id DESC
        `,
      );
      const updateValueEvidence = db.prepare(
        `
          UPDATE "values"
          SET confidence = ?, last_tested_at = ?, last_contradicted_at = NULL,
              support_count = ?, contradiction_count = 0, evidence_episode_ids = ?
          WHERE id = ?
        `,
      );

      for (const row of valueRows) {
        const supportRows = valueSupportRows.all(row.id) as Array<{
          ts: number;
          provenance_episode_ids: unknown;
        }>;
        const supportCount = supportRows.length;
        const lastTestedAt = supportCount === 0 ? null : Number(supportRows[0]?.ts ?? 0);
        const confidence = computeEvidenceConfidence(supportCount, 0);
        updateValueEvidence.run(
          confidence,
          lastTestedAt,
          supportCount,
          JSON.stringify(getRecentDistinctEpisodeIds(supportRows, 3)),
          row.id,
        );
      }

      const traitRows = db
        .prepare(
          `
            SELECT id, state
            FROM traits
            ORDER BY label ASC
          `,
        )
        .all() as Array<{ id: string; state: string | null }>;
      const traitSupportRows = db.prepare(
        `
          SELECT ts, provenance_episode_ids
          FROM trait_reinforcement_events
          WHERE trait_id = ? AND provenance_kind = 'episodes'
          ORDER BY ts DESC, id DESC
        `,
      );
      const updateTraitEvidence = db.prepare(
        `
          UPDATE traits
          SET confidence = ?, last_tested_at = ?, last_contradicted_at = NULL,
              support_count = ?, contradiction_count = 0, evidence_episode_ids = ?
          WHERE id = ?
        `,
      );

      for (const row of traitRows) {
        const supportRows = traitSupportRows.all(row.id) as Array<{
          ts: number;
          provenance_episode_ids: unknown;
        }>;
        const supportCount = supportRows.length;
        const lastTestedAt = supportCount === 0 ? null : Number(supportRows[0]?.ts ?? 0);
        const confidence = computeEvidenceConfidence(supportCount, 0);
        updateTraitEvidence.run(
          confidence,
          lastTestedAt,
          supportCount,
          JSON.stringify(getRecentDistinctEpisodeIds(supportRows, 3)),
          row.id,
        );
      }
    },
  },
  {
    id: 261,
    name: "allow-autonomy-open-question-source",
    up: (db) => {
      db.exec(`
        CREATE TABLE open_questions__next (
          id TEXT PRIMARY KEY,
          question TEXT NOT NULL,
          urgency REAL NOT NULL,
          status TEXT NOT NULL CHECK (status IN ('open', 'resolved', 'abandoned')),
          related_episode_ids TEXT NOT NULL,
          related_semantic_node_ids TEXT NOT NULL,
          source TEXT NOT NULL CHECK (
            source IN ('user', 'reflection', 'contradiction', 'ruminator', 'overseer', 'autonomy')
          ),
          created_at INTEGER NOT NULL,
          last_touched INTEGER NOT NULL,
          resolution_episode_id TEXT,
          resolution_note TEXT,
          resolved_at INTEGER,
          abandoned_reason TEXT,
          abandoned_at INTEGER,
          dedupe_key TEXT,
          provenance_kind TEXT,
          provenance_episode_ids TEXT,
          provenance_process TEXT
        );

        INSERT INTO open_questions__next (
          id,
          question,
          urgency,
          status,
          related_episode_ids,
          related_semantic_node_ids,
          source,
          created_at,
          last_touched,
          resolution_episode_id,
          resolution_note,
          resolved_at,
          abandoned_reason,
          abandoned_at,
          dedupe_key,
          provenance_kind,
          provenance_episode_ids,
          provenance_process
        )
        SELECT
          id,
          question,
          urgency,
          status,
          related_episode_ids,
          related_semantic_node_ids,
          source,
          created_at,
          last_touched,
          resolution_episode_id,
          resolution_note,
          resolved_at,
          abandoned_reason,
          abandoned_at,
          dedupe_key,
          provenance_kind,
          provenance_episode_ids,
          provenance_process
        FROM open_questions;

        DROP TABLE open_questions;
        ALTER TABLE open_questions__next RENAME TO open_questions;

        CREATE INDEX IF NOT EXISTS idx_open_questions_status_urgency
          ON open_questions (status, urgency DESC, last_touched DESC);
        CREATE UNIQUE INDEX IF NOT EXISTS idx_open_questions_dedupe_key
          ON open_questions (dedupe_key);
      `);
    },
  },
  {
    id: 262,
    name: "add-goal-last-progress-ts",
    up: (db) => {
      if (!tableHasColumn(db, "goals", "last_progress_ts")) {
        db.exec("ALTER TABLE goals ADD COLUMN last_progress_ts INTEGER");
      }

      if (!tableExists(db, "identity_events")) {
        return;
      }

      db.exec(`
        UPDATE goals
        SET last_progress_ts = (
          SELECT MAX(ts)
          FROM identity_events
          WHERE record_type = 'goal'
            AND record_id = goals.id
            AND action IN ('update_progress', 'update')
            AND COALESCE(json_extract(old_value_json, '$.progress_notes'), '__null__')
              != COALESCE(json_extract(new_value_json, '$.progress_notes'), '__null__')
        )
        WHERE last_progress_ts IS NULL;
      `);
    },
  },
  {
    id: 263,
    name: "add-growth-marker-provenance",
    up: (db) => {
      if (!tableHasColumn(db, "growth_markers", "provenance_kind")) {
        db.exec("ALTER TABLE growth_markers ADD COLUMN provenance_kind TEXT");
      }

      if (!tableHasColumn(db, "growth_markers", "provenance_episode_ids")) {
        db.exec("ALTER TABLE growth_markers ADD COLUMN provenance_episode_ids TEXT");
      }

      if (!tableHasColumn(db, "growth_markers", "provenance_process")) {
        db.exec("ALTER TABLE growth_markers ADD COLUMN provenance_process TEXT");
      }

      const rows = db
        .prepare(
          `
            SELECT id, evidence_episode_ids, source_process, provenance_kind
            FROM growth_markers
            ORDER BY created_at ASC, id ASC
          `,
        )
        .all() as Array<Record<string, unknown>>;
      const updateGrowthMarkerProvenance = db.prepare(
        `
          UPDATE growth_markers
          SET provenance_kind = ?, provenance_episode_ids = ?, provenance_process = ?, source_process = ?
          WHERE id = ?
        `,
      );

      for (const row of rows) {
        if (typeof row.provenance_kind === "string" && row.provenance_kind.length > 0) {
          continue;
        }

        const evidenceEpisodeIds = parseStoredIdArray(row.evidence_episode_ids).filter(
          (value) => value.length > 0,
        );
        const sourceProcess =
          typeof row.source_process === "string" && row.source_process.trim().length > 0
            ? row.source_process.trim()
            : null;

        if (evidenceEpisodeIds.length > 0) {
          updateGrowthMarkerProvenance.run(
            "episodes",
            JSON.stringify(evidenceEpisodeIds),
            null,
            sourceProcess ?? "growth-marker-detector",
            row.id,
          );
          continue;
        }

        updateGrowthMarkerProvenance.run(
          "offline",
          "[]",
          sourceProcess ?? "growth-marker-detector",
          sourceProcess ?? "growth-marker-detector",
          row.id,
        );
      }
    },
  },
  {
    id: 264,
    name: "allow-online-self-event-provenance",
    up: (db) => {
      db.exec(`
        CREATE TABLE trait_reinforcement_events__next (
          id INTEGER PRIMARY KEY AUTOINCREMENT,
          trait_id TEXT NOT NULL,
          delta REAL NOT NULL,
          ts INTEGER NOT NULL,
          provenance_kind TEXT NOT NULL CHECK (
            provenance_kind IN ('episodes', 'manual', 'system', 'offline', 'online')
          ),
          provenance_episode_ids TEXT NOT NULL DEFAULT '[]',
          provenance_process TEXT
        );

        INSERT INTO trait_reinforcement_events__next (
          id,
          trait_id,
          delta,
          ts,
          provenance_kind,
          provenance_episode_ids,
          provenance_process
        )
        SELECT
          id,
          trait_id,
          delta,
          ts,
          provenance_kind,
          provenance_episode_ids,
          provenance_process
        FROM trait_reinforcement_events;

        DROP TABLE trait_reinforcement_events;
        ALTER TABLE trait_reinforcement_events__next RENAME TO trait_reinforcement_events;

        CREATE INDEX IF NOT EXISTS idx_trait_reinforcement_events_trait_ts
          ON trait_reinforcement_events (trait_id, ts DESC, id DESC);

        CREATE TABLE value_reinforcement_events__next (
          id INTEGER PRIMARY KEY AUTOINCREMENT,
          value_id TEXT NOT NULL,
          ts INTEGER NOT NULL,
          provenance_kind TEXT NOT NULL CHECK (
            provenance_kind IN ('episodes', 'manual', 'system', 'offline', 'online')
          ),
          provenance_episode_ids TEXT NOT NULL DEFAULT '[]',
          provenance_process TEXT
        );

        INSERT INTO value_reinforcement_events__next (
          id,
          value_id,
          ts,
          provenance_kind,
          provenance_episode_ids,
          provenance_process
        )
        SELECT
          id,
          value_id,
          ts,
          provenance_kind,
          provenance_episode_ids,
          provenance_process
        FROM value_reinforcement_events;

        DROP TABLE value_reinforcement_events;
        ALTER TABLE value_reinforcement_events__next RENAME TO value_reinforcement_events;

        CREATE INDEX IF NOT EXISTS idx_value_reinforcement_events_value_ts
          ON value_reinforcement_events (value_id, ts DESC, id DESC);

        CREATE TABLE value_contradiction_events__next (
          id INTEGER PRIMARY KEY AUTOINCREMENT,
          value_id TEXT NOT NULL,
          ts INTEGER NOT NULL,
          weight REAL NOT NULL DEFAULT 1,
          provenance_kind TEXT NOT NULL CHECK (
            provenance_kind IN ('episodes', 'manual', 'system', 'offline', 'online')
          ),
          provenance_episode_ids TEXT NOT NULL DEFAULT '[]',
          provenance_process TEXT
        );

        INSERT INTO value_contradiction_events__next (
          id,
          value_id,
          ts,
          weight,
          provenance_kind,
          provenance_episode_ids,
          provenance_process
        )
        SELECT
          id,
          value_id,
          ts,
          weight,
          provenance_kind,
          provenance_episode_ids,
          provenance_process
        FROM value_contradiction_events;

        DROP TABLE value_contradiction_events;
        ALTER TABLE value_contradiction_events__next RENAME TO value_contradiction_events;

        CREATE INDEX IF NOT EXISTS idx_value_contradiction_events_value_ts
          ON value_contradiction_events (value_id, ts DESC, id DESC);

        CREATE TABLE trait_contradiction_events__next (
          id INTEGER PRIMARY KEY AUTOINCREMENT,
          trait_id TEXT NOT NULL,
          ts INTEGER NOT NULL,
          weight REAL NOT NULL DEFAULT 1,
          provenance_kind TEXT NOT NULL CHECK (
            provenance_kind IN ('episodes', 'manual', 'system', 'offline', 'online')
          ),
          provenance_episode_ids TEXT NOT NULL DEFAULT '[]',
          provenance_process TEXT
        );

        INSERT INTO trait_contradiction_events__next (
          id,
          trait_id,
          ts,
          weight,
          provenance_kind,
          provenance_episode_ids,
          provenance_process
        )
        SELECT
          id,
          trait_id,
          ts,
          weight,
          provenance_kind,
          provenance_episode_ids,
          provenance_process
        FROM trait_contradiction_events;

        DROP TABLE trait_contradiction_events;
        ALTER TABLE trait_contradiction_events__next RENAME TO trait_contradiction_events;

        CREATE INDEX IF NOT EXISTS idx_trait_contradiction_events_trait_ts
          ON trait_contradiction_events (trait_id, ts DESC, id DESC);
      `);
    },
  },
  {
    id: 265,
    name: "allow-deliberator-open-question-source",
    up: (db) => {
      db.exec(`
        CREATE TABLE open_questions__next (
          id TEXT PRIMARY KEY,
          question TEXT NOT NULL,
          urgency REAL NOT NULL,
          status TEXT NOT NULL CHECK (status IN ('open', 'resolved', 'abandoned')),
          related_episode_ids TEXT NOT NULL,
          related_semantic_node_ids TEXT NOT NULL,
          source TEXT NOT NULL CHECK (
            source IN ('user', 'reflection', 'contradiction', 'ruminator', 'overseer', 'autonomy', 'deliberator')
          ),
          created_at INTEGER NOT NULL,
          last_touched INTEGER NOT NULL,
          resolution_episode_id TEXT,
          resolution_note TEXT,
          resolved_at INTEGER,
          abandoned_reason TEXT,
          abandoned_at INTEGER,
          dedupe_key TEXT,
          provenance_kind TEXT,
          provenance_episode_ids TEXT,
          provenance_process TEXT
        );

        INSERT INTO open_questions__next (
          id,
          question,
          urgency,
          status,
          related_episode_ids,
          related_semantic_node_ids,
          source,
          created_at,
          last_touched,
          resolution_episode_id,
          resolution_note,
          resolved_at,
          abandoned_reason,
          abandoned_at,
          dedupe_key,
          provenance_kind,
          provenance_episode_ids,
          provenance_process
        )
        SELECT
          id,
          question,
          urgency,
          status,
          related_episode_ids,
          related_semantic_node_ids,
          source,
          created_at,
          last_touched,
          resolution_episode_id,
          resolution_note,
          resolved_at,
          abandoned_reason,
          abandoned_at,
          dedupe_key,
          provenance_kind,
          provenance_episode_ids,
          provenance_process
        FROM open_questions;

        DROP TABLE open_questions;
        ALTER TABLE open_questions__next RENAME TO open_questions;

        CREATE INDEX IF NOT EXISTS idx_open_questions_status_urgency
          ON open_questions (status, urgency DESC, last_touched DESC);
        CREATE UNIQUE INDEX IF NOT EXISTS idx_open_questions_dedupe_key
          ON open_questions (dedupe_key);
      `);
    },
  },
] as const satisfies readonly Migration[];
