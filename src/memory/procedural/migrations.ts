import type { Migration } from "../../storage/sqlite/index.js";
import { tableHasColumn } from "../../storage/sqlite/migrations-utils.js";
import { serializeJsonValue } from "../../util/json-value.js";
import { deriveProceduralContextKey, parseLegacyProceduralContextKey } from "./context.js";

function nextAvailableContextKey(
  db: { prepare: (sql: string) => { get: (...args: unknown[]) => unknown } },
  skillId: string,
  baseKey: string,
): string {
  let candidate = baseKey;
  let suffix = 1;

  while (
    db
      .prepare(
        `
          SELECT 1
          FROM skill_context_stats
          WHERE skill_id = ? AND context_key = ?
        `,
      )
      .get(skillId, candidate) !== undefined
  ) {
    candidate = `${baseKey}:legacy-${suffix}`;
    suffix += 1;
  }

  return candidate;
}

export const proceduralMigrations = [
  {
    id: 172,
    name: "create-skills-table",
    up: `
      CREATE TABLE IF NOT EXISTS skills (
        id TEXT PRIMARY KEY,
        applies_when TEXT NOT NULL,
        approach TEXT NOT NULL,
        alpha REAL NOT NULL,
        beta REAL NOT NULL,
        attempts INTEGER NOT NULL,
        successes INTEGER NOT NULL,
        failures INTEGER NOT NULL,
        alternatives TEXT NOT NULL,
        source_episode_ids TEXT NOT NULL,
        last_used INTEGER,
        last_successful INTEGER,
        created_at INTEGER NOT NULL,
        updated_at INTEGER NOT NULL
      );
      CREATE INDEX IF NOT EXISTS idx_skills_updated_at
        ON skills (updated_at DESC);
    `,
  },
  {
    id: 173,
    name: "create-procedural-evidence-table",
    up: `
      CREATE TABLE IF NOT EXISTS procedural_evidence (
        id TEXT PRIMARY KEY,
        pending_attempt_snapshot TEXT NOT NULL,
        classification TEXT NOT NULL,
        evidence_text TEXT NOT NULL,
        resolved_episode_ids TEXT NOT NULL,
        audience_entity_id TEXT,
        consumed_at INTEGER,
        created_at INTEGER NOT NULL
      );
      CREATE INDEX IF NOT EXISTS idx_procedural_evidence_unconsumed
        ON procedural_evidence (consumed_at, created_at);
      CREATE INDEX IF NOT EXISTS idx_procedural_evidence_audience
        ON procedural_evidence (audience_entity_id);
    `,
  },
  {
    id: 174,
    name: "add-procedural-evidence-grounded",
    up: `
      ALTER TABLE procedural_evidence
        ADD COLUMN grounded INTEGER NOT NULL DEFAULT 1;
    `,
  },
  {
    id: 175,
    name: "add-procedural-evidence-skill-actually-applied",
    up: `
      ALTER TABLE procedural_evidence
        ADD COLUMN skill_actually_applied INTEGER NOT NULL DEFAULT 1;
    `,
  },
  {
    id: 176,
    name: "create-skill-context-stats-table",
    up: `
      CREATE TABLE IF NOT EXISTS skill_context_stats (
        skill_id TEXT NOT NULL,
        context_key TEXT NOT NULL,
        procedural_context_json TEXT,
        alpha REAL NOT NULL,
        beta REAL NOT NULL,
        attempts INTEGER NOT NULL,
        successes INTEGER NOT NULL,
        failures INTEGER NOT NULL,
        last_used INTEGER,
        last_successful INTEGER,
        updated_at INTEGER NOT NULL,
        PRIMARY KEY (skill_id, context_key)
      );
      CREATE INDEX IF NOT EXISTS idx_skill_context_stats_context
        ON skill_context_stats (context_key, updated_at DESC);
      CREATE INDEX IF NOT EXISTS idx_skill_context_stats_skill
        ON skill_context_stats (skill_id, updated_at DESC);
    `,
  },
  {
    id: 177,
    name: "add-procedural-evidence-context",
    up: (db) => {
      if (!tableHasColumn(db, "procedural_evidence", "procedural_context")) {
        db.exec("ALTER TABLE procedural_evidence ADD COLUMN procedural_context TEXT");
      }
    },
  },
  {
    id: 178,
    name: "add-skill-supersession-metadata",
    up: (db) => {
      if (!tableHasColumn(db, "skills", "status")) {
        db.exec("ALTER TABLE skills ADD COLUMN status TEXT NOT NULL DEFAULT 'active'");
      }
      if (!tableHasColumn(db, "skills", "superseded_by")) {
        db.exec("ALTER TABLE skills ADD COLUMN superseded_by TEXT NOT NULL DEFAULT '[]'");
      }
      if (!tableHasColumn(db, "skills", "superseded_at")) {
        db.exec("ALTER TABLE skills ADD COLUMN superseded_at INTEGER");
      }
      if (!tableHasColumn(db, "skills", "splitting_at")) {
        db.exec("ALTER TABLE skills ADD COLUMN splitting_at INTEGER");
      }
      db.exec(`
        CREATE INDEX IF NOT EXISTS idx_skills_status_updated
          ON skills (status, updated_at DESC);
      `);
    },
  },
  {
    id: 179,
    name: "add-skill-split-attempt-timestamp",
    up: (db) => {
      if (!tableHasColumn(db, "skills", "last_split_attempt_at")) {
        db.exec("ALTER TABLE skills ADD COLUMN last_split_attempt_at INTEGER");
      }
      db.exec(`
        CREATE INDEX IF NOT EXISTS idx_skills_split_attempt
          ON skills (status, last_split_attempt_at DESC);
      `);
    },
  },
  {
    id: 180,
    name: "add-skill-split-failure-metadata",
    up: (db) => {
      if (!tableHasColumn(db, "skills", "split_failure_count")) {
        db.exec("ALTER TABLE skills ADD COLUMN split_failure_count INTEGER NOT NULL DEFAULT 0");
      }
      if (!tableHasColumn(db, "skills", "last_split_error")) {
        db.exec("ALTER TABLE skills ADD COLUMN last_split_error TEXT");
      }
      db.exec(`
        CREATE INDEX IF NOT EXISTS idx_skills_split_failures
          ON skills (status, split_failure_count, updated_at DESC);
      `);
    },
  },
  {
    id: 181,
    name: "add-skill-manual-review-flag",
    up: (db) => {
      if (!tableHasColumn(db, "skills", "requires_manual_review")) {
        db.exec("ALTER TABLE skills ADD COLUMN requires_manual_review INTEGER NOT NULL DEFAULT 0");
      }
      db.exec(`
        CREATE INDEX IF NOT EXISTS idx_skills_manual_review
          ON skills (requires_manual_review, updated_at DESC);
      `);
    },
  },
  {
    id: 182,
    name: "backfill-procedural-context-keys-v2",
    up: (db) => {
      if (!tableHasColumn(db, "skill_context_stats", "procedural_context_json")) {
        db.exec("ALTER TABLE skill_context_stats ADD COLUMN procedural_context_json TEXT");
      }

      const rows = db
        .prepare(
          `
            SELECT skill_id, context_key
            FROM skill_context_stats
            WHERE context_key NOT LIKE 'v2:%'
          `,
        )
        .all() as Array<{ skill_id: string; context_key: string }>;
      const update = db.prepare(
        `
          UPDATE skill_context_stats
          SET context_key = ?,
              procedural_context_json = COALESCE(procedural_context_json, ?)
          WHERE skill_id = ? AND context_key = ?
        `,
      );

      db.transaction(() => {
        for (const row of rows) {
          const parsed = parseLegacyProceduralContextKey(row.context_key);

          if (parsed === null) {
            continue;
          }

          const baseKey = deriveProceduralContextKey(parsed);
          const existing = db
            .prepare(
              `
                SELECT 1
                FROM skill_context_stats
                WHERE skill_id = ? AND context_key = ?
              `,
            )
            .get(row.skill_id, baseKey);
          const nextKey =
            existing === undefined ? baseKey : nextAvailableContextKey(db, row.skill_id, baseKey);

          update.run(nextKey, serializeJsonValue(parsed), row.skill_id, row.context_key);
        }
      })();
    },
  },
  {
    id: 183,
    name: "add-skill-context-stats-procedural-context-json",
    up: (db) => {
      if (!tableHasColumn(db, "skill_context_stats", "procedural_context_json")) {
        db.exec("ALTER TABLE skill_context_stats ADD COLUMN procedural_context_json TEXT");
      }

      const rows = db
        .prepare(
          `
            SELECT skill_id, context_key
            FROM skill_context_stats
            WHERE procedural_context_json IS NULL
          `,
        )
        .all() as Array<{ skill_id: string; context_key: string }>;
      const update = db.prepare(
        `
          UPDATE skill_context_stats
          SET procedural_context_json = ?
          WHERE skill_id = ? AND context_key = ?
        `,
      );

      db.transaction(() => {
        for (const row of rows) {
          const parsed = parseLegacyProceduralContextKey(row.context_key);

          if (parsed === null) {
            continue;
          }

          update.run(serializeJsonValue(parsed), row.skill_id, row.context_key);
        }
      })();
    },
  },
] as const satisfies readonly Migration[];
