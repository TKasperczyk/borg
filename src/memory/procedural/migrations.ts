import type { Migration } from "../../storage/sqlite/index.js";

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
] as const satisfies readonly Migration[];
