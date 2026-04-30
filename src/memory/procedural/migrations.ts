import type { Migration } from "../../storage/sqlite/index.js";

export const proceduralMigrations = [
  {
    id: 1,
    name: "procedural_initial_schema",
    up: (db) => {
      db.exec(`
        CREATE TABLE skills (
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
          updated_at INTEGER NOT NULL,
          status TEXT NOT NULL DEFAULT 'active',
          superseded_by TEXT NOT NULL DEFAULT '[]',
          superseded_at INTEGER,
          splitting_at INTEGER,
          last_split_attempt_at INTEGER,
          split_failure_count INTEGER NOT NULL DEFAULT 0,
          last_split_error TEXT,
          requires_manual_review INTEGER NOT NULL DEFAULT 0
        );

        CREATE INDEX IF NOT EXISTS idx_skills_updated_at
          ON skills (updated_at DESC);
        CREATE INDEX IF NOT EXISTS idx_skills_status_updated
          ON skills (status, updated_at DESC);
        CREATE INDEX IF NOT EXISTS idx_skills_split_attempt
          ON skills (status, last_split_attempt_at DESC);
        CREATE INDEX IF NOT EXISTS idx_skills_split_failures
          ON skills (status, split_failure_count, updated_at DESC);
        CREATE INDEX IF NOT EXISTS idx_skills_manual_review
          ON skills (requires_manual_review, updated_at DESC);

        CREATE TABLE procedural_evidence (
          id TEXT PRIMARY KEY,
          pending_attempt_snapshot TEXT NOT NULL,
          classification TEXT NOT NULL,
          evidence_text TEXT NOT NULL,
          resolved_episode_ids TEXT NOT NULL,
          audience_entity_id TEXT,
          consumed_at INTEGER,
          created_at INTEGER NOT NULL,
          grounded INTEGER NOT NULL DEFAULT 1,
          skill_actually_applied INTEGER NOT NULL DEFAULT 1,
          procedural_context TEXT
        );

        CREATE INDEX IF NOT EXISTS idx_procedural_evidence_unconsumed
          ON procedural_evidence (consumed_at, created_at);
        CREATE INDEX IF NOT EXISTS idx_procedural_evidence_audience
          ON procedural_evidence (audience_entity_id);

        CREATE TABLE skill_context_stats (
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
      `);
    },
  },
] as const satisfies readonly Migration[];
