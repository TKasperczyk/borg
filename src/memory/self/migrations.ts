import type { Migration } from "../../storage/sqlite/index.js";
import { tableHasColumn } from "../../storage/sqlite/migrations-utils.js";

export const selfMigrations = [
  {
    id: 1,
    name: "self_baseline",
    up: (db) => {
      db.exec(`
        CREATE TABLE "values" (
          id TEXT PRIMARY KEY,
          record_version INTEGER NOT NULL DEFAULT 1,
          label TEXT NOT NULL,
          description TEXT NOT NULL,
          priority REAL NOT NULL,
          created_at INTEGER NOT NULL,
          last_affirmed INTEGER,
          provenance_kind TEXT,
          provenance_episode_ids TEXT,
          provenance_process TEXT,
          state TEXT,
          established_at INTEGER,
          confidence REAL,
          last_tested_at INTEGER,
          last_contradicted_at INTEGER,
          support_count INTEGER,
          contradiction_count INTEGER,
          evidence_episode_ids TEXT
        );
        CREATE TABLE value_sources (
          value_id TEXT NOT NULL,
          episode_id TEXT NOT NULL,
          PRIMARY KEY (value_id, episode_id),
          FOREIGN KEY (value_id) REFERENCES "values"(id) ON DELETE CASCADE
        );
        CREATE TABLE goals (
          id TEXT PRIMARY KEY,
          record_version INTEGER NOT NULL DEFAULT 1,
          description TEXT NOT NULL,
          priority REAL NOT NULL,
          parent_goal_id TEXT,
          status TEXT NOT NULL CHECK (status IN ('active', 'done', 'abandoned', 'blocked')),
          progress_notes TEXT,
          created_at INTEGER NOT NULL,
          target_at INTEGER,
          provenance_kind TEXT,
          provenance_episode_ids TEXT,
          provenance_process TEXT,
          last_progress_ts INTEGER,
          audience_entity_id TEXT,
          owner_entity_id TEXT,
          source_stream_entry_ids TEXT, canonicalized_by_artifact_entry_id TEXT NULL,
          FOREIGN KEY (parent_goal_id) REFERENCES goals(id) ON DELETE SET NULL
        );
        CREATE INDEX idx_goals_audience_status_priority
          ON goals (audience_entity_id, status, priority DESC, created_at ASC);
        CREATE INDEX idx_goals_owner_status_priority
          ON goals (owner_entity_id, status, priority DESC, created_at ASC);
        CREATE TABLE traits (
          label TEXT PRIMARY KEY,
          record_version INTEGER NOT NULL DEFAULT 1,
          strength REAL NOT NULL,
          last_reinforced INTEGER NOT NULL,
          last_decayed INTEGER,
          id TEXT,
          provenance_kind TEXT,
          provenance_episode_ids TEXT,
          provenance_process TEXT,
          state TEXT,
          established_at INTEGER,
          confidence REAL,
          last_tested_at INTEGER,
          last_contradicted_at INTEGER,
          support_count INTEGER,
          contradiction_count INTEGER,
          evidence_episode_ids TEXT
        );
        CREATE UNIQUE INDEX idx_traits_id
          ON traits (id)
          WHERE id IS NOT NULL;
        CREATE TABLE autobiographical_periods (
          id TEXT PRIMARY KEY,
          record_version INTEGER NOT NULL DEFAULT 1,
          label TEXT NOT NULL,
          start_ts INTEGER NOT NULL,
          end_ts INTEGER,
          narrative TEXT NOT NULL,
          key_episode_ids TEXT NOT NULL,
          themes TEXT NOT NULL,
          created_at INTEGER NOT NULL,
          last_updated INTEGER NOT NULL,
          disclosure_label TEXT,
          provenance_kind TEXT,
          provenance_episode_ids TEXT,
          provenance_process TEXT
        );
        CREATE UNIQUE INDEX autobiographical_single_open
          ON autobiographical_periods (CASE WHEN end_ts IS NULL THEN 1 END)
          WHERE end_ts IS NULL;
        CREATE INDEX idx_autobiographical_periods_end
          ON autobiographical_periods (end_ts);
        CREATE INDEX idx_autobiographical_periods_start
          ON autobiographical_periods (start_ts DESC);
        CREATE TABLE growth_markers (
          id TEXT PRIMARY KEY,
          record_version INTEGER NOT NULL DEFAULT 1,
          ts INTEGER NOT NULL,
          category TEXT NOT NULL CHECK (
            category IN ('skill', 'value', 'habit', 'relationship', 'understanding')
          ),
          what_changed TEXT NOT NULL,
          before_description TEXT,
          after_description TEXT,
          evidence_episode_ids TEXT NOT NULL,
          disclosure_label TEXT,
          confidence REAL NOT NULL,
          source_process TEXT NOT NULL,
          created_at INTEGER NOT NULL,
          provenance_kind TEXT,
          provenance_episode_ids TEXT,
          provenance_process TEXT
        );
        CREATE INDEX idx_growth_markers_category
          ON growth_markers (category);
        CREATE INDEX idx_growth_markers_ts
          ON growth_markers (ts DESC);
        CREATE TABLE open_questions (
          id TEXT PRIMARY KEY,
          record_version INTEGER NOT NULL DEFAULT 1,
          question TEXT NOT NULL,
          urgency REAL NOT NULL,
          status TEXT NOT NULL CHECK (status IN ('open', 'resolved', 'abandoned')),
          goal_id TEXT,
          related_episode_ids TEXT NOT NULL,
          related_semantic_node_ids TEXT NOT NULL,
          source TEXT NOT NULL CHECK (
            source IN (
              'user',
              'reflection',
              'contradiction',
              'ruminator',
              'overseer',
              'autonomy',
              'deliberator'
            )
          ),
          created_at INTEGER NOT NULL,
          last_touched INTEGER NOT NULL,
          resolution_evidence_episode_ids TEXT NOT NULL DEFAULT '[]',
          resolution_evidence_stream_entry_ids TEXT NOT NULL DEFAULT '[]',
          resolution_disclosure_label TEXT,
          resolution_note TEXT,
          resolved_at INTEGER,
          abandoned_reason TEXT,
          abandoned_at INTEGER,
          dedupe_key TEXT,
          provenance_kind TEXT,
          provenance_episode_ids TEXT,
          provenance_process TEXT,
          audience_entity_id TEXT,
          unresolved_rumination_ticks INTEGER NOT NULL DEFAULT 0,
          last_ruminated_at INTEGER
        , resolved_by_artifact_entry_id TEXT NULL);
        CREATE INDEX idx_open_questions_audience_status_urgency
          ON open_questions (audience_entity_id, status, urgency DESC, last_touched DESC);
        CREATE UNIQUE INDEX idx_open_questions_dedupe_key
          ON open_questions (dedupe_key);
        CREATE INDEX idx_open_questions_status_urgency
          ON open_questions (status, urgency DESC, last_touched DESC);
        CREATE TABLE trait_reinforcement_events (
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
        CREATE INDEX idx_trait_reinforcement_events_trait_ts
          ON trait_reinforcement_events (trait_id, ts DESC, id DESC);
        CREATE TABLE value_reinforcement_events (
          id INTEGER PRIMARY KEY AUTOINCREMENT,
          value_id TEXT NOT NULL,
          ts INTEGER NOT NULL,
          provenance_kind TEXT NOT NULL CHECK (
            provenance_kind IN ('episodes', 'manual', 'system', 'offline', 'online')
          ),
          provenance_episode_ids TEXT NOT NULL DEFAULT '[]',
          provenance_process TEXT
        );
        CREATE INDEX idx_value_reinforcement_events_value_ts
          ON value_reinforcement_events (value_id, ts DESC, id DESC);
        CREATE TABLE value_contradiction_events (
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
        CREATE INDEX idx_value_contradiction_events_value_ts
          ON value_contradiction_events (value_id, ts DESC, id DESC);
        CREATE TABLE trait_contradiction_events (
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
        CREATE INDEX idx_trait_contradiction_events_trait_ts
          ON trait_contradiction_events (trait_id, ts DESC, id DESC);
      `);
    },
  },
  {
    id: 2,
    name: "self_disclosure_labels",
    up: (db) => {
      if (!tableHasColumn(db, "autobiographical_periods", "disclosure_label")) {
        db.exec("ALTER TABLE autobiographical_periods ADD COLUMN disclosure_label TEXT");
      }
      if (!tableHasColumn(db, "growth_markers", "disclosure_label")) {
        db.exec("ALTER TABLE growth_markers ADD COLUMN disclosure_label TEXT");
      }
      if (!tableHasColumn(db, "open_questions", "resolution_disclosure_label")) {
        db.exec("ALTER TABLE open_questions ADD COLUMN resolution_disclosure_label TEXT");
      }
    },
  },
  {
    id: 3,
    name: "open_question_ruminations",
    up: (db) => {
      db.exec(`
        CREATE TABLE IF NOT EXISTS open_question_ruminations (
          id INTEGER PRIMARY KEY AUTOINCREMENT,
          open_question_id TEXT NOT NULL,
          note TEXT NOT NULL,
          tensions TEXT NOT NULL DEFAULT '[]',
          connected_open_question_ids TEXT NOT NULL DEFAULT '[]',
          evidence_episode_ids TEXT NOT NULL DEFAULT '[]',
          evidence_stream_entry_ids TEXT NOT NULL DEFAULT '[]',
          source_process TEXT NOT NULL,
          source_run_id TEXT,
          source_turn_id TEXT,
          provenance_kind TEXT NOT NULL,
          provenance_episode_ids TEXT NOT NULL DEFAULT '[]',
          provenance_process TEXT,
          created_at INTEGER NOT NULL,
          FOREIGN KEY (open_question_id) REFERENCES open_questions(id) ON DELETE CASCADE
        );
        CREATE INDEX IF NOT EXISTS idx_open_question_ruminations_question_created
          ON open_question_ruminations (open_question_id, created_at DESC, id DESC);
      `);
    },
  },
] as const satisfies readonly Migration[];
