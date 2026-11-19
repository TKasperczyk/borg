-- Live table DDL at dev cc7877c4; schema only, no live rows.
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
              'associator',
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
          last_ruminated_at INTEGER,
          resolved_by_artifact_entry_id TEXT NULL
        , disclosure_label TEXT, consecutive_refusals INTEGER NOT NULL DEFAULT 0, last_rumination_attempt_outcome TEXT, last_rumination_attempt_at INTEGER, refusal_backoff_until INTEGER, resolution_corrections TEXT NOT NULL DEFAULT '[]', resolution_version INTEGER NULL);
CREATE INDEX idx_open_questions_audience_status_urgency
          ON open_questions (audience_entity_id, status, urgency DESC, last_touched DESC);
CREATE UNIQUE INDEX idx_open_questions_dedupe_key
          ON open_questions (dedupe_key);
CREATE INDEX idx_open_questions_status_urgency
          ON open_questions (status, urgency DESC, last_touched DESC);
