import type { Migration } from "../../storage/sqlite/index.js";

export const creatorDirectiveMigrations = [
  {
    id: 1,
    name: "creator_directives_initial_schema",
    up: (db) => {
      db.exec(`
        CREATE TABLE creator_directives (
          id TEXT PRIMARY KEY,
          record_version INTEGER NOT NULL DEFAULT 1,
          status TEXT NOT NULL CHECK (status IN ('active', 'superseded', 'revoked')),
          kind TEXT NOT NULL CHECK (
            kind IN (
              'self_identity',
              'subject_fact',
              'disclosure_boundary',
              'response_policy',
              'routing_instruction'
            )
          ),
          created_by_entity_id TEXT NOT NULL,
          source_session_id TEXT NOT NULL,
          authorization_stream_entry_ids TEXT NOT NULL,
          content_source_stream_entry_ids TEXT NOT NULL,
          subject_kind TEXT NOT NULL CHECK (
            subject_kind IN ('borg_self', 'entity', 'system', 'unknown')
          ),
          subject_entity_id TEXT NULL,
          semantic_slot TEXT NULL CHECK (
            semantic_slot IS NULL OR semantic_slot IN ('public_name')
          ),
          canonical_fact TEXT NULL,
          operational_directive TEXT NOT NULL,
          content_scope TEXT NOT NULL CHECK (
            content_scope IN ('operator_only', 'public', 'allow_list', 'subject_only', 'all_except')
          ),
          allowed_entity_ids TEXT NOT NULL DEFAULT '[]',
          excluded_entity_ids TEXT NOT NULL DEFAULT '[]',
          subject_may_know INTEGER NULL CHECK (
            subject_may_know IS NULL OR subject_may_know IN (0, 1)
          ),
          mention_policy TEXT NOT NULL CHECK (
            mention_policy IN (
              'proactive',
              'answer_if_asked',
              'only_if_topic_raised',
              'never_mention'
            )
          ),
          denied_audience_behavior TEXT NOT NULL CHECK (
            denied_audience_behavior IN ('omit', 'render_boundary_when_relevant')
          ),
          boundary_prompt TEXT NULL,
          topic_tags TEXT NOT NULL DEFAULT '[]',
          priority INTEGER NOT NULL,
          superseded_by TEXT NULL,
          revoked_reason TEXT NULL,
          created_at INTEGER NOT NULL,
          updated_at INTEGER NOT NULL
        );

        CREATE INDEX IF NOT EXISTS creator_directives_status_priority_idx
          ON creator_directives(status, priority DESC, created_at ASC);
        CREATE INDEX IF NOT EXISTS creator_directives_kind_status_idx
          ON creator_directives(kind, status);
        CREATE INDEX IF NOT EXISTS creator_directives_created_by_idx
          ON creator_directives(created_by_entity_id, status);
        CREATE INDEX IF NOT EXISTS creator_directives_source_session_idx
          ON creator_directives(source_session_id, status);
        CREATE INDEX IF NOT EXISTS creator_directives_subject_idx
          ON creator_directives(subject_kind, subject_entity_id, status);
        CREATE INDEX IF NOT EXISTS creator_directives_slot_conflict_idx
          ON creator_directives(status, kind, subject_kind, subject_entity_id, semantic_slot);
        CREATE INDEX IF NOT EXISTS creator_directives_content_scope_idx
          ON creator_directives(content_scope, status);
      `);
    },
  },
] as const satisfies readonly Migration[];
