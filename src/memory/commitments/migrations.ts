import type { Migration } from "../../storage/sqlite/index.js";

export const commitmentMigrations = [
  {
    id: 1,
    name: "commitment_baseline",
    up: (db) => {
      db.exec(`
        CREATE TABLE entities (
          id TEXT PRIMARY KEY,
          canonical_name TEXT NOT NULL,
          aliases TEXT NOT NULL,
          created_at INTEGER NOT NULL
        , name_provenance TEXT NOT NULL DEFAULT 'unknown', kind TEXT NULL CHECK (
              kind IS NULL OR kind IN ('person', 'group', 'self', 'abstract')
            ), borg_role TEXT NULL CHECK (
            borg_role IS NULL OR borg_role IN ('creator')
          ));
        CREATE INDEX entities_kind_idx
          ON entities(kind);
        CREATE INDEX entities_name_idx
          ON entities(canonical_name);
        CREATE TABLE commitments (
          id TEXT PRIMARY KEY,
          record_version INTEGER NOT NULL DEFAULT 1,
          type TEXT NOT NULL,
          directive TEXT NOT NULL,
          priority INTEGER NOT NULL,
          made_to_entity TEXT NULL,
          restricted_audience TEXT NULL,
          about_entity TEXT NULL,
          committed_by_entity_id TEXT NULL,
          source_episode_ids TEXT NOT NULL,
          created_at INTEGER NOT NULL,
          expires_at INTEGER NULL,
          revoked_at INTEGER NULL,
          superseded_by TEXT NULL,
          provenance_kind TEXT,
          provenance_episode_ids TEXT,
          provenance_process TEXT,
          expired_at INTEGER,
          revoked_reason TEXT,
          revoke_provenance_kind TEXT,
          revoke_provenance_episode_ids TEXT,
          revoke_provenance_process TEXT
        , source_stream_entry_ids TEXT NULL, directive_family TEXT NULL, last_reinforced_at INTEGER NULL, closure_pressure_relevance TEXT NOT NULL DEFAULT 'neutral', canonicalized_by_artifact_entry_id TEXT NULL, kind TEXT NOT NULL DEFAULT 'assistant_commitment' CHECK (
              kind IN (
                'assistant_commitment',
                'audience_rule',
                'participant_preference',
                'boundary',
                'process_norm'
              )
            ), enforcement_class TEXT NULL CHECK (
              enforcement_class IS NULL OR enforcement_class IN ('critical', 'advisory')
            ), critical_domain TEXT NULL CHECK (
              critical_domain IS NULL OR critical_domain IN (
                'privacy',
                'audience_scope',
                'safety',
                'explicit_no_disclosure',
                'internal_tool_hygiene'
              )
            ));
        CREATE INDEX commitments_about_idx
          ON commitments(about_entity);
        CREATE INDEX commitments_audience_idx
          ON commitments(restricted_audience);
        CREATE INDEX commitments_committed_by_idx
          ON commitments(committed_by_entity_id);
        CREATE INDEX commitments_critical_domain_idx
          ON commitments(critical_domain);
        CREATE INDEX commitments_directive_family_idx
          ON commitments(directive_family, restricted_audience, made_to_entity);
        CREATE INDEX commitments_enforcement_class_idx
          ON commitments(enforcement_class);
        CREATE INDEX commitments_kind_idx
          ON commitments(kind);
      `);
    },
  },
  {
    id: 2,
    name: "entity_external_ids",
    up: (db) => {
      db.exec(`
        CREATE TABLE entity_external_ids (
          source TEXT NOT NULL,
          external_id TEXT NOT NULL,
          entity_id TEXT NOT NULL,
          created_at INTEGER NOT NULL,
          updated_at INTEGER NOT NULL,
          PRIMARY KEY (source, external_id),
          FOREIGN KEY (entity_id) REFERENCES entities(id) ON DELETE CASCADE
        );
        CREATE INDEX entity_external_ids_entity_idx
          ON entity_external_ids(entity_id);
      `);
    },
  },
  {
    id: 3,
    name: "commitments_updated_at",
    up: (db) => {
      db.exec(`
        ALTER TABLE commitments
        ADD COLUMN updated_at INTEGER NULL;
      `);
    },
  },
] as const satisfies readonly Migration[];
