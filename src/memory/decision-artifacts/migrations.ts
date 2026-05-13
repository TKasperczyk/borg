import type { Migration } from "../../storage/sqlite/index.js";

export const decisionArtifactMigrations = [
  {
    id: 1,
    name: "decision_artifacts_initial_schema",
    up: `
      CREATE TABLE decision_artifacts (
        audience_entity_id TEXT PRIMARY KEY,
        record_version INTEGER NOT NULL DEFAULT 1,
        created_at INTEGER NOT NULL,
        updated_at INTEGER NOT NULL,
        last_compiled_at INTEGER NULL,
        last_compiled_stream_entry_id TEXT NULL
      );

      CREATE TABLE decision_artifact_entries (
        id TEXT PRIMARY KEY,
        audience_entity_id TEXT NOT NULL,
        kind TEXT NOT NULL CHECK (
          kind IN ('locked', 'live', 'tentative', 'invalidated', 'pending')
        ),
        text TEXT NOT NULL,
        owner_entity_id TEXT NULL,
        provenance_stream_entry_ids TEXT NOT NULL,
        last_updated_stream_entry_ids TEXT NOT NULL,
        created_at INTEGER NOT NULL,
        last_updated_at INTEGER NOT NULL,
        superseded_by_id TEXT NULL,
        rank INTEGER NOT NULL DEFAULT 0,
        FOREIGN KEY (audience_entity_id)
          REFERENCES decision_artifacts(audience_entity_id)
          ON DELETE CASCADE,
        FOREIGN KEY (superseded_by_id)
          REFERENCES decision_artifact_entries(id)
          ON DELETE RESTRICT
      );

      CREATE INDEX IF NOT EXISTS idx_decision_artifact_entries_audience_rank
        ON decision_artifact_entries(audience_entity_id, rank ASC, created_at ASC);
      CREATE INDEX IF NOT EXISTS idx_decision_artifact_entries_kind
        ON decision_artifact_entries(kind);
      CREATE INDEX IF NOT EXISTS idx_decision_artifact_entries_superseded
        ON decision_artifact_entries(superseded_by_id);
    `,
  },
] as const satisfies readonly Migration[];
