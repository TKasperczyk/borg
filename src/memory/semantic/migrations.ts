import type { Migration } from "../../storage/sqlite/index.js";
import { REVIEW_QUEUE_BASELINE_SQL } from "../review-queue/migrations.js";

export const semanticMigrations = [
  {
    id: 1,
    name: "semantic_baseline",
    up: (db) => {
      db.exec(`
        CREATE TABLE semantic_nodes (
          id TEXT PRIMARY KEY,
          kind TEXT NOT NULL,
          label TEXT NOT NULL,
          description TEXT NOT NULL,
          domain TEXT NULL,
          aliases TEXT NOT NULL,
          confidence REAL NOT NULL,
          source_episode_ids TEXT NOT NULL,
          created_at INTEGER NOT NULL,
          updated_at INTEGER NOT NULL,
          last_verified_at INTEGER NOT NULL,
          archived INTEGER NOT NULL DEFAULT 0,
          superseded_by TEXT NULL
        , status TEXT NOT NULL DEFAULT 'active', corrected_by TEXT NULL, superseded_at INTEGER NULL, observation_metadata TEXT NULL);
        CREATE INDEX semantic_nodes_kind_idx
          ON semantic_nodes(kind);
        CREATE INDEX semantic_nodes_label_idx
          ON semantic_nodes(label);
        CREATE INDEX semantic_nodes_observation_metadata_idx
          ON semantic_nodes(observation_metadata)
          WHERE observation_metadata IS NOT NULL;
        CREATE INDEX semantic_nodes_status_updated_idx
          ON semantic_nodes(status, updated_at DESC);
        CREATE TABLE semantic_edges (
          id TEXT PRIMARY KEY,
          from_node_id TEXT NOT NULL,
          to_node_id TEXT NOT NULL,
          relation TEXT NOT NULL,
          confidence REAL NOT NULL,
          evidence_episode_ids TEXT NOT NULL,
          created_at INTEGER NOT NULL,
          last_verified_at INTEGER NOT NULL,
          valid_from INTEGER NOT NULL,
          valid_to INTEGER NULL,
          invalidated_at INTEGER NULL,
          invalidated_by_edge_id TEXT NULL,
          invalidated_by_review_id INTEGER NULL,
          invalidated_by_process TEXT NULL,
          invalidated_reason TEXT NULL
        );
        CREATE INDEX semantic_edges_from_relation_validity_idx
          ON semantic_edges(from_node_id, relation, valid_from, valid_to);
        CREATE INDEX semantic_edges_invalidated_at_idx
          ON semantic_edges(invalidated_at);
        CREATE UNIQUE INDEX semantic_edges_open_unique_idx
          ON semantic_edges(from_node_id, to_node_id, relation)
          WHERE valid_to IS NULL;
        CREATE INDEX semantic_edges_to_relation_validity_idx
          ON semantic_edges(to_node_id, relation, valid_from, valid_to);
        CREATE TRIGGER semantic_edges_invalidation_outbox_insert
        AFTER UPDATE OF valid_to ON semantic_edges
        WHEN OLD.valid_to IS NULL AND NEW.valid_to IS NOT NULL
        BEGIN
          INSERT INTO semantic_edge_invalidation_events (
            edge_id,
            valid_to,
            invalidated_at,
            processed_at
          ) VALUES (
            NEW.id,
            NEW.valid_to,
            COALESCE(NEW.invalidated_at, NEW.valid_to),
            NULL
          );
        END;
        ${REVIEW_QUEUE_BASELINE_SQL}
        CREATE TABLE semantic_belief_dependencies (
          target_type TEXT NOT NULL CHECK (target_type IN ('semantic_node', 'semantic_edge')),
          target_id TEXT NOT NULL,
          source_edge_id TEXT NOT NULL,
          dependency_kind TEXT NOT NULL CHECK (dependency_kind IN ('supports', 'derived_from')),
          created_at INTEGER NOT NULL,
          PRIMARY KEY (target_type, target_id, source_edge_id, dependency_kind)
        );
        CREATE INDEX semantic_belief_dependencies_source_idx
          ON semantic_belief_dependencies(source_edge_id);
        CREATE TABLE semantic_edge_invalidation_events (
          id INTEGER PRIMARY KEY AUTOINCREMENT,
          edge_id TEXT NOT NULL,
          valid_to INTEGER NOT NULL,
          invalidated_at INTEGER NOT NULL,
          processed_at INTEGER NULL
        );
        CREATE TABLE semantic_node_vector_sync_outbox (
          id INTEGER PRIMARY KEY AUTOINCREMENT,
          node_id TEXT NOT NULL,
          reason TEXT NOT NULL,
          created_at INTEGER NOT NULL,
          generation INTEGER NOT NULL DEFAULT 1,
          attempts INTEGER NOT NULL DEFAULT 0,
          last_attempt_at INTEGER NULL,
          last_error TEXT NULL,
          UNIQUE(node_id)
        );
        CREATE INDEX semantic_node_vector_sync_outbox_created_idx
          ON semantic_node_vector_sync_outbox(created_at, id);
      `);
    },
  },
] as const satisfies readonly Migration[];
