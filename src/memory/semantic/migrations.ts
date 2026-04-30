import type { Migration, SqliteDatabase } from "../../storage/sqlite/index.js";
import { tableExists, tableHasColumn } from "../../storage/sqlite/migrations-utils.js";

function createSemanticEdgeValidityIndexes(db: SqliteDatabase): void {
  db.exec(`
    CREATE UNIQUE INDEX IF NOT EXISTS semantic_edges_open_unique_idx
      ON semantic_edges(from_node_id, to_node_id, relation)
      WHERE valid_to IS NULL;
    CREATE INDEX IF NOT EXISTS semantic_edges_from_relation_validity_idx
      ON semantic_edges(from_node_id, relation, valid_from, valid_to);
    CREATE INDEX IF NOT EXISTS semantic_edges_to_relation_validity_idx
      ON semantic_edges(to_node_id, relation, valid_from, valid_to);
    CREATE INDEX IF NOT EXISTS semantic_edges_invalidated_at_idx
      ON semantic_edges(invalidated_at);
  `);
}

export const semanticMigrations: Migration[] = [
  {
    id: 130,
    name: "semantic_nodes_edges_review_queue",
    up: `
      CREATE TABLE IF NOT EXISTS semantic_nodes (
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
      );

      CREATE INDEX IF NOT EXISTS semantic_nodes_kind_idx
        ON semantic_nodes(kind);
      CREATE INDEX IF NOT EXISTS semantic_nodes_label_idx
        ON semantic_nodes(label);

      CREATE TABLE IF NOT EXISTS semantic_edges (
        id TEXT PRIMARY KEY,
        from_node_id TEXT NOT NULL,
        to_node_id TEXT NOT NULL,
        relation TEXT NOT NULL,
        confidence REAL NOT NULL,
        evidence_episode_ids TEXT NOT NULL,
        created_at INTEGER NOT NULL,
        last_verified_at INTEGER NOT NULL,
        UNIQUE(from_node_id, to_node_id, relation)
      );

      CREATE INDEX IF NOT EXISTS semantic_edges_from_relation_idx
        ON semantic_edges(from_node_id, relation);
      CREATE INDEX IF NOT EXISTS semantic_edges_to_relation_idx
        ON semantic_edges(to_node_id, relation);

      CREATE TABLE IF NOT EXISTS review_queue (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        kind TEXT NOT NULL,
        refs TEXT NOT NULL,
        reason TEXT NOT NULL,
        created_at INTEGER NOT NULL,
        resolved_at INTEGER NULL,
        resolution TEXT NULL
      );

      CREATE INDEX IF NOT EXISTS review_queue_kind_idx
        ON review_queue(kind);
      CREATE INDEX IF NOT EXISTS review_queue_open_idx
        ON review_queue(resolved_at);
    `,
  },
  {
    id: 131,
    name: "backfill-review-queue-proposed-provenance",
    up: (db) => {
      const rows = db.prepare("SELECT id, refs FROM review_queue ORDER BY id ASC").all() as Array<{
        id: number;
        refs: string | null;
      }>;
      const update = db.prepare("UPDATE review_queue SET refs = ? WHERE id = ?");

      for (const row of rows) {
        let refs: Record<string, unknown>;

        try {
          const parsed = JSON.parse(row.refs ?? "{}") as unknown;
          refs =
            parsed !== null && typeof parsed === "object" && !Array.isArray(parsed)
              ? (parsed as Record<string, unknown>)
              : {};
        } catch {
          refs = {};
        }

        if (refs.proposed_provenance !== undefined) {
          continue;
        }

        refs.proposed_provenance =
          refs.provenance ??
          ({
            kind: "manual",
          } satisfies { kind: "manual" });
        update.run(JSON.stringify(refs), row.id);
      }
    },
  },
  {
    id: 132,
    name: "semantic_nodes_domain",
    up: (db) => {
      if (tableHasColumn(db, "semantic_nodes", "domain")) {
        return;
      }

      db.prepare("ALTER TABLE semantic_nodes ADD COLUMN domain TEXT NULL").run();
    },
  },
  {
    id: 133,
    name: "semantic_edge_validity",
    up: (db) => {
      if (!tableExists(db, "semantic_edges")) {
        return;
      }

      const hasValidityColumns =
        tableHasColumn(db, "semantic_edges", "valid_from") &&
        tableHasColumn(db, "semantic_edges", "valid_to") &&
        tableHasColumn(db, "semantic_edges", "invalidated_at") &&
        tableHasColumn(db, "semantic_edges", "invalidated_by_edge_id") &&
        tableHasColumn(db, "semantic_edges", "invalidated_by_review_id") &&
        tableHasColumn(db, "semantic_edges", "invalidated_by_process") &&
        tableHasColumn(db, "semantic_edges", "invalidated_reason");

      if (hasValidityColumns) {
        createSemanticEdgeValidityIndexes(db);
        return;
      }

      db.exec(`
        CREATE TABLE semantic_edges__next (
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

        INSERT INTO semantic_edges__next (
          id,
          from_node_id,
          to_node_id,
          relation,
          confidence,
          evidence_episode_ids,
          created_at,
          last_verified_at,
          valid_from,
          valid_to,
          invalidated_at,
          invalidated_by_edge_id,
          invalidated_by_review_id,
          invalidated_by_process,
          invalidated_reason
        )
        SELECT
          id,
          from_node_id,
          to_node_id,
          relation,
          confidence,
          evidence_episode_ids,
          created_at,
          last_verified_at,
          created_at,
          NULL,
          NULL,
          NULL,
          NULL,
          NULL,
          NULL
        FROM semantic_edges;

        DROP TABLE semantic_edges;
        ALTER TABLE semantic_edges__next RENAME TO semantic_edges;
      `);

      createSemanticEdgeValidityIndexes(db);
    },
  },
  {
    id: 134,
    name: "supports_edge_direction_flip",
    up: (db) => {
      if (!tableExists(db, "semantic_edges")) {
        return;
      }

      // Sprint 52: supports edges were previously created as
      // `insight --supports--> target`, which is backwards from the
      // natural reading "X is evidence for Y". Retrieval walks supports
      // OUT, so a query that matches the original target node could not
      // surface the new insight. Flip existing supports edges so
      // `from --supports--> to` reads as "from is evidence supporting to".
      //
      // Done row-by-row with a reverse-edge check to avoid violating the
      // partial unique index on (from, to, relation) WHERE valid_to IS NULL.
      const openSupports = db
        .prepare(
          "SELECT id, from_node_id, to_node_id FROM semantic_edges WHERE relation = 'supports' AND valid_to IS NULL",
        )
        .all() as Array<{ id: string; from_node_id: string; to_node_id: string }>;
      const reverseOpenStmt = db.prepare(
        "SELECT id FROM semantic_edges WHERE relation = 'supports' AND valid_to IS NULL AND from_node_id = ? AND to_node_id = ?",
      );
      const updateStmt = db.prepare(
        "UPDATE semantic_edges SET from_node_id = ?, to_node_id = ? WHERE id = ?",
      );

      for (const edge of openSupports) {
        const conflict = reverseOpenStmt.get(edge.to_node_id, edge.from_node_id) as
          | { id: string }
          | undefined;

        if (conflict !== undefined && conflict.id !== edge.id) {
          continue;
        }

        updateStmt.run(edge.to_node_id, edge.from_node_id, edge.id);
      }

      db.exec(
        "UPDATE semantic_edges SET from_node_id = to_node_id, to_node_id = from_node_id WHERE relation = 'supports' AND valid_to IS NOT NULL",
      );
    },
  },
  {
    id: 135,
    name: "semantic_belief_revision_substrate",
    up: (db) => {
      db.exec(`
        CREATE TABLE IF NOT EXISTS semantic_belief_dependencies (
          target_type TEXT NOT NULL CHECK (target_type IN ('semantic_node', 'semantic_edge')),
          target_id TEXT NOT NULL,
          source_edge_id TEXT NOT NULL,
          dependency_kind TEXT NOT NULL CHECK (dependency_kind IN ('supports', 'derived_from')),
          created_at INTEGER NOT NULL,
          PRIMARY KEY (target_type, target_id, source_edge_id, dependency_kind)
        );

        CREATE INDEX IF NOT EXISTS semantic_belief_dependencies_source_idx
          ON semantic_belief_dependencies(source_edge_id);

        CREATE TABLE IF NOT EXISTS semantic_edge_invalidation_events (
          id INTEGER PRIMARY KEY AUTOINCREMENT,
          edge_id TEXT NOT NULL,
          valid_to INTEGER NOT NULL,
          invalidated_at INTEGER NOT NULL,
          processed_at INTEGER NULL
        );
      `);

      if (!tableExists(db, "semantic_edges")) {
        return;
      }

      db.exec(`
        CREATE TRIGGER IF NOT EXISTS semantic_edges_invalidation_outbox_insert
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

        INSERT OR IGNORE INTO semantic_belief_dependencies (
          target_type,
          target_id,
          source_edge_id,
          dependency_kind,
          created_at
        )
        SELECT
          'semantic_node',
          to_node_id,
          id,
          'supports',
          created_at
        FROM semantic_edges
        WHERE relation = 'supports'
          AND valid_to IS NULL;
      `);
    },
  },
  {
    id: 136,
    name: "review_queue_belief_revision_target_index",
    up: (db) => {
      if (!tableExists(db, "review_queue")) {
        return;
      }

      db.exec(`
        CREATE INDEX IF NOT EXISTS review_queue_belief_revision_target_idx
          ON review_queue (
            json_extract(refs, '$.target_type'),
            json_extract(refs, '$.target_id'),
            created_at DESC,
            id DESC
          )
          WHERE kind = 'belief_revision'
            AND resolved_at IS NULL;
      `);
    },
  },
  {
    id: 137,
    name: "semantic_node_vector_sync_outbox",
    up: `
      CREATE TABLE IF NOT EXISTS semantic_node_vector_sync_outbox (
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

      CREATE INDEX IF NOT EXISTS semantic_node_vector_sync_outbox_created_idx
        ON semantic_node_vector_sync_outbox(created_at, id);
    `,
  },
  {
    id: 138,
    name: "semantic_node_vector_sync_outbox_generation",
    up: (db) => {
      if (
        !tableExists(db, "semantic_node_vector_sync_outbox") ||
        tableHasColumn(db, "semantic_node_vector_sync_outbox", "generation")
      ) {
        return;
      }

      db.prepare(
        "ALTER TABLE semantic_node_vector_sync_outbox ADD COLUMN generation INTEGER NOT NULL DEFAULT 1",
      ).run();
    },
  },
];
