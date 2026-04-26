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
];
