import type { Migration } from "../../storage/sqlite/index.js";

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
      const rows = db
        .prepare("SELECT id, refs FROM review_queue ORDER BY id ASC")
        .all() as Array<{ id: number; refs: string | null }>;
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
];
