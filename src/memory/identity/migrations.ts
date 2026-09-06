import type { Migration } from "../../storage/sqlite/index.js";
import { tableHasColumn } from "../../storage/sqlite/migrations-utils.js";
import { mapGoalRow } from "../self/shared/sql-mapping.js";
import { IdentityEventRepository } from "./repository.js";

export const identityMigrations = [
  {
    id: 1,
    name: "identity_baseline",
    up: (db) => {
      db.exec(`
        CREATE TABLE identity_events (
          id INTEGER PRIMARY KEY AUTOINCREMENT,
          record_type TEXT NOT NULL,
          record_id TEXT NOT NULL,
          action TEXT NOT NULL,
          old_value_json TEXT,
          new_value_json TEXT,
          reason TEXT,
          provenance_kind TEXT NOT NULL CHECK (
            provenance_kind IN (
              'episodes',
              'manual',
              'system',
              'offline',
              'online',
              'online_reflector'
            )
          ),
          provenance_episode_ids TEXT NOT NULL DEFAULT '[]',
          provenance_stream_entry_ids TEXT NOT NULL DEFAULT '[]',
          provenance_process TEXT,
          review_item_id INTEGER,
          overwrite_without_review INTEGER NOT NULL DEFAULT 0,
          ts INTEGER NOT NULL
        );
        CREATE INDEX idx_identity_events_record_ts
          ON identity_events (record_type, record_id, ts DESC, id DESC);
        CREATE INDEX idx_identity_events_ts
          ON identity_events (ts DESC, id DESC);
      `);
    },
  },
  {
    id: 2,
    name: "repair_unnamed_goal_blocks",
    up: (db) => {
      if (!tableHasColumn(db, "goals", "block_history_json")) return;
      const events = new IdentityEventRepository({ db });
      const rows = db
        .prepare("SELECT * FROM goals WHERE status = 'blocked' AND block_history_json = '[]'")
        .all() as Record<string, unknown>[];
      for (const row of rows) {
        const oldGoal = mapGoalRow(row);
        const next = {
          ...oldGoal,
          status: "active" as const,
          record_version: (oldGoal.record_version ?? 1) + 1,
        };
        db.prepare(
          "UPDATE goals SET status = 'active', record_version = record_version + 1 WHERE id = ?",
        ).run(oldGoal.id);
        events.record({
          record_type: "goal",
          record_id: oldGoal.id,
          action: "unblock",
          old_value: oldGoal,
          new_value: next,
          reason:
            "repair_unnamed_goal_blocks migration: legacy blocked row has no named blocker or block time; reactivated without inventing either",
          provenance: { kind: "system" },
        });
      }
    },
  },
] as const satisfies readonly Migration[];
