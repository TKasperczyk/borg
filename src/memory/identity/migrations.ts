import type { Migration } from "../../storage/sqlite/index.js";
import { tableHasColumn } from "../../storage/sqlite/migrations-utils.js";
import { mapGoalRow } from "../self/shared/sql-mapping.js";
import { IdentityEventRepository } from "./repository.js";
import { legacyUnknownGoalBlock } from "../self/goal-blocks.js";
import { serializeJsonValue } from "../../util/json-value.js";

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
          block_history: [legacyUnknownGoalBlock()],
          record_version: (oldGoal.record_version ?? 1) + 1,
        };
        db.prepare(
          "UPDATE goals SET block_history_json = ?, record_version = record_version + 1 WHERE id = ?",
        ).run(serializeJsonValue(next.block_history), oldGoal.id);
        events.record({
          record_type: "goal",
          record_id: oldGoal.id,
          action: "legacy_block_metadata",
          old_value: oldGoal,
          new_value: next,
          reason:
            "repair_unnamed_goal_blocks migration: preserve blocked status; legacy blocker, attempt and block time not recorded",
          provenance: { kind: "system" },
        });
      }
    },
  },
  {
    id: 3,
    name: "goal_block_labels_and_deadline_basis",
    up: (db) => {
      if (!tableHasColumn(db, "goals", "target_assigned_at")) return;
      const events = new IdentityEventRepository({ db });
      const rows = db.prepare("SELECT * FROM goals").all() as Record<string, unknown>[];
      for (const row of rows) {
        const goal = mapGoalRow(row);
        // An already-applied version of migration 2 can be reversed exactly
        // only while its resulting row is still untouched. Later real changes
        // are preserved. The original migration event remains in the audit.
        const mistakenRelease = db
          .prepare(
            `
          SELECT id, new_value_json FROM identity_events
          WHERE record_type = 'goal' AND record_id = ? AND provenance_kind = 'system'
            AND action = 'unblock' AND reason = ? ORDER BY id DESC LIMIT 1
        `,
          )
          .get(
            goal.id,
            "repair_unnamed_goal_blocks migration: legacy blocked row has no named blocker or block time; reactivated without inventing either",
          ) as { id: number; new_value_json: string } | undefined;
        const restoreLegacy =
          goal.status === "active" &&
          (goal.block_history?.length ?? 0) === 0 &&
          mistakenRelease !== undefined &&
          JSON.parse(mistakenRelease.new_value_json).record_version === goal.record_version;
        const history =
          (goal.status === "blocked" && (goal.block_history?.length ?? 0) === 0) || restoreLegacy
            ? [legacyUnknownGoalBlock()]
            : (goal.block_history ?? []);
        db.prepare("UPDATE goals SET block_history_json = ? WHERE id = ?").run(
          serializeJsonValue(history),
          goal.id,
        );
        if (restoreLegacy) {
          db.prepare(
            "UPDATE goals SET status = 'blocked', record_version = record_version + 1 WHERE id = ?",
          ).run(goal.id);
          events.record({
            record_type: "goal",
            record_id: goal.id,
            action: "legacy_block_metadata",
            old_value: goal,
            new_value: {
              ...goal,
              status: "blocked",
              block_history: history,
              record_version: (goal.record_version ?? 1) + 1,
            },
            reason: `restore unchanged legacy block incorrectly released by migration identity event ${mistakenRelease.id}`,
            provenance: { kind: "system" },
          });
        }
        // Recover an assignment only from an actual recorded target change.
        // No event means unknown; creation time is not a guessed assignment.
        if (goal.target_at !== null && goal.target_assigned_at == null) {
          const assignment = db
            .prepare(
              `
            SELECT ts FROM identity_events WHERE record_type = 'goal' AND record_id = ?
              AND json_extract(new_value_json, '$.target_at') = ?
              AND json_extract(old_value_json, '$.target_at') IS NOT json_extract(new_value_json, '$.target_at')
            ORDER BY id DESC LIMIT 1
          `,
            )
            .get(goal.id, goal.target_at) as { ts: number } | undefined;
          if (assignment !== undefined)
            db.prepare("UPDATE goals SET target_assigned_at = ? WHERE id = ?").run(
              assignment.ts,
              goal.id,
            );
        }
      }
    },
  },
] as const satisfies readonly Migration[];
