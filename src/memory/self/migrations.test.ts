import { mkdtempSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { identityMigrations, IdentityEventRepository } from "../identity/index.js";
import { GoalsRepository } from "./goals-repository.js";
import { goalSchedulingTimes, currentGoalBlock } from "./goal-blocks.js";
import { createGoalId } from "../../util/ids.js";
import { ManualClock } from "../../util/clock.js";
import { composeMigrations } from "../../storage/sqlite/index.js";
import { describe, expect, it } from "vitest";

import { openDatabase } from "../../storage/sqlite/index.js";
import { selfMigrations } from "./migrations.js";

describe("self migrations", () => {
  it("applies the complete migration set to a fresh database", () => {
    const db = openDatabase(":memory:", { migrations: selfMigrations });

    try {
      const migrationNames = selfMigrations.map((migration) => migration.name);
      expect(new Set(migrationNames).size).toBe(migrationNames.length);
      expect(migrationNames.slice(-3)).toEqual([
        "goal_counterparty_entity_id",
        "goal_named_block_history",
        "goal_deadline_assignment_basis",
      ]);
      expect(db.listAppliedMigrations().map((migration) => migration.name)).toEqual(migrationNames);
      expect(
        db
          .prepare("SELECT name FROM sqlite_master WHERE type = 'table' AND name = ?")
          .get("open_question_rumination_stamps"),
      ).toEqual({ name: "open_question_rumination_stamps" });
      expect(
        (db.pragma("table_info(goals)") as Array<{ name: string }>).map((column) => column.name),
      ).toContain("counterparty_entity_id");
    } finally {
      db.close();
    }
  });
});

it("preserves a legacy unnamed block in an audited migration without inventing a blocker", () => {
  const dataDir = mkdtempSync(join(tmpdir(), "borg-block-migration-"));
  const path = join(dataDir, "legacy.db");
  try {
    const oldDb = openDatabase(path, {
      migrations: composeMigrations(selfMigrations.slice(0, 9), identityMigrations.slice(0, 1)),
    });
    const goal = { id: createGoalId(), description: "資料待ち" };
    oldDb
      .prepare(
        "INSERT INTO goals (id, description, priority, status, created_at, progress_notes, target_at, provenance_kind) VALUES (?, ?, 3, 'blocked', 1000, 'Earlier attempt', 7000, 'manual')",
      )
      .run(goal.id, goal.description);
    oldDb.close();
    const db = openDatabase(path, {
      migrations: composeMigrations(selfMigrations, identityMigrations),
    });
    try {
      const goals = new GoalsRepository({
        db,
        identityEventRepository: new IdentityEventRepository({ db }),
      });
      expect(goals.get(goal.id)).toMatchObject({
        status: "blocked",
        description: goal.description,
        progress_notes: "Earlier attempt",
        block_history: [
          {
            blocker: { kind: "legacy_unknown" },
            attempt_status: "not_recorded",
            blocked_at: null,
            disclosure_label: { disclosure_class: "unknown" },
          },
        ],
      });
      expect(new IdentityEventRepository({ db }).list({ recordId: goal.id })[0]).toMatchObject({
        action: "legacy_block_metadata",
        old_value: { status: "blocked" },
        new_value: { status: "blocked" },
        reason: expect.stringContaining("preserve blocked status"),
      });
      goals.reconcileBlocks();
      expect(goals.get(goal.id)?.status).toBe("blocked");
      expect(
        goals.listActiveFollowupDueCandidatesReadOnly({
          lookaheadMs: 10_000,
          staleMs: 1,
          limit: 5,
        }),
      ).toEqual([]);
      expect(() =>
        goals.block(
          goal.id,
          {
            blocker: { kind: "legacy_unknown" },
            attempt_status: "not_recorded",
            reason: "x",
          } as never,
          { kind: "manual" },
        ),
      ).toThrow();
      const released = goals.unblock(goal.id, "Reprise explicite", { kind: "manual" });
      expect(released.status).toBe("active");
      expect(released.block_history?.[0]?.blocked_at).toBeNull();
    } finally {
      db.close();
    }
  } finally {
    rmSync(dataDir, { recursive: true, force: true });
  }
});

it("recovers recorded deadline assignments and corrects only untouched mistaken legacy releases", () => {
  const dataDir = mkdtempSync(join(tmpdir(), "borg-block-upgrade-"));
  const path = join(dataDir, "upgrade.db");
  const clock = new ManualClock(1_000);
  try {
    const db = openDatabase(path, {
      migrations: composeMigrations(selfMigrations, identityMigrations.slice(0, 2)),
    });
    const events = new IdentityEventRepository({ db, clock });
    const goals = new GoalsRepository({ db, clock, identityEventRepository: events });
    const untouched = goals.add({
      description: "legacy",
      priority: 1,
      provenance: { kind: "manual" },
    });
    events.record({
      record_type: "goal",
      record_id: untouched.id,
      action: "unblock",
      old_value: { ...untouched, status: "blocked" },
      new_value: untouched,
      provenance: { kind: "system" },
      reason:
        "repair_unnamed_goal_blocks migration: legacy blocked row has no named blocker or block time; reactivated without inventing either",
    });
    const edited = goals.add({
      description: "later edited",
      priority: 1,
      provenance: { kind: "manual" },
    });
    events.record({
      record_type: "goal",
      record_id: edited.id,
      action: "unblock",
      old_value: { ...edited, status: "blocked" },
      new_value: edited,
      provenance: { kind: "system" },
      reason:
        "repair_unnamed_goal_blocks migration: legacy blocked row has no named blocker or block time; reactivated without inventing either",
    });
    goals.update(edited.id, { progress_notes: "Intentional later progress" }, { kind: "manual" });
    const assigned = goals.add({
      description: "deadline",
      priority: 1,
      targetAt: 10_000,
      provenance: { kind: "manual" },
    });
    clock.advance(4_000);
    goals.update(assigned.id, { target_at: 7_000 }, { kind: "manual" });
    db.prepare("UPDATE goals SET target_assigned_at = NULL WHERE id = ?").run(assigned.id);
    db.close();
    const upgraded = openDatabase(path, {
      migrations: composeMigrations(selfMigrations, identityMigrations),
    });
    try {
      const rows = new GoalsRepository({ db: upgraded, clock });
      expect(currentGoalBlock(rows.get(untouched.id)!)?.blocker.kind).toBe("legacy_unknown");
      expect(rows.get(untouched.id)?.status).toBe("blocked");
      expect(rows.get(edited.id)).toMatchObject({
        status: "active",
        progress_notes: "Intentional later progress",
        block_history: [],
      });
      expect(rows.get(assigned.id)?.target_assigned_at).toBe(5_000);
      expect(goalSchedulingTimes(rows.get(assigned.id)!).targetAt).toBe(7_000);
    } finally {
      upgraded.close();
    }
  } finally {
    rmSync(dataDir, { recursive: true, force: true });
  }
});
