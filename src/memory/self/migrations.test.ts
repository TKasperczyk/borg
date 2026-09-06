import { mkdtempSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { identityMigrations, IdentityEventRepository } from "../identity/index.js";
import { GoalsRepository } from "./goals-repository.js";
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
        "open_question_rumination_run_stamps",
        "goal_counterparty_entity_id",
        "goal_named_block_history",
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
      migrations: composeMigrations(selfMigrations.slice(0, -1), identityMigrations.slice(0, 1)),
    });
    const goal = new GoalsRepository({ db: oldDb }).add({
      description: "資料待ち",
      priority: 3,
      progressNotes: "Earlier attempt",
      provenance: { kind: "manual" },
    });
    oldDb.prepare("UPDATE goals SET status = 'blocked' WHERE id = ?").run(goal.id);
    oldDb.close();
    const db = openDatabase(path, {
      migrations: composeMigrations(selfMigrations, identityMigrations),
    });
    try {
      expect(new GoalsRepository({ db }).get(goal.id)).toMatchObject({
        status: "active",
        description: goal.description,
        progress_notes: "Earlier attempt",
        block_history: [],
      });
      expect(new IdentityEventRepository({ db }).list({ recordId: goal.id })[0]).toMatchObject({
        action: "unblock",
        old_value: { status: "blocked" },
        new_value: { status: "active" },
        reason: expect.stringContaining("legacy blocked row has no named blocker"),
      });
    } finally {
      db.close();
    }
  } finally {
    rmSync(dataDir, { recursive: true, force: true });
  }
});
