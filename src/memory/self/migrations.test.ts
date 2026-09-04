import { describe, expect, it } from "vitest";

import { openDatabase } from "../../storage/sqlite/index.js";
import { selfMigrations } from "./migrations.js";

describe("self migrations", () => {
  it("applies the complete migration set to a fresh database", () => {
    const db = openDatabase(":memory:", { migrations: selfMigrations });

    try {
      const migrationNames = selfMigrations.map((migration) => migration.name);
      expect(new Set(migrationNames).size).toBe(migrationNames.length);
      expect(migrationNames.slice(-2)).toEqual([
        "open_question_rumination_run_stamps",
        "goal_counterparty_entity_id",
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
