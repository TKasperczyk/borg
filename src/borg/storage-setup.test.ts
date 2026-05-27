import { mkdtempSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";

import { afterEach, describe, expect, it } from "vitest";

import { openDatabase, type Migration, type SqliteDatabase } from "../storage/sqlite/index.js";
import { createMigrations } from "./storage-setup.js";

describe("borg storage setup migrations", () => {
  const tempDirs: string[] = [];

  afterEach(() => {
    while (tempDirs.length > 0) {
      rmSync(tempDirs.pop() as string, { recursive: true, force: true });
    }
  });

  function tableExists(db: SqliteDatabase, name: string): boolean {
    return (
      db.prepare("SELECT name FROM sqlite_master WHERE type = 'table' AND name = ?").get(name) !==
      undefined
    );
  }

  it("runs the Sprint C table drop on databases that applied the old operator-advice migration", () => {
    const tempDir = mkdtempSync(join(tmpdir(), "borg-"));
    tempDirs.push(tempDir);
    const dbPath = join(tempDir, "borg.db");
    const migrations = createMigrations();
    const dropMigration = migrations.find(
      (migration) => migration.name === "drop_operator_advice_table",
    );

    if (dropMigration === undefined) {
      throw new Error("drop_operator_advice_table migration missing");
    }
    expect(dropMigration.id % 1_000_000).toBe(2);

    const legacyOperatorAdviceMigration = {
      id: dropMigration.id - 1,
      name: "operator_advice_initial_schema",
      up: `
        CREATE TABLE operator_advice (
          id TEXT PRIMARY KEY,
          session_id TEXT NULL,
          audience_entity_id TEXT NULL,
          text TEXT NOT NULL,
          created_at INTEGER NOT NULL,
          expires_at INTEGER NULL,
          consumed_at INTEGER NULL,
          consumed_by_turn_id TEXT NULL,
          canceled_at INTEGER NULL
        );
      `,
    } satisfies Migration;

    const legacyDb = openDatabase(dbPath, {
      migrations: [legacyOperatorAdviceMigration],
    });
    try {
      expect(tableExists(legacyDb, "operator_advice")).toBe(true);
    } finally {
      legacyDb.close();
    }

    const migratedDb = openDatabase(dbPath, {
      migrations,
    });
    try {
      expect(tableExists(migratedDb, "operator_advice")).toBe(false);
      expect(
        migratedDb.listAppliedMigrations().some((migration) => migration.id === dropMigration.id),
      ).toBe(true);
    } finally {
      migratedDb.close();
    }
  });
});
