import { mkdtempSync, rmSync } from "node:fs";
import { join } from "node:path";
import { tmpdir } from "node:os";

import { afterEach, describe, expect, it } from "vitest";

import { openDatabase } from "../../storage/sqlite/index.js";

import { proceduralMigrations } from "./migrations.js";

describe("procedural migrations", () => {
  const tempDirs: string[] = [];

  afterEach(() => {
    while (tempDirs.length > 0) {
      rmSync(tempDirs.pop() as string, { recursive: true, force: true });
    }
  });

  it("adds skill context stats and procedural evidence context columns idempotently", () => {
    const tempDir = mkdtempSync(join(tmpdir(), "borg-"));
    const dbPath = join(tempDir, "procedural.db");
    tempDirs.push(tempDir);

    const legacyDb = openDatabase(dbPath, {
      migrations: proceduralMigrations.filter((migration) => migration.id < 176),
    });
    legacyDb.close();

    const upgradedDb = openDatabase(dbPath, {
      migrations: proceduralMigrations,
    });

    try {
      const evidenceColumns = upgradedDb
        .prepare("PRAGMA table_info(procedural_evidence)")
        .all() as Array<{
        name: string;
      }>;
      const statsColumns = upgradedDb
        .prepare("PRAGMA table_info(skill_context_stats)")
        .all() as Array<{
        name: string;
      }>;
      const skillColumns = upgradedDb.prepare("PRAGMA table_info(skills)").all() as Array<{
        name: string;
      }>;

      expect(evidenceColumns.map((column) => column.name)).toContain("procedural_context");
      expect(skillColumns.map((column) => column.name)).toEqual(
        expect.arrayContaining([
          "status",
          "superseded_by",
          "superseded_at",
          "splitting_at",
          "last_split_attempt_at",
          "split_failure_count",
          "last_split_error",
          "requires_manual_review",
        ]),
      );
      expect(statsColumns.map((column) => column.name)).toEqual([
        "skill_id",
        "context_key",
        "alpha",
        "beta",
        "attempts",
        "successes",
        "failures",
        "last_used",
        "last_successful",
        "updated_at",
      ]);
    } finally {
      upgradedDb.close();
    }

    const reopenedDb = openDatabase(dbPath, {
      migrations: proceduralMigrations,
    });
    reopenedDb.close();
  });
});
