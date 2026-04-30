import { mkdtempSync, rmSync } from "node:fs";
import { join } from "node:path";
import { tmpdir } from "node:os";

import { afterEach, describe, expect, it } from "vitest";

import { openDatabase } from "../../storage/sqlite/index.js";

import { deriveProceduralContextKey } from "./context.js";
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

    const initialDb = openDatabase(dbPath, {
      migrations: proceduralMigrations.filter((migration) => migration.id < 176),
    });
    initialDb.close();

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
        "procedural_context_json",
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

  it("does not duplicate or corrupt v2 context stats when migrations rerun", () => {
    const tempDir = mkdtempSync(join(tmpdir(), "borg-"));
    const dbPath = join(tempDir, "procedural.db");
    tempDirs.push(tempDir);

    const db = openDatabase(dbPath, {
      migrations: proceduralMigrations,
    });
    const contextKey = deriveProceduralContextKey({
      problem_kind: "planning",
      domain_tags: ["roadmap"],
      audience_scope: "self",
    });

    try {
      db.prepare(
        `
          INSERT INTO skill_context_stats (
            skill_id, context_key, procedural_context_json, alpha, beta, attempts, successes,
            failures, last_used, last_successful, updated_at
          ) VALUES (?, ?, ?, 2, 1, 1, 1, 0, 100, 100, 100)
        `,
      ).run(
        "skill_a",
        contextKey,
        JSON.stringify({
          problem_kind: "planning",
          domain_tags: ["roadmap"],
          audience_scope: "self",
        }),
      );
    } finally {
      db.close();
    }

    const reopenedDb = openDatabase(dbPath, {
      migrations: proceduralMigrations,
    });

    try {
      const rows = reopenedDb
        .prepare(
          "SELECT context_key, procedural_context_json FROM skill_context_stats ORDER BY context_key ASC",
        )
        .all() as Array<{ context_key: string; procedural_context_json: string }>;

      expect(rows).toHaveLength(1);
      expect(rows[0]).toEqual({
        context_key: contextKey,
        procedural_context_json: JSON.stringify({
          problem_kind: "planning",
          domain_tags: ["roadmap"],
          audience_scope: "self",
        }),
      });
    } finally {
      reopenedDb.close();
    }
  });
});
