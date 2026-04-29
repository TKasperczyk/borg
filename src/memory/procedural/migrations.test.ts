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

  it("backfills parseable legacy procedural context keys to v2 with collision suffixes", () => {
    const tempDir = mkdtempSync(join(tmpdir(), "borg-"));
    const dbPath = join(tempDir, "procedural.db");
    tempDirs.push(tempDir);

    const legacyDb = openDatabase(dbPath, {
      migrations: proceduralMigrations.filter((migration) => migration.id < 182),
    });

    try {
      legacyDb
        .prepare(
          `
            INSERT INTO skill_context_stats (
              skill_id, context_key, alpha, beta, attempts, successes, failures,
              last_used, last_successful, updated_at
            ) VALUES (?, ?, 2, 1, 1, 1, 0, 100, 100, 100)
          `,
        )
        .run("skill_a", "code_debugging:typescript:self");
      legacyDb
        .prepare(
          `
            INSERT INTO skill_context_stats (
              skill_id, context_key, alpha, beta, attempts, successes, failures,
              last_used, last_successful, updated_at
            ) VALUES (?, ?, 1, 2, 1, 0, 1, 200, NULL, 200)
          `,
        )
        .run("skill_a", "not:parseable:legacy:key");
    } finally {
      legacyDb.close();
    }

    const upgradedDb = openDatabase(dbPath, {
      migrations: proceduralMigrations,
    });

    try {
      const rows = upgradedDb
        .prepare(
          "SELECT context_key, procedural_context_json FROM skill_context_stats ORDER BY context_key ASC",
        )
        .all() as Array<{ context_key: string; procedural_context_json: string | null }>;

      expect(rows.map((row) => row.context_key)).toEqual([
        "not:parseable:legacy:key",
        expect.stringMatching(/^v2:/),
      ]);
      expect(
        JSON.parse(rows.find((row) => row.context_key.startsWith("v2:"))!.procedural_context_json!),
      ).toEqual({
        problem_kind: "code_debugging",
        domain_tags: ["typescript"],
        audience_scope: "self",
      });
    } finally {
      upgradedDb.close();
    }
  });

  it("suffixes real v2 collisions after canonicalization and preserves both rows", () => {
    const tempDir = mkdtempSync(join(tmpdir(), "borg-"));
    const dbPath = join(tempDir, "procedural.db");
    tempDirs.push(tempDir);

    const legacyDb = openDatabase(dbPath, {
      migrations: proceduralMigrations.filter((migration) => migration.id < 182),
    });

    try {
      const insert = legacyDb.prepare(
        `
          INSERT INTO skill_context_stats (
            skill_id, context_key, alpha, beta, attempts, successes, failures,
            last_used, last_successful, updated_at
          ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        `,
      );

      insert.run("skill_a", "code_debugging:typescript:self", 2, 1, 1, 1, 0, 100, 100, 100);
      insert.run("skill_a", "code_debugging:TypeScript:self", 1, 2, 1, 0, 1, 200, null, 200);
    } finally {
      legacyDb.close();
    }

    const upgradedDb = openDatabase(dbPath, {
      migrations: proceduralMigrations,
    });

    try {
      const baseKey = deriveProceduralContextKey({
        problem_kind: "code_debugging",
        domain_tags: ["typescript"],
        audience_scope: "self",
      });
      const rows = upgradedDb
        .prepare("SELECT context_key FROM skill_context_stats ORDER BY context_key ASC")
        .all() as Array<{ context_key: string }>;

      expect(rows.map((row) => row.context_key)).toEqual([baseKey, `${baseKey}:legacy-1`]);
    } finally {
      upgradedDb.close();
    }
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
