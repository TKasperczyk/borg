import { mkdtempSync, rmSync } from "node:fs";
import { join } from "node:path";
import { tmpdir } from "node:os";

import { afterEach, describe, expect, it } from "vitest";

import { openDatabase } from "../../storage/sqlite/index.js";
import { FixedClock } from "../../util/clock.js";
import { StorageError } from "../../util/errors.js";

import { identityMigrations } from "./migrations.js";
import { IdentityEventRepository } from "./repository.js";

describe("identity migrations", () => {
  const tempDirs: string[] = [];

  afterEach(() => {
    while (tempDirs.length > 0) {
      rmSync(tempDirs.pop() as string, { recursive: true, force: true });
    }
  });

  it("upgrades an empty identity event table to allow online writes", () => {
    const tempDir = mkdtempSync(join(tmpdir(), "borg-"));
    const dbPath = join(tempDir, "identity.db");
    tempDirs.push(tempDir);

    const initialDb = openDatabase(dbPath, {
      migrations: identityMigrations.filter((migration) => migration.id < 251),
    });
    initialDb.close();

    const db = openDatabase(dbPath, {
      migrations: identityMigrations,
    });
    const repository = new IdentityEventRepository({
      db,
      clock: new FixedClock(2_000),
    });

    try {
      repository.record({
        record_type: "goal",
        record_id: "goal_aaaaaaaaaaaaaaaa",
        action: "update",
        provenance: {
          kind: "online",
          process: "reflector",
        },
      });

      expect(repository.list({ recordType: "goal", recordId: "goal_aaaaaaaaaaaaaaaa" })).toEqual(
        expect.arrayContaining([
          expect.objectContaining({
            provenance: {
              kind: "online",
              process: "reflector",
            },
          }),
        ]),
      );
    } finally {
      db.close();
    }
  });

  it("fails migration 251 loudly when identity events already exist", () => {
    const tempDir = mkdtempSync(join(tmpdir(), "borg-"));
    const dbPath = join(tempDir, "identity.db");
    tempDirs.push(tempDir);

    const db = openDatabase(dbPath, {
      migrations: identityMigrations.filter((migration) => migration.id < 251),
    });

    try {
      new IdentityEventRepository({
        db,
        clock: new FixedClock(1_000),
      }).record({
        record_type: "goal",
        record_id: "goal_aaaaaaaaaaaaaaaa",
        action: "create",
        provenance: {
          kind: "manual",
        },
      });

      const migration = identityMigrations.find((item) => item.id === 251);
      if (migration === undefined || typeof migration.up !== "function") {
        throw new Error("Migration 251 fixture is not callable");
      }

      let thrown: unknown;
      try {
        migration.up(db);
      } catch (error) {
        thrown = error;
      }

      expect(thrown).toBeInstanceOf(StorageError);
      expect((thrown as StorageError).code).toBe(
        "IDENTITY_EVENTS_MIGRATION_REQUIRES_FRESH_DATABASE",
      );
      expect((thrown as StorageError).message).toContain("identity_events table has 1 rows");
    } finally {
      db.close();
    }
  });
});
