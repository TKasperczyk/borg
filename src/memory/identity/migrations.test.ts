import { mkdtempSync, rmSync } from "node:fs";
import { join } from "node:path";
import { tmpdir } from "node:os";

import { afterEach, describe, expect, it } from "vitest";

import { openDatabase } from "../../storage/sqlite/index.js";
import { FixedClock } from "../../util/clock.js";

import { identityMigrations } from "./migrations.js";
import { IdentityEventRepository } from "./repository.js";

describe("identity migrations", () => {
  const tempDirs: string[] = [];

  afterEach(() => {
    while (tempDirs.length > 0) {
      rmSync(tempDirs.pop() as string, { recursive: true, force: true });
    }
  });

  it("upgrades identity event provenance checks to allow online writes", () => {
    const tempDir = mkdtempSync(join(tmpdir(), "borg-"));
    const dbPath = join(tempDir, "identity.db");
    tempDirs.push(tempDir);

    const legacyDb = openDatabase(dbPath, {
      migrations: identityMigrations.filter((migration) => migration.id < 251),
    });

    try {
      new IdentityEventRepository({
        db: legacyDb,
        clock: new FixedClock(1_000),
      }).record({
        record_type: "goal",
        record_id: "goal_aaaaaaaaaaaaaaaa",
        action: "create",
        provenance: {
          kind: "manual",
        },
      });
    } finally {
      legacyDb.close();
    }

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
});
