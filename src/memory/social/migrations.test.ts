import { mkdtempSync, rmSync } from "node:fs";
import { join } from "node:path";
import { tmpdir } from "node:os";

import { afterEach, describe, expect, it } from "vitest";

import { openDatabase } from "../../storage/sqlite/index.js";

import { socialMigrations } from "./migrations.js";
import { SocialRepository } from "./repository.js";

describe("social migrations", () => {
  const tempDirs: string[] = [];

  afterEach(() => {
    while (tempDirs.length > 0) {
      rmSync(tempDirs.pop() as string, { recursive: true, force: true });
    }
  });

  it("upgrades social event provenance checks to allow online writes", () => {
    const tempDir = mkdtempSync(join(tmpdir(), "borg-"));
    const dbPath = join(tempDir, "social.db");
    tempDirs.push(tempDir);

    const initialDb = openDatabase(dbPath, {
      migrations: socialMigrations.filter((migration) => migration.id < 241),
    });
    initialDb.close();

    const db = openDatabase(dbPath, {
      migrations: socialMigrations,
    });
    const repository = new SocialRepository({ db });

    try {
      repository.recordInteraction("ent_onlineeeeeeeeeee" as never, {
        provenance: {
          kind: "online",
          process: "reflector",
        },
      });

      expect(repository.listEvents("ent_onlineeeeeeeeeee" as never)).toEqual([
        expect.objectContaining({
          provenance: {
            kind: "online",
            process: "reflector",
          },
        }),
      ]);
    } finally {
      db.close();
    }
  });
});
