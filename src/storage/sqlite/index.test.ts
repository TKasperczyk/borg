import { mkdtempSync, rmSync } from "node:fs";
import { join } from "node:path";
import { tmpdir } from "node:os";

import { afterEach, describe, expect, it } from "vitest";

import { openDatabase } from "./index.js";

describe("sqlite storage", () => {
  const tempDirs: string[] = [];

  afterEach(() => {
    while (tempDirs.length > 0) {
      rmSync(tempDirs.pop() as string, { recursive: true, force: true });
    }
  });

  it("runs migrations and caches prepared statements", () => {
    const tempDir = mkdtempSync(join(tmpdir(), "borg-"));
    tempDirs.push(tempDir);

    const db = openDatabase(join(tempDir, "borg.db"), {
      migrations: [
        {
          id: 1,
          name: "create-items",
          up: "CREATE TABLE items (id INTEGER PRIMARY KEY, name TEXT NOT NULL)",
        },
        {
          id: 2,
          name: "seed-items",
          up: (database) => {
            database.prepare("INSERT INTO items (name) VALUES (?)").run("alpha");
          },
        },
      ],
    });

    try {
      const statement = db.prepare("SELECT name FROM items WHERE id = ?");
      expect(db.prepare("SELECT name FROM items WHERE id = ?")).toBe(statement);
      expect(
        statement.get(1) as {
          name: string;
        },
      ).toEqual({ name: "alpha" });
      expect(db.listAppliedMigrations().map((migration) => migration.id)).toEqual([1, 2]);
    } finally {
      db.close();
    }
  });
});
