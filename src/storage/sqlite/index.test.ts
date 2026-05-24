import { mkdtempSync, rmSync } from "node:fs";
import { join } from "node:path";
import { tmpdir } from "node:os";

import { afterEach, describe, expect, it } from "vitest";

import { StorageError } from "../../util/errors.js";
import { composeMigrations, openDatabase } from "./index.js";

describe("sqlite storage", () => {
  const tempDirs: string[] = [];

  function expectComposeError(fn: () => void, message: RegExp): void {
    expect(fn).toThrow(StorageError);
    expect(fn).toThrow(message);
  }

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

  it("composes multi-band migrations without disturbing within-band runtime order", () => {
    const tempDir = mkdtempSync(join(tmpdir(), "borg-"));
    tempDirs.push(tempDir);
    const applied: string[] = [];

    const db = openDatabase(join(tempDir, "borg.db"), {
      migrations: composeMigrations(
        [
          { id: 1, name: "alpha-1", up: () => applied.push("alpha-1") },
          { id: 2, name: "alpha-2", up: () => applied.push("alpha-2") },
          { id: 3, name: "alpha-3", up: () => applied.push("alpha-3") },
        ],
        [
          { id: 1, name: "beta-1", up: () => applied.push("beta-1") },
          { id: 2, name: "beta-2", up: () => applied.push("beta-2") },
        ],
      ),
    });

    try {
      expect(applied).toEqual(["alpha-1", "alpha-2", "alpha-3", "beta-1", "beta-2"]);
      expect(db.listAppliedMigrations().map((migration) => migration.id)).toEqual([
        1, 2, 3, 1_000_001, 1_000_002,
      ]);
    } finally {
      db.close();
    }
  });

  it("keeps identical source ids unique across composed bands", () => {
    const migrations = composeMigrations(
      [{ id: 1, name: "alpha", up: "" }],
      [{ id: 1, name: "beta", up: "" }],
    );

    expect(migrations.map((migration) => migration.id)).toEqual([1, 1_000_001]);
    expect(new Set(migrations.map((migration) => migration.id)).size).toBe(2);
  });

  it("rejects non-positive composed source ids in later bands", () => {
    expectComposeError(
      () =>
        composeMigrations([{ id: 1, name: "alpha", up: "" }], [{ id: 0, name: "zero", up: "" }]),
      /Migration source id 0 must be an integer in \[1, 999999\]: zero:0/,
    );

    expectComposeError(
      () =>
        composeMigrations([{ id: 1, name: "alpha", up: "" }], [
          { id: -1, name: "negative", up: "" },
        ]),
      /Migration source id -1 must be an integer in \[1, 999999\]: negative:-1/,
    );
  });

  it("rejects composed source ids outside a band range", () => {
    expectComposeError(
      () =>
        composeMigrations([{ id: 1, name: "alpha", up: "" }], [
          { id: 1_000_000, name: "too-large", up: "" },
        ]),
      /Migration source id 1000000 must be an integer in \[1, 999999\]: too-large:1000000/,
    );

    expectComposeError(
      () =>
        composeMigrations([{ id: 1, name: "alpha", up: "" }], [
          { id: 1_000_001, name: "much-too-large", up: "" },
        ]),
      /Migration source id 1000001 must be an integer in \[1, 999999\]: much-too-large:1000001/,
    );
  });
});
