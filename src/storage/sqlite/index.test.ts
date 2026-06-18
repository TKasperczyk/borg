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
            database.prepare("INSERT INTO items (name) VALUES (?)").run(["bravo"]);
          },
        },
      ],
    });

    try {
      const statement = db.prepare("SELECT name FROM items WHERE id = ?");
      expect(db.prepare("SELECT name FROM items WHERE id = ?")).toBe(statement);
      const row = statement.get([1]) as {
        name: string;
      };

      expect(Object.getPrototypeOf(row)).toBe(Object.prototype);
      expect(row).toEqual({ name: "alpha" });
      expect(
        db
          .prepare("SELECT name FROM items WHERE id IN (?, ?) ORDER BY id ASC")
          .all([1, 2])
          .map((item) => item.name),
      ).toEqual(["alpha", "bravo"]);
      expect(
        Array.from(
          db.prepare("SELECT name FROM items WHERE id IN (?, ?) ORDER BY id ASC").iterate([1, 2]),
          (item) => {
            expect(Object.getPrototypeOf(item)).toBe(Object.prototype);
            return item.name;
          },
        ),
      ).toEqual(["alpha", "bravo"]);
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

  it("preserves pragma helper behavior", () => {
    const tempDir = mkdtempSync(join(tmpdir(), "borg-"));
    tempDirs.push(tempDir);

    const db = openDatabase(join(tempDir, "borg.db"));

    try {
      expect(db.pragma("journal_mode", { simple: true })).toBe("wal");
      expect(db.pragma("foreign_keys", { simple: true })).toBe(1);
      expect(db.pragma("wal_checkpoint(TRUNCATE)")).toEqual([
        expect.objectContaining({
          busy: 0,
        }),
      ]);
    } finally {
      db.close();
    }
  });

  it("preserves transaction variants, rollback, and nested savepoint behavior", () => {
    const db = openDatabase(":memory:");

    try {
      db.exec("CREATE TABLE tx_items (name TEXT NOT NULL)");

      const observesTransaction = db.transaction(() => db.raw.inTransaction);
      expect(observesTransaction()).toBe(true);
      expect(observesTransaction.default()).toBe(true);
      expect(observesTransaction.deferred()).toBe(true);
      expect(observesTransaction.immediate()).toBe(true);
      expect(observesTransaction.exclusive()).toBe(true);
      expect(db.raw.inTransaction).toBe(false);

      const insert = db.prepare("INSERT INTO tx_items (name) VALUES (?)");
      const outer = db.transaction(() => {
        insert.run("outer");

        try {
          db.transaction(() => {
            insert.run("inner");
            throw new Error("inner rollback");
          })();
        } catch (error) {
          expect((error as Error).message).toBe("inner rollback");
        }

        insert.run("after-inner");
        return db.raw.inTransaction;
      });

      expect(outer.immediate()).toBe(true);
      expect(db.raw.inTransaction).toBe(false);
      expect(
        db
          .prepare("SELECT name FROM tx_items ORDER BY rowid ASC")
          .all()
          .map((row) => row.name),
      ).toEqual(["outer", "after-inner"]);

      const fails = db.transaction(() => {
        insert.run("top-level");
        throw new Error("top rollback");
      });

      expect(() => fails()).toThrow("top rollback");
      expect(
        db
          .prepare("SELECT name FROM tx_items ORDER BY rowid ASC")
          .all()
          .map((row) => row.name),
      ).toEqual(["outer", "after-inner"]);
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
        composeMigrations(
          [{ id: 1, name: "alpha", up: "" }],
          [{ id: -1, name: "negative", up: "" }],
        ),
      /Migration source id -1 must be an integer in \[1, 999999\]: negative:-1/,
    );
  });

  it("rejects composed source ids outside a band range", () => {
    expectComposeError(
      () =>
        composeMigrations(
          [{ id: 1, name: "alpha", up: "" }],
          [{ id: 1_000_000, name: "too-large", up: "" }],
        ),
      /Migration source id 1000000 must be an integer in \[1, 999999\]: too-large:1000000/,
    );

    expectComposeError(
      () =>
        composeMigrations(
          [{ id: 1, name: "alpha", up: "" }],
          [{ id: 1_000_001, name: "much-too-large", up: "" }],
        ),
      /Migration source id 1000001 must be an integer in \[1, 999999\]: much-too-large:1000001/,
    );
  });
});
