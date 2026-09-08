import { mkdtempSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";

import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import { createMigrations } from "../../borg/storage-setup.js";
import { StorageError } from "../../util/errors.js";
import productionMigrations from "../fixtures/prod-migrations.json";
import { composeMigrations, openDatabase, type Migration, type SqliteDatabase } from "./index.js";

describe("migration name reconciliation", () => {
  let tempDir: string;
  let db: SqliteDatabase | undefined;

  function reopenDatabase(migrations: readonly Migration[] = []): SqliteDatabase {
    db?.close();
    db = undefined;
    db = openDatabase(join(tempDir, "borg.db"), { migrations });
    return db;
  }

  beforeEach(() => {
    tempDir = mkdtempSync(join(tmpdir(), "borg-migrations-"));
  });

  afterEach(() => {
    db?.close();
    db = undefined;
    rmSync(tempDir, { recursive: true, force: true });
  });

  it("reconciles the production fixture without rerunning applied migrations", () => {
    const migrations = createMigrations();
    const migrationsByName = new Map(migrations.map((migration) => [migration.name, migration]));
    // Replay the real definitions using the ids and names recorded before the merge.
    const legacyMigrations = productionMigrations.map(({ id, name }) => {
      const migration = migrationsByName.get(name);
      if (migration === undefined) {
        throw new Error(`Missing migration definition for ${name}`);
      }
      return { ...migration, id };
    });
    const legacyDb = reopenDatabase(legacyMigrations);
    const stamp = legacyDb.prepare("UPDATE _migrations SET applied_at = ? WHERE id = ?");
    for (const [index, migration] of productionMigrations.entries()) {
      stamp.run(1_700_000_000_000 + index, migration.id);
    }
    const before = legacyDb.listAppliedMigrations();
    expect(before).toHaveLength(65);
    expect(before.map(({ id, name }) => ({ id, name }))).toEqual(productionMigrations);
    const responseLane = legacyDb
      .prepare("SELECT * FROM sqlite_master WHERE name = 'idx_stream_entry_response_lane'")
      .get();
    expect(responseLane).toBeDefined();
    expect(
      legacyDb
        .prepare(
          "SELECT name FROM sqlite_master WHERE type = 'table' AND name IN (?, ?) ORDER BY name",
        )
        .all("agent_deliveries", "agent_delivery_ack_receipts"),
    ).toEqual([{ name: "agent_deliveries" }, { name: "agent_delivery_ack_receipts" }]);

    // The response-lane CREATE INDEX is not idempotent, so a rerun would fail here.
    const upgradedDb = reopenDatabase(migrations);
    const after = upgradedDb.listAppliedMigrations();
    const afterByName = new Map(after.map((migration) => [migration.name, migration]));
    for (const [name, id] of [
      ["agent_deliveries", 28_000_001],
      ["agent_delivery_ack_receipts", 28_000_002],
      ["stream_entry_response_lane", 16_000_006],
    ] as const) {
      expect(afterByName.get(name)).toEqual({
        ...before.find((migration) => migration.name === name),
        id,
      });
    }
    for (const migration of before) {
      expect(afterByName.get(migration.name)?.applied_at).toBe(migration.applied_at);
    }
    expect(after.map(({ id, name }) => ({ id, name }))).toEqual(
      migrations.map(({ id, name }) => ({ id, name })).sort((left, right) => left.id - right.id),
    );
    for (const [name, id] of [
      ["operator_attention_index", 27_000_001],
      ["stream_entry_blocker_and_answered_edge_lookups", 16_000_004],
      ["stream_entry_observed_answered_edges", 16_000_005],
    ] as const) {
      expect(afterByName.get(name)).toMatchObject({ id, name });
    }
    const schema = upgradedDb.prepare("SELECT * FROM sqlite_master ORDER BY type, name").all();
    expect(schema).toEqual(
      expect.arrayContaining([
        responseLane,
        expect.objectContaining({ type: "table", name: "operator_attention_records" }),
        expect.objectContaining({ type: "index", name: "idx_operator_attention_latest" }),
        expect.objectContaining({ type: "index", name: "idx_stream_entry_blocker_inbound" }),
        expect.objectContaining({
          type: "index",
          name: "idx_stream_entry_latest_answered_edge",
          sql: expect.stringContaining("'agent_observed'"),
        }),
      ]),
    );

    const reopenedDb = reopenDatabase(migrations);
    expect(reopenedDb.listAppliedMigrations()).toEqual(after);
    expect(reopenedDb.prepare("SELECT * FROM sqlite_master ORDER BY type, name").all()).toEqual(
      schema,
    );
    expect(reopenedDb.prepare("SELECT total_changes() AS count").get()).toEqual({ count: 0 });
  });

  it("swaps applied migration ids without rerunning them", () => {
    const seededDb = reopenDatabase();
    seededDb.exec(`
      INSERT INTO _migrations (id, name, applied_at) VALUES
        (1, 'alpha', 101), (2, 'beta', 202);
    `);
    const up = vi.fn();
    const upgradedDb = reopenDatabase([
      { id: 1, name: "beta", up },
      { id: 2, name: "alpha", up },
    ]);

    expect(upgradedDb.listAppliedMigrations()).toEqual([
      { id: 1, name: "beta", applied_at: 202 },
      { id: 2, name: "alpha", applied_at: 101 },
    ]);
    expect(up).not.toHaveBeenCalled();
  });

  it("reconciles chained id moves and applies migrations at vacated ids", () => {
    const seededDb = reopenDatabase();
    seededDb.exec(`
      INSERT INTO _migrations (id, name, applied_at) VALUES
        (1, 'alpha', 101), (2, 'beta', 202), (3, 'gamma', 303);
    `);
    const up = vi.fn();
    const upgradedDb = reopenDatabase([
      { id: 1, name: "new-migration", up: "CREATE TABLE new_items (id INTEGER PRIMARY KEY)" },
      { id: 2, name: "alpha", up },
      { id: 3, name: "beta", up },
      { id: 4, name: "gamma", up },
    ]);

    expect(upgradedDb.listAppliedMigrations()).toEqual([
      { id: 1, name: "new-migration", applied_at: expect.any(Number) },
      { id: 2, name: "alpha", applied_at: 101 },
      { id: 3, name: "beta", applied_at: 202 },
      { id: 4, name: "gamma", applied_at: 303 },
    ]);
    expect(upgradedDb.prepare("SELECT * FROM new_items").all()).toEqual([]);
    expect(up).not.toHaveBeenCalled();
  });

  it("preserves unknown migrations and avoids occupied temporary ids", () => {
    const seededDb = reopenDatabase();
    seededDb.exec(`
      INSERT INTO _migrations (id, name, applied_at) VALUES
        (-1, 'unknown-negative', 100), (-3, 'another-unknown-negative', 300),
        (1, 'alpha', 101), (2, 'beta', 202), (99, 'unknown-positive', 999);
    `);
    const up = vi.fn();
    const upgradedDb = reopenDatabase([
      { id: 2, name: "alpha", up },
      { id: 3, name: "beta", up },
    ]);

    expect(upgradedDb.listAppliedMigrations()).toEqual([
      { id: -3, name: "another-unknown-negative", applied_at: 300 },
      { id: -1, name: "unknown-negative", applied_at: 100 },
      { id: 2, name: "alpha", applied_at: 101 },
      { id: 3, name: "beta", applied_at: 202 },
      { id: 99, name: "unknown-positive", applied_at: 999 },
    ]);
    expect(up).not.toHaveBeenCalled();
  });

  it.each([false, true])(
    "rejects an unknown name at a composed id and rolls back reconciliation (already applied: %s)",
    (alreadyApplied) => {
      const seededDb = reopenDatabase();
      seededDb.exec(`
        INSERT INTO _migrations (id, name, applied_at) VALUES
          (1, 'alpha', 101), (3, 'unknown-migration', 303);
      `);
      if (alreadyApplied) {
        seededDb.exec("INSERT INTO _migrations (id, name, applied_at) VALUES (4, 'beta', 404)");
      }
      const before = seededDb.listAppliedMigrations();
      const up = vi.fn();
      const migrations = [
        { id: 2, name: "alpha", up },
        { id: 3, name: "beta", up },
      ];

      const reopen = () => reopenDatabase(migrations);
      expect(reopen).toThrow(StorageError);
      expect(reopen).toThrow(/Migration id 3.*beta.*unknown-migration/);
      expect(reopenDatabase().listAppliedMigrations()).toEqual(before);
      expect(up).not.toHaveBeenCalled();
    },
  );

  it("rejects duplicate migration names before applying migrations", () => {
    const up = vi.fn();
    const migrations = composeMigrations(
      [{ id: 1, name: "duplicate-name", up }],
      [{ id: 1, name: "duplicate-name", up }],
    );
    const reopen = () => reopenDatabase(migrations);

    expect(reopen).toThrow(StorageError);
    expect(reopen).toThrow("Duplicate migration name duplicate-name");
    expect(up).not.toHaveBeenCalled();
    expect(reopenDatabase().listAppliedMigrations()).toEqual([]);
  });
});
