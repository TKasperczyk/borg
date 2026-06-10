import { mkdtempSync, rmSync } from "node:fs";
import { join } from "node:path";
import { tmpdir } from "node:os";

import { afterEach, describe, expect, it } from "vitest";

import {
  LANCEDB_OPTIMIZE_CLEANUP_GRACE_MS,
  LanceDbStore,
  schema,
  utf8Field,
  vectorField,
} from "./index.js";

describe("lancedb storage", () => {
  const tempDirs: string[] = [];

  afterEach(() => {
    while (tempDirs.length > 0) {
      rmSync(tempDirs.pop() as string, { recursive: true, force: true });
    }
  });

  it("opens tables, upserts rows, lists rows, searches, and removes rows", async () => {
    const tempDir = mkdtempSync(join(tmpdir(), "borg-"));
    tempDirs.push(tempDir);

    const store = new LanceDbStore({
      uri: join(tempDir, "lancedb"),
    });

    try {
      const table = await store.openTable({
        name: "items",
        schema: schema([utf8Field("id"), utf8Field("label", true), vectorField("vector", 3)]),
      });

      await table.upsert(
        [
          { id: "a", label: "first", vector: [1, 0, 0] },
          { id: "b", label: "second", vector: [0, 1, 0] },
        ],
        { on: "id" },
      );

      await table.upsert([{ id: "a", label: "updated", vector: [1, 0, 0] }], {
        on: "id",
      });

      const listed = await table.list({ limit: 10 });
      const searched = await table.search([1, 0, 0], {
        limit: 1,
        vectorColumn: "vector",
      });

      expect(listed).toHaveLength(2);
      expect(listed.find((row) => row.id === "a")?.label).toBe("updated");
      expect(searched[0]?.id).toBe("a");

      await table.remove("id = 'b'");
      expect((await table.list()).map((row) => row.id)).toEqual(["a"]);

      table.close();
    } finally {
      await store.close();
    }
  });

  it("optimizes all LanceDB tables and preserves rows while reducing fragments", async () => {
    const tempDir = mkdtempSync(join(tmpdir(), "borg-"));
    tempDirs.push(tempDir);

    const store = new LanceDbStore({
      uri: join(tempDir, "lancedb"),
    });

    try {
      const items = await store.openTable({
        name: "items",
        schema: schema([utf8Field("id"), utf8Field("label"), vectorField("vector", 3)]),
      });
      const cachedItems = await store.openTable({
        name: "items",
        schema: schema([utf8Field("id"), utf8Field("label"), vectorField("vector", 3)]),
      });
      const other = await store.openTable({
        name: "other",
        schema: schema([utf8Field("id"), vectorField("vector", 3)]),
      });

      for (let index = 0; index < 30; index += 1) {
        await items.upsert(
          [
            {
              id: `item-${index}`,
              label: `label-${index}`,
              vector: [index, 0, 1],
            },
          ],
          { on: "id" },
        );
      }

      await other.upsert([{ id: "other-1", vector: [0, 1, 0] }], { on: "id" });

      const beforeStats = await items.stats();
      const beforeRows = await items.list();

      expect(beforeStats.fragmentStats.numFragments).toBeGreaterThan(1);
      expect(beforeRows).toHaveLength(30);

      const now = Date.now();
      const result = await store.optimizeStorage({ now });

      const afterStats = await items.stats();
      const afterRows = await items.list();
      const cachedAfterRows = await cachedItems.list();
      const itemResult = result.tables.find((table) => table.table === "items");

      expect(result.cleanupOlderThan).toBe(now - LANCEDB_OPTIMIZE_CLEANUP_GRACE_MS);
      expect(result.durationMs).toEqual(expect.any(Number));
      expect(result.tables.map((table) => table.table).sort()).toEqual(["items", "other"]);
      expect(itemResult).toMatchObject({
        table: "items",
        status: "ok",
        fragmentsRemoved: expect.any(Number),
        fragmentsAdded: expect.any(Number),
        versionsPruned: expect.any(Number),
        bytesRemoved: expect.any(Number),
        durationMs: expect.any(Number),
      });
      expect(afterStats.fragmentStats.numFragments).toBeLessThan(
        beforeStats.fragmentStats.numFragments,
      );
      expect(afterRows).toHaveLength(beforeRows.length);
      expect(afterRows.map((row) => row.id).sort()).toEqual(beforeRows.map((row) => row.id).sort());
      expect(cachedAfterRows.map((row) => row.id).sort()).toEqual(
        beforeRows.map((row) => row.id).sort(),
      );

      if (itemResult?.status !== "ok") {
        throw new Error("Expected items table optimization to succeed");
      }

      expect(itemResult.versionsPruned).toBe(0);

      const pruneResult = await store.optimizeStorage({
        now: now + LANCEDB_OPTIMIZE_CLEANUP_GRACE_MS + 1_000,
      });
      const prunedItemResult = pruneResult.tables.find((table) => table.table === "items");

      if (prunedItemResult?.status !== "ok") {
        throw new Error("Expected items table pruning optimization to succeed");
      }

      expect(prunedItemResult.versionsPruned).toBeGreaterThan(0);

      items.close();
      cachedItems.close();
      other.close();
    } finally {
      await store.close();
    }
  });

  it("continues optimizing remaining tables when one table fails", async () => {
    const optimizeCalls: string[] = [];
    const makeTable = (name: string, shouldFail = false) => ({
      name,
      isOpen: () => true,
      checkoutLatest: async () => {},
      optimize: async () => {
        optimizeCalls.push(name);

        if (shouldFail) {
          throw new Error("table unavailable");
        }

        return {
          compaction: {
            fragmentsRemoved: 2,
            fragmentsAdded: 1,
            filesRemoved: 2,
            filesAdded: 1,
          },
          prune: {
            bytesRemoved: 10,
            oldVersionsRemoved: 3,
          },
        };
      },
      close: () => {},
    });
    const tables = new Map([
      ["ok", makeTable("ok")],
      ["bad", makeTable("bad", true)],
      ["later", makeTable("later")],
    ]);
    const connection = {
      tableNames: async () => ["ok", "bad", "later"],
      openTable: async (name: string) => tables.get(name),
      close: () => {},
    };
    const store = new LanceDbStore({
      uri: "unused",
      connection: connection as never,
    });

    const result = await store.optimizeStorage();

    expect(optimizeCalls).toEqual(["ok", "bad", "later"]);
    expect(result.tables.map((table) => [table.table, table.status])).toEqual([
      ["ok", "ok"],
      ["bad", "error"],
      ["later", "ok"],
    ]);
    expect(result.tables[0]).toMatchObject({
      table: "ok",
      status: "ok",
      fragmentsRemoved: 2,
      fragmentsAdded: 1,
      versionsPruned: 3,
      bytesRemoved: 10,
    });
    expect(result.tables[1]).toMatchObject({
      table: "bad",
      status: "error",
      error: {
        message: "Failed to optimize LanceDB table bad",
        code: "BORG_STORAGE_ERROR",
      },
    });
  });

  it("checks schema compatibility on createEmptyTable handles before returning", async () => {
    let addColumnsCalled = false;
    const fakeTable = {
      name: "items",
      schema: async () => schema([utf8Field("id")]),
      addColumns: async () => {
        addColumnsCalled = true;
      },
      checkoutLatest: async () => {},
      close: () => {},
    };
    const connection = {
      tableNames: async () => [],
      createEmptyTable: async () => fakeTable,
      openTable: async () => fakeTable,
      close: () => {},
    };
    const store = new LanceDbStore({
      uri: "unused",
      connection: connection as never,
    });

    const table = await store.openTable({
      name: "items",
      schema: schema([utf8Field("id"), utf8Field("label", true)]),
    });

    expect(addColumnsCalled).toBe(true);
    table.close();
  });
});
