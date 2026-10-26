import { mkdtempSync, rmSync } from "node:fs";
import { join } from "node:path";
import { tmpdir } from "node:os";

import { afterEach, describe, expect, it, vi } from "vitest";

import {
  LANCEDB_OPTIMIZE_CLEANUP_GRACE_MS,
  LANCEDB_INDEX_CACHE_BYTES,
  LANCEDB_METADATA_CACHE_BYTES,
  LanceDbStore,
  LanceDbTable,
  schema,
  utf8Field,
  vectorField,
} from "./index.js";

describe("lancedb storage", () => {
  const tempDirs: string[] = [];

  afterEach(() => {
    vi.restoreAllMocks();
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
      expect(Object.getPrototypeOf(listed[0])).toBe(Object.prototype);
      expect(searched[0]?.vector).toEqual(new Float32Array([1, 0, 0]));
      expect(store.memoryStats()).toMatchObject({
        cache: {
          indexMaxBytes: LANCEDB_INDEX_CACHE_BYTES,
          metadataMaxBytes: LANCEDB_METADATA_CACHE_BYTES,
          sizeBytes: expect.any(Number),
          approxItems: expect.any(Number),
        },
        tables: [
          {
            table: "items",
            openHandles: 1,
            listQueries: 1,
            searchQueries: 1,
            upserts: 2,
            rowsRead: 3,
          },
        ],
      });
      // This observes the actual native session used by the connection, not an
      // unused JS Session object created alongside an unconfigured connection.
      expect(store.memoryStats().cache!.sizeBytes).toBeGreaterThan(0);
      const counters = store.memoryStats().tables;
      expect(await store.tableStats()).toEqual([
        expect.objectContaining({
          table: "items",
          fragments: expect.any(Number),
          version: expect.any(Number),
          versions: expect.any(Number),
        }),
      ]);
      expect(store.memoryStats().tables).toEqual(counters);

      await table.remove("id = 'b'");
      expect((await table.list()).map((row) => row.id)).toEqual(["a"]);

      table.close();
    } finally {
      await store.close();
    }
  });

  it.each(["stats", "version", "listVersions"] as const)(
    "reports a stable diagnostic code and logs the cause when %s fails",
    async (method) => {
      const cause = new Error("native details at /private/lance/table/_versions/123.manifest");
      const logger = vi.spyOn(console, "error").mockImplementation(() => {});
      const native = {
        name: "episodes",
        stats: vi.fn(async () => ({ fragmentStats: { numFragments: 2, numSmallFragments: 2 } })),
        version: vi.fn(async () => 3),
        listVersions: vi.fn(async () => [1, 2, 3]),
      };
      native[method].mockRejectedValueOnce(cause);
      const table = new LanceDbTable(native as never);
      const result = await table.diagnosticStats();
      expect(result).toEqual({
        table: "episodes",
        fragments: null,
        smallFragments: null,
        version: null,
        versions: null,
        durationMs: expect.any(Number),
        errorCode: "LANCEDB_TABLE_STATS_FAILED",
      });
      expect(JSON.stringify(result)).not.toContain(cause.message);
      expect(logger).toHaveBeenCalledOnce();
      const details = logger.mock.calls[0]![1];
      expect(details).toMatchObject({ table: "episodes", code: "LANCEDB_TABLE_STATS_FAILED" });
      // stats() wraps storage errors; version()/listVersions() pass them through.
      expect(method === "stats" ? details.cause.cause : details.cause).toBe(cause);
    },
  );

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
      expect(store.memoryStats().lastOptimization).toMatchObject({
        errors: 0,
        versionsPruned: expect.any(Number),
        completedAt: expect.any(Number),
      });

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

  it("closes all table handles on store shutdown even while JS references survive", async () => {
    const tempDir = mkdtempSync(join(tmpdir(), "borg-close-"));
    tempDirs.push(tempDir);
    const store = new LanceDbStore({
      uri: join(tempDir, "lancedb"),
      indexCacheBytes: 1024 * 1024,
      metadataCacheBytes: 1024 * 1024,
    });
    const options = { name: "items", schema: schema([utf8Field("id")]) };
    const first = await store.openTable(options);
    const second = await store.openTable(options);
    expect(store.memoryStats().tables[0]?.openHandles).toBe(2);
    await store.close();
    expect(first.isOpen()).toBe(false);
    expect(second.isOpen()).toBe(false);
    expect(store.memoryStats()).toMatchObject({ cache: null, tables: [] });
    await store.close();
  });

  it("closes a temporary handle when schema validation fails", async () => {
    const close = vi.fn();
    const connection = {
      tableNames: async () => ["items"],
      openTable: async () => ({
        name: "items",
        schema: async () => schema([vectorField("id", 3)]),
        close,
      }),
      close: vi.fn(),
    };
    const store = new LanceDbStore({ uri: "unused", connection: connection as never });
    await expect(
      store.openTable({ name: "items", schema: schema([utf8Field("id")]) }),
    ).rejects.toMatchObject({ code: "LANCEDB_SCHEMA_MISMATCH" });
    expect(close).toHaveBeenCalledOnce();
    expect(store.memoryStats().tables).toEqual([]);
    await store.close();
  });

  it("rejects invalid native cache capacities before opening a connection", () => {
    for (const indexCacheBytes of [-1, 0.5, Number.NaN, Number.MAX_SAFE_INTEGER + 1]) {
      expect(() => new LanceDbStore({ uri: "unused", indexCacheBytes })).toThrow("cache sizes");
    }
  });
});
