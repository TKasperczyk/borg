import { mkdtempSync, rmSync } from "node:fs";
import { join } from "node:path";
import { tmpdir } from "node:os";

import { afterEach, describe, expect, it } from "vitest";

import { LanceDbStore, schema, utf8Field, vectorField } from "./index.js";

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
});
