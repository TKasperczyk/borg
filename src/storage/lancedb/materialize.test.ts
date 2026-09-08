import { makeArrowTable } from "@lancedb/lancedb";
import { Field, Float64, List, Struct, Table, Utf8, vectorFromArray } from "apache-arrow";
import { describe, expect, it } from "vitest";

import { materializeRows } from "./materialize.js";
import { schema, utf8Field, vectorField } from "./index.js";

describe("Arrow result ownership", () => {
  it("detaches rows and vector cells from their parent batches", async () => {
    const arrow = makeArrowTable(
      [
        { id: "a", embedding: [1, 2, 3] },
        { id: "b", embedding: [4, 5, 6] },
      ],
      { schema: schema([utf8Field("id"), vectorField("embedding", 3)]) },
    );
    let finished = false;
    async function* batches() {
      try {
        yield* arrow.batches;
      } finally {
        finished = true;
      }
    }
    const rows = await materializeRows(batches());
    expect(finished).toBe(true);
    expect(rows).toEqual([
      { id: "a", embedding: [1, 2, 3] },
      { id: "b", embedding: [4, 5, 6] },
    ]);
    expect(Object.getPrototypeOf(rows[0])).toBe(Object.prototype);
    expect(Object.getOwnPropertySymbols(rows[0]!)).toEqual([]);
    expect(Array.isArray(rows[0]!.embedding)).toBe(true);
    // Mutating the source Arrow buffers must not affect returned data.
    arrow.getChild("embedding")!.get(0).set(0, 99);
    expect(rows[0]!.embedding).toEqual([1, 2, 3]);
  });

  it("copies nested struct/list cells and consumes every batch", async () => {
    const arrow = new Table({
      id: vectorFromArray(["a"], new Utf8()),
      meta: vectorFromArray(
        [{ labels: ["one", "two"], count: 2 }],
        new Struct([
          new Field("labels", new List(new Field("item", new Utf8(), false)), false),
          new Field("count", new Float64(), false),
        ]),
      ),
    });
    async function* batches() {
      yield* arrow.batches;
      yield* arrow.batches;
    }
    const rows = await materializeRows(batches());
    expect(rows).toEqual([
      { id: "a", meta: { labels: ["one", "two"], count: 2 } },
      { id: "a", meta: { labels: ["one", "two"], count: 2 } },
    ]);
    expect(Object.getPrototypeOf(rows[0]!.meta)).toBe(Object.prototype);
    expect(rows[0]!.meta).not.toBe(rows[1]!.meta);
  });

  it("surfaces iterator errors while allowing iterator cleanup", async () => {
    let finished = false;
    async function* batches() {
      try {
        yield* makeArrowTable([{ id: "a" }]).batches;
        throw new Error("scan failed");
      } finally {
        finished = true;
      }
    }
    await expect(materializeRows(batches())).rejects.toThrow("scan failed");
    expect(finished).toBe(true);
  });
});
