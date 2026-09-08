import { makeArrowTable } from "@lancedb/lancedb";
import {
  Field,
  FixedSizeList,
  Float32,
  Float64,
  List,
  Struct,
  Table,
  Utf8,
  Vector,
  makeData,
  vectorFromArray,
} from "apache-arrow";
import { describe, expect, it } from "vitest";

import { toFloat32Array } from "../codecs.js";
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
      { id: "a", embedding: new Float32Array([1, 2, 3]) },
      { id: "b", embedding: new Float32Array([4, 5, 6]) },
    ]);
    expect(Object.getPrototypeOf(rows[0])).toBe(Object.prototype);
    expect(Object.getOwnPropertySymbols(rows[0]!)).toEqual([]);
    const embedding = rows[0]!.embedding as Float32Array;
    expect(embedding).toBeInstanceOf(Float32Array);
    const source = arrow.getChild("embedding")!.get(0).toArray() as Float32Array;
    expect(embedding.buffer).not.toBe(source.buffer);
    // Mutating the source Arrow buffers must not affect returned data.
    arrow.getChild("embedding")!.get(0).set(0, 99);
    expect(embedding).toEqual(new Float32Array([1, 2, 3]));
    embedding[1] = 88;
    expect(source[1]).toBe(2);
  });

  it("materializes a 3,000-row, 4096-dimension scan into independently owned hydration buffers", async () => {
    const rowCount = 3000;
    const dimensions = 4096;
    const source = new Float32Array(rowCount * dimensions);
    for (let row = 0; row < rowCount; row++) {
      source[row * dimensions] = row + 0.25;
      source[(row + 1) * dimensions - 1] = -row - 0.5;
    }
    // Construct the representative Arrow scan without boxed JS fixture vectors.
    const floats = new Float32();
    const arrow = new Table({
      embedding: new Vector([
        makeData({
          type: new FixedSizeList(dimensions, new Field("item", floats, false)),
          length: rowCount,
          child: makeData({ type: floats, data: source, length: source.length }),
        }),
      ]),
    });
    async function* batches() {
      yield* arrow.slice(0, 1500).batches;
      // Exercise nonzero offsets as well as consuming multiple batches.
      yield* arrow.slice(1500).batches;
    }
    const rows = await materializeRows(batches());
    expect(rows).toHaveLength(rowCount);
    const buffers = new Set<ArrayBufferLike>();
    let payloadBytes = 0;
    for (const [index, row] of rows.entries()) {
      const vector = row.embedding as Float32Array;
      expect(vector).toBeInstanceOf(Float32Array);
      expect(vector.length).toBe(dimensions);
      expect(vector.byteOffset).toBe(0);
      expect(vector.buffer.byteLength).toBe(dimensions * Float32Array.BYTES_PER_ELEMENT);
      expect(vector.buffer).not.toBe(source.buffer);
      expect(vector[0]).toBe(index + 0.25);
      expect(vector[dimensions - 1]).toBe(-index - 0.5);
      expect(
        toFloat32Array(vector, {
          arrayLikeErrorMessage: "invalid vector",
          nonFiniteErrorMessage: "non-finite vector",
          errorCode: "TEST_VECTOR_INVALID",
        }),
      ).toBe(vector);
      buffers.add(vector.buffer);
      payloadBytes += vector.byteLength;
    }
    expect(buffers.size).toBe(rowCount);
    expect(payloadBytes).toBe(49_152_000);
    source.fill(99);
    for (const [index, row] of rows.entries()) {
      const vector = row.embedding as Float32Array;
      expect(vector[0]).toBe(index + 0.25);
      expect(vector[dimensions - 1]).toBe(-index - 0.5);
    }
  });

  it("copies dense float64 lists directly into detached float32 hydration vectors", async () => {
    const arrow = new Table({
      embedding: vectorFromArray(
        [
          [1 / 3, 2, 3],
          [4, 5, 6],
        ],
        new FixedSizeList(3, new Field("item", new Float64(), false)),
      ),
    });
    async function* batches() {
      yield* arrow.batches;
    }
    const rows = await materializeRows(batches());
    const source = arrow.getChild("embedding")!.get(0)!.toArray();
    const embedding = rows[0]!.embedding as Float32Array;
    expect(embedding).toEqual(new Float32Array([1 / 3, 2, 3]));
    expect(embedding.buffer).not.toBe(source.buffer);
    source[0] = 99;
    expect(embedding[0]).toBe(Math.fround(1 / 3));
    embedding[1] = 88;
    expect(source[1]).toBe(2);
  });

  it("preserves nulls and nested lists through recursive conversion", async () => {
    const arrow = new Table({
      nullable: vectorFromArray(
        [[1, null, 3], null, [5, 6, 7]],
        new FixedSizeList(3, new Field("item", new Float32(), true)),
      ),
      nested: vectorFromArray(
        [[[1, 2], [3]], [[4]], null],
        new List(new Field("item", new List(new Field("item", new Float64(), false)), false)),
      ),
    });
    async function* batches() {
      yield* arrow.batches;
    }
    const rows = await materializeRows(batches());
    expect(rows).toEqual([
      { nullable: [1, null, 3], nested: [[1, 2], [3]] },
      { nullable: null, nested: [[4]] },
      { nullable: new Float32Array([5, 6, 7]), nested: null },
    ]);
    expect(Array.isArray(rows[0]!.nullable)).toBe(true);
    // A nullable schema alone must not defeat the dense-vector fast path.
    expect(rows[2]!.nullable).toBeInstanceOf(Float32Array);
    arrow.getChild("nested")!.get(0)!.get(0)!.set(0, 99);
    expect(rows[0]!.nested).toEqual([[1, 2], [3]]);
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
