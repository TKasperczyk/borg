import { describe, expect, it } from "vitest";

import {
  dedupePreservingOrder,
  mapWithConcurrency,
  sortStrings,
  uniqueStrings,
} from "./collections.js";

describe("dedupePreservingOrder", () => {
  it("keeps the first occurrence of each string-like value", () => {
    expect(dedupePreservingOrder(["b", "a", "b", "c", "a"] as const)).toEqual(["b", "a", "c"]);
  });
});

describe("uniqueStrings", () => {
  it("dedupes strings without trimming or normalizing", () => {
    expect(uniqueStrings([" a", "a", " a", "A"])).toEqual([" a", "a", "A"]);
  });
});

describe("sortStrings", () => {
  it("returns a localeCompare-sorted copy", () => {
    const input = ["b", "a", "c"] as const;

    expect(sortStrings(input)).toEqual(["a", "b", "c"]);
    expect(input).toEqual(["b", "a", "c"]);
  });
});

describe("mapWithConcurrency", () => {
  it("preserves input order while bounding active mappers", async () => {
    let active = 0;
    let maxActive = 0;

    const result = await mapWithConcurrency([1, 2, 3, 4, 5], 2, async (value) => {
      active += 1;
      maxActive = Math.max(maxActive, active);
      await Promise.resolve();
      active -= 1;
      return value * 2;
    });

    expect(result).toEqual([2, 4, 6, 8, 10]);
    expect(maxActive).toBe(2);
  });
});
