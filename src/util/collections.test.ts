import { describe, expect, it } from "vitest";

import { dedupePreservingOrder, sortStrings, uniqueStrings } from "./collections.js";

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
