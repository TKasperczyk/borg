import { describe, expect, it } from "vitest";

import { DisjointSet } from "./disjoint-set.js";

describe("DisjointSet", () => {
  it("can preserve left-root union behavior", () => {
    const set = new DisjointSet<number>(() => -1);

    set.union(1, 2);
    set.union(2, 3);

    expect(set.find(1)).toBe(1);
    expect(set.find(2)).toBe(1);
    expect(set.find(3)).toBe(1);
  });

  it("can select lexicographic minimum roots", () => {
    const set = new DisjointSet<string>((leftRoot, rightRoot) => leftRoot.localeCompare(rightRoot));

    set.union("b", "c");
    set.union("a", "c");

    expect(set.find("a")).toBe("a");
    expect(set.find("b")).toBe("a");
    expect(set.find("c")).toBe("a");
  });
});
