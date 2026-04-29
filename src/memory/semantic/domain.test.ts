import { describe, expect, it } from "vitest";

import { canonicalizeDomain } from "./domain.js";

describe("semantic domain canonicalization", () => {
  it("normalizes case and whitespace", () => {
    expect(canonicalizeDomain("  Tech  ")).toBe("tech");
    expect(canonicalizeDomain("  artisanal-craft  ")).toBe("artisanal-craft");
    expect(canonicalizeDomain("   ")).toBeNull();
    expect(canonicalizeDomain(null)).toBeNull();
  });

  it("does not infer English synonym buckets", () => {
    expect(canonicalizeDomain("technology")).toBe("technology");
    expect(canonicalizeDomain("Persons")).toBe("persons");
    expect(canonicalizeDomain(" location ")).toBe("location");
    expect(canonicalizeDomain("Cuisine")).toBe("cuisine");
  });
});
