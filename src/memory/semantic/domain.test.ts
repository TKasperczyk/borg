import { describe, expect, it } from "vitest";

import { DOMAIN_SYNONYMS, canonicalizeDomain } from "./domain.js";

describe("semantic domain canonicalization", () => {
  it("normalizes case and whitespace", () => {
    expect(canonicalizeDomain("  Tech  ")).toBe("tech");
    expect(canonicalizeDomain("  artisanal-craft  ")).toBe("artisanal-craft");
    expect(canonicalizeDomain("   ")).toBeNull();
    expect(canonicalizeDomain(null)).toBeNull();
  });

  it("maps configured synonyms to canonical buckets", () => {
    expect(DOMAIN_SYNONYMS.technology).toBe("tech");
    expect(DOMAIN_SYNONYMS.persons).toBe("people");
    expect(DOMAIN_SYNONYMS.location).toBe("places");
    expect(DOMAIN_SYNONYMS.cuisine).toBe("food");
    expect(canonicalizeDomain("technology")).toBe("tech");
    expect(canonicalizeDomain("Persons")).toBe("people");
    expect(canonicalizeDomain(" location ")).toBe("places");
    expect(canonicalizeDomain("Cuisine")).toBe("food");
  });
});
