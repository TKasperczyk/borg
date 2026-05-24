import { describe, expect, it } from "vitest";

import { getDistance, toSimilarity } from "./vector-results.js";

describe("lancedb vector result helpers", () => {
  it("converts distance values to clamped similarity scores", () => {
    expect(toSimilarity(0)).toBe(1);
    expect(toSimilarity(0.25)).toBe(0.75);
    expect(toSimilarity(1)).toBe(0);
    expect(toSimilarity(4)).toBe(0);
    expect(toSimilarity(-0.5)).toBe(1);
  });

  it("maps undefined distance to zero similarity", () => {
    expect(toSimilarity(undefined)).toBe(0);
  });

  it("preserves existing NaN behavior for similarity conversion", () => {
    expect(toSimilarity(Number.NaN)).toBeNaN();
  });

  it("reads finite numeric LanceDB distance values from rows", () => {
    expect(getDistance({ _distance: 0 })).toBe(0);
    expect(getDistance({ _distance: 0.42 })).toBe(0.42);
    expect(getDistance({ _distance: 999 })).toBe(999);
  });

  it("treats missing, null, and non-finite row distances as absent", () => {
    expect(getDistance({})).toBeUndefined();
    expect(getDistance({ _distance: undefined })).toBeUndefined();
    expect(getDistance({ _distance: null })).toBeUndefined();
    expect(getDistance({ _distance: Number.NaN })).toBeUndefined();
    expect(getDistance({ _distance: Number.POSITIVE_INFINITY })).toBeUndefined();
  });
});
