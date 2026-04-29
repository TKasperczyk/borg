import { describe, expect, it } from "vitest";

import {
  bestVectorSimilarity,
  cosineSimilarity,
  cosineSimilarity01,
} from "./embedding-similarity.js";

describe("embedding similarity helpers", () => {
  it("computes cosine similarity for aligned and opposite vectors", () => {
    expect(cosineSimilarity(new Float32Array([1, 0]), new Float32Array([1, 0]))).toBeCloseTo(1);
    expect(cosineSimilarity(new Float32Array([1, 0]), new Float32Array([-1, 0]))).toBeCloseTo(-1);
  });

  it("maps cosine similarity into a zero-to-one score", () => {
    expect(cosineSimilarity01(new Float32Array([1, 0]), new Float32Array([1, 0]))).toBeCloseTo(1);
    expect(cosineSimilarity01(new Float32Array([1, 0]), new Float32Array([-1, 0]))).toBeCloseTo(0);
  });

  it("returns the best non-negative cosine score among candidate vectors", () => {
    const target = new Float32Array([1, 0]);

    expect(
      bestVectorSimilarity(target, [
        new Float32Array([-1, 0]),
        new Float32Array([0.8, 0.2]),
        new Float32Array([0, 1]),
      ]),
    ).toBeGreaterThan(0.95);
  });

  it("returns zero for empty, orthogonal, opposite, or zero-vector candidates", () => {
    expect(bestVectorSimilarity(new Float32Array([1, 0]), [])).toBe(0);
    expect(bestVectorSimilarity(new Float32Array([1, 0]), [new Float32Array([0, 1])])).toBe(0);
    expect(bestVectorSimilarity(new Float32Array([1, 0]), [new Float32Array([-1, 0])])).toBe(0);
    expect(cosineSimilarity(new Float32Array([1, 0]), new Float32Array([0, 0]))).toBe(0);
  });

  it("treats NaN and Infinity components as zero", () => {
    expect(
      cosineSimilarity(new Float32Array([1, Number.NaN]), new Float32Array([1, 0])),
    ).toBeCloseTo(1);
    expect(
      cosineSimilarity(new Float32Array([1, Number.POSITIVE_INFINITY]), new Float32Array([1, 0])),
    ).toBeCloseTo(1);
    expect(
      bestVectorSimilarity(new Float32Array([1, Number.NaN]), [new Float32Array([1, 0])]),
    ).toBeCloseTo(1);
  });

  it("keeps finite correlated vectors sane", () => {
    expect(cosineSimilarity(new Float32Array([2, 2]), new Float32Array([1, 1]))).toBeCloseTo(1);
    expect(cosineSimilarity01(new Float32Array([2, 2]), new Float32Array([1, 1]))).toBeCloseTo(1);
  });

  it("rejects dimension mismatches", () => {
    expect(() => cosineSimilarity(new Float32Array([1, 0]), new Float32Array([1]))).toThrow(
      RangeError,
    );
  });
});
