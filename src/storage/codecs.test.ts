import { describe, expect, it } from "vitest";

import { toFloat32Array, type Float32ArrayCodecOptions } from "./codecs.js";

const VECTOR_CODEC = {
  arrayLikeErrorMessage: "Vector must be array-like",
  nonFiniteErrorMessage: "Vector contains a non-finite value",
  errorCode: "VECTOR_INVALID",
} satisfies Float32ArrayCodecOptions;

describe("toFloat32Array", () => {
  it("coerces arrays, typed arrays, and array-like objects", () => {
    expect(Array.from(toFloat32Array([1, 2], VECTOR_CODEC))).toEqual([1, 2]);
    expect(Array.from(toFloat32Array(new Uint8Array([3, 4]), VECTOR_CODEC))).toEqual([3, 4]);
    expect(Array.from(toFloat32Array({ 0: 5, 1: 6, length: 2 }, VECTOR_CODEC))).toEqual([5, 6]);
  });

  it("throws for non-array-like and non-finite values", () => {
    expect(() => toFloat32Array("vector", VECTOR_CODEC)).toThrow("Vector must be array-like");
    expect(() => toFloat32Array([Number.NaN], VECTOR_CODEC)).toThrow(
      "Vector contains a non-finite value",
    );
  });
});
