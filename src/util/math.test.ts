import { describe, expect, it } from "vitest";

import {
  clamp,
  clampPositiveIntegerOrFallback,
  coercePositiveIntegerOrFallback,
  coerceUnitIntervalOrFallback,
  optionalPositiveIntegerOption,
  positiveIntegerOptionOrFallback,
  positiveIntegerRecordParamOrFallback,
  requirePositiveInteger,
  requireUnitInterval,
  unitIntervalOptionOrFallback,
  halfLifeDecay,
} from "./math.js";
import { positiveIntegerValue } from "./parse.js";

describe("clamp", () => {
  it("keeps values inside the inclusive bounds", () => {
    expect(clamp(5, 0, 10)).toBe(5);
    expect(clamp(-1, 0, 10)).toBe(0);
    expect(clamp(11, 0, 10)).toBe(10);
  });
});

describe("halfLifeDecay", () => {
  it("returns the standard exponential decay factor", () => {
    const halfLife = 12;

    expect(halfLifeDecay(halfLife, halfLife)).toBe(0.5);
    expect(halfLifeDecay(0, halfLife)).toBe(1);
  });
});

describe("positive integer helpers", () => {
  it("coerces positive finite numbers or falls back", () => {
    expect(coercePositiveIntegerOrFallback(3.9, 10)).toBe(3);
    expect(coercePositiveIntegerOrFallback(0, 10)).toBe(10);
    expect(coercePositiveIntegerOrFallback(Number.NaN, 10)).toBe(10);
    expect(coercePositiveIntegerOrFallback(undefined, 10)).toBe(10);
  });

  it("clamps floored finite numbers to at least one or falls back", () => {
    expect(clampPositiveIntegerOrFallback(3.9, 10)).toBe(3);
    expect(clampPositiveIntegerOrFallback(0, 10)).toBe(1);
    expect(clampPositiveIntegerOrFallback(Number.POSITIVE_INFINITY, 10)).toBe(10);
    expect(clampPositiveIntegerOrFallback(undefined, 10)).toBe(10);
  });

  it("requires positive integer options and params with explicit fallback semantics", () => {
    expect(requirePositiveInteger(3, "count")).toBe(3);
    expect(() => requirePositiveInteger(1.5, "count")).toThrow("count must be a positive integer");
    expect(positiveIntegerOptionOrFallback(undefined, 5, "count")).toBe(5);
    expect(optionalPositiveIntegerOption(undefined, "count")).toBeUndefined();
    expect(positiveIntegerRecordParamOrFallback({ count: 4 }, "count", 5)).toBe(4);
    expect(positiveIntegerRecordParamOrFallback({ count: 4 }, "count", 5)).toBe(
      positiveIntegerValue(4),
    );
    expect(positiveIntegerRecordParamOrFallback({ count: "4" }, "count", 5)).toBe(5);
  });
});

describe("unit interval helpers", () => {
  it("coerces or requires unit intervals according to the helper contract", () => {
    expect(coerceUnitIntervalOrFallback(1.5, 0.4)).toBe(1);
    expect(coerceUnitIntervalOrFallback(-0.5, 0.4)).toBe(0);
    expect(coerceUnitIntervalOrFallback(Number.NaN, 0.4)).toBe(0.4);
    expect(requireUnitInterval(0.5, "threshold")).toBe(0.5);
    expect(() => requireUnitInterval(1.5, "threshold")).toThrow(
      "threshold must be between 0 and 1",
    );
    expect(unitIntervalOptionOrFallback(undefined, 0.4, "threshold")).toBe(0.4);
  });
});
