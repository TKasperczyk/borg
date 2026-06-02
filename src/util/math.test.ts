import { describe, expect, it } from "vitest";

import { clamp, halfLifeDecay } from "./math.js";

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
