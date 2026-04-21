import { describe, expect, it } from "vitest";

import { FixedClock, ManualClock, SystemClock } from "./clock.js";

describe("clock", () => {
  it("provides fixed and manual clocks for tests", () => {
    const fixedClock = new FixedClock(123);
    const manualClock = new ManualClock(10);

    expect(fixedClock.now()).toBe(123);
    expect(manualClock.now()).toBe(10);
    expect(manualClock.advance(5)).toBe(15);
    manualClock.set(99);
    expect(manualClock.now()).toBe(99);
  });

  it("system clock returns a unix timestamp", () => {
    expect(SystemClock.prototype.now.call(new SystemClock())).toBeGreaterThan(0);
  });
});
