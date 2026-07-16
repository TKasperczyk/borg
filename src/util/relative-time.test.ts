import { describe, expect, it } from "vitest";

import { formatRelativeDuration } from "./relative-time.js";

describe("formatRelativeDuration", () => {
  it("formats second, minute, hour, and day buckets without an age suffix", () => {
    expect(formatRelativeDuration(41_000)).toBe("~41s");
    expect(formatRelativeDuration(5 * 60_000)).toBe("5m");
    expect(formatRelativeDuration(2 * 60 * 60_000)).toBe("2h");
    expect(formatRelativeDuration(24 * 60 * 60_000)).toBe("1d");
    expect(formatRelativeDuration(3 * 24 * 60 * 60_000)).toBe("3d");
  });
});
