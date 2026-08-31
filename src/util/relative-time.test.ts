import { describe, expect, it } from "vitest";

import { formatRelativeAge, formatRelativeDuration, formatRelativeUntil } from "./relative-time.js";

describe("formatRelativeDuration", () => {
  it("formats second, minute, hour, and day buckets without an age suffix", () => {
    expect(formatRelativeDuration(41_000)).toBe("~41s");
    expect(formatRelativeDuration(5 * 60_000)).toBe("5m");
    expect(formatRelativeDuration(2 * 60 * 60_000)).toBe("2h");
    expect(formatRelativeDuration(24 * 60 * 60_000)).toBe("1d");
    expect(formatRelativeDuration(3 * 24 * 60 * 60_000)).toBe("3d");
  });
});

describe("formatRelativeUntil", () => {
  it("formats future timestamps without changing formatRelativeAge semantics", () => {
    expect(formatRelativeUntil(1_000_000 + 41_000, 1_000_000)).toBe("in ~41s");
    expect(formatRelativeUntil(1_000_000 + 5 * 60_000, 1_000_000)).toBe("in 5m");
    expect(formatRelativeAge(1_000_000 + 5 * 60_000, 1_000_000)).toBe("~0s ago");
  });
});
