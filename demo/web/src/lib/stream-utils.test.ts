import { afterEach, describe, expect, it, vi } from "vitest";

import { formatTimestamp, formatTimestampForKey, formatTimestampRange } from "./stream-utils";

describe("timestamp formatting", () => {
  afterEach(() => {
    vi.useRealTimers();
  });

  it("uses time for today, compact date for this year, and full date for older years", () => {
    vi.useFakeTimers();
    vi.setSystemTime(new Date(2026, 5, 10, 12, 0, 0));

    expect(formatTimestamp(new Date(2026, 5, 10, 8, 9, 5).getTime())).toBe("08:09:05");
    expect(formatTimestamp(new Date(2026, 5, 9, 8, 9, 5).getTime())).toBe("Jun 9 08:09");
    expect(formatTimestamp(new Date(2025, 11, 31, 23, 59, 5).getTime())).toBe("Dec 31 2025 23:59");
  });

  it("normalizes ranges so later timestamps never render first", () => {
    vi.useFakeTimers();
    vi.setSystemTime(new Date(2026, 5, 10, 12, 0, 0));

    const later = new Date(2026, 5, 10, 13, 16, 54).getTime();
    const earlier = new Date(2026, 5, 10, 8, 54, 43).getTime();

    expect(formatTimestampRange(later, earlier)).toBe("08:54:43 - 13:16:54");
  });

  it("formats numeric epoch values only when the key is structurally timestamp-like", () => {
    vi.useFakeTimers();
    vi.setSystemTime(new Date(2026, 5, 10, 12, 0, 0));

    const ts = new Date(2026, 5, 10, 8, 9, 5).getTime();
    expect(formatTimestampForKey("created_at", ts)).toBe("08:09:05");
    expect(formatTimestampForKey("created", ts)).toBe("08:09:05");
    expect(formatTimestampForKey("score", ts)).toBeNull();
    expect(formatTimestampForKey("created_at", 42)).toBeNull();
  });
});
