import { describe, expect, it } from "vitest";

import { isoInstantEpochMsSchema, parseIsoInstant } from "./iso-instant.js";

describe("parseIsoInstant", () => {
  it("parses zoned instants and rejects unzoned ones when an offset is required", () => {
    expect(parseIsoInstant("2026-01-31T09:30:00Z")).toBe(Date.UTC(2026, 0, 31, 9, 30));
    expect(parseIsoInstant("2026-01-31T09:30:00+02:00", { requireOffset: true })).toBe(
      Date.UTC(2026, 0, 31, 7, 30),
    );
    expect(parseIsoInstant("2026-01-31T09:30:00", { requireOffset: true })).toBeUndefined();
    expect(parseIsoInstant("not a date")).toBeUndefined();
  });
});

describe("isoInstantEpochMsSchema", () => {
  const schema = isoInstantEpochMsSchema("When it happened.");

  it("parses an ISO-8601 date-time with a zone into epoch milliseconds", () => {
    expect(schema.parse("1970-01-01T00:00:01.500Z")).toBe(1_500);
    expect(schema.parse(" 2026-09-13T12:00:00+02:00 ")).toBe(Date.UTC(2026, 8, 13, 10));
  });

  it("rejects a bare digit run, an unzoned time, a calendar date and prose", () => {
    for (const value of [
      "1788936711815",
      "2026-09-13T12:00:00",
      "2026-09-13",
      "next Tuesday",
      "",
    ]) {
      const result = schema.safeParse(value);
      expect(result.success, value).toBe(false);
    }
    const digits = schema.safeParse("1788936711815");
    expect(digits.success).toBe(false);
    if (!digits.success) {
      expect(digits.error.issues[0]?.message).toContain("ISO-8601");
    }
  });

  it("is a string with the description on the wire and a number after parsing", () => {
    expect(schema.description).toBe("When it happened.");
    const wire = schema.safeParse("2026-09-13T12:00:00Z");
    expect(wire.success && typeof wire.data).toBe("number");
  });
});
