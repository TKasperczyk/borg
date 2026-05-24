import { describe, expect, it } from "vitest";

import { isPlainRecord } from "./guards.js";

describe("isPlainRecord", () => {
  it("accepts non-null objects and rejects arrays and primitives", () => {
    expect(isPlainRecord({ value: 1 })).toBe(true);
    expect(isPlainRecord(Object.create(null))).toBe(true);
    expect(isPlainRecord([])).toBe(false);
    expect(isPlainRecord(null)).toBe(false);
    expect(isPlainRecord("value")).toBe(false);
  });
});
