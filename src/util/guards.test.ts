import { describe, expect, it } from "vitest";

import { isNodeError, isPlainRecord } from "./guards.js";

describe("isPlainRecord", () => {
  it("accepts non-null objects and rejects arrays and primitives", () => {
    expect(isPlainRecord({ value: 1 })).toBe(true);
    expect(isPlainRecord(Object.create(null))).toBe(true);
    expect(isPlainRecord([])).toBe(false);
    expect(isPlainRecord(null)).toBe(false);
    expect(isPlainRecord("value")).toBe(false);
  });
});

describe("isNodeError", () => {
  it("accepts Error instances with string code fields", () => {
    const error = new Error("missing") as NodeJS.ErrnoException;
    error.code = "ENOENT";

    expect(isNodeError(error)).toBe(true);
  });

  it("rejects non-errors and errors without string codes", () => {
    expect(isNodeError(new Error("plain"))).toBe(false);
    expect(isNodeError({ code: "ENOENT" })).toBe(false);
  });
});
