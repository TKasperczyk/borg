import { describe, expect, it } from "vitest";

import { parsePositiveIntegerValue, positiveIntegerValue } from "./parse.js";

describe("parsePositiveIntegerValue", () => {
  it("accepts positive integer numbers and numeric strings", () => {
    expect(parsePositiveIntegerValue(3)).toBe(3);
    expect(parsePositiveIntegerValue("3")).toBe(3);
  });

  it("rejects zero, decimals, and non-numeric strings", () => {
    expect(parsePositiveIntegerValue(0)).toBeNull();
    expect(parsePositiveIntegerValue("1.5")).toBeNull();
    expect(parsePositiveIntegerValue("abc")).toBeNull();
  });
});

describe("positiveIntegerValue", () => {
  it("accepts only positive integer numbers", () => {
    expect(positiveIntegerValue(2)).toBe(2);
    expect(positiveIntegerValue("2")).toBeNull();
  });
});
