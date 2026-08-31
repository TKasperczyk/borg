import { describe, expect, it } from "vitest";

import { utf16SafePrefixEnd, utf16SafeSuffixStart } from "./utf16-boundary.js";

describe("UTF-16-safe boundaries", () => {
  it("moves a prefix cut before an astral character split", () => {
    const value = "head😀tail";
    const nominalBoundary = value.indexOf("😀") + 1;
    const safeBoundary = utf16SafePrefixEnd(value, nominalBoundary);

    expect(value.slice(0, safeBoundary)).toBe("head");
    expect(value.slice(0, safeBoundary)).not.toMatch(/[\uD800-\uDFFF]$/u);
  });

  it("moves a suffix cut after an astral character split", () => {
    const value = "head😀tail";
    const nominalBoundary = value.indexOf("😀") + 1;
    const safeBoundary = utf16SafeSuffixStart(value, nominalBoundary);

    expect(value.slice(safeBoundary)).toBe("tail");
    expect(value.slice(safeBoundary)).not.toMatch(/^[\uD800-\uDFFF]/u);
  });

  it("leaves code-point boundaries and clamped outer boundaries unchanged", () => {
    const value = "head😀tail";

    expect(utf16SafePrefixEnd(value, value.indexOf("😀"))).toBe(value.indexOf("😀"));
    expect(utf16SafeSuffixStart(value, value.indexOf("tail"))).toBe(value.indexOf("tail"));
    expect(utf16SafePrefixEnd(value, Number.POSITIVE_INFINITY)).toBe(value.length);
    expect(utf16SafeSuffixStart(value, -10)).toBe(0);
  });
});
