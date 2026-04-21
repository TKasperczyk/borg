import { describe, expect, it } from "vitest";

import { assertJsonValue, serializeJsonValue } from "./json-value.js";

describe("json-value", () => {
  it("accepts nested JSON-compatible values", () => {
    expect(() =>
      assertJsonValue({
        ok: true,
        nested: ["a", 1, null, { fine: false }],
      }),
    ).not.toThrow();

    expect(
      serializeJsonValue({
        ok: true,
        nested: ["a", 1, null],
      }),
    ).toBe('{"ok":true,"nested":["a",1,null]}');
  });

  it("rejects unsupported values", () => {
    expect(() => assertJsonValue({ bad: undefined })).toThrow(/\$\.bad contains undefined/);
    expect(() => assertJsonValue({ bad: 1n })).toThrow(/\$\.bad contains a bigint/);
    expect(() => assertJsonValue({ bad: Symbol("x") })).toThrow(/\$\.bad contains a symbol/);
    expect(() => assertJsonValue({ bad: () => "x" })).toThrow(/\$\.bad contains a function/);
    expect(() => assertJsonValue({ bad: new Date() })).toThrow(
      /\$\.bad contains a non-plain object/,
    );
  });
});
