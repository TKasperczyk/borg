import { describe, expect, it } from "vitest";

import { serializeJsonValue } from "../util/json-value.js";

import { offlineProcessError } from "./process-errors.js";

describe("offlineProcessError", () => {
  it("omits the code key when the thrown value has no .code property", () => {
    const err = offlineProcessError("semantic-extractor", new TypeError("boom"));

    expect("code" in err).toBe(false);
    expect(() => serializeJsonValue(err)).not.toThrow();
  });

  it("omits the code key when includeErrorCode is false and no explicit code is passed", () => {
    const wrapped = new Error("inner");
    (wrapped as Error & { code?: string }).code = "SOMETHING";
    const err = offlineProcessError("semantic-extractor", wrapped, {
      includeErrorCode: false,
    });

    expect("code" in err).toBe(false);
    expect(() => serializeJsonValue(err)).not.toThrow();
  });

  it("includes the code key when the thrown value carries one", () => {
    const wrapped = new Error("inner");
    (wrapped as Error & { code?: string }).code = "STORAGE_FAILED";
    const err = offlineProcessError("semantic-extractor", wrapped);

    expect(err.code).toBe("STORAGE_FAILED");
    expect(() => serializeJsonValue(err)).not.toThrow();
  });

  it("prefers an explicit code option over the thrown value's code", () => {
    const wrapped = new Error("inner");
    (wrapped as Error & { code?: string }).code = "FROM_ERROR";
    const err = offlineProcessError("semantic-extractor", wrapped, {
      code: "FROM_OPTIONS",
    });

    expect(err.code).toBe("FROM_OPTIONS");
  });

  it("omits target_type and target_id when not provided", () => {
    const err = offlineProcessError("semantic-extractor", new Error("x"));

    expect("target_type" in err).toBe(false);
    expect("target_id" in err).toBe(false);
  });

  it("survives serialization when the underlying error has no .code property", () => {
    const errs = [
      offlineProcessError("semantic-extractor", new TypeError("a")),
      offlineProcessError("semantic-extractor", "string thrown"),
      offlineProcessError("semantic-extractor", { weird: "object" }),
    ];

    for (const err of errs) {
      expect(() => serializeJsonValue({ content: { errors: [err] } })).not.toThrow();
    }
  });
});
