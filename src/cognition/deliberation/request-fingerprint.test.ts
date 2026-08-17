import { describe, expect, it } from "vitest";

import { fingerprintCanonicalValue } from "./request-fingerprint.js";

describe("fingerprintCanonicalValue", () => {
  it("normalizes object key order recursively", () => {
    expect(fingerprintCanonicalValue({ outer: { second: 2, first: 1 }, tail: true })).toEqual(
      fingerprintCanonicalValue({ tail: true, outer: { first: 1, second: 2 } }),
    );
  });

  it("preserves array order", () => {
    expect(fingerprintCanonicalValue(["first", "second"]).canonicalSha256).not.toBe(
      fingerprintCanonicalValue(["second", "first"]).canonicalSha256,
    );
  });

  it("fingerprints astral text by its exact JSON representation", () => {
    expect(fingerprintCanonicalValue({ text: "Sol 🌌" })).toEqual(
      fingerprintCanonicalValue({ text: "Sol 🌌" }),
    );
    expect(fingerprintCanonicalValue({ text: "Sol 🌌" }).canonicalSha256).not.toBe(
      fingerprintCanonicalValue({ text: "Sol 🌍" }).canonicalSha256,
    );
  });

  it("keeps distinct finite numeric values distinct", () => {
    expect(fingerprintCanonicalValue(1).canonicalSha256).not.toBe(
      fingerprintCanonicalValue(1.25).canonicalSha256,
    );
    expect(fingerprintCanonicalValue(-12.5e6)).toEqual(fingerprintCanonicalValue(-12_500_000));
  });

  it("pins JSON-compatible undefined normalization", () => {
    expect(fingerprintCanonicalValue({ kept: true, dropped: undefined })).toEqual(
      fingerprintCanonicalValue({ kept: true }),
    );
    expect(fingerprintCanonicalValue([undefined])).toEqual(fingerprintCanonicalValue([null]));
    expect(fingerprintCanonicalValue(undefined).canonicalSha256).not.toBe(
      fingerprintCanonicalValue(null).canonicalSha256,
    );
  });
});
