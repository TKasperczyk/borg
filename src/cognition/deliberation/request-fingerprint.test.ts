import { describe, expect, it } from "vitest";

import { fingerprintCanonicalValue, fingerprintSystemSurface } from "./request-fingerprint.js";
import { fingerprintPlannerSurface } from "./planner-context-capture.js";
import type { LLMSystemBlock } from "../../llm/index.js";

describe("fingerprintCanonicalValue", () => {
  it("counts cache markers independently of blocks and fingerprints the uncached tail", () => {
    const system: LLMSystemBlock[] = ["global", "audience", "standing", "overlay", "fast"].map(
      (text, index) => ({
        type: "text",
        text,
        ...(index === 4
          ? {}
          : { cache_control: { type: "ephemeral", ttl: index < 2 ? "1h" : "5m" } }),
      }),
    );
    const fingerprint = fingerprintSystemSurface(system);
    expect(fingerprint).toMatchObject({ systemBlockCount: 5, cacheBreakpointCount: 4 });
    expect(fingerprintPlannerSurface({ system })).toEqual(fingerprint);
    const changed = [...system.slice(0, 4), { type: "text" as const, text: "next turn" }];
    expect(fingerprintSystemSurface(changed).systemSha256).not.toBe(fingerprint.systemSha256);
    expect(fingerprintSystemSurface(changed).transportSha256).not.toBe(fingerprint.transportSha256);
  });
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
