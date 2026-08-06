import { describe, expect, it } from "vitest";

import { clampMaxOutputTokens, getModelMaxOutputTokens } from "./max-tokens.js";

describe("llm max token ceilings", () => {
  it("returns the Opus ceiling", () => {
    expect(getModelMaxOutputTokens("claude-opus-4-6")).toBe(64_000);
  });

  it("returns the Opus/Sonnet ceiling for newer generations, not the fallback", () => {
    // Regression guard: a version-pinned family match sent claude-opus-5 to the
    // 8_192 fallback, silently cutting the output ceiling by ~87% on a model bump.
    expect(getModelMaxOutputTokens("claude-opus-5")).toBe(64_000);
    expect(getModelMaxOutputTokens("claude-sonnet-5")).toBe(64_000);
    expect(clampMaxOutputTokens("claude-opus-5", 80_000)).toBe(64_000);
  });

  it("returns the Haiku ceiling", () => {
    expect(getModelMaxOutputTokens("claude-haiku-4-5")).toBe(32_000);
    // The exact dated id the recallExpansion/imagePerception slots run on.
    expect(getModelMaxOutputTokens("claude-haiku-4-5-20251001")).toBe(32_000);
  });

  it("is not pinned to a version digit, so a future release keeps its ceiling", () => {
    // NOTE: the ids below are deliberately hypothetical -- at time of writing
    // Opus and Sonnet have shipped a 5, Haiku has not (latest is 4.5). They
    // exist here only to prove the family matcher is version-generic: a future
    // release must not fall through to the 8_192 default the way claude-opus-5
    // did when these patterns were pinned to `-4`.
    expect(getModelMaxOutputTokens("claude-haiku-6")).toBe(32_000);
    expect(getModelMaxOutputTokens("claude-opus-9")).toBe(64_000);
  });

  it("returns the Qwen 3 ceiling for gateway-prefixed and bare model names", () => {
    expect(getModelMaxOutputTokens("generative-apis/qwen3-235b-a22b-instruct-2507")).toBe(16_384);
    expect(getModelMaxOutputTokens("  QWEN3.235B-A22B-INSTRUCT-2507  ")).toBe(16_384);
    expect(getModelMaxOutputTokens("Qwen/Qwen3-235B-A22B-Instruct-2507")).toBe(16_384);
  });

  it("falls back for unknown models", () => {
    expect(getModelMaxOutputTokens("custom-model")).toBe(8_192);
  });

  it("clamps requested output tokens for Qwen, Claude, and unknown models", () => {
    expect(clampMaxOutputTokens("generative-apis/qwen3-235b-a22b-instruct-2507", 20_000)).toBe(
      16_384,
    );
    expect(clampMaxOutputTokens("qwen3-235b-a22b-instruct-2507", 8_192)).toBe(8_192);
    expect(clampMaxOutputTokens("claude-opus-4-6", 80_000)).toBe(64_000);
    expect(clampMaxOutputTokens("custom-model", 20_000)).toBe(8_192);
  });
});
