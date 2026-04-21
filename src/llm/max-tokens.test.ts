import { describe, expect, it } from "vitest";

import { getModelMaxOutputTokens } from "./max-tokens.js";

describe("llm max token ceilings", () => {
  it("returns the Opus ceiling", () => {
    expect(getModelMaxOutputTokens("claude-opus-4-7")).toBe(64_000);
  });

  it("returns the Haiku ceiling", () => {
    expect(getModelMaxOutputTokens("claude-haiku-4-5")).toBe(32_000);
  });

  it("falls back for unknown models", () => {
    expect(getModelMaxOutputTokens("custom-model")).toBe(8_192);
  });
});
