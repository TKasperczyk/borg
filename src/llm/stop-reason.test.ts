import { describe, expect, it } from "vitest";
import { LLMRefusalError } from "./index.js";
import { LLMError } from "../util/errors.js";
import { classifyLlmStopReason, getLlmErrorStopReason } from "./stop-reason.js";

describe("provider stop reason classification", () => {
  it.each([
    ["end_turn", "completed"],
    ["stop_sequence", "completed"],
    ["tool_use", "tool_use"],
    ["max_tokens", "max_tokens"],
    ["pause_turn", "pause_turn"],
    ["refusal", "refusal"],
    ["model_context_window_exceeded", "context_window_exceeded"],
    ["future_reason", "unknown"],
    [null, "unknown"],
  ])("classifies Anthropic %s without losing its raw value", (raw, kind) => {
    expect(classifyLlmStopReason(raw)).toEqual({ kind, raw });
  });
  it.each([
    ["stop", "completed"],
    ["tool_calls", "tool_use"],
    ["function_call", "tool_use"],
    ["length", "max_tokens"],
    ["refusal", "unknown"],
    ["content_filter", "unknown"],
    ["future_reason", "unknown"],
  ])("keeps provider distinctions for OpenAI-compatible %s", (raw, kind) => {
    expect(classifyLlmStopReason(raw!, "openai")).toEqual({ kind, raw });
  });
  it("reads protocol metadata and legacy refusal codes through causes, never error prose", () => {
    for (const cause of [
      new LLMRefusalError(),
      new LLMError("legacy", { code: "LLM_REFUSAL_EXHAUSTED" }),
    ]) {
      expect(getLlmErrorStopReason(new Error("wrapper", { cause }))).toEqual({
        kind: "refusal",
        raw: "refusal",
      });
    }
    expect(getLlmErrorStopReason(new Error("refusal LLM_REFUSAL_EXHAUSTED"))).toEqual({
      kind: "unknown",
      raw: null,
    });
  });
});
