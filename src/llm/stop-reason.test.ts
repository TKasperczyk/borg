import { describe, expect, it } from "vitest";
import { LLMRefusalError } from "./index.js";
import { LLMError, LLMToolArgumentsError } from "../util/errors.js";
import {
  classifyLlmStopReason,
  getLlmErrorStopReason,
  normalizeLlmStopReason,
} from "./stop-reason.js";

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

  it.each(["max_tokens", "length"])("keeps raw %s as a cut despite other metadata", (raw) => {
    for (const kind of ["unknown", "completed", "tool_use", "refusal"] as const) {
      const classification = { kind, raw: "other" };
      expect(normalizeLlmStopReason(raw, classification)).toEqual({ kind: "max_tokens", raw });
      const cause = new LLMToolArgumentsError("Invalid arguments", {
        toolName: "EmitResult",
        stopReason: raw,
        stopReasonClassification: classification,
      });
      expect(getLlmErrorStopReason(new Error("wrapper", { cause }))).toEqual({
        kind: "max_tokens",
        raw,
      });
    }
    expect(
      getLlmErrorStopReason(
        new LLMToolArgumentsError("Invalid", { toolName: "EmitResult", stopReason: raw }),
      ),
    ).toEqual({ kind: "max_tokens", raw });
  });

  it.each([
    ["tool_use", "tool_use"],
    ["end_turn", "completed"],
    ["refusal", "refusal"],
    ["pause_turn", "pause_turn"],
    ["future_reason", "unknown"],
    ["", "unknown"],
  ])("does not classify explicit raw %s as a cut despite cut metadata", (raw, kind) => {
    const classification = { kind: "max_tokens", raw: "length" } as const;
    expect(normalizeLlmStopReason(raw!, classification)).toEqual({ kind, raw });
    const cause = new LLMToolArgumentsError("Invalid arguments", {
      toolName: "EmitResult",
      stopReason: raw!,
      stopReasonClassification: classification,
    });
    expect(getLlmErrorStopReason(new Error("wrapper", { cause }))).toEqual({ kind, raw });
  });

  it("retains compatible provider details and uses cut metadata when the raw reason is absent", () => {
    const cut = { kind: "max_tokens", raw: "length" } as const;
    expect(normalizeLlmStopReason("max_tokens", cut)).toEqual(cut);
    expect(normalizeLlmStopReason(null, cut)).toEqual(cut);
    expect(
      getLlmErrorStopReason(
        new LLMToolArgumentsError("Invalid", {
          toolName: "EmitResult",
          stopReason: null,
          stopReasonClassification: cut,
        }),
      ),
    ).toEqual(cut);
    expect(normalizeLlmStopReason("stop", undefined, "openai")).toEqual({
      kind: "completed",
      raw: "stop",
    });
  });
});
