import { inspect } from "node:util";

import { describe, expect, it } from "vitest";
import { LLMRefusalError } from "./index.js";
import { LLMError, LLMToolArgumentsError } from "../util/errors.js";
import {
  classifyLlmStopReason,
  getLlmErrorStopDetails,
  getLlmErrorStopReason,
  normalizeLlmStopReason,
  readLlmRefusalStopDetails,
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
  it("reads the provider's refusal details as sent, including a category its SDK does not list", () => {
    expect(
      readLlmRefusalStopDetails({
        stop_reason: "refusal",
        stop_details: { type: "refusal", category: "cyber", explanation: "Declined: cyber." },
      }),
    ).toEqual({ category: "cyber", explanation: "Declined: cyber." });
    // The category is an open set upstream; an unlisted one is kept, never mapped or dropped.
    expect(
      readLlmRefusalStopDetails({
        stop_details: { type: "refusal", category: "reasoning_extraction", explanation: null },
      }),
    ).toEqual({ category: "reasoning_extraction", explanation: null });
    // Details with no named category are still details.
    expect(
      readLlmRefusalStopDetails({ stop_details: { type: "refusal", category: null } }),
    ).toEqual({ category: null, explanation: null });
    // A field of the wrong shape is not coerced into a claim.
    expect(
      readLlmRefusalStopDetails({ stop_details: { category: 7, explanation: ["why"] } }),
    ).toEqual({ category: null, explanation: null });
    // A category is a label; one too long to be a label is cut, on a character boundary.
    expect(
      readLlmRefusalStopDetails({
        stop_details: { category: `${"c".repeat(199)}\u{1F512}tail`, explanation: null },
      }),
    ).toEqual({ category: "c".repeat(199), explanation: null });
  });

  it("reports no refusal details where the provider sent none", () => {
    for (const carrier of [
      { stop_reason: "refusal", stop_details: null },
      { stop_reason: "refusal" },
      { stop_details: "refused" },
      { stop_details: ["cyber"] },
      // An object that names neither field is not the provider answering "no category".
      { stop_details: {} },
      { stop_details: { type: "refusal" } },
      null,
      undefined,
      "refusal",
    ]) {
      expect(readLlmRefusalStopDetails(carrier)).toBeUndefined();
    }
  });

  it("bounds a long explanation on a character boundary and says how long it was", () => {
    // 1,999 units, then an astral character whose two units straddle the 2,000 bound.
    const explanation = `${"x".repeat(1_999)}\u{1F512}${"y".repeat(500)}`;
    const details = readLlmRefusalStopDetails({ stop_details: { category: "bio", explanation } });

    expect(details).toEqual({
      category: "bio",
      explanation: "x".repeat(1_999),
      explanation_chars: explanation.length,
    });
    // At the bound nothing is cut and no length is added.
    expect(
      readLlmRefusalStopDetails({
        stop_details: { category: "bio", explanation: "z".repeat(2_000) },
      }),
    ).toEqual({ category: "bio", explanation: "z".repeat(2_000) });
  });

  it("finds an exhausted refusal's details through a cause chain, and nothing on other errors", () => {
    const details = { category: "cyber", explanation: "Declined: cyber." };

    expect(new LLMRefusalError(details).stopDetails).toEqual(details);
    expect(
      getLlmErrorStopDetails(new Error("wrapper", { cause: new LLMRefusalError(details) })),
    ).toEqual(details);
    // A refusal the provider gave no details for carries none, and says nothing invented.
    expect(new LLMRefusalError()).not.toHaveProperty("stopDetails");
    expect(getLlmErrorStopDetails(new LLMRefusalError())).toBeUndefined();
    expect(
      getLlmErrorStopDetails(new LLMError("other", { code: "LLM_CALL_TIMED_OUT" })),
    ).toBeUndefined();
    expect(getLlmErrorStopDetails(new Error("stopDetails cyber"))).toBeUndefined();
    // Something that is not details is not read as details, even on a refusal.
    for (const stray of [["cyber"], "cyber", { category: 7, explanation: null }, null]) {
      const refusal = Object.assign(new LLMError("legacy", { code: "LLM_REFUSAL_EXHAUSTED" }), {
        stopDetails: stray,
      });
      expect(getLlmErrorStopDetails(refusal)).toBeUndefined();
    }
    // An error whose stop reason is not a refusal reports none, whatever it carries.
    expect(
      getLlmErrorStopDetails(
        Object.assign(new LLMError("other", { code: "LLM_CALL_TIMED_OUT" }), {
          stopDetails: { category: "cyber", explanation: null },
        }),
      ),
    ).toBeUndefined();
  });

  it("keeps the provider's explanation off an error that is logged whole or copied", () => {
    const refusal = new LLMRefusalError({ category: "cyber", explanation: "PROVIDER PROSE" });

    // Read by name, by the tracing that gates it.
    expect(refusal.stopDetails).toEqual({ category: "cyber", explanation: "PROVIDER PROSE" });
    // Not enumerable, so nothing that walks the error's own fields carries it along.
    expect(Object.keys(refusal)).not.toContain("stopDetails");
    expect({ ...refusal }).not.toHaveProperty("stopDetails");
    expect(inspect(refusal)).not.toContain("PROVIDER PROSE");
    expect(JSON.stringify(refusal)).not.toContain("PROVIDER PROSE");
    expect(inspect(new Error("wrapper", { cause: refusal }))).not.toContain("PROVIDER PROSE");
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
