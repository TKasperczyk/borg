import { describe, expect, it, vi } from "vitest";
import { LLMRefusalError } from "../llm/index.js";
import { LLMError } from "../util/errors.js";
import { attachLlmErrorUsage } from "../llm/usage.js";
import { StructuredToolCallError } from "../llm/structured-tool-call.js";

import type { LLMCompleteResult, LLMMessage, LLMToolDefinition } from "../llm/index.js";
import { compositeTracer, type TurnTracer } from "./tracer.js";
import {
  countCompletePromptChars,
  summarizeToolSchemas,
  traceLlmCallError,
  traceLlmCallResponse,
  traceLlmCallRetryHook,
  traceLlmCallStarted,
} from "./llm-call-trace.js";

function createTracer() {
  const emit = vi.fn<TurnTracer["emit"]>();

  return {
    enabled: true,
    includePayloads: false,
    emit,
  } satisfies TurnTracer & { emit: typeof emit };
}

describe("llm call trace helpers", () => {
  const messages: readonly LLMMessage[] = [
    { role: "user", content: "hello" },
    { role: "assistant", content: "hi" },
  ];

  const tools: readonly LLMToolDefinition[] = [
    {
      name: "EmitThing",
      inputSchema: {
        type: "object",
        properties: {
          name: { type: "string" },
          score: { type: "number" },
        },
        required: ["name"],
      },
    },
    {
      name: "EmitEmpty",
      inputSchema: {
        type: "object",
      },
    },
  ];

  it("counts complete prompt characters using the established trace formula", () => {
    expect(countCompletePromptChars("system", messages)).toBe(
      "system".length + "user".length + "hello".length + "assistant".length + "hi".length,
    );
  });

  it("summarizes tool schemas without including full schema payloads", () => {
    expect(summarizeToolSchemas(tools)).toEqual([
      {
        name: "EmitThing",
        propertyCount: 2,
        required: ["name"],
      },
      {
        name: "EmitEmpty",
        propertyCount: 0,
        required: [],
      },
    ]);
  });

  it("emits llm_call.started with optional tool schemas and extra payloads", () => {
    const tracer = createTracer();

    traceLlmCallStarted({
      tracer,
      turnId: "turn_1",
      label: "test_label",
      model: "model-a",
      systemPrompt: "system",
      messages,
      tools,
      extra: { prompt: { type: "object" } },
    });

    expect(tracer.emit).toHaveBeenCalledWith("llm_call.started", {
      turnId: "turn_1",
      label: "test_label",
      model: "model-a",
      promptCharCount: countCompletePromptChars("system", messages),
      toolSchemas: summarizeToolSchemas(tools),
      prompt: { type: "object" },
    });
  });

  it("omits toolSchemas when no tools are supplied", () => {
    const tracer = createTracer();

    traceLlmCallStarted({
      tracer,
      turnId: "turn_1",
      label: "closure_loop_classifier",
      model: "model-a",
      systemPrompt: "system",
      messages,
    });

    expect(tracer.emit).toHaveBeenCalledWith("llm_call.started", {
      turnId: "turn_1",
      label: "closure_loop_classifier",
      model: "model-a",
      promptCharCount: countCompletePromptChars("system", messages),
    });
  });

  it("emits llm_call.completed responses with usage and caller-provided shape", () => {
    const tracer = createTracer();
    const response: LLMCompleteResult = {
      text: "ok",
      input_tokens: 7,
      output_tokens: 3,
      cache_creation_input_tokens: 11,
      cache_read_input_tokens: 13,
      stop_reason: "tool_use",
      tool_calls: [{ id: "toolu_1", name: "EmitThing", input: { name: "x" } }],
    };

    traceLlmCallResponse({
      tracer,
      turnId: "turn_1",
      label: "test_label",
      response,
      responseShape: { textLength: 2 },
      extra: { response: { type: "object" } },
    });

    expect(tracer.emit).toHaveBeenCalledWith("llm_call.completed", {
      turnId: "turn_1",
      label: "test_label",
      responseShape: { textLength: 2 },
      stopReason: "tool_use",
      stopReasonClassification: { kind: "tool_use", raw: "tool_use" },
      usage: {
        inputTokens: 7,
        outputTokens: 3,
        cacheCreationInputTokens: 11,
        cacheReadInputTokens: 13,
      },
      response: { type: "object" },
    });
  });

  it("keeps reserved response fields when extra uses the same keys", () => {
    const tracer = createTracer();
    const response: LLMCompleteResult = {
      text: "ok",
      input_tokens: 7,
      output_tokens: 3,
      stop_reason: "end_turn",
      tool_calls: [],
    };

    traceLlmCallResponse({
      tracer,
      turnId: "turn_1",
      label: "actual_label",
      response,
      responseShape: { textLength: 2 },
      extra: {
        label: "wrong_label",
        responseShape: { wrong: true },
        stopReason: "wrong_stop",
        usage: null,
      },
    });

    expect(tracer.emit).toHaveBeenCalledWith("llm_call.completed", {
      turnId: "turn_1",
      label: "actual_label",
      responseShape: { textLength: 2 },
      stopReason: "end_turn",
      stopReasonClassification: { kind: "completed", raw: "end_turn" },
      usage: {
        inputTokens: 7,
        outputTokens: 3,
      },
    });
  });

  it("emits llm_call.completed errors with the established null stop and usage fields", () => {
    const tracer = createTracer();

    traceLlmCallError({
      tracer,
      turnId: "turn_1",
      label: "test_label",
      error: new Error("failed"),
    });

    expect(tracer.emit).toHaveBeenCalledWith("llm_call.completed", {
      turnId: "turn_1",
      label: "test_label",
      responseShape: { error: "failed" },
      stopReason: null,
      stopReasonClassification: { kind: "unknown", raw: null },
      usage: null,
    });
  });

  it("traces a wrapped exhausted refusal with provider reason and accounted usage", () => {
    const tracer = createTracer();
    const refusal = new LLMRefusalError();
    attachLlmErrorUsage(refusal, { input_tokens: 12, output_tokens: 6 });
    const error = new StructuredToolCallError("wrapped", {
      kind: "llm_failed",
      toolName: "Emit",
      acceptedToolNames: ["Emit"],
      stopReason: refusal.stopReason,
      stopReasonClassification: refusal.stopReasonClassification,
      cause: refusal,
      usage: { input_tokens: 12, output_tokens: 6 },
    });
    traceLlmCallError({ tracer, turnId: "turn_1", label: "refusal", error });
    expect(tracer.emit).toHaveBeenCalledWith(
      "llm_call.completed",
      expect.objectContaining({
        stopReason: "refusal",
        stopReasonClassification: { kind: "refusal", raw: "refusal" },
        usage: { inputTokens: 12, outputTokens: 6 },
      }),
    );
    // The provider sent no details with this refusal, so the trace claims none.
    const traced = tracer.emit.mock.calls[0]?.[1] ?? {};
    expect(traced).not.toHaveProperty("stopDetailsCategory");
    expect(traced).not.toHaveProperty("stopDetailsExplanation");
  });

  describe("the provider's reason for a refusal", () => {
    const details = { category: "reasoning_extraction", explanation: "Declined for this request." };
    const payloadTracer = () => ({ ...createTracer(), includePayloads: true });
    const exhausted = () =>
      new Error("wrapped by a caller", { cause: new LLMRefusalError(details) });

    it("traces the category on an exhausted refusal, and the explanation only with payloads", () => {
      const redacted = createTracer();
      const full = payloadTracer();

      traceLlmCallError({
        tracer: redacted,
        turnId: "turn_1",
        label: "refusal",
        error: exhausted(),
      });
      traceLlmCallError({ tracer: full, turnId: "turn_1", label: "refusal", error: exhausted() });

      expect(redacted.emit.mock.calls[0]?.[1]).toMatchObject({
        stopReason: "refusal",
        stopDetailsCategory: "reasoning_extraction",
      });
      expect(redacted.emit.mock.calls[0]?.[1]).not.toHaveProperty("stopDetailsExplanation");
      expect(full.emit.mock.calls[0]?.[1]).toMatchObject({
        stopDetailsCategory: "reasoning_extraction",
        stopDetailsExplanation: "Declined for this request.",
      });
    });

    it("traces the same on the retry of a refused response", () => {
      const redacted = createTracer();
      const full = payloadTracer();
      const event = {
        attempt: 2,
        kind: "refusal" as const,
        code: "LLM_REFUSAL",
        retry_transport: "unary" as const,
        stop_reason_classification: { kind: "refusal" as const, raw: "refusal" },
        stop_details: {
          category: null,
          explanation: "No named category.",
          explanation_chars: 9_000,
        },
      };

      traceLlmCallRetryHook({ tracer: redacted, turnId: "turn_1", label: "refusal" })?.(event);
      traceLlmCallRetryHook({ tracer: full, turnId: "turn_1", label: "refusal" })?.(event);

      // A null category is the provider's answer and is traced as one; the cut length is a count.
      expect(redacted.emit.mock.calls[0]?.[1]).toMatchObject({
        kind: "refusal",
        stopDetailsCategory: null,
        stopDetailsExplanationChars: 9_000,
      });
      expect(redacted.emit.mock.calls[0]?.[1]).not.toHaveProperty("stopDetailsExplanation");
      expect(full.emit.mock.calls[0]?.[1]).toMatchObject({
        stopDetailsCategory: null,
        stopDetailsExplanation: "No named category.",
        stopDetailsExplanationChars: 9_000,
      });
      // A retry that is not a refusal has no such fields.
      const stall = createTracer();
      traceLlmCallRetryHook({ tracer: stall, turnId: "turn_1", label: "stall" })?.({
        attempt: 2,
        kind: "stall",
        retry_transport: "unary",
      });
      expect(stall.emit.mock.calls[0]?.[1]).not.toHaveProperty("stopDetailsCategory");
    });

    it("traces no category beside a stop reason that is not a refusal", () => {
      const tracer = payloadTracer();
      const timeout = Object.assign(new LLMError("timed out", { code: "LLM_CALL_TIMED_OUT" }), {
        stopDetails: details,
      });

      traceLlmCallError({ tracer, turnId: "turn_1", label: "timeout", error: timeout });

      const traced = tracer.emit.mock.calls[0]?.[1] ?? {};
      expect(traced).toMatchObject({ stopReason: null });
      expect(traced).not.toHaveProperty("stopDetailsCategory");
      expect(traced).not.toHaveProperty("stopDetailsExplanation");
    });

    it("gives a redacted sink the category and keeps the explanation to the sink that takes payloads", () => {
      const redacted = createTracer();
      const full = payloadTracer();

      traceLlmCallError({
        tracer: compositeTracer([redacted, full]),
        turnId: "turn_1",
        label: "refusal",
        error: exhausted(),
      });

      expect(redacted.emit.mock.calls[0]?.[1]).toMatchObject({
        stopDetailsCategory: "reasoning_extraction",
      });
      expect(redacted.emit.mock.calls[0]?.[1]).not.toHaveProperty("stopDetailsExplanation");
      expect(full.emit.mock.calls[0]?.[1]).toMatchObject({
        stopDetailsCategory: "reasoning_extraction",
        stopDetailsExplanation: "Declined for this request.",
      });
    });
  });
});
