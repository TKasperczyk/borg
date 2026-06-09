import { describe, expect, it, vi } from "vitest";

import type { LLMCompleteResult, LLMMessage, LLMToolDefinition } from "../llm/index.js";
import type { TurnTracer } from "./tracer.js";
import {
  countCompletePromptChars,
  summarizeToolSchemas,
  traceLlmCallError,
  traceLlmCallResponse,
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
      usage: null,
    });
  });
});
