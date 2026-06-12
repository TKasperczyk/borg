import { describe, expect, it, vi } from "vitest";
import { z } from "zod";

import { BudgetExceededError } from "../util/errors.js";
import type {
  LLMClient,
  LLMCompleteOptions,
  LLMCompleteResult,
  LLMToolDefinition,
} from "./index.js";
import {
  callStructuredTool,
  isStructuredToolCallError,
  StructuredToolCallError,
} from "./structured-tool-call.js";

const TOOL_NAME = "EmitThing";
const ALIAS_TOOL_NAME = "EmitThingAlias";
const TOOL: LLMToolDefinition = {
  name: TOOL_NAME,
  inputSchema: {
    type: "object",
    properties: {
      value: { type: "string" },
    },
    required: ["value"],
  },
};
const schema = z.object({
  value: z.string(),
});

function completeResult(
  toolCalls: LLMCompleteResult["tool_calls"],
  stopReason: LLMCompleteResult["stop_reason"] = "tool_use",
): LLMCompleteResult {
  return {
    text: "",
    input_tokens: 3,
    output_tokens: 4,
    stop_reason: stopReason,
    tool_calls: toolCalls,
  };
}

function client(response: LLMCompleteResult | Error): LLMClient {
  return {
    complete: vi.fn(async () => {
      if (response instanceof Error) {
        throw response;
      }

      return response;
    }),
    converse: vi.fn(async () => {
      throw new Error("not used");
    }),
  };
}

describe("callStructuredTool", () => {
  it("selects the accepted tool call and parses its input", async () => {
    const result = await callStructuredTool({
      llmClient: client(
        completeResult([
          {
            id: "toolu_1",
            name: TOOL_NAME,
            input: { value: "ok" },
          },
        ]),
      ),
      request: {
        model: "model",
        system: "system",
        messages: [{ role: "user", content: "message" }],
        tools: [TOOL],
        tool_choice: { type: "tool", name: TOOL_NAME },
        budget: "test",
      },
      toolName: TOOL_NAME,
      parse: (input) => schema.parse(input),
    });

    expect(result.parsed).toEqual({ value: "ok" });
    expect(result.toolCall.name).toBe(TOOL_NAME);
  });

  it("selects an accepted alias when the primary tool name is absent", async () => {
    const result = await callStructuredTool({
      llmClient: client(
        completeResult([
          {
            id: "toolu_1",
            name: ALIAS_TOOL_NAME,
            input: { value: "alias" },
          },
        ]),
      ),
      request: {
        model: "model",
        system: "system",
        messages: [{ role: "user", content: "message" }],
        tools: [TOOL],
        tool_choice: { type: "tool", name: TOOL_NAME },
        budget: "test",
      },
      toolName: TOOL_NAME,
      acceptedToolNames: [TOOL_NAME, ALIAS_TOOL_NAME],
      parse: (input) => schema.parse(input),
    });

    expect(result.parsed).toEqual({ value: "alias" });
    expect(result.toolCall.name).toBe(ALIAS_TOOL_NAME);
  });

  it("selects the first accepted tool call in response order", async () => {
    const result = await callStructuredTool({
      llmClient: client(
        completeResult([
          {
            id: "toolu_ignored",
            name: "OtherTool",
            input: { value: "ignored" },
          },
          {
            id: "toolu_alias",
            name: ALIAS_TOOL_NAME,
            input: { value: "first" },
          },
          {
            id: "toolu_primary",
            name: TOOL_NAME,
            input: { value: "second" },
          },
        ]),
      ),
      request: {
        model: "model",
        system: "system",
        messages: [{ role: "user", content: "message" }],
        tools: [TOOL],
        tool_choice: { type: "tool", name: TOOL_NAME },
        budget: "test",
      },
      toolName: TOOL_NAME,
      acceptedToolNames: [TOOL_NAME, ALIAS_TOOL_NAME],
      parse: (input) => schema.parse(input),
    });

    expect(result.parsed).toEqual({ value: "first" });
    expect(result.toolCall.id).toBe("toolu_alias");
  });

  it("throws a typed missing-tool error with stop reason", async () => {
    await expect(() =>
      callStructuredTool({
        llmClient: client(completeResult([], "max_tokens")),
        request: {
          model: "model",
          system: "system",
          messages: [{ role: "user", content: "message" }],
          tools: [TOOL],
          tool_choice: { type: "tool", name: TOOL_NAME },
          budget: "test",
        },
        toolName: TOOL_NAME,
        parse: (input) => schema.parse(input),
      }),
    ).rejects.toMatchObject({
      kind: "missing_tool_call",
      toolName: TOOL_NAME,
      stopReason: "max_tokens",
    });
  });

  it("wraps parser failures as invalid payload errors with stop reason", async () => {
    try {
      await callStructuredTool({
        llmClient: client(
          completeResult([
            {
              id: "toolu_1",
              name: TOOL_NAME,
              input: { value: 42 },
            },
          ], "max_tokens"),
        ),
        request: {
          model: "model",
          system: "system",
          messages: [{ role: "user", content: "message" }],
          tools: [TOOL],
          tool_choice: { type: "tool", name: TOOL_NAME },
          budget: "test",
        },
        toolName: TOOL_NAME,
        parse: (input) => schema.parse(input),
      });
      throw new Error("expected call to fail");
    } catch (error) {
      expect(isStructuredToolCallError(error, "invalid_payload")).toBe(true);
      expect((error as StructuredToolCallError).cause).toBeInstanceOf(z.ZodError);
      expect((error as StructuredToolCallError).stopReason).toBe("max_tokens");
    }
  });

  it("wraps LLM failures with the original cause and emits traced errors", async () => {
    const originalError = new Error("transport failed");
    const emit = vi.fn();

    try {
      await callStructuredTool({
        llmClient: client(originalError),
        request: {
          model: "model",
          system: "system",
          messages: [{ role: "user", content: "message" }],
          tools: [TOOL],
          tool_choice: { type: "tool", name: TOOL_NAME },
          budget: "test",
        },
        toolName: TOOL_NAME,
        parse: (input) => schema.parse(input),
        trace: {
          tracer: {
            enabled: true,
            includePayloads: false,
            emit,
          },
          turnId: "turn_1",
          label: "test_call",
        },
      });
      throw new Error("expected call to fail");
    } catch (error) {
      expect(isStructuredToolCallError(error, "llm_failed")).toBe(true);
      expect((error as StructuredToolCallError).cause).toBe(originalError);
      expect((error as StructuredToolCallError).stopReason).toBeNull();
    }

    expect(emit).toHaveBeenCalledWith(
      "llm_call.started",
      expect.objectContaining({ label: "test_call" }),
    );
    expect(emit).toHaveBeenCalledWith(
      "llm_call.completed",
      expect.objectContaining({
        label: "test_call",
        responseShape: { error: "transport failed" },
        stopReason: null,
        usage: null,
      }),
    );
  });

  it("wires transport-retry tracing into the request", async () => {
    const emit = vi.fn();
    const complete = vi.fn(async (request: LLMCompleteOptions) => {
      request.onTransportRetry?.({
        attempt: 2,
        kind: "stall",
        code: "LLM_STREAM_EVENT_STALLED",
        retry_transport: "unary",
      });

      return completeResult([
        {
          id: "toolu_1",
          name: TOOL_NAME,
          input: { value: "ok" },
        },
      ]);
    });

    await callStructuredTool({
      llmClient: {
        complete,
        converse: vi.fn(async () => {
          throw new Error("not used");
        }),
      },
      request: {
        model: "model",
        system: "system",
        messages: [{ role: "user", content: "message" }],
        tools: [TOOL],
        tool_choice: { type: "tool", name: TOOL_NAME },
        budget: "test",
      },
      toolName: TOOL_NAME,
      parse: (input) => schema.parse(input),
      trace: {
        tracer: {
          enabled: true,
          includePayloads: false,
          emit,
        },
        turnId: "turn_1",
        label: "test_call",
      },
    });

    expect(emit).toHaveBeenCalledWith("llm_call.retried", {
      turnId: "turn_1",
      label: "test_call",
      attempt: 2,
      kind: "stall",
      code: "LLM_STREAM_EVENT_STALLED",
      retry_transport: "unary",
    });
  });

  it("can omit tool schemas from started traces", async () => {
    const emit = vi.fn();
    await callStructuredTool({
      llmClient: client(
        completeResult([
          {
            id: "toolu_1",
            name: TOOL_NAME,
            input: { value: "ok" },
          },
        ]),
      ),
      request: {
        model: "model",
        system: "system",
        messages: [{ role: "user", content: "message" }],
        tools: [TOOL],
        tool_choice: { type: "tool", name: TOOL_NAME },
        budget: "test",
      },
      toolName: TOOL_NAME,
      parse: (input) => schema.parse(input),
      trace: {
        tracer: {
          enabled: true,
          includePayloads: false,
          emit,
        },
        turnId: "turn_1",
        label: "test_call",
        includeToolSchemas: false,
      },
    });

    expect(emit).toHaveBeenCalledWith(
      "llm_call.started",
      expect.not.objectContaining({ toolSchemas: expect.anything() }),
    );
    expect(emit).toHaveBeenCalledWith(
      "llm_call.completed",
      expect.objectContaining({ label: "test_call" }),
    );
  });

  it("passes budget errors through untouched and emits traced errors", async () => {
    const budgetError = new BudgetExceededError("budget");
    const emit = vi.fn();

    await expect(() =>
      callStructuredTool({
        llmClient: client(budgetError),
        request: {
          model: "model",
          system: "system",
          messages: [{ role: "user", content: "message" }],
          tools: [TOOL],
          tool_choice: { type: "tool", name: TOOL_NAME },
          budget: "test",
        },
        toolName: TOOL_NAME,
        parse: (input) => schema.parse(input),
        trace: {
          tracer: {
            enabled: true,
            includePayloads: false,
            emit,
          },
          turnId: "turn_1",
          label: "test_call",
        },
      }),
    ).rejects.toBe(budgetError);

    expect(emit).toHaveBeenCalledWith(
      "llm_call.completed",
      expect.objectContaining({
        label: "test_call",
        responseShape: { error: "budget" },
        stopReason: null,
        usage: null,
      }),
    );
  });
});
