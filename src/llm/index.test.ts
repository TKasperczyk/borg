import { describe, expect, it, vi } from "vitest";

import type { Message } from "@anthropic-ai/sdk/resources/messages/messages.js";

import { ConfigError } from "../util/errors.js";
import { AnthropicLLMClient, FakeLLMClient, type TokenUsageEvent } from "./index.js";

describe("llm", () => {
  it("wraps anthropic messages and extracts tool calls", async () => {
    const usageEvents: TokenUsageEvent[] = [];

    const message = {
      id: "msg_1",
      container: null,
      content: [
        { type: "text", text: "Hello", citations: null },
        {
          type: "tool_use",
          id: "toolu_1",
          caller: { type: "direct" },
          name: "lookup",
          input: { id: 1 },
        },
      ],
      model: "claude-sonnet-4-5",
      role: "assistant",
      stop_details: null,
      stop_reason: "tool_use",
      stop_sequence: null,
      type: "message",
      usage: {
        cache_creation: null,
        cache_creation_input_tokens: null,
        cache_read_input_tokens: null,
        input_tokens: 12,
        output_tokens: 7,
        server_tool_use: null,
      },
    } as unknown as Message;

    const create = vi.fn().mockResolvedValue(message);
    const client = new AnthropicLLMClient({
      client: {
        messages: { create },
      },
      usageSink: async (event) => {
        usageEvents.push(event);
      },
    });

    const result = await client.complete({
      model: "claude-sonnet-4-5",
      system: "be concise",
      messages: [{ role: "user", content: "hello" }],
      tools: [
        {
          name: "lookup",
          inputSchema: {
            type: "object",
            properties: { id: { type: "number" } },
            required: ["id"],
          },
        },
      ],
      max_tokens: 128,
      budget: "test",
    });

    expect(result).toEqual({
      text: "Hello",
      input_tokens: 12,
      output_tokens: 7,
      stop_reason: "tool_use",
      tool_calls: [
        {
          id: "toolu_1",
          name: "lookup",
          input: { id: 1 },
        },
      ],
    });
    expect(create).toHaveBeenCalledTimes(1);
    expect(usageEvents).toEqual([
      {
        budget: "test",
        model: "claude-sonnet-4-5",
        input_tokens: 12,
        output_tokens: 7,
      },
    ]);
  });

  it("requires an api key when no anthropic client is injected", () => {
    expect(() => new AnthropicLLMClient()).toThrow(ConfigError);
  });

  it("supports scripted fake llm responses", async () => {
    const usageSink = vi.fn();
    const client = new FakeLLMClient({
      responses: [
        {
          text: "ok",
          input_tokens: 1,
          output_tokens: 2,
          stop_reason: "end_turn",
          tool_calls: [],
        },
      ],
      usageSink,
    });

    const result = await client.complete({
      model: "fake",
      messages: [{ role: "user", content: "hi" }],
      max_tokens: 8,
      budget: "test",
    });

    expect(result.text).toBe("ok");
    expect(client.requests).toHaveLength(1);
    expect(usageSink).toHaveBeenCalledWith({
      budget: "test",
      model: "fake",
      input_tokens: 1,
      output_tokens: 2,
    });
  });
});
