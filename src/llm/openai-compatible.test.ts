import { describe, expect, it } from "vitest";

import { OpenAICompatibleLLMClient, type OpenAIChatCompletionsClient } from "./openai-compatible.js";
import { callStructuredTool, isStructuredToolCallError } from "./structured-tool-call.js";
import type { LLMCompleteOptions, LLMConverseOptions, LLMToolDefinition } from "./index.js";
import { ConfigError, LLMError } from "../util/errors.js";

type CapturedParams = Record<string, unknown>;

function fakeClient(
  response: unknown,
  capture?: (params: CapturedParams) => void,
): OpenAIChatCompletionsClient {
  return {
    chat: {
      completions: {
        // eslint-disable-next-line @typescript-eslint/require-await
        async create(params) {
          capture?.(params);
          // eslint-disable-next-line @typescript-eslint/no-explicit-any
          return response as any;
        },
      },
    },
  };
}

const TOOL: LLMToolDefinition = {
  name: "EmitFacts",
  description: "Emit extracted facts",
  inputSchema: { type: "object", properties: { facts: { type: "array" } }, required: ["facts"] },
};

function completeOptions(overrides: Partial<LLMCompleteOptions> = {}): LLMCompleteOptions {
  return {
    model: "generative-apis/qwen3-235b-a22b-instruct-2507",
    budget: "memory.extract",
    messages: [{ role: "user", content: "remember this" }],
    tools: [TOOL],
    tool_choice: { type: "tool", name: "EmitFacts" },
    ...overrides,
  };
}

function toolCallResponse(args: string) {
  return {
    choices: [
      {
        message: {
          content: null,
          tool_calls: [{ id: "call_1", type: "function", function: { name: "EmitFacts", arguments: args } }],
        },
        finish_reason: "tool_calls",
      },
    ],
    usage: { prompt_tokens: 11, completion_tokens: 7 },
  };
}

describe("OpenAICompatibleLLMClient", () => {
  it("maps a forced tool_choice to OpenAI function-forcing and decodes tool args", async () => {
    let captured: CapturedParams = {};
    const client = new OpenAICompatibleLLMClient({
      client: fakeClient(toolCallResponse('{"facts":["a","b"]}'), (p) => (captured = p)),
    });

    const result = await client.complete(completeOptions());

    expect(captured.model).toBe("generative-apis/qwen3-235b-a22b-instruct-2507");
    expect(captured.tool_choice).toEqual({ type: "function", function: { name: "EmitFacts" } });
    expect(captured.parallel_tool_calls).toBe(false);
    expect(captured.max_tokens).toBe(8192);
    expect(captured.max_completion_tokens).toBeUndefined();
    expect(captured.tools).toEqual([
      {
        type: "function",
        function: { name: "EmitFacts", description: "Emit extracted facts", parameters: TOOL.inputSchema },
      },
    ]);

    expect(result.stop_reason).toBe("tool_use");
    expect(result.tool_calls).toEqual([{ id: "call_1", name: "EmitFacts", input: { facts: ["a", "b"] } }]);
    expect(result.input_tokens).toBe(11);
    expect(result.output_tokens).toBe(7);
  });

  it("flattens a system block array and prepends it as a system message", async () => {
    let captured: CapturedParams = {};
    const client = new OpenAICompatibleLLMClient({
      client: fakeClient(toolCallResponse("{}"), (p) => (captured = p)),
    });

    await client.complete(
      completeOptions({
        system: [
          { type: "text", text: "line one", cache_control: { type: "ephemeral" } },
          { type: "text", text: "line two" },
        ],
      }),
    );

    const messages = captured.messages as Array<{ role: string; content: string }>;
    expect(messages[0]).toEqual({ role: "system", content: "line one\n\nline two" });
    expect(messages[1]).toEqual({ role: "user", content: "remember this" });
  });

  it("maps {type:'any'} to 'required' and {type:'auto'} to 'auto'", async () => {
    let captured: CapturedParams = {};
    const client = new OpenAICompatibleLLMClient({
      client: fakeClient(toolCallResponse("{}"), (p) => (captured = p)),
    });

    await client.complete(completeOptions({ tool_choice: { type: "any" } }));
    expect(captured.tool_choice).toBe("required");

    await client.complete(completeOptions({ tool_choice: { type: "auto" } }));
    expect(captured.tool_choice).toBe("auto");
  });

  it("honors the max_completion_tokens field switch for stricter gateways", async () => {
    let captured: CapturedParams = {};
    const client = new OpenAICompatibleLLMClient({
      maxTokensField: "max_completion_tokens",
      client: fakeClient(toolCallResponse("{}"), (p) => (captured = p)),
    });

    await client.complete(completeOptions({ max_tokens: 512 }));
    expect(captured.max_completion_tokens).toBe(512);
    expect(captured.max_tokens).toBeUndefined();
  });

  it("throws LLMError on unparseable tool arguments (becomes llm_failed via callStructuredTool)", async () => {
    const client = new OpenAICompatibleLLMClient({
      client: fakeClient(toolCallResponse("{not json")),
    });

    await expect(client.complete(completeOptions())).rejects.toBeInstanceOf(LLMError);

    const failure = await callStructuredTool({
      llmClient: client,
      request: completeOptions(),
      toolName: "EmitFacts",
      parse: (input) => input,
    }).catch((error: unknown) => error);

    expect(isStructuredToolCallError(failure, "llm_failed")).toBe(true);
  });

  it("drives callStructuredTool end-to-end on a well-formed forced tool call", async () => {
    const client = new OpenAICompatibleLLMClient({
      client: fakeClient(toolCallResponse('{"facts":["x"]}')),
    });

    const { parsed, toolCall } = await callStructuredTool<{ facts: string[] }>({
      llmClient: client,
      request: completeOptions(),
      toolName: "EmitFacts",
      parse: (input) => input as { facts: string[] },
    });

    expect(toolCall.name).toBe("EmitFacts");
    expect(parsed).toEqual({ facts: ["x"] });
  });

  it("maps stop/length finish reasons and reads plain text", async () => {
    const client = new OpenAICompatibleLLMClient({
      client: fakeClient({
        choices: [{ message: { content: "hello", tool_calls: null }, finish_reason: "stop" }],
        usage: { prompt_tokens: 3, completion_tokens: 2 },
      }),
    });

    const result = await client.complete(completeOptions({ tools: undefined, tool_choice: undefined }));
    expect(result.text).toBe("hello");
    expect(result.stop_reason).toBe("end_turn");
    expect(result.tool_calls).toEqual([]);
  });

  it("translates converse content blocks (text, tool_use, tool_result) into OpenAI messages", async () => {
    let captured: CapturedParams = {};
    const client = new OpenAICompatibleLLMClient({
      client: fakeClient(
        {
          choices: [
            {
              message: {
                content: "done",
                tool_calls: [
                  { id: "c2", type: "function", function: { name: "EmitFacts", arguments: '{"facts":[]}' } },
                ],
              },
              finish_reason: "tool_calls",
            },
          ],
          usage: { prompt_tokens: 1, completion_tokens: 1 },
        },
        (p) => (captured = p),
      ),
    });

    const options: LLMConverseOptions = {
      model: "qwen",
      budget: "x",
      messages: [
        { role: "user", content: [{ type: "text", text: "hi" }] },
        {
          role: "assistant",
          content: [
            { type: "text", text: "calling" },
            { type: "tool_use", id: "c1", name: "EmitFacts", input: { facts: [] } },
          ],
        },
        { role: "user", content: [{ type: "tool_result", tool_use_id: "c1", content: "ok" }] },
      ],
    };

    const result = await client.converse(options);

    const messages = captured.messages as Array<Record<string, unknown>>;
    expect(messages).toEqual([
      { role: "user", content: "hi" },
      {
        role: "assistant",
        content: "calling",
        tool_calls: [{ id: "c1", type: "function", function: { name: "EmitFacts", arguments: '{"facts":[]}' } }],
      },
      { role: "tool", tool_call_id: "c1", content: "ok" },
    ]);

    expect(result.messageBlocks).toEqual([
      { type: "text", text: "done" },
      { type: "tool_use", id: "c2", name: "EmitFacts", input: { facts: [] } },
    ]);
  });

  it("rejects image_ref blocks in converse (no attachment resolver)", async () => {
    const client = new OpenAICompatibleLLMClient({ client: fakeClient({ choices: [] }) });

    const options: LLMConverseOptions = {
      model: "qwen",
      budget: "x",
      // eslint-disable-next-line @typescript-eslint/no-explicit-any
      messages: [{ role: "user", content: [{ type: "image_ref", attachment_id: "att_1" as any }] }],
    };

    await expect(client.converse(options)).rejects.toBeInstanceOf(LLMError);
  });

  it("requires baseUrl and apiKey when no client is injected", () => {
    expect(() => new OpenAICompatibleLLMClient({ apiKey: "k" })).toThrow(ConfigError);
    expect(() => new OpenAICompatibleLLMClient({ baseUrl: "https://x/v1" })).toThrow(ConfigError);
  });
});
