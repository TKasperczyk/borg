import { afterEach, describe, expect, it, vi } from "vitest";
import { z } from "zod";
import { APIConnectionError } from "@anthropic-ai/sdk";
import type { Message } from "@anthropic-ai/sdk/resources/messages/messages.js";
import { mkdtempSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";

import { AnthropicLLMClient, type LLMClient } from "./index.js";
import { OpenAICompatibleLLMClient } from "./openai-compatible.js";
import { callStructuredTool } from "./structured-tool-call.js";
import { writeJsonFileAtomic } from "../util/atomic-write.js";

const TOOL_NAME = "EmitResult";
const REPAIR = "The previous output was cut. Emit one complete result within 16000 output tokens.";
const schema = z.object({ value: z.string() });
const request = {
  model: "claude-opus-5-5",
  tools: [{ name: TOOL_NAME, inputSchema: { type: "object" as const } }],
  tool_choice: { type: "tool" as const, name: TOOL_NAME },
  messages: [{ role: "user" as const, content: "Produce the result." }],
  max_tokens: 16_000,
  budget: "structured-transport-test",
};

type Reply = {
  stop: "max_tokens" | "length" | "end_turn" | "tool_use" | "refusal";
  input?: unknown;
};

function anthropicReply(reply: Reply): Message {
  return {
    id: "msg_1",
    type: "message",
    role: "assistant",
    model: request.model,
    content:
      reply.input === undefined
        ? [{ type: "text", text: "Unfinished response.", citations: null }]
        : [{ type: "tool_use", id: "call_1", name: TOOL_NAME, input: reply.input }],
    stop_reason: reply.stop,
    stop_sequence: null,
    usage: { input_tokens: 12, output_tokens: 7 },
  } as Message;
}

function injectedClient(provider: "anthropic" | "openai", replies: Array<Reply | Error>) {
  const nextReply = () => {
    const reply = replies.shift();
    if (reply === undefined) throw new Error("Unexpected provider call");
    if (reply instanceof Error) throw reply;
    return reply;
  };
  if (provider === "openai") {
    const create = vi.fn(async (_params: unknown, _options?: unknown) => {
      const reply = nextReply();
      return {
        choices: [
          {
            message: {
              role: "assistant" as const,
              content: reply.input === undefined ? "Unfinished response." : "",
              ...(reply.input === undefined
                ? {}
                : {
                    tool_calls: [
                      {
                        id: "call_1",
                        type: "function" as const,
                        function: { name: TOOL_NAME, arguments: JSON.stringify(reply.input) },
                      },
                    ],
                  }),
            },
            finish_reason:
              reply.stop === "max_tokens" || reply.stop === "length"
                ? "length"
                : reply.stop === "tool_use"
                  ? "tool_calls"
                  : "stop",
          },
        ],
        usage: { prompt_tokens: 12, completion_tokens: 7 },
      };
    });
    return {
      client: new OpenAICompatibleLLMClient({ client: { chat: { completions: { create } } } }),
      create,
    };
  }
  const create = vi.fn(async (_params: unknown, _options?: unknown): Promise<Message> => {
    return anthropicReply(nextReply());
  });
  return { client: new AnthropicLLMClient({ client: { messages: { create } } }), create };
}

const tempDirs: string[] = [];
afterEach(() => {
  vi.restoreAllMocks();
  vi.unstubAllGlobals();
  while (tempDirs.length > 0) rmSync(tempDirs.pop()!, { recursive: true, force: true });
});

describe.each(["anthropic", "openai"] as const)(
  "structured repair over %s transport",
  (provider) => {
    function run(client: LLMClient) {
      return callStructuredTool({
        llmClient: client,
        request,
        toolName: TOOL_NAME,
        outputLimitRepairInstruction: REPAIR,
        parse: (input) => schema.parse(input),
      });
    }

    it("retries a no-tool cut once with cut guidance and no replay", async () => {
      vi.spyOn(console, "warn").mockImplementation(() => undefined);
      const { client, create } = injectedClient(provider, [
        { stop: "max_tokens" },
        { stop: "tool_use", input: { value: "complete" } },
      ]);
      const result = await run(client);
      expect(result.parsed).toEqual({ value: "complete" });
      expect(result.usage).toMatchObject({ input_tokens: 24, output_tokens: 14 });
      expect(create).toHaveBeenCalledTimes(2);
      expect(create.mock.calls[1]![0]).toMatchObject({
        messages: [...request.messages, { role: "user", content: REPAIR }],
      });
      for (const [params] of create.mock.calls) {
        expect(params).not.toHaveProperty("outputLimitRepairBudget");
      }
    });

    it("ends two no-tool cuts with the output-limit error", async () => {
      vi.spyOn(console, "warn").mockImplementation(() => undefined);
      const { client, create } = injectedClient(provider, [
        { stop: "max_tokens" },
        { stop: "max_tokens" },
      ]);
      await expect(run(client)).rejects.toMatchObject({
        kind: "llm_failed",
        stopReason: "max_tokens",
        cause: { message: "Structured tool response reached its output limit" },
        usage: { input_tokens: 24, output_tokens: 14 },
      });
      expect(create).toHaveBeenCalledTimes(2);
    });

    it.each([true, false])(
      "keeps the normal missing-tool reminder with instruction=%s",
      async (guided) => {
        vi.spyOn(console, "warn").mockImplementation(() => undefined);
        const { client, create } = injectedClient(provider, [
          { stop: "end_turn" },
          { stop: "tool_use", input: { value: "complete" } },
        ]);
        const result = await callStructuredTool({
          llmClient: client,
          request,
          toolName: TOOL_NAME,
          ...(guided ? { outputLimitRepairInstruction: REPAIR } : {}),
          parse: (input) => schema.parse(input),
        });
        expect(result.parsed).toEqual({ value: "complete" });
        expect(create).toHaveBeenCalledTimes(2);
        expect(create.mock.calls[1]![0]).toMatchObject({
          messages: [
            ...request.messages,
            expect.objectContaining({ role: "assistant" }),
            { role: "user", content: `Answer only by calling the ${TOOL_NAME} tool.` },
          ],
        });
      },
    );

    it("retries a cut without caller guidance without replaying it", async () => {
      vi.spyOn(console, "warn").mockImplementation(() => undefined);
      const { client, create } = injectedClient(provider, [
        { stop: "max_tokens" },
        { stop: "tool_use", input: { value: "complete" } },
      ]);
      const result = await callStructuredTool({
        llmClient: client,
        request,
        toolName: TOOL_NAME,
        parse: (input) => schema.parse(input),
      });
      expect(result.parsed).toEqual({ value: "complete" });
      expect(create).toHaveBeenCalledTimes(2);
      expect(create.mock.calls[1]![0]).toMatchObject({ max_tokens: 16_000 });
      expect(create.mock.calls[1]![0]).toMatchObject({
        messages: [
          ...request.messages,
          { role: "user", content: expect.stringContaining("discarded whole") },
        ],
      });
    });

    it("ends two unguided no-tool cuts without a reminder", async () => {
      vi.spyOn(console, "warn").mockImplementation(() => undefined);
      const { client, create } = injectedClient(provider, [
        { stop: "max_tokens" },
        { stop: "max_tokens" },
      ]);
      await expect(
        callStructuredTool({
          llmClient: client,
          request,
          toolName: TOOL_NAME,
          parse: (input) => schema.parse(input),
        }),
      ).rejects.toMatchObject({
        kind: "llm_failed",
        stopReason: "max_tokens",
        message: expect.stringContaining("16000"),
      });
      expect(create).toHaveBeenCalledTimes(2);
    });

    it.each([
      [
        { stop: "end_turn" },
        { stop: "tool_use", input: {} },
        { stop: "tool_use", input: { value: "complete" } },
      ],
      [
        { stop: "tool_use", input: {} },
        { stop: "end_turn" },
        { stop: "tool_use", input: { value: "complete" } },
      ],
    ] satisfies Reply[][])("preserves a normal three-call path: %j", async (...replies) => {
      vi.spyOn(console, "warn").mockImplementation(() => undefined);
      const { client, create } = injectedClient(provider, replies);
      const result = await callStructuredTool({
        llmClient: client,
        request,
        toolName: TOOL_NAME,
        parse: (input) => schema.parse(input),
      });
      expect(result.parsed).toEqual({ value: "complete" });
      expect(create).toHaveBeenCalledTimes(3);
    });

    it("does not follow a cut on the reminder with another repair", async () => {
      vi.spyOn(console, "warn").mockImplementation(() => undefined);
      const { client, create } = injectedClient(provider, [
        { stop: "end_turn" },
        { stop: "max_tokens" },
      ]);
      await expect(
        callStructuredTool({
          llmClient: client,
          request,
          toolName: TOOL_NAME,
          parse: (input) => schema.parse(input),
        }),
      ).rejects.toMatchObject({ kind: "llm_failed", stopReason: "max_tokens" });
      expect(create).toHaveBeenCalledTimes(2);
    });

    it.each([
      { first: "end_turn", second: "max_tokens", kind: "llm_failed", stop: "max_tokens" },
      { first: "max_tokens", second: "end_turn", kind: "missing_tool_call", stop: "end_turn" },
    ] as const)(
      "bounds a mixed $first then $second path at two calls",
      async ({ first, second, kind, stop }) => {
        vi.spyOn(console, "warn").mockImplementation(() => undefined);
        const { client, create } = injectedClient(provider, [{ stop: first }, { stop: second }]);
        await expect(run(client)).rejects.toMatchObject({
          kind,
          stopReason: stop,
          usage: { input_tokens: 24, output_tokens: 14 },
        });
        expect(create).toHaveBeenCalledTimes(2);
      },
    );

    it("does not start a third schema repair after a reminder used the second call", async () => {
      vi.spyOn(console, "warn").mockImplementation(() => undefined);
      const { client, create } = injectedClient(provider, [
        { stop: "end_turn" },
        { stop: "tool_use", input: {} },
        { stop: "max_tokens" },
      ]);
      await expect(run(client)).rejects.toMatchObject({ kind: "invalid_payload" });
      expect(create).toHaveBeenCalledTimes(2);
    });
  },
);

it("skips the reminder even when an Anthropic-compatible transport returns raw length", async () => {
  vi.spyOn(console, "warn").mockImplementation(() => undefined);
  const { client, create } = injectedClient("anthropic", [
    { stop: "length" },
    { stop: "tool_use", input: { value: "complete" } },
  ]);
  const result = await callStructuredTool({
    llmClient: client,
    request,
    toolName: TOOL_NAME,
    parse: (input) => schema.parse(input),
  });
  expect(result.parsed).toEqual({ value: "complete" });
  expect(create).toHaveBeenCalledTimes(2);
  expect(create.mock.calls[1]![0]).toMatchObject({
    max_tokens: 16_000,
    messages: [
      ...request.messages,
      { role: "user", content: expect.stringContaining("discarded whole") },
    ],
  });
});

it.each([
  { steps: ["connection", "valid"], dispatches: 2 },
  { steps: ["connection", "schema", "valid"], dispatches: 3 },
  { steps: ["connection", "connection", "valid"], dispatches: 3 },
  { steps: ["refusal", "cut", "valid"], dispatches: 3 },
  { steps: ["schema", "connection", "valid"], dispatches: 3 },
  { steps: ["cut", "refusal", "valid"], dispatches: 3 },
] as const)(
  "preserves Anthropic recovery and HEAD dispatch count for $steps",
  async ({ steps, dispatches }) => {
    vi.spyOn(console, "warn").mockImplementation(() => undefined);
    const replies = steps.map((step): Reply | Error => {
      switch (step) {
        case "connection":
          return new APIConnectionError({ message: "Connection reset" });
        case "refusal":
          return { stop: "refusal" };
        case "cut":
          return { stop: "max_tokens" };
        case "schema":
          return { stop: "tool_use", input: {} };
        case "valid":
          return { stop: "tool_use", input: { value: "complete" } };
      }
    });
    const { client, create } = injectedClient("anthropic", replies);
    const result = await callStructuredTool({
      llmClient: client,
      request,
      toolName: TOOL_NAME,
      outputLimitRepairInstruction: REPAIR,
      parse: (input) => schema.parse(input),
    });
    expect(result.parsed).toEqual({ value: "complete" });
    expect(create).toHaveBeenCalledTimes(dispatches);
    const responseCount = steps.filter((step) => step !== "connection").length;
    expect(result.usage).toMatchObject({
      input_tokens: 12 * responseCount,
      output_tokens: 7 * responseCount,
    });
  },
);

it("preserves OAuth refresh then schema repair: three message dispatches, as at HEAD", async () => {
  vi.spyOn(console, "warn").mockImplementation(() => undefined);
  const tempDir = mkdtempSync(join(tmpdir(), "borg-logical-repair-"));
  tempDirs.push(tempDir);
  const credentialsPath = join(tempDir, "credentials.json");
  writeJsonFileAtomic(credentialsPath, {
    claudeAiOauth: {
      accessToken: "stale-access",
      refreshToken: "refresh-token",
      expiresAt: Date.now() + 3_600_000,
    },
  });
  const messages: Record<string, unknown>[] = [];
  let refreshes = 0;
  const fetchMock = vi.fn(
    async (input: Parameters<typeof fetch>[0], init?: Parameters<typeof fetch>[1]) => {
      if (new URL(String(input)).pathname === "/v1/oauth/token") {
        refreshes += 1;
        return Response.json({
          access_token: "fresh-access",
          refresh_token: "fresh-refresh",
          expires_in: 3600,
        });
      }
      messages.push(JSON.parse(String(init?.body)) as Record<string, unknown>);
      if (messages.length === 1)
        return Response.json({ error: { message: "unauthorized" } }, { status: 401 });
      return Response.json(
        anthropicReply({
          stop: "tool_use",
          input: messages.length === 2 ? {} : { value: "complete" },
        }),
      );
    },
  );
  vi.stubGlobal("fetch", fetchMock);
  const client = new AnthropicLLMClient({ env: { BORG_CLAUDE_CREDENTIALS_PATH: credentialsPath } });
  const result = await callStructuredTool({
    llmClient: client,
    request,
    toolName: TOOL_NAME,
    outputLimitRepairInstruction: REPAIR,
    parse: (input) => schema.parse(input),
  });
  expect(result.parsed).toEqual({ value: "complete" });
  expect(messages).toHaveLength(3);
  expect(refreshes).toBe(1);
  expect(fetchMock).toHaveBeenCalledTimes(4);
  expect(messages[1]!.messages).toEqual(messages[0]!.messages);
  expect(messages[2]!.messages).toEqual([
    ...request.messages,
    { role: "user", content: expect.stringContaining("failed client-side schema validation") },
  ]);
});

it.each(["connection", "429 with Retry-After"] as const)(
  "preserves OpenAI SDK recovery for %s: two dispatches, as at HEAD",
  async (failure) => {
    vi.spyOn(console, "warn").mockImplementation(() => undefined);
    let dispatches = 0;
    const fetchMock = vi.fn(async () => {
      dispatches += 1;
      if (dispatches === 1) {
        if (failure === "connection") throw new TypeError("Connection reset");
        return Response.json(
          { error: { message: "Rate limited" } },
          {
            status: 429,
            headers: { "Retry-After": "0.01" },
          },
        );
      }
      return Response.json({
        choices: [
          {
            message: {
              role: "assistant",
              content: null,
              tool_calls: [
                {
                  id: "call_1",
                  type: "function",
                  function: { name: TOOL_NAME, arguments: JSON.stringify({ value: "complete" }) },
                },
              ],
            },
            finish_reason: "tool_calls",
          },
        ],
        usage: { prompt_tokens: 12, completion_tokens: 7 },
      });
    });
    vi.stubGlobal("fetch", fetchMock);
    const client = new OpenAICompatibleLLMClient({
      apiKey: "test-key",
      baseUrl: "https://provider.invalid/v1",
    });
    const result = await callStructuredTool({
      llmClient: client,
      request,
      toolName: TOOL_NAME,
      outputLimitRepairInstruction: REPAIR,
      parse: (input) => schema.parse(input),
    });
    expect(result.parsed).toEqual({ value: "complete" });
    expect(fetchMock).toHaveBeenCalledTimes(2);
    expect(result.usage).toMatchObject({ input_tokens: 12, output_tokens: 7 });
  },
);
