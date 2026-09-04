import { mkdtempSync, rmSync } from "node:fs";
import { join } from "node:path";
import { tmpdir } from "node:os";

import { afterEach, describe, expect, it, vi } from "vitest";
import { z } from "zod";

import type { Message } from "@anthropic-ai/sdk/resources/messages/messages.js";

import { writeJsonFileAtomic } from "../util/atomic-write.js";
import { AuthError, LLMError } from "../util/errors.js";
import {
  AnthropicLLMClient,
  CLAUDE_CODE_IDENTITY_BLOCK_TEXT,
  LLMStructuredOutputParseError,
  createOAuthFetch,
  toStructuredOutputFormat,
  type TokenUsageEvent,
} from "./index.js";
import { FakeLLMClient } from "./test-support/fake-client.js";

function createTempCredentialsPath(tempDirs: string[]): string {
  const tempDir = mkdtempSync(join(tmpdir(), "borg-llm-"));
  tempDirs.push(tempDir);
  return join(tempDir, "credentials.json");
}

function createMessageBody(overrides: Partial<Message> = {}): Message {
  return {
    id: "msg_1",
    container: null,
    content: [
      {
        type: "text",
        text: "Hello",
        citations: null,
      },
    ],
    model: "claude-sonnet-4-5",
    role: "assistant",
    stop_details: null,
    stop_reason: "end_turn",
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
    ...overrides,
  } as unknown as Message;
}

function jsonResponse(body: unknown, status = 200): Response {
  return new Response(JSON.stringify(body), {
    status,
    headers: {
      "content-type": "application/json",
    },
  });
}

function delayedAbortAwareResponse(
  signal: AbortSignal | null | undefined,
  delayMs: number,
  createResponse: () => Response,
): Promise<Response> {
  return new Promise<Response>((resolve, reject) => {
    if (signal?.aborted === true) {
      reject(signal.reason);
      return;
    }

    const timer = setTimeout(() => {
      signal?.removeEventListener("abort", onAbort);
      resolve(createResponse());
    }, delayMs);
    const onAbort = () => {
      clearTimeout(timer);
      reject(signal?.reason);
    };

    signal?.addEventListener("abort", onAbort, { once: true });
  });
}

function createSseResponse(events: readonly string[]): Response {
  const encoder = new TextEncoder();
  const stream = new ReadableStream<Uint8Array>({
    start(controller) {
      controller.enqueue(encoder.encode(events.join("\n\n")));
      controller.close();
    },
  });

  return new Response(stream, {
    status: 200,
    headers: {
      "content-type": "text/event-stream",
    },
  });
}

function delay(ms: number): Promise<void> {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

function createChunkedSseResponse(
  chunks: readonly string[],
  options: { close?: boolean; delayMs?: number; onCancel?: (reason: unknown) => void } = {},
): Response {
  const encoder = new TextEncoder();
  let index = 0;

  const stream = new ReadableStream<Uint8Array>({
    async pull(controller) {
      if (index >= chunks.length) {
        if (options.close !== false) {
          controller.close();
          return;
        }

        await new Promise(() => undefined);
        return;
      }

      if (options.delayMs !== undefined) {
        await delay(options.delayMs);
      }

      controller.enqueue(encoder.encode(chunks[index] ?? ""));
      index += 1;
    },
    cancel(reason) {
      options.onCancel?.(reason);
    },
  });

  return new Response(stream, {
    status: 200,
    headers: {
      "content-type": "text/event-stream",
    },
  });
}

describe("llm", () => {
  const tempDirs: string[] = [];

  afterEach(() => {
    vi.useRealTimers();
    vi.restoreAllMocks();
    vi.unstubAllGlobals();

    while (tempDirs.length > 0) {
      rmSync(tempDirs.pop() as string, { recursive: true, force: true });
    }
  });

  it("wraps anthropic messages and extracts tool calls", async () => {
    const usageEvents: TokenUsageEvent[] = [];

    const message = createMessageBody({
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
      stop_reason: "tool_use",
    });

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

  it("clamps explicit Anthropic output requests to the model ceiling", async () => {
    const create = vi.fn().mockResolvedValue(createMessageBody());
    const client = new AnthropicLLMClient({
      client: {
        messages: { create },
      },
    });

    await client.complete({
      model: "claude-opus-4-6",
      messages: [{ role: "user", content: "hello" }],
      max_tokens: 80_000,
      budget: "test",
    });

    expect(create.mock.calls[0]?.[0]).toMatchObject({ max_tokens: 64_000 });
  });

  it.each(["oauth", "api-key"] as const)(
    "sets an explicit SDK client timeout for %s non-streaming max-token requests",
    async (authKind) => {
      let requestBody: Record<string, unknown> | undefined;
      const fetchMock = vi.fn(
        async (_input: Parameters<typeof fetch>[0], init?: Parameters<typeof fetch>[1]) => {
          requestBody = JSON.parse(String(init?.body)) as Record<string, unknown>;
          return jsonResponse(createMessageBody());
        },
      );
      vi.stubGlobal("fetch", fetchMock);
      const client = new AnthropicLLMClient(
        authKind === "oauth"
          ? {
              env: {
                ANTHROPIC_AUTH_TOKEN: "oauth-token",
              },
            }
          : {
              authMode: "api-key",
              apiKey: "api-key",
            },
      );

      await expect(
        client.complete({
          model: "claude-sonnet-4-5",
          messages: [{ role: "user", content: "hello" }],
          max_tokens: 64_000,
          budget: "test",
        }),
      ).resolves.toMatchObject({ text: "Hello" });

      expect(fetchMock).toHaveBeenCalledTimes(1);
      expect(requestBody).toMatchObject({ max_tokens: 64_000 });
    },
  );

  it("names the auth cause in the message rather than only in the dropped cause chain", async () => {
    // Callers record the top-level message and drop the cause chain, so an OAuth failure that
    // says only "Failed to complete Anthropic request" is indistinguishable from every other
    // request failure at the only surface that keeps it.
    vi.stubGlobal(
      "fetch",
      vi.fn(
        async () =>
          new Response(JSON.stringify({ error: { error: { message: "invalid x-api-key" } } }), {
            status: 401,
            headers: { "content-type": "application/json" },
          }),
      ),
    );
    const client = new AnthropicLLMClient({ env: { ANTHROPIC_AUTH_TOKEN: "oauth-token" } });

    const failure = await client
      .complete({
        model: "claude-sonnet-4-5",
        messages: [{ role: "user", content: "hello" }],
        budget: "test",
      })
      .then(
        () => null,
        (error: unknown) => error,
      );

    expect(failure).toBeInstanceOf(LLMError);
    expect((failure as LLMError).message).toMatch(
      /^Failed to complete Anthropic request: OAuth .+/,
    );
  });

  it("passes structured output config and extracts parsed JSON text", async () => {
    const outputConfig = {
      format: {
        type: "json_schema" as const,
        schema: {
          type: "object",
          properties: {
            ok: { type: "boolean" },
          },
          required: ["ok"],
          additionalProperties: false,
        },
      },
    };
    const create = vi.fn().mockResolvedValue(
      createMessageBody({
        content: [{ type: "text", text: '{"ok":true}', citations: null }],
        stop_reason: "end_turn",
      }),
    );
    const client = new AnthropicLLMClient({
      client: {
        messages: { create },
      },
    });

    const result = await client.complete({
      model: "claude-sonnet-4-5",
      messages: [{ role: "user", content: "return ok" }],
      output_config: outputConfig,
      max_tokens: 128,
      budget: "test",
    });

    expect(create.mock.calls[0]?.[0]).toMatchObject({
      output_config: outputConfig,
    });
    expect(result).toMatchObject({
      text: '{"ok":true}',
      structured_output: { ok: true },
      stop_reason: "end_turn",
    });
  });

  it("throws a typed structured-output parse error for non-JSON response text", async () => {
    const create = vi.fn().mockResolvedValue(
      createMessageBody({
        content: [{ type: "text", text: "I cannot comply.", citations: null }],
        stop_reason: "refusal",
      }),
    );
    const client = new AnthropicLLMClient({
      client: {
        messages: { create },
      },
    });

    const promise = client.complete({
      model: "claude-sonnet-4-5",
      messages: [{ role: "user", content: "return ok" }],
      output_config: {
        format: {
          type: "json_schema",
          schema: {
            type: "object",
            properties: { ok: { type: "boolean" } },
            required: ["ok"],
            additionalProperties: false,
          },
        },
      },
      max_tokens: 128,
      budget: "test",
    });

    await expect(promise).rejects.toBeInstanceOf(LLMStructuredOutputParseError);
    await expect(promise).rejects.toMatchObject({
      code: "LLM_STRUCTURED_OUTPUT_PARSE_FAILED",
      rawText: "I cannot comply.",
    });
  });

  it("preserves discriminator and literal constraints in structured-output schemas", () => {
    const format = toStructuredOutputFormat(
      z
        .object({
          discourse_act: z.enum(["answer", "no_output"]),
          claim: z.discriminatedUnion("kind", [
            z
              .object({
                kind: z.literal("user_fact"),
                confidence: z.enum(["direct", "inferred"]),
              })
              .strict(),
            z
              .object({
                kind: z.literal("interpretation"),
                persistence_allowed: z.literal(false),
              })
              .strict(),
          ]),
        })
        .strict(),
    );
    const serialized = JSON.stringify(format.schema);

    // Structured outputs must enforce value-level constraints at the API layer;
    // this guards against SDK/schema-conversion regressions that turn them into prose.
    expect(serialized).toContain('"enum":["answer","no_output"]');
    expect(serialized).toContain('"const":"user_fact"');
    expect(serialized).toContain('"const":"interpretation"');
    expect(serialized).toContain('"const":false');
  });

  it("keeps PascalCase tool names unchanged through the OAuth fetch wrapper", async () => {
    const fetchMock = vi.fn(
      async (input: Parameters<typeof fetch>[0], init?: Parameters<typeof fetch>[1]) => {
        const url = new URL(String(input));
        expect(url.pathname).toBe("/v1/messages");
        expect(url.searchParams.getAll("beta")).toEqual(["true"]);

        const body = JSON.parse(String(init?.body)) as {
          tools: Array<{ name: string }>;
          tool_choice: { name: string };
        };
        expect(body.tools[0]?.name).toBe("EmitEpisodeCandidates");
        expect(body.tool_choice.name).toBe("EmitEpisodeCandidates");

        return jsonResponse(
          createMessageBody({
            content: [
              {
                type: "tool_use",
                id: "toolu_1",
                caller: { type: "direct" },
                name: "EmitEpisodeCandidates",
                input: { id: 1 },
              },
            ],
            stop_reason: "tool_use",
          }),
        );
      },
    );
    vi.stubGlobal("fetch", fetchMock);

    const oauthFetch = createOAuthFetch();
    const response = await oauthFetch("https://api.anthropic.com/v1/messages", {
      method: "POST",
      body: JSON.stringify({
        tools: [{ name: "EmitEpisodeCandidates" }],
        tool_choice: { type: "tool", name: "EmitEpisodeCandidates" },
      }),
    });

    expect(((await response.json()) as Message).content[0]).toMatchObject({
      type: "tool_use",
      name: "EmitEpisodeCandidates",
    });
  });

  it("capitalizes lowercase OAuth tool names on request and restores them on JSON responses", async () => {
    const fetchMock = vi.fn(
      async (input: Parameters<typeof fetch>[0], init?: Parameters<typeof fetch>[1]) => {
        const url = new URL(String(input));
        expect(url.pathname).toBe("/v1/messages");
        expect(url.searchParams.getAll("beta")).toEqual(["true"]);

        const body = JSON.parse(String(init?.body)) as {
          tools: Array<{ name: string }>;
          tool_choice: { name: string };
        };
        expect(body.tools[0]?.name).toBe("Lookup");
        expect(body.tool_choice.name).toBe("Lookup");

        return jsonResponse(
          createMessageBody({
            content: [
              {
                type: "tool_use",
                id: "toolu_1",
                caller: { type: "direct" },
                name: "Lookup",
                input: { id: 1 },
              },
            ],
            stop_reason: "tool_use",
          }),
        );
      },
    );
    vi.stubGlobal("fetch", fetchMock);

    const oauthFetch = createOAuthFetch();
    const response = await oauthFetch("https://api.anthropic.com/v1/messages", {
      method: "POST",
      body: JSON.stringify({
        tools: [{ name: "lookup" }],
        tool_choice: { type: "tool", name: "lookup" },
      }),
    });

    expect(((await response.json()) as Message).content[0]).toMatchObject({
      type: "tool_use",
      name: "lookup",
    });
  });

  it("rewrites dotted OAuth tool names on request and restores them on JSON responses", async () => {
    const fetchMock = vi.fn(
      async (input: Parameters<typeof fetch>[0], init?: Parameters<typeof fetch>[1]) => {
        const url = new URL(String(input));
        expect(url.pathname).toBe("/v1/messages");
        expect(url.searchParams.getAll("beta")).toEqual(["true"]);

        const body = JSON.parse(String(init?.body)) as {
          tools: Array<{ name: string }>;
          tool_choice: { name: string };
        };
        expect(body.tools[0]?.name).toBe("Tool_episodic_search");
        expect(body.tool_choice.name).toBe("Tool_episodic_search");

        return jsonResponse(
          createMessageBody({
            content: [
              {
                type: "tool_use",
                id: "toolu_1",
                caller: { type: "direct" },
                name: "Tool_episodic_search",
                input: { query: "planning" },
              },
            ],
            stop_reason: "tool_use",
          }),
        );
      },
    );
    vi.stubGlobal("fetch", fetchMock);

    const oauthFetch = createOAuthFetch();
    const response = await oauthFetch("https://api.anthropic.com/v1/messages", {
      method: "POST",
      body: JSON.stringify({
        tools: [{ name: "tool.episodic.search" }],
        tool_choice: { type: "tool", name: "tool.episodic.search" },
      }),
    });

    expect(((await response.json()) as Message).content[0]).toMatchObject({
      type: "tool_use",
      name: "tool.episodic.search",
    });
  });

  it("rewrites dotted OAuth tool_use names in outbound message history", async () => {
    const fetchMock = vi.fn(
      async (_input: Parameters<typeof fetch>[0], init?: Parameters<typeof fetch>[1]) => {
        const body = JSON.parse(String(init?.body)) as {
          messages: Array<{
            role: string;
            content: Array<{ type: string; name?: string }>;
          }>;
        };

        expect(body.messages[1]?.content[0]).toMatchObject({
          type: "tool_use",
          name: "Tool_episodic_search",
        });

        return jsonResponse(createMessageBody());
      },
    );
    vi.stubGlobal("fetch", fetchMock);

    const oauthFetch = createOAuthFetch();
    await oauthFetch("https://api.anthropic.com/v1/messages", {
      method: "POST",
      body: JSON.stringify({
        tools: [{ name: "tool.episodic.search" }],
        messages: [
          {
            role: "user",
            content: [{ type: "text", text: "search memory" }],
          },
          {
            role: "assistant",
            content: [
              {
                type: "tool_use",
                id: "toolu_1",
                name: "tool.episodic.search",
                input: { query: "planning" },
              },
            ],
          },
        ],
      }),
    });
  });

  it("rewrites mixed OAuth tool batches per name instead of lowercasing everything", async () => {
    vi.stubGlobal(
      "fetch",
      vi.fn(async (_input: Parameters<typeof fetch>[0], init?: Parameters<typeof fetch>[1]) => {
        const body = JSON.parse(String(init?.body)) as {
          tools: Array<{ name: string }>;
          tool_choice: { name: string };
        };

        expect(body.tools.map((tool) => tool.name)).toEqual([
          "EmitEpisodeCandidates",
          "Lookup",
          "mcp__diagnostics",
        ]);
        expect(body.tool_choice.name).toBe("EmitEpisodeCandidates");

        return jsonResponse(
          createMessageBody({
            content: [
              {
                type: "tool_use",
                id: "toolu_1",
                caller: { type: "direct" },
                name: "EmitEpisodeCandidates",
                input: { episode: 1 },
              },
              {
                type: "tool_use",
                id: "toolu_2",
                caller: { type: "direct" },
                name: "Lookup",
                input: { id: 2 },
              },
              {
                type: "tool_use",
                id: "toolu_3",
                caller: { type: "direct" },
                name: "mcp__diagnostics",
                input: { id: 3 },
              },
            ],
            stop_reason: "tool_use",
          }),
        );
      }),
    );

    const oauthFetch = createOAuthFetch();
    const response = await oauthFetch("https://api.anthropic.com/v1/messages", {
      method: "POST",
      body: JSON.stringify({
        tools: [
          { name: "EmitEpisodeCandidates" },
          { name: "lookup" },
          { name: "mcp__diagnostics" },
        ],
        tool_choice: { type: "tool", name: "EmitEpisodeCandidates" },
      }),
    });

    const content = ((await response.json()) as Message).content;
    expect(content[0]).toMatchObject({ type: "tool_use", name: "EmitEpisodeCandidates" });
    expect(content[1]).toMatchObject({ type: "tool_use", name: "lookup" });
    expect(content[2]).toMatchObject({ type: "tool_use", name: "mcp__diagnostics" });
  });

  it("rewrites OAuth tool names inside SSE responses using the per-request transform map", async () => {
    vi.stubGlobal(
      "fetch",
      vi.fn(async () =>
        createSseResponse([
          'data: {"type":"content_block_start","content_block":{"type":"tool_use","id":"toolu_1","name":"EmitEpisodeCandidates","input":{"id":1}}}',
          'data: {"type":"content_block_start","content_block":{"type":"tool_use","id":"toolu_2","name":"Lookup","input":{"id":2}}}',
          "data: [DONE]",
        ]),
      ),
    );

    const oauthFetch = createOAuthFetch();
    const response = await oauthFetch("https://api.anthropic.com/v1/messages", {
      method: "POST",
      body: JSON.stringify({
        tools: [{ name: "EmitEpisodeCandidates" }, { name: "lookup" }],
        tool_choice: { type: "tool", name: "lookup" },
      }),
    });

    const text = await response.text();
    expect(text).toContain('"name":"EmitEpisodeCandidates"');
    expect(text).toContain('"name":"lookup"');
    expect(text).not.toContain('"name":"Lookup"');
  });

  it("fails OAuth SSE consumption when the byte stream stalls", async () => {
    vi.stubGlobal(
      "fetch",
      vi.fn(async () =>
        createChunkedSseResponse(
          [
            'data: {"type":"message_start","message":{"id":"msg_1"}}\n\n',
            "event: ping\ndata: {}\n\n",
          ],
          { close: false },
        ),
      ),
    );

    const oauthFetch = createOAuthFetch({ sseInactivityTimeoutMs: 20 });
    const response = await oauthFetch("https://api.anthropic.com/v1/messages", {
      method: "POST",
      body: JSON.stringify({ messages: [{ role: "user", content: "hello" }] }),
    });

    await expect(response.text()).rejects.toMatchObject({
      code: "LLM_STREAM_STALLED",
      message: "Anthropic SSE stream stalled after 20ms without a chunk",
    });
  });

  it("fails ping-only OAuth SSE streams at the first message-event bound", async () => {
    vi.useFakeTimers();
    const onCancel = vi.fn();
    vi.stubGlobal(
      "fetch",
      vi.fn(async () =>
        createChunkedSseResponse(
          Array.from({ length: 20 }, (_, index) =>
            index % 2 === 0 ? "event: ping\ndata: {}\n\n" : 'data: {"type":"ping"}\n\n',
          ),
          { close: false, delayMs: 5, onCancel },
        ),
      ),
    );

    const oauthFetch = createOAuthFetch({
      sseInactivityTimeoutMs: 100,
      sseFirstMessageEventTimeoutMs: 30,
      sseMessageEventGapTimeoutMs: 100,
    });
    const response = await oauthFetch("https://api.anthropic.com/v1/messages", {
      method: "POST",
      body: JSON.stringify({ messages: [{ role: "user", content: "hello" }] }),
    });

    const textResult = response.text().then(
      (value) => ({ status: "resolved" as const, value }),
      (reason: unknown) => ({ status: "rejected" as const, reason }),
    );

    await vi.advanceTimersByTimeAsync(35);
    const result = await textResult;

    expect(result.status).toBe("rejected");
    expect(result).toMatchObject({
      reason: {
        code: "LLM_STREAM_EVENT_STALLED",
        message: "Anthropic SSE stream stalled for 30ms before the first message event",
      },
    });
    expect(result).not.toMatchObject({
      reason: {
        code: "LLM_STREAM_STALLED",
      },
    });
    expect(onCancel).toHaveBeenCalledWith(
      expect.objectContaining({ code: "LLM_STREAM_EVENT_STALLED" }),
    );
  });

  it("allows OAuth SSE chunks that arrive before the inactivity deadline", async () => {
    vi.stubGlobal(
      "fetch",
      vi.fn(async () =>
        createChunkedSseResponse(
          [
            'data: {"type":"message_start","message":{"id":"msg_1"}}\n\n',
            "event: ping\ndata: {}\n\n",
            "data: [DONE]\n\n",
          ],
          { close: true, delayMs: 5 },
        ),
      ),
    );

    const oauthFetch = createOAuthFetch({ sseInactivityTimeoutMs: 50 });
    const response = await oauthFetch("https://api.anthropic.com/v1/messages", {
      method: "POST",
      body: JSON.stringify({ messages: [{ role: "user", content: "hello" }] }),
    });

    await expect(response.text()).resolves.toBe(
      'data: {"type":"message_start","message":{"id":"msg_1"}}\n\n' +
        "event: ping\ndata: {}\n\n" +
        "data: [DONE]\n\n",
    );
  });

  it("allows ping gaps between message events when they stay within event bounds", async () => {
    vi.useFakeTimers();
    vi.stubGlobal(
      "fetch",
      vi.fn(async () =>
        createChunkedSseResponse(
          [
            'data: {"type":"message_start","message":{"id":"msg_1"}}\n\n',
            "event: ping\ndata: {}\n\n",
            ": keep-alive\n\n",
            'data: {"type":"ping"}\n\n',
            'data: {"type":"content_block_start","content_block":{"type":"tool_use","id":"toolu_1","name":"Lookup","input":{}}}\n\n',
            "data: [DONE]\n\n",
          ],
          { close: true, delayMs: 5 },
        ),
      ),
    );

    const oauthFetch = createOAuthFetch({
      sseInactivityTimeoutMs: 100,
      sseFirstMessageEventTimeoutMs: 30,
      sseMessageEventGapTimeoutMs: 30,
    });
    const response = await oauthFetch("https://api.anthropic.com/v1/messages", {
      method: "POST",
      body: JSON.stringify({
        tools: [{ name: "lookup" }],
        tool_choice: { type: "tool", name: "lookup" },
      }),
    });

    const textPromise = response.text();
    await vi.advanceTimersByTimeAsync(40);

    await expect(textPromise).resolves.toBe(
      'data: {"type":"message_start","message":{"id":"msg_1"}}\n\n' +
        "event: ping\ndata: {}\n\n" +
        ": keep-alive\n\n" +
        'data: {"type":"ping"}\n\n' +
        'data: {"type":"content_block_start","content_block":{"type":"tool_use","id":"toolu_1","name":"lookup","input":{}}}\n\n' +
        "data: [DONE]\n\n",
    );
  });

  it("keeps the byte-silence watchdog responsible for fully silent SSE streams", async () => {
    vi.useFakeTimers();
    vi.stubGlobal(
      "fetch",
      vi.fn(async () => createChunkedSseResponse([], { close: false })),
    );

    const oauthFetch = createOAuthFetch({
      sseInactivityTimeoutMs: 20,
      sseFirstMessageEventTimeoutMs: 100,
      sseMessageEventGapTimeoutMs: 100,
    });
    const response = await oauthFetch("https://api.anthropic.com/v1/messages", {
      method: "POST",
      body: JSON.stringify({ messages: [{ role: "user", content: "hello" }] }),
    });

    const textResult = response.text().then(
      (value) => ({ status: "resolved" as const, value }),
      (reason: unknown) => ({ status: "rejected" as const, reason }),
    );

    await vi.advanceTimersByTimeAsync(25);
    const result = await textResult;

    expect(result).toMatchObject({
      status: "rejected",
      reason: {
        code: "LLM_STREAM_STALLED",
        message: "Anthropic SSE stream stalled after 20ms without a chunk",
      },
    });
  });

  it("fails when the gap between SSE message events exceeds the event bound", async () => {
    vi.useFakeTimers();
    vi.stubGlobal(
      "fetch",
      vi.fn(async () =>
        createChunkedSseResponse(
          [
            'data: {"type":"message_start","message":{"id":"msg_1"}}\n\n',
            "event: ping\ndata: {}\n\n",
            'data: {"type":"ping"}\n\n',
            ": still waiting\n\n",
            "event: ping\ndata: {}\n\n",
          ],
          { close: false, delayMs: 5 },
        ),
      ),
    );

    const oauthFetch = createOAuthFetch({
      sseInactivityTimeoutMs: 100,
      sseFirstMessageEventTimeoutMs: 100,
      sseMessageEventGapTimeoutMs: 17,
    });
    const response = await oauthFetch("https://api.anthropic.com/v1/messages", {
      method: "POST",
      body: JSON.stringify({ messages: [{ role: "user", content: "hello" }] }),
    });

    const textResult = response.text().then(
      (value) => ({ status: "resolved" as const, value }),
      (reason: unknown) => ({ status: "rejected" as const, reason }),
    );

    await vi.advanceTimersByTimeAsync(30);
    const result = await textResult;

    expect(result).toMatchObject({
      status: "rejected",
      reason: {
        code: "LLM_STREAM_EVENT_STALLED",
        message: "Anthropic SSE stream stalled for 17ms between message events",
      },
    });
  });

  it("clears OAuth SSE inactivity timers after normal completion", async () => {
    vi.useFakeTimers();
    vi.stubGlobal(
      "fetch",
      vi.fn(async () =>
        createChunkedSseResponse([
          'data: {"type":"message_start","message":{"id":"msg_1"}}\n\n',
          "data: [DONE]\n\n",
        ]),
      ),
    );

    const oauthFetch = createOAuthFetch({ sseInactivityTimeoutMs: 20 });
    const response = await oauthFetch("https://api.anthropic.com/v1/messages", {
      method: "POST",
      body: JSON.stringify({ messages: [{ role: "user", content: "hello" }] }),
    });

    await expect(response.text()).resolves.toBe(
      'data: {"type":"message_start","message":{"id":"msg_1"}}\n\n' + "data: [DONE]\n\n",
    );
    expect(vi.getTimerCount()).toBe(0);

    await vi.advanceTimersByTimeAsync(25);
    expect(vi.getTimerCount()).toBe(0);
  });

  it("fails typed when an OAuth unary JSON body read stalls", async () => {
    vi.useFakeTimers();
    vi.stubGlobal(
      "fetch",
      vi.fn(async () => {
        const stream = new ReadableStream<Uint8Array>({
          async pull() {
            await new Promise(() => undefined);
          },
        });

        return new Response(stream, {
          status: 200,
          headers: {
            "content-type": "application/json",
          },
        });
      }),
    );

    const oauthFetch = createOAuthFetch({ unaryBodyTimeoutMs: 20 });
    const resultPromise = oauthFetch("https://api.anthropic.com/v1/messages", {
      method: "POST",
      body: JSON.stringify({ messages: [{ role: "user", content: "hello" }] }),
    }).then(
      (value) => ({ status: "resolved" as const, value }),
      (reason: unknown) => ({ status: "rejected" as const, reason }),
    );

    await vi.advanceTimersByTimeAsync(25);
    const result = await resultPromise;

    expect(result).toMatchObject({
      status: "rejected",
      reason: {
        code: "LLM_CALL_TIMED_OUT",
        message: "Anthropic body-read LLM deadline exceeded after 20ms",
      },
    });
  });

  it("aborts OAuth fetches when response headers never arrive", async () => {
    vi.useFakeTimers();
    let observedSignal: AbortSignal | undefined;
    vi.stubGlobal(
      "fetch",
      vi.fn(
        async (_input: Parameters<typeof fetch>[0], init?: Parameters<typeof fetch>[1]) =>
          await new Promise<Response>((_resolve, reject) => {
            observedSignal = init?.signal ?? undefined;
            init?.signal?.addEventListener("abort", () => reject(init.signal?.reason), {
              once: true,
            });
          }),
      ),
    );

    const oauthFetch = createOAuthFetch({ fetchHeadersTimeoutMs: 20 });
    const resultPromise = oauthFetch("https://api.anthropic.com/v1/messages", {
      method: "POST",
      body: JSON.stringify({ messages: [{ role: "user", content: "hello" }] }),
    }).then(
      (value) => ({ status: "resolved" as const, value }),
      (reason: unknown) => ({ status: "rejected" as const, reason }),
    );

    await vi.advanceTimersByTimeAsync(25);
    const result = await resultPromise;

    expect(observedSignal?.aborted).toBe(true);
    expect(result).toMatchObject({
      status: "rejected",
      reason: {
        code: "LLM_CALL_TIMED_OUT",
        message: "Anthropic headers LLM deadline exceeded after 20ms",
      },
    });
  });

  it("does not let the OAuth headers timeout abort a normal slow streaming body", async () => {
    vi.useFakeTimers();
    let observedSignal: AbortSignal | undefined;
    vi.stubGlobal(
      "fetch",
      vi.fn(async (_input: Parameters<typeof fetch>[0], init?: Parameters<typeof fetch>[1]) => {
        observedSignal = init?.signal ?? undefined;
        return createChunkedSseResponse(
          ['data: {"type":"message_start","message":{"id":"msg_1"}}\n\n', "data: [DONE]\n\n"],
          { close: true, delayMs: 50 },
        );
      }),
    );

    const oauthFetch = createOAuthFetch({
      fetchHeadersTimeoutMs: 20,
      sseInactivityTimeoutMs: 100,
      sseFirstMessageEventTimeoutMs: 100,
      sseMessageEventGapTimeoutMs: 100,
    });
    const response = await oauthFetch("https://api.anthropic.com/v1/messages", {
      method: "POST",
      body: JSON.stringify({ messages: [{ role: "user", content: "hello" }] }),
    });

    const textPromise = response.text();
    await vi.advanceTimersByTimeAsync(120);

    await expect(textPromise).resolves.toBe(
      'data: {"type":"message_start","message":{"id":"msg_1"}}\n\n' + "data: [DONE]\n\n",
    );
    expect(observedSignal?.aborted).toBe(false);
  });

  it("lets buffered OAuth complete responses outlive the generic header guard", async () => {
    vi.useFakeTimers();
    let observedSignal: AbortSignal | undefined;
    const fetchMock = vi.fn(
      async (_input: Parameters<typeof fetch>[0], init?: Parameters<typeof fetch>[1]) => {
        observedSignal = init?.signal ?? undefined;
        return await delayedAbortAwareResponse(observedSignal, 50, () =>
          jsonResponse(createMessageBody()),
        );
      },
    );
    vi.stubGlobal("fetch", fetchMock);

    const client = new AnthropicLLMClient({
      env: {
        ANTHROPIC_AUTH_TOKEN: "oauth-token",
      },
      oauthFetchHeadersTimeoutMs: 20,
      unaryCallTimeoutMs: 20,
      transportStallMaxRetries: 0,
    });
    let settled = false;
    const resultPromise = client.complete({
      model: "claude-sonnet-4-5",
      messages: [{ role: "user", content: "hello" }],
      max_tokens: 32,
      timeoutMs: 100,
      budget: "test",
    });
    void resultPromise.then(
      () => {
        settled = true;
      },
      () => {
        settled = true;
      },
    );

    await vi.advanceTimersByTimeAsync(25);
    expect(settled).toBe(false);
    expect(observedSignal?.aborted).toBe(false);

    await vi.advanceTimersByTimeAsync(30);

    expect(fetchMock).toHaveBeenCalledTimes(1);
    await expect(resultPromise).resolves.toMatchObject({ text: "Hello" });
  });

  it("aligns buffered OAuth converse headers with the unary call deadline", async () => {
    vi.useFakeTimers();
    let observedSignal: AbortSignal | undefined;
    const fetchMock = vi.fn(
      async (_input: Parameters<typeof fetch>[0], init?: Parameters<typeof fetch>[1]) => {
        observedSignal = init?.signal ?? undefined;
        return await delayedAbortAwareResponse(observedSignal, 50, () =>
          jsonResponse(createMessageBody()),
        );
      },
    );
    vi.stubGlobal("fetch", fetchMock);

    const client = new AnthropicLLMClient({
      env: {
        ANTHROPIC_AUTH_TOKEN: "oauth-token",
      },
      oauthFetchHeadersTimeoutMs: 20,
      unaryCallTimeoutMs: 100,
      transportStallMaxRetries: 0,
    });
    const resultPromise = client.converse({
      model: "claude-sonnet-4-5",
      messages: [{ role: "user", content: [{ type: "text", text: "hello" }] }],
      max_tokens: 32,
      budget: "test",
    });

    await vi.advanceTimersByTimeAsync(25);
    expect(observedSignal?.aborted).toBe(false);

    await vi.advanceTimersByTimeAsync(30);

    expect(fetchMock).toHaveBeenCalledTimes(1);
    await expect(resultPromise).resolves.toMatchObject({
      messageBlocks: [{ type: "text", text: "Hello" }],
    });
  });

  it("cancels wedged OAuth unary bodies and surfaces typed through complete", async () => {
    vi.useFakeTimers();
    let cancelReason: unknown;
    const fetchMock = vi.fn(async () => {
      const stream = new ReadableStream<Uint8Array>({
        cancel(reason) {
          cancelReason = reason;
        },
      });

      return new Response(stream, {
        status: 200,
        headers: {
          "content-type": "application/json",
        },
      });
    });
    vi.stubGlobal("fetch", fetchMock);

    const client = new AnthropicLLMClient({
      env: {
        ANTHROPIC_AUTH_TOKEN: "oauth-token",
      },
      oauthUnaryBodyTimeoutMs: 20,
      unaryCallTimeoutMs: 1_000,
      transportStallMaxRetries: 0,
    });
    const resultPromise = client
      .complete({
        model: "claude-sonnet-4-5",
        messages: [{ role: "user", content: "hello" }],
        max_tokens: 32,
        budget: "test",
      })
      .then(
        (value) => ({ status: "resolved" as const, value }),
        (reason: unknown) => ({ status: "rejected" as const, reason }),
      );

    await vi.advanceTimersByTimeAsync(25);
    const result = await resultPromise;

    expect(fetchMock).toHaveBeenCalledTimes(1);
    expect(cancelReason).toMatchObject({
      code: "LLM_CALL_TIMED_OUT",
      message: "Anthropic body-read LLM deadline exceeded after 20ms",
    });
    expect(result).toMatchObject({
      status: "rejected",
      reason: {
        code: "LLM_CALL_TIMED_OUT",
        message: "Anthropic body-read LLM deadline exceeded after 20ms",
      },
    });
  });

  it("surfaces OAuth header deadlines through streaming without SDK retries", async () => {
    vi.useFakeTimers();
    const fetchMock = vi.fn(
      async (_input: Parameters<typeof fetch>[0], init?: Parameters<typeof fetch>[1]) =>
        await new Promise<Response>((_resolve, reject) => {
          init?.signal?.addEventListener("abort", () => reject(init.signal?.reason), {
            once: true,
          });
        }),
    );
    vi.stubGlobal("fetch", fetchMock);

    const client = new AnthropicLLMClient({
      env: {
        ANTHROPIC_AUTH_TOKEN: "oauth-token",
      },
      oauthFetchHeadersTimeoutMs: 20,
      streamingCallTimeoutMs: 1_000,
      transportStallMaxRetries: 0,
    });
    const resultPromise = client
      .streamComplete({
        model: "claude-sonnet-4-5",
        messages: [{ role: "user", content: "hello" }],
        max_tokens: 32,
        budget: "test",
      })
      .then(
        (value) => ({ status: "resolved" as const, value }),
        (reason: unknown) => ({ status: "rejected" as const, reason }),
      );

    await vi.advanceTimersByTimeAsync(25);
    const result = await resultPromise;

    expect(fetchMock).toHaveBeenCalledTimes(1);
    expect(result).toMatchObject({
      status: "rejected",
      reason: {
        code: "LLM_CALL_TIMED_OUT",
        message: "Anthropic headers LLM deadline exceeded after 20ms",
      },
    });
  });

  it("retries plain Anthropic connection failures with Borg-owned bounds", async () => {
    const fetchMock = vi.fn(async () => {
      throw new TypeError("fetch failed");
    });
    vi.stubGlobal("fetch", fetchMock);

    const retryEvents: unknown[] = [];
    const client = new AnthropicLLMClient({
      env: {
        ANTHROPIC_AUTH_TOKEN: "oauth-token",
      },
    });

    await expect(
      client.complete({
        model: "claude-sonnet-4-5",
        messages: [{ role: "user", content: "hello" }],
        max_tokens: 32,
        budget: "test",
        onTransportRetry: (event) => retryEvents.push(event),
      }),
    ).rejects.toMatchObject({
      code: "LLM_CONNECTION_FAILED",
      message: "Anthropic connection failed after 3 attempts",
    });
    expect(fetchMock).toHaveBeenCalledTimes(3);
    expect(retryEvents).toEqual([
      { attempt: 2, kind: "connection", retry_transport: "unary" },
      { attempt: 3, kind: "connection", retry_transport: "unary" },
    ]);
  });

  it("retries a unary call once when it dies with a stall-class error", async () => {
    const create = vi
      .fn()
      .mockRejectedValueOnce(
        new LLMError("Anthropic SSE stream stalled for 180000ms between message events", {
          code: "LLM_STREAM_EVENT_STALLED",
        }),
      )
      .mockResolvedValueOnce(createMessageBody());
    const client = new AnthropicLLMClient({
      client: {
        messages: {
          create,
          stream: vi.fn(),
        },
      },
    });

    await expect(
      client.complete({
        model: "claude-sonnet-4-5",
        messages: [{ role: "user", content: "hello" }],
        max_tokens: 32,
        budget: "test",
      }),
    ).resolves.toMatchObject({ text: "Hello" });
    expect(create).toHaveBeenCalledTimes(2);
  });

  it("exhausts stall retries and surfaces the typed stall error", async () => {
    const create = vi.fn(async () => {
      throw new LLMError("Anthropic SSE stream stalled for 180000ms between message events", {
        code: "LLM_STREAM_EVENT_STALLED",
      });
    });
    const client = new AnthropicLLMClient({
      client: {
        messages: {
          create,
          stream: vi.fn(),
        },
      },
    });

    await expect(
      client.complete({
        model: "claude-sonnet-4-5",
        messages: [{ role: "user", content: "hello" }],
        max_tokens: 32,
        budget: "test",
      }),
    ).rejects.toMatchObject({
      code: "LLM_STREAM_EVENT_STALLED",
    });
    expect(create).toHaveBeenCalledTimes(2);
  });

  it("never retries LLM_CONNECTION_FAILED even when its cause chain holds a stall code", async () => {
    const create = vi.fn(async () => {
      throw new LLMError("Anthropic connection failed after 3 attempts", {
        code: "LLM_CONNECTION_FAILED",
        cause: new LLMError("Anthropic SSE stream stalled for 20ms between message events", {
          code: "LLM_STREAM_EVENT_STALLED",
        }),
      });
    });
    const client = new AnthropicLLMClient({
      client: {
        messages: {
          create,
          stream: vi.fn(),
        },
      },
    });

    await expect(
      client.complete({
        model: "claude-sonnet-4-5",
        messages: [{ role: "user", content: "hello" }],
        max_tokens: 32,
        budget: "test",
      }),
    ).rejects.toMatchObject({
      code: "LLM_CONNECTION_FAILED",
    });
    expect(create).toHaveBeenCalledTimes(1);
  });

  it("honors transportStallMaxRetries: 0 with a single attempt", async () => {
    const create = vi.fn(async () => {
      throw new LLMError("Anthropic SSE stream stalled for 180000ms between message events", {
        code: "LLM_STREAM_EVENT_STALLED",
      });
    });
    const client = new AnthropicLLMClient({
      client: {
        messages: {
          create,
          stream: vi.fn(),
        },
      },
      transportStallMaxRetries: 0,
    });

    await expect(
      client.complete({
        model: "claude-sonnet-4-5",
        messages: [{ role: "user", content: "hello" }],
        max_tokens: 32,
        budget: "test",
      }),
    ).rejects.toMatchObject({
      code: "LLM_STREAM_EVENT_STALLED",
    });
    expect(create).toHaveBeenCalledTimes(1);
  });

  it("retries a stalled stream non-streaming without duplicating forwarded deltas", async () => {
    const stalledStream = {
      async *[Symbol.asyncIterator]() {
        yield {
          type: "content_block_delta",
          index: 0,
          delta: { type: "text_delta", text: "Hel" },
        };
        throw new LLMError("Anthropic SSE stream stalled for 20ms between message events", {
          code: "LLM_STREAM_EVENT_STALLED",
        });
      },
      finalMessage: vi.fn(),
    };
    const streamFactory = vi.fn().mockReturnValueOnce(stalledStream as never);
    const create = vi.fn(async () => createMessageBody());
    const deltas: string[] = [];
    const retryEvents: unknown[] = [];
    const client = new AnthropicLLMClient({
      client: {
        messages: {
          create,
          stream: streamFactory,
        },
      },
    });

    await expect(
      client.streamComplete({
        model: "claude-sonnet-4-5",
        messages: [{ role: "user", content: "hello" }],
        max_tokens: 32,
        budget: "test",
        onTextDelta: (text) => deltas.push(text),
        onTransportRetry: (event) => retryEvents.push(event),
      }),
    ).resolves.toMatchObject({ text: "Hello" });

    // The retry goes non-streaming: one stream attempt, one unary attempt, and
    // the stalled attempt's forwarded prefix is never re-emitted.
    expect(streamFactory).toHaveBeenCalledTimes(1);
    expect(create).toHaveBeenCalledTimes(1);
    expect(deltas.join("")).toBe("Hel");
    expect(retryEvents).toEqual([
      {
        attempt: 2,
        kind: "stall",
        code: "LLM_STREAM_EVENT_STALLED",
        retry_transport: "unary",
      },
    ]);
  });

  it("retries a stalled OAuth SSE stream non-streaming through the real fetch path", async () => {
    vi.useFakeTimers();
    const requestBodies: string[] = [];
    const requestSignals: Array<AbortSignal | undefined> = [];
    const fetchMock = vi.fn(
      async (_input: Parameters<typeof fetch>[0], init?: Parameters<typeof fetch>[1]) => {
        requestBodies.push(String(init?.body));
        requestSignals.push(init?.signal ?? undefined);

        if (requestBodies.length === 1) {
          return createChunkedSseResponse(
            [
              `event: message_start\ndata: ${JSON.stringify({
                type: "message_start",
                message: createMessageBody({
                  content: [],
                  stop_reason: null,
                  usage: {
                    cache_creation: null,
                    cache_creation_input_tokens: null,
                    cache_read_input_tokens: null,
                    input_tokens: 12,
                    output_tokens: 0,
                    server_tool_use: null,
                  },
                } as unknown as Partial<Message>),
              })}\n\n`,
              "event: ping\ndata: {}\n\n",
            ],
            { close: false },
          );
        }

        return await delayedAbortAwareResponse(init?.signal, 50, () =>
          jsonResponse(createMessageBody()),
        );
      },
    );
    vi.stubGlobal("fetch", fetchMock);

    const client = new AnthropicLLMClient({
      env: {
        ANTHROPIC_AUTH_TOKEN: "oauth-token",
      },
      oauthSseInactivityTimeoutMs: 10,
      oauthFetchHeadersTimeoutMs: 20,
      streamingCallTimeoutMs: 200,
    });

    const resultPromise = client.streamComplete({
      model: "claude-sonnet-4-5",
      messages: [{ role: "user", content: "hello" }],
      max_tokens: 32,
      budget: "test",
    });

    await vi.advanceTimersByTimeAsync(15);
    expect(fetchMock).toHaveBeenCalledTimes(2);

    await vi.advanceTimersByTimeAsync(25);
    expect(requestSignals[1]?.aborted).toBe(false);

    await vi.advanceTimersByTimeAsync(30);
    await expect(resultPromise).resolves.toMatchObject({ text: "Hello" });

    expect(fetchMock).toHaveBeenCalledTimes(2);
    expect(requestBodies[0]).toContain('"stream":true');
    expect(requestBodies[1]).not.toContain('"stream":true');
  });

  it("does not burn stall retries after the outer call deadline fires", async () => {
    vi.useFakeTimers();
    const fetchMock = vi.fn(
      async (_input: Parameters<typeof fetch>[0], init?: Parameters<typeof fetch>[1]) =>
        await new Promise<Response>((_resolve, reject) => {
          init?.signal?.addEventListener("abort", () => reject(init.signal?.reason), {
            once: true,
          });
        }),
    );
    vi.stubGlobal("fetch", fetchMock);

    const client = new AnthropicLLMClient({
      env: {
        ANTHROPIC_AUTH_TOKEN: "oauth-token",
      },
      oauthFetchHeadersTimeoutMs: 10_000,
      unaryCallTimeoutMs: 20,
    });
    const resultPromise = client
      .complete({
        model: "claude-sonnet-4-5",
        messages: [{ role: "user", content: "hello" }],
        max_tokens: 32,
        budget: "test",
      })
      .then(
        (value) => ({ status: "resolved" as const, value }),
        (reason: unknown) => ({ status: "rejected" as const, reason }),
      );

    await vi.advanceTimersByTimeAsync(25);
    const result = await resultPromise;

    expect(fetchMock).toHaveBeenCalledTimes(1);
    expect(result).toMatchObject({
      status: "rejected",
      reason: {
        code: "LLM_CALL_TIMED_OUT",
        message: "Anthropic unary LLM call timed out after 20ms",
      },
    });
  });

  it("surfaces OAuth SSE stalls at top level through the real SDK stream path", async () => {
    vi.stubGlobal(
      "fetch",
      vi.fn(async () =>
        createChunkedSseResponse(
          [
            `event: message_start\ndata: ${JSON.stringify({
              type: "message_start",
              message: createMessageBody({
                content: [],
                stop_reason: null,
                usage: {
                  cache_creation: null,
                  cache_creation_input_tokens: null,
                  cache_read_input_tokens: null,
                  input_tokens: 12,
                  output_tokens: 0,
                  server_tool_use: null,
                },
              } as unknown as Partial<Message>),
            })}\n\n`,
            "event: ping\ndata: {}\n\n",
          ],
          { close: false },
        ),
      ),
    );
    const client = new AnthropicLLMClient({
      env: {
        ANTHROPIC_AUTH_TOKEN: "oauth-token",
      },
      oauthSseInactivityTimeoutMs: 20,
    });

    let caught: unknown;
    try {
      await client.streamComplete({
        model: "claude-sonnet-4-5",
        messages: [{ role: "user", content: "hello" }],
        max_tokens: 32,
        budget: "test",
      });
    } catch (error) {
      caught = error;
    }

    expect(caught).toMatchObject({
      code: "LLM_STREAM_STALLED",
      message: "Anthropic SSE stream stalled after 20ms without a chunk",
    });
    expect(caught).toHaveProperty("cause");
  });

  it("preserves Request method, headers, and body in the OAuth fetch wrapper", async () => {
    const requestBody = JSON.stringify({
      messages: [{ role: "user", content: "hello" }],
    });
    const fetchMock = vi.fn(
      async (input: Parameters<typeof fetch>[0], init?: Parameters<typeof fetch>[1]) => {
        const url = new URL(String(input));
        const headers = new Headers(init?.headers);

        expect(url.pathname).toBe("/v1/messages");
        expect(url.searchParams.get("beta")).toBe("true");
        expect(init?.method).toBe("POST");
        expect(headers.get("content-type")).toBe("application/json");
        expect(headers.get("x-borg-test")).toBe("preserve-me");
        await expect(new Response(init?.body ?? null).text()).resolves.toBe(requestBody);

        return jsonResponse(createMessageBody());
      },
    );
    vi.stubGlobal("fetch", fetchMock);

    const oauthFetch = createOAuthFetch();
    await expect(
      oauthFetch(
        new Request("https://api.anthropic.com/v1/messages", {
          method: "POST",
          headers: {
            "content-type": "application/json",
            "x-borg-test": "preserve-me",
          },
          body: requestBody,
        }),
      ),
    ).resolves.toBeInstanceOf(Response);
  });

  it("prefers API key auth when available", async () => {
    const credentialsPath = createTempCredentialsPath(tempDirs);
    writeJsonFileAtomic(credentialsPath, {
      claudeAiOauth: {
        accessToken: "oauth-access",
        refreshToken: "oauth-refresh",
        expiresAt: Date.now() + 3_600_000,
      },
    });

    const fetchMock = vi.fn(
      async (_input: Parameters<typeof fetch>[0], init?: Parameters<typeof fetch>[1]) => {
        const headers = new Headers(init?.headers);
        const body = JSON.parse(String(init?.body)) as { system: string };

        expect(headers.get("x-api-key")).toBe("sk-test");
        expect(body.system).toBe("be concise");

        return jsonResponse(createMessageBody());
      },
    );
    vi.stubGlobal("fetch", fetchMock);

    const client = new AnthropicLLMClient({
      env: {
        ANTHROPIC_API_KEY: "sk-test",
        BORG_CLAUDE_CREDENTIALS_PATH: credentialsPath,
      },
    });

    await expect(
      client.complete({
        model: "claude-sonnet-4-5",
        system: "be concise",
        messages: [{ role: "user", content: "hello" }],
        max_tokens: 32,
        budget: "test",
      }),
    ).resolves.toMatchObject({
      text: "Hello",
    });

    const url = new URL(String(fetchMock.mock.calls[0]?.[0]));
    expect(url.searchParams.has("beta")).toBe(false);
  });

  it("builds an OAuth client from env auth token and prepends the identity block", async () => {
    const fetchMock = vi.fn(
      async (input: Parameters<typeof fetch>[0], init?: Parameters<typeof fetch>[1]) => {
        const url = new URL(String(input));
        const headers = new Headers(init?.headers);
        const body = JSON.parse(String(init?.body)) as {
          system: Array<{ type: string; text: string }>;
          tools: Array<{ name: string }>;
        };

        expect(url.searchParams.get("beta")).toBe("true");
        expect(headers.get("anthropic-beta")).toContain("claude-code-20250219");
        expect(headers.get("user-agent")).toContain("claude-cli/2.1.2");
        expect(body.system[0]?.text).toBe(CLAUDE_CODE_IDENTITY_BLOCK_TEXT);
        expect(body.system[1]?.text).toBe("be concise");
        expect(body.tools[0]?.name).toBe("Lookup");

        return jsonResponse(
          createMessageBody({
            content: [
              { type: "text", text: "Hello", citations: null },
              {
                type: "tool_use",
                id: "toolu_1",
                caller: { type: "direct" },
                name: "Lookup",
                input: { id: 1 },
              },
            ],
            stop_reason: "tool_use",
          }),
        );
      },
    );
    vi.stubGlobal("fetch", fetchMock);

    const client = new AnthropicLLMClient({
      env: {
        ANTHROPIC_AUTH_TOKEN: "oauth-token",
      },
    });

    await expect(
      client.complete({
        model: "claude-sonnet-4-5",
        system: "be concise",
        messages: [{ role: "user", content: "hello" }],
        tools: [
          {
            name: "lookup",
            inputSchema: {
              type: "object",
            },
          },
        ],
        max_tokens: 32,
        budget: "test",
      }),
    ).resolves.toMatchObject({
      tool_calls: [
        {
          name: "lookup",
        },
      ],
    });
  });

  it("prepends the OAuth identity block without flattening string or block-array system input", async () => {
    const systems: Array<Array<{ type: string; text: string }>> = [];
    const fetchMock = vi.fn(
      async (_input: Parameters<typeof fetch>[0], init?: Parameters<typeof fetch>[1]) => {
        const body = JSON.parse(String(init?.body)) as {
          system: Array<{ type: string; text: string }>;
        };

        systems.push(body.system);
        return jsonResponse(createMessageBody());
      },
    );
    vi.stubGlobal("fetch", fetchMock);

    const client = new AnthropicLLMClient({
      env: {
        ANTHROPIC_AUTH_TOKEN: "oauth-token",
      },
    });

    await client.complete({
      model: "claude-sonnet-4-5",
      system: "be concise",
      messages: [{ role: "user", content: "hello" }],
      max_tokens: 32,
      budget: "test",
    });

    await client.complete({
      model: "claude-sonnet-4-5",
      system: [
        { type: "text", text: "be concise" },
        { type: "text", text: "cite sources" },
      ],
      messages: [{ role: "user", content: "hello again" }],
      max_tokens: 32,
      budget: "test",
    });

    expect(systems).toEqual([
      [
        { type: "text", text: CLAUDE_CODE_IDENTITY_BLOCK_TEXT },
        { type: "text", text: "be concise" },
      ],
      [
        { type: "text", text: CLAUDE_CODE_IDENTITY_BLOCK_TEXT },
        { type: "text", text: "be concise" },
        { type: "text", text: "cite sources" },
      ],
    ]);
  });

  it("omits temperature and thinking for Opus requests in OAuth mode", async () => {
    const fetchMock = vi.fn(
      async (_input: Parameters<typeof fetch>[0], init?: Parameters<typeof fetch>[1]) => {
        const body = JSON.parse(String(init?.body)) as Record<string, unknown>;

        expect(body.temperature).toBeUndefined();
        expect(body.thinking).toBeUndefined();

        return jsonResponse(createMessageBody());
      },
    );
    vi.stubGlobal("fetch", fetchMock);

    const client = new AnthropicLLMClient({
      env: {
        ANTHROPIC_AUTH_TOKEN: "oauth-token",
      },
    });

    await expect(
      client.complete({
        model: "claude-opus-4-6",
        system: "be concise",
        messages: [{ role: "user", content: "hello" }],
        tools: [
          {
            name: "EmitEpisodeCandidates",
            inputSchema: {
              type: "object",
            },
          },
        ],
        tool_choice: { type: "tool", name: "EmitEpisodeCandidates" },
        temperature: 0,
        thinking: { type: "disabled" },
        max_tokens: 32,
        budget: "test",
      }),
    ).resolves.toMatchObject({
      text: "Hello",
    });
  });

  it("treats a newer Opus generation as Opus for temperature and manual thinking", async () => {
    // Regression guard: the Opus family gate must not be pinned to a version
    // digit. Opus 5 rejects both `temperature` and manual budget_tokens
    // thinking, so a gate that only matched claude-opus-4* would 400 here.
    const fetchMock = vi.fn(
      async (_input: Parameters<typeof fetch>[0], init?: Parameters<typeof fetch>[1]) => {
        const body = JSON.parse(String(init?.body)) as Record<string, unknown>;

        expect(body.temperature).toBeUndefined();
        expect(body.thinking).toBeUndefined();

        return jsonResponse(createMessageBody());
      },
    );
    vi.stubGlobal("fetch", fetchMock);

    const client = new AnthropicLLMClient({
      env: {
        ANTHROPIC_AUTH_TOKEN: "oauth-token",
      },
    });

    await expect(
      client.complete({
        model: "claude-opus-5",
        system: "be concise",
        messages: [{ role: "user", content: "hello" }],
        temperature: 0,
        thinking: { type: "enabled", budget_tokens: 4_000 },
        max_tokens: 32,
        budget: "test",
      }),
    ).resolves.toMatchObject({
      text: "Hello",
    });
  });

  it("sends adaptive thinking and effort for Opus with auto tool_choice", async () => {
    const fetchMock = vi.fn(
      async (_input: Parameters<typeof fetch>[0], init?: Parameters<typeof fetch>[1]) => {
        const body = JSON.parse(String(init?.body)) as Record<string, unknown>;

        // Adaptive thinking flows through on Opus (only manual budget_tokens is
        // omitted there); effort rides in output_config.
        expect(body.thinking).toEqual({ type: "adaptive" });
        expect(body.output_config).toEqual({ effort: "max" });

        return jsonResponse(createMessageBody());
      },
    );
    vi.stubGlobal("fetch", fetchMock);

    const client = new AnthropicLLMClient({
      env: {
        ANTHROPIC_AUTH_TOKEN: "oauth-token",
      },
    });

    await expect(
      client.complete({
        model: "claude-opus-4-8",
        system: "be concise",
        messages: [{ role: "user", content: "hello" }],
        thinking: { type: "adaptive" },
        effort: "max",
        max_tokens: 32,
        budget: "test",
      }),
    ).resolves.toMatchObject({
      text: "Hello",
    });
  });

  it("omits thinking and effort when tool_choice forces tool use", async () => {
    const fetchMock = vi.fn(
      async (_input: Parameters<typeof fetch>[0], init?: Parameters<typeof fetch>[1]) => {
        const body = JSON.parse(String(init?.body)) as Record<string, unknown>;

        // The API rejects thinking under forced tool use, so both thinking and the
        // effort that rides with it are dropped.
        expect(body.thinking).toBeUndefined();
        expect(body.output_config).toBeUndefined();

        return jsonResponse(createMessageBody());
      },
    );
    vi.stubGlobal("fetch", fetchMock);

    const client = new AnthropicLLMClient({
      env: {
        ANTHROPIC_AUTH_TOKEN: "oauth-token",
      },
    });

    await expect(
      client.complete({
        model: "claude-opus-4-8",
        system: "be concise",
        messages: [{ role: "user", content: "hello" }],
        tools: [{ name: "EmitX", inputSchema: { type: "object" } }],
        tool_choice: { type: "any" },
        thinking: { type: "adaptive" },
        effort: "max",
        max_tokens: 32,
        budget: "test",
      }),
    ).resolves.toMatchObject({
      text: "Hello",
    });
  });

  it("captures thinking blocks from converse responses so they can round-trip", async () => {
    const fetchMock = vi.fn(async () =>
      jsonResponse(
        createMessageBody({
          content: [
            {
              type: "thinking",
              thinking: "step-by-step reasoning",
              signature: "sig-abc",
            },
            {
              type: "text",
              text: "done",
              citations: null,
            },
          ],
        } as unknown as Partial<Message>),
      ),
    );
    vi.stubGlobal("fetch", fetchMock);

    const client = new AnthropicLLMClient({
      env: {
        ANTHROPIC_AUTH_TOKEN: "oauth-token",
      },
    });

    const result = await client.converse({
      model: "claude-opus-4-8",
      system: "be concise",
      messages: [{ role: "user", content: [{ type: "text", text: "hi" }] }],
      thinking: { type: "adaptive" },
      effort: "max",
      max_tokens: 32,
      budget: "test",
    });

    expect(result.messageBlocks).toEqual([
      { type: "thinking", thinking: "step-by-step reasoning", signature: "sig-abc" },
      { type: "text", text: "done" },
    ]);
  });

  it("preserves non-Opus temperature settings", async () => {
    const fetchMock = vi.fn(
      async (_input: Parameters<typeof fetch>[0], init?: Parameters<typeof fetch>[1]) => {
        const body = JSON.parse(String(init?.body)) as Record<string, unknown>;

        expect(body.temperature).toBe(0.3);
        expect(body.thinking).toEqual({ type: "disabled" });

        return jsonResponse(createMessageBody());
      },
    );
    vi.stubGlobal("fetch", fetchMock);

    const client = new AnthropicLLMClient({
      env: {
        ANTHROPIC_AUTH_TOKEN: "oauth-token",
      },
    });

    await expect(
      client.complete({
        model: "claude-haiku-4-5",
        system: "be concise",
        messages: [{ role: "user", content: "hello" }],
        temperature: 0.3,
        thinking: { type: "disabled" },
        max_tokens: 32,
        budget: "test",
      }),
    ).resolves.toMatchObject({
      text: "Hello",
    });
  });

  it("routes requests through ANTHROPIC_BASE_URL when set", async () => {
    const requestedUrls: string[] = [];
    const fetchMock = vi.fn(async (input: Parameters<typeof fetch>[0]) => {
      const url = input instanceof Request ? input.url : String(input);
      requestedUrls.push(url);
      return jsonResponse(createMessageBody());
    });
    vi.stubGlobal("fetch", fetchMock);

    const client = new AnthropicLLMClient({
      env: {
        ANTHROPIC_AUTH_TOKEN: "oauth-token",
        ANTHROPIC_BASE_URL: "https://aiproxy.example.com",
      },
    });

    await client.complete({
      model: "claude-sonnet-4-5",
      system: "be concise",
      messages: [{ role: "user", content: "hello" }],
      max_tokens: 32,
      budget: "test",
    });

    expect(requestedUrls.length).toBeGreaterThan(0);
    for (const url of requestedUrls) {
      expect(url.startsWith("https://aiproxy.example.com")).toBe(true);
    }
  });

  it("builds an OAuth client from the shared credentials file", async () => {
    const credentialsPath = createTempCredentialsPath(tempDirs);
    writeJsonFileAtomic(credentialsPath, {
      claudeAiOauth: {
        accessToken: "oauth-access",
        refreshToken: "oauth-refresh",
        expiresAt: Date.now() + 3_600_000,
      },
    });

    const fetchMock = vi.fn(
      async (_input: Parameters<typeof fetch>[0], init?: Parameters<typeof fetch>[1]) => {
        const body = JSON.parse(String(init?.body)) as {
          system: Array<{ type: string; text: string }>;
        };

        expect(body.system[0]?.text).toBe(CLAUDE_CODE_IDENTITY_BLOCK_TEXT);
        return jsonResponse(createMessageBody());
      },
    );
    vi.stubGlobal("fetch", fetchMock);

    const client = new AnthropicLLMClient({
      env: {
        BORG_CLAUDE_CREDENTIALS_PATH: credentialsPath,
      },
    });

    await expect(
      client.complete({
        model: "claude-sonnet-4-5",
        system: "be concise",
        messages: [{ role: "user", content: "hello" }],
        max_tokens: 32,
        budget: "test",
      }),
    ).resolves.toMatchObject({
      text: "Hello",
    });
  });

  it("picks up a credentials-file account swap without a restart", async () => {
    // A long-lived process resolved auth once at startup and held the previous
    // account's token for its whole lifetime: swapping accounts after
    // exhausting a subscription answers 429, not 401, so the auth-failure
    // refresh path never fires. Observed live on 2026-08-09 -- new credentials
    // were on disk at 09:24 and a wake at 09:28 still failed.
    const credentialsPath = createTempCredentialsPath(tempDirs);
    writeJsonFileAtomic(credentialsPath, {
      claudeAiOauth: {
        accessToken: "first-account-access",
        refreshToken: "first-account-refresh",
        expiresAt: Date.now() + 3_600_000,
      },
    });

    const seenTokens: string[] = [];
    const fetchMock = vi.fn(
      async (_input: Parameters<typeof fetch>[0], init?: Parameters<typeof fetch>[1]) => {
        seenTokens.push(new Headers(init?.headers).get("authorization") ?? "");
        return jsonResponse(createMessageBody());
      },
    );
    vi.stubGlobal("fetch", fetchMock);

    const client = new AnthropicLLMClient({
      env: {
        BORG_CLAUDE_CREDENTIALS_PATH: credentialsPath,
      },
    });
    const call = async (): Promise<unknown> =>
      client.complete({
        model: "claude-sonnet-4-5",
        messages: [{ role: "user", content: "hello" }],
        max_tokens: 32,
        budget: "test",
      });

    await call();
    // Same credentials: the cached client must be reused, not re-resolved.
    await call();

    writeJsonFileAtomic(credentialsPath, {
      claudeAiOauth: {
        accessToken: "second-account-access",
        refreshToken: "second-account-refresh",
        expiresAt: Date.now() + 3_600_000,
      },
    });

    await call();

    expect(seenTokens).toEqual([
      "Bearer first-account-access",
      "Bearer first-account-access",
      "Bearer second-account-access",
    ]);
  });

  it("throws an auth error when no credentials are available", async () => {
    const credentialsPath = createTempCredentialsPath(tempDirs);
    const client = new AnthropicLLMClient({
      env: {
        BORG_CLAUDE_CREDENTIALS_PATH: credentialsPath,
      },
    });

    await expect(
      client.complete({
        model: "claude-sonnet-4-5",
        messages: [{ role: "user", content: "hello" }],
        max_tokens: 32,
        budget: "test",
      }),
    ).rejects.toBeInstanceOf(AuthError);
  });

  it("retries initialization after a transient auth resolution failure", async () => {
    const credentialsPath = createTempCredentialsPath(tempDirs);
    const client = new AnthropicLLMClient({
      env: {
        BORG_CLAUDE_CREDENTIALS_PATH: credentialsPath,
      },
    });

    await expect(
      client.complete({
        model: "claude-sonnet-4-5",
        messages: [{ role: "user", content: "hello" }],
        max_tokens: 32,
        budget: "test",
      }),
    ).rejects.toBeInstanceOf(AuthError);

    writeJsonFileAtomic(credentialsPath, {
      claudeAiOauth: {
        accessToken: "oauth-access",
        refreshToken: "oauth-refresh",
        expiresAt: Date.now() + 3_600_000,
      },
    });
    vi.stubGlobal(
      "fetch",
      vi.fn(async () => jsonResponse(createMessageBody())),
    );

    await expect(
      client.complete({
        model: "claude-sonnet-4-5",
        messages: [{ role: "user", content: "hello again" }],
        max_tokens: 32,
        budget: "test",
      }),
    ).resolves.toMatchObject({
      text: "Hello",
    });
  });

  it("retries once after a 401 by refreshing shared OAuth credentials", async () => {
    const credentialsPath = createTempCredentialsPath(tempDirs);
    writeJsonFileAtomic(credentialsPath, {
      claudeAiOauth: {
        accessToken: "stale-access",
        refreshToken: "refresh-token",
        expiresAt: Date.now() + 3_600_000,
      },
    });

    let messageCalls = 0;
    let refreshCalls = 0;
    vi.stubGlobal(
      "fetch",
      vi.fn(async (input: Parameters<typeof fetch>[0]) => {
        const url = new URL(String(input));

        if (url.pathname === "/v1/oauth/token") {
          refreshCalls += 1;
          return jsonResponse({
            access_token: "fresh-access",
            refresh_token: "fresh-refresh",
            expires_in: 3600,
          });
        }

        if (url.pathname === "/v1/messages") {
          messageCalls += 1;
          if (messageCalls === 1) {
            return new Response(JSON.stringify({ error: { message: "unauthorized" } }), {
              status: 401,
              headers: {
                "content-type": "application/json",
              },
            });
          }

          return jsonResponse(createMessageBody());
        }

        return new Response("unexpected", { status: 500 });
      }),
    );

    const client = new AnthropicLLMClient({
      env: {
        BORG_CLAUDE_CREDENTIALS_PATH: credentialsPath,
      },
    });

    await expect(
      client.complete({
        model: "claude-sonnet-4-5",
        system: "be concise",
        messages: [{ role: "user", content: "hello" }],
        max_tokens: 32,
        budget: "test",
      }),
    ).resolves.toMatchObject({
      text: "Hello",
    });

    expect(messageCalls).toBe(2);
    expect(refreshCalls).toBe(1);
  });

  it("streams EmitSelfReport text without forwarding raw partial JSON", async () => {
    const stream = {
      async *[Symbol.asyncIterator]() {
        yield {
          type: "content_block_start",
          index: 0,
          content_block: {
            type: "tool_use",
            id: "toolu_self_report",
            name: "EmitSelfReport",
            input: {},
          },
        };
        yield {
          type: "content_block_delta",
          index: 0,
          delta: {
            type: "input_json_delta",
            partial_json: '{"kind":"self_report","text":"I am ',
          },
        };
        yield {
          type: "content_block_delta",
          index: 0,
          delta: {
            type: "input_json_delta",
            partial_json: 'steady","persistence_class":"assistant_self_report"}',
          },
        };
        yield {
          type: "content_block_stop",
          index: 0,
        };
      },
      finalMessage: vi.fn(async () =>
        createMessageBody({
          content: [
            {
              type: "tool_use",
              id: "toolu_self_report",
              caller: { type: "direct" },
              name: "EmitSelfReport",
              input: {
                kind: "self_report",
                text: "I am steady",
                persistence_class: "assistant_self_report",
              },
            },
          ],
          stop_reason: "tool_use",
        }),
      ),
    };
    const create = vi.fn();
    const streamFactory = vi.fn(() => stream as never);
    const deltas: string[] = [];
    const client = new AnthropicLLMClient({
      client: {
        messages: {
          create,
          stream: streamFactory,
        },
      },
    });

    await expect(
      client.streamComplete({
        model: "claude-sonnet-4-5",
        messages: [{ role: "user", content: "report" }],
        tools: [
          {
            name: "EmitSelfReport",
            inputSchema: {
              type: "object",
            },
          },
        ],
        tool_choice: { type: "tool", name: "EmitSelfReport" },
        max_tokens: 64,
        budget: "test",
        onTextDelta: (text) => deltas.push(text),
      }),
    ).resolves.toMatchObject({
      stop_reason: "tool_use",
    });

    expect(create).not.toHaveBeenCalled();
    expect(streamFactory).toHaveBeenCalledTimes(1);
    expect(deltas.join("")).toBe("I am steady");
    expect(deltas.join("")).not.toContain("{");
    expect(deltas.join("")).not.toContain("self_report");
  });

  it("lifts stalled stream errors from the Anthropic request failure cause", async () => {
    const stalledError = new LLMError("Anthropic SSE stream stalled after 20ms without a chunk", {
      code: "LLM_STREAM_STALLED",
    });
    const stream = {
      async *[Symbol.asyncIterator]() {
        throw stalledError;
      },
      finalMessage: vi.fn(),
    };
    const client = new AnthropicLLMClient({
      client: {
        messages: {
          create: vi.fn(),
          stream: vi.fn(() => stream as never),
        },
      },
      transportStallMaxRetries: 0,
    });

    await expect(
      client.streamComplete({
        model: "claude-sonnet-4-5",
        messages: [{ role: "user", content: "hello" }],
        max_tokens: 32,
        budget: "test",
      }),
    ).rejects.toMatchObject({
      code: "LLM_STREAM_STALLED",
      cause: {
        code: "LLM_STREAM_STALLED",
      },
    });
  });

  it("lifts event-stalled stream errors through the same retryable transport path", async () => {
    const eventStalledError = new LLMError(
      "Anthropic SSE stream stalled for 30ms before the first message event",
      {
        code: "LLM_STREAM_EVENT_STALLED",
      },
    );
    const stream = {
      async *[Symbol.asyncIterator]() {
        throw new Error("SDK stream wrapper", { cause: eventStalledError });
      },
      finalMessage: vi.fn(),
    };
    const client = new AnthropicLLMClient({
      client: {
        messages: {
          create: vi.fn(),
          stream: vi.fn(() => stream as never),
        },
      },
      transportStallMaxRetries: 0,
    });

    await expect(
      client.streamComplete({
        model: "claude-sonnet-4-5",
        messages: [{ role: "user", content: "hello" }],
        max_tokens: 32,
        budget: "test",
      }),
    ).rejects.toMatchObject({
      code: "LLM_STREAM_EVENT_STALLED",
      cause: {
        cause: {
          code: "LLM_STREAM_EVENT_STALLED",
        },
      },
    });
  });

  it("times out hung streaming SDK calls and aborts the request signal", async () => {
    vi.useFakeTimers();
    let observedSignal: AbortSignal | undefined;
    const stream = {
      async *[Symbol.asyncIterator]() {
        await new Promise(() => undefined);
      },
      finalMessage: vi.fn(),
    };
    const streamFactory = vi.fn((_params: unknown, options?: { signal?: AbortSignal | null }) => {
      observedSignal = options?.signal ?? undefined;
      return stream as never;
    });
    const client = new AnthropicLLMClient({
      client: {
        messages: {
          create: vi.fn(),
          stream: streamFactory,
        },
      },
      streamingCallTimeoutMs: 20,
      oauthSseInactivityTimeoutMs: 1_000,
      oauthSseFirstMessageEventTimeoutMs: 1_000,
      oauthSseMessageEventGapTimeoutMs: 1_000,
    });

    const resultPromise = client
      .streamComplete({
        model: "claude-sonnet-4-5",
        messages: [{ role: "user", content: "hello" }],
        max_tokens: 32,
        budget: "test",
      })
      .then(
        (value) => ({ status: "resolved" as const, value }),
        (reason: unknown) => ({ status: "rejected" as const, reason }),
      );

    await vi.advanceTimersByTimeAsync(25);
    const result = await resultPromise;

    expect(observedSignal?.aborted).toBe(true);
    expect(result).toMatchObject({
      status: "rejected",
      reason: {
        code: "LLM_CALL_TIMED_OUT",
        message: "Anthropic streaming LLM call timed out after 20ms",
      },
    });
  });

  it("clears outer deadline timers after fast unary SDK calls", async () => {
    vi.useFakeTimers();
    const create = vi.fn().mockResolvedValue(createMessageBody());
    const client = new AnthropicLLMClient({
      client: {
        messages: { create },
      },
      unaryCallTimeoutMs: 20,
    });

    await expect(
      client.complete({
        model: "claude-sonnet-4-5",
        messages: [{ role: "user", content: "hello" }],
        max_tokens: 32,
        budget: "test",
      }),
    ).resolves.toMatchObject({
      text: "Hello",
    });

    expect(vi.getTimerCount()).toBe(0);
    await vi.advanceTimersByTimeAsync(25);
    expect(vi.getTimerCount()).toBe(0);
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

  it("forwards block-typed converse messages without flattening them", async () => {
    const create = vi.fn().mockResolvedValue(
      createMessageBody({
        content: [
          { type: "text", text: "Checking", citations: null },
          {
            type: "tool_use",
            id: "toolu_1",
            caller: { type: "direct" },
            name: "lookup",
            input: { id: 1 },
          },
        ],
        stop_reason: "tool_use",
      }),
    );
    const client = new AnthropicLLMClient({
      client: {
        messages: { create },
      },
    });

    const result = await client.converse({
      model: "claude-sonnet-4-5",
      system: "be concise",
      messages: [
        {
          role: "user",
          content: [{ type: "text", text: "hello" }],
        },
        {
          role: "assistant",
          content: [
            {
              type: "tool_use",
              id: "toolu_prev",
              name: "lookup",
              input: { id: 7 },
            },
          ],
        },
        {
          role: "user",
          content: [
            {
              type: "tool_result",
              tool_use_id: "toolu_prev",
              content: '{"value":7}',
            },
          ],
        },
      ],
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

    expect(create).toHaveBeenCalledTimes(1);
    expect(create.mock.calls[0]?.[0]).toMatchObject({
      messages: [
        {
          role: "user",
          content: [{ type: "text", text: "hello" }],
        },
        {
          role: "assistant",
          content: [{ type: "tool_use", id: "toolu_prev", name: "lookup", input: { id: 7 } }],
        },
        {
          role: "user",
          content: [
            {
              type: "tool_result",
              tool_use_id: "toolu_prev",
              content: '{"value":7}',
            },
          ],
        },
      ],
    });
    expect(result).toEqual({
      messageBlocks: [
        { type: "text", text: "Checking" },
        { type: "tool_use", id: "toolu_1", name: "lookup", input: { id: 1 } },
      ],
      input_tokens: 12,
      output_tokens: 7,
      stop_reason: "tool_use",
    });
  });

  it("translates image_ref blocks to Anthropic base64 image blocks without reordering", async () => {
    const create = vi.fn().mockResolvedValue(
      createMessageBody({
        content: [{ type: "text", text: "seen", citations: null }],
      }),
    );
    const attachmentBytes = Buffer.from("image-bytes");
    const client = new AnthropicLLMClient({
      client: {
        messages: { create },
      },
      attachmentResolver: (attachmentId) => {
        expect(attachmentId).toBe("att_aaaaaaaaaaaaaaaa");
        return {
          mediaType: "image/png",
          bytes: attachmentBytes,
        };
      },
    });

    await client.converse({
      model: "claude-sonnet-4-5",
      messages: [
        {
          role: "user",
          content: [
            { type: "text", text: "describe this" },
            { type: "image_ref", attachment_id: "att_aaaaaaaaaaaaaaaa" as never },
          ],
        },
      ],
      max_tokens: 128,
      budget: "test",
    });

    expect(create.mock.calls[0]?.[0]).toMatchObject({
      messages: [
        {
          role: "user",
          content: [
            { type: "text", text: "describe this" },
            {
              type: "image",
              source: {
                type: "base64",
                media_type: "image/png",
                data: attachmentBytes.toString("base64"),
              },
            },
          ],
        },
      ],
    });
  });

  it("preserves multi-image label adjacency in Anthropic conversation payloads", async () => {
    const create = vi.fn().mockResolvedValue(
      createMessageBody({
        content: [{ type: "text", text: "seen", citations: null }],
      }),
    );
    const firstAttachmentBytes = Buffer.from("first-image");
    const secondAttachmentBytes = Buffer.from("second-image");
    const client = new AnthropicLLMClient({
      client: {
        messages: { create },
      },
      attachmentResolver: (attachmentId) => {
        if (attachmentId === "att_aaaaaaaaaaaaaaaa") {
          return {
            mediaType: "image/png",
            bytes: firstAttachmentBytes,
          };
        }

        expect(attachmentId).toBe("att_bbbbbbbbbbbbbbbb");
        return {
          mediaType: "image/png",
          bytes: secondAttachmentBytes,
        };
      },
    });

    await client.converse({
      model: "claude-sonnet-4-5",
      messages: [
        {
          role: "user",
          content: [
            { type: "text", text: "Label A" },
            { type: "image_ref", attachment_id: "att_aaaaaaaaaaaaaaaa" as never },
            { type: "text", text: "Label B" },
            { type: "image_ref", attachment_id: "att_bbbbbbbbbbbbbbbb" as never },
          ],
        },
      ],
      max_tokens: 128,
      budget: "test",
    });

    const content = create.mock.calls[0]?.[0].messages[0]?.content;

    expect(content).toEqual([
      { type: "text", text: "Label A" },
      {
        type: "image",
        source: {
          type: "base64",
          media_type: "image/png",
          data: firstAttachmentBytes.toString("base64"),
        },
      },
      { type: "text", text: "Label B" },
      {
        type: "image",
        source: {
          type: "base64",
          media_type: "image/png",
          data: secondAttachmentBytes.toString("base64"),
        },
      },
    ]);
  });

  it("supports scripted fake llm block conversations", async () => {
    const client = new FakeLLMClient({
      responses: [
        [
          {
            type: "tool_use",
            id: "toolu_1",
            name: "tool.episodic.search",
            input: { query: "planning" },
          },
        ],
        [
          {
            type: "text",
            text: "done",
          },
        ],
      ],
    });

    const first = await client.converse({
      model: "fake",
      messages: [{ role: "user", content: [{ type: "text", text: "hi" }] }],
      max_tokens: 8,
      budget: "test",
    });
    const second = await client.converse({
      model: "fake",
      messages: [{ role: "user", content: [{ type: "text", text: "continue" }] }],
      max_tokens: 8,
      budget: "test",
    });

    expect(first).toEqual({
      messageBlocks: [
        {
          type: "tool_use",
          id: "toolu_1",
          name: "tool.episodic.search",
          input: { query: "planning" },
        },
      ],
      input_tokens: 0,
      output_tokens: 0,
      stop_reason: "tool_use",
    });
    expect(second.messageBlocks).toEqual([{ type: "text", text: "done" }]);
    expect(client.converseRequests).toHaveLength(2);
  });
});
