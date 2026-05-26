import { mkdtempSync, rmSync } from "node:fs";
import { join } from "node:path";
import { tmpdir } from "node:os";

import { afterEach, describe, expect, it, vi } from "vitest";
import { z } from "zod";

import type { Message } from "@anthropic-ai/sdk/resources/messages/messages.js";

import { writeJsonFileAtomic } from "../util/atomic-write.js";
import { AuthError } from "../util/errors.js";
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

describe("llm", () => {
  const tempDirs: string[] = [];

  afterEach(() => {
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
        model: "claude-opus-4-7",
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
