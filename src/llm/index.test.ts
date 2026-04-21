import { mkdtempSync, rmSync } from "node:fs";
import { join } from "node:path";
import { tmpdir } from "node:os";

import { afterEach, describe, expect, it, vi } from "vitest";

import type { Message } from "@anthropic-ai/sdk/resources/messages/messages.js";

import { writeJsonFileAtomic } from "../util/atomic-write.js";
import { AuthError } from "../util/errors.js";
import {
  AnthropicLLMClient,
  CLAUDE_CODE_IDENTITY_BLOCK_TEXT,
  FakeLLMClient,
  createOAuthFetch,
  type TokenUsageEvent,
} from "./index.js";

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
