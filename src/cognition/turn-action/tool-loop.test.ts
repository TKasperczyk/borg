import { mkdtempSync, rmSync } from "node:fs";
import { join } from "node:path";
import { tmpdir } from "node:os";

import { afterEach, describe, expect, it, vi } from "vitest";
import { z } from "zod";

import { Borg } from "../../borg.js";
import { executiveMigrations, ExecutiveStepsRepository } from "../../executive/index.js";
import { GoalsRepository, selfMigrations } from "../../memory/self/index.js";
import { createTestConfig, TestEmbeddingClient } from "../../offline/test-support.js";
import { composeMigrations, openDatabase } from "../../storage/sqlite/index.js";
import { StreamReader, StreamWriter } from "../../stream/index.js";
import {
  ToolDispatcher,
  createGoalsRetireTool,
  createOpenQuestionsCreateTool,
  type ToolDefinition,
} from "../../tools/index.js";
import { ManualClock } from "../../util/clock.js";
import { LLMError } from "../../util/errors.js";
import { DEFAULT_SESSION_ID } from "../../util/ids.js";
import { type LLMContentBlockMessage, type LLMConverseOptions } from "../../llm/index.js";
import { FakeLLMClient } from "../../llm/test-support/fake-client.js";
import { FinalizerToolTranscriptCollector } from "../deliberation/finalizer-tool-transcript.js";
import {
  AUTONOMOUS_TOOL_LOOP_TOOL_ROUND_BUDGET_MS,
  AUTONOMOUS_TOOL_LOOP_WALL_CLOCK_BUDGET_MS,
  executeToolLoop,
} from "./tool-loop.js";

function createDispatcher(tempDir: string, clock = new ManualClock(1_000)): ToolDispatcher {
  return new ToolDispatcher({
    clock,
    createStreamWriter: (sessionId) =>
      new StreamWriter({
        dataDir: tempDir,
        sessionId,
        clock,
      }),
  });
}

function baseMessages(text = "hi"): LLMContentBlockMessage[] {
  return [
    {
      role: "user",
      content: [
        {
          type: "text",
          text,
        },
      ],
    },
  ];
}

async function openTestBorg(tempDir: string, llm = new FakeLLMClient()) {
  return Borg.open({
    config: createTestConfig({
      dataDir: tempDir,
      perception: {
        llmEnabled: false,
      },
      embedding: {
        baseUrl: "http://localhost:1234/v1",
        apiKey: "test",
        model: "test-embed",
        dims: 4,
      },
      anthropic: {
        auth: "api-key",
        apiKey: "test",
        models: {
          cognition: "test-cognition",
          background: "test-background",
          extraction: "test-extraction",
        },
      },
    }),
    clock: new ManualClock(1_000_000),
    embeddingDimensions: 4,
    embeddingClient: new TestEmbeddingClient(),
    llmClient: llm,
    liveExtraction: false,
  });
}

describe("executeToolLoop", () => {
  const tempDirs: string[] = [];

  afterEach(() => {
    while (tempDirs.length > 0) {
      rmSync(tempDirs.pop() as string, { recursive: true, force: true });
    }
  });

  it("returns immediately on a text-only response", async () => {
    const tempDir = mkdtempSync(join(tmpdir(), "borg-"));
    tempDirs.push(tempDir);
    const dispatcher = createDispatcher(tempDir);
    const llm = new FakeLLMClient({
      responses: ["done"],
    });

    const result = await executeToolLoop({
      llmClient: llm,
      dispatcher,
      sessionId: DEFAULT_SESSION_ID,
      model: "fake",
      systemPrompt: "be concise",
      initialMessages: baseMessages(),
      tools: [],
      origin: "deliberator",
      budget: "test",
    });

    expect(result).toMatchObject({
      text: "done",
      iterations: 0,
      toolCallsMade: [],
      stopReason: "text",
    });
    expect(llm.converseRequests[0]?.tools).toBeUndefined();
  });

  it("returns immediately when a terminal tool is called", async () => {
    const tempDir = mkdtempSync(join(tmpdir(), "borg-"));
    tempDirs.push(tempDir);
    const dispatcher = createDispatcher(tempDir);
    const terminalTool: ToolDefinition = {
      name: "no_output",
      description: "Terminal suppression signal.",
      allowedOrigins: ["deliberator"],
      writeScope: "read",
      inputSchema: z.object({}).strict(),
      outputSchema: z.object({}).strict(),
      async invoke() {
        throw new Error("terminal tools are not dispatched");
      },
    };
    const llm = new FakeLLMClient({
      responses: [
        [
          {
            type: "text",
            text: "discard me",
          },
          {
            type: "tool_use",
            id: "toolu_terminal",
            name: "no_output",
            input: {},
          },
        ],
      ],
    });

    const result = await executeToolLoop({
      llmClient: llm,
      dispatcher,
      sessionId: DEFAULT_SESSION_ID,
      model: "fake",
      systemPrompt: "be concise",
      initialMessages: baseMessages(),
      tools: [terminalTool],
      terminalToolNames: ["no_output"],
      origin: "deliberator",
      budget: "test",
    });

    expect(result).toMatchObject({
      text: "discard me",
      iterations: 0,
      toolCallsMade: [],
      stopReason: "terminal_tool",
    });
    expect(result.terminalToolCalls).toMatchObject([
      {
        id: "toolu_terminal",
        name: "no_output",
        input: {},
      },
    ]);
    expect(llm.converseRequests).toHaveLength(1);
    expect(
      new StreamReader({
        dataDir: tempDir,
        sessionId: DEFAULT_SESSION_ID,
      }).tail(10),
    ).toEqual([]);
  });

  it("runs a single tool-use round and returns the final text", async () => {
    const tempDir = mkdtempSync(join(tmpdir(), "borg-"));
    tempDirs.push(tempDir);
    const dispatcher = createDispatcher(tempDir);
    dispatcher.register({
      name: "tool.test.echo",
      description: "Echo test input.",
      allowedOrigins: ["autonomous", "deliberator"],
      writeScope: "read",
      inputSchema: z.object({
        value: z.string().min(1),
      }),
      outputSchema: z.object({
        echoed: z.string().min(1),
      }),
      async invoke(input: { value: string }) {
        return { echoed: input.value };
      },
    });
    const llm = new FakeLLMClient({
      responses: [
        [
          {
            type: "tool_use",
            id: "toolu_1",
            name: "tool.test.echo",
            input: { value: "hello" },
          },
        ],
        "final answer",
      ],
    });

    const result = await executeToolLoop({
      llmClient: llm,
      dispatcher,
      sessionId: DEFAULT_SESSION_ID,
      model: "fake",
      systemPrompt: "be concise",
      initialMessages: baseMessages(),
      tools: dispatcher.listTools("deliberator"),
      origin: "deliberator",
      budget: "test",
    });

    expect(result).toMatchObject({
      text: "final answer",
      iterations: 1,
      stopReason: "text",
      toolCallsMade: [
        {
          callId: "toolu_1",
          name: "tool.test.echo",
          input: { value: "hello" },
          output: { echoed: "hello" },
          ok: true,
        },
      ],
    });
    expect(llm.converseRequests[1]?.messages.at(-1)).toEqual({
      role: "user",
      content: [
        {
          type: "tool_result",
          tool_use_id: "toolu_1",
          content: '{"echoed":"hello"}',
        },
      ],
    });

    const entries = new StreamReader({
      dataDir: tempDir,
      sessionId: DEFAULT_SESSION_ID,
    }).tail(2);
    expect(entries.map((entry) => entry.kind)).toEqual(["tool_call", "tool_result"]);
  });

  it("aggregates cache token fields across tool-loop iterations", async () => {
    const tempDir = mkdtempSync(join(tmpdir(), "borg-"));
    tempDirs.push(tempDir);
    const dispatcher = createDispatcher(tempDir);
    dispatcher.register({
      name: "tool.test.echo",
      description: "Echo test input.",
      allowedOrigins: ["deliberator"],
      writeScope: "read",
      inputSchema: z.object({
        value: z.string().min(1),
      }),
      outputSchema: z.object({
        echoed: z.string().min(1),
      }),
      async invoke(input: { value: string }) {
        return { echoed: input.value };
      },
    });
    const llm = new FakeLLMClient({
      responses: [
        {
          messageBlocks: [
            {
              type: "tool_use",
              id: "toolu_1",
              name: "tool.test.echo",
              input: { value: "hello" },
            },
          ],
          input_tokens: 10,
          output_tokens: 2,
          cache_creation_input_tokens: 3,
          cache_read_input_tokens: 5,
          stop_reason: "tool_use",
        },
        {
          messageBlocks: [
            {
              type: "text",
              text: "final answer",
            },
          ],
          input_tokens: 20,
          output_tokens: 4,
          cache_creation_input_tokens: 7,
          cache_read_input_tokens: 11,
          stop_reason: "end_turn",
        },
      ],
    });

    const result = await executeToolLoop({
      llmClient: llm,
      dispatcher,
      sessionId: DEFAULT_SESSION_ID,
      model: "fake",
      systemPrompt: "be concise",
      initialMessages: baseMessages(),
      tools: dispatcher.listTools("deliberator"),
      origin: "deliberator",
      budget: "test",
    });

    expect(result.usage).toEqual({
      input_tokens: 30,
      output_tokens: 6,
      cache_creation_input_tokens: 10,
      cache_read_input_tokens: 16,
      stop_reason: "end_turn",
    });
  });

  it("re-transforms prior assistant tool_use names in OAuth transport history", async () => {
    const tempDir = mkdtempSync(join(tmpdir(), "borg-"));
    tempDirs.push(tempDir);
    const dispatcher = createDispatcher(tempDir);
    dispatcher.register({
      name: "tool.episodic.search",
      description: "Search episodic memory.",
      allowedOrigins: ["deliberator"],
      writeScope: "read",
      inputSchema: z.object({
        query: z.string().min(1),
      }),
      outputSchema: z.object({
        ok: z.literal(true),
      }),
      async invoke() {
        return { ok: true } as const;
      },
    });
    const llm = new FakeLLMClient({
      oauthToolNameTransport: true,
      responses: [
        [
          {
            type: "tool_use",
            id: "toolu_oauth",
            name: "tool.episodic.search",
            input: { query: "Marta" },
          },
        ],
        (options: LLMConverseOptions) => {
          const assistantMessage = options.messages.find((message) => message.role === "assistant");

          expect(assistantMessage?.content).toContainEqual({
            type: "tool_use",
            id: "toolu_oauth",
            name: "Tool_episodic_search",
            input: { query: "Marta" },
          });

          return "done";
        },
      ],
    });

    const result = await executeToolLoop({
      llmClient: llm,
      dispatcher,
      sessionId: DEFAULT_SESSION_ID,
      model: "fake",
      systemPrompt: "be concise",
      initialMessages: baseMessages(),
      tools: dispatcher.listTools("deliberator"),
      origin: "deliberator",
      budget: "test",
    });

    expect(result.toolCallsMade).toMatchObject([
      {
        name: "tool.episodic.search",
        ok: true,
      },
    ]);
    expect(llm.converseRequests[1]?.messages[1]).toEqual({
      role: "assistant",
      content: [
        {
          type: "tool_use",
          id: "toolu_oauth",
          name: "Tool_episodic_search",
          input: { query: "Marta" },
        },
      ],
    });
  });

  it("executes multiple tool uses sequentially in model order", async () => {
    const tempDir = mkdtempSync(join(tmpdir(), "borg-"));
    tempDirs.push(tempDir);
    const dispatcher = createDispatcher(tempDir);
    const events: string[] = [];

    dispatcher.register({
      name: "tool.test.sequence",
      description: "Records start/end order.",
      allowedOrigins: ["autonomous", "deliberator"],
      writeScope: "read",
      inputSchema: z.object({
        value: z.string().min(1),
      }),
      outputSchema: z.object({
        echoed: z.string().min(1),
      }),
      async invoke(input: { value: string }) {
        events.push(`start:${input.value}`);
        await Promise.resolve();
        events.push(`end:${input.value}`);
        return { echoed: input.value };
      },
    });
    const llm = new FakeLLMClient({
      responses: [
        [
          {
            type: "tool_use",
            id: "toolu_1",
            name: "tool.test.sequence",
            input: { value: "first" },
          },
          {
            type: "tool_use",
            id: "toolu_2",
            name: "tool.test.sequence",
            input: { value: "second" },
          },
        ],
        "done",
      ],
    });

    const loopPromise = executeToolLoop({
      llmClient: llm,
      dispatcher,
      sessionId: DEFAULT_SESSION_ID,
      model: "fake",
      systemPrompt: "be concise",
      initialMessages: baseMessages(),
      tools: dispatcher.listTools("deliberator"),
      origin: "deliberator",
      budget: "test",
    });

    const result = await loopPromise;
    expect(result).toMatchObject({
      toolCallsMade: [
        {
          callId: "toolu_1",
          name: "tool.test.sequence",
          input: { value: "first" },
          output: { echoed: "first" },
          ok: true,
        },
        {
          callId: "toolu_2",
          name: "tool.test.sequence",
          input: { value: "second" },
          output: { echoed: "second" },
          ok: true,
        },
      ],
    });
    expect(events).toEqual(["start:first", "end:first", "start:second", "end:second"]);

    const entries = new StreamReader({
      dataDir: tempDir,
      sessionId: DEFAULT_SESSION_ID,
    }).tail(4);
    expect(entries.map((entry) => entry.kind)).toEqual([
      "tool_call",
      "tool_result",
      "tool_call",
      "tool_result",
    ]);
    expect(entries[0]?.content).toMatchObject({
      call_id: "toolu_1",
      tool_name: "tool.test.sequence",
    });
    expect(entries[2]?.content).toMatchObject({
      call_id: "toolu_2",
      tool_name: "tool.test.sequence",
    });
  });

  it("forces a final text-only call after hitting the iteration cap", async () => {
    const tempDir = mkdtempSync(join(tmpdir(), "borg-"));
    tempDirs.push(tempDir);
    const dispatcher = createDispatcher(tempDir);
    dispatcher.register({
      name: "tool.test.loop",
      description: "Loop helper.",
      allowedOrigins: ["autonomous", "deliberator"],
      writeScope: "read",
      inputSchema: z.object({
        value: z.string().min(1),
      }),
      outputSchema: z.object({
        echoed: z.string().min(1),
      }),
      async invoke(input: { value: string }) {
        return { echoed: input.value };
      },
    });
    const llm = new FakeLLMClient({
      responses: [
        [
          {
            type: "tool_use",
            id: "toolu_1",
            name: "tool.test.loop",
            input: { value: "one" },
          },
        ],
        [
          {
            type: "tool_use",
            id: "toolu_2",
            name: "tool.test.loop",
            input: { value: "two" },
          },
        ],
        "forced final answer",
      ],
    });

    const result = await executeToolLoop({
      llmClient: llm,
      dispatcher,
      sessionId: DEFAULT_SESSION_ID,
      model: "fake",
      systemPrompt: "be concise",
      initialMessages: baseMessages(),
      tools: dispatcher.listTools("deliberator"),
      origin: "deliberator",
      budget: "test",
      maxIterations: 2,
    });

    expect(result).toMatchObject({
      text: "forced final answer",
      iterations: 2,
      stopReason: "max_iterations",
    });
    expect(llm.converseRequests[2]?.tools).toBeUndefined();
  });

  it("returns an error tool result for unknown tools and lets the model recover", async () => {
    const tempDir = mkdtempSync(join(tmpdir(), "borg-"));
    tempDirs.push(tempDir);
    const dispatcher = createDispatcher(tempDir);
    const advertisedTool: ToolDefinition = {
      name: "tool.test.unknown",
      description: "Advertised but not registered.",
      allowedOrigins: ["deliberator"],
      writeScope: "read",
      inputSchema: z.object({}).strict(),
      outputSchema: z.object({}).strict(),
      async invoke() {
        return {};
      },
    };
    const llm = new FakeLLMClient({
      responses: [
        [
          {
            type: "tool_use",
            id: "toolu_1",
            name: "tool.test.unknown",
            input: {},
          },
        ],
        "recovered",
      ],
    });

    const result = await executeToolLoop({
      llmClient: llm,
      dispatcher,
      sessionId: DEFAULT_SESSION_ID,
      model: "fake",
      systemPrompt: "be concise",
      initialMessages: baseMessages(),
      tools: [advertisedTool],
      origin: "deliberator",
      budget: "test",
    });

    expect(result).toMatchObject({
      text: "recovered",
      toolCallsMade: [
        {
          callId: "toolu_1",
          name: "tool.test.unknown",
          input: {},
          ok: false,
        },
      ],
    });
    expect(llm.converseRequests[1]?.messages.at(-1)).toEqual({
      role: "user",
      content: [
        {
          type: "tool_result",
          tool_use_id: "toolu_1",
          content: "Unknown tool",
          is_error: true,
        },
      ],
    });
  });

  it("returns an error tool result when the model requests a tool outside this loop's advertised surface", async () => {
    const tempDir = mkdtempSync(join(tmpdir(), "borg-"));
    tempDirs.push(tempDir);
    const dispatcher = createDispatcher(tempDir);
    let invoked = false;
    const visibleTool: ToolDefinition = {
      name: "tool.test.visible",
      description: "Exposed to this loop.",
      allowedOrigins: ["autonomous", "deliberator"],
      writeScope: "read",
      inputSchema: z.object({}).strict(),
      outputSchema: z.object({
        ok: z.literal(true),
      }),
      async invoke() {
        return { ok: true } as const;
      },
    };
    dispatcher.register(visibleTool);
    dispatcher.register({
      name: "tool.test.hidden",
      description: "Registered but not exposed to this loop.",
      allowedOrigins: ["autonomous", "deliberator"],
      writeScope: "write",
      inputSchema: z.object({
        value: z.string().min(1),
      }),
      outputSchema: z.object({
        echoed: z.string().min(1),
      }),
      async invoke(input: { value: string }) {
        invoked = true;
        return { echoed: input.value };
      },
    });
    const llm = new FakeLLMClient({
      responses: [
        [
          {
            type: "tool_use",
            id: "toolu_hidden",
            name: "tool.test.hidden",
            input: { value: "secret" },
          },
        ],
        "recovered",
      ],
    });

    const result = await executeToolLoop({
      llmClient: llm,
      dispatcher,
      sessionId: DEFAULT_SESSION_ID,
      model: "fake",
      systemPrompt: "be concise",
      initialMessages: baseMessages(),
      tools: [visibleTool],
      origin: "deliberator",
      budget: "test",
    });

    expect(invoked).toBe(false);
    expect(result).toMatchObject({
      text: "recovered",
      toolCallsMade: [
        {
          callId: "toolu_hidden",
          name: "tool.test.hidden",
          input: { value: "secret" },
          ok: false,
          durationMs: 0,
        },
      ],
    });
    expect(llm.converseRequests[1]?.messages.at(-1)).toEqual({
      role: "user",
      content: [
        {
          type: "tool_result",
          tool_use_id: "toolu_hidden",
          content: "tool tool.test.hidden not available in this context",
          is_error: true,
        },
      ],
    });

    const entries = new StreamReader({
      dataDir: tempDir,
      sessionId: DEFAULT_SESSION_ID,
    }).tail(2);
    expect(entries.map((entry) => entry.kind)).toEqual(["tool_call", "tool_result"]);
    expect(entries[0]?.content).toMatchObject({
      call_id: "toolu_hidden",
      tool_name: "tool.test.hidden",
      skipped: true,
      skip_reason: "tool_not_available_in_context",
    });
    expect(entries[1]?.content).toMatchObject({
      call_id: "toolu_hidden",
      ok: false,
      error: "tool_not_available_in_context",
      duration_ms: 0,
    });
  });

  it("returns an error for an advertised but turn-unavailable terminal tool and allows recovery", async () => {
    const tempDir = mkdtempSync(join(tmpdir(), "borg-"));
    tempDirs.push(tempDir);
    const dispatcher = createDispatcher(tempDir);
    let blockedInvoked = false;
    const blockedTerminal: ToolDefinition = {
      name: "tool.test.blockedTerminal",
      description: "Origin-static schema disabled by live policy.",
      allowedOrigins: ["deliberator"],
      writeScope: "read",
      inputSchema: z.object({}).strict(),
      outputSchema: z.object({}).strict(),
      async invoke() {
        blockedInvoked = true;
        return {};
      },
    };
    const enabledTerminal: ToolDefinition = {
      ...blockedTerminal,
      name: "tool.test.enabledTerminal",
      async invoke() {
        throw new Error("terminal tools are not dispatched");
      },
    };
    dispatcher.register(blockedTerminal);
    dispatcher.register(enabledTerminal);
    const llm = new FakeLLMClient({
      responses: [
        [
          {
            type: "tool_use",
            id: "toolu_blocked_terminal",
            name: blockedTerminal.name,
            input: {},
          },
        ],
        [
          {
            type: "tool_use",
            id: "toolu_enabled_terminal",
            name: enabledTerminal.name,
            input: {},
          },
        ],
      ],
    });

    const result = await executeToolLoop({
      llmClient: llm,
      dispatcher,
      sessionId: DEFAULT_SESSION_ID,
      model: "fake",
      systemPrompt: "be concise",
      initialMessages: baseMessages(),
      tools: [blockedTerminal, enabledTerminal],
      terminalToolNames: [blockedTerminal.name, enabledTerminal.name],
      unavailableToolNames: [blockedTerminal.name],
      origin: "deliberator",
      budget: "test",
    });

    expect(blockedInvoked).toBe(false);
    expect(llm.converseRequests[0]?.tools?.map((tool) => tool.name)).toEqual([
      blockedTerminal.name,
      enabledTerminal.name,
    ]);
    expect(llm.converseRequests[1]?.messages.at(-1)).toEqual({
      role: "user",
      content: [
        {
          type: "tool_result",
          tool_use_id: "toolu_blocked_terminal",
          content: `tool ${blockedTerminal.name} not available in this context`,
          is_error: true,
        },
      ],
    });
    expect(result.stopReason).toBe("terminal_tool");
    expect(result.terminalToolCalls.map((call) => call.name)).toEqual([enabledTerminal.name]);
  });

  it("rejects unavailable siblings before accepting an enabled terminal call", async () => {
    const tempDir = mkdtempSync(join(tmpdir(), "borg-"));
    tempDirs.push(tempDir);
    const dispatcher = createDispatcher(tempDir);
    const enabledTerminal: ToolDefinition = {
      name: "tool.test.enabledTerminal",
      description: "Enabled terminal emission.",
      allowedOrigins: ["deliberator"],
      writeScope: "read",
      inputSchema: z.object({}).strict(),
      outputSchema: z.object({}).strict(),
      async invoke() {
        throw new Error("terminal tools are not dispatched");
      },
    };
    const unavailableTerminal: ToolDefinition = {
      ...enabledTerminal,
      name: "tool.test.unavailableTerminal",
    };
    const unavailableOutbound: ToolDefinition = {
      ...enabledTerminal,
      name: "tool.test.unavailableOutbound",
    };
    dispatcher.register(enabledTerminal);
    dispatcher.register(unavailableTerminal);
    dispatcher.register(unavailableOutbound);
    const llm = new FakeLLMClient({
      responses: [
        [
          {
            type: "tool_use",
            id: "toolu_enabled_too_early",
            name: enabledTerminal.name,
            input: {},
          },
          {
            type: "tool_use",
            id: "toolu_unavailable_terminal",
            name: unavailableTerminal.name,
            input: {},
          },
          {
            type: "tool_use",
            id: "toolu_unavailable_outbound",
            name: unavailableOutbound.name,
            input: {},
          },
        ],
        [
          {
            type: "tool_use",
            id: "toolu_enabled_retry",
            name: enabledTerminal.name,
            input: {},
          },
        ],
      ],
    });

    const result = await executeToolLoop({
      llmClient: llm,
      dispatcher,
      sessionId: DEFAULT_SESSION_ID,
      model: "fake",
      systemPrompt: "be concise",
      initialMessages: baseMessages(),
      tools: [enabledTerminal, unavailableTerminal, unavailableOutbound],
      terminalToolNames: [enabledTerminal.name, unavailableTerminal.name],
      unavailableToolNames: [unavailableTerminal.name, unavailableOutbound.name],
      maxToolCallsPerIteration: 1,
      origin: "deliberator",
      budget: "test",
    });

    expect(result.stopReason).toBe("terminal_tool");
    expect(result.terminalToolCalls.map((call) => call.id)).toEqual(["toolu_enabled_retry"]);
    expect(llm.converseRequests).toHaveLength(2);
    expect(llm.converseRequests[1]?.messages.at(-1)).toEqual({
      role: "user",
      content: [
        {
          type: "tool_result",
          tool_use_id: "toolu_enabled_too_early",
          content:
            "Terminal emission was not accepted because a sibling tool call was unavailable. Handle the tool error, then call exactly one enabled terminal emission tool again.",
          is_error: true,
        },
        {
          type: "tool_result",
          tool_use_id: "toolu_unavailable_terminal",
          content: `tool ${unavailableTerminal.name} not available in this context`,
          is_error: true,
        },
        {
          type: "tool_result",
          tool_use_id: "toolu_unavailable_outbound",
          content: `tool ${unavailableOutbound.name} not available in this context`,
          is_error: true,
        },
      ],
    });
    const skippedCalls = new StreamReader({
      dataDir: tempDir,
      sessionId: DEFAULT_SESSION_ID,
    })
      .tail(6)
      .filter((entry) => entry.kind === "tool_call")
      .map((entry) => entry.content);
    expect(skippedCalls).toMatchObject([
      {
        call_id: "toolu_enabled_too_early",
        skipped: true,
        skip_reason: "terminal_deferred_for_unavailable_sibling",
      },
      {
        call_id: "toolu_unavailable_terminal",
        skipped: true,
        skip_reason: "tool_not_available_in_context",
      },
      {
        call_id: "toolu_unavailable_outbound",
        skipped: true,
        skip_reason: "tool_not_available_in_context",
      },
    ]);
  });

  it("returns an error tool result for invalid tool input and lets the model recover", async () => {
    const tempDir = mkdtempSync(join(tmpdir(), "borg-"));
    tempDirs.push(tempDir);
    const dispatcher = createDispatcher(tempDir);
    dispatcher.register({
      name: "tool.test.strict",
      description: "Validates input strictly.",
      allowedOrigins: ["autonomous", "deliberator"],
      writeScope: "read",
      inputSchema: z.object({
        value: z.string().min(1),
      }),
      outputSchema: z.object({
        echoed: z.string().min(1),
      }),
      async invoke(input: { value: string }) {
        return { echoed: input.value };
      },
    });
    const llm = new FakeLLMClient({
      responses: [
        [
          {
            type: "tool_use",
            id: "toolu_1",
            name: "tool.test.strict",
            input: { value: "" },
          },
        ],
        "recovered",
      ],
    });

    const result = await executeToolLoop({
      llmClient: llm,
      dispatcher,
      sessionId: DEFAULT_SESSION_ID,
      model: "fake",
      systemPrompt: "be concise",
      initialMessages: baseMessages(),
      tools: dispatcher.listTools("deliberator"),
      origin: "deliberator",
      budget: "test",
    });

    expect(result.toolCallsMade[0]).toMatchObject({
      callId: "toolu_1",
      name: "tool.test.strict",
      input: { value: "" },
      ok: false,
    });
    expect(llm.converseRequests[1]?.messages.at(-1)).toEqual({
      role: "user",
      content: [
        expect.objectContaining({
          type: "tool_result",
          tool_use_id: "toolu_1",
          is_error: true,
        }),
      ],
    });
  });

  it("logs synthetic skipped entries for over-cap tool uses", async () => {
    const tempDir = mkdtempSync(join(tmpdir(), "borg-"));
    tempDirs.push(tempDir);
    const dispatcher = createDispatcher(tempDir);
    let invokeCount = 0;
    dispatcher.register({
      name: "tool.test.capped",
      description: "Only the first tool call should execute.",
      allowedOrigins: ["autonomous", "deliberator"],
      writeScope: "read",
      inputSchema: z.object({
        value: z.string().min(1),
      }),
      outputSchema: z.object({
        echoed: z.string().min(1),
      }),
      async invoke(input: { value: string }) {
        invokeCount += 1;
        return { echoed: input.value };
      },
    });
    const llm = new FakeLLMClient({
      responses: [
        [
          {
            type: "tool_use",
            id: "toolu_1",
            name: "tool.test.capped",
            input: { value: "first" },
          },
          {
            type: "tool_use",
            id: "toolu_2",
            name: "tool.test.capped",
            input: { value: "second" },
          },
          {
            type: "tool_use",
            id: "toolu_3",
            name: "tool.test.capped",
            input: { value: "third" },
          },
        ],
        "done",
      ],
    });

    const result = await executeToolLoop({
      llmClient: llm,
      dispatcher,
      sessionId: DEFAULT_SESSION_ID,
      model: "fake",
      systemPrompt: "be concise",
      initialMessages: baseMessages(),
      tools: dispatcher.listTools("deliberator"),
      origin: "deliberator",
      budget: "test",
      maxToolCallsPerIteration: 1,
    });

    expect(invokeCount).toBe(1);
    expect(result.toolCallsMade).toMatchObject([
      {
        callId: "toolu_1",
        name: "tool.test.capped",
        input: { value: "first" },
        output: { echoed: "first" },
        ok: true,
      },
      {
        callId: "toolu_2",
        name: "tool.test.capped",
        input: { value: "second" },
        ok: false,
        durationMs: 0,
      },
      {
        callId: "toolu_3",
        name: "tool.test.capped",
        input: { value: "third" },
        ok: false,
        durationMs: 0,
      },
    ]);

    const entries = new StreamReader({
      dataDir: tempDir,
      sessionId: DEFAULT_SESSION_ID,
    }).tail(6);
    expect(entries.map((entry) => entry.kind)).toEqual([
      "tool_call",
      "tool_result",
      "tool_call",
      "tool_result",
      "tool_call",
      "tool_result",
    ]);
    expect(entries[0]?.content).toMatchObject({
      call_id: "toolu_1",
      tool_name: "tool.test.capped",
    });
    expect(entries[2]?.content).toMatchObject({
      call_id: "toolu_2",
      tool_name: "tool.test.capped",
      skipped: true,
      skip_reason: "max_tool_calls_per_iteration",
    });
    expect(entries[3]?.content).toMatchObject({
      call_id: "toolu_2",
      ok: false,
      error: "max_tool_calls_per_iteration",
      duration_ms: 0,
    });
    expect(entries[4]?.content).toMatchObject({
      call_id: "toolu_3",
      tool_name: "tool.test.capped",
      skipped: true,
      skip_reason: "max_tool_calls_per_iteration",
    });
    expect(entries[5]?.content).toMatchObject({
      call_id: "toolu_3",
      ok: false,
      error: "max_tool_calls_per_iteration",
      duration_ms: 0,
    });
  });

  it("allows five real goal retirements plus follow-up work across six autonomous tool rounds", async () => {
    const tempDir = mkdtempSync(join(tmpdir(), "borg-"));
    tempDirs.push(tempDir);
    const clock = new ManualClock(1_000);
    const db = openDatabase(":memory:", {
      migrations: composeMigrations(selfMigrations, executiveMigrations),
    });
    const executiveStepsRepository = new ExecutiveStepsRepository({ db, clock });
    const goalsRepository = new GoalsRepository({ db, clock, executiveStepsRepository });
    const goals = Array.from({ length: 5 }, (_, index) =>
      goalsRepository.add({
        description: `Autonomous batch goal ${index + 1}`,
        priority: 10 - index,
        provenance: { kind: "manual" },
      }),
    );
    const dispatcher = createDispatcher(tempDir, clock);
    dispatcher.register(createGoalsRetireTool({ goalsRepository }));
    dispatcher.register({
      name: "tool.test.autonomous-followup",
      description: "Apply journal or step work after the goal retirements.",
      allowedOrigins: ["autonomous"],
      writeScope: "write",
      inputSchema: z.object({ ordinal: z.number().int().positive() }).strict(),
      outputSchema: z.object({ applied: z.number().int().positive() }).strict(),
      async invoke(input: { ordinal: number }) {
        return { applied: input.ordinal };
      },
    });
    const retirementCall = (goal: (typeof goals)[number], ordinal: number) => ({
      type: "tool_use" as const,
      id: `toolu_autonomous_${ordinal}`,
      name: "tool.goals.retire",
      input: {
        goal_id: goal.id,
        reason: "The batched wake verified that this goal is complete.",
      },
    });
    const followupCall = (ordinal: number) => ({
      type: "tool_use" as const,
      id: `toolu_autonomous_${ordinal}`,
      name: "tool.test.autonomous-followup",
      input: { ordinal },
    });
    const llm = new FakeLLMClient({
      responses: [
        goals.map(retirementCall),
        [followupCall(6)],
        [followupCall(7)],
        [followupCall(8)],
        [followupCall(9)],
        [followupCall(10)],
        "finished ten goal-scoped actions",
      ],
    });

    try {
      const result = await executeToolLoop({
        llmClient: llm,
        dispatcher,
        sessionId: DEFAULT_SESSION_ID,
        model: "fake",
        systemPrompt: "Act on every presented goal before finishing.",
        initialMessages: baseMessages(),
        tools: dispatcher.listTools("autonomous"),
        origin: "autonomous",
        budget: "test",
      });

      expect(result).toMatchObject({
        text: "finished ten goal-scoped actions",
        iterations: 6,
        stopReason: "text",
      });
      expect(result.toolCallsMade).toHaveLength(10);
      expect(result.toolCallsMade.every((call) => call.ok)).toBe(true);
      expect(result.toolCallsMade.slice(0, 5).map((call) => call.name)).toEqual(
        Array(5).fill("tool.goals.retire"),
      );
      expect(goals.map((goal) => goalsRepository.get(goal.id)?.status)).toEqual(
        Array(5).fill("abandoned"),
      );
    } finally {
      db.close();
    }
  });

  it("runs the eighth autonomous tool round and forces the ninth request to finalize", async () => {
    const tempDir = mkdtempSync(join(tmpdir(), "borg-"));
    tempDirs.push(tempDir);
    const clock = new ManualClock(2_000);
    const dispatcher = createDispatcher(tempDir, clock);
    const tool: ToolDefinition = {
      name: "tool.test.eight-round-boundary",
      description: "Record one autonomous boundary action.",
      allowedOrigins: ["autonomous"],
      writeScope: "write",
      inputSchema: z.object({ round: z.number().int().positive() }).strict(),
      outputSchema: z.object({ round: z.number().int().positive() }).strict(),
      async invoke(input: { round: number }) {
        return input;
      },
    };
    dispatcher.register(tool);
    const llm = new FakeLLMClient({
      responses: [
        ...Array.from({ length: 8 }, (_, index) => [
          {
            type: "tool_use" as const,
            id: `toolu_round_${index + 1}`,
            name: tool.name,
            input: { round: index + 1 },
          },
        ]),
        "finalized after eight rounds",
      ],
    });

    const result = await executeToolLoop({
      llmClient: llm,
      dispatcher,
      sessionId: DEFAULT_SESSION_ID,
      model: "fake",
      initialMessages: baseMessages(),
      tools: [tool],
      origin: "autonomous",
      budget: "test",
      clock,
    });

    expect(result).toMatchObject({
      text: "finalized after eight rounds",
      iterations: 8,
      stopReason: "max_iterations",
    });
    expect(result.toolCallsMade).toHaveLength(8);
    expect(llm.converseRequests).toHaveLength(9);
    expect(llm.converseRequests.slice(0, 8).every((request) => request.tools !== undefined)).toBe(
      true,
    );
    expect(llm.converseRequests[8]?.tools).toBeUndefined();
  });

  it("dispatches five autonomous calls in one round and skips the sixth", async () => {
    const tempDir = mkdtempSync(join(tmpdir(), "borg-"));
    tempDirs.push(tempDir);
    const clock = new ManualClock(3_000);
    const dispatcher = createDispatcher(tempDir, clock);
    const invoked: number[] = [];
    const tool: ToolDefinition = {
      name: "tool.test.five-call-boundary",
      description: "Record one autonomous per-round action.",
      allowedOrigins: ["autonomous"],
      writeScope: "write",
      inputSchema: z.object({ ordinal: z.number().int().positive() }).strict(),
      outputSchema: z.object({ ordinal: z.number().int().positive() }).strict(),
      async invoke(input: { ordinal: number }) {
        invoked.push(input.ordinal);
        return input;
      },
    };
    dispatcher.register(tool);
    const llm = new FakeLLMClient({
      responses: [
        Array.from({ length: 6 }, (_, index) => ({
          type: "tool_use" as const,
          id: `toolu_call_${index + 1}`,
          name: tool.name,
          input: { ordinal: index + 1 },
        })),
        "continued after the capped round",
      ],
    });

    const result = await executeToolLoop({
      llmClient: llm,
      dispatcher,
      sessionId: DEFAULT_SESSION_ID,
      model: "fake",
      initialMessages: baseMessages(),
      tools: [tool],
      origin: "autonomous",
      budget: "test",
      clock,
    });

    expect(invoked).toEqual([1, 2, 3, 4, 5]);
    expect(result.toolCallsMade).toHaveLength(6);
    expect(result.toolCallsMade.slice(0, 5).every((call) => call.ok)).toBe(true);
    expect(result.toolCallsMade[5]).toMatchObject({
      callId: "toolu_call_6",
      ok: false,
    });
  });

  it("honors explicit autonomous iteration and per-round call overrides", async () => {
    const tempDir = mkdtempSync(join(tmpdir(), "borg-"));
    tempDirs.push(tempDir);
    const clock = new ManualClock(4_000);
    const dispatcher = createDispatcher(tempDir, clock);
    const invoked: number[] = [];
    const tool: ToolDefinition = {
      name: "tool.test.autonomous-override",
      description: "Record one overridden autonomous action.",
      allowedOrigins: ["autonomous"],
      writeScope: "write",
      inputSchema: z.object({ ordinal: z.number().int().positive() }).strict(),
      outputSchema: z.object({ ordinal: z.number().int().positive() }).strict(),
      async invoke(input: { ordinal: number }) {
        invoked.push(input.ordinal);
        return input;
      },
    };
    dispatcher.register(tool);
    const llm = new FakeLLMClient({
      responses: [
        [
          { type: "tool_use", id: "toolu_override_1", name: tool.name, input: { ordinal: 1 } },
          { type: "tool_use", id: "toolu_override_2", name: tool.name, input: { ordinal: 2 } },
        ],
        "finalized under explicit overrides",
      ],
    });

    const result = await executeToolLoop({
      llmClient: llm,
      dispatcher,
      sessionId: DEFAULT_SESSION_ID,
      model: "fake",
      initialMessages: baseMessages(),
      tools: [tool],
      origin: "autonomous",
      budget: "test",
      maxIterations: 1,
      maxToolCallsPerIteration: 1,
      clock,
    });

    expect(invoked).toEqual([1]);
    expect(result).toMatchObject({ iterations: 1, stopReason: "max_iterations" });
    expect(result.toolCallsMade).toHaveLength(2);
    expect(result.toolCallsMade[1]?.ok).toBe(false);
    expect(llm.converseRequests[1]?.tools).toBeUndefined();
  });

  it("cuts off tool rounds at the autonomous wall-clock boundary and still finalizes", async () => {
    const tempDir = mkdtempSync(join(tmpdir(), "borg-"));
    tempDirs.push(tempDir);
    const clock = new ManualClock(5_000);
    const dispatcher = createDispatcher(tempDir, clock);
    const tool: ToolDefinition = {
      name: "tool.test.autonomous-wall-clock",
      description: "Record one near-deadline autonomous action.",
      allowedOrigins: ["autonomous"],
      writeScope: "write",
      inputSchema: z.object({ round: z.number().int().positive() }).strict(),
      outputSchema: z.object({ round: z.number().int().positive() }).strict(),
      async invoke(input: { round: number }) {
        return input;
      },
    };
    dispatcher.register(tool);
    const toolCall = (round: number) => [
      {
        type: "tool_use" as const,
        id: `toolu_deadline_${round}`,
        name: tool.name,
        input: { round },
      },
    ];
    const llm = new FakeLLMClient({
      responses: [
        () => {
          clock.advance(AUTONOMOUS_TOOL_LOOP_TOOL_ROUND_BUDGET_MS - 1);
          return toolCall(1);
        },
        () => {
          clock.advance(1);
          return toolCall(2);
        },
        "finalized after the wall-clock cutoff",
      ],
    });

    const result = await executeToolLoop({
      llmClient: llm,
      dispatcher,
      sessionId: DEFAULT_SESSION_ID,
      model: "fake",
      initialMessages: baseMessages(),
      tools: [tool],
      origin: "autonomous",
      budget: "test",
      clock,
    });

    expect(AUTONOMOUS_TOOL_LOOP_WALL_CLOCK_BUDGET_MS).toBe(24 * 60_000);
    expect(result).toMatchObject({
      text: "finalized after the wall-clock cutoff",
      iterations: 2,
      stopReason: "max_iterations",
    });
    expect(result.toolCallsMade).toHaveLength(2);
    expect(llm.converseRequests.map((request) => request.timeoutMs)).toEqual([
      12 * 60_000,
      1,
      12 * 60_000,
    ]);
    expect(llm.converseRequests[0]?.tools).toBeDefined();
    expect(llm.converseRequests[1]?.tools).toBeDefined();
    expect(llm.converseRequests[2]?.tools).toBeUndefined();
  });

  it("returns completed work instead of throwing when the final wall-clock allowance expires", async () => {
    const tempDir = mkdtempSync(join(tmpdir(), "borg-"));
    tempDirs.push(tempDir);
    const clock = new ManualClock(6_000);
    const dispatcher = createDispatcher(tempDir, clock);
    const tool: ToolDefinition = {
      name: "tool.test.autonomous-hard-deadline",
      description: "Record work completed before the hard deadline.",
      allowedOrigins: ["autonomous"],
      writeScope: "write",
      inputSchema: z.object({}).strict(),
      outputSchema: z.object({ applied: z.literal(true) }).strict(),
      async invoke() {
        return { applied: true } as const;
      },
    };
    dispatcher.register(tool);
    const llm = new FakeLLMClient({
      responses: [
        () => {
          clock.advance(AUTONOMOUS_TOOL_LOOP_TOOL_ROUND_BUDGET_MS);
          return [
            { type: "tool_use" as const, id: "toolu_before_deadline", name: tool.name, input: {} },
          ];
        },
        () => {
          clock.advance(
            AUTONOMOUS_TOOL_LOOP_WALL_CLOCK_BUDGET_MS - AUTONOMOUS_TOOL_LOOP_TOOL_ROUND_BUDGET_MS,
          );
          throw new LLMError("Final wall-clock allowance expired", {
            code: "LLM_CALL_TIMED_OUT",
          });
        },
      ],
    });

    const result = await executeToolLoop({
      llmClient: llm,
      dispatcher,
      sessionId: DEFAULT_SESSION_ID,
      model: "fake",
      initialMessages: baseMessages(),
      tools: [tool],
      origin: "autonomous",
      budget: "test",
      clock,
    });

    expect(result).toMatchObject({
      iterations: 1,
      stopReason: "max_iterations",
    });
    expect(result.toolCallsMade).toHaveLength(1);
    expect(result.toolCallsMade[0]?.ok).toBe(true);
    expect(llm.converseRequests[1]?.tools).toBeUndefined();
  });

  it("observes every nonterminal result across iterations without treating the terminal call as replayable", async () => {
    const tempDir = mkdtempSync(join(tmpdir(), "borg-"));
    tempDirs.push(tempDir);
    const dispatcher = createDispatcher(tempDir);
    const strictTool: ToolDefinition = {
      name: "tool.test.observed",
      description: "Observed test tool.",
      allowedOrigins: ["deliberator"],
      writeScope: "read",
      inputSchema: z.object({ value: z.string().min(1) }).strict(),
      outputSchema: z.object({ echoed: z.string() }).strict(),
      async invoke(input: { value: string }) {
        return { echoed: input.value };
      },
    };
    const terminalTool: ToolDefinition = {
      name: "EmitAnswer",
      description: "Terminal test tool.",
      allowedOrigins: ["deliberator"],
      writeScope: "read",
      inputSchema: z.object({ text: z.string() }).strict(),
      outputSchema: z.object({}).strict(),
      async invoke() {
        throw new Error("terminal tools are not dispatched");
      },
    };
    dispatcher.register(strictTool);
    const collector = new FinalizerToolTranscriptCollector();
    const llm = new FakeLLMClient({
      responses: [
        [
          {
            type: "tool_use",
            id: "toolu_success",
            name: strictTool.name,
            input: { value: "🌌" },
          },
          {
            type: "tool_use",
            id: "toolu_validation",
            name: strictTool.name,
            input: { value: "" },
          },
          {
            type: "tool_use",
            id: "toolu_cap",
            name: strictTool.name,
            input: { value: "capped" },
          },
        ],
        [
          {
            type: "tool_use",
            id: "toolu_unavailable",
            name: "tool.test.not_advertised",
            input: { value: "unavailable" },
          },
        ],
        [
          {
            type: "tool_use",
            id: "toolu_terminal",
            name: terminalTool.name,
            input: { text: "done" },
          },
        ],
      ],
    });

    const result = await executeToolLoop({
      llmClient: llm,
      dispatcher,
      sessionId: DEFAULT_SESSION_ID,
      model: "fake",
      initialMessages: baseMessages(),
      tools: [strictTool, terminalTool],
      terminalToolNames: [terminalTool.name],
      origin: "deliberator",
      budget: "test",
      maxToolCallsPerIteration: 2,
      toolResultObserver: collector,
    });
    const transcript = collector.finish({
      requestBinding: null,
      expectedEventCount: result.toolCallsMade.length,
      sourceCompleted: true,
    }).transcript;

    expect(result.stopReason).toBe("terminal_tool");
    expect(result.terminalToolCalls.map((call) => call.id)).toEqual(["toolu_terminal"]);
    expect(transcript.complete).toBe(true);
    expect(transcript.events.map((event) => event.disposition)).toEqual([
      "dispatched",
      "dispatched",
      "skipped_iteration_cap",
      "skipped_unavailable",
    ]);
    expect(
      transcript.events.map((event) => [event.ordinal, event.iteration, event.batch_position]),
    ).toEqual([
      [1, 1, 1],
      [2, 1, 2],
      [3, 1, 3],
      [4, 2, 1],
    ]);
    expect(transcript.events[0]).toMatchObject({
      call_id: "toolu_success",
      raw_arguments: { value: "🌌" },
      result: { ok: true, output: { echoed: "🌌" } },
    });
    expect(transcript.events[1]?.result).toMatchObject({ ok: false, error: expect.any(String) });
    expect(transcript.events[2]?.result).toEqual({
      ok: false,
      error: "Skipped because this turn allows at most 2 tool calls per iteration.",
    });
    expect(transcript.events[3]?.result).toEqual({
      ok: false,
      error: "tool tool.test.not_advertised not available in this context",
    });
  });

  it("swallows observer failures and marks the observer incomplete", async () => {
    const tempDir = mkdtempSync(join(tmpdir(), "borg-"));
    tempDirs.push(tempDir);
    const dispatcher = createDispatcher(tempDir);
    const tool: ToolDefinition = {
      name: "tool.test.observer_failure",
      description: "Observer failure test tool.",
      allowedOrigins: ["deliberator"],
      writeScope: "read",
      inputSchema: z.object({}).strict(),
      outputSchema: z.object({ ok: z.literal(true) }).strict(),
      async invoke() {
        return { ok: true } as const;
      },
    };
    dispatcher.register(tool);
    const observer = {
      observe() {
        throw new Error("capture observer failed");
      },
      markIncomplete: vi.fn(() => {
        throw new Error("capture failure marker also failed");
      }),
    };
    const llm = new FakeLLMClient({
      responses: [
        [{ type: "tool_use", id: "toolu_observer", name: tool.name, input: {} }],
        "live loop continued",
      ],
    });

    await expect(
      executeToolLoop({
        llmClient: llm,
        dispatcher,
        sessionId: DEFAULT_SESSION_ID,
        model: "fake",
        initialMessages: baseMessages(),
        tools: [tool],
        origin: "deliberator",
        budget: "test",
        toolResultObserver: observer,
      }),
    ).resolves.toMatchObject({ text: "live loop continued" });
    expect(observer.markIncomplete).toHaveBeenCalledOnce();
  });

  it("allows deliberator-origin write tools and records deliberator open questions", async () => {
    const tempDir = mkdtempSync(join(tmpdir(), "borg-"));
    tempDirs.push(tempDir);
    const borg = await openTestBorg(tempDir);
    const dispatcher = createDispatcher(tempDir);

    try {
      dispatcher.register(
        createOpenQuestionsCreateTool({
          createOpenQuestion: (input) => borg.self.openQuestions.add(input),
        }),
      );
      const llm = new FakeLLMClient({
        responses: [
          [
            {
              type: "tool_use",
              id: "toolu_1",
              name: "tool.openQuestions.create",
              input: {
                question: "What should I verify before I answer?",
              },
            },
          ],
          "Logged it.",
        ],
      });

      const result = await executeToolLoop({
        llmClient: llm,
        dispatcher,
        sessionId: DEFAULT_SESSION_ID,
        model: "fake",
        systemPrompt: "be concise",
        initialMessages: baseMessages(),
        tools: dispatcher.listTools("deliberator"),
        origin: "deliberator",
        budget: "test",
      });

      expect(result.text).toBe("Logged it.");
      const openQuestions = borg.self.openQuestions.list({ limit: 10 });
      expect(
        openQuestions.find(
          (question) => question.question === "What should I verify before I answer?",
        ),
      ).toMatchObject({
        source: "deliberator",
      });
    } finally {
      await borg.close();
    }
  });
});
