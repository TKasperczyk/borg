import { mkdtempSync, rmSync } from "node:fs";
import { join } from "node:path";
import { tmpdir } from "node:os";

import { afterEach, describe, expect, it } from "vitest";
import { z } from "zod";

import { Borg } from "../../borg.js";
import { TestEmbeddingClient } from "../../offline/test-support.js";
import { StreamReader, StreamWriter } from "../../stream/index.js";
import { ToolDispatcher, createOpenQuestionsCreateTool, type ToolDefinition } from "../../tools/index.js";
import { ManualClock } from "../../util/clock.js";
import { DEFAULT_SESSION_ID } from "../../util/ids.js";
import { FakeLLMClient, type LLMContentBlockMessage } from "../../llm/index.js";
import { executeToolLoop } from "./tool-loop.js";

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
    config: {
      dataDir: tempDir,
      perception: {
        useLlmFallback: false,
        modeWhenLlmAbsent: "problem_solving",
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
    },
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
      async invoke(input) {
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
      async invoke(input) {
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
      async invoke(input) {
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
      async invoke(input) {
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
      async invoke(input) {
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
      async invoke(input) {
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
        openQuestions.find((question) => question.question === "What should I verify before I answer?"),
      ).toMatchObject({
        source: "deliberator",
      });
    } finally {
      await borg.close();
    }
  });
});
