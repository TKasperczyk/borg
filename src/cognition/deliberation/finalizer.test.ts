import { mkdtempSync, rmSync } from "node:fs";
import { join } from "node:path";
import { tmpdir } from "node:os";

import { afterEach, describe, expect, it } from "vitest";
import { z } from "zod";

import { FakeLLMClient } from "../../llm/index.js";
import { StreamWriter } from "../../stream/index.js";
import { ToolDispatcher, type ToolDefinition } from "../../tools/index.js";
import { FixedClock } from "../../util/clock.js";
import { DEFAULT_SESSION_ID } from "../../util/ids.js";
import { runFinalizer, type CacheableFinalizerSystemPrompt } from "./finalizer.js";

function createDispatcher(tempDirs: string[]): ToolDispatcher {
  const tempDir = mkdtempSync(join(tmpdir(), "borg-finalizer-"));
  tempDirs.push(tempDir);
  const clock = new FixedClock(0);

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

const extraTool: ToolDefinition = {
  name: "tool.memory.write",
  description: "A non-emission tool that must not be exposed to the finalizer.",
  allowedOrigins: ["deliberator"],
  writeScope: "write",
  inputSchema: z.object({}).strict(),
  outputSchema: z.object({}).strict(),
  async invoke() {
    return {};
  },
};

async function runEmissionFinalizer(
  llm: FakeLLMClient,
  tempDirs: string[],
  options: {
    tools?: readonly ToolDefinition[];
    cacheableSystemPrompt?: CacheableFinalizerSystemPrompt;
    additionalPromptSections?: readonly (string | null)[];
  } = {},
) {
  return runFinalizer({
    llmClient: llm,
    dispatcher: createDispatcher(tempDirs),
    sessionId: DEFAULT_SESSION_ID,
    model: "fake",
    baseSystemPrompt: "Legacy base dynamic prompt.",
    cacheableSystemPrompt: options.cacheableSystemPrompt ?? {
      staticPrefix: "Stable static prompt.",
      dynamicContent: "Base dynamic prompt.",
    },
    initialMessages: [
      {
        role: "user",
        content: [
          {
            type: "text",
            text: "Please respond.",
          },
        ],
      },
    ],
    tools: options.tools ?? [extraTool],
    userEntryId: undefined,
    maxTokens: 256,
    path: "system_1",
    mode: "emission_tools",
    ...(options.additionalPromptSections === undefined
      ? {}
      : { additionalPromptSections: options.additionalPromptSections }),
  });
}

describe("runFinalizer emission tools", () => {
  const tempDirs: string[] = [];

  afterEach(() => {
    while (tempDirs.length > 0) {
      rmSync(tempDirs.pop() as string, { recursive: true, force: true });
    }
  });

  it("exposes only emission tools with a cacheable static system block", async () => {
    const llm = new FakeLLMClient({
      responses: [
        {
          messageBlocks: [
            {
              type: "tool_use",
              id: "toolu_answer",
              name: "EmitAnswer",
              input: { text: "Answer." },
            },
          ],
          input_tokens: 4,
          output_tokens: 2,
          stop_reason: "tool_use",
        },
      ],
    });

    const result = await runEmissionFinalizer(llm, tempDirs);

    expect(result.decision).toEqual({
      kind: "answer",
      text: "Answer.",
      source: "tool",
    });
    expect(llm.requests[0]?.tools?.map((tool) => tool.name)).toEqual([
      "EmitAnswer",
      "EmitNoOutput",
      "EmitSelfReport",
    ]);
    expect(llm.requests[0]?.tools?.some((tool) => "cache_control" in tool)).toBe(false);
    expect(llm.requests[0]?.system).toEqual([
      expect.objectContaining({
        type: "text",
        cache_control: { type: "ephemeral", ttl: "1h" },
        text: expect.stringContaining(
          "Call exactly ONE of EmitAnswer / EmitNoOutput / EmitSelfReport per turn.",
        ),
      }),
      expect.objectContaining({
        type: "text",
        text: "Base dynamic prompt.",
      }),
    ]);
  });

  it("keeps the static system block byte-identical when dynamic context changes", async () => {
    const firstLlm = new FakeLLMClient({
      responses: [
        {
          messageBlocks: [
            {
              type: "tool_use",
              id: "toolu_first",
              name: "EmitAnswer",
              input: { text: "First." },
            },
          ],
          input_tokens: 4,
          output_tokens: 2,
          stop_reason: "tool_use",
        },
      ],
    });
    const secondLlm = new FakeLLMClient({
      responses: [
        {
          messageBlocks: [
            {
              type: "tool_use",
              id: "toolu_second",
              name: "EmitAnswer",
              input: { text: "Second." },
            },
          ],
          input_tokens: 4,
          output_tokens: 2,
          stop_reason: "tool_use",
        },
      ],
    });

    await runEmissionFinalizer(firstLlm, tempDirs, {
      cacheableSystemPrompt: {
        staticPrefix: "Stable static prompt.",
        dynamicContent: "Dynamic context one.",
      },
      additionalPromptSections: ["Evidence ledger one."],
    });
    await runEmissionFinalizer(secondLlm, tempDirs, {
      cacheableSystemPrompt: {
        staticPrefix: "Stable static prompt.",
        dynamicContent: "Dynamic context two.",
      },
      additionalPromptSections: ["Evidence ledger two."],
    });

    const firstSystem = firstLlm.requests[0]?.system as readonly { text: string }[];
    const secondSystem = secondLlm.requests[0]?.system as readonly { text: string }[];

    expect(firstSystem[0]?.text).toBe(secondSystem[0]?.text);
    expect(firstSystem[1]?.text).toBe("Dynamic context one.\n\nEvidence ledger one.");
    expect(secondSystem[1]?.text).toBe("Dynamic context two.\n\nEvidence ledger two.");
  });

  it("treats free text without an emission tool as a protocol failure", async () => {
    const llm = new FakeLLMClient({
      responses: ["I forgot to call the finalizer tool."],
    });

    const result = await runEmissionFinalizer(llm, tempDirs);

    expect(result.decision).toEqual({
      kind: "invalid_tool",
      toolName: "none",
      reason: "expected exactly one emission tool call, got 0",
    });
  });

  it("treats empty EmitAnswer text as an empty finalizer", async () => {
    const llm = new FakeLLMClient({
      responses: [
        {
          messageBlocks: [
            {
              type: "tool_use",
              id: "toolu_answer_empty",
              name: "EmitAnswer",
              input: { text: "" },
            },
          ],
          input_tokens: 4,
          output_tokens: 2,
          stop_reason: "tool_use",
        },
      ],
    });

    const result = await runEmissionFinalizer(llm, tempDirs);

    expect(result.decision).toEqual({ kind: "empty" });
  });

  it("treats empty EmitSelfReport text as an empty finalizer", async () => {
    const llm = new FakeLLMClient({
      responses: [
        {
          messageBlocks: [
            {
              type: "tool_use",
              id: "toolu_self_report_empty",
              name: "EmitSelfReport",
              input: { text: "" },
            },
          ],
          input_tokens: 4,
          output_tokens: 2,
          stop_reason: "tool_use",
        },
      ],
    });

    const result = await runEmissionFinalizer(llm, tempDirs);

    expect(result.decision).toEqual({ kind: "empty" });
  });

  it("rejects parallel terminal emission tool calls", async () => {
    const llm = new FakeLLMClient({
      responses: [
        {
          messageBlocks: [
            {
              type: "tool_use",
              id: "toolu_answer",
              name: "EmitAnswer",
              input: { text: "Answer." },
            },
            {
              type: "tool_use",
              id: "toolu_no_output",
              name: "EmitNoOutput",
              input: { reason: "natural_close" },
            },
          ],
          input_tokens: 4,
          output_tokens: 2,
          stop_reason: "tool_use",
        },
      ],
    });

    const result = await runEmissionFinalizer(llm, tempDirs);

    expect(result.decision).toEqual({
      kind: "invalid_tool",
      toolName: "multiple",
      reason: "expected exactly one emission tool call, got 2",
    });
  });
});
