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
import { runFinalizer } from "./finalizer.js";

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
  tools: readonly ToolDefinition[] = [extraTool],
) {
  return runFinalizer({
    llmClient: llm,
    dispatcher: createDispatcher(tempDirs),
    sessionId: DEFAULT_SESSION_ID,
    model: "fake",
    baseSystemPrompt: "Base dynamic prompt.",
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
    tools,
    userEntryId: undefined,
    maxTokens: 256,
    path: "system_1",
    mode: "emission_tools",
  });
}

describe("runFinalizer emission tools", () => {
  const tempDirs: string[] = [];

  afterEach(() => {
    while (tempDirs.length > 0) {
      rmSync(tempDirs.pop() as string, { recursive: true, force: true });
    }
  });

  it("exposes only emission tools without dead cache-control markers", async () => {
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
        text: expect.stringContaining(
          "Call exactly ONE of EmitAnswer / EmitNoOutput / EmitSelfReport per turn.",
        ),
      }),
      expect.objectContaining({
        type: "text",
        text: "Base dynamic prompt.",
      }),
    ]);
    expect(
      (llm.requests[0]?.system as readonly Record<string, unknown>[]).some(
        (block) => "cache_control" in block,
      ),
    ).toBe(false);
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
