import { mkdtempSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";

import { afterEach, describe, expect, it } from "vitest";

import { Borg, FakeLLMClient, ManualClock } from "../index.js";
import { TestEmbeddingClient } from "../offline/test-support.js";

async function openTestBorg(tempDir: string, llm: FakeLLMClient) {
  return Borg.open({
    config: {
      dataDir: tempDir,
      perception: {
        useLlmFallback: false,
        modeWhenLlmAbsent: "idle",
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

function createEmptyReflectionResponse() {
  return {
    text: "",
    input_tokens: 4,
    output_tokens: 2,
    stop_reason: "tool_use" as const,
    tool_calls: [
      {
        id: "toolu_reflection",
        name: "EmitTurnReflection",
        input: {
          advanced_goals: [],
          procedural_outcomes: [],
          trait_demonstrations: [],
          intent_updates: [],
        },
      },
    ],
  };
}

describe("self-turn recency", () => {
  const tempDirs: string[] = [];

  afterEach(() => {
    while (tempDirs.length > 0) {
      rmSync(tempDirs.pop() as string, { recursive: true, force: true });
    }
  });

  it("excludes prior self-turns from the next user turn", async () => {
    const tempDir = mkdtempSync(join(tmpdir(), "borg-"));
    tempDirs.push(tempDir);
    const llm = new FakeLLMClient({
      responses: [
        {
          text: "I reflected on the last few turns.",
          input_tokens: 8,
          output_tokens: 4,
          stop_reason: "end_turn",
          tool_calls: [],
        },
        {
          text: "Fresh answer for the user.",
          input_tokens: 8,
          output_tokens: 4,
          stop_reason: "end_turn",
          tool_calls: [],
        },
        createEmptyReflectionResponse(),
      ],
    });
    const borg = await openTestBorg(tempDir, llm);

    try {
      await borg.turn({
        userMessage: "Pause and reflect on recent changes.",
        audience: "self",
        origin: "autonomous",
        stakes: "low",
      });
      await borg.turn({
        userMessage: "What changed since yesterday?",
        stakes: "low",
      });

      expect(llm.requests).toHaveLength(3);
      expect(llm.requests[1]?.messages).toEqual([
        {
          role: "user",
          content: "What changed since yesterday?",
        },
      ]);
    } finally {
      await borg.close();
    }
  });

  it("includes prior self-turns when another autonomous self-turn runs", async () => {
    const tempDir = mkdtempSync(join(tmpdir(), "borg-"));
    tempDirs.push(tempDir);
    const llm = new FakeLLMClient({
      responses: [
        {
          text: "I reflected on the last few turns.",
          input_tokens: 8,
          output_tokens: 4,
          stop_reason: "end_turn",
          tool_calls: [],
        },
        {
          text: "I continued the reflection.",
          input_tokens: 8,
          output_tokens: 4,
          stop_reason: "end_turn",
          tool_calls: [],
        },
      ],
    });
    const borg = await openTestBorg(tempDir, llm);

    try {
      await borg.turn({
        userMessage: "Pause and reflect on recent changes.",
        audience: "self",
        origin: "autonomous",
        stakes: "low",
      });
      await borg.turn({
        userMessage: "Continue the reflection with any new pattern you notice.",
        audience: "self",
        origin: "autonomous",
        stakes: "low",
      });

      expect(llm.requests).toHaveLength(2);
      expect(llm.requests[1]?.messages).toEqual([
        {
          role: "user",
          content: "Pause and reflect on recent changes.",
        },
        {
          role: "assistant",
          content: "I reflected on the last few turns.",
        },
        {
          role: "user",
          content: "Continue the reflection with any new pattern you notice.",
        },
      ]);
    } finally {
      await borg.close();
    }
  });
});
