import { mkdtempSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";

import { afterEach, describe, expect, it } from "vitest";

import {
  Borg,
  ManualClock,
} from "../index.js";
import {
  FakeLLMClient,
  createFakeEmitAnswerResponse,
} from "../llm/test-support/fake-client.js";
import { createTestConfig, TestEmbeddingClient } from "../offline/test-support.js";

async function openTestBorg(tempDir: string, llm: FakeLLMClient) {
  return Borg.open({
    config: createTestConfig({
      dataDir: tempDir,
      perception: {
        useLlmFallback: false,
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
        createFakeEmitAnswerResponse("I reflected on the last few turns.", {
            inputTokens: 8,
            outputTokens: 4,
          }),
        createFakeEmitAnswerResponse("Fresh answer for the user.", {
            inputTokens: 8,
            outputTokens: 4,
          }),
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

      const finalizerRequests = llm.requests.filter((request) =>
        request.budget?.startsWith("cognition-system"),
      );

      expect(finalizerRequests).toHaveLength(2);
      expect(finalizerRequests[1]?.messages).toEqual([
        {
          role: "user",
          content: "What changed since yesterday?",
        },
      ]);
    } finally {
      await borg.close();
    }
  });

  it("does not treat prior autonomous wakes as dialogue recency", async () => {
    const tempDir = mkdtempSync(join(tmpdir(), "borg-"));
    tempDirs.push(tempDir);
    const llm = new FakeLLMClient({
      responses: [
        createFakeEmitAnswerResponse("I reflected on the last few turns.", {
            inputTokens: 8,
            outputTokens: 4,
          }),
        createFakeEmitAnswerResponse("I continued the reflection.", {
            inputTokens: 8,
            outputTokens: 4,
          }),
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

      const finalizerRequests = llm.requests.filter((request) =>
        request.budget?.startsWith("cognition-system"),
      );

      expect(finalizerRequests).toHaveLength(2);
      expect(finalizerRequests[1]?.messages).toEqual([
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
