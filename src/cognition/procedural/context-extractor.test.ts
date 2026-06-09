import { describe, expect, it, vi } from "vitest";

import { FakeLLMClient } from "../../llm/test-support/fake-client.js";
import { ProceduralContextExtractor } from "./context-extractor.js";

const TOOL_NAME = "EmitProceduralContext";

function input() {
  return {
    userMessage: "The TypeScript build is failing on a generic constraint.",
    recentMessages: [],
    perception: {
      mode: "problem_solving" as const,
      entities: [],
    },
    isSelfAudience: false,
    audienceEntityId: null,
  };
}

describe("ProceduralContextExtractor", () => {
  it("degrades to null when no LLM is configured", async () => {
    const onDegraded = vi.fn();
    const extractor = new ProceduralContextExtractor({ onDegraded });

    await expect(extractor.extract(input())).resolves.toBeNull();
    expect(onDegraded).toHaveBeenCalledWith("llm_unavailable", undefined);
  });

  it("extracts a procedural context from the structured tool call", async () => {
    const llm = new FakeLLMClient({
      responses: [
        {
          text: "",
          input_tokens: 12,
          output_tokens: 8,
          stop_reason: "tool_use",
          tool_calls: [
            {
              id: "toolu_context",
              name: TOOL_NAME,
              input: {
                problem_kind: "code_debugging",
                domain_tags: ["typescript", "build"],
                confidence: 0.9,
              },
            },
          ],
        },
      ],
    });
    const extractor = new ProceduralContextExtractor({
      llmClient: llm,
      model: "model",
    });

    await expect(extractor.extract(input())).resolves.toMatchObject({
      problem_kind: "code_debugging",
      domain_tags: ["typescript", "build"],
      audience_scope: "unknown",
    });
  });
});
