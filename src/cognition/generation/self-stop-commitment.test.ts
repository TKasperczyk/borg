import { describe, expect, it } from "vitest";

import { FakeLLMClient } from "../../llm/index.js";
import { StopCommitmentExtractor } from "./self-stop-commitment.js";

function commitmentResponse(classification: "stop_until_substantive_content" | "none") {
  return {
    text: "",
    input_tokens: 4,
    output_tokens: 2,
    stop_reason: "tool_use" as const,
    tool_calls: [
      {
        id: "toolu_stop",
        name: "EmitStopCommitmentClassification",
        input: {
          classification,
          reason:
            classification === "stop_until_substantive_content"
              ? "Assistant committed to stop until real content arrives."
              : "Local wording only.",
          confidence: 0.9,
        },
      },
    ],
  };
}

describe("StopCommitmentExtractor", () => {
  it("classifies explicit stop commitments through the LLM tool", async () => {
    const llm = new FakeLLMClient({
      responses: [commitmentResponse("stop_until_substantive_content")],
    });
    const extractor = new StopCommitmentExtractor({
      llmClient: llm,
      model: "test",
    });

    await expect(
      extractor.extract({
        userMessage: "No.",
        agentResponse: "I'm going to stop responding to these until real content arrives.",
      }),
    ).resolves.toEqual({
      reason: "Assistant committed to stop until real content arrives.",
      confidence: 0.9,
    });
    expect(llm.requests).toHaveLength(1);
  });

  it("does not call the LLM when the response has no stop trigger", async () => {
    const llm = new FakeLLMClient();
    const extractor = new StopCommitmentExtractor({
      llmClient: llm,
      model: "test",
    });

    await expect(
      extractor.extract({
        userMessage: "Thanks.",
        agentResponse: "You're welcome.",
      }),
    ).resolves.toBeNull();
    expect(llm.requests).toHaveLength(0);
  });
});
