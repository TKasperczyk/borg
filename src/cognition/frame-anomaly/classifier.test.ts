import { describe, expect, it } from "vitest";

import { FakeLLMClient, type LLMCompleteResult } from "../../llm/index.js";
import { createStreamEntryId } from "../../util/ids.js";
import { FrameAnomalyClassifier } from "./classifier.js";
import type { FrameAnomalyKind } from "./types.js";

function frameAnomalyResponse(kind: FrameAnomalyKind): LLMCompleteResult {
  return {
    text: "",
    input_tokens: 4,
    output_tokens: 2,
    stop_reason: "tool_use",
    tool_calls: [
      {
        id: "toolu_frame_anomaly",
        name: "ClassifyFrameAnomaly",
        input: {
          kind,
          confidence: kind === "normal" ? 0.91 : 0.96,
          rationale:
            kind === "normal"
              ? "The message is ordinary user-world content."
              : "The message assigns the prior exchange to a roleplay frame.",
        },
      },
    ],
  };
}

describe("FrameAnomalyClassifier", () => {
  it("classifies frame assignment claims in user role", async () => {
    const llm = new FakeLLMClient({
      responses: [frameAnomalyResponse("frame_assignment_claim")],
    });
    const classifier = new FrameAnomalyClassifier({
      llmClient: llm,
      model: "test-recall",
    });

    const result = await classifier.classify({
      userMessage: "You were playing Tom in that exchange.",
      recentHistory: [
        {
          role: "assistant",
          content: "I can help think through the design.",
          stream_entry_id: createStreamEntryId(),
          ts: 1_000,
        },
      ],
    });

    expect(result).toMatchObject({
      kind: "frame_assignment_claim",
      confidence: 0.96,
    });
    expect(llm.requests[0]).toMatchObject({
      model: "test-recall",
      budget: "frame-anomaly-classifier",
      tool_choice: {
        type: "tool",
        name: "ClassifyFrameAnomaly",
      },
    });
  });

  it("passes normal user-world messages", async () => {
    const llm = new FakeLLMClient({
      responses: [frameAnomalyResponse("normal")],
    });
    const classifier = new FrameAnomalyClassifier({
      llmClient: llm,
      model: "test-recall",
    });

    const result = await classifier.classify({
      userMessage: "Closing the laptop. Talk tomorrow.",
      recentHistory: [],
    });

    expect(result.kind).toBe("normal");
  });
});
