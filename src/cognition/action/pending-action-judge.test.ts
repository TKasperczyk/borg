import { describe, expect, it } from "vitest";

import { FakeLLMClient } from "../../llm/index.js";
import { LLMPendingActionJudge } from "./pending-action-judge.js";

describe("LLMPendingActionJudge", () => {
  it("accepts LLM-classified operational follow-ups", async () => {
    const llm = new FakeLLMClient({
      responses: [
        {
          text: "",
          input_tokens: 1,
          output_tokens: 1,
          stop_reason: "tool_use",
          tool_calls: [
            {
              id: "tool_1",
              name: "ClassifyPendingAction",
              input: {
                classification: "action",
                reason: "Concrete future follow-up.",
                confidence: 0.9,
              },
            },
          ],
        },
      ],
    });
    const judge = new LLMPendingActionJudge({ llmClient: llm, model: "judge-model" });

    await expect(
      judge.judge({
        description: "Ask Tom tomorrow whether he wants to revisit the Valencia tutor.",
        next_action: "Ask Tom tomorrow about the Valencia tutor",
      }),
    ).resolves.toEqual({
      accepted: true,
      reason: "Concrete future follow-up.",
      confidence: 0.9,
      degraded: false,
    });
    expect(llm.requests[0]?.tool_choice).toEqual({
      type: "tool",
      name: "ClassifyPendingAction",
    });
  });

  it("rejects LLM-classified belief mutations", async () => {
    const llm = new FakeLLMClient({
      responses: [
        {
          text: "",
          input_tokens: 1,
          output_tokens: 1,
          stop_reason: "tool_use",
          tool_calls: [
            {
              id: "tool_1",
              name: "ClassifyPendingAction",
              input: {
                classification: "non_action",
                reason: "This is a relationship claim.",
                confidence: 0.95,
              },
            },
          ],
        },
      ],
    });
    const judge = new LLMPendingActionJudge({ llmClient: llm, model: "judge-model" });

    await expect(
      judge.judge({
        description: "Tom's partner is unnamed; Maya is a separate person.",
        next_action: "Remember that Maya is separate",
      }),
    ).resolves.toEqual({
      accepted: false,
      reason: "This is a relationship claim.",
      confidence: 0.95,
      degraded: false,
    });
  });
});
