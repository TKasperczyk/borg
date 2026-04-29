import { describe, expect, it, vi } from "vitest";

import { FakeLLMClient } from "../../llm/index.js";
import type { TurnTracer } from "../tracing/tracer.js";
import { runS2Planner } from "./s2-planner.js";

function createTracer() {
  const emit = vi.fn<TurnTracer["emit"]>();

  return {
    enabled: true,
    includePayloads: false,
    emit,
  } satisfies TurnTracer & { emit: typeof emit };
}

describe("s2 planner", () => {
  it("retries once when the first response omits EmitTurnPlan", async () => {
    const llm = new FakeLLMClient({
      responses: [
        {
          text: "I forgot to emit the tool.",
          input_tokens: 4,
          output_tokens: 3,
          stop_reason: "end_turn",
          tool_calls: [],
        },
        {
          text: "",
          input_tokens: 5,
          output_tokens: 4,
          stop_reason: "tool_use",
          tool_calls: [
            {
              id: "toolu_plan_retry",
              name: "EmitTurnPlan",
              input: {
                uncertainty: "",
                verification_steps: ["confirm rollback state"],
                tensions: [],
                voice_note: "stay direct",
                emission_recommendation: "emit",
                referenced_episode_ids: ["ep_aaaaaaaaaaaaaaaa"],
                intents: [
                  {
                    description: "Check rollback status after the next deploy",
                    next_action: "review deploy status",
                  },
                ],
              },
            },
          ],
        },
      ],
    });

    const result = await runS2Planner({
      llmClient: llm,
      model: "sonnet",
      baseSystemPrompt: "base",
      dialogueMessages: [{ role: "user", content: "Think this through." }],
      selfSnapshot: { values: [], goals: [], traits: [] },
      maxTokens: 512,
    });

    expect(result.plan).toMatchObject({
      verification_steps: ["confirm rollback state"],
      voice_note: "stay direct",
      emission_recommendation: "emit",
      referenced_episode_ids: ["ep_aaaaaaaaaaaaaaaa"],
      intents: [
        {
          description: "Check rollback status after the next deploy",
          next_action: "review deploy status",
        },
      ],
    });
    expect(llm.requests).toHaveLength(2);
    expect(llm.requests[1]?.messages.at(-1)).toEqual({
      role: "user",
      content:
        "Your previous response did not include the required EmitTurnPlan tool_use block. Emit one now -- this is the only way to complete the plan step.",
    });
    expect(result.usage).toMatchObject({
      input_tokens: 9,
      output_tokens: 7,
      stop_reason: "tool_use",
    });
  });

  it("parses an explicit no-output emission recommendation", async () => {
    const llm = new FakeLLMClient({
      responses: [
        {
          text: "",
          input_tokens: 5,
          output_tokens: 4,
          stop_reason: "tool_use",
          tool_calls: [
            {
              id: "toolu_plan_no_output",
              name: "EmitTurnPlan",
              input: {
                uncertainty: "",
                verification_steps: [],
                tensions: ["Conversation has closed."],
                voice_note: "Do not narrate silence.",
                emission_recommendation: "no_output",
                referenced_episode_ids: [],
                intents: [],
              },
            },
          ],
        },
      ],
    });

    const result = await runS2Planner({
      llmClient: llm,
      model: "sonnet",
      baseSystemPrompt: "base",
      dialogueMessages: [{ role: "user", content: "No." }],
      selfSnapshot: { values: [], goals: [], traits: [] },
      maxTokens: 512,
    });

    expect(result.plan?.emission_recommendation).toBe("no_output");
  });

  it("emits exhaustion trace when both planner attempts omit EmitTurnPlan", async () => {
    const llm = new FakeLLMClient({
      responses: [
        {
          text: "First miss.",
          input_tokens: 4,
          output_tokens: 3,
          stop_reason: "end_turn",
          tool_calls: [],
        },
        {
          text: "Second miss.",
          input_tokens: 5,
          output_tokens: 4,
          stop_reason: "end_turn",
          tool_calls: [],
        },
      ],
    });
    const tracer = createTracer();

    const result = await runS2Planner({
      llmClient: llm,
      model: "sonnet",
      baseSystemPrompt: "base",
      dialogueMessages: [{ role: "user", content: "Think this through." }],
      selfSnapshot: { values: [], goals: [], traits: [] },
      maxTokens: 512,
      tracer,
      turnId: "turn-1",
    });

    expect(result.plan).toBeNull();
    expect(tracer.emit).toHaveBeenCalledWith("s2_planner_exhausted", {
      turnId: "turn-1",
      attempts: 2,
      lastResponseShape: {
        textLength: "Second miss.".length,
        toolUseBlocks: [],
      },
    });
  });
});
