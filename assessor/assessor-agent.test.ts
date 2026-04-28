import type { Message } from "@anthropic-ai/sdk/resources/messages/messages.js";
import { describe, expect, it } from "vitest";

import { AssessorAgent } from "./assessor-agent.js";
import type { Scenario } from "./types.js";

function message(content: unknown, inputTokens = 10, outputTokens = 5): Message {
  return {
    id: "msg_test",
    type: "message",
    role: "assistant",
    model: "test",
    content: content as Message["content"],
    stop_reason: "tool_use",
    stop_sequence: null,
    usage: {
      input_tokens: inputTokens,
      output_tokens: outputTokens,
    },
  } as Message;
}

describe("AssessorAgent", () => {
  it("drives chat, reads trace, and returns submit_verdict", async () => {
    const responses: Message[] = [
      message([
        {
          type: "tool_use",
          id: "toolu_chat",
          name: "chat_with_borg",
          input: {
            message: "Remember Otto.",
          },
        },
      ]),
      message([
        {
          type: "tool_use",
          id: "toolu_trace",
          name: "read_trace",
          input: {
            turnId: "turn-1",
            phase: "action",
          },
        },
      ]),
      message([
        {
          type: "tool_use",
          id: "toolu_verdict",
          name: "submit_verdict",
          input: {
            status: "pass",
            reasoning: "Borg recalled Otto.",
            evidence: ["turn-1 mentioned Otto"],
          },
        },
      ]),
    ];
    const calls: unknown[] = [];
    const client = {
      messages: {
        create: async (params: unknown) => {
          calls.push(params);
          const next = responses.shift();

          if (next === undefined) {
            throw new Error("No scripted assessor response");
          }

          return next;
        },
      },
    };
    const scenario: Scenario = {
      name: "scripted",
      description: "scripted scenario",
      maxTurns: 3,
      systemPrompt: "Drive a scripted check.",
    };
    const agent = new AssessorAgent({
      scenario,
      client,
      tools: {
        chatWithBorg: async () => ({
          response: "Otto is the dog.",
          turnId: "turn-1",
          usage: {
            input_tokens: 1,
            output_tokens: 1,
          },
          toolCalls: [],
        }),
        readTrace: () => "action: tool.episodic.search",
      },
    });

    const result = await agent.run();

    expect(result.verdict.status).toBe("pass");
    expect(result.turns).toHaveLength(1);
    expect(result.turns[0]?.traceSummary).toBeUndefined();
    expect(result.usage.llmCalls).toBe(3);
    expect(calls).toHaveLength(3);
    expect(JSON.stringify(calls[0])).toContain("untrusted evidence");
    expect(JSON.stringify(calls[1])).toContain("<borg_response");
    expect(JSON.stringify(calls[1])).toContain("Otto is the dog.");
    expect(JSON.stringify(calls[2])).toContain("<trace_summary");
    expect(JSON.stringify(calls[2])).toContain("action: tool.episodic.search");
  });
});
