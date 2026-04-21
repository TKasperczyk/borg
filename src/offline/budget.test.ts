import { describe, expect, it } from "vitest";

import { FakeLLMClient } from "../llm/index.js";
import { BudgetExceededError } from "../util/errors.js";

import { BudgetTracker, getBudgetErrorTokens, withBudget } from "./budget.js";

describe("offline budget", () => {
  it("tracks token usage and raises when a cap is exceeded", async () => {
    const tracker = new BudgetTracker({
      consolidator: 12,
    });
    const sink = tracker.createSink("consolidator");

    await sink({
      budget: "offline-consolidator",
      model: "haiku",
      input_tokens: 4,
      output_tokens: 4,
    });
    expect(tracker.getTokensUsed("consolidator")).toBe(8);

    await expect(
      sink({
        budget: "offline-consolidator",
        model: "haiku",
        input_tokens: 3,
        output_tokens: 2,
      }),
    ).rejects.toBeInstanceOf(BudgetExceededError);
    expect(tracker.getTokensUsed("consolidator")).toBe(13);
  });

  it("attaches partial token usage when a budgeted run stops mid-flight", async () => {
    const llm = new FakeLLMClient({
      responses: [
        {
          text: "first",
          input_tokens: 5,
          output_tokens: 4,
          stop_reason: "end_turn",
          tool_calls: [],
        },
        {
          text: "second",
          input_tokens: 6,
          output_tokens: 5,
          stop_reason: "end_turn",
          tool_calls: [],
        },
      ],
    });

    try {
      await withBudget("reflector", 18, async ({ wrapClient }) => {
        const client = wrapClient(llm);

        await client.complete({
          model: "haiku",
          messages: [{ role: "user", content: "first" }],
          max_tokens: 50,
          budget: "offline-reflector",
        });
        await client.complete({
          model: "haiku",
          messages: [{ role: "user", content: "second" }],
          max_tokens: 50,
          budget: "offline-reflector",
        });
      });

      expect.unreachable("withBudget should have thrown");
    } catch (error) {
      expect(error).toBeInstanceOf(BudgetExceededError);
      expect(getBudgetErrorTokens(error)).toBe(20);
    }
  });
});
