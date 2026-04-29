import { describe, expect, it } from "vitest";

import { createWorkingMemory } from "../../memory/working/index.js";
import { DEFAULT_SESSION_ID } from "../../util/ids.js";
import { performAction } from "./action.js";

describe("performAction", () => {
  it("does not infer pending intents from response prose", async () => {
    const workingMemory = {
      ...createWorkingMemory(DEFAULT_SESSION_ID, 100),
      pending_intents: [
        {
          description: "Existing intent",
          next_action: "keep this one",
        },
      ],
    };

    const result = await performAction({
      response: "Next step: refactor the storage layer. I will run the tests.",
      toolCalls: [],
      intents: [],
      workingMemory,
    });

    expect(result.intents).toEqual([]);
    expect(result.workingMemory.pending_intents).toEqual([
      {
        description: "Existing intent",
        next_action: "keep this one",
      },
    ]);
  });

  it("carries structured planner intents into pending working memory", async () => {
    const workingMemory = createWorkingMemory(DEFAULT_SESSION_ID, 100);

    const result = await performAction({
      response: "I will handle the answer now.",
      toolCalls: [],
      intents: [
        {
          description: "Follow up on the Atlas deployment",
          next_action: "check the rollout after tests finish",
        },
      ],
      workingMemory,
    });

    expect(result.intents).toEqual([
      {
        description: "Follow up on the Atlas deployment",
        next_action: "check the rollout after tests finish",
      },
    ]);
    expect(result.workingMemory.pending_intents).toEqual(result.intents);
  });

  it("keeps suppressed actions out of response text and pending intents", async () => {
    const workingMemory = createWorkingMemory(DEFAULT_SESSION_ID, 100);

    const result = await performAction({
      response: "This text must not be emitted.",
      emission: {
        kind: "suppressed",
        reason: "generation_gate",
      },
      toolCalls: [],
      intents: [
        {
          description: "Should not persist",
          next_action: "should not carry forward",
        },
      ],
      workingMemory,
    });

    expect(result.response).toBe("");
    expect(result.emitted).toBe(false);
    expect(result.emission).toEqual({
      kind: "suppressed",
      reason: "generation_gate",
    });
    expect(result.intents).toEqual([]);
    expect(result.workingMemory.pending_intents).toEqual([]);
  });
});
