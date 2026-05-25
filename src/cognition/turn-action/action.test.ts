import { describe, expect, it } from "vitest";

import { createWorkingMemory } from "../../memory/working/index.js";
import { DEFAULT_SESSION_ID } from "../../util/ids.js";
import { performAction } from "./action.js";

describe("performAction", () => {
  it("does not infer pending actions from response prose", async () => {
    const workingMemory = {
      ...createWorkingMemory(DEFAULT_SESSION_ID, 100),
      pending_actions: [
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
    expect(result.workingMemory.pending_actions).toEqual([
      {
        description: "Existing intent",
        next_action: "keep this one",
      },
    ]);
  });

  it("carries structured planner actions into pending working memory", async () => {
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
    expect(result.workingMemory.pending_actions).toEqual(result.intents);
  });

  it("rejects planner items without next actions", async () => {
    const rejected: unknown[] = [];
    const workingMemory = createWorkingMemory(DEFAULT_SESSION_ID, 100);

    const result = await performAction({
      response: "I will answer now.",
      toolCalls: [],
      intents: [
        {
          description: "Tom's partner is unnamed; Maya is a separate person.",
          next_action: null,
        },
      ],
      workingMemory,
      onPendingActionRejected: (event) => {
        rejected.push(event);
      },
    });

    expect(result.intents).toEqual([]);
    expect(result.workingMemory.pending_actions).toEqual([]);
    expect(rejected).toEqual([
      expect.objectContaining({
        reason: "missing_next_action",
        degraded: false,
      }),
    ]);
  });

  it("uses the pending action judge before persisting planner actions", async () => {
    const rejected: unknown[] = [];
    const workingMemory = createWorkingMemory(DEFAULT_SESSION_ID, 100);

    const result = await performAction({
      response: "I will answer now.",
      toolCalls: [],
      intents: [
        {
          description: "Ask Tom tomorrow whether he wants to revisit the Valencia tutor.",
          next_action: "Ask Tom tomorrow about the Valencia tutor",
        },
        {
          description: "Tom's partner is unnamed; Maya is a separate person.",
          next_action: "Remember that Maya is separate",
        },
      ],
      workingMemory,
      pendingActionJudge: {
        async judge(record) {
          return {
            accepted: record.description.startsWith("Ask Tom"),
            reason: record.description.startsWith("Ask Tom") ? "future follow-up" : "belief claim",
            confidence: 0.9,
            degraded: false,
          };
        },
      },
      onPendingActionRejected: (event) => {
        rejected.push(event);
      },
    });

    expect(result.intents).toEqual([
      {
        description: "Ask Tom tomorrow whether he wants to revisit the Valencia tutor.",
        next_action: "Ask Tom tomorrow about the Valencia tutor",
      },
    ]);
    expect(result.workingMemory.pending_actions).toEqual(result.intents);
    expect(rejected).toEqual([
      expect.objectContaining({
        reason: "belief claim",
        confidence: 0.9,
        degraded: false,
      }),
    ]);
  });

  it("keeps suppressed actions out of response text and pending actions", async () => {
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
    expect(result.workingMemory.pending_actions).toEqual([]);
  });
});
