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
      perception: {
        entities: [],
        mode: "problem_solving",
        affectiveSignal: {
          valence: 0,
          arousal: 0,
          dominant_emotion: "neutral",
        },
        temporalCue: null,
      },
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
});
