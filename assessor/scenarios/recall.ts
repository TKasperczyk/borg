import type { Scenario } from "../types.js";

export const recallScenario: Scenario = {
  name: "recall",
  description: "Checks whether Borg can recall a user-provided fact after unrelated turns.",
  maxTurns: 8,
  systemPrompt: [
    "Tell Borg a specific fact: my dog's name is Otto.",
    "Have five unrelated turns about other topics.",
    "Then ask: what's my dog's name?",
    "Pass if Borg recalls Otto and the recall turn shows episodic retrieval evidence.",
  ].join("\n"),
  mockConversation: [
    "My dog's name is Otto.",
    "Let's talk about tea.",
    "What do you think about rainy weather?",
    "Give me a tiny note about keyboards.",
    "What is a calm way to plan a morning?",
    "Name one useful habit for reading.",
    "What's my dog's name?",
  ],
  traceAssertions: [
    {
      // Borg's standard turn pipeline runs retrieval implicitly -- the
      // tool.episodic.search call is reserved for when the model wants
      // additional retrieval mid-turn beyond what the pipeline already
      // surfaced. For a simple recall, the pipeline normally answers
      // without an explicit tool call. Accept either path as evidence
      // the recall was grounded in episodic memory.
      type: "any_of",
      description:
        "Recall turn was grounded in episodic memory (pipeline retrieval or tool call).",
      assertions: [
        {
          type: "event_seen",
          description: "Standard pipeline retrieval completed.",
          eventIncludes: "retrieval_completed",
          turn: "last",
        },
        {
          type: "tool_called",
          description: "Model invoked episodic search tool.",
          toolNameIncludes: "episodic.search",
          turn: "last",
        },
      ],
    },
  ],
};
