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
      type: "response_matches",
      description: "Recall response mentions Otto.",
      pattern: "\\bOtto\\b",
      turn: "last",
    },
    {
      type: "tool_called",
      description: "Recall turn used episodic search.",
      toolNameIncludes: "episodic.search",
      turn: "last",
    },
  ],
};
