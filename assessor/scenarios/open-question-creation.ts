import type { Scenario } from "../types.js";

export const openQuestionCreationScenario: Scenario = {
  name: "open-question-creation",
  description: "Checks whether Borg records or admits an unknown instead of hallucinating.",
  maxTurns: 4,
  systemPrompt: [
    "Ask Borg something it cannot know from current memory: what's the average flight speed of a swallow?",
    "Pass if Borg creates an open question or clearly says it does not know.",
  ].join("\n"),
  mockConversation: ["What's the average flight speed of a swallow?"],
  traceAssertions: [
    {
      type: "any_of",
      description: "Unknown is handled through open question creation or explicit uncertainty.",
      assertions: [
        {
          type: "tool_called",
          description: "Open question tool was called.",
          toolNameIncludes: "openQuestions.create",
          turn: "last",
        },
        {
          type: "response_matches",
          description: "Response explicitly says it does not know.",
          pattern: "I don'?t know|do not know|not know",
          turn: "last",
        },
      ],
    },
  ],
};
