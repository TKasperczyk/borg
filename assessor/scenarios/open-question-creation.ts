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
  // No structural assertion: Borg can plainly admit ignorance or name an unresolved question for reflection to capture after the response.
};
