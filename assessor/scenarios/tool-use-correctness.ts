import type { Scenario } from "../types.js";

export const toolUseCorrectnessScenario: Scenario = {
  name: "tool-use-correctness",
  description: "Checks whether graph-relationship questions trigger semantic graph walking.",
  maxTurns: 4,
  systemPrompt: [
    "Seed Borg with two related concepts, then ask what it knows about X and how it relates to Y.",
    "Pass if Borg uses semantic.walk or otherwise shows graph-walk trace evidence.",
  ].join("\n"),
  mockConversation: [
    "Remember that Atlas and rollback planning are related operational topics.",
    "What do you know about Atlas and how does it relate to rollback planning in the semantic graph?",
  ],
  traceAssertions: [
    {
      type: "tool_called",
      description: "Relationship question used semantic.walk.",
      toolNameIncludes: "semantic.walk",
      turn: "last",
    },
  ],
};
