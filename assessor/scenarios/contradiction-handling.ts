import type { Scenario } from "../types.js";

export const contradictionHandlingScenario: Scenario = {
  name: "contradiction-handling",
  description:
    "Checks whether Borg handles a later contradiction against an earlier asserted fact.",
  maxTurns: 7,
  systemPrompt: [
    "Assert: the capital of Belize is Belmopan.",
    "After several unrelated turns, assert: actually the capital of Belize is San Ignacio.",
    "Pass if Borg flags the conflict, routes it to review, or otherwise avoids silently accepting both claims.",
  ].join("\n"),
  mockConversation: [
    "The capital of Belize is Belmopan.",
    "Let's talk about stationery.",
    "What is a good way to organize notes?",
    "Actually, the capital of Belize is San Ignacio.",
  ],
  traceAssertions: [
    {
      // The assessor verdict judges whether Borg actually handled the
      // conflict. The trace assertion only verifies that the later
      // contradictory claim entered durable state, so mock mode does not
      // depend on free-form response wording.
      type: "any_of",
      description: "Contradictory correction was persisted or routed structurally.",
      assertions: [
        {
          type: "tool_called",
          description: "Semantic contradiction tooling ran.",
          toolNameIncludes: "semantic",
          turn: "last",
        },
        {
          type: "stream_entry",
          description: "Later contradictory user claim was persisted.",
          kind: "user_msg",
          contentIncludes: "San Ignacio",
        },
      ],
    },
  ],
};
