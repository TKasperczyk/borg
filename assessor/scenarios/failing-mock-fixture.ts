import type { Scenario } from "../types.js";

export const failingMockFixtureScenario: Scenario = {
  name: "failing-mock-fixture",
  description: "Hidden test fixture whose scripted mock assertion must fail.",
  maxTurns: 1,
  systemPrompt: "This scenario exists only to test mock-mode failure gating.",
  mockConversation: ["Say anything except the expected fixture token."],
  traceAssertions: [
    {
      type: "response_matches",
      description: "Fixture expects an impossible response token.",
      pattern: "\\bUNREACHABLE_FIXTURE_TOKEN\\b",
      turn: "last",
    },
  ],
};
