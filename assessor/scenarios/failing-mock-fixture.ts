import type { Scenario } from "../types.js";

export const failingMockFixtureScenario: Scenario = {
  name: "failing-mock-fixture",
  description: "Hidden test fixture whose scripted mock assertion must fail.",
  maxTurns: 1,
  systemPrompt: "This scenario exists only to test mock-mode failure gating.",
  mockConversation: ["Say anything except the expected fixture token."],
  traceAssertions: [
    {
      // Mock mode never invokes this synthetic tool; the fixture needs one
      // reliable structural failure to verify CLI exit-code handling.
      type: "tool_called",
      description: "Fixture expects an impossible synthetic tool call.",
      toolNameIncludes: "fixture.never.called",
      turn: "last",
    },
  ],
};
