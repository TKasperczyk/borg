import type { Scenario } from "../types.js";

export const goalProgressTrackingScenario: Scenario = {
  name: "goal-progress-tracking",
  description: "Checks whether progress against an active goal is attributed and reflected.",
  maxTurns: 6,
  seedGoals: [
    {
      key: "dinner-party",
      description: "Help the user prepare for a dinner party next week.",
      priority: 10,
    },
  ],
  systemPrompt: [
    "A dinner-party preparation goal is already seeded in Borg self-memory for this scenario.",
    "Reference that goal conversationally: help me prepare for a dinner party next week.",
    "Later, mention a concrete progress step: I bought wine.",
    "Pass if Borg tracks that progress on the seeded goal, including last_progress_ts and progress notes.",
  ].join("\n"),
  mockConversation: [
    "I have a goal: help me prepare for a dinner party next week.",
    "I bought wine for the dinner party.",
  ],
  traceAssertions: [
    {
      type: "goal_progress",
      description: "Seeded dinner-party goal records concrete progress.",
      goalKey: "dinner-party",
      progressIncludes: "bought wine",
    },
  ],
};
