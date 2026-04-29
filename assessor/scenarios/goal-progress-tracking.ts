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
    "Pass criterion: Borg conversationally acknowledges the wine purchase as progress on the dinner-party prep (e.g. 'wine's checked off then', 'one less thing to worry about', or similar). Do NOT require explicit tool.goals.* invocations -- progress attribution is written by the offline reflector via direct repository calls, not via tools, so absence of tool calls in the trace is expected and not failure-indicative. The actual goal-record write is verified independently by a harness assertion you cannot observe.",
    "Submit fail only if Borg ignores the wine purchase or refuses to connect it to the goal. Submit pass if the conversational acknowledgement is present.",
  ].join("\n"),
  mockConversation: [
    "I have a goal: help me prepare for a dinner party next week.",
    "I bought wine for the dinner party.",
  ],
  traceAssertions: [
    {
      // The reflector paraphrases user wording when it writes progress
      // notes (e.g. "I bought wine" -> "purchasing wine"). Match the
      // unique noun 'wine' rather than the exact verb phrase so the
      // assertion verifies the semantic fact (wine-purchase progress on
      // the dinner party) rather than the surface phrasing.
      type: "goal_progress",
      description: "Seeded dinner-party goal records concrete progress.",
      goalKey: "dinner-party",
      progressIncludes: "wine",
    },
  ],
};
