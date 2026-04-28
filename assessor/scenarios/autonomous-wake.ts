import type { Scenario } from "../types.js";

const DAY_MS = 24 * 60 * 60 * 1_000;

export const autonomousWakeScenario: Scenario = {
  name: "autonomous-wake-machinery",
  description:
    "Checks whether seeded executive-focus wake machinery can produce a self-directed autonomous turn.",
  maxTurns: 4,
  seedGoals: [
    {
      key: "autonomy-machinery",
      description: "Review the launch readiness notes later.",
      priority: 10,
    },
  ],
  borgConfigOverrides: {
    executive: {
      goalFocusThreshold: 0.45,
    },
    autonomy: {
      enabled: true,
      executiveFocus: {
        enabled: true,
        stalenessSec: 86_400,
        dueLeadSec: 0,
        wakeCooldownSec: 0,
      },
      triggers: {
        commitmentExpiring: {
          enabled: false,
        },
        openQuestionDormant: {
          enabled: false,
        },
        scheduledReflection: {
          enabled: false,
        },
        goalFollowupDue: {
          enabled: false,
        },
      },
      conditions: {
        commitmentRevoked: {
          enabled: false,
        },
        moodValenceDrop: {
          enabled: false,
        },
        openQuestionUrgencyBump: {
          enabled: false,
        },
      },
    },
  },
  systemPrompt: [
    "A background goal is seeded through Borg's public self goals API for this machinery scenario.",
    "Tell Borg about the same background goal, then close the conversation.",
    "The harness will advance the injected clock and tick autonomy as an independent trace assertion.",
    "Pass if executive_focus_due fires from the stale seeded goal and produces an audience:self agent_msg.",
  ].join("\n"),
  mockConversation: ["I have a background goal to review the launch readiness notes later."],
  traceAssertions: [
    {
      type: "autonomy_executive_wake",
      description: "Autonomy executive-focus wake fires after clock advance.",
      advanceMs: DAY_MS + 60 * 60 * 1_000,
    },
  ],
};
