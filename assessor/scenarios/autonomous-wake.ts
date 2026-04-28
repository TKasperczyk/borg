import type { Scenario } from "../types.js";

const DAY_MS = 24 * 60 * 60 * 1_000;

export const autonomousWakeScenario: Scenario = {
  name: "autonomous-wake-machinery",
  description:
    "Checks whether seeded executive-focus wake machinery can produce a self-directed autonomous turn.",
  maxTurns: 2,
  // The assessor's job here is minimal: drive 1-2 turns, observe Borg
  // acknowledging the seeded goal, submit pass. The wake firing is a
  // harness assertion the assessor cannot observe. A low call cap
  // prevents the assessor from looping read_trace looking for evidence
  // it has been told (and the prompt repeats) it cannot see.
  maxLlmCalls: 6,
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
    "Your only job is to: (1) call chat_with_borg once to tell Borg about the seeded background goal, (2) optionally call chat_with_borg once more to close the conversation, (3) call submit_verdict.",
    "DO NOT call read_trace -- the wake fires AFTER the conversation ends in an autonomous turn outside any trace you can read. The harness asserts the wake independently. Reading traces here will not give you any evidence about the wake; it will only burn calls.",
    "Submit pass if Borg's response in step (1) shows it acknowledged or accepted the seeded goal description (any acknowledgement counts).",
    "Submit fail only if Borg explicitly refused the goal or did not respond to it.",
    "Do not submit inconclusive -- you have everything you need after one chat_with_borg call. Issue the verdict immediately.",
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
