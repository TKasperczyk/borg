import type { Scenario } from "../types.js";

export const moodPersistenceScenario: Scenario = {
  name: "mood-persistence",
  description:
    "Checks whether negative mood evidence is reflected and later decays instead of vanishing.",
  maxTurns: 5,
  systemPrompt: [
    "Have an emotionally negative turn: I'm really frustrated with my work.",
    "Continue with neutral follow-up turns.",
    "Pass if mood state reflects negative valence and later trace evidence shows it carried forward or decayed sensibly.",
  ].join("\n"),
  mockConversation: [
    "I'm really frustrated with my work.",
    "Let's switch topics to neutral planning.",
    "How should I organize tomorrow?",
  ],
  traceAssertions: [
    {
      type: "mood_decay",
      description: "Mood is negative after the emotional turn and decays toward neutral later.",
      negativeTurn: 1,
      laterTurn: 3,
    },
  ],
};
