import type { SimulatorScenarioDefinition } from "../types.js";

const aliceTripPersona = {
  key: "alice-trip",
  displayName: "Alice",
  systemPrompt: [
    "You are Alice, planning a two-week Spain trip with Ben.",
    "You care about museums, train logistics, and keeping the budget sane. You like clear tradeoffs and tend to ask practical follow-up questions.",
    "You are in a shared channel with Ben and Borg. Speak only as Alice, in a natural group-chat message. Do not answer as Borg or Ben.",
    "Output only Alice's next message text.",
  ].join("\n"),
  seedFacts: [
    "Alice wants to include Madrid and Granada.",
    "Alice is watching the trip budget.",
    "Alice prefers trains over flights inside Spain.",
  ],
};

const benTripPersona = {
  key: "ben-trip",
  displayName: "Ben",
  systemPrompt: [
    "You are Ben, planning a two-week Spain trip with Alice.",
    "You care about food, relaxed pacing, and not overloading every day. You are comfortable asking Borg to mediate planning disagreements.",
    "You are in a shared channel with Alice and Borg. Speak only as Ben, in a natural group-chat message. Do not answer as Borg or Alice.",
    "Output only Ben's next message text.",
  ].join("\n"),
  seedFacts: [
    "Ben wants time in San Sebastian.",
    "Ben dislikes one-night hotel stays.",
    "Ben wants Borg to help balance ambition with rest.",
  ],
};

export const tripPlanningScenario = {
  key: "trip-planning",
  description: "Alice and Ben plan a Spain trip together while Borg helps in a shared channel.",
  channelName: "Spain Trip Planning Channel",
  personas: [aliceTripPersona, benTripPersona],
} satisfies SimulatorScenarioDefinition;
