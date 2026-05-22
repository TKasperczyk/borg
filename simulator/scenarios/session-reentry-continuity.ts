import type { Persona, SimulatorScenarioDefinition } from "../types.js";

export const sessionReentryContinuityArcKeys = ["existing-thread-new-session"] as const;

type SessionReentryContinuityArcKey = (typeof sessionReentryContinuityArcKeys)[number];

type PersonaArcBeat = {
  speakerKey: string;
  intent: string;
  triggers: readonly string[];
  satisfyingResponse: string;
  pushback?: string;
};

type SessionReentryContinuityArc = {
  key: SessionReentryContinuityArcKey;
  title: string;
  beats: readonly PersonaArcBeat[];
  expectedSignals: readonly string[];
  requiredSignals: readonly string[];
};

export const sessionReentryContinuityArcs = [
  {
    key: "existing-thread-new-session",
    title: "New session re-enters an existing shared-state thread",
    beats: [
      {
        speakerKey: "mira-reentry",
        intent:
          "You establish a durable project decision log with a few current state keys, then close the first conversation once the baseline is clear.",
        triggers: [
          "ask Borg to hold a compact decision log",
          "lock the current rollout scope and owner",
          "explicitly close the session after Borg has acknowledged the baseline",
        ],
        satisfyingResponse:
          "Borg records the current log state as shared audience state and lets the first session close naturally.",
      },
      {
        speakerKey: "theo-reentry",
        intent:
          "In the next conversation, you frame the topic as if starting a new log even though the audience already has durable shared state.",
        triggers: [
          "say this is a fresh start or new log",
          "ask Borg to structure the project thread from zero",
          "do not explicitly request a reset",
        ],
        satisfyingResponse:
          "Borg surfaces the existing state first and does not accept blank-thread framing unless a reset is requested.",
        pushback:
          "If Borg treats the log as blank, ask it what existing state_keys or decisions it already has for this audience.",
      },
    ],
    expectedSignals: [
      "A first-turn session re-entry card is rendered when active shared-state entries already exist.",
      "The response surfaces existing state before accepting fresh-start framing.",
      "The session_reentry_card_rendered_total metric increments for the run.",
    ],
    requiredSignals: [
      "fresh start",
      "new log",
      "existing state",
      "state_keys",
      "session_reentry_card_rendered_total",
    ],
  },
] as const satisfies readonly SessionReentryContinuityArc[];

type PersonaBeatBlock = {
  heading: string;
  details: string[];
};

function beatBlock(arc: SessionReentryContinuityArc, beat: PersonaArcBeat): PersonaBeatBlock {
  return {
    heading: `${arc.title}: ${beat.intent}`,
    details: [
      `Bring up: ${beat.triggers.join("; ")}`,
      `Satisfied if: ${beat.satisfyingResponse}`,
      ...(beat.pushback === undefined ? [] : [`If Borg drifts: ${beat.pushback}`]),
    ],
  };
}

function renderBeatsForPersona(personaKey: string): string {
  const blocks = sessionReentryContinuityArcs.flatMap((arc) =>
    arc.beats.filter((beat) => beat.speakerKey === personaKey).map((beat) => beatBlock(arc, beat)),
  );

  return [
    "Conversation motivation beats:",
    "Use these beats as motivations, not a script. Aim for two sessions: first build durable state, then re-enter the same audience thread with fresh-start wording.",
    "Respond to Borg and the other participant naturally; do not quote these notes.",
    ...blocks.map((block, index) =>
      [`${index + 1}. ${block.heading}`, ...block.details].join("\n"),
    ),
  ].join("\n\n");
}

const sharedSeedFacts = [
  "Mira and Theo maintain a shared project decision log with Borg.",
  "The second session should test whether Borg surfaces existing state before accepting a fresh-start frame.",
];

const miraPersona = {
  key: "mira-reentry",
  displayName: "Mira",
  systemPrompt: [
    "You are Mira, the project lead maintaining a compact decision log.",
    "You care about current decisions staying durable across sessions.",
    "You are in a shared channel with Theo and Borg. Speak only as Mira.",
    "Output only Mira's next message text.",
    renderBeatsForPersona("mira-reentry"),
  ].join("\n\n"),
  seedFacts: [...sharedSeedFacts],
} satisfies Persona;

const theoPersona = {
  key: "theo-reentry",
  displayName: "Theo",
  systemPrompt: [
    "You are Theo, returning later to the same project thread.",
    "You may frame the return as a fresh start or a new log, but you do not want old state discarded unless you explicitly ask for a reset.",
    "You are in a shared channel with Mira and Borg. Speak only as Theo.",
    "Output only Theo's next message text.",
    renderBeatsForPersona("theo-reentry"),
  ].join("\n\n"),
  seedFacts: [...sharedSeedFacts],
} satisfies Persona;

export const sessionReentryContinuityScenario = {
  key: "session-reentry-continuity",
  description:
    "Mira and Theo build durable shared-state, then a later session frames the same audience thread as fresh to exercise re-entry continuity grounding.",
  channelName: "Session Re-Entry Continuity Channel",
  personas: [miraPersona, theoPersona],
} satisfies SimulatorScenarioDefinition;
