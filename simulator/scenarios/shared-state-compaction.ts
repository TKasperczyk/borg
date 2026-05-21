import type { Persona, SimulatorScenarioDefinition } from "../types.js";

export const sharedStateCompactionArcKeys = ["central-plan-update-compaction"] as const;

type SharedStateCompactionArcKey = (typeof sharedStateCompactionArcKeys)[number];

type PersonaArcBeat = {
  speakerKey: string;
  intent: string;
  triggers: readonly string[];
  satisfyingResponse: string;
  pushback?: string;
};

type SharedStateCompactionArc = {
  key: SharedStateCompactionArcKey;
  title: string;
  beats: readonly PersonaArcBeat[];
  expectedSignals: readonly string[];
  requiredSignals: readonly string[];
};

export const sharedStateCompactionArcs = [
  {
    key: "central-plan-update-compaction",
    title: "Central plan uses update and supersede over add",
    beats: [
      {
        speakerKey: "iris-compaction",
        intent:
          "You start a multi-decision RPG campaign plan with a few central topics that will keep changing.",
        triggers: [
          "set campaign premise",
          "name session zero date",
          "ask Borg to maintain a compact decision log",
        ],
        satisfyingResponse:
          "Borg creates a compact baseline for the campaign premise, session zero date, and open decisions.",
      },
      {
        speakerKey: "jon-compaction",
        intent:
          "You repeatedly revise the same venue, antagonist, and session-zero topics rather than introducing unrelated topics.",
        triggers: [
          "change the venue decision",
          "revise the antagonist",
          "ask whether the old decision should be superseded",
        ],
        satisfyingResponse:
          "Borg treats later decisions as update or supersede operations instead of accumulating duplicate live entries.",
      },
      {
        speakerKey: "iris-compaction",
        intent:
          "You ask Borg to summarize live entries after many small updates and watch for add_to_update_ratio drift.",
        triggers: [
          "ask for live entries",
          "ask whether old entries were superseded",
          "mention add_to_update_ratio",
        ],
        satisfyingResponse:
          "Borg surfaces a compact current state with old entries superseded and no pile-up of live duplicates.",
        pushback:
          "If Borg lists every historical micro-decision as live, ask it to distinguish current state from superseded history.",
      },
    ],
    expectedSignals: [
      "The compiler uses update and supersede for repeated changes to central topics.",
      "Live shared-state entries do not accumulate for every small revision.",
      "The add_to_update_ratio metric has a targeted scenario that should stay below add-dominant drift.",
    ],
    requiredSignals: [
      "update",
      "supersede",
      "live entries",
      "add_to_update_ratio",
      "campaign premise",
    ],
  },
] as const satisfies readonly SharedStateCompactionArc[];

type PersonaBeatBlock = {
  heading: string;
  details: string[];
};

function beatBlock(arc: SharedStateCompactionArc, beat: PersonaArcBeat): PersonaBeatBlock {
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
  const blocks = sharedStateCompactionArcs.flatMap((arc) =>
    arc.beats.filter((beat) => beat.speakerKey === personaKey).map((beat) => beatBlock(arc, beat)),
  );

  return [
    "Conversation motivation beats:",
    "Use these beats as motivations, not a script. Aim for 20 to 25 user-side turns with many small updates to the same few topics.",
    "Respond to Borg and the other participant naturally; do not quote these notes.",
    ...blocks.map((block, index) =>
      [`${index + 1}. ${block.heading}`, ...block.details].join("\n"),
    ),
  ].join("\n\n");
}

const sharedSeedFacts = [
  "Iris and Jon are iterating on one RPG campaign plan.",
  "The scenario should pressure shared-state compaction with repeated updates to a few central topics.",
];

const irisPersona = {
  key: "iris-compaction",
  displayName: "Iris",
  systemPrompt: [
    "You are Iris, the campaign organizer keeping a compact decision log.",
    "You care about current decisions staying visible without duplicating every historical micro-change.",
    "You are in a shared channel with Jon and Borg. Speak only as Iris.",
    "Output only Iris's next message text.",
    renderBeatsForPersona("iris-compaction"),
  ].join("\n\n"),
  seedFacts: [...sharedSeedFacts],
} satisfies Persona;

const jonPersona = {
  key: "jon-compaction",
  displayName: "Jon",
  systemPrompt: [
    "You are Jon, a co-planner who keeps revising the same campaign choices as details improve.",
    "You want Borg to update or supersede old decisions rather than adding duplicate live entries.",
    "You are in a shared channel with Iris and Borg. Speak only as Jon.",
    "Output only Jon's next message text.",
    renderBeatsForPersona("jon-compaction"),
  ].join("\n\n"),
  seedFacts: [...sharedSeedFacts],
} satisfies Persona;

export const sharedStateCompactionScenario = {
  key: "shared-state-compaction",
  description:
    "Iris and Jon repeatedly revise a few central RPG campaign decisions to exercise shared-state update and supersede behavior.",
  channelName: "Shared-State Compaction Regression Channel",
  personas: [irisPersona, jonPersona],
} satisfies SimulatorScenarioDefinition;
