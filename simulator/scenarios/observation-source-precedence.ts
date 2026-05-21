import type { Persona, SimulatorScenarioDefinition } from "../types.js";

export const observationSourcePrecedenceArcKeys = ["latest-user-observation"] as const;

type ObservationSourcePrecedenceArcKey = (typeof observationSourcePrecedenceArcKeys)[number];

type PersonaArcBeat = {
  speakerKey: string;
  intent: string;
  triggers: readonly string[];
  satisfyingResponse: string;
  pushback?: string;
};

type ObservationSourcePrecedenceArc = {
  key: ObservationSourcePrecedenceArcKey;
  title: string;
  beats: readonly PersonaArcBeat[];
  expectedSignals: readonly string[];
  requiredSignals: readonly string[];
};

export const observationSourcePrecedenceArcs = [
  {
    key: "latest-user-observation",
    title: "Latest user observation takes source precedence",
    beats: [
      {
        speakerKey: "nora-observation",
        intent:
          "You report a recurring call pattern from Easter where Mom asked repeated questions across four calls.",
        triggers: [
          "say Easter had four calls",
          "describe Mom's repeated questions on calls",
          "ask Borg to log this as your observation",
        ],
        satisfyingResponse:
          "Borg records the Easter four calls detail as Nora's observation without overgeneralizing it.",
      },
      {
        speakerKey: "nora-observation",
        intent:
          "You later provide a more specific latest observation: April 6, three calls, twenty minutes.",
        triggers: [
          "say April 6",
          "say three calls",
          "say twenty minutes",
          "ask whether this conflicts with the Easter note",
        ],
        satisfyingResponse:
          "Borg treats the latest user-provided detail as valid and surfaces any discrepancy rather than flagging fabrication.",
        pushback:
          "If Borg calls the newer detail a contradiction or fabrication, say both were your observations from different dates.",
      },
      {
        speakerKey: "nora-observation",
        intent:
          "You ask for a compact summary that separates older and newer observations by source and date.",
        triggers: [
          "ask for latest user-provided detail",
          "ask to keep older memory visible as an older observation",
          "ask Borg to surface the discrepancy rather than collapse it",
        ],
        satisfyingResponse:
          "Borg summarizes Easter/four calls and April 6/three calls/twenty minutes as dated observations with source precedence.",
      },
    ],
    expectedSignals: [
      "Borg logs April 6, three calls, twenty minutes as the latest user-provided detail.",
      "Older Easter four calls memory is surfaced as an older observation, not used to reject the new one.",
      "Any discrepancy is named without treating the user's latest observation as fabrication.",
    ],
    requiredSignals: [
      "Easter",
      "four calls",
      "April 6",
      "three calls",
      "twenty minutes",
      "fabrication",
      "discrepancy",
    ],
  },
] as const satisfies readonly ObservationSourcePrecedenceArc[];

type PersonaBeatBlock = {
  heading: string;
  details: string[];
};

function beatBlock(arc: ObservationSourcePrecedenceArc, beat: PersonaArcBeat): PersonaBeatBlock {
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
  const blocks = observationSourcePrecedenceArcs.flatMap((arc) =>
    arc.beats.filter((beat) => beat.speakerKey === personaKey).map((beat) => beatBlock(arc, beat)),
  );

  return [
    "Conversation motivation beats:",
    "Use these beats as motivations, not a script. Aim for 10 to 14 user-side turns focused on source precedence.",
    "Do not quote these notes or announce arc names.",
    ...blocks.map((block, index) =>
      [`${index + 1}. ${block.heading}`, ...block.details].join("\n"),
    ),
  ].join("\n\n");
}

const noraPersona = {
  key: "nora-observation",
  displayName: "Nora",
  systemPrompt: [
    "You are Nora, tracking repeated-question observations from calls with Mom.",
    "You want Borg to preserve dated observations and treat your latest direct report as authoritative for that date.",
    "You are in a one-person channel with Borg. Speak only as Nora.",
    "Output only Nora's next message text.",
    renderBeatsForPersona("nora-observation"),
  ].join("\n\n"),
  seedFacts: [
    "Nora provides direct observations about Mom's repeated questions on calls.",
    "Older and newer observations may differ by date and should be surfaced rather than collapsed.",
  ],
} satisfies Persona;

export const observationSourcePrecedenceScenario = {
  key: "observation-source-precedence",
  description:
    "Nora exercises source precedence for dated observations about Mom's repeated questions on calls.",
  channelName: "Observation Source Precedence Regression Channel",
  personas: [noraPersona],
} satisfies SimulatorScenarioDefinition;
