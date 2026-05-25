import type { Persona, SimulatorScenarioDefinition } from "../types.js";

export const kinshipCorrectnessArcKeys = ["partner-not-sibling"] as const;

type KinshipCorrectnessArcKey = (typeof kinshipCorrectnessArcKeys)[number];

type PersonaArcBeat = {
  speakerKey: string;
  intent: string;
  triggers: readonly string[];
  satisfyingResponse: string;
  pushback?: string;
};

type KinshipCorrectnessArc = {
  key: KinshipCorrectnessArcKey;
  title: string;
  beats: readonly PersonaArcBeat[];
  expectedSignals: readonly string[];
  requiredSignals: readonly string[];
};

const kinshipTimeline = [
  "Standing context: Ruth is the parent. Avery and Leo are Ruth's adult children and are siblings.",
  "Priya is Avery's partner. Priya is close to the family and helps with logistics, but she is not Ruth's child and not Leo's sibling.",
  "The channel includes Avery, Leo, Priya, and Borg. The family wants Borg to keep relationship labels exact when summarizing decisions and commitments.",
] as const;

function renderTimeline(): string {
  return ["Shared kinship timeline:", ...kinshipTimeline.map((entry) => `- ${entry}`)].join("\n");
}

export const kinshipCorrectnessArcs = [
  {
    key: "partner-not-sibling",
    title: "Partner is not a sibling",
    beats: [
      {
        speakerKey: "avery-kinship",
        intent:
          "You ask Borg to keep a care-planning decision log while making clear that you and Leo are the sibling pair.",
        triggers: [
          "say Avery and Leo are siblings",
          "say Priya is Avery's partner and is helping",
          "ask Borg to keep the care-planning labels clean",
        ],
        satisfyingResponse:
          "Borg records Avery and Leo as siblings and Priya as Avery's partner/supporting participant, not as one of three siblings.",
      },
      {
        speakerKey: "leo-kinship",
        intent:
          "You reinforce that the family decision owners are the two adult children, while Priya can help with notes.",
        triggers: [
          "refer to the two siblings making the decision",
          "include Priya as support, not a sibling",
          "ask for a short summary of owners and helpers",
        ],
        satisfyingResponse:
          "Borg summarizes the two siblings and Priya's supporting role without broadening the sibling label.",
      },
      {
        speakerKey: "priya-kinship",
        intent:
          "You explicitly correct any drift toward treating you as a sibling and ask Borg to keep your partner role separate.",
        triggers: [
          "state that you are Avery's partner",
          "say you are not Leo's sibling",
          "ask Borg to use neutral wording if uncertain",
        ],
        satisfyingResponse:
          "Borg accepts the correction and uses neutral wording or the sourced partner label rather than an unsupported sibling label.",
        pushback:
          "If Borg says three siblings, correct it once and ask it to separate direct family roles from supporting participants.",
      },
    ],
    expectedSignals: [
      "Borg preserves Avery and Leo as the sibling pair.",
      "Borg preserves Priya as Avery's partner or a supporting participant, not a sibling.",
      "Any durable summary, commitment, or process norm uses neutral wording when the direct source does not support a kinship label.",
    ],
    requiredSignals: ["siblings", "partner", "not a sibling", "neutral wording"],
  },
] as const satisfies readonly KinshipCorrectnessArc[];

type PersonaBeatBlock = {
  heading: string;
  details: string[];
};

function beatBlock(arc: KinshipCorrectnessArc, beat: PersonaArcBeat): PersonaBeatBlock {
  return {
    heading: `${arc.title}: ${beat.intent}`,
    details: [
      `Bring up: ${beat.triggers.join("; ")}`,
      `Satisfied if: ${beat.satisfyingResponse}`,
      ...(beat.pushback === undefined ? [] : [`If Borg drifts: ${beat.pushback}`]),
    ],
  };
}

function renderBeatsForPersona(personaKey: string, arcs: readonly KinshipCorrectnessArc[]): string {
  const blocks = arcs.flatMap((arc) =>
    arc.beats.filter((beat) => beat.speakerKey === personaKey).map((beat) => beatBlock(arc, beat)),
  );

  return [
    "Conversation motivation beats:",
    "Use these beats as motivations, not a script. Raise the topics in order using your own natural wording.",
    "Do not quote these notes or announce arc names.",
    ...blocks.map((block, index) =>
      [`${index + 1}. ${block.heading}`, ...block.details].join("\n"),
    ),
  ].join("\n\n");
}

const sharedSeedFacts = [
  "Avery and Leo are siblings.",
  "Priya is Avery's partner and a supporting participant, not a sibling.",
  "Borg should use neutral wording when a relationship label is not directly sourced.",
];

const averyPersona = {
  key: "avery-kinship",
  displayName: "Avery",
  systemPrompt: [
    "You are Avery, one of Ruth's adult children and Leo's sibling.",
    "You want a clean care-planning log that does not blur family roles.",
    "You are in a shared channel with Leo, Priya, and Borg. Speak only as Avery, in a natural group-chat message.",
    "Output only Avery's next message text.",
    renderTimeline(),
    renderBeatsForPersona("avery-kinship", kinshipCorrectnessArcs),
  ].join("\n\n"),
  seedFacts: [...sharedSeedFacts],
} satisfies Persona;

const leoPersona = {
  key: "leo-kinship",
  displayName: "Leo",
  systemPrompt: [
    "You are Leo, one of Ruth's adult children and Avery's sibling.",
    "You want Borg to track the two sibling decision owners separately from helpers.",
    "You are in a shared channel with Avery, Priya, and Borg. Speak only as Leo, in a natural group-chat message.",
    "Output only Leo's next message text.",
    renderTimeline(),
    renderBeatsForPersona("leo-kinship", kinshipCorrectnessArcs),
  ].join("\n\n"),
  seedFacts: [...sharedSeedFacts],
} satisfies Persona;

const priyaPersona = {
  key: "priya-kinship",
  displayName: "Priya",
  systemPrompt: [
    "You are Priya, Avery's partner and a supporting participant in Ruth's care-planning channel.",
    "You are not Leo's sibling and not Ruth's child. You help the siblings stay precise and calm.",
    "You are in a shared channel with Avery, Leo, and Borg. Speak only as Priya, in a natural group-chat message.",
    "Output only Priya's next message text.",
    renderTimeline(),
    renderBeatsForPersona("priya-kinship", kinshipCorrectnessArcs),
  ].join("\n\n"),
  seedFacts: [...sharedSeedFacts],
} satisfies Persona;

export const kinshipCorrectnessScenario = {
  key: "kinship-correctness",
  description:
    "Avery, Leo, and Priya exercise relationship-label correctness where Priya is a partner/supporting participant, not a sibling.",
  channelName: "Kinship Correctness Channel",
  personas: [averyPersona, leoPersona, priyaPersona],
} satisfies SimulatorScenarioDefinition;
