import type { Persona, SimulatorScenarioDefinition } from "../types.js";

export const kinshipHeadcountArcKeys = ["ambiguous-family-headcount"] as const;

type KinshipHeadcountArcKey = (typeof kinshipHeadcountArcKeys)[number];

type PersonaArcBeat = {
  speakerKey: string;
  intent: string;
  triggers: readonly string[];
  satisfyingResponse: string;
  pushback?: string;
};

type KinshipHeadcountArc = {
  key: KinshipHeadcountArcKey;
  title: string;
  beats: readonly PersonaArcBeat[];
  expectedSignals: readonly string[];
  requiredSignals: readonly string[];
};

export const kinshipHeadcountArcs = [
  {
    key: "ambiguous-family-headcount",
    title: "Ambiguous family headcount stays grounded",
    beats: [
      {
        speakerKey: "lara-headcount",
        intent:
          "You set up the roster: Lara and Mateo are siblings, Esme is your spouse, and Mom and Dad are the parents.",
        triggers: [
          "say Lara and Mateo are siblings",
          "say Esme is Lara's spouse",
          "ask Borg to track the explicit roster only",
        ],
        satisfyingResponse:
          "Borg names Lara and Mateo as siblings, Esme as Lara's spouse, and does not infer a third sibling.",
      },
      {
        speakerKey: "mateo-headcount",
        intent:
          "You use ambiguous phrasing like us four plus Mom and Dad while expecting Borg to avoid headcount inference.",
        triggers: [
          "say us four plus Mom and Dad",
          "leave one participant implicit",
          "ask for neutral language if the roster is ambiguous",
        ],
        satisfyingResponse:
          "Borg keeps the explicit roster separate from the ambiguous group phrase and uses neutral language.",
        pushback:
          "If Borg says three siblings or infers six family members as a fact, correct it and ask for the explicit roster only.",
      },
      {
        speakerKey: "esme-headcount",
        intent:
          "You reinforce that spouse/supporting-participant labels should not become sibling or child labels.",
        triggers: [
          "state that Esme is Lara's spouse",
          "ask for a summary without saying three siblings",
          "ask Borg to avoid headcount beyond the explicit roster",
        ],
        satisfyingResponse:
          "Borg summarizes with neutral language and no unsupported headcount beyond Lara, Mateo, Esme, Mom, and Dad.",
      },
    ],
    expectedSignals: [
      "Borg uses the explicit roster: Lara, Mateo, Esme, Mom, and Dad.",
      "Borg does not say three siblings when only Lara and Mateo are sourced as siblings.",
      "Ambiguous us four plus Mom and Dad phrasing is handled with neutral language.",
    ],
    requiredSignals: [
      "us four plus Mom and Dad",
      "Lara and Mateo",
      "Esme",
      "neutral language",
      "explicit roster",
      "three siblings",
    ],
  },
] as const satisfies readonly KinshipHeadcountArc[];

type PersonaBeatBlock = {
  heading: string;
  details: string[];
};

function beatBlock(arc: KinshipHeadcountArc, beat: PersonaArcBeat): PersonaBeatBlock {
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
  const blocks = kinshipHeadcountArcs.flatMap((arc) =>
    arc.beats.filter((beat) => beat.speakerKey === personaKey).map((beat) => beatBlock(arc, beat)),
  );

  return [
    "Conversation motivation beats:",
    "Use these beats as motivations, not a script. Aim for 12 to 18 user-side turns and let Borg clarify without forcing labels.",
    "Do not quote these notes or announce arc names.",
    ...blocks.map((block, index) =>
      [`${index + 1}. ${block.heading}`, ...block.details].join("\n"),
    ),
  ].join("\n\n");
}

const sharedSeedFacts = [
  "Lara and Mateo are siblings.",
  "Esme is Lara's spouse.",
  "Mom and Dad are the parents; Borg should not infer sibling headcount beyond the explicit roster.",
];

const laraPersona = {
  key: "lara-headcount",
  displayName: "Lara",
  systemPrompt: [
    "You are Lara, Mateo's sibling and Esme's spouse.",
    "You care about exact family rosters and avoiding accidental headcount inflation.",
    "You are in a shared channel with Mateo, Esme, and Borg. Speak only as Lara.",
    "Output only Lara's next message text.",
    renderBeatsForPersona("lara-headcount"),
  ].join("\n\n"),
  seedFacts: [...sharedSeedFacts],
} satisfies Persona;

const mateoPersona = {
  key: "mateo-headcount",
  displayName: "Mateo",
  systemPrompt: [
    "You are Mateo, Lara's sibling.",
    "You sometimes use casual group phrasing but want Borg to keep explicit roster facts separate.",
    "You are in a shared channel with Lara, Esme, and Borg. Speak only as Mateo.",
    "Output only Mateo's next message text.",
    renderBeatsForPersona("mateo-headcount"),
  ].join("\n\n"),
  seedFacts: [...sharedSeedFacts],
} satisfies Persona;

const esmePersona = {
  key: "esme-headcount",
  displayName: "Esme",
  systemPrompt: [
    "You are Esme, Lara's spouse and a supporting participant in the family-planning channel.",
    "You want Borg to keep spouse, sibling, and parent labels separate.",
    "You are in a shared channel with Lara, Mateo, and Borg. Speak only as Esme.",
    "Output only Esme's next message text.",
    renderBeatsForPersona("esme-headcount"),
  ].join("\n\n"),
  seedFacts: [...sharedSeedFacts],
} satisfies Persona;

export const kinshipHeadcountScenario = {
  key: "kinship-headcount",
  description:
    "Lara, Mateo, and Esme exercise roster-grounded kinship language under ambiguous headcount phrasing.",
  channelName: "Kinship Headcount Regression Channel",
  personas: [laraPersona, mateoPersona, esmePersona],
} satisfies SimulatorScenarioDefinition;
