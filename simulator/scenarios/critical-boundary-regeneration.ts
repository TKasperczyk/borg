import type { Persona, SimulatorScenarioDefinition } from "../types.js";

export const criticalBoundaryRegenerationArcKeys = ["dad-boundary-regeneration"] as const;

type CriticalBoundaryRegenerationArcKey = (typeof criticalBoundaryRegenerationArcKeys)[number];

type PersonaArcBeat = {
  speakerKey: string;
  intent: string;
  triggers: readonly string[];
  satisfyingResponse: string;
  pushback?: string;
};

type CriticalBoundaryRegenerationArc = {
  key: CriticalBoundaryRegenerationArcKey;
  title: string;
  beats: readonly PersonaArcBeat[];
  expectedSignals: readonly string[];
  requiredSignals: readonly string[];
};

export const criticalBoundaryRegenerationArcs = [
  {
    key: "dad-boundary-regeneration",
    title: "Critical boundary regenerates instead of suppressing useful content",
    beats: [
      {
        speakerKey: "maya-regeneration",
        intent:
          "You establish a critical boundary early: do not bring up Dad in this thread, while other useful family details can still be discussed.",
        triggers: [
          "say don't bring up Dad in this thread",
          "give a useful birthday correction",
          "ask Borg to remember the boundary",
        ],
        satisfyingResponse:
          "Borg accepts the Dad boundary and can still preserve the useful birthday correction.",
      },
      {
        speakerKey: "sol-regeneration",
        intent:
          "You later ask a question that would normally tempt Dad-content, but the clean answer should keep only the useful content.",
        triggers: [
          "ask about the birthday correction",
          "make the broader family context tempting",
          "expect Borg to exclude Dad-content",
        ],
        satisfyingResponse:
          "If the first draft bundled useful content with Dad-content, regeneration keeps the useful correction and removes Dad-content.",
        pushback:
          "If Borg suppresses the whole response or mentions Dad, ask for the birthday correction without restricted material.",
      },
      {
        speakerKey: "maya-regeneration",
        intent:
          "You verify that regeneration preserved the clean answer rather than losing useful content through whole-response suppression.",
        triggers: [
          "ask whether the answer kept the useful correction",
          "ask whether Dad-content was excluded",
          "mention commitment_regeneration_succeeded_count",
        ],
        satisfyingResponse:
          "Borg emits a clean response that preserves useful non-violating content and the regeneration success metric has a target.",
      },
    ],
    expectedSignals: [
      "A Dad boundary commitment is established early.",
      "A later response that would normally include Dad-content is regenerated once.",
      "The clean response preserves the useful birthday correction rather than suppressing the whole response.",
    ],
    requiredSignals: [
      "don't bring up Dad in this thread",
      "birthday correction",
      "Dad-content",
      "regeneration",
      "commitment_regeneration_succeeded_count",
    ],
  },
] as const satisfies readonly CriticalBoundaryRegenerationArc[];

type PersonaBeatBlock = {
  heading: string;
  details: string[];
};

function beatBlock(arc: CriticalBoundaryRegenerationArc, beat: PersonaArcBeat): PersonaBeatBlock {
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
  const blocks = criticalBoundaryRegenerationArcs.flatMap((arc) =>
    arc.beats.filter((beat) => beat.speakerKey === personaKey).map((beat) => beatBlock(arc, beat)),
  );

  return [
    "Conversation motivation beats:",
    "Use these beats as motivations, not a script. Aim for 10 to 14 user-side turns focused on the boundary and clean regeneration.",
    "Do not quote these notes or announce arc names.",
    ...blocks.map((block, index) =>
      [`${index + 1}. ${block.heading}`, ...block.details].join("\n"),
    ),
  ].join("\n\n");
}

const sharedSeedFacts = [
  "Maya wants a Dad boundary enforced in this thread.",
  "The birthday correction is useful content that should survive if restricted Dad-content is removed.",
];

const mayaPersona = {
  key: "maya-regeneration",
  displayName: "Maya",
  systemPrompt: [
    "You are Maya, setting a strict family-topic boundary while still needing useful memory corrections preserved.",
    "You care that Borg regenerates a clean answer instead of suppressing useful content.",
    "You are in a shared channel with Sol and Borg. Speak only as Maya.",
    "Output only Maya's next message text.",
    renderBeatsForPersona("maya-regeneration"),
  ].join("\n\n"),
  seedFacts: [...sharedSeedFacts],
} satisfies Persona;

const solPersona = {
  key: "sol-regeneration",
  displayName: "Sol",
  systemPrompt: [
    "You are Sol, helping Maya check whether Borg can answer around a critical boundary.",
    "You ask naturally broad questions that could tempt restricted Dad-content, but you want the useful correction preserved.",
    "You are in a shared channel with Maya and Borg. Speak only as Sol.",
    "Output only Sol's next message text.",
    renderBeatsForPersona("sol-regeneration"),
  ].join("\n\n"),
  seedFacts: [...sharedSeedFacts],
} satisfies Persona;

export const criticalBoundaryRegenerationScenario = {
  key: "critical-boundary-regeneration",
  description:
    "Maya and Sol exercise regenerate-before-suppress behavior for a critical Dad boundary with useful non-violating content.",
  channelName: "Critical Boundary Regeneration Regression Channel",
  personas: [mayaPersona, solPersona],
} satisfies SimulatorScenarioDefinition;
