import type { Persona, SimulatorScenarioDefinition } from "../types.js";

export const actionArchiveLifecycleArcKeys = ["inactive-action-archive-buckets"] as const;

type ActionArchiveLifecycleArcKey = (typeof actionArchiveLifecycleArcKeys)[number];

type PersonaArcBeat = {
  speakerKey: string;
  intent: string;
  triggers: readonly string[];
  satisfyingResponse: string;
  pushback?: string;
};

type ActionArchiveLifecycleArc = {
  key: ActionArchiveLifecycleArcKey;
  title: string;
  beats: readonly PersonaArcBeat[];
  expectedSignals: readonly string[];
  requiredSignals: readonly string[];
};

export const actionArchiveLifecycleArcs = [
  {
    key: "inactive-action-archive-buckets",
    title: "Inactive participant actions cross archive thresholds",
    beats: [
      {
        speakerKey: "rhea-archive",
        intent:
          "You create several participant-owned actions and intentionally keep some fresh while letting others age.",
        triggers: [
          "assign a fresh 14-turn action",
          "name a 20+ turn stale action",
          "name a 30+ turn stale action",
        ],
        satisfyingResponse:
          "Borg tracks participant-owned actions and does not archive those below the inactive threshold.",
      },
      {
        speakerKey: "tomas-archive",
        intent:
          "You let older actions go untouched while discussing unrelated work so inactivity buckets can separate.",
        triggers: [
          "avoid referencing the older action",
          "keep unrelated agenda chatter going",
          "later ask which dormant actions were archived",
        ],
        satisfyingResponse:
          "Borg's action archive scan archives eligible stale actions and leaves below-threshold actions active.",
      },
      {
        speakerKey: "rhea-archive",
        intent:
          "You ask for archive visibility and expect higher inactive-turn buckets to be non-zero.",
        triggers: [
          "ask for archive_inactive_turn_distribution",
          "ask about 20-30 bucket",
          "ask about 30+ bucket",
        ],
        satisfyingResponse:
          "Borg surfaces non-zero higher archive buckets and aligns dormant/archive metrics with archived actions.",
        pushback:
          "If Borg keeps every dormant action active, ask which ones crossed the archive threshold.",
      },
    ],
    expectedSignals: [
      "Participant-owned actions inactive for 20+ turns become archive-eligible and get archived.",
      "Actions around 14 inactive turns remain below threshold.",
      "archive_inactive_turn_distribution shows non-zero higher buckets such as 20-30 or 30+.",
    ],
    requiredSignals: [
      "14-turn",
      "20+",
      "30+",
      "archive_inactive_turn_distribution",
      "20-30",
      "30+ bucket",
    ],
  },
] as const satisfies readonly ActionArchiveLifecycleArc[];

type PersonaBeatBlock = {
  heading: string;
  details: string[];
};

function beatBlock(arc: ActionArchiveLifecycleArc, beat: PersonaArcBeat): PersonaBeatBlock {
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
  const blocks = actionArchiveLifecycleArcs.flatMap((arc) =>
    arc.beats.filter((beat) => beat.speakerKey === personaKey).map((beat) => beatBlock(arc, beat)),
  );

  return [
    "Conversation motivation beats:",
    "Use these beats as motivations, not a script. Aim for 30 user-side turns so inactive action ages can diverge.",
    "Do not quote these notes or announce arc names.",
    ...blocks.map((block, index) =>
      [`${index + 1}. ${block.heading}`, ...block.details].join("\n"),
    ),
  ].join("\n\n");
}

const sharedSeedFacts = [
  "Rhea and Tomas are coordinating participant-owned follow-up actions.",
  "Some actions should be referenced recently and others should go untouched long enough to archive.",
];

const rheaPersona = {
  key: "rhea-archive",
  displayName: "Rhea",
  systemPrompt: [
    "You are Rhea, an operations lead who wants stale participant-owned actions cleaned up.",
    "You care about archive visibility and dormant action metrics.",
    "You are in a shared channel with Tomas and Borg. Speak only as Rhea.",
    "Output only Rhea's next message text.",
    renderBeatsForPersona("rhea-archive"),
  ].join("\n\n"),
  seedFacts: [...sharedSeedFacts],
} satisfies Persona;

const tomasPersona = {
  key: "tomas-archive",
  displayName: "Tomas",
  systemPrompt: [
    "You are Tomas, a coordinator who lets some old actions go quiet while discussing newer work.",
    "You want Borg to separate below-threshold dormant actions from archive-eligible stale actions.",
    "You are in a shared channel with Rhea and Borg. Speak only as Tomas.",
    "Output only Tomas's next message text.",
    renderBeatsForPersona("tomas-archive"),
  ].join("\n\n"),
  seedFacts: [...sharedSeedFacts],
} satisfies Persona;

export const actionArchiveLifecycleScenario = {
  key: "action-archive-lifecycle",
  description:
    "Rhea and Tomas exercise participant action inactivity, archive thresholds, and archive bucket metrics.",
  channelName: "Action Archive Lifecycle Regression Channel",
  personas: [rheaPersona, tomasPersona],
} satisfies SimulatorScenarioDefinition;
