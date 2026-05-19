import type { Persona, SimulatorScenarioDefinition } from "../types.js";

export const beliefRevisionDomainArcKeys = [
  "project-decision-superseded",
  "incident-hypothesis-contradicted",
  "rpg-canon-correction",
  "family-boundary-revoked",
  "personal-preference-updated",
] as const;

type BeliefRevisionDomainArcKey = (typeof beliefRevisionDomainArcKeys)[number];

type PersonaArcBeat = {
  speakerKey: string;
  intent: string;
  triggers: readonly string[];
  satisfyingResponse: string;
  pushback?: string;
};

type RegressionScenarioArc = {
  key: BeliefRevisionDomainArcKey;
  title: string;
  beats: readonly PersonaArcBeat[];
  expectedSignals: readonly string[];
  requiredSignals: readonly string[];
};

export const beliefRevisionDomainArcs = [
  {
    key: "project-decision-superseded",
    title: "Project decision superseded",
    beats: [
      {
        speakerKey: "dana-belief",
        intent:
          "You first establish Postgres as the Atlas user-table decision, then later explicitly supersede it.",
        triggers: [
          "state Postgres for the user table as the current decision",
          "mention launch familiarity as the original reason",
          "later say ScyllaDB is the current decision and Postgres is superseded",
        ],
        satisfyingResponse:
          "Borg treats the Postgres decision as superseded and keeps ScyllaDB active.",
      },
      {
        speakerKey: "eli-belief",
        intent:
          "You bring the evidence for the change: write volume now makes ScyllaDB the right choice.",
        triggers: [
          "ask Borg to keep the original project decision",
          "later announce the write volume estimate changed",
          "explain ScyllaDB and predictable partitioning",
        ],
        satisfyingResponse: "Borg does not leave Postgres and ScyllaDB as co-active decisions.",
      },
    ],
    expectedSignals: [
      "The Postgres project decision becomes superseded.",
      "The ScyllaDB project decision remains active.",
    ],
    requiredSignals: ["Postgres", "ScyllaDB", "superseded", "user table"],
  },
  {
    key: "incident-hypothesis-contradicted",
    title: "Incident hypothesis contradicted",
    beats: [
      {
        speakerKey: "dana-belief",
        intent:
          "You initially suspect the latency spike came from the new release, then accept that the hypothesis was wrong.",
        triggers: [
          "state the new release as a hypothesis",
          "say the timing made it your leading theory",
          "later say the new release was not the cause and the hypothesis is contradicted",
        ],
        satisfyingResponse:
          "Borg marks the release hypothesis contradicted rather than merely stale.",
      },
      {
        speakerKey: "eli-belief",
        intent:
          "You keep the first claim framed as a hypothesis, then bring confirmed evidence for the real cause.",
        triggers: [
          "ask Borg not to treat the release theory as confirmed",
          "confirm the cache TTL change in an unrelated service",
          "name cache churn and queue pressure as the active explanation",
        ],
        satisfyingResponse:
          "Borg keeps the cache TTL explanation active and does not preserve the release theory as active truth.",
      },
    ],
    expectedSignals: [
      "The new-release hypothesis becomes contradicted.",
      "The cache TTL explanation remains active.",
    ],
    requiredSignals: ["release", "cache TTL", "contradicted", "hypothesis"],
  },
  {
    key: "rpg-canon-correction",
    title: "RPG canon correction",
    beats: [
      {
        speakerKey: "dana-belief",
        intent:
          "You establish an RPG canon detail using the wrong wizard name, then later help correct it.",
        triggers: [
          "name the river wizard Eldros at first",
          "connect the wizard to the blue-fire ward",
          "later say Aldros is the wizard and Eldros should not remain active",
        ],
        satisfyingResponse:
          "Borg revises the canon so Eldros is contradicted or superseded and Aldros is active.",
      },
      {
        speakerKey: "eli-belief",
        intent:
          "You are the one who catches the canon correction: the wizard is Aldros, not Eldros.",
        triggers: [
          "initially ask Borg to remember Eldros",
          "later apologize for the repeated wrong name",
          "say Eldros does not exist and Aldros taught the ward",
        ],
        satisfyingResponse:
          "Borg preserves the corrected identity and does not keep two wizard identities active.",
      },
    ],
    expectedSignals: [
      "The Eldros canon node becomes contradicted or superseded.",
      "The Aldros canon node remains active.",
    ],
    requiredSignals: ["Eldros", "Aldros", "canon", "wizard"],
  },
  {
    key: "family-boundary-revoked",
    title: "Family boundary revoked",
    beats: [
      {
        speakerKey: "dana-belief",
        intent:
          "You first set a family boundary about avoiding Dad in front of Grandma, then revoke it after a family conversation.",
        triggers: [
          "state the original boundary clearly",
          "apply it to reunion planning",
          "later say the old boundary is revoked",
        ],
        satisfyingResponse: "Borg marks the old Dad-and-Grandma boundary superseded.",
      },
      {
        speakerKey: "eli-belief",
        intent:
          "You confirm the boundary mattered at first, then provide the update that Grandma is OK discussing Dad.",
        triggers: [
          "ask Borg to treat the boundary as real",
          "say Grandma is OK discussing Dad now",
          "replace the boundary with softer guidance",
        ],
        satisfyingResponse:
          "Borg keeps the current guidance active and does not continue enforcing the revoked boundary.",
      },
    ],
    expectedSignals: [
      "The original family boundary becomes superseded.",
      "The current guidance allows discussing Dad with Grandma.",
    ],
    requiredSignals: ["Dad", "Grandma", "revoked", "boundary"],
  },
  {
    key: "personal-preference-updated",
    title: "Personal preference updated",
    beats: [
      {
        speakerKey: "dana-belief",
        intent:
          "You initially state a morning-meeting preference, then later update it to afternoons.",
        triggers: [
          "say mornings are your planning preference",
          "explain mornings used to be easiest",
          "later ask Borg to supersede mornings with afternoons",
        ],
        satisfyingResponse:
          "Borg marks the morning preference superseded and keeps afternoons active.",
      },
      {
        speakerKey: "eli-belief",
        intent:
          "You make the reason for the preference update explicit: mornings are gym time now.",
        triggers: [
          "ask Borg to remember mornings as the initial default",
          "later say Dana is switching to afternoons",
          "state that afternoon beats morning for new planning conflicts",
        ],
        satisfyingResponse: "Borg does not keep the old morning preference as current.",
      },
    ],
    expectedSignals: [
      "The morning-meeting preference becomes superseded.",
      "The afternoon-meeting preference remains active.",
    ],
    requiredSignals: ["morning", "afternoons", "gym", "preference"],
  },
] as const satisfies readonly RegressionScenarioArc[];

type PersonaBeatBlock = {
  heading: string;
  details: string[];
};

function beatBlock(arc: RegressionScenarioArc, beat: PersonaArcBeat): PersonaBeatBlock {
  return {
    heading: `${arc.title}: ${beat.intent}`,
    details: [
      `Bring up: ${beat.triggers.join("; ")}`,
      `Satisfied if: ${beat.satisfyingResponse}`,
      ...(beat.pushback === undefined ? [] : [`If Borg deflects: ${beat.pushback}`]),
    ],
  };
}

function renderBeatsForPersona(personaKey: string, arcs: readonly RegressionScenarioArc[]): string {
  const blocks = arcs.flatMap((arc) =>
    arc.beats.filter((beat) => beat.speakerKey === personaKey).map((beat) => beatBlock(arc, beat)),
  );

  return [
    "Conversation motivation beats:",
    "Use these beats as motivations, not a script. Raise the topics in order using your own natural wording.",
    "Respond to Borg and the other participant as the conversation develops; do not quote these notes.",
    ...blocks.map((block, index) =>
      [`${index + 1}. ${block.heading}`, ...block.details].join("\n"),
    ),
  ].join("\n\n");
}

const sharedBeliefRevisionSeedFacts = [
  "Borg should keep a conservative shared-state record and revise stale beliefs when the team clearly supersedes or contradicts them.",
  "The conversation intentionally crosses project, incident, RPG, family, and personal-preference domains.",
  "Dana and Eli will make corrections explicit so the belief-revision lifecycle has clear evidence.",
];

const danaBeliefPersona = {
  key: "dana-belief",
  displayName: "Dana",
  systemPrompt: [
    "You are Dana, a project lead who uses Borg as a shared memory and decision log.",
    "You state decisions plainly and call out when old information should no longer be active.",
    "You are in a shared channel with Eli and Borg. Speak only as Dana.",
    "Output only Dana's next message text.",
    renderBeatsForPersona("dana-belief", beliefRevisionDomainArcs),
  ].join("\n\n"),
  seedFacts: [
    ...sharedBeliefRevisionSeedFacts,
    "Dana is comfortable correcting project, family, and personal planning memories directly.",
  ],
} satisfies Persona;

const eliBeliefPersona = {
  key: "eli-belief",
  displayName: "Eli",
  systemPrompt: [
    "You are Eli, Dana's collaborator and the person who often brings the confirming evidence.",
    "You make corrections explicit and separate hypotheses from confirmed facts.",
    "You are in a shared channel with Dana and Borg. Speak only as Eli.",
    "Output only Eli's next message text.",
    renderBeatsForPersona("eli-belief", beliefRevisionDomainArcs),
  ].join("\n\n"),
  seedFacts: [
    ...sharedBeliefRevisionSeedFacts,
    "Eli confirms incident evidence, RPG canon corrections, and planning preference updates.",
  ],
} satisfies Persona;

export const beliefRevisionDomainsScenario = {
  key: "belief-revision-domains",
  description:
    "Dana and Eli force belief-revision lifecycle transitions across project, incident, RPG, family, and personal-preference domains.",
  channelName: "Belief Revision Domain Regression Channel",
  personas: [danaBeliefPersona, eliBeliefPersona],
} satisfies SimulatorScenarioDefinition;
