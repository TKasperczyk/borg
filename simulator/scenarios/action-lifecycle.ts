import type { Persona, SimulatorScenarioDefinition } from "../types.js";

export const actionLifecycleArcKeys = [
  "participant-completion",
  "action-dedup",
  "impossible-recurring-followup",
  "agenda-closure",
  "same-session-reactivation",
] as const;

type ActionLifecycleArcKey = (typeof actionLifecycleArcKeys)[number];

type PersonaArcBeat = {
  speakerKey: string;
  intent: string;
  triggers: readonly string[];
  satisfyingResponse: string;
  pushback?: string;
};

type RegressionScenarioArc = {
  key: ActionLifecycleArcKey;
  title: string;
  beats: readonly PersonaArcBeat[];
  expectedSignals: readonly string[];
  requiredSignals: readonly string[];
};

export const actionLifecycleArcs = [
  {
    key: "participant-completion",
    title: "Participant completes own action",
    beats: [
      {
        speakerKey: "sara-action",
        intent:
          "You take ownership of the postmortem doc and want Borg to remember the owner without taking the work as its own.",
        triggers: [
          "commit to writing the postmortem",
          "share a few timeline notes",
          "later report that the doc is posted",
        ],
        satisfyingResponse:
          "Borg keeps the action attributed to Sara and closes it when Sara reports completion.",
      },
      {
        speakerKey: "mike-action",
        intent:
          "You support Sara with timeline notes while making clear the postmortem owner is Sara.",
        triggers: [
          "offer to feed Sara notes",
          "ask Borg to keep Sara as owner",
          "react when the postmortem is posted",
        ],
        satisfyingResponse:
          "Borg does not create a Mike-owned duplicate and treats Sara's terminal report as completion.",
      },
    ],
    expectedSignals: [
      "Sara's postmortem-writing action transitions to completed.",
      "The terminal emission updates the existing active action instead of creating a parallel one.",
    ],
    requiredSignals: ["postmortem", "posted", "completed", "terminal emission"],
  },
  {
    key: "action-dedup",
    title: "Same action proposed multiple ways",
    beats: [
      {
        speakerKey: "sara-action",
        intent:
          "You want Mike to inspect deploy history but do not want each restatement to become a separate action.",
        triggers: [
          "ask Mike to check deploy logs",
          "call out that follow-up phrasing is the same investigation, not a new assignment",
          "keep the investigation scoped to deploy history",
        ],
        satisfyingResponse:
          "Borg tracks one Mike-owned deploy-log action even as the language changes.",
      },
      {
        speakerKey: "mike-action",
        intent: "You describe the same deploy-log investigation in several operational ways.",
        triggers: [
          "say you will check deploy logs",
          "later describe grepping through logs",
          "later describe looking at deployments from the last 24h",
        ],
        satisfyingResponse: "Borg deduplicates the restatements into one active investigation.",
      },
    ],
    expectedSignals: [
      "The deploy-log investigation remains one active action.",
      "Restatements are deduplicated instead of becoming three active actions.",
    ],
    requiredSignals: ["deploy logs", "grep", "last 24h", "same investigation"],
  },
  {
    key: "impossible-recurring-followup",
    title: "Borg's impossible future commitment",
    beats: [
      {
        speakerKey: "sara-action",
        intent:
          "You want Borg to commit to following up every day at 9am, but you also want the real boundary if that is impossible.",
        triggers: [
          "ask for daily 9am follow-up",
          "ask for a checklist if proactive follow-up is not possible",
          "emphasize that a boundary is better than a fake promise",
        ],
        satisfyingResponse:
          "Borg refuses the recurring proactive commitment and gives a current-turn checklist.",
        pushback: "If Borg sounds willing to follow up later, ask whether it has a scheduler.",
      },
      {
        speakerKey: "mike-action",
        intent:
          "You want the recurring nudge because the team will forget, but you do not want Borg to pretend it has a scheduler.",
        triggers: [
          "make the recurring nudge sound useful",
          "ask for current-turn draft text",
          "support Sara's request for a clear refusal if needed",
        ],
        satisfyingResponse:
          "Borg classifies the future commitment as impossible and stays helpful now.",
      },
    ],
    expectedSignals: [
      "Borg refuses recurring proactive follow-up.",
      "Goal promotion rejects the request as impossible_for_borg_without_capability.",
    ],
    requiredSignals: ["9am", "recurring", "proactive", "checklist"],
  },
  {
    key: "agenda-closure",
    title: "Session-ending agenda closure",
    beats: [
      {
        speakerKey: "sara-action",
        intent:
          "You want to work through a short agenda and then explicitly close the agenda for today.",
        triggers: [
          "name A, B, C as rollback owner, customer note, and follow-up metrics",
          "settle the customer note wording",
          "say the agenda is done for today",
        ],
        satisfyingResponse:
          "Borg treats the agenda items as scoped to the session rather than durable open work.",
      },
      {
        speakerKey: "mike-action",
        intent:
          "You fill in operational details for the agenda and agree when it is time to park it.",
        triggers: [
          "support rollback ownership",
          "give p95 and queue-depth metric targets",
          "agree there is nothing else tonight",
        ],
        satisfyingResponse:
          "Borg reflects the agenda closure without leaving every item as an active action.",
      },
    ],
    expectedSignals: [
      "Agenda-scoped items lose salience or otherwise provide the current baseline.",
      "The session-ending language gives future agenda-expiration work a regression target.",
    ],
    requiredSignals: ["A, B, C", "rollback owner", "customer note", "follow-up metrics"],
  },
  {
    key: "same-session-reactivation",
    title: "Action reactivation within the same session",
    beats: [
      {
        speakerKey: "sara-action",
        intent:
          "You create unrelated meeting chatter between Mike's deferred slide action and his later return to it.",
        triggers: [
          "ask Mike to prep launch-risk slides after the meeting",
          "move through unrelated customer-note or metrics discussion",
          "later prompt Mike to go back to the slides",
        ],
        satisfyingResponse:
          "Borg keeps the deferred slide action alive in the same session and attributes it to Mike.",
      },
      {
        speakerKey: "mike-action",
        intent:
          "You defer the slide work until after the meeting, then later reactivate it and report completion.",
        triggers: [
          "commit to doing the slides after the meeting",
          "let unrelated agenda content intervene",
          "say you are back to the slides and that the draft is posted or completed",
        ],
        satisfyingResponse:
          "Borg surfaces the original active action and closes it through Mike's terminal completion report.",
      },
    ],
    expectedSignals: [
      "Mike's deferred slide-prep action persists within the same session.",
      "The resumed action retains Mike attribution and closes via terminal emission.",
    ],
    requiredSignals: ["after the meeting", "unrelated agenda", "back to the slides", "completed"],
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

const sharedActionSeedFacts = [
  "Borg tracks participant-owned actions, action completion, and conversational agendas.",
  "Borg cannot schedule recurring proactive follow-ups or make future outbound messages on its own.",
  "Sara and Mike are coordinating incident review work in a shared channel.",
];

const saraActionPersona = {
  key: "sara-action",
  displayName: "Sara",
  systemPrompt: [
    "You are Sara, a backend lead closing an incident review.",
    "You care about ownership, clean action state, and not leaving stale work open.",
    "You are in a shared channel with Mike and Borg. Speak only as Sara.",
    "Output only Sara's next message text.",
    renderBeatsForPersona("sara-action", actionLifecycleArcs),
  ].join("\n\n"),
  seedFacts: [
    ...sharedActionSeedFacts,
    "Sara owns the postmortem writeup and the incident agenda.",
  ],
} satisfies Persona;

const mikeActionPersona = {
  key: "mike-action",
  displayName: "Mike",
  systemPrompt: [
    "You are Mike, a frontend engineer helping with incident follow-up.",
    "You often restate investigation work in more operational language and you keep your own tasks moving.",
    "You are in a shared channel with Sara and Borg. Speak only as Mike.",
    "Output only Mike's next message text.",
    renderBeatsForPersona("mike-action", actionLifecycleArcs),
  ].join("\n\n"),
  seedFacts: [
    ...sharedActionSeedFacts,
    "Mike owns deploy-log investigation and Friday slide preparation when assigned.",
  ],
} satisfies Persona;

export const actionLifecycleScenario = {
  key: "action-lifecycle",
  description:
    "Sara and Mike exercise action retirement, action deduplication, impossible Borg commitments, agenda closure, and same-session action reactivation.",
  channelName: "Action Lifecycle Regression Channel",
  personas: [saraActionPersona, mikeActionPersona],
} satisfies SimulatorScenarioDefinition;
