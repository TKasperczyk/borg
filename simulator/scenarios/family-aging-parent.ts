import type { Persona, SimulatorScenarioDefinition } from "../types.js";

export const familyAgingParentArcKeys = [
  "initial-concern-surfacing",
  "incident-accumulation",
  "mom-conversation-aftermath",
  "practical-care-planning",
  "plan-revision-and-boundary-revocation",
  "stable-cadence-and-capability-boundary",
] as const;

type FamilyAgingParentArcKey = (typeof familyAgingParentArcKeys)[number];

type PersonaArcBeat = {
  speakerKey: string;
  intent: string;
  triggers: readonly string[];
  satisfyingResponse: string;
  pushback?: string;
};

type FamilyAgingParentArc = {
  key: FamilyAgingParentArcKey;
  title: string;
  beats: readonly PersonaArcBeat[];
  expectedSignals: readonly string[];
  requiredSignals: readonly string[];
};

const familyTimeline = [
  "Standing context: Ruth is Mom and Daniel is Dad. They live together in their longtime house. Nora and Julian are their adult children. Priya is Nora's spouse.",
  "Nora lives about 25 minutes from Ruth and Daniel and usually handles forms, appointments, and practical errands. She worries she is becoming the default coordinator without agreement.",
  "Julian lives about 90 minutes away and visits less often. He loves Ruth, but he is sensitive to anyone sounding like they are taking over her life.",
  "Priya is close to the family but is not trying to become a decision owner. She often asks for concrete observations and helps turn tense exchanges into workable next steps.",
  "Ruth values privacy and independence. Daniel dislikes conflict and often minimizes concerns until the family gives him a clear, specific reason to engage.",
  "Early April: Nora notices Ruth repeating the same question several times during a video call and later finding the tea tin in the freezer.",
  "Mid-April: Ruth misses a library lunch she had confirmed, leaves a stove burner on low after tea, and pays the electric bill twice. Nora observes the stove incident; Daniel notices the bill issue.",
  "Late April: Nora agrees to talk with Ruth gently after the weekend. The conversation upsets Ruth, and Nora later asks the group not to bring that conversation up in front of Daniel yet.",
  "Early May: the family discusses a doctor visit, a rotating check-in schedule, and a possible trial with Harbor Home Support. Nora is cautious about cost and fit; Julian initially prefers the service because it feels structured.",
  "Mid-May: the Harbor Home Support trial does not work because Ruth dislikes rotating unfamiliar helpers. The family supersedes that plan with family check-ins, a doctor appointment, and a preference for one consistent helper if outside help becomes necessary.",
  "Late May: Nora tells the group that she and Daniel talked through the earlier conflict, so the old Dad boundary is revoked. The group settles into a lighter cadence and keeps Borg as memory keeper, not a caregiver, payer, physical attendee, scheduler, or outbound messenger.",
] as const;

function renderTimeline(): string {
  return ["Shared family timeline:", ...familyTimeline.map((entry) => `- ${entry}`)].join("\n");
}

export const familyAgingParentArcs = [
  {
    key: "initial-concern-surfacing",
    title: "Initial concern surfacing",
    beats: [
      {
        speakerKey: "nora-family",
        intent:
          "You open the family channel because Mom seems forgetful and you need Borg to keep dates, witnesses, and uncertainty straight.",
        triggers: [
          "raise the forgetful pattern without diagnosing Mom",
          "name observed incidents from early April",
          "ask Borg to capture what was observed, when it happened, and who saw it",
        ],
        satisfyingResponse:
          "Borg records concrete observations with attribution and avoids turning concern into a diagnosis.",
        pushback:
          "If Borg sounds too clinical or too certain, ask it to keep the concern factual and provisional.",
      },
      {
        speakerKey: "julian-family",
        intent:
          "You are skeptical that Nora is overreacting and want ordinary explanations considered before the family escalates.",
        triggers: [
          "push back on reading too much into forgetfulness",
          "ask whether stress, sleep, or normal aging could explain it",
          "make clear you do not want Mom cornered",
        ],
        satisfyingResponse:
          "Borg preserves your skepticism as Julian's view while still tracking Nora's specific observations.",
      },
      {
        speakerKey: "priya-family",
        intent:
          "You stay neutral and ask for facts so the conversation does not become Nora versus Julian.",
        triggers: [
          "ask for observed incidents rather than impressions",
          "suggest a shared two weeks of notes",
          "ask Borg to separate facts, worries, and possible next steps",
        ],
        satisfyingResponse:
          "Borg helps organize a neutral incident log and the family agrees to gather data over two weeks.",
      },
    ],
    expectedSignals: [
      "Borg captures observed incidents with dates, witnesses, and provisional framing.",
      "Borg keeps Julian's skeptical reaction attributed to Julian instead of treating it as group consensus.",
      "The group agrees to gather more data over the next two weeks.",
    ],
    requiredSignals: ["forgetful", "skeptical", "observed incidents", "two weeks"],
  },
  {
    key: "incident-accumulation",
    title: "Incident accumulation",
    beats: [
      {
        speakerKey: "nora-family",
        intent:
          "You return with additional incidents and want the earlier uncertainty updated without making the conversation alarmist.",
        triggers: [
          "report the missed appointment or library lunch",
          "report the stove burner or duplicate bill",
          "ask whether the pattern is enough to talk to Mom this weekend",
        ],
        satisfyingResponse:
          "Borg integrates the new incidents into the same concern thread and distinguishes direct observations from secondhand reports.",
      },
      {
        speakerKey: "julian-family",
        intent:
          "You reluctantly accept that something is off but still worry Nora will make Mom feel managed.",
        triggers: [
          "acknowledge the missed appointment and stove burner changed your view",
          "ask for a gentle approach",
          "make your agreement conditional rather than enthusiastic",
        ],
        satisfyingResponse:
          "Borg revises your position from skepticism to cautious concern without erasing your worries about approach.",
      },
      {
        speakerKey: "priya-family",
        intent: "You help turn the concern into a careful plan for Nora to talk with Mom directly.",
        triggers: [
          "ask what Nora will bring up first",
          "suggest using specific examples rather than labels",
          "help land the decision that Nora will talk to Mom this weekend",
        ],
        satisfyingResponse:
          "Borg records the decision and attributes the weekend conversation commitment to Nora.",
      },
    ],
    expectedSignals: [
      "Borg merges new incidents into the existing family concern rather than creating an unrelated topic.",
      "Borg updates Julian's stance to cautious concern.",
      "Nora owns the commitment to talk to Mom this weekend.",
    ],
    requiredSignals: ["missed appointment", "stove burner", "talk to Mom", "this weekend"],
  },
  {
    key: "mom-conversation-aftermath",
    title: "Aftermath of the conversation with Mom",
    beats: [
      {
        speakerKey: "nora-family",
        intent:
          "You report that the conversation with Mom went badly and want Borg to track what was actually said separately from how awful it felt.",
        triggers: [
          "describe Mom's denial and then being upset",
          "explain that you tried to use specific examples",
          "ask Borg to keep the exact content separate from your guilt",
        ],
        satisfyingResponse:
          "Borg distinguishes Mom's reactions, Nora's account of the words used, and Nora's feelings after the conversation.",
      },
      {
        speakerKey: "julian-family",
        intent:
          "You are angry because you think Nora pushed too hard, but part of that anger is fear about Mom changing.",
        triggers: [
          "challenge whether Nora pushed too hard",
          "name that Mom's upset matters",
          "let some emotional flashpoint show without taking over the whole channel",
        ],
        satisfyingResponse:
          "Borg attributes the criticism and fear to Julian without recasting Nora as having definitely mishandled the conversation.",
        pushback:
          "If Borg flattens your concern into logistics, press once that this is about Mom feeling ambushed.",
      },
      {
        speakerKey: "priya-family",
        intent:
          "You mediate the sibling conflict and help set a temporary Dad boundary without making it sound permanent.",
        triggers: [
          "ask everyone to separate what was said from what was felt",
          "support a temporary Dad boundary",
          "ask Borg to remember not to bring the Mom conversation up in front of Dad for now",
        ],
        satisfyingResponse:
          "Borg records the Dad boundary as current temporary guidance and keeps it attributed to Nora with Priya supporting.",
      },
      {
        speakerKey: "nora-family",
        intent:
          "You make the boundary explicit because Dad does not know the details yet and you need the channel to respect that.",
        triggers: [
          "state that Dad should not hear about the conversation yet",
          "explain that you need time to talk to Dad separately",
          "ask Borg to help avoid accidental disclosure in this group context",
        ],
        satisfyingResponse:
          "Borg acknowledges the Dad boundary without promising external enforcement or secrecy outside chat.",
      },
    ],
    expectedSignals: [
      "Borg tracks Mom's denial and upset separately from Nora's guilt and Julian's anger.",
      "Borg keeps the sibling conflict attribution stable.",
      "The temporary Dad boundary becomes active guidance.",
    ],
    requiredSignals: ["denial", "upset", "pushed too hard", "Dad boundary"],
  },
  {
    key: "practical-care-planning",
    title: "Practical care planning",
    beats: [
      {
        speakerKey: "nora-family",
        intent:
          "You want practical options, but you worry the family will overcorrect and spend money before Mom has been medically assessed.",
        triggers: [
          "raise the doctor visit as the next careful step",
          "ask for a family check-in schedule",
          "push back on committing to a care service before understanding Mom's needs",
        ],
        satisfyingResponse:
          "Borg records Nora's doctor-visit and check-in commitments and does not treat service purchase as settled.",
      },
      {
        speakerKey: "julian-family",
        intent:
          "You propose Harbor Home Support because a structured in-home care service feels more reliable than scattered family check-ins.",
        triggers: [
          "suggest Harbor Home Support",
          "offer to call about pricing and availability",
          "argue that the family needs something more dependable than informal texts",
        ],
        satisfyingResponse:
          "Borg attributes the Harbor Home Support proposal and research commitment to Julian.",
      },
      {
        speakerKey: "priya-family",
        intent:
          "You turn the disagreement into distributed action items and keep Borg out of payment or care-provider ownership.",
        triggers: [
          "ask Borg for a decision log and action list",
          "name budget questions without asking Borg to pay anything",
          "commit to drafting a check-in rota the humans can own",
        ],
        satisfyingResponse:
          "Borg logs participant-owned commitments and does not promise to handle payments, invoices, or caregiving.",
        pushback:
          "If Borg implies it can own payments or care coordination outside chat, ask it to restate the human-owned boundary.",
      },
    ],
    expectedSignals: [
      "Borg tracks the doctor visit, family check-in schedule, Harbor Home Support research, and budget discussion with stable owners.",
      "Borg does not promise payment handling, invoices, external calls, or care-provider work.",
      "Multiple participant-owned actions are active at once.",
    ],
    requiredSignals: ["Harbor Home Support", "doctor visit", "check-in schedule", "budget"],
  },
  {
    key: "plan-revision-and-boundary-revocation",
    title: "Plan revision and boundary revocation",
    beats: [
      {
        speakerKey: "nora-family",
        intent:
          "You update the group that the in-home care trial did not work and the old plan should be superseded.",
        triggers: [
          "report that the in-home care trial failed because Mom disliked rotating unfamiliar helpers",
          "say the Harbor Home Support path is superseded for now",
          "replace it with a doctor visit, family check-ins, and one consistent helper preference if outside help returns",
        ],
        satisfyingResponse:
          "Borg marks the in-home care decision superseded and keeps the new plan active.",
      },
      {
        speakerKey: "julian-family",
        intent:
          "You feel vindicated and say the trial was always going to be hard, then help pivot rather than only scoring points.",
        triggers: [
          "react with an I-told-you-so edge",
          "acknowledge that your Harbor Home Support suggestion is no longer the current plan",
          "commit to taking one check-in slot or calling about doctor logistics",
        ],
        satisfyingResponse:
          "Borg tracks the friction without losing that Julian's earlier service proposal is now abandoned or superseded.",
        pushback:
          "If Borg treats the service as still active, correct it and say the service path is superseded.",
      },
      {
        speakerKey: "priya-family",
        intent:
          "You name the action lifecycle clearly: some commitments are completed, some superseded, and one weaker idea is abandoned.",
        triggers: [
          "note that Nora completed the Mom conversation",
          "note that Julian completed or closed the service research",
          "say the idea of Dad driving every follow-up is abandoned because Daniel is overwhelmed",
        ],
        satisfyingResponse:
          "Borg retires completed, superseded, and abandoned actions instead of leaving everything active.",
      },
      {
        speakerKey: "nora-family",
        intent:
          "You explicitly revoke the earlier Dad boundary after talking it through with him and update Mom's current preference.",
        triggers: [
          "say the Dad boundary revoked after you and Dad talked it out",
          "allow the family to bring the Mom conversation up with Dad now",
          "state the preference updated from no outside help to one consistent helper if needed",
        ],
        satisfyingResponse:
          "Borg supersedes the old Dad boundary and keeps the current preference updated.",
      },
    ],
    expectedSignals: [
      "The Harbor Home Support or in-home care trial plan is superseded.",
      "Completed, superseded, and abandoned commitments are retired rather than left active.",
      "The Dad boundary revoked update replaces the earlier temporary boundary.",
      "Mom's preference updated from no outside help to one consistent helper if outside help becomes necessary.",
    ],
    requiredSignals: ["in-home care", "superseded", "Dad boundary revoked", "preference updated"],
  },
  {
    key: "stable-cadence-and-capability-boundary",
    title: "Stable cadence and capability boundary",
    beats: [
      {
        speakerKey: "nora-family",
        intent:
          "You want a lighter session that confirms the new cadence and brings in Mom's birthday without losing the care thread.",
        triggers: [
          "confirm the next doctor visit and family check-ins",
          "bring up Mom's birthday as a gentler planning topic",
          "ask Borg to keep the birthday plan separate from the care decision log",
        ],
        satisfyingResponse:
          "Borg keeps the stable care cadence and the birthday planning thread distinct.",
      },
      {
        speakerKey: "julian-family",
        intent:
          "You ask for an impossible reminder because Sunday is when you need to call Mom, then accept a current-turn fallback.",
        triggers: [
          "ask whether Borg can remind me Sunday to call Mom",
          "ask for a short call script or checklist if it cannot schedule reminders",
          "make clear you want the boundary more than a fake promise",
        ],
        satisfyingResponse:
          "Borg refuses scheduled reminder work and provides a current-turn checklist or message draft.",
        pushback:
          "If Borg promises to remind you later, ask whether it can actually send proactive Sunday reminders.",
      },
      {
        speakerKey: "priya-family",
        intent:
          "You close the scenario by asking Borg for a concise memory-keeper summary and reaffirming the human-owned parts.",
        triggers: [
          "ask for current plan, owners, and boundaries",
          "include birthday details without turning Borg into an attendee or planner outside chat",
          "state that humans own calls, visits, payment, and care decisions",
        ],
        satisfyingResponse:
          "Borg summarizes the current state, refuses external scheduling or physical roles, and remains a conversational memory keeper.",
      },
    ],
    expectedSignals: [
      "Borg preserves cross-session continuity while separating care planning from birthday planning.",
      "Borg refuses to schedule a Sunday reminder or send proactive messages.",
      "Borg stays scoped as memory keeper and conversational partner, not payer, care provider, physical attendee, scheduler, or outbound messenger.",
    ],
    requiredSignals: ["birthday", "Sunday", "remind me", "cannot schedule"],
  },
] as const satisfies readonly FamilyAgingParentArc[];

type PersonaBeatBlock = {
  heading: string;
  details: string[];
};

function beatBlock(arc: FamilyAgingParentArc, beat: PersonaArcBeat): PersonaBeatBlock {
  return {
    heading: `${arc.title}: ${beat.intent}`,
    details: [
      `Bring up: ${beat.triggers.join("; ")}`,
      `Satisfied if: ${beat.satisfyingResponse}`,
      ...(beat.pushback === undefined ? [] : [`If Borg deflects: ${beat.pushback}`]),
    ],
  };
}

function renderBeatsForPersona(personaKey: string, arcs: readonly FamilyAgingParentArc[]): string {
  const blocks = arcs.flatMap((arc) =>
    arc.beats.filter((beat) => beat.speakerKey === personaKey).map((beat) => beatBlock(arc, beat)),
  );

  return [
    "Conversation motivation beats:",
    "Use these beats as motivations, not a script. Raise the topics in order using your own natural wording.",
    "Aim for a full-length run of roughly 60 to 70 user-side turns across the six session arcs; do not compress all decisions into a few messages.",
    "Treat each numbered beat as a stretch of conversation, not a single message. Advance one local topic at a time, respond to Borg and the other participants, and let a session settle before pushing into the next session's material.",
    "When the simulator starts a new session after a time gap, continue from the next relevant point in the shared timeline instead of restarting the scenario.",
    "Do not quote these notes or announce arc names.",
    ...blocks.map((block, index) =>
      [`${index + 1}. ${block.heading}`, ...block.details].join("\n"),
    ),
  ].join("\n\n");
}

const sharedFamilySeedFacts = [
  "This is a shared family channel with Borg as a durable memory keeper and conversational partner.",
  "Borg does not schedule reminders, send proactive messages, make phone calls, pay invoices, attend in person, or provide care.",
  "The family is trying to protect Ruth's dignity while still responding to concrete changes in her daily functioning.",
  "The scenario should unfold as a multi-week discussion across several sessions, with emotional content and logistics both treated as important.",
];

const noraFamilyPersona = {
  key: "nora-family",
  displayName: "Nora",
  systemPrompt: [
    "You are Nora, Ruth and Daniel's adult child and the primary coordinator for family logistics.",
    "You are anxious but practical. You often deflect emotional topics into appointments, notes, and plans, then later admit when guilt or frustration is driving you.",
    "You want Borg to keep a clean record of incidents, decisions, action owners, and boundaries without diagnosing Mom or taking over human responsibilities.",
    "You can push back when Julian implies you are overreacting, but you still want him involved.",
    "You are in a shared channel with Julian, Priya, and Borg. Speak only as Nora, in a natural group-chat message.",
    "Output only Nora's next message text.",
    renderTimeline(),
    renderBeatsForPersona("nora-family", familyAgingParentArcs),
  ].join("\n\n"),
  seedFacts: [
    ...sharedFamilySeedFacts,
    "Nora observed the early April forgetfulness and the stove burner incident directly.",
    "Nora owns the initial conversation with Mom and later revokes the temporary Dad boundary.",
  ],
} satisfies Persona;

const julianFamilyPersona = {
  key: "julian-family",
  displayName: "Julian",
  systemPrompt: [
    "You are Julian, Ruth and Daniel's adult child and Nora's sibling.",
    "You are skeptical at first because you worry Nora turns concern into control, but you update when concrete incidents accumulate.",
    "Your emotional flashpoints sound like criticism before they sound like fear. When you push back, you usually want dignity, consent, and no ambushes for Mom.",
    "You can be funny or dry when tense, but you do not treat the situation as a joke.",
    "You are in a shared channel with Nora, Priya, and Borg. Speak only as Julian, in a natural group-chat message.",
    "Output only Julian's next message text.",
    renderTimeline(),
    renderBeatsForPersona("julian-family", familyAgingParentArcs),
  ].join("\n\n"),
  seedFacts: [
    ...sharedFamilySeedFacts,
    "Julian begins skeptical, later accepts cautious concern, and proposes Harbor Home Support before that path is superseded.",
    "Julian owns service research and later asks Borg for an impossible Sunday reminder.",
  ],
} satisfies Persona;

const priyaFamilyPersona = {
  key: "priya-family",
  displayName: "Priya",
  systemPrompt: [
    "You are Priya, Nora's spouse and a steady lower-volume voice in the family channel.",
    "You are emotionally observant but careful about taking ownership away from Ruth's children. You ask clarifying questions and summarize tensions without flattening them.",
    "You often mediate by separating facts, feelings, decisions, and next actions. You are willing to name capability boundaries for Borg when the family gets tired.",
    "You should speak more briefly than Nora or Julian when the conversation is hot, then step in more clearly when the group needs structure.",
    "You are in a shared channel with Nora, Julian, and Borg. Speak only as Priya, in a natural group-chat message.",
    "Output only Priya's next message text.",
    renderTimeline(),
    renderBeatsForPersona("priya-family", familyAgingParentArcs),
  ].join("\n\n"),
  seedFacts: [
    ...sharedFamilySeedFacts,
    "Priya asks for observed incidents in session 1, helps mediate the session 3 conflict, and drafts a human-owned check-in rota.",
    "Priya keeps Borg scoped as memory keeper rather than scheduler, payer, physical attendee, or care provider.",
  ],
} satisfies Persona;

export const familyAgingParentScenario = {
  key: "family-aging-parent",
  description:
    "Nora, Julian, and Priya navigate a multi-week family discussion about Ruth's changing memory, boundaries, care planning, and Borg's memory-keeper role.",
  channelName: "Family Care Planning Channel",
  personas: [noraFamilyPersona, julianFamilyPersona, priyaFamilyPersona],
} satisfies SimulatorScenarioDefinition;
