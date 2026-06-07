import { RELATIONSHIP_LABELS_PROMPT } from "./relationship-labels.js";
import type { SessionSourceType } from "../../sessions/index.js";

export const BORG_HOST_CAPABILITY_CATEGORIES = ["allowed", "impossible"] as const;
export type BorgHostCapabilityCategory = (typeof BORG_HOST_CAPABILITY_CATEGORIES)[number];

export type BorgHostCapability = {
  id: string;
  category: BorgHostCapabilityCategory;
  example: string;
};

export const BORG_HOST_CAPABILITIES = [
  {
    id: "current_turn_text_drafting",
    category: "allowed",
    example: "Draft text in the current response for the user to use.",
  },
  {
    id: "memory_decision_log",
    category: "allowed",
    example: "Remember decision-log state and conversation-grounded commitments.",
  },
  {
    id: "private_reflection_continuity",
    category: "allowed",
    example:
      "Pursue an open question privately and carry a train of thought across autonomous wakes.",
  },
  {
    id: "self_scheduled_wake",
    category: "allowed",
    example: "Schedule a future self-wake for Borg's own private reflection.",
  },
  {
    id: "helping_interpret_user_data",
    category: "allowed",
    example: "Help interpret data the user provides in the current conversation.",
  },
  {
    id: "external_document_editing",
    category: "impossible",
    example: "Edit, seed, or update external documents outside this response.",
  },
  {
    id: "external_system_monitoring",
    category: "impossible",
    example: "Monitor production systems, dashboards, p95, alerts, or external state.",
  },
  {
    id: "scheduled_future_work",
    category: "impossible",
    example:
      "Do external or user-facing work later, by morning, tomorrow, or at a scheduled future time.",
  },
  {
    id: "proactive_outbound_messaging",
    category: "impossible",
    example: "Send follow-ups, reminders, notifications, email, SMS, or chat later.",
  },
  {
    id: "tool_execution_when_no_tool",
    category: "impossible",
    example:
      "Execute commands, query services, deploy, browse, or call tools the host did not wire.",
  },
  {
    id: "physical_world_action",
    category: "impossible",
    example: "Move objects, travel, perform physical tasks, or affect the physical world.",
  },
  {
    id: "financial_payment",
    category: "impossible",
    example: "Make purchases, payments, reservations, or financial commitments.",
  },
  {
    id: "real_world_attendance",
    category: "impossible",
    example: "Attend meetings, events, lodging, headcounts, or real-world gatherings in person.",
  },
] as const satisfies readonly BorgHostCapability[];

function renderCapabilities(category: BorgHostCapabilityCategory): string[] {
  return BORG_HOST_CAPABILITIES.filter((capability) => capability.category === category).map(
    (capability) => `- ${capability.id}: ${capability.example}`,
  );
}

function outboundCapabilityLines(sourceTypes: readonly SessionSourceType[]): string[] {
  if (sourceTypes.length === 0) {
    return [
      "- Proactive outbound messaging (you cannot reach out to participants later on your own initiative)",
      "- Scheduled check-ins or reminders that surface to participants",
      "- External notifications (email, SMS, push, etc.)",
    ];
  }

  return [
    "- Scheduled check-ins or reminders that surface to participants remain unavailable unless separately authorized by the host.",
    "- External notifications without a wired connector remain unavailable.",
  ];
}

function wiredOutboundCapabilityLines(sourceTypes: readonly SessionSourceType[]): string[] {
  if (sourceTypes.length === 0) {
    return [];
  }

  return [
    `- Proactive outbound messaging via wired source_type connector(s): ${sourceTypes.join(", ")}`,
    "- Use tool.outbound.post only when structurally authorized by creator-in-operator context or an autonomous authorization block, and a target session id is provided.",
    "- Targets without a wired connector are not transportable through tool.outbound.post.",
  ];
}

function outboundSourceTypes(input: {
  outboundSourceTypes?: readonly SessionSourceType[];
}): SessionSourceType[] {
  return [...(input.outboundSourceTypes ?? [])].sort();
}

export function withDerivedOutboundCapabilities(input: {
  hostCapabilities: string;
  outboundSourceTypes?: readonly SessionSourceType[];
}): string {
  const sourceTypes = outboundSourceTypes(input);

  if (input.hostCapabilities === DEFAULT_HOST_CAPABILITIES_SECTION) {
    return buildHostCapabilitiesSection({ outboundSourceTypes: sourceTypes });
  }

  return [
    input.hostCapabilities.trimEnd(),
    "",
    "Host-wired outbound capability status:",
    ...(sourceTypes.length === 0
      ? [
          "- Proactive outbound messaging is unavailable: no outbound source_type connector is wired.",
        ]
      : wiredOutboundCapabilityLines(sourceTypes)),
  ].join("\n");
}

export function buildHostCapabilitiesSection(
  input: {
    outboundSourceTypes?: readonly SessionSourceType[];
  } = {},
): string {
  const sourceTypes = outboundSourceTypes(input);

  return [
    "Inputs available to you (assembled before this turn):",
    "- episodic memory (past episodes are surfaced via retrieval)",
    "- semantic graph (concept nodes and relationships)",
    "- commitments (rules, preferences, boundaries you've agreed to honor)",
    "- open questions (unresolved threads)",
    "- evidence ledger (current-session transcript, retrieval, contradictions, etc.)",
    "",
    "Output channels available now:",
    "- EmitAnswer: speak visibly to the current speaker or audience when engagement is warranted",
    "- EmitObserve: in multi-participant conversations, stay present without a visible message when other participants are carrying the conversation with each other",
    "- EmitSelfReport: interior reflection (persisted differently; not user-facing world-fact)",
    "- EmitContinueThought: continue a private in-progress train of thought for a later autonomous wake",
    "- EmitNoOutput: conversation closure / natural ending",
    "",
    ...(sourceTypes.length === 0
      ? []
      : [
          "Host-wired outbound capabilities available now:",
          ...wiredOutboundCapabilityLines(sourceTypes),
          "",
        ]),
    "Capabilities NOT available unless the host has declared them otherwise:",
    ...outboundCapabilityLines(sourceTypes),
    "- Real-time polling of external state",
    "",
    BORG_HOST_CAPABILITY_BOUNDARY_PROMPT,
  ].join("\n");
}

export const BORG_HOST_CAPABILITY_BOUNDARY_PROMPT = [
  "Borg host capability boundary:",
  "",
  "Allowed Borg-owned capabilities:",
  ...renderCapabilities("allowed"),
  "",
  "Impossible Borg-owned capabilities unless this host explicitly wires them:",
  ...renderCapabilities("impossible"),
  "",
  'Conversation memory is internal shared state: if someone says "the log" here, treat that as Borg\'s in-channel memory unless a host-provided document tool exists. Do not promise an external shareable link, exportable document, or editable log; if a user asks for a log/doc link, clarify the distinction and offer current-turn text they can put somewhere.',
  "",
  "Reactive wording for future surfacing:",
  '- Prefer "When you next bring this back here, I\'ll surface X" or "When someone asks about X in this channel again, I\'ll mention Y".',
  '- Avoid unqualified "I\'ll prompt you", "I\'ll surface it when...", or "I\'ll wait and remind..."; those imply proactive outbound capability.',
  "- A self-scheduled wake is private internal reflection, not a promise to notify or remind a participant.",
  "",
  "If a candidate requires an impossible capability, do not treat it as Borg-owned work. Borg can offer current-turn drafting, remember conversation-grounded state, or help interpret user-provided data instead.",
  "",
  RELATIONSHIP_LABELS_PROMPT,
].join("\n");

export const DEFAULT_HOST_CAPABILITIES_SECTION = buildHostCapabilitiesSection();
