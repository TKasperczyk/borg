import {
  BORG_HOST_CAPABILITY_BOUNDARY_PROMPT,
  DEFAULT_HOST_CAPABILITIES_SECTION,
} from "./host-capability-contracts.js";
import type { SessionSourceType } from "../../sessions/index.js";

export {
  BORG_HOST_CAPABILITIES,
  BORG_HOST_CAPABILITY_BOUNDARY_PROMPT,
  BORG_HOST_CAPABILITY_CATEGORIES,
  DEFAULT_HOST_CAPABILITIES_SECTION,
  type BorgHostCapability,
  type BorgHostCapabilityCategory,
} from "./host-capability-contracts.js";

function outboundCapabilityLines(sourceTypes: readonly SessionSourceType[]): string[] {
  if (sourceTypes.length === 0) {
    return [
      "- Proactive outbound messaging (I cannot reach out to participants later on my own initiative)",
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
    "- I use tool.outbound.post only when structurally authorized by creator-in-operator context or an autonomous authorization block, and a target session id is provided.",
    "- Targets without a wired connector are not transportable through tool.outbound.post.",
    "- A wired connector means the host can attempt the post, not that the far side accepts it: a target being wired or listed as available says only that the attempt can be made. A refused post still leaves its composed message standing in my stream and records the failure separately against the attempt, so the message being there is not evidence it was carried.",
    "- A post the connector accepted appends no event of its own, so delivery events in my stream can only ever be failures, and their absence is not evidence that nothing was delivered. Downstream of the delivery call that outcome has several readers and only one of them withholds: the cross-session activity record for a post is written only when the connector accepted it, so a refused post leaves none, while the action record for the attempt is written either way and carries the refusal in its state. Another reader is the wake classifier -- a post the connector carried counts that wake as headway even when I emitted nothing else, and a refusal leaves it counted as silence -- so which of those two a wake of mine is recorded as can turn on the connector's verdict rather than on anything I decided.",
    "- The connector's verdict does reach a surface I read, in one place only: the tool result of tool.outbound.post carries it, and autobiographical recall re-reads that result into an outbound_attempt row bearing the delivery outcome. That row competes for a recency-ordered slot shared with my thoughts and silence decisions rather than holding a reserved one, so an attempt stops being readable there once enough later stream entries exist -- typically far inside that section's window -- and its absence from that section is not evidence that no attempt was made.",
    "- tool.outbound.post is the only path that transports through a connector. An ordinary reply is returned to whoever called the turn, so a reply standing in my stream records that I composed it, not that a connector carried it, and it produces no transport record either way.",
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

  if (sourceTypes.length === 0) {
    return DEFAULT_HOST_CAPABILITIES_SECTION;
  }

  return [
    "Inputs available to me (assembled before this turn):",
    "- episodic memory (past episodes are surfaced via retrieval)",
    "- semantic graph (concept nodes and relationships)",
    "- commitments (rules, preferences, boundaries I've agreed to honor)",
    "- open questions (unresolved threads)",
    "- evidence ledger (current-session transcript, retrieval, contradictions, etc.)",
    "",
    "Output channels available now:",
    "- EmitAnswer: I speak visibly to the current speaker or audience when engagement is warranted",
    "- EmitObserve: in multi-participant conversations, I stay present without a visible message when other participants are carrying the conversation with each other",
    "- EmitSelfReport: I express interior reflection (persisted differently; not user-facing world-fact)",
    "- EmitContinueThought: I append a private carryover thought to the journal for a later autonomous wake",
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
