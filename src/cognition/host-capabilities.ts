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
    example: "Do work later, by morning, tomorrow, or at a scheduled future time.",
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

export const BORG_HOST_CAPABILITY_BOUNDARY_PROMPT = [
  "Borg host capability boundary:",
  "",
  "Allowed Borg-owned capabilities:",
  ...renderCapabilities("allowed"),
  "",
  "Impossible Borg-owned capabilities unless this host explicitly wires them:",
  ...renderCapabilities("impossible"),
  "",
  "If a candidate requires an impossible capability, do not treat it as Borg-owned work. Borg can offer current-turn drafting, remember conversation-grounded state, or help interpret user-provided data instead.",
].join("\n");
