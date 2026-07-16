import type { PromptKey } from "./registry.js";

export const PROMPT_SURFACES = {
  baseDirect: "base_direct",
  baseUntrustedSections: "base_untrusted_sections",
  baseTrustedGuidanceSections: "base_trusted_guidance_sections",
  cacheableStaticPrefix: "cacheable_static_prefix",
  cacheableDynamic: "cacheable_dynamic",
  cacheableTrustedDynamicSections: "cacheable_trusted_dynamic_sections",
  finalizerStaticSystem: "finalizer_static_system",
  finalizerDynamicSystem: "finalizer_dynamic_system",
  s2PlannerSystem: "s2_planner_system",
  evidenceLedgerFraming: "evidence_ledger_framing",
  compactPlannerFraming: "compact_planner_framing",
  commitmentRegenerationInstruction: "commitment_regeneration_instruction",
  directedOutboundFraming: "directed_outbound_framing",
} as const;

export type PromptSurface = (typeof PROMPT_SURFACES)[keyof typeof PROMPT_SURFACES];

export type PromptSurfaceSource = {
  file: string;
  exportName?: string;
};

export type PromptSurfacePlacement = {
  surface: PromptSurface;
  order: number;
};

export type PromptSurfaceRenderContext = {
  renderBlock: (id: string) => string | null;
};

export type PromptSurfaceAdditionalSection = {
  blockId: string;
  text: string;
};

export type PromptSurfaceBlock = {
  id: string;
  owner: string;
  purpose: string;
  renderCondition: string;
  source: PromptSurfaceSource;
  tag?: string;
  editableKey?: PromptKey;
  approxLines: number | null;
  approxChars: number | null;
  surfaces: readonly PromptSurfacePlacement[];
  render: (context: PromptSurfaceRenderContext) => string | null;
};

function block(
  input: Omit<PromptSurfaceBlock, "render" | "approxLines" | "approxChars"> & {
    approxLines?: number | null;
    approxChars?: number | null;
    render?: PromptSurfaceBlock["render"];
  },
): PromptSurfaceBlock {
  return {
    ...input,
    approxLines: input.approxLines ?? null,
    approxChars: input.approxChars ?? null,
    render: input.render ?? ((context) => context.renderBlock(input.id)),
  };
}

export const PROMPT_SURFACE_BLOCKS = [
  block({
    id: "base_identity_preamble",
    owner: "cognition.prompts",
    purpose: "Root identity contract.",
    renderCondition: "always",
    source: {
      file: "src/cognition/prompts/base-identity.ts",
      exportName: "BASE_IDENTITY_PREAMBLE",
    },
    editableKey: "base_identity_preamble",
    surfaces: [
      { surface: PROMPT_SURFACES.baseDirect, order: 10 },
      { surface: PROMPT_SURFACES.cacheableStaticPrefix, order: 10 },
    ],
  }),
  block({
    id: "self_architecture",
    owner: "cognition.prompts",
    purpose: "Memory architecture and cognition posture.",
    renderCondition: "always",
    source: {
      file: "src/cognition/prompts/base-identity.ts",
      exportName: "SELF_ARCHITECTURE_SECTION",
    },
    editableKey: "self_architecture",
    surfaces: [
      { surface: PROMPT_SURFACES.baseDirect, order: 20 },
      { surface: PROMPT_SURFACES.cacheableStaticPrefix, order: 20 },
    ],
  }),
  block({
    id: "voice_and_posture",
    owner: "cognition.prompts",
    purpose: "Default voice and interaction stance.",
    renderCondition: "always",
    source: {
      file: "src/cognition/prompts/base-identity.ts",
      exportName: "VOICE_AND_POSTURE_SECTION",
    },
    editableKey: "voice_and_posture",
    surfaces: [
      { surface: PROMPT_SURFACES.baseDirect, order: 30 },
      { surface: PROMPT_SURFACES.cacheableStaticPrefix, order: 30 },
    ],
  }),
  block({
    id: "epistemic_posture",
    owner: "cognition.prompts",
    purpose: "Grounding and uncertainty posture.",
    renderCondition: "always",
    source: {
      file: "src/cognition/prompts/base-identity.ts",
      exportName: "EPISTEMIC_POSTURE_SECTION",
    },
    editableKey: "epistemic_posture",
    surfaces: [
      { surface: PROMPT_SURFACES.baseDirect, order: 40 },
      { surface: PROMPT_SURFACES.cacheableStaticPrefix, order: 40 },
    ],
  }),
  block({
    id: "identity_posture",
    owner: "cognition.prompts",
    purpose: "Identity-bearing memory and self-reference posture.",
    renderCondition: "always",
    source: {
      file: "src/cognition/prompts/base-identity.ts",
      exportName: "IDENTITY_POSTURE_SECTION",
    },
    editableKey: "identity_posture",
    surfaces: [
      { surface: PROMPT_SURFACES.baseDirect, order: 50 },
      { surface: PROMPT_SURFACES.cacheableStaticPrefix, order: 50 },
    ],
  }),
  block({
    id: "participation_posture",
    owner: "cognition.prompts",
    purpose: "Participation and presence posture.",
    renderCondition: "always",
    source: {
      file: "src/cognition/prompts/participation.ts",
      exportName: "PARTICIPATION_POSTURE_SECTION",
    },
    editableKey: "participation_posture",
    surfaces: [
      { surface: PROMPT_SURFACES.baseDirect, order: 60 },
      { surface: PROMPT_SURFACES.cacheableStaticPrefix, order: 60 },
    ],
  }),
  block({
    id: "loop_breaking_posture",
    owner: "cognition.prompts",
    purpose: "No-output and closure-loop posture.",
    renderCondition: "always",
    source: {
      file: "src/cognition/prompts/participation.ts",
      exportName: "LOOP_BREAKING_POSTURE_SECTION",
    },
    surfaces: [
      { surface: PROMPT_SURFACES.baseDirect, order: 70 },
      { surface: PROMPT_SURFACES.cacheableStaticPrefix, order: 70 },
    ],
  }),
  block({
    id: "base_untrusted_data_block",
    owner: "cognition.deliberation",
    purpose: "Wrapper for prompt-visible dynamic memory/state data.",
    renderCondition: "renders when at least one untrusted section has content",
    source: {
      file: "src/cognition/prompts/base-identity.ts",
      exportName: "UNTRUSTED_DATA_PREAMBLE",
    },
    surfaces: [
      { surface: PROMPT_SURFACES.baseDirect, order: 80 },
      { surface: PROMPT_SURFACES.cacheableDynamic, order: 20 },
    ],
  }),
  block({
    id: "base_trusted_guidance_block",
    owner: "cognition.deliberation",
    purpose: "Wrapper for trusted standing guidance in the direct base prompt.",
    renderCondition: "renders when trusted guidance sections have content",
    source: {
      file: "src/cognition/prompts/base-identity.ts",
      exportName: "TRUSTED_GUIDANCE_PREAMBLE",
    },
    surfaces: [{ surface: PROMPT_SURFACES.baseDirect, order: 90 }],
  }),
  block({
    id: "current_user_message_reminder",
    owner: "cognition.prompts",
    purpose: "Reminder that the current user message is authoritative for this turn.",
    renderCondition: "always",
    source: {
      file: "src/cognition/prompts/base-identity.ts",
      exportName: "CURRENT_USER_MESSAGE_REMINDER",
    },
    surfaces: [
      { surface: PROMPT_SURFACES.baseDirect, order: 100 },
      { surface: PROMPT_SURFACES.cacheableDynamic, order: 30 },
    ],
  }),
  block({
    id: "group_chat_sender_scoping_reminder",
    owner: "cognition.prompts",
    purpose: "Group-chat sender scoping reminder.",
    renderCondition: "group audience only",
    source: {
      file: "src/cognition/prompts/participation.ts",
      exportName: "GROUP_CHAT_SENDER_SCOPING_REMINDER",
    },
    surfaces: [
      { surface: PROMPT_SURFACES.baseDirect, order: 110 },
      { surface: PROMPT_SURFACES.cacheableDynamic, order: 40 },
    ],
  }),
  block({
    id: "trusted_guidance_preamble",
    owner: "cognition.prompts",
    purpose: "Trusted guidance boundary preamble in the cacheable static prefix.",
    renderCondition: "always",
    source: {
      file: "src/cognition/prompts/base-identity.ts",
      exportName: "TRUSTED_GUIDANCE_PREAMBLE",
    },
    surfaces: [{ surface: PROMPT_SURFACES.cacheableStaticPrefix, order: 80 }],
  }),
  block({
    id: "borg_host_capabilities",
    owner: "cognition.prompts",
    purpose: "Host capability contract.",
    renderCondition: "always; content may be configured or overridden",
    source: {
      file: "src/cognition/prompts/host-capability-contracts.ts",
      exportName: "DEFAULT_HOST_CAPABILITIES_SECTION",
    },
    tag: "borg_host_capabilities",
    editableKey: "host_capabilities",
    surfaces: [
      { surface: PROMPT_SURFACES.baseTrustedGuidanceSections, order: 70 },
      { surface: PROMPT_SURFACES.cacheableStaticPrefix, order: 90 },
    ],
  }),
  block({
    id: "base_trusted_dynamic_guidance_block",
    owner: "cognition.deliberation",
    purpose: "Trusted dynamic guidance block in the cacheable finalizer dynamic content.",
    renderCondition: "renders when trusted dynamic guidance sections have content",
    source: { file: "src/cognition/deliberation/prompt/system-prompt.ts" },
    surfaces: [{ surface: PROMPT_SURFACES.cacheableDynamic, order: 10 }],
  }),
  block({
    id: "borg_current_time",
    owner: "cognition.deliberation",
    purpose: "Turn current-time anchor for interpreting recalled-memory recency labels.",
    renderCondition: "nowMs provided",
    source: {
      file: "src/cognition/deliberation/prompt/system-prompt.ts",
      exportName: "renderCurrentTimeSection",
    },
    tag: "borg_current_time",
    surfaces: [
      { surface: PROMPT_SURFACES.baseTrustedGuidanceSections, order: 5 },
      { surface: PROMPT_SURFACES.cacheableTrustedDynamicSections, order: 5 },
    ],
  }),
  block({
    id: "borg_participation_policy",
    owner: "cognition.sessions",
    purpose: "Session participation policy guidance.",
    renderCondition: "non-active participation policy",
    source: {
      file: "src/cognition/deliberation/prompt/system-prompt.ts",
      exportName: "renderParticipationPolicy",
    },
    tag: "borg_participation_policy",
    surfaces: [
      { surface: PROMPT_SURFACES.baseTrustedGuidanceSections, order: 10 },
      { surface: PROMPT_SURFACES.cacheableTrustedDynamicSections, order: 10 },
    ],
  }),
  block({
    id: "borg_creator_identity",
    owner: "cognition.authority",
    purpose: "Creator identity and authority context.",
    renderCondition: "creator identity present",
    source: {
      file: "src/cognition/deliberation/prompt/system-prompt.ts",
      exportName: "renderCreatorIdentity",
    },
    tag: "borg_creator_identity",
    surfaces: [
      { surface: PROMPT_SURFACES.baseTrustedGuidanceSections, order: 20 },
      { surface: PROMPT_SURFACES.cacheableTrustedDynamicSections, order: 20 },
    ],
  }),
  block({
    id: "borg_memory_disclosure_guidance",
    owner: "retrieval",
    purpose: "Disclosure and common-ground guidance for recalled memory.",
    renderCondition: "always",
    source: {
      file: "src/retrieval/recall-context.ts",
      exportName: "MEMORY_DISCLOSURE_GUIDANCE_FOR_MODEL",
    },
    tag: "borg_memory_disclosure_guidance",
    surfaces: [
      { surface: PROMPT_SURFACES.baseTrustedGuidanceSections, order: 30 },
      { surface: PROMPT_SURFACES.cacheableTrustedDynamicSections, order: 30 },
    ],
  }),
  block({
    id: "borg_standing_with_audience",
    owner: "cognition.deliberation",
    purpose: "Audience standing, authority, directives, commitments, and cross-session awareness.",
    renderCondition: "always",
    source: {
      file: "src/cognition/deliberation/prompt/system-prompt.ts",
      exportName: "buildStandingWithAudienceSection",
    },
    tag: "borg_standing_with_audience",
    surfaces: [
      { surface: PROMPT_SURFACES.baseTrustedGuidanceSections, order: 40 },
      { surface: PROMPT_SURFACES.cacheableTrustedDynamicSections, order: 40 },
    ],
  }),
  block({
    id: "borg_autonomous_reflection",
    owner: "autonomy",
    purpose: "Autonomous reflection posture and reachable outbound targets.",
    renderCondition: "autonomous turns only",
    source: {
      file: "src/cognition/deliberation/prompt/system-prompt.ts",
      exportName: "buildAutonomousOutboundAuthorizationSection",
    },
    tag: "borg_autonomous_reflection",
    surfaces: [
      { surface: PROMPT_SURFACES.baseTrustedGuidanceSections, order: 50 },
      { surface: PROMPT_SURFACES.cacheableTrustedDynamicSections, order: 50 },
    ],
  }),
  block({
    id: "borg_held_preferences",
    owner: "memory.self",
    purpose: "Established self values and traits.",
    renderCondition: "established values or traits present",
    source: {
      file: "src/cognition/deliberation/prompt/system-prompt.ts",
      exportName: "summarizeHeldPreferences",
    },
    tag: "borg_held_preferences",
    surfaces: [
      { surface: PROMPT_SURFACES.baseTrustedGuidanceSections, order: 60 },
      { surface: PROMPT_SURFACES.cacheableTrustedDynamicSections, order: 60 },
    ],
  }),
  block({
    id: "borg_procedural_guidance",
    owner: "memory.procedural",
    purpose: "Selected procedural skill guidance.",
    renderCondition: "problem-solving mode",
    source: {
      file: "src/cognition/deliberation/prompt/system-prompt.ts",
      exportName: "summarizeSelectedSkill",
    },
    tag: "borg_procedural_guidance",
    surfaces: [
      { surface: PROMPT_SURFACES.baseTrustedGuidanceSections, order: 80 },
      { surface: PROMPT_SURFACES.cacheableTrustedDynamicSections, order: 70 },
    ],
  }),
  block({
    id: "borg_discourse_control",
    owner: "cognition.generation",
    purpose: "Closure and discourse-control guidance.",
    renderCondition: "non-autonomous discourse stop/closure state",
    source: {
      file: "src/cognition/deliberation/prompt/system-prompt.ts",
      exportName: "summarizeDiscourseControl",
    },
    tag: "borg_discourse_control",
    surfaces: [
      { surface: PROMPT_SURFACES.baseTrustedGuidanceSections, order: 90 },
      { surface: PROMPT_SURFACES.cacheableTrustedDynamicSections, order: 80 },
    ],
  }),
  block({
    id: "borg_mechanism_evidence",
    owner: "cognition.generation",
    purpose: "Model-visible evidence about recent turn machinery outcomes.",
    renderCondition: "recent suppression or regeneration evidence exists",
    source: {
      file: "src/cognition/deliberation/prompt/system-prompt.ts",
      exportName: "summarizeMechanismEvidence",
    },
    tag: "borg_mechanism_evidence",
    surfaces: [
      { surface: PROMPT_SURFACES.baseTrustedGuidanceSections, order: 85 },
      { surface: PROMPT_SURFACES.cacheableTrustedDynamicSections, order: 75 },
    ],
  }),
  block({
    id: "borg_frame_anomaly_gate",
    owner: "cognition.frame_anomaly",
    purpose: "Frame-anomaly safety guidance for current user message.",
    renderCondition: "frame anomaly present",
    source: {
      file: "src/cognition/deliberation/prompt/system-prompt.ts",
      exportName: "summarizeFrameAnomalyGate",
    },
    tag: "borg_frame_anomaly_gate",
    surfaces: [
      { surface: PROMPT_SURFACES.baseTrustedGuidanceSections, order: 100 },
      { surface: PROMPT_SURFACES.cacheableTrustedDynamicSections, order: 90 },
    ],
  }),
  ...(
    [
      [
        "borg_self_snapshot",
        "memory.self",
        "Self snapshot.",
        "always unless only held preferences render",
      ],
      ["borg_executive_focus", "executive", "Selected executive focus.", "selected goal present"],
      [
        "borg_current_period",
        "memory.self",
        "Current autobiographical period.",
        "current period present",
      ],
      ["borg_recent_growth", "memory.self", "Recent growth markers.", "growth markers present"],
      ["borg_working_state", "memory.working", "Working memory state.", "always"],
      [
        "borg_recent_completed_actions",
        "memory.actions",
        "Recent completed actions.",
        "ledger inactive and completed actions present",
      ],
      [
        "borg_affective_trajectory",
        "memory.affective",
        "Affective trajectory.",
        "affective history present",
      ],
      [
        "borg_audience_profile",
        "memory.social",
        "Audience and participant social profiles.",
        "social profile or participant profiles present",
      ],
      [
        "borg_thread_roster",
        "cognition.perception",
        "Prompt-visible thread roster.",
        "thread roster present",
      ],
      [
        "borg_retrieved_evidence",
        "retrieval",
        "Legacy retrieved evidence.",
        "evidence ledger inactive",
      ],
      [
        "borg_retrieval_confidence",
        "retrieval",
        "Retrieval confidence signal.",
        "retrieval confidence present",
      ],
      [
        "contradiction_signal",
        "retrieval",
        "Contradiction routing signal.",
        "contradiction signal present",
      ],
      ["borg_open_questions", "memory.self", "Open questions.", "reflective mode only"],
      [
        "borg_pending_corrections",
        "memory.review_queue",
        "Pending corrections.",
        "pending corrections present",
      ],
      [
        "borg_autonomy_trigger",
        "autonomy",
        "Autonomy wake trigger context.",
        "autonomy trigger present",
      ],
    ] as const
  ).map(([id, owner, purpose, renderCondition], index) =>
    block({
      id,
      owner,
      purpose,
      renderCondition,
      source: { file: "src/cognition/deliberation/prompt/system-prompt.ts" },
      tag: id,
      surfaces: [{ surface: PROMPT_SURFACES.baseUntrustedSections, order: (index + 1) * 10 }],
    }),
  ),
  block({
    id: "finalizer_emission_protocol",
    owner: "cognition.deliberation",
    purpose: "Terminal emission-tool protocol and common finalizer instructions.",
    renderCondition: "always on finalizer calls",
    source: {
      file: "src/cognition/deliberation/finalizer.ts",
      exportName: "buildEmissionFinalizerInstructions",
    },
    surfaces: [{ surface: PROMPT_SURFACES.finalizerStaticSystem, order: 10 }],
  }),
  block({
    id: "finalizer_cacheable_static_prefix",
    owner: "cognition.deliberation",
    purpose: "Cacheable base static prompt prefix appended after finalizer protocol.",
    renderCondition: "cacheable base system prompt present",
    source: { file: "src/cognition/deliberation/finalizer.ts" },
    surfaces: [{ surface: PROMPT_SURFACES.finalizerStaticSystem, order: 20 }],
  }),
  block({
    id: "finalizer_base_dynamic_prompt",
    owner: "cognition.deliberation",
    purpose: "Base dynamic system prompt text for finalizer.",
    renderCondition: "always on finalizer calls",
    source: { file: "src/cognition/deliberation/finalizer.ts" },
    surfaces: [{ surface: PROMPT_SURFACES.finalizerDynamicSystem, order: 10 }],
  }),
  block({
    id: "s2_planner_base_system_prompt",
    owner: "cognition.deliberation",
    purpose: "Base prompt passed into S2 planner.",
    renderCondition: "always on S2 planner calls",
    source: { file: "src/cognition/deliberation/s2-planner.ts" },
    surfaces: [{ surface: PROMPT_SURFACES.s2PlannerSystem, order: 10 }],
  }),
  block({
    id: "borg_voice_anchors",
    owner: "cognition.deliberation",
    purpose: "Planner-specific voice anchors.",
    renderCondition: "S2 planner with established value anchors",
    source: {
      file: "src/cognition/deliberation/prompt/voice-anchors.ts",
      exportName: "summarizeVoiceAnchors",
    },
    tag: "borg_voice_anchors",
    surfaces: [{ surface: PROMPT_SURFACES.s2PlannerSystem, order: 20 }],
  }),
  block({
    id: "borg_unresolved_contradiction_open_questions",
    owner: "cognition.deliberation",
    purpose: "Forced-S2 planner note for unresolved contradiction open questions.",
    renderCondition: "forced S2 routing by open-question contradiction",
    source: {
      file: "src/cognition/deliberation/deliberator.ts",
      exportName: "renderForcedContradictionOpenQuestionsPrompt",
    },
    tag: "borg_unresolved_contradiction_open_questions",
    surfaces: [{ surface: PROMPT_SURFACES.s2PlannerSystem, order: 50 }],
  }),
  block({
    id: "s2_planner_autonomous_want",
    owner: "cognition.deliberation",
    purpose: "Autonomous S2 planner first question for naming any want before weighing.",
    renderCondition: "autonomous S2 planner calls",
    source: { file: "src/cognition/deliberation/s2-planner.ts" },
    surfaces: [{ surface: PROMPT_SURFACES.s2PlannerSystem, order: 55 }],
  }),
  block({
    id: "s2_planner_directive",
    owner: "cognition.deliberation",
    purpose: "Directive to emit exactly one structured S2 plan.",
    renderCondition: "always on S2 planner calls",
    source: { file: "src/cognition/deliberation/s2-planner.ts" },
    surfaces: [{ surface: PROMPT_SURFACES.s2PlannerSystem, order: 60 }],
  }),
  block({
    id: "borg_additional_retrieval",
    owner: "retrieval",
    purpose: "Secondary S2 retrieval results for the finalizer.",
    renderCondition: "S2 finalizer only",
    source: { file: "src/cognition/deliberation/deliberator.ts" },
    tag: "borg_additional_retrieval",
    surfaces: [{ surface: PROMPT_SURFACES.finalizerDynamicSystem, order: 40 }],
  }),
  block({
    id: "borg_s2_plan",
    owner: "cognition.deliberation",
    purpose: "Planner advisory context passed to finalizer.",
    renderCondition: "S2 finalizer with non-empty plan",
    source: {
      file: "src/cognition/deliberation/prompt/plan-rendering.ts",
      exportName: "formatTurnPlanForPrompt",
    },
    tag: "borg_s2_plan",
    surfaces: [{ surface: PROMPT_SURFACES.finalizerDynamicSystem, order: 50 }],
  }),
  block({
    id: "borg_session_reentry_continuity",
    owner: "cognition.session_reentry",
    purpose: "Trusted first-turn continuity guidance from durable shared state.",
    renderCondition: "first user turn with active prior shared state for the audience",
    source: {
      file: "src/cognition/session-reentry-continuity.ts",
      exportName: "buildSessionReentryContinuityPrompt",
    },
    tag: "borg_session_reentry_continuity",
    surfaces: [
      { surface: PROMPT_SURFACES.finalizerDynamicSystem, order: 20 },
      { surface: PROMPT_SURFACES.s2PlannerSystem, order: 30 },
    ],
  }),
  block({
    id: "borg_evidence_ledger",
    owner: "cognition.evidence_ledger",
    purpose: "Full finalizer evidence ledger framing.",
    renderCondition: "evidence ledger active",
    source: {
      file: "src/cognition/evidence-ledger/finalizer-ledger.ts",
      exportName: "renderEvidenceLedger",
    },
    tag: "borg_evidence_ledger",
    surfaces: [
      { surface: PROMPT_SURFACES.finalizerDynamicSystem, order: 30 },
      { surface: PROMPT_SURFACES.evidenceLedgerFraming, order: 10 },
    ],
  }),
  block({
    id: "borg_compact_planner_ledger",
    owner: "cognition.evidence_ledger",
    purpose: "Compact S2 planner evidence ledger framing.",
    renderCondition: "S2 planner with evidence ledger",
    source: {
      file: "src/cognition/evidence-ledger/compact-planner.ts",
      exportName: "buildCompactPlannerLedgerPrompt",
    },
    tag: "borg_compact_planner_ledger",
    surfaces: [
      { surface: PROMPT_SURFACES.s2PlannerSystem, order: 40 },
      { surface: PROMPT_SURFACES.compactPlannerFraming, order: 10 },
    ],
  }),
  block({
    id: "borg_commitment_regeneration_instruction",
    owner: "cognition.commitments",
    purpose: "One-shot finalizer regeneration instruction after critical commitment violation.",
    renderCondition: "critical commitment guard violation requiring regeneration",
    source: {
      file: "src/cognition/commitments/guard-runner.ts",
      exportName: "buildRegenerationPromptSection",
    },
    tag: "borg_commitment_regeneration_instruction",
    surfaces: [
      { surface: PROMPT_SURFACES.finalizerDynamicSystem, order: 60 },
      { surface: PROMPT_SURFACES.commitmentRegenerationInstruction, order: 10 },
    ],
  }),
  block({
    id: "finalizer_invalid_tool_retry_instruction",
    owner: "cognition.deliberation",
    purpose:
      "Invalid-tool corrective carrier. buildFinalizerCallOptions INTERCEPTS this section id " +
      "and routes its text into a trailing message adjacent to generation (not the system " +
      "tail, where it sat behind the transcript and went unread). The deliberator is the only " +
      "PRODUCTION caller and always strips it before render, so it never renders on this " +
      "system surface in production. The placement below still serves a direct runFinalizer " +
      "caller (e.g. tests) that supplies this section id -- such a caller would system-render it.",
    renderCondition:
      "finalizer retry after invalid terminal tool emission (intercepted; see purpose)",
    source: {
      file: "src/cognition/deliberation/deliberator.ts",
      exportName: "buildInvalidToolFinalizerRetryPromptSection",
    },
    surfaces: [{ surface: PROMPT_SURFACES.finalizerDynamicSystem, order: 70 }],
  }),
  block({
    id: "borg_directed_outbound_instruction",
    owner: "outbound",
    purpose: "Directed outbound user-message framing.",
    renderCondition: "directed outbound turn",
    source: {
      file: "src/outbound/outbound-turn.ts",
      exportName: "formatDirectedOutboundInstruction",
    },
    tag: "borg_directed_outbound_instruction",
    surfaces: [{ surface: PROMPT_SURFACES.directedOutboundFraming, order: 10 }],
  }),
] as const satisfies readonly PromptSurfaceBlock[];

export function promptSurfaceBlocksForSurface(surface: PromptSurface): PromptSurfaceBlock[] {
  return PROMPT_SURFACE_BLOCKS.filter((entry) =>
    entry.surfaces.some((placement) => placement.surface === surface),
  ).sort((left, right) => {
    const leftOrder = left.surfaces.find((placement) => placement.surface === surface)?.order ?? 0;
    const rightOrder =
      right.surfaces.find((placement) => placement.surface === surface)?.order ?? 0;

    return leftOrder - rightOrder;
  });
}

export function renderPromptSurface(
  surface: PromptSurface,
  context: PromptSurfaceRenderContext,
): string | null {
  const rendered = promptSurfaceBlocksForSurface(surface)
    .map((entry) => entry.render(context))
    .filter((section): section is string => section !== null);

  return rendered.length === 0 ? null : rendered.join("\n\n");
}

export function renderPromptSurfaceAdditionalBlock(
  blockId: string,
  sections: readonly PromptSurfaceAdditionalSection[] | null | undefined,
): string | null {
  const rendered = (sections ?? [])
    .filter((section) => section.blockId === blockId)
    .map((section) => section.text);

  return rendered.length === 0 ? null : rendered.join("\n\n");
}

export function promptSurfaceCountsBySurface(): Record<PromptSurface, number> {
  const counts = Object.fromEntries(
    Object.values(PROMPT_SURFACES).map((surface) => [surface, 0]),
  ) as Record<PromptSurface, number>;

  for (const entry of PROMPT_SURFACE_BLOCKS) {
    for (const placement of entry.surfaces) {
      counts[placement.surface] += 1;
    }
  }

  return counts;
}
