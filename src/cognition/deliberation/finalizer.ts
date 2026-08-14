// Routes S1/S2 final response generation through the deliberator tool loop.
import { z } from "zod";

import type {
  LLMClient,
  LLMContentBlockMessage,
  LLMConverseOptions,
  LLMSystemBlock,
} from "../../llm/index.js";
import { willSendThinkingUnderAutoToolChoice } from "../../llm/index.js";
import type { ToolDefinition, ToolDispatcher } from "../../tools/dispatcher.js";
import type { BorgRole } from "../../memory/commitments/index.js";
import type { SessionAudienceRole } from "../../sessions/index.js";
import type { EntityId, SessionId } from "../../util/ids.js";
import type { TurnOrigin } from "../types.js";
import {
  emitTurnTokenFlushTrace,
  emitTurnTokenTrace,
  type TurnTracer,
} from "../../tracing/tracer.js";
import { executeToolLoop, type ToolLoopResult } from "../turn-action/index.js";
import {
  FINALIZER_NO_OUTPUT_PRIMARY_REASONS,
  FINALIZER_NO_OUTPUT_SEMANTIC_CATEGORIES,
  deriveFinalizerNoOutputPrimaryReason,
  messageDiscourseControlSchema,
  replyTargetSchema,
  type FinalizerNoOutputPrimaryReason,
  type FinalizerNoOutputSemanticCategory,
  type FinalizerNoOutputStructuralFlag,
  type MessageDiscourseControl,
  type ReplyTarget,
} from "../generation/types.js";
import {
  PROMPT_SURFACES,
  renderPromptSurfaceAdditionalBlock,
  renderPromptSurface,
  type PromptSurfaceAdditionalSection,
  type PromptSurfaceRenderContext,
} from "../prompts/prompt-surface-registry.js";
import { RELATIONSHIP_LABELS_PROMPT } from "../prompts/relationship-labels.js";
import { resolveFinalizerNonTerminalTools } from "./autonomous-finalizer-tools.js";
import { OUTBOUND_POST_TOOL_NAME } from "../../tools/internal/outbound-post-name.js";
import {
  buildCompactFinalizerSystemPrompt,
  type FinalizerContextTraceSummary,
  type FinalizerSurfaceVariant,
} from "./prompt/finalizer-context.js";
import type { BuildBaseSystemPromptOptions } from "./prompt/system-prompt.js";
import type { DeliberationContext } from "./types.js";
import { toTraceJsonValue } from "../../tracing/tracer.js";
import type {
  FinalizerCaptureOutcome,
  FinalizerContextCapture,
} from "./finalizer-context-capture.js";
import {
  fingerprintCanonicalRequest,
  type CanonicalRequestFingerprint,
} from "./request-fingerprint.js";
import { estimatePromptTokens } from "../../util/token-estimate.js";

export const EMIT_ANSWER_FINALIZER_TOOL_NAME = "EmitAnswer";
export const EMIT_OBSERVE_FINALIZER_TOOL_NAME = "EmitObserve";
export const EMIT_NO_OUTPUT_FINALIZER_TOOL_NAME = "EmitNoOutput";
export const EMIT_SELF_REPORT_FINALIZER_TOOL_NAME = "EmitSelfReport";
export const EMIT_CONTINUE_THOUGHT_FINALIZER_TOOL_NAME = "EmitContinueThought";

const emitTextToolInputSchema = z
  .object({
    text: z.string(),
    reply_target: replyTargetSchema.optional(),
    discourse_control: messageDiscourseControlSchema.optional(),
  })
  .strict();

const finalizerNoOutputSemanticCategorySchema = z.enum(FINALIZER_NO_OUTPUT_SEMANTIC_CATEGORIES);
const finalizerNoOutputPrimaryReasonSchema = z.enum(FINALIZER_NO_OUTPUT_PRIMARY_REASONS);

const emitNoOutputToolInputSchema = z
  .object({
    reason: z.string().min(1),
    primary_no_output_reason: finalizerNoOutputPrimaryReasonSchema.optional(),
    no_output_categories: z.array(finalizerNoOutputSemanticCategorySchema).optional(),
  })
  .strict();

const emitObserveToolInputSchema = z
  .object({
    reason: z.string().min(1),
  })
  .strict();

const emitSelfReportToolInputSchema = z
  .object({
    kind: z.literal("self_report"),
    text: z.string(),
    persistence_class: z.literal("assistant_self_report"),
    discourse_control: messageDiscourseControlSchema.optional(),
  })
  .strict();

const emitContinueThoughtToolInputSchema = z
  .object({
    text: z.string(),
  })
  .strict();

const EMIT_ANSWER_FINALIZER_TOOL: ToolDefinition = {
  name: EMIT_ANSWER_FINALIZER_TOOL_NAME,
  description:
    "I emit my visible response for this turn. I put the complete user-visible response in text. I use this for ordinary answers, questions, acknowledgments, challenges, and continuations. When the response is primarily addressed to one named participant, I also set reply_target to kind=entity with their prompt-visible entity_id; I default to kind=audience (or omit) when speaking to the whole channel or multiple participants.",
  menuSummary: "Speak visibly for the current turn.",
  allowedOrigins: ["deliberator"],
  writeScope: "read",
  inputSchema: emitTextToolInputSchema,
  outputSchema: z.object({}).strict(),
  async invoke() {
    return {};
  },
};

const EMIT_NO_OUTPUT_FINALIZER_TOOL: ToolDefinition = {
  name: EMIT_NO_OUTPUT_FINALIZER_TOOL_NAME,
  description:
    "I emit no visible message for this turn because the conversation has reached a natural close, the user has ended the exchange, or continuing would only produce ritual closure tokens. Different from EmitObserve, which is only for multi-participant conversations where I remain present while other participants continue. When using this tool, I choose one primary_no_output_reason and classify the silence in no_output_categories using the provided enum; I use [] when uncertain.",
  menuSummary: "End the turn with no visible message.",
  allowedOrigins: ["deliberator"],
  writeScope: "read",
  inputSchema: emitNoOutputToolInputSchema,
  outputSchema: z.object({}).strict(),
  async invoke() {
    return {};
  },
};

const EMIT_OBSERVE_FINALIZER_TOOL: ToolDefinition = {
  name: EMIT_OBSERVE_FINALIZER_TOOL_NAME,
  description:
    "I choose to observe the current conversation without producing a visible message. I use this only when <borg_audience_profile> shows a Participants list with multiple entries and they are talking to each other rather than to me: my input is not needed, the conversation flows naturally without me, or adding to it would interrupt rather than help. Different from EmitNoOutput, which signals conversation closure.",
  menuSummary: "Stay present without a visible message in a multi-participant exchange.",
  allowedOrigins: ["deliberator"],
  writeScope: "read",
  inputSchema: emitObserveToolInputSchema,
  outputSchema: z.object({}).strict(),
  async invoke() {
    return {};
  },
};

const EMIT_SELF_REPORT_FINALIZER_TOOL: ToolDefinition = {
  name: EMIT_SELF_REPORT_FINALIZER_TOOL_NAME,
  description:
    "I emit a first-person interior self-report from my perspective. I set kind to self_report, persistence_class to assistant_self_report, and put the complete user-visible response in text. The text is shown to the user like EmitAnswer.",
  menuSummary: "Speak visibly as a first-person interior self-report.",
  allowedOrigins: ["deliberator"],
  writeScope: "read",
  inputSchema: emitSelfReportToolInputSchema,
  outputSchema: z.object({}).strict(),
  async invoke() {
    return {};
  },
};

const EMIT_CONTINUE_THOUGHT_FINALIZER_TOOL: ToolDefinition = {
  name: EMIT_CONTINUE_THOUGHT_FINALIZER_TOOL_NAME,
  description:
    "I append a private carryover thought to the self-private journal so it is available to a later autonomous wake. I put the complete self-private thought in text. This is not user-facing, has no audience, makes no disclosure decision, and does not post a message.",
  menuSummary:
    "Append the carryover thought to the private journal and end the autonomous interval.",
  allowedOrigins: ["deliberator"],
  writeScope: "read",
  inputSchema: emitContinueThoughtToolInputSchema,
  outputSchema: z.object({}).strict(),
  async invoke() {
    return {};
  },
};

const EMISSION_FINALIZER_TOOLS = [
  EMIT_ANSWER_FINALIZER_TOOL,
  EMIT_OBSERVE_FINALIZER_TOOL,
  EMIT_NO_OUTPUT_FINALIZER_TOOL,
  EMIT_SELF_REPORT_FINALIZER_TOOL,
  EMIT_CONTINUE_THOUGHT_FINALIZER_TOOL,
] as const;

const EMISSION_FINALIZER_TOOL_NAMES = [
  EMIT_ANSWER_FINALIZER_TOOL_NAME,
  EMIT_OBSERVE_FINALIZER_TOOL_NAME,
  EMIT_NO_OUTPUT_FINALIZER_TOOL_NAME,
  EMIT_SELF_REPORT_FINALIZER_TOOL_NAME,
  EMIT_CONTINUE_THOUGHT_FINALIZER_TOOL_NAME,
] as const;

export type EmissionToolName = (typeof EMISSION_FINALIZER_TOOL_NAMES)[number];

export function resolveAvailableEmissionTools(
  allowedEmissions: readonly EmissionToolName[] | undefined,
  turnOrigin?: TurnOrigin,
): ToolDefinition[] {
  const allowedEmissionNames = new Set<string>(
    resolveAvailableEmissionNames(allowedEmissions, turnOrigin),
  );

  return EMISSION_FINALIZER_TOOLS.filter((tool) => allowedEmissionNames.has(tool.name));
}

const EMIT_ANSWER_FINALIZER_INSTRUCTION =
  "I use EmitAnswer for an ordinary visible response when I should speak. I put the complete user-visible response in text. I use reply_target.kind=entity with a prompt-visible entity_id when the response is primarily addressed to a single named participant -- including when answering a question from a specific speaker, when addressing one person by name, or when a participant has asked to be addressed directly. I use reply_target.kind=audience (or omit) when the response speaks to the channel as a whole.";
const EMIT_OBSERVE_FINALIZER_INSTRUCTION =
  "I use EmitObserve only in multi-participant contexts where <borg_audience_profile> shows a Participants list with multiple entries and the current exchange is participant-to-participant rather than directed to me. I put a concise durable reason in reason. This is an active observation, not a closure signal.";
const FINALIZER_SELF_REFERENTIAL_MEMORY_VOICE_GUIDANCE = [
  "Self-referential memory voice: when a prompt-visible structure identifies content as self-owned, I write self-referential content -- what I did, decided, believe, wonder, or commit to -- in the first person, in the same language as the memory, using that language's natural grammar.",
  "In pro-drop languages, first person may be carried by verb conjugation; I do not force a pronoun token or use any language-specific pronoun list.",
  "I refer to every other participant by their canonical name or stable handle; I do not use deictic participant placeholders for other people.",
  "For joint actions involving me and others, I name the others explicitly rather than collapsing the memory into a first-person-plural group reference; this keeps the memory stable when recalled for a different audience.",
  "I keep statements about the world, or about what other agents did or said, in their natural third-person form.",
].join(" ");
const EMIT_NO_OUTPUT_FINALIZER_INSTRUCTION = `I use EmitNoOutput to produce no visible message for this turn. I put a concise reason in reason. ${FINALIZER_SELF_REFERENTIAL_MEMORY_VOICE_GUIDANCE} I apply this to reason, which is persisted as decision_rationale.`;
const DEFAULT_EMIT_NO_OUTPUT_FINALIZER_INSTRUCTION = `I use EmitNoOutput only when the conversation has reached a natural close, the user has ended the exchange, or continuing would only produce ritual closure tokens. I put a concise reason in reason. ${FINALIZER_SELF_REFERENTIAL_MEMORY_VOICE_GUIDANCE} I apply this to reason, which is persisted as decision_rationale.`;
const EMIT_SELF_REPORT_FINALIZER_INSTRUCTION =
  "I use EmitSelfReport for first-person expression of my interior state, identity reflection, voice, or boundary. EmitSelfReport must include kind=self_report, persistence_class=assistant_self_report, and text. It is shown to the user exactly like EmitAnswer and persisted as assistant_self_report.";
const EMIT_CONTINUE_THOUGHT_FINALIZER_INSTRUCTION =
  "I use EmitContinueThought to carry a private thought into the next autonomous reflection wake by appending it to the self-private journal. It is not user-facing, has no audience, makes no disclosure decision, and does not post a visible message.";
const EMIT_DISCOURSE_CONTROL_INSTRUCTION =
  "For EmitAnswer or EmitSelfReport, I set discourse_control.kind=stop_until_substantive_content ONLY when the visible response commits me to emit nothing until substantive new user content appears; I do not set it for ordinary topic boundaries, local explanations, or style commitments.";
const EMIT_NO_OUTPUT_CLASSIFICATION_INSTRUCTIONS = [
  'When emitting EmitNoOutput, I choose ONE primary_no_output_reason that best captures why silence is the right output: "closure" when the message is a closure-shaped wrap-up, goodbye, sign-off, or terminal beat; "user_to_user" when the current message is between two human participants and I was not addressed; "when_borg_addressed" when I was explicitly addressed but no useful response is warranted (rare); "low_value_echo" when any visible response would only acknowledge or echo with no new content; "other" for any other principled reason for silence.',
  'When emitting EmitNoOutput, I classify the silence with no_output_categories: "user_to_user" if the current message is between two human participants and I was not addressed; "when_borg_addressed" if I was explicitly addressed but no useful response is warranted; "closure" if the message is a closure-shaped acknowledgment, sign-off, or terminal beat. If multiple apply, I list all. I use [] if uncertain.',
] as const;

const COMMON_FINALIZER_INSTRUCTIONS = [
  "I do not hide factual or source-sensitive content. If a name, place, number, date, callback, action state, relational/profile detail, or claim about my own prior behavior cannot be grounded in prompt-visible evidence, I remove it or phrase it qualitatively.",
  RELATIONSHIP_LABELS_PROMPT,
  "I use the Attribution Matrix and Attribution Sidebar as authoritative for who said, committed, decided, or reasoned what. Assistant rationale entries are my prior reasoning, not participant claims.",
  "When a named entity is supported by evidence that uses only a pronoun or descriptive noun phrase for the predicate, I do not present the name and predicate together unless the prompt-visible evidence also establishes that the name belongs to that entity.",
] as const;

export function resolveAvailableEmissionNames(
  allowedEmissions: readonly EmissionToolName[] | undefined,
  turnOrigin?: TurnOrigin,
): EmissionToolName[] {
  const allowed =
    allowedEmissions === undefined ? null : new Set<EmissionToolName>(allowedEmissions);

  return EMISSION_FINALIZER_TOOL_NAMES.filter(
    (name) =>
      (allowed === null || allowed.has(name)) &&
      (name !== EMIT_CONTINUE_THOUGHT_FINALIZER_TOOL_NAME || turnOrigin === "autonomous"),
  );
}

function formatEmissionToolList(names: readonly EmissionToolName[]): string {
  if (names.length === 0) {
    return "no terminal tools";
  }

  if (names.length === 1) {
    return names[0]!;
  }

  if (names.length === 2) {
    return `${names[0]!} and ${names[1]!}`;
  }

  return `${names.slice(0, -1).join(", ")}, and ${names[names.length - 1]!}`;
}

// The terminal-emission contract, stated as the first thing in the protocol block. The
// model's output channel for a turn IS a single terminal tool call; loose prose outside one
// is discarded by the harness, never delivered. This is the structural fact -- not a judgment
// of content -- and it was previously only implied by "available tools" framing plus one
// aside, which let a thinking-mode turn (tool_choice:auto) occasionally answer in prose and
// emit nothing. Naming the mechanism up front is the fix.
function emissionContractPreamble(availableEmissionNames: readonly EmissionToolName[]): string {
  const onePhrase =
    availableEmissionNames.length === 1
      ? `my only terminal emission tool, ${availableEmissionNames[0]}`
      : `exactly one of ${formatEmissionToolList(availableEmissionNames)}`;
  return [
    `Every turn I take ends with one terminal emission tool call: I call ${onePhrase}, never zero and never more than one.`,
    "That single tool call is my entire output for the turn. Any text I write outside a terminal emission tool call -- even a complete, well-formed reply in plain prose -- is internal scratch that the harness never delivers to anyone. Whatever I want to say or decide, I put it inside the terminal tool; writing prose instead of calling one emits nothing at all.",
  ].join(" ");
}

function buildEmissionToolInstructions(
  availableEmissionNames: readonly EmissionToolName[],
): string {
  const available = new Set(availableEmissionNames);
  const contractPreamble = emissionContractPreamble(availableEmissionNames);
  const noOutputInstructions = [
    EMIT_NO_OUTPUT_FINALIZER_INSTRUCTION,
    ...EMIT_NO_OUTPUT_CLASSIFICATION_INSTRUCTIONS,
  ];

  if (availableEmissionNames.length === 1 && available.has(EMIT_NO_OUTPUT_FINALIZER_TOOL_NAME)) {
    return [contractPreamble, "", ...noOutputInstructions].join("\n");
  }

  if (
    availableEmissionNames.length === 2 &&
    available.has(EMIT_OBSERVE_FINALIZER_TOOL_NAME) &&
    available.has(EMIT_NO_OUTPUT_FINALIZER_TOOL_NAME)
  ) {
    return [contractPreamble, "", EMIT_OBSERVE_FINALIZER_INSTRUCTION, ...noOutputInstructions].join(
      "\n",
    );
  }

  if (availableEmissionNames.length === EMISSION_FINALIZER_TOOL_NAMES.length) {
    return [
      contractPreamble,
      "",
      EMIT_ANSWER_FINALIZER_INSTRUCTION,
      EMIT_DISCOURSE_CONTROL_INSTRUCTION,
      `${EMIT_OBSERVE_FINALIZER_INSTRUCTION} In ordinary one-to-one turns, I prefer EmitAnswer when I should speak or EmitNoOutput when the conversation has closed.`,
      DEFAULT_EMIT_NO_OUTPUT_FINALIZER_INSTRUCTION,
      ...EMIT_NO_OUTPUT_CLASSIFICATION_INSTRUCTIONS,
      EMIT_SELF_REPORT_FINALIZER_INSTRUCTION,
      EMIT_CONTINUE_THOUGHT_FINALIZER_INSTRUCTION,
    ].join("\n");
  }

  if (availableEmissionNames.length === 0) {
    return "No terminal emission tools are available.";
  }

  return [
    contractPreamble,
    "",
    ...(available.has(EMIT_ANSWER_FINALIZER_TOOL_NAME) ? [EMIT_ANSWER_FINALIZER_INSTRUCTION] : []),
    ...(available.has(EMIT_ANSWER_FINALIZER_TOOL_NAME) ||
    available.has(EMIT_SELF_REPORT_FINALIZER_TOOL_NAME)
      ? [EMIT_DISCOURSE_CONTROL_INSTRUCTION]
      : []),
    ...(available.has(EMIT_OBSERVE_FINALIZER_TOOL_NAME)
      ? [EMIT_OBSERVE_FINALIZER_INSTRUCTION]
      : []),
    ...(available.has(EMIT_NO_OUTPUT_FINALIZER_TOOL_NAME) ? noOutputInstructions : []),
    ...(available.has(EMIT_SELF_REPORT_FINALIZER_TOOL_NAME)
      ? [EMIT_SELF_REPORT_FINALIZER_INSTRUCTION]
      : []),
    ...(available.has(EMIT_CONTINUE_THOUGHT_FINALIZER_TOOL_NAME)
      ? [EMIT_CONTINUE_THOUGHT_FINALIZER_INSTRUCTION]
      : []),
  ].join("\n");
}

function buildEmissionFinalizerInstructions(
  allowedEmissions: readonly EmissionToolName[] | undefined,
  outboundToolAvailable: boolean,
  turnOrigin?: TurnOrigin,
): string {
  return [
    buildEmissionToolInstructions(resolveAvailableEmissionNames(allowedEmissions, turnOrigin)),
    ...(outboundToolAvailable
      ? [
          "",
          "Non-terminal outbound tool: when a structurally authorized creator in an operator session asks me to send a message into another session, or when an autonomous turn has an authorized target listed in <reachable_threads>, I call tool.outbound.post first with the target_session_id and an instruction for the target-scoped composition turn. I wait for the tool result, then call exactly one terminal emission tool for the current turn. I do not expose tool names, session ids, or dispatch internals in visible text.",
        ]
      : []),
    "",
    ...COMMON_FINALIZER_INSTRUCTIONS,
  ].join("\n");
}

// Sprint 9.9: the static prefix is currently ~1,845 estimated tokens, below
// Opus's 4,096-token cache minimum. The marker is a no-op today but is kept
// in place so the structural ordering is cache-aware -- whenever future
// sprints add legitimate static self-knowledge content (evidence-ledger
// catalog, memory-band semantics, audience invariants, richer tool
// descriptions) and the prefix crosses the threshold, caching activates
// without a code change. Precedent 8d.6.11 removed a dead marker on
// TURN_PLAN_TOOL because there was no plausible content path; here there is.
const FINALIZER_STATIC_PREFIX_CACHE_CONTROL = { type: "ephemeral", ttl: "1h" } as const;

// The large dynamic prefix repeats within a turn, usually inside five minutes.
// A 1h marker charges a higher write premium and loses on turns without a repeat.
const FINALIZER_DYNAMIC_PREFIX_CACHE_CONTROL = { type: "ephemeral", ttl: "5m" } as const;

export const FINALIZER_REGENERATION_PROMPT_BLOCK_IDS = [
  "borg_commitment_regeneration_instruction",
  "finalizer_invalid_tool_retry_instruction",
] as const;

export type CacheableFinalizerSystemPrompt = {
  staticPrefix: string;
  dynamicContent: string;
};

export type RunFinalizerOptions = {
  llmClient: LLMClient;
  dispatcher: ToolDispatcher;
  sessionId: SessionId;
  audienceEntityId?: EntityId | null;
  model: string;
  baseSystemPrompt: string;
  initialMessages: readonly LLMContentBlockMessage[];
  userEntryId: string | undefined;
  maxTokens: number;
  thinking?: LLMConverseOptions["thinking"];
  effort?: LLMConverseOptions["effort"];
  path: "system_1" | "system_2";
  additionalPromptSections?: readonly PromptSurfaceAdditionalSection[];
  cacheableSystemPrompt?: CacheableFinalizerSystemPrompt;
  finalizerDynamicPromptCacheEnabled?: boolean;
  finalizerSurfaceVariant?: FinalizerSurfaceVariant;
  compactSurface?: {
    context: DeliberationContext;
    baseSystemPromptOptions: BuildBaseSystemPromptOptions;
  };
  allowedEmissions?: readonly EmissionToolName[];
  outboundToolAvailable?: boolean;
  nonTerminalTools?: readonly ToolDefinition[];
  turnOrigin?: TurnOrigin;
  currentSenderBorgRole?: BorgRole | null;
  sessionAudienceRole?: SessionAudienceRole;
  structuralNoOutputFlags?: readonly FinalizerNoOutputStructuralFlag[];
  tracer?: TurnTracer;
  turnId?: string;
  finalizerAttempt?: "initial" | "regenerate";
  onRequestPrepared?: (request: LLMConverseOptions, attempt: number) => void;
  finalizerContextCapture?: FinalizerContextCapture;
};

export type EmissionDecision =
  | {
      kind: "answer";
      text: string;
      source: "tool" | "text";
      reply_target?: ReplyTarget;
      discourse_control?: MessageDiscourseControl;
    }
  | {
      kind: "self_report";
      text: string;
      persistence_class: "assistant_self_report";
      discourse_control?: MessageDiscourseControl;
    }
  | {
      kind: "continue_thought";
      text: string;
    }
  | {
      kind: "no_output";
      reason: string;
      primary_no_output_reason?: FinalizerNoOutputPrimaryReason;
      no_output_categories: FinalizerNoOutputSemanticCategory[];
    }
  | {
      kind: "observe";
      reason: string;
    }
  | {
      kind: "empty";
    }
  | {
      kind: "invalid_tool";
      toolName: string;
      reason: string;
    };

export type FinalizerResult = ToolLoopResult & {
  decision: EmissionDecision;
};

function buildDynamicSystemPrompt(options: RunFinalizerOptions): string {
  const renderContext = createFinalizerPromptSurfaceRenderContext(options);

  return renderPromptSurface(PROMPT_SURFACES.finalizerDynamicSystem, renderContext) ?? "";
}

function isFinalizerRegenerationPromptBlock(blockId: string): boolean {
  return FINALIZER_REGENERATION_PROMPT_BLOCK_IDS.some(
    (regenerationBlockId) => regenerationBlockId === blockId,
  );
}

function buildDynamicSystemPromptParts(options: RunFinalizerOptions): {
  stable: string;
  regeneration: string | null;
} {
  const full = buildDynamicSystemPrompt(options);
  const stableOptions =
    options.additionalPromptSections === undefined
      ? options
      : {
          ...options,
          additionalPromptSections: options.additionalPromptSections.filter(
            (section) => !isFinalizerRegenerationPromptBlock(section.blockId),
          ),
        };
  const stable = buildDynamicSystemPrompt(stableOptions);

  return {
    stable,
    // Content blocks are forwarded verbatim, so retain the surface renderer's exact
    // boundary bytes on the extracted suffix.
    regeneration: full === stable ? null : full.slice(stable.length),
  };
}

function createFinalizerPromptSurfaceRenderContext(
  options: RunFinalizerOptions,
): PromptSurfaceRenderContext {
  const outboundAvailableForPrompt =
    options.nonTerminalTools?.some((tool) => tool.name === OUTBOUND_POST_TOOL_NAME) ?? false;

  return {
    renderBlock: (id) => {
      switch (id) {
        case "finalizer_emission_protocol":
          return buildEmissionFinalizerInstructions(
            options.allowedEmissions,
            outboundAvailableForPrompt,
            options.turnOrigin,
          );
        case "finalizer_cacheable_static_prefix":
          return options.cacheableSystemPrompt?.staticPrefix ?? null;
        case "finalizer_base_dynamic_prompt":
          return options.cacheableSystemPrompt?.dynamicContent ?? options.baseSystemPrompt;
        case "borg_session_reentry_continuity":
        case "borg_evidence_ledger":
        case "borg_additional_retrieval":
        case "borg_s2_plan":
        case "borg_commitment_regeneration_instruction":
        // finalizer_invalid_tool_retry_instruction is intercepted by buildFinalizerCallOptions
        // and routed to a trailing message, so the deliberator (the only production caller)
        // never passes it here and it renders null in production. This case still renders the
        // section if a direct runFinalizer caller (e.g. a test) supplies it.
        case "finalizer_invalid_tool_retry_instruction":
          return renderPromptSurfaceAdditionalBlock(id, options.additionalPromptSections);
        default:
          return null;
      }
    },
  };
}

function buildStaticSystemPrompt(options: RunFinalizerOptions): string {
  return (
    renderPromptSurface(
      PROMPT_SURFACES.finalizerStaticSystem,
      createFinalizerPromptSurfaceRenderContext(options),
    ) ?? ""
  );
}

function legacyFinalizerTraceSummary(
  path: RunFinalizerOptions["path"],
  system: readonly LLMSystemBlock[],
): FinalizerContextTraceSummary {
  const staticText = system[0]?.text ?? "";
  const turnText = system
    .slice(1)
    .map((block) => block.text)
    .join("\n\n");
  const totalText = system.map((block) => block.text).join("\n\n");
  const sections = Object.fromEntries(
    system.map((block, index) => {
      const label =
        index === 0 ? "legacy_static" : index === 1 ? "legacy_dynamic" : "legacy_suffix";
      return [
        label,
        {
          chars: block.text.length,
          estimatedTokens: estimatePromptTokens(block.text),
          rowCount: 0,
          truncationCount: 0,
          omissionCount: 0,
          cacheTier: index === 0 ? "terminal_static_head" : "terminal_turn_context",
        },
      ];
    }),
  ) as FinalizerContextTraceSummary["sections"];
  return {
    variant: "legacy",
    path,
    sections,
    blocks: {
      terminal_static_head: {
        chars: staticText.length,
        estimatedTokens: estimatePromptTokens(staticText),
        ttl: "1h",
      },
      terminal_durable_global: { chars: 0, estimatedTokens: 0, ttl: "1h" },
      terminal_durable_audience: { chars: 0, estimatedTokens: 0, ttl: "1h" },
      terminal_turn_context: {
        chars: turnText.length,
        estimatedTokens: estimatePromptTokens(turnText),
        ttl: "5m",
      },
    },
    totalChars: totalText.length,
    totalEstimatedTokens: estimatePromptTokens(totalText),
    rowCount: 0,
    truncationCount: 0,
    omissionCount: 0,
  };
}

export function buildFinalizerSystemPrompt(options: RunFinalizerOptions): {
  system: readonly LLMSystemBlock[];
  traceSummary: FinalizerContextTraceSummary | null;
} {
  if (options.finalizerSurfaceVariant === "compact") {
    if (options.compactSurface === undefined) {
      throw new TypeError("Compact finalizer surface requires compactSurface context");
    }
    const dynamicPrompt = buildDynamicSystemPromptParts(options);
    const stableAdditionalSections = options.additionalPromptSections?.filter(
      (section) => !isFinalizerRegenerationPromptBlock(section.blockId),
    );
    const compact = buildCompactFinalizerSystemPrompt({
      context: options.compactSurface.context,
      baseSystemPromptOptions: options.compactSurface.baseSystemPromptOptions,
      staticHead: buildStaticSystemPrompt(options),
      path: options.path,
      additionalPromptSections: stableAdditionalSections,
    });
    return {
      system: [
        ...compact.system,
        ...(dynamicPrompt.regeneration === null
          ? []
          : [{ type: "text" as const, text: dynamicPrompt.regeneration }]),
      ],
      traceSummary: compact.traceSummary,
    };
  }

  const staticPromptBlock = {
    type: "text" as const,
    text: buildStaticSystemPrompt(options),
    ...(options.cacheableSystemPrompt === undefined
      ? {}
      : { cache_control: FINALIZER_STATIC_PREFIX_CACHE_CONTROL }),
  };

  if (options.finalizerDynamicPromptCacheEnabled === false) {
    const system: LLMSystemBlock[] = [
      staticPromptBlock,
      {
        type: "text" as const,
        text: buildDynamicSystemPrompt(options),
      },
    ];
    return {
      system,
      traceSummary: legacyFinalizerTraceSummary(options.path, system),
    };
  }

  const dynamicPrompt = buildDynamicSystemPromptParts(options);

  const system: LLMSystemBlock[] = [
    staticPromptBlock,
    {
      type: "text",
      text: dynamicPrompt.stable,
      // Regeneration is a one-shot whose reasoning parameters differ from the call
      // that populated the prefix, so another five-minute write cannot amortize.
      ...(dynamicPrompt.regeneration === null
        ? { cache_control: FINALIZER_DYNAMIC_PREFIX_CACHE_CONTROL }
        : {}),
    },
    // Keep one-shot instructions outside the reusable dynamic prefix.
    ...(dynamicPrompt.regeneration === null
      ? []
      : [
          {
            type: "text" as const,
            text: dynamicPrompt.regeneration,
          },
        ]),
  ];
  return {
    system,
    traceSummary: legacyFinalizerTraceSummary(options.path, system),
  };
}

function invalidToolDecision(toolName: string, reason: string): EmissionDecision {
  return {
    kind: "invalid_tool",
    toolName,
    reason,
  };
}

function decisionFromEmissionToolResult(result: ToolLoopResult): EmissionDecision {
  // Emission-tool mode is a strict protocol: exactly one terminal tool call
  // carries the behavior choice. Free text without a tool call is a protocol
  // violation and maps to finalizer_failed via invalid_tool, not to
  // empty_finalizer. empty_finalizer is reserved for an explicit EmitAnswer("")
  // or EmitSelfReport("") call.
  if (result.terminalToolCalls.length !== 1) {
    return invalidToolDecision(
      result.terminalToolCalls.length === 0 ? "none" : "multiple",
      `expected exactly one emission tool call, got ${result.terminalToolCalls.length}`,
    );
  }

  const terminalCall = result.terminalToolCalls[0]!;

  if (terminalCall.name === EMIT_ANSWER_FINALIZER_TOOL_NAME) {
    const parsed = emitTextToolInputSchema.safeParse(terminalCall.input);

    if (!parsed.success) {
      return invalidToolDecision(terminalCall.name, parsed.error.message);
    }

    return parsed.data.text.trim().length === 0
      ? { kind: "empty" }
      : {
          kind: "answer",
          text: parsed.data.text,
          source: "tool",
          ...(parsed.data.reply_target === undefined
            ? {}
            : { reply_target: parsed.data.reply_target }),
          ...(parsed.data.discourse_control === undefined
            ? {}
            : { discourse_control: parsed.data.discourse_control }),
        };
  }

  if (terminalCall.name === EMIT_SELF_REPORT_FINALIZER_TOOL_NAME) {
    const parsed = emitSelfReportToolInputSchema.safeParse(terminalCall.input);

    if (!parsed.success) {
      return invalidToolDecision(terminalCall.name, parsed.error.message);
    }

    return parsed.data.text.trim().length === 0
      ? { kind: "empty" }
      : {
          kind: "self_report",
          text: parsed.data.text,
          persistence_class: "assistant_self_report",
          ...(parsed.data.discourse_control === undefined
            ? {}
            : { discourse_control: parsed.data.discourse_control }),
        };
  }

  if (terminalCall.name === EMIT_CONTINUE_THOUGHT_FINALIZER_TOOL_NAME) {
    const parsed = emitContinueThoughtToolInputSchema.safeParse(terminalCall.input);

    return parsed.success
      ? {
          kind: "continue_thought",
          text: parsed.data.text,
        }
      : invalidToolDecision(terminalCall.name, parsed.error.message);
  }

  if (terminalCall.name === EMIT_NO_OUTPUT_FINALIZER_TOOL_NAME) {
    const parsed = emitNoOutputToolInputSchema.safeParse(terminalCall.input);

    return parsed.success
      ? {
          kind: "no_output",
          reason: parsed.data.reason,
          ...(parsed.data.primary_no_output_reason === undefined
            ? {}
            : { primary_no_output_reason: parsed.data.primary_no_output_reason }),
          no_output_categories: parsed.data.no_output_categories ?? [],
        }
      : invalidToolDecision(terminalCall.name, parsed.error.message);
  }

  if (terminalCall.name === EMIT_OBSERVE_FINALIZER_TOOL_NAME) {
    const parsed = emitObserveToolInputSchema.safeParse(terminalCall.input);

    return parsed.success
      ? { kind: "observe", reason: parsed.data.reason }
      : invalidToolDecision(terminalCall.name, parsed.error.message);
  }

  return invalidToolDecision(terminalCall.name, "unknown terminal emission tool");
}

function structuralNoOutputFlagsForTrace(
  options: RunFinalizerOptions,
  decision: EmissionDecision,
): FinalizerNoOutputStructuralFlag[] | undefined {
  if (decision.kind !== "no_output") {
    return undefined;
  }

  return [
    ...new Set<FinalizerNoOutputStructuralFlag>([
      ...(options.structuralNoOutputFlags ?? []),
      ...(decision.no_output_categories.includes("when_borg_addressed")
        ? (["borg_directly_addressed"] as const)
        : []),
    ]),
  ];
}

function emitFinalizerTrace(options: RunFinalizerOptions, decision: EmissionDecision): void {
  if (options.tracer?.enabled !== true || options.turnId === undefined) {
    return;
  }

  const structuralNoOutputFlags = structuralNoOutputFlagsForTrace(options, decision);
  const primaryNoOutputReason =
    decision.kind === "no_output"
      ? (decision.primary_no_output_reason ??
        deriveFinalizerNoOutputPrimaryReason(decision.no_output_categories))
      : undefined;

  options.tracer.emit("finalizer.completed", {
    turnId: options.turnId,
    session_id: options.sessionId,
    path: options.path,
    mode: "emission_tools",
    decision: decision.kind,
    attempt: options.finalizerAttempt ?? "initial",
    ...(decision.kind === "answer" ||
    decision.kind === "self_report" ||
    decision.kind === "continue_thought"
      ? { text_length: decision.text.length }
      : {}),
    ...(decision.kind === "answer" && decision.reply_target !== undefined
      ? { reply_target: decision.reply_target }
      : {}),
    ...(decision.kind === "answer" || decision.kind === "self_report"
      ? decision.discourse_control === undefined
        ? {}
        : { discourse_control: decision.discourse_control }
      : {}),
    ...(decision.kind === "no_output"
      ? {
          reason: decision.reason,
          primary_no_output_reason: primaryNoOutputReason,
          no_output_categories: [...decision.no_output_categories],
          ...(structuralNoOutputFlags === undefined
            ? {}
            : { structural_no_output_flags: structuralNoOutputFlags }),
        }
      : {}),
    ...(decision.kind === "observe" ? { reason: decision.reason } : {}),
    ...(decision.kind === "self_report" ? { persistence_class: decision.persistence_class } : {}),
    ...(decision.kind === "invalid_tool"
      ? { tool_name: decision.toolName, reason: decision.reason }
      : {}),
  });
}

function finalizerFlushText(result: ToolLoopResult, decision: EmissionDecision): string {
  if (decision.kind === "answer" || decision.kind === "self_report") {
    return decision.text;
  }

  // For no_output / observe / continue_thought / invalid, there is no user-facing
  // visible text. result.text may hold loose model prose emitted alongside the
  // tool under auto tool_choice -- never flush it as the turn's visible output.
  return "";
}

type FinalizerCaptureState = {
  selected: boolean;
  timestamp: number | undefined;
  systems: {
    legacy: NonNullable<LLMConverseOptions["system"]>;
    compact: NonNullable<LLMConverseOptions["system"]>;
  } | null;
  request: LLMConverseOptions | null;
  requestFingerprint: CanonicalRequestFingerprint | null;
  attempts: number;
};

type PreparedFinalizerPrompt = {
  rendered: ReturnType<typeof buildFinalizerSystemPrompt>;
  capture: FinalizerCaptureState;
};

function prepareFinalizerPrompt(
  options: RunFinalizerOptions,
  nonTerminalTools: readonly ToolDefinition[],
): PreparedFinalizerPrompt {
  const rendered = buildFinalizerSystemPrompt({
    ...options,
    nonTerminalTools,
  });
  const captureSelected = options.finalizerContextCapture?.shouldCapture() === true;
  const captureTimestamp = captureSelected
    ? options.finalizerContextCapture?.capturedAt()
    : undefined;
  let captureSystems: FinalizerCaptureState["systems"] = null;

  if (captureSelected && options.compactSurface !== undefined) {
    try {
      captureSystems = {
        legacy: buildFinalizerSystemPrompt({
          ...options,
          nonTerminalTools,
          finalizerSurfaceVariant: "legacy",
        }).system,
        compact: buildFinalizerSystemPrompt({
          ...options,
          nonTerminalTools,
          finalizerSurfaceVariant: "compact",
        }).system,
      };
    } catch (error) {
      // Evaluation capture is best-effort. The already-rendered live surface
      // remains authoritative; failure to assemble its alternative must not
      // interrupt a production finalizer call.
      captureSystems = null;
      options.finalizerContextCapture?.recordAssemblyFailure(
        { turnId: options.turnId, sessionId: options.sessionId },
        error,
      );
    }
  }

  return {
    rendered,
    capture: {
      selected: captureSelected,
      timestamp: captureTimestamp,
      systems: captureSystems,
      request: null,
      requestFingerprint: null,
      attempts: 0,
    },
  };
}

function recordPreparedFinalizerRequest(
  state: FinalizerCaptureState,
  request: LLMConverseOptions,
  attempt: number,
): void {
  state.attempts = Math.max(state.attempts, attempt);
  if (state.request !== null || !state.selected) {
    return;
  }

  try {
    state.request = JSON.parse(JSON.stringify(request)) as LLMConverseOptions;
    state.requestFingerprint = fingerprintCanonicalRequest(request);
  } catch {
    // Missing request fidelity makes the record ineligible for live replay;
    // observation must never block the live call.
    state.request = null;
    state.requestFingerprint = null;
  }
}

async function captureFinalizerOutcome(
  options: RunFinalizerOptions,
  state: FinalizerCaptureState,
  outcome: FinalizerCaptureOutcome,
  usedNonTerminalTools: boolean,
): Promise<void> {
  if (
    state.timestamp === undefined ||
    state.systems === null ||
    options.finalizerContextCapture === undefined ||
    options.compactSurface === undefined
  ) {
    return;
  }

  await options.finalizerContextCapture.capture({
    capturedAt: state.timestamp,
    turnId: options.turnId,
    sessionId: options.sessionId,
    path: options.path,
    attemptKind: options.finalizerAttempt ?? "initial",
    liveSurfaceVariant: options.finalizerSurfaceVariant ?? "legacy",
    context: options.compactSurface.context,
    legacySystem: state.systems.legacy,
    compactSystem: state.systems.compact,
    liveRequest: state.request,
    liveRequestFingerprint: state.requestFingerprint,
    outcome,
    usedNonTerminalTools,
  });
}

function emitFinalizerContextTrace(
  options: RunFinalizerOptions,
  summary: FinalizerContextTraceSummary | null,
): void {
  if (summary === null || options.tracer?.enabled !== true || options.turnId === undefined) {
    return;
  }

  options.tracer.emit("deliberation.finalizer_context.completed", {
    turnId: options.turnId,
    session_id: options.sessionId,
    path: options.path,
    attempt: options.finalizerAttempt ?? "initial",
    variant: summary.variant,
    sections: toTraceJsonValue(summary.sections),
    blocks: toTraceJsonValue(summary.blocks),
    total_chars: summary.totalChars,
    total_estimated_tokens: summary.totalEstimatedTokens,
    row_count: summary.rowCount,
    truncation_count: summary.truncationCount,
    omission_count: summary.omissionCount,
    message_count: options.initialMessages.length,
    message_chars: options.initialMessages.reduce(
      (sum, message) =>
        sum +
        message.content.reduce(
          (messageSum, block) => messageSum + (block.type === "text" ? block.text.length : 0),
          0,
        ),
      0,
    ),
  });
}

export async function runFinalizer(options: RunFinalizerOptions): Promise<FinalizerResult> {
  const toolProvenance =
    options.userEntryId === undefined ? undefined : { user_entry_id: options.userEntryId };
  const emissionTools = resolveAvailableEmissionTools(options.allowedEmissions, options.turnOrigin);
  const nonTerminalTools = resolveFinalizerNonTerminalTools({
    dispatcher: options.dispatcher,
    turnOrigin: options.turnOrigin,
    outboundToolAvailable: options.outboundToolAvailable,
  });
  const preparedPrompt = prepareFinalizerPrompt(options, nonTerminalTools);
  const renderedSystemPrompt = preparedPrompt.rendered;
  const systemPrompt = renderedSystemPrompt.system;
  const captureState = preparedPrompt.capture;
  emitFinalizerContextTrace(options, renderedSystemPrompt.traceSummary);
  const availableTools = [...emissionTools, ...nonTerminalTools];
  const terminalToolNames = emissionTools.map((tool) => tool.name);
  const effectiveThinking =
    options.finalizerAttempt === "regenerate" ? undefined : options.thinking;
  const effectiveEffort = options.finalizerAttempt === "regenerate" ? undefined : options.effort;
  // Auto tool_choice iff thinking will actually reach the model -- otherwise force
  // an emission tool so a structured emission stays guaranteed (e.g. manual
  // thinking on Opus is omitted by the client, so forcing is correct there).
  const useAutoToolChoice = willSendThinkingUnderAutoToolChoice(options.model, effectiveThinking);
  let tokenSequence = 0;

  let result: ToolLoopResult;
  try {
    result = await executeToolLoop({
      llmClient: options.llmClient,
      dispatcher: options.dispatcher,
      sessionId: options.sessionId,
      audienceEntityId: options.audienceEntityId,
      model: options.model,
      systemPrompt,
      initialMessages: options.initialMessages,
      tools: availableTools,
      origin: options.turnOrigin === "autonomous" ? "autonomous" : "deliberator",
      turnOrigin: options.turnOrigin,
      currentSenderBorgRole: options.currentSenderBorgRole,
      sessionAudienceRole: options.sessionAudienceRole,
      provenance: toolProvenance,
      maxTokens: options.maxTokens,
      ...(effectiveThinking === undefined ? {} : { thinking: effectiveThinking }),
      ...(effectiveEffort === undefined ? {} : { effort: effectiveEffort }),
      // Emission-tool protocol: the answer lives in the terminal tool input, so any
      // loose text the model emits under auto tool_choice must not reach the stream.
      suppressRawTextStream: true,
      // Thinking requires auto tool_choice -- the API rejects forced tool use with
      // thinking active. When thinking will be sent, omit toolChoice (auto): the
      // model thinks, then calls exactly one emission tool. The emission is read
      // from the terminal tool (loose text never leaks), and the invalid-tool retry
      // net covers the rare turn that emits no terminal tool. Otherwise force a tool
      // ("any") so an emission is guaranteed.
      ...(useAutoToolChoice || emissionTools.length === 0
        ? {}
        : { toolChoice: { type: "any" as const } }),
      budget: options.path === "system_1" ? "cognition-system-1" : "cognition-system-2",
      tracer: options.tracer,
      turnId: options.turnId,
      traceLabel: `${options.path}_finalizer`,
      terminalToolNames,
      stream: true,
      onTextDelta: (chunkText) => {
        tokenSequence += 1;
        emitTurnTokenTrace({
          tracer: options.tracer,
          turnId: options.turnId,
          sessionId: options.sessionId,
          phase: "final",
          chunkText,
          sequence: tokenSequence,
        });
      },
      onRequestPrepared: (request, attempt) => {
        recordPreparedFinalizerRequest(captureState, request, attempt);
        options.onRequestPrepared?.(request, attempt);
      },
    });
  } catch (error) {
    const code =
      error !== null && typeof error === "object" && "code" in error
        ? String((error as { code?: unknown }).code)
        : undefined;
    await captureFinalizerOutcome(
      options,
      captureState,
      {
        status: "threw",
        attempts: captureState.attempts,
        structuralReason: "finalizer_error",
        error: {
          name: error instanceof Error ? error.name : "UnknownThrownValue",
          message: error instanceof Error ? error.message : String(error),
          ...(code === undefined ? {} : { code }),
        },
      },
      false,
    );
    throw error;
  }
  const decision = decisionFromEmissionToolResult(result);
  await captureFinalizerOutcome(
    options,
    captureState,
    {
      status: "completed",
      attempts: captureState.attempts,
      structuralReason:
        result.toolCallsMade.length > 0
          ? "nonterminal_tool_loop"
          : result.terminalToolCalls.length === 1
            ? "terminal_emission"
            : "no_terminal_emission",
      decisionKind: decision.kind,
      decision,
      terminalToolCalls: result.terminalToolCalls,
      reasoningText: result.text,
      usage: result.usage,
    },
    result.toolCallsMade.length > 0,
  );

  if (tokenSequence > 0) {
    emitTurnTokenFlushTrace({
      tracer: options.tracer,
      turnId: options.turnId,
      sessionId: options.sessionId,
      phase: "final",
      fullText: finalizerFlushText(result, decision),
    });
  }
  emitFinalizerTrace(options, decision);

  return {
    ...result,
    decision,
  };
}
