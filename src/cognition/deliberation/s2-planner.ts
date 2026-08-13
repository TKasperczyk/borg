// Runs the System 2 structured planner call and parses its EmitTurnPlan output.
import { z } from "zod";

import {
  type LLMClient,
  type LLMCompleteOptions,
  type LLMMessage,
  type LLMSystemBlock,
  type LLMToolCall,
  type LLMToolDefinition,
  isRetryableLlmTransportError,
  toToolInputSchema,
  willSendThinkingUnderAutoToolChoice,
} from "../../llm/index.js";
import type { TurnTracer } from "../../tracing/tracer.js";
import { toTraceJsonValue } from "../../tracing/tracer.js";
import {
  summarizeToolResponseShape,
  traceLlmCallResponse,
  traceLlmCallRetryHook,
  traceLlmCallStarted,
} from "../../tracing/llm-call-trace.js";
import { intentRecordSchema, type TurnOrigin } from "../types.js";
import type { EmissionRecommendation } from "../generation/types.js";
import type { SessionId } from "../../util/ids.js";
import type { JsonValue } from "../../util/json-value.js";
import type { DeliberationUsage, SelfSnapshot } from "./types.js";
import { DEFAULT_DELIBERATION_PLAN_CALL_TIMEOUT_MS } from "./constants.js";
import { renderTaggedPromptSection } from "./prompt/sections.js";
import { summarizeVoiceAnchors } from "./prompt/voice-anchors.js";
import { mergeDeliberationUsage, usageFromCompleteResult } from "./usage.js";
import {
  AUTONOMOUS_WANT_PROMPT_BLOCK,
  buildPlannerDirective,
  TURN_PLAN_TOOL_NAME,
} from "./prompt/planner-contract.js";
import type {
  CompactPlannerSystemPrompt,
  PlannerContextTraceSummary,
} from "./prompt/planner-context.js";
import { estimatePromptTokens } from "../../util/token-estimate.js";
import {
  PROMPT_SURFACES,
  renderPromptSurfaceAdditionalBlock,
  renderPromptSurface,
  type PromptSurfaceAdditionalSection,
  type PromptSurfaceRenderContext,
} from "../prompts/prompt-surface-registry.js";

const AUTONOMOUS_WANT_FIELD_DESCRIPTION =
  "Answered before every field below: setting aside whether it is justified, is there anything I want in this interval? Named plainly in my own words. Naming a want does not oblige me to act on it. Empty string when nothing genuinely surfaces -- an empty answer is complete, and I do not manufacture a want to fill it.";

const turnPlanSchema = z.object({
  uncertainty: z
    .string()
    .describe(
      "What's unclear about the current participant input that matters for the engagement decision or answer? Empty string if nothing.",
    ),
  verification_steps: z
    .array(z.string())
    .describe(
      "Short phrases describing what I should double-check or re-retrieve before engaging. Empty array if nothing.",
    ),
  tensions: z
    .array(z.string())
    .describe(
      "Conflicts or contradictions in what I already know that need to be reconciled if I respond. Empty array if none.",
    ),
  voice_note: z
    .string()
    .describe(
      "How the voice and posture should land for this specific turn. Empty string if default voice fits.",
    ),
  emission_recommendation: z
    .enum(["emit", "no_output"])
    .default("emit")
    .describe(
      "I use no_output only when the conversation has naturally closed and the correct current-turn behavior is to emit no visible message at all; otherwise I use emit and let the finalizer choose visible speech or observation.",
    ),
  intents: z
    .array(intentRecordSchema)
    .describe(
      "Follow-up intent records to carry into working memory after this turn. I include only concrete future actions I actually intend to track, not stylistic next-step wording.",
    ),
});

const autonomousTurnPlanSchema = z.object({
  want: z.string().default("").describe(AUTONOMOUS_WANT_FIELD_DESCRIPTION),
  ...turnPlanSchema.shape,
});

type BaseTurnPlan = z.infer<typeof turnPlanSchema>;

export type TurnPlan = BaseTurnPlan & { want?: string };
export type TurnPlanEmissionRecommendation = EmissionRecommendation;

export { TURN_PLAN_TOOL_NAME } from "./prompt/planner-contract.js";

const TURN_PLAN_TOOL_DESCRIPTION =
  "I emit a structured plan for this reflective/high-stakes turn before the final engagement decision. The plan is passed back to me in the final-response call so I can execute against it. I emit follow-up intents only for concrete future actions worth carrying in working memory.";

function createTurnPlanTool(schema: z.ZodType): LLMToolDefinition {
  return {
    name: TURN_PLAN_TOOL_NAME,
    description: TURN_PLAN_TOOL_DESCRIPTION,
    inputSchema: toToolInputSchema(schema),
    // The compact planner's final static-head system block owns the cache
    // breakpoint; Anthropic's prefix at that point already includes this
    // stable tool schema. A second marker here would spend a breakpoint and
    // leave the legacy rollback path paying for an otherwise ineligible head.
  };
}

const TURN_PLAN_TOOL: LLMToolDefinition = createTurnPlanTool(turnPlanSchema);
const AUTONOMOUS_TURN_PLAN_TOOL: LLMToolDefinition = createTurnPlanTool(autonomousTurnPlanSchema);

function resolveTurnPlanTool(turnOrigin: TurnOrigin | undefined): LLMToolDefinition {
  return turnOrigin === "autonomous" ? AUTONOMOUS_TURN_PLAN_TOOL : TURN_PLAN_TOOL;
}

function resolveTurnPlanSchema(turnOrigin: TurnOrigin | undefined): z.ZodType<TurnPlan> {
  return turnOrigin === "autonomous" ? autonomousTurnPlanSchema : turnPlanSchema;
}

const PLANNER_RETRY_HINT =
  "My previous response did not include the required EmitTurnPlan tool_use block. I emit one now -- this is the only way to complete the plan step.";

export type RunS2PlannerOptions = {
  llmClient: LLMClient;
  model: string;
  baseSystemPrompt: string;
  dialogueMessages: readonly LLMMessage[];
  selfSnapshot: SelfSnapshot;
  additionalPromptSections?: readonly PromptSurfaceAdditionalSection[];
  maxTokens: number;
  thinking?: LLMCompleteOptions["thinking"];
  effort?: LLMCompleteOptions["effort"];
  tracer?: TurnTracer;
  turnId?: string;
  sessionId?: SessionId;
  turnOrigin?: TurnOrigin;
  plannerSurface?: { variant: "legacy" } | ({ variant: "compact" } & CompactPlannerSystemPrompt);
};

export type S2PlannerResult = {
  plan: TurnPlan | null;
  reasoning: string;
  usage: DeliberationUsage;
};

const EMPTY_S2_PLANNER_USAGE = {
  input_tokens: 0,
  output_tokens: 0,
  stop_reason: null,
} satisfies DeliberationUsage;

function degradedPlannerResult(
  options: RunS2PlannerOptions,
  input: {
    attempts: number;
    lastResponseShape: JsonValue;
    reasoning?: string;
    usage?: DeliberationUsage;
  },
): S2PlannerResult {
  if (options.tracer?.enabled === true && options.turnId !== undefined) {
    options.tracer.emit("deliberation.planner.degraded", {
      turnId: options.turnId,
      ...(options.sessionId === undefined ? {} : { session_id: options.sessionId }),
      attempts: input.attempts,
      lastResponseShape: input.lastResponseShape,
    });
  }

  return {
    plan: null,
    reasoning: input.reasoning ?? "",
    usage: input.usage ?? { ...EMPTY_S2_PLANNER_USAGE },
  };
}

function createS2PlannerPromptSurfaceRenderContext(
  options: RunS2PlannerOptions,
): PromptSurfaceRenderContext {
  return {
    renderBlock: (id) => {
      switch (id) {
        case "s2_planner_base_system_prompt":
          return options.baseSystemPrompt;
        case "borg_voice_anchors":
          return renderTaggedPromptSection(
            "borg_voice_anchors",
            summarizeVoiceAnchors(options.selfSnapshot),
          );
        case "borg_session_reentry_continuity":
        case "borg_compact_planner_ledger":
        case "borg_unresolved_contradiction_open_questions":
          return renderPromptSurfaceAdditionalBlock(id, options.additionalPromptSections);
        case "s2_planner_autonomous_want":
          return options.turnOrigin === "autonomous" ? AUTONOMOUS_WANT_PROMPT_BLOCK : null;
        case "s2_planner_directive":
          return buildPlannerDirective();
        default:
          return null;
      }
    },
  };
}

function legacyPlannerTraceSummary(systemPrompt: string): PlannerContextTraceSummary {
  return {
    variant: "legacy",
    sections: {
      legacy_full_surface: {
        chars: systemPrompt.length,
        estimatedTokens: estimatePromptTokens(systemPrompt),
        rowCount: 0,
        truncationCount: 0,
        omissionCount: 0,
        criticalOverflow: false,
      },
    },
    targetTokens: null,
    totalChars: systemPrompt.length,
    totalEstimatedTokens: estimatePromptTokens(systemPrompt),
    rowCount: 0,
    truncationCount: 0,
    omissionCount: 0,
    criticalOverflow: false,
    overallOverflow: false,
  };
}

function plannerContextSummaryForTrace(summary: PlannerContextTraceSummary): JsonValue {
  return {
    variant: summary.variant,
    sections: Object.fromEntries(
      Object.entries(summary.sections).map(([label, section]) => [
        label,
        {
          chars: section.chars,
          estimated_tokens: section.estimatedTokens,
          row_count: section.rowCount,
          truncation_count: section.truncationCount,
          omission_count: section.omissionCount,
          critical_overflow: section.criticalOverflow,
        },
      ]),
    ),
    target_tokens: summary.targetTokens,
    total_chars: summary.totalChars,
    total_estimated_tokens: summary.totalEstimatedTokens,
    row_count: summary.rowCount,
    truncation_count: summary.truncationCount,
    omission_count: summary.omissionCount,
    critical_overflow: summary.criticalOverflow,
    overall_overflow: summary.overallOverflow,
  };
}

export async function runS2Planner(options: RunS2PlannerOptions): Promise<S2PlannerResult> {
  const surface = options.plannerSurface;
  const legacySystemPrompt = (): string =>
    renderPromptSurface(
      PROMPT_SURFACES.s2PlannerSystem,
      createS2PlannerPromptSurfaceRenderContext(options),
    ) ?? "";
  const systemPrompt: string | readonly LLMSystemBlock[] =
    surface?.variant === "compact" ? surface.system : legacySystemPrompt();
  const traceSummary =
    surface?.variant === "compact"
      ? surface.traceSummary
      : legacyPlannerTraceSummary(systemPrompt as string);
  const tools = [resolveTurnPlanTool(options.turnOrigin)];
  let result: PlannerAttemptResult;

  try {
    result = await callPlannerAttempt(
      options,
      systemPrompt,
      traceSummary,
      tools,
      options.dialogueMessages,
    );
  } catch (error) {
    if (!isRetryableLlmTransportError(error)) {
      throw error;
    }

    return degradedPlannerResult(options, {
      attempts: 1,
      lastResponseShape: {
        error: error.message,
        code: error.code,
      },
    });
  }

  let usage = result.usage;

  if (result.extraction.plan === null) {
    // Retry forces the plan tool (which precludes thinking) so a plan is
    // guaranteed even when the thinking-mode first attempt emitted no tool call.
    const firstResult = result;

    try {
      result = await callPlannerAttempt(
        { ...options, thinking: undefined, effort: undefined },
        systemPrompt,
        traceSummary,
        tools,
        [
          ...options.dialogueMessages,
          {
            role: "user",
            content: PLANNER_RETRY_HINT,
          },
        ],
      );
    } catch (error) {
      if (!isRetryableLlmTransportError(error)) {
        throw error;
      }

      return degradedPlannerResult(options, {
        attempts: 2,
        lastResponseShape: {
          error: error.message,
          code: error.code,
        },
        reasoning: firstResult.planner.text,
        usage,
      });
    }

    usage = mergeDeliberationUsage(usage, result.usage);
  }

  if (result.extraction.plan === null) {
    return degradedPlannerResult(options, {
      attempts: 2,
      lastResponseShape: summarizeToolResponseShape(result.planner),
      reasoning: result.planner.text,
      usage,
    });
  }

  return {
    plan: result.extraction.plan,
    reasoning: result.planner.text,
    usage,
  };
}

type PlannerAttemptResult = {
  planner: Awaited<ReturnType<LLMClient["complete"]>>;
  extraction: ExtractTurnPlanResult;
  usage: DeliberationUsage;
};

async function callPlannerAttempt(
  options: RunS2PlannerOptions,
  systemPrompt: string | readonly LLMSystemBlock[],
  traceSummary: PlannerContextTraceSummary,
  tools: readonly LLMToolDefinition[],
  messages: readonly LLMMessage[],
): Promise<PlannerAttemptResult> {
  const systemPromptForTrace =
    typeof systemPrompt === "string"
      ? systemPrompt
      : systemPrompt.map((block) => block.text).join("\n\n");
  traceLlmCallStarted({
    tracer: options.tracer,
    turnId: options.turnId,
    sessionId: options.sessionId,
    label: "s2_planner",
    model: options.model,
    systemPrompt: systemPromptForTrace,
    messages,
    tools,
    extra: {
      planner_surface_variant: traceSummary.variant,
      planner_context_summary: plannerContextSummaryForTrace(traceSummary),
      ...(options.tracer?.includePayloads === true
        ? {
            prompt: toTraceJsonValue({
              system: systemPrompt,
              messages,
              tools,
            }),
          }
        : {}),
    },
  });

  const onTransportRetry = traceLlmCallRetryHook({
    tracer: options.tracer,
    turnId: options.turnId,
    sessionId: options.sessionId,
    label: "s2_planner",
  });
  const completeOptions = {
    model: options.model,
    system: systemPrompt,
    messages,
    tools,
    // Thinking requires auto tool_choice (the API rejects forced tool use with
    // thinking). When thinking will actually be sent, omit tool_choice so the
    // model may think before emitting EmitTurnPlan; a missing plan tool-call is
    // already handled (retry hint forces the tool, then degraded plan=null).
    // Otherwise force the plan tool.
    ...(willSendThinkingUnderAutoToolChoice(options.model, options.thinking)
      ? {}
      : { tool_choice: { type: "tool" as const, name: TURN_PLAN_TOOL_NAME } }),
    max_tokens: options.maxTokens,
    ...(options.thinking === undefined ? {} : { thinking: options.thinking }),
    ...(options.effort === undefined ? {} : { effort: options.effort }),
    ...(onTransportRetry === undefined ? {} : { onTransportRetry }),
    timeoutMs: DEFAULT_DELIBERATION_PLAN_CALL_TIMEOUT_MS,
    budget: "cognition-plan",
  } satisfies LLMCompleteOptions;
  const planner = await options.llmClient.complete(completeOptions);
  const extraction = extractTurnPlan(planner.tool_calls, options.turnOrigin);

  if (options.tracer?.enabled === true && options.turnId !== undefined) {
    traceLlmCallResponse({
      tracer: options.tracer,
      turnId: options.turnId,
      sessionId: options.sessionId,
      label: "s2_planner",
      response: planner,
      responseShape: summarizeToolResponseShape(planner),
      extra:
        options.tracer.includePayloads === true
          ? {
              response: toTraceJsonValue({
                text: planner.text,
                toolCalls: planner.tool_calls,
              }),
            }
          : undefined,
    });
    options.tracer.emit("deliberation.plan.completed", {
      turnId: options.turnId,
      ...(options.sessionId === undefined ? {} : { session_id: options.sessionId }),
      success: extraction.plan !== null,
      ...(extraction.reason === null ? {} : { reason: extraction.reason }),
    });
  }

  return {
    planner,
    extraction,
    usage: usageFromCompleteResult(planner),
  };
}

type ExtractTurnPlanResult = {
  plan: TurnPlan | null;
  reason: string | null;
};

function extractTurnPlan(
  toolCalls: readonly LLMToolCall[],
  turnOrigin: TurnOrigin | undefined,
): ExtractTurnPlanResult {
  const call = toolCalls.find((entry) => entry.name === TURN_PLAN_TOOL_NAME);

  if (call === undefined) {
    return {
      plan: null,
      reason: "missing_emit_turn_plan_tool_use",
    };
  }

  const parsed = resolveTurnPlanSchema(turnOrigin).safeParse(call.input);
  if (!parsed.success) {
    return {
      plan: null,
      reason: "invalid_emit_turn_plan_input",
    };
  }

  return {
    plan: parsed.data,
    reason: null,
  };
}
