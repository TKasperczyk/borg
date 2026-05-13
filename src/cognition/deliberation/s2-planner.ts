// Runs the System 2 structured planner call and parses its EmitTurnPlan output.
import { z } from "zod";

import {
  type LLMClient,
  type LLMCompleteOptions,
  type LLMMessage,
  type LLMToolCall,
  type LLMToolDefinition,
  toToolInputSchema,
} from "../../llm/index.js";
import type { JsonValue } from "../../util/json-value.js";
import type { TurnTracer } from "../tracing/tracer.js";
import { buildUsageTraceBlock, toTraceJsonValue } from "../tracing/tracer.js";
import { intentRecordSchema } from "../types.js";
import type { EmissionRecommendation } from "../generation/types.js";
import type { DeliberationUsage, SelfSnapshot } from "./types.js";
import { renderTaggedPromptSection } from "./prompt/sections.js";
import { summarizeVoiceAnchors } from "./prompt/voice-anchors.js";

const turnPlanSchema = z.object({
  uncertainty: z
    .string()
    .describe(
      "What's unclear about the current participant input that matters for the engagement decision or answer? Empty string if nothing.",
    ),
  verification_steps: z
    .array(z.string())
    .describe(
      "Short phrases describing what you should double-check or re-retrieve before engaging. Empty array if nothing.",
    ),
  tensions: z
    .array(z.string())
    .describe(
      "Conflicts or contradictions in what you already know that need to be reconciled if you respond. Empty array if none.",
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
      "Use no_output only when the conversation has naturally closed and the correct current-turn behavior is to emit no assistant message at all; otherwise use emit and let the finalizer choose visible speech or observation.",
    ),
  referenced_episode_ids: z
    .array(z.string())
    .describe(
      "List the episode_ids from borg_retrieved_evidence that you actually used as evidence; empty if none were drawn on.",
    ),
  intents: z
    .array(intentRecordSchema)
    .describe(
      "Follow-up intent records to carry into working memory after this turn. Include only concrete future actions you actually intend to track, not stylistic next-step wording.",
    ),
});

export type TurnPlan = z.infer<typeof turnPlanSchema>;
export type TurnPlanEmissionRecommendation = EmissionRecommendation;

export const TURN_PLAN_TOOL_NAME = "EmitTurnPlan";

const TURN_PLAN_TOOL: LLMToolDefinition = {
  name: TURN_PLAN_TOOL_NAME,
  description:
    "Emit a structured plan for this reflective/high-stakes turn before the final engagement decision. The plan is passed back to you in the final-response call so you can execute against it. List the episode_ids from borg_retrieved_evidence that you actually used as evidence; empty if none were drawn on. Emit follow-up intents only for concrete future actions worth carrying in working memory.",
  inputSchema: toToolInputSchema(turnPlanSchema),
  // Sprint 8d.6.5 placed cache_control here, but v39 traces (codex
  // 1b0384c3) showed it was a no-op: TURN_PLAN_TOOL JSON is ~2.2KB,
  // well under Opus 4.7's 4096-token minimum cacheable prefix. The
  // single 6505-token cache_create observed on call 1 was actually
  // the retry path's per-turn prefix, which never gets reused because
  // the planner's baseSystemPrompt is fully dynamic. Removing the
  // marker eliminates the wasted 1.25x cache write on retries. The
  // planner has no stable >=4096-token prefix to cache today, so it
  // doesn't get caching until that changes.
};

const PLANNER_RETRY_HINT =
  "Your previous response did not include the required EmitTurnPlan tool_use block. Emit one now -- this is the only way to complete the plan step.";

export type RunS2PlannerOptions = {
  llmClient: LLMClient;
  model: string;
  baseSystemPrompt: string;
  dialogueMessages: readonly LLMMessage[];
  selfSnapshot: SelfSnapshot;
  additionalPromptSections?: readonly (string | null)[];
  maxTokens: number;
  thinking?: LLMCompleteOptions["thinking"];
  tracer?: TurnTracer;
  turnId?: string;
};

export type S2PlannerResult = {
  plan: TurnPlan | null;
  reasoning: string;
  usage: DeliberationUsage;
};

export async function runS2Planner(options: RunS2PlannerOptions): Promise<S2PlannerResult> {
  const plannerVoiceAnchors = renderTaggedPromptSection(
    "borg_voice_anchors",
    summarizeVoiceAnchors(options.selfSnapshot),
  );
  const systemPrompt = [
    options.baseSystemPrompt,
    plannerVoiceAnchors,
    ...(options.additionalPromptSections ?? []),
    [
      "You are about to decide whether and how to engage with a reflective, high-stakes, or contradictory turn.",
      `Emit a structured plan by calling the ${TURN_PLAN_TOOL_NAME} tool exactly once.`,
      "The plan is passed back to you in the next call so you can execute it. Keep it short and grounded in the current turn -- do NOT try to draft the answer itself here.",
      "Set emission_recommendation='no_output' only when the conversation has naturally closed. Do not describe silence in voice_note.",
      "Use plan.intents only for concrete future actions you mean to carry into later turns. Leave it empty when no follow-up state should persist.",
    ].join("\n"),
  ]
    .filter((section): section is string => section !== null)
    .join("\n\n");
  const tools = [TURN_PLAN_TOOL];
  let result = await callPlannerAttempt(options, systemPrompt, tools, options.dialogueMessages);
  let usage = result.usage;

  if (result.extraction.plan === null) {
    result = await callPlannerAttempt(options, systemPrompt, tools, [
      ...options.dialogueMessages,
      {
        role: "user",
        content: PLANNER_RETRY_HINT,
      },
    ]);
    usage = aggregatePlannerUsage(usage, result.usage);
  }

  if (
    result.extraction.plan === null &&
    options.tracer?.enabled === true &&
    options.turnId !== undefined
  ) {
    options.tracer.emit("s2_planner_exhausted", {
      turnId: options.turnId,
      attempts: 2,
      lastResponseShape: summarizePlannerResponseShape(result.planner),
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

function aggregatePlannerUsage(
  current: DeliberationUsage,
  next: DeliberationUsage,
): DeliberationUsage {
  const cacheCreation =
    current.cache_creation_input_tokens === undefined &&
    next.cache_creation_input_tokens === undefined
      ? undefined
      : (current.cache_creation_input_tokens ?? 0) + (next.cache_creation_input_tokens ?? 0);
  const cacheRead =
    current.cache_read_input_tokens === undefined && next.cache_read_input_tokens === undefined
      ? undefined
      : (current.cache_read_input_tokens ?? 0) + (next.cache_read_input_tokens ?? 0);
  return {
    input_tokens: current.input_tokens + next.input_tokens,
    output_tokens: current.output_tokens + next.output_tokens,
    stop_reason: next.stop_reason,
    ...(cacheCreation === undefined ? {} : { cache_creation_input_tokens: cacheCreation }),
    ...(cacheRead === undefined ? {} : { cache_read_input_tokens: cacheRead }),
  };
}

function summarizePlannerResponseShape(planner: Awaited<ReturnType<LLMClient["complete"]>>) {
  return {
    textLength: planner.text.length,
    toolUseBlocks: planner.tool_calls.map((call) => ({
      id: call.id,
      name: call.name,
    })),
  };
}

async function callPlannerAttempt(
  options: RunS2PlannerOptions,
  systemPrompt: string,
  tools: readonly LLMToolDefinition[],
  messages: readonly LLMMessage[],
): Promise<PlannerAttemptResult> {
  if (options.tracer?.enabled === true && options.turnId !== undefined) {
    options.tracer.emit("llm_call_started", {
      turnId: options.turnId,
      label: "s2_planner",
      model: options.model,
      promptCharCount: countCompletePromptChars(systemPrompt, messages),
      toolSchemas: summarizeToolSchemas(tools),
      ...(options.tracer.includePayloads
        ? {
            prompt: toTraceJsonValue({
              system: systemPrompt,
              messages,
              tools,
            }),
          }
        : {}),
    });
  }

  const planner = await options.llmClient.complete({
    model: options.model,
    system: systemPrompt,
    messages,
    tools,
    tool_choice: { type: "tool", name: TURN_PLAN_TOOL_NAME },
    max_tokens: options.maxTokens,
    ...(options.thinking === undefined ? {} : { thinking: options.thinking }),
    budget: "cognition-plan",
  });
  const extraction = extractTurnPlan(planner.tool_calls);

  if (options.tracer?.enabled === true && options.turnId !== undefined) {
    options.tracer.emit("llm_call_response", {
      turnId: options.turnId,
      label: "s2_planner",
      responseShape: summarizePlannerResponseShape(planner),
      stopReason: planner.stop_reason,
      usage: buildUsageTraceBlock(planner),
      ...(options.tracer.includePayloads
        ? {
            response: toTraceJsonValue({
              text: planner.text,
              toolCalls: planner.tool_calls,
            }),
          }
        : {}),
    });
    options.tracer.emit("plan_extraction", {
      turnId: options.turnId,
      success: extraction.plan !== null,
      ...(extraction.reason === null ? {} : { reason: extraction.reason }),
    });
  }

  return {
    planner,
    extraction,
    usage: {
      input_tokens: planner.input_tokens,
      output_tokens: planner.output_tokens,
      stop_reason: planner.stop_reason,
      ...(planner.cache_creation_input_tokens === undefined
        ? {}
        : { cache_creation_input_tokens: planner.cache_creation_input_tokens }),
      ...(planner.cache_read_input_tokens === undefined
        ? {}
        : { cache_read_input_tokens: planner.cache_read_input_tokens }),
    },
  };
}

type ExtractTurnPlanResult = {
  plan: TurnPlan | null;
  reason: string | null;
};

function extractTurnPlan(toolCalls: readonly LLMToolCall[]): ExtractTurnPlanResult {
  const call = toolCalls.find((entry) => entry.name === TURN_PLAN_TOOL_NAME);

  if (call === undefined) {
    return {
      plan: null,
      reason: "missing_emit_turn_plan_tool_use",
    };
  }

  const parsed = turnPlanSchema.safeParse(call.input);
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

function countCompletePromptChars(systemPrompt: string, messages: readonly LLMMessage[]): number {
  return (
    systemPrompt.length +
    messages.reduce((sum, message) => sum + message.role.length + message.content.length, 0)
  );
}

function summarizeToolSchemas(tools: readonly LLMToolDefinition[]): JsonValue {
  return tools.map((tool) => ({
    name: tool.name,
    propertyCount:
      tool.inputSchema.properties === undefined
        ? 0
        : Object.keys(tool.inputSchema.properties).length,
    required: Array.isArray(tool.inputSchema.required) ? tool.inputSchema.required.map(String) : [],
  }));
}
