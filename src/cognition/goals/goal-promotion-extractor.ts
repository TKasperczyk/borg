import { z } from "zod";

import { executiveStepKindSchema, type ExecutiveStepKind } from "../../executive/types.js";
import {
  type LLMClient,
  type LLMCompleteResult,
  type LLMMessage,
  type LLMToolDefinition,
  toToolInputSchema,
} from "../../llm/index.js";
import { goalIdSchema, type GoalRecord } from "../../memory/self/index.js";
import type { JsonValue } from "../../util/json-value.js";
import type { EntityId } from "../../util/ids.js";
import type { RecencyMessage } from "../recency/index.js";
import type { TurnTracer } from "../tracing/tracer.js";

const CONFIDENCE_THRESHOLD = 0.85;
const MAX_PROMOTIONS_PER_TURN = 3;
const GOAL_PROMOTION_TOOL_NAME = "EmitGoalPromotion";

const initialExecutiveStepSchema = z
  .object({
    description: z
      .string()
      .trim()
      .min(1)
      .describe("A concrete first executive step Borg can take or track for this new goal."),
    kind: executiveStepKindSchema.describe("The operational kind of the first step."),
    due_at: z
      .number()
      .finite()
      .nullable()
      .optional()
      .describe("Optional due timestamp in Unix epoch milliseconds. Use null if absent."),
    rationale: z.string().trim().min(1).describe("Why this step follows from the goal request."),
  })
  .strict();

const goalPromotionSchema = z
  .object({
    classification: z
      .enum(["promote", "none"])
      .optional()
      .describe("Use promote only when Borg has an ongoing role for this goal."),
    description: z
      .string()
      .trim()
      .min(1)
      .describe("Concise durable goal Borg should carry forward."),
    priority: z
      .number()
      .finite()
      .min(0)
      .max(10)
      .describe(
        "Relative priority from 0 to 10. Prefer moderate values unless urgency is explicit.",
      ),
    target_at: z
      .number()
      .finite()
      .nullable()
      .describe("Target completion timestamp in Unix epoch milliseconds, or null if no deadline."),
    reason: z
      .string()
      .trim()
      .min(1)
      .describe("Semantic reason Borg has an ongoing tracking, support, or follow-up role."),
    confidence: z
      .number()
      .min(0)
      .max(1)
      .describe("Confidence that the current user turn creates a durable Borg-carried goal."),
    duplicate_of_goal_id: goalIdSchema
      .nullable()
      .describe("Existing active goal id if this turn refers to an existing goal; null otherwise."),
    initial_step: initialExecutiveStepSchema
      .nullable()
      .optional()
      .describe("Optional first executive step for a newly promoted goal."),
  })
  .strict();

const goalPromotionOutputSchema = z
  .object({
    promotions: z
      .array(goalPromotionSchema)
      .describe(
        "Goal-promotion candidates. Emit an empty array when no new Borg-carried goal is created.",
      ),
  })
  .strict();

const GOAL_PROMOTION_TOOL = {
  name: GOAL_PROMOTION_TOOL_NAME,
  description:
    "Extract durable goals only when Borg has an ongoing tracking, support, reminder, or follow-up role.",
  inputSchema: toToolInputSchema(goalPromotionOutputSchema),
} satisfies LLMToolDefinition;

const GOAL_PROMOTION_SYSTEM_PROMPT = [
  "Classify whether the current user turn creates a durable goal Borg should carry as active self-memory.",
  "Promote only when the user asks Borg to track, support, remind, follow up, keep organized, or otherwise carry an ongoing role; or when the turn clearly establishes that Borg has committed to ongoing support.",
  "Do not promote a goal just because the user mentions a possible intention, appointment, task, wish, plan, or event. Those may be pending actions or ordinary conversation, not Borg goals.",
  "Judge semantic intent across languages. Do not rely on wording, punctuation, capitalization, or phrase shapes.",
  "If an existing active goal already covers the request, set duplicate_of_goal_id and do not create a new goal.",
  "Use target_at only for a real goal deadline. Use the supplied temporal cue as context, not as an automatic trigger.",
  "When uncertain, emit no promotions. Return only the required tool call.",
  "",
  "Examples:",
  "- Help me track my italki shortlist -> promote, because Borg has a tracking role.",
  "- I might book italki tonight -> no promotion, because Borg has no ongoing role.",
  "- Postmortem Monday, help me keep this straight -> promote with an initial step.",
  "- Doctor appointment Tuesday -> no promotion unless the user asks Borg to track or follow up.",
].join("\n");

type GoalPromotionToolInput = z.infer<typeof goalPromotionOutputSchema>;

class MissingGoalPromotionToolCallError extends Error {}

export type GoalPromotionInitialStep = {
  description: string;
  kind: ExecutiveStepKind;
  due_at: number | null;
  rationale: string;
};

export type GoalPromotionCandidate = {
  description: string;
  priority: number;
  target_at: number | null;
  reason: string;
  confidence: number;
  initial_step: GoalPromotionInitialStep | null;
};

export type GoalPromotionExtractorDegradedReason =
  | "llm_unavailable"
  | "llm_failed"
  | "missing_tool_call"
  | "invalid_payload";

export type GoalPromotionExtractorOptions = {
  llmClient?: LLMClient;
  model?: string;
  tracer?: TurnTracer;
  turnId?: string;
  onDegraded?: (
    reason: GoalPromotionExtractorDegradedReason,
    error?: unknown,
  ) => Promise<void> | void;
};

export type ExtractGoalPromotionInput = {
  userMessage: string;
  recentHistory: readonly RecencyMessage[];
  audienceEntityId: EntityId | null;
  temporalCue: unknown;
  activeGoals: readonly Pick<GoalRecord, "id" | "description" | "priority" | "target_at">[];
};

function toCandidates(input: GoalPromotionToolInput): GoalPromotionCandidate[] {
  const candidates: GoalPromotionCandidate[] = [];

  for (const promotion of input.promotions.slice(0, MAX_PROMOTIONS_PER_TURN)) {
    if (
      promotion.classification === "none" ||
      promotion.confidence < CONFIDENCE_THRESHOLD ||
      promotion.duplicate_of_goal_id !== null
    ) {
      continue;
    }

    const description = promotion.description.trim();
    const reason = promotion.reason.trim();

    if (description.length === 0 || reason.length === 0) {
      continue;
    }

    candidates.push({
      description,
      priority: promotion.priority,
      target_at: promotion.target_at,
      reason,
      confidence: promotion.confidence,
      initial_step:
        promotion.initial_step === null || promotion.initial_step === undefined
          ? null
          : {
              description: promotion.initial_step.description.trim(),
              kind: promotion.initial_step.kind,
              due_at: promotion.initial_step.due_at ?? null,
              rationale: promotion.initial_step.rationale.trim(),
            },
    });
  }

  return candidates;
}

function parseResponse(result: LLMCompleteResult): GoalPromotionCandidate[] {
  const call = result.tool_calls.find((toolCall) => toolCall.name === GOAL_PROMOTION_TOOL_NAME);

  if (call === undefined) {
    throw new MissingGoalPromotionToolCallError(
      `Goal promotion extractor did not emit ${GOAL_PROMOTION_TOOL_NAME}`,
    );
  }

  const parsed = goalPromotionOutputSchema.safeParse(call.input);

  if (!parsed.success) {
    throw parsed.error;
  }

  return toCandidates(parsed.data);
}

function buildGoalPromotionMessages(input: ExtractGoalPromotionInput): LLMMessage[] {
  return [
    {
      role: "user",
      content: JSON.stringify({
        current_user_message: input.userMessage,
        recent_history: input.recentHistory.slice(-8).map((message) => ({
          role: message.role,
          content: message.content,
        })),
        audience_entity_id: input.audienceEntityId,
        temporal_cue: input.temporalCue,
        active_goals: input.activeGoals.map((goal) => ({
          id: goal.id,
          description: goal.description,
          priority: goal.priority,
          target_at: goal.target_at,
        })),
      }),
    },
  ];
}

function summarizeGoalPromotionResponseShape(response: LLMCompleteResult): JsonValue {
  return {
    textLength: response.text.length,
    toolUseBlocks: response.tool_calls.map((call) => ({
      id: call.id,
      name: call.name,
    })),
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

function traceLlmCallStarted(options: {
  tracer?: TurnTracer;
  turnId?: string;
  model: string;
  messages: readonly LLMMessage[];
  tools: readonly LLMToolDefinition[];
}): void {
  if (options.tracer?.enabled === true && options.turnId !== undefined) {
    options.tracer.emit("llm_call_started", {
      turnId: options.turnId,
      label: "goal_promotion_extractor",
      model: options.model,
      promptCharCount: countCompletePromptChars(GOAL_PROMOTION_SYSTEM_PROMPT, options.messages),
      toolSchemas: summarizeToolSchemas(options.tools),
    });
  }
}

function traceLlmCallResponse(options: {
  tracer?: TurnTracer;
  turnId?: string;
  response: LLMCompleteResult;
}): void {
  if (options.tracer?.enabled === true && options.turnId !== undefined) {
    options.tracer.emit("llm_call_response", {
      turnId: options.turnId,
      label: "goal_promotion_extractor",
      responseShape: summarizeGoalPromotionResponseShape(options.response),
      stopReason: options.response.stop_reason,
      usage: {
        inputTokens: options.response.input_tokens,
        outputTokens: options.response.output_tokens,
      },
    });
  }
}

function traceLlmCallError(options: {
  tracer?: TurnTracer;
  turnId?: string;
  error: unknown;
}): void {
  if (options.tracer?.enabled === true && options.turnId !== undefined) {
    options.tracer.emit("llm_call_response", {
      turnId: options.turnId,
      label: "goal_promotion_extractor",
      responseShape: {
        error: options.error instanceof Error ? options.error.message : String(options.error),
      },
      stopReason: null,
      usage: null,
    });
  }
}

export class GoalPromotionExtractor {
  constructor(private readonly options: GoalPromotionExtractorOptions = {}) {}

  private async degraded(
    reason: GoalPromotionExtractorDegradedReason,
    error?: unknown,
  ): Promise<GoalPromotionCandidate[]> {
    try {
      await this.options.onDegraded?.(reason, error);
    } catch {
      // Best-effort degraded-mode logging only.
    }

    return [];
  }

  async extract(input: ExtractGoalPromotionInput): Promise<GoalPromotionCandidate[]> {
    if (this.options.llmClient === undefined || this.options.model === undefined) {
      return this.degraded("llm_unavailable");
    }

    const messages = buildGoalPromotionMessages(input);
    const tools = [GOAL_PROMOTION_TOOL];

    traceLlmCallStarted({
      tracer: this.options.tracer,
      turnId: this.options.turnId,
      model: this.options.model,
      messages,
      tools,
    });

    let response: LLMCompleteResult;

    try {
      response = await this.options.llmClient.complete({
        model: this.options.model,
        system: GOAL_PROMOTION_SYSTEM_PROMPT,
        messages,
        tools,
        tool_choice: { type: "tool", name: GOAL_PROMOTION_TOOL_NAME },
        max_tokens: 768,
        budget: "goal-promotion-extractor",
      });
    } catch (error) {
      traceLlmCallError({
        tracer: this.options.tracer,
        turnId: this.options.turnId,
        error,
      });

      return this.degraded("llm_failed", error);
    }

    traceLlmCallResponse({
      tracer: this.options.tracer,
      turnId: this.options.turnId,
      response,
    });

    try {
      return parseResponse(response);
    } catch (error) {
      return this.degraded(
        error instanceof MissingGoalPromotionToolCallError
          ? "missing_tool_call"
          : error instanceof z.ZodError
            ? "invalid_payload"
            : "llm_failed",
        error,
      );
    }
  }
}

export { GOAL_PROMOTION_TOOL_NAME };
