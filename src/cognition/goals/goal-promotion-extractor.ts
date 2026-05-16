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
import { buildUsageTraceBlock, type TurnTracer } from "../tracing/tracer.js";

const CONFIDENCE_THRESHOLD = 0.85;
const MAX_PROMOTIONS_PER_TURN = 3;
const GOAL_PROMOTION_MAX_TOKENS = 1536;
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

const goalPromotionEnvelopeSchema = z
  .object({
    promotions: z.array(z.unknown()),
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
  "When speaker_entity_id is supplied and the current speaker creates a durable first-person goal, treat that speaker as the goal owner. In group chat, first-person user goals belong to the current sender, not the group, unless the message explicitly says the group is acting.",
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

type ParsedGoalPromotion = z.infer<typeof goalPromotionSchema>;
type GoalPromotionEnvelopeInput = z.infer<typeof goalPromotionEnvelopeSchema>;

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

export type GoalPromotionSkippedReason =
  | "missing_description"
  | "invalid_priority"
  | "invalid_target_at"
  | "invalid_reason"
  | "invalid_confidence"
  | "invalid_duplicate_of_goal_id"
  | "invalid_initial_step"
  | "invalid_classification"
  | "invalid_promotion";

export type GoalPromotionSkippedPromotion = {
  candidate_index: number;
  reason: GoalPromotionSkippedReason;
};

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
  speakerEntityId?: EntityId | null;
  speakerDisplayName?: string | null;
  temporalCue: unknown;
  activeGoals: readonly Pick<GoalRecord, "id" | "description" | "priority" | "target_at">[];
};

type GoalPromotionParseResult = {
  candidates: GoalPromotionCandidate[];
  validPromotionCount: number;
  skippedPromotions: GoalPromotionSkippedPromotion[];
};

function toCandidates(promotions: readonly ParsedGoalPromotion[]): GoalPromotionCandidate[] {
  const candidates: GoalPromotionCandidate[] = [];

  for (const promotion of promotions.slice(0, MAX_PROMOTIONS_PER_TURN)) {
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

function skippedReasonFromIssue(issue: {
  path: readonly (string | number | symbol)[];
}): GoalPromotionSkippedReason {
  const field = issue.path[0];

  if (field === "description") {
    return "missing_description";
  }

  if (field === "priority") {
    return "invalid_priority";
  }

  if (field === "target_at") {
    return "invalid_target_at";
  }

  if (field === "reason") {
    return "invalid_reason";
  }

  if (field === "confidence") {
    return "invalid_confidence";
  }

  if (field === "duplicate_of_goal_id") {
    return "invalid_duplicate_of_goal_id";
  }

  if (field === "initial_step") {
    return "invalid_initial_step";
  }

  if (field === "classification") {
    return "invalid_classification";
  }

  return "invalid_promotion";
}

function skippedReasonFromError(error: z.ZodError): GoalPromotionSkippedReason {
  const duplicateIssue = error.issues.find((issue) => issue.path[0] === "duplicate_of_goal_id");

  return skippedReasonFromIssue(duplicateIssue ?? error.issues[0] ?? { path: [] });
}

function parsePromotions(envelope: GoalPromotionEnvelopeInput): {
  promotions: ParsedGoalPromotion[];
  skippedPromotions: GoalPromotionSkippedPromotion[];
} {
  const promotions: ParsedGoalPromotion[] = [];
  const skippedPromotions: GoalPromotionSkippedPromotion[] = [];

  for (const [candidateIndex, rawPromotion] of envelope.promotions.entries()) {
    const parsed = goalPromotionSchema.safeParse(rawPromotion);

    if (!parsed.success) {
      skippedPromotions.push({
        candidate_index: candidateIndex,
        reason: skippedReasonFromError(parsed.error),
      });
      continue;
    }

    promotions.push(parsed.data);
  }

  return {
    promotions,
    skippedPromotions,
  };
}

function parseResponse(result: LLMCompleteResult): GoalPromotionParseResult {
  const call = result.tool_calls.find((toolCall) => toolCall.name === GOAL_PROMOTION_TOOL_NAME);

  if (call === undefined) {
    throw new MissingGoalPromotionToolCallError(
      `Goal promotion extractor did not emit ${GOAL_PROMOTION_TOOL_NAME}`,
    );
  }

  const parsed = goalPromotionEnvelopeSchema.safeParse(call.input);

  if (!parsed.success) {
    throw parsed.error;
  }

  const { promotions, skippedPromotions } = parsePromotions(parsed.data);

  return {
    candidates: toCandidates(promotions),
    validPromotionCount: promotions.length,
    skippedPromotions,
  };
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
        speaker_entity_id: input.speakerEntityId ?? null,
        speaker_display_name: input.speakerDisplayName ?? null,
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
      usage: buildUsageTraceBlock(options.response),
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

function degradedReasonForParseError(error: unknown): GoalPromotionExtractorDegradedReason {
  if (error instanceof MissingGoalPromotionToolCallError) {
    return "missing_tool_call";
  }

  if (error instanceof z.ZodError) {
    return "invalid_payload";
  }

  return "llm_failed";
}

function traceExtractorCompleted(options: {
  tracer?: TurnTracer;
  turnId?: string;
  parseResult?: GoalPromotionParseResult;
  degraded: boolean;
  fatalReason?: GoalPromotionExtractorDegradedReason;
}): void {
  if (options.tracer?.enabled !== true || options.turnId === undefined) {
    return;
  }

  const skippedPromotions = options.parseResult?.skippedPromotions ?? [];
  const skippedPromotionCount = skippedPromotions.length;
  const validPromotionCount = options.parseResult?.validPromotionCount ?? 0;

  options.tracer.emit("goal_promotion_extractor_completed", {
    turnId: options.turnId,
    candidates_emitted: options.parseResult?.candidates.length ?? 0,
    valid_promotion_count: validPromotionCount,
    skipped_promotion_count: skippedPromotionCount,
    salvaged_promotion_count: skippedPromotionCount > 0 ? validPromotionCount : 0,
    skipped_promotions: skippedPromotions.map((promotion) => ({ ...promotion })),
    degraded: options.degraded,
    ...(options.fatalReason === undefined ? {} : { fatal_reason: options.fatalReason }),
  });
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

  private async complete(input: {
    messages: readonly LLMMessage[];
    tools: readonly LLMToolDefinition[];
  }): Promise<LLMCompleteResult> {
    traceLlmCallStarted({
      tracer: this.options.tracer,
      turnId: this.options.turnId,
      model: this.options.model as string,
      messages: input.messages,
      tools: input.tools,
    });

    try {
      const response = await (this.options.llmClient as LLMClient).complete({
        model: this.options.model as string,
        system: GOAL_PROMOTION_SYSTEM_PROMPT,
        messages: input.messages,
        tools: input.tools,
        tool_choice: { type: "tool", name: GOAL_PROMOTION_TOOL_NAME },
        max_tokens: GOAL_PROMOTION_MAX_TOKENS,
        budget: "goal-promotion-extractor",
      });

      traceLlmCallResponse({
        tracer: this.options.tracer,
        turnId: this.options.turnId,
        response,
      });

      return response;
    } catch (error) {
      traceLlmCallError({
        tracer: this.options.tracer,
        turnId: this.options.turnId,
        error,
      });

      throw error;
    }
  }

  async extract(input: ExtractGoalPromotionInput): Promise<GoalPromotionCandidate[]> {
    if (this.options.llmClient === undefined || this.options.model === undefined) {
      return this.degraded("llm_unavailable");
    }

    const messages = buildGoalPromotionMessages(input);
    const tools = [GOAL_PROMOTION_TOOL];

    let response: LLMCompleteResult;

    try {
      response = await this.complete({
        messages,
        tools,
      });
    } catch (error) {
      return this.degraded("llm_failed", error);
    }

    try {
      const parseResult = parseResponse(response);

      traceExtractorCompleted({
        tracer: this.options.tracer,
        turnId: this.options.turnId,
        parseResult,
        degraded: false,
      });

      return parseResult.candidates;
    } catch (error) {
      const reason = degradedReasonForParseError(error);

      traceExtractorCompleted({
        tracer: this.options.tracer,
        turnId: this.options.turnId,
        degraded: true,
        fatalReason: reason,
      });

      return this.degraded(reason, error);
    }
  }
}

export { GOAL_PROMOTION_TOOL_NAME };
