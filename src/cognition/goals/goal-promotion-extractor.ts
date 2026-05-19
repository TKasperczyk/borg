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
import { BORG_HOST_CAPABILITY_BOUNDARY_PROMPT } from "../host-capabilities.js";
import type { RecencyMessage } from "../recency/index.js";
import { buildUsageTraceBlock, type TurnTracer } from "../tracing/tracer.js";

const CONFIDENCE_THRESHOLD = 0.85;
const MAX_PROMOTIONS_PER_TURN = 3;
const GOAL_PROMOTION_MAX_TOKENS = 1536;
const GOAL_PROMOTION_TOOL_NAME = "EmitGoalPromotion";

export const GOAL_PROMOTION_CLASSIFICATIONS = [
  "durable_borg_goal",
  "one_off",
  "not_borg_responsibility",
  "impossible_for_borg_without_capability",
  "already_represented",
  "none",
] as const;

export const goalPromotionClassificationSchema = z.enum(GOAL_PROMOTION_CLASSIFICATIONS);
export type GoalPromotionClassification = z.infer<typeof goalPromotionClassificationSchema>;

const GOAL_PROMOTION_CLASSIFICATION_COUNT_KEYS = [
  ...GOAL_PROMOTION_CLASSIFICATIONS,
  "invalid_classification",
] as const;

type GoalPromotionClassificationCountKey =
  (typeof GOAL_PROMOTION_CLASSIFICATION_COUNT_KEYS)[number];

const goalPromotionBatchSchema = z.enum(["single", "explicit_multiple"]).default("single");
type GoalPromotionBatch = z.infer<typeof goalPromotionBatchSchema>;

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
    classification: goalPromotionClassificationSchema.describe(
      "Classify the candidate by memory kind. Only durable_borg_goal can become a GoalRecord.",
    ),
    description: z
      .string()
      .trim()
      .min(1)
      .describe("Concise description of the candidate memory item."),
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
      .describe("Semantic reason for the classification, grounded in the current user turn."),
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
    durable_goal_batch: goalPromotionBatchSchema.describe(
      "Use single unless the current turn explicitly asks Borg to track multiple separate ongoing responsibilities.",
    ),
    promotions: z
      .array(goalPromotionSchema)
      .describe(
        "Goal-promotion taxonomy candidates. Emit an empty array when nothing is relevant.",
      ),
  })
  .strict();

const goalPromotionEnvelopeSchema = z
  .object({
    durable_goal_batch: goalPromotionBatchSchema,
    promotions: z.array(z.unknown()),
  })
  .strict();

const GOAL_PROMOTION_TOOL = {
  name: GOAL_PROMOTION_TOOL_NAME,
  description:
    "Classify goal-like candidates by memory kind and emit durable Borg goals only when warranted.",
  inputSchema: toToolInputSchema(goalPromotionOutputSchema),
} satisfies LLMToolDefinition;

const GOAL_PROMOTION_SYSTEM_PROMPT = [
  "Classify goal-like candidates from the current user turn by memory kind.",
  "Only durable_borg_goal becomes active goal self-memory; other labels are rejected from goal storage but explain why.",
  "Set durable_goal_batch=explicit_multiple only when the current turn explicitly asks Borg to track multiple separate ongoing responsibilities; otherwise use single.",
  "Judge semantic intent across languages. Do not rely on wording, punctuation, capitalization, or phrase shapes.",
  "When speaker_entity_id is supplied and the current speaker creates a durable first-person goal, treat that speaker as the goal owner. In group chat, first-person user goals belong to the current sender, not the group, unless the message explicitly says the group is acting.",
  "If a supplied active goal already covers the request, classify as already_represented and set duplicate_of_goal_id.",
  "Use target_at only for a real goal deadline. Use the supplied temporal cue as context, not as an automatic trigger.",
  "Borg-owned durable goals must stay inside the host capability boundary below.",
  BORG_HOST_CAPABILITY_BOUNDARY_PROMPT,
  "",
  "Classifications:",
  "- durable_borg_goal: Borg tracks conversation-state and memory-state across turns, maintains remembered decision-log context, or revisits an unresolved conversation responsibility. Tracking or monitoring means memory/conversation-state tracking unless the host explicitly provides external monitoring capability.",
  "- one_off: finite event, task, reply, or fact; no ongoing tracking.",
  '- not_borg_responsibility: responsibility belongs to a participant or external system, such as "user will deploy", "friend will respond", or "Ben will pull flight numbers".',
  "- impossible_for_borg_without_capability: the candidate asks Borg to do work Borg cannot do without host capabilities, such as monitoring p95, sending something later, scheduled document edits, physical attendance, payments, proactive notifications, external tool execution, or production/dashboard monitoring.",
  "- already_represented: a supplied active goal already covers the candidate.",
  "- none: not memory-worthy at all.",
  "",
  "Prefer one durable_borg_goal at most. When uncertain, emit no promotions. Return only the required tool call.",
  "",
  "Examples:",
  "- Coding: track refactor decisions across sessions -> durable_borg_goal; read this file -> one_off; user will deploy -> not_borg_responsibility; Borg will monitor p95 -> impossible_for_borg_without_capability.",
  "- Relationships: support through job search -> durable_borg_goal; send one message -> one_off; friend will respond -> not_borg_responsibility.",
  "- Planning: track the plan across sessions -> durable_borg_goal; execute tonight's agenda -> one_off; Ben will pull flight numbers -> not_borg_responsibility; Borg will attend in person or make payments -> impossible_for_borg_without_capability.",
  "- Monitor deployment cleanup and job-search support -> explicit_multiple only if both are separate ongoing Borg memory/conversation responsibilities, not external monitoring.",
].join("\n");

type ParsedGoalPromotion = z.infer<typeof goalPromotionSchema>;
type GoalPromotionEnvelopeInput = z.infer<typeof goalPromotionEnvelopeSchema>;
type ParsedGoalPromotionWithIndex = {
  candidateIndex: number;
  promotion: ParsedGoalPromotion;
};

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
  duplicate_of_goal_id: GoalRecord["id"] | null;
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
  activeGoals: readonly Pick<
    GoalRecord,
    "id" | "description" | "priority" | "target_at" | "owner_entity_id"
  >[];
};

type GoalPromotionParseResult = {
  candidates: GoalPromotionCandidate[];
  validPromotionCount: number;
  skippedPromotions: GoalPromotionSkippedPromotion[];
  classificationCounts: Record<GoalPromotionClassificationCountKey, number>;
  rejectedPromotions: GoalPromotionRejectedPromotion[];
  rejectedLowConfidenceCount: number;
  rejectedByCapCount: number;
};

type GoalPromotionRejectedReason = "non_durable_classification" | "low_confidence" | "cap_exceeded";

type GoalPromotionRejectedPromotion = {
  candidate_index: number;
  classification: GoalPromotionClassification;
  description_excerpt: string;
  reason: GoalPromotionRejectedReason;
};

type GoalPromotionCandidateWithMeta = {
  candidate: GoalPromotionCandidate;
  candidateIndex: number;
};

function zeroClassificationCounts(): Record<GoalPromotionClassificationCountKey, number> {
  return {
    durable_borg_goal: 0,
    one_off: 0,
    not_borg_responsibility: 0,
    impossible_for_borg_without_capability: 0,
    already_represented: 0,
    none: 0,
    invalid_classification: 0,
  };
}

function incrementClassificationCount(
  counts: Record<GoalPromotionClassificationCountKey, number>,
  key: GoalPromotionClassificationCountKey,
): void {
  counts[key] += 1;
}

function descriptionExcerpt(description: string): string {
  return description.trim().slice(0, 60);
}

function rejectedPromotion(input: {
  candidateIndex: number;
  classification: GoalPromotionClassification;
  description: string;
  reason: GoalPromotionRejectedReason;
}): GoalPromotionRejectedPromotion {
  return {
    candidate_index: input.candidateIndex,
    classification: input.classification,
    description_excerpt: descriptionExcerpt(input.description),
    reason: input.reason,
  };
}

function candidateFromPromotion(promotion: ParsedGoalPromotion): GoalPromotionCandidate {
  return {
    description: promotion.description.trim(),
    priority: promotion.priority,
    target_at: promotion.target_at,
    reason: promotion.reason.trim(),
    confidence: promotion.confidence,
    duplicate_of_goal_id: promotion.duplicate_of_goal_id,
    initial_step:
      promotion.initial_step === null || promotion.initial_step === undefined
        ? null
        : {
            description: promotion.initial_step.description.trim(),
            kind: promotion.initial_step.kind,
            due_at: promotion.initial_step.due_at ?? null,
            rationale: promotion.initial_step.rationale.trim(),
          },
  };
}

function toCandidates(input: {
  promotions: readonly ParsedGoalPromotionWithIndex[];
  durableGoalBatch: GoalPromotionBatch;
}): {
  candidates: GoalPromotionCandidate[];
  rejectedPromotions: GoalPromotionRejectedPromotion[];
  rejectedLowConfidenceCount: number;
  rejectedByCapCount: number;
} {
  const durableCandidates: GoalPromotionCandidateWithMeta[] = [];
  const rejectedPromotions: GoalPromotionRejectedPromotion[] = [];
  let rejectedLowConfidenceCount = 0;

  for (const parsed of input.promotions) {
    const promotion = parsed.promotion;

    if (promotion.classification !== "durable_borg_goal") {
      rejectedPromotions.push(
        rejectedPromotion({
          candidateIndex: parsed.candidateIndex,
          classification: promotion.classification,
          description: promotion.description,
          reason: "non_durable_classification",
        }),
      );
      continue;
    }

    if (promotion.confidence < CONFIDENCE_THRESHOLD) {
      rejectedLowConfidenceCount += 1;
      rejectedPromotions.push(
        rejectedPromotion({
          candidateIndex: parsed.candidateIndex,
          classification: promotion.classification,
          description: promotion.description,
          reason: "low_confidence",
        }),
      );
      continue;
    }

    durableCandidates.push({
      candidate: candidateFromPromotion(promotion),
      candidateIndex: parsed.candidateIndex,
    });
  }

  const cap = input.durableGoalBatch === "explicit_multiple" ? MAX_PROMOTIONS_PER_TURN : 1;
  const rankedCandidates =
    input.durableGoalBatch === "explicit_multiple"
      ? durableCandidates
      : [...durableCandidates].sort((left, right) => {
          const confidenceDelta = right.candidate.confidence - left.candidate.confidence;

          return confidenceDelta === 0
            ? left.candidateIndex - right.candidateIndex
            : confidenceDelta;
        });
  const acceptedCandidates = rankedCandidates.slice(0, cap);
  const cappedCandidates = rankedCandidates.slice(cap);

  for (const capped of cappedCandidates) {
    rejectedPromotions.push(
      rejectedPromotion({
        candidateIndex: capped.candidateIndex,
        classification: "durable_borg_goal",
        description: capped.candidate.description,
        reason: "cap_exceeded",
      }),
    );
  }

  return {
    candidates: acceptedCandidates.map((candidate) => candidate.candidate),
    rejectedPromotions,
    rejectedLowConfidenceCount,
    rejectedByCapCount: cappedCandidates.length,
  };
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
  promotions: ParsedGoalPromotionWithIndex[];
  skippedPromotions: GoalPromotionSkippedPromotion[];
  classificationCounts: Record<GoalPromotionClassificationCountKey, number>;
} {
  const promotions: ParsedGoalPromotionWithIndex[] = [];
  const skippedPromotions: GoalPromotionSkippedPromotion[] = [];
  const classificationCounts = zeroClassificationCounts();

  for (const [candidateIndex, rawPromotion] of envelope.promotions.entries()) {
    const parsed = goalPromotionSchema.safeParse(rawPromotion);

    if (!parsed.success) {
      const reason = skippedReasonFromError(parsed.error);

      if (reason === "invalid_classification") {
        incrementClassificationCount(classificationCounts, "invalid_classification");
      }

      skippedPromotions.push({
        candidate_index: candidateIndex,
        reason,
      });
      continue;
    }

    incrementClassificationCount(classificationCounts, parsed.data.classification);
    promotions.push({
      candidateIndex,
      promotion: parsed.data,
    });
  }

  return {
    promotions,
    skippedPromotions,
    classificationCounts,
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

  const { promotions, skippedPromotions, classificationCounts } = parsePromotions(parsed.data);
  const candidates = toCandidates({
    promotions,
    durableGoalBatch: parsed.data.durable_goal_batch,
  });

  return {
    candidates: candidates.candidates,
    validPromotionCount: promotions.length,
    skippedPromotions,
    classificationCounts,
    rejectedPromotions: candidates.rejectedPromotions,
    rejectedLowConfidenceCount: candidates.rejectedLowConfidenceCount,
    rejectedByCapCount: candidates.rejectedByCapCount,
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
          owner_entity_id: goal.owner_entity_id ?? null,
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
  const classificationCounts =
    options.parseResult?.classificationCounts ?? zeroClassificationCounts();
  const rejectedPromotions = options.parseResult?.rejectedPromotions ?? [];
  const rejectedByClassification = zeroClassificationCounts();

  for (const rejection of rejectedPromotions) {
    if (rejection.reason === "non_durable_classification") {
      incrementClassificationCount(rejectedByClassification, rejection.classification);
    }

    options.tracer.emit("goal_promotion_classification_rejected", {
      turnId: options.turnId,
      classification: rejection.classification,
      description_excerpt: rejection.description_excerpt,
      reason: rejection.reason,
    });
  }

  options.tracer.emit("goal_promotion_extractor_completed", {
    turnId: options.turnId,
    candidates_emitted: options.parseResult?.candidates.length ?? 0,
    valid_promotion_count: validPromotionCount,
    skipped_promotion_count: skippedPromotionCount,
    salvaged_promotion_count: skippedPromotionCount > 0 ? validPromotionCount : 0,
    skipped_promotions: skippedPromotions.map((promotion) => ({ ...promotion })),
    classification_counts: classificationCounts,
    rejected_by_classification: rejectedByClassification,
    rejected_low_confidence: options.parseResult?.rejectedLowConfidenceCount ?? 0,
    rejected_by_cap: options.parseResult?.rejectedByCapCount ?? 0,
    rejected_invalid_enum: classificationCounts.invalid_classification,
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
