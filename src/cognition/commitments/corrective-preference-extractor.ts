import { z } from "zod";

import {
  type LLMClient,
  type LLMCompleteResult,
  type LLMMessage,
  type LLMToolDefinition,
  toToolInputSchema,
} from "../../llm/index.js";
import {
  closurePressureRelevanceSchema,
  commitmentIdSchema,
  commitmentTypeSchema,
  normalizeDirectiveFamily,
} from "../../memory/commitments/index.js";
import type { JsonValue } from "../../util/json-value.js";
import {
  entityIdHelpers,
  streamEntryIdHelpers,
  type CommitmentId,
  type EntityId,
  type StreamEntryId,
} from "../../util/ids.js";
import type { RecencyMessage } from "../recency/index.js";
import type { TurnTracer } from "../tracing/tracer.js";

const CONFIDENCE_THRESHOLD = 0.8;
const CORRECTIVE_PREFERENCE_TOOL_NAME = "EmitCorrectivePreference";

const correctivePreferenceEntityIdSchema = z
  .string()
  .refine((value) => entityIdHelpers.is(value), {
    message: "Invalid corrective preference entity id",
  })
  .transform((value) => value as EntityId);

const correctivePreferenceStreamEntryIdSchema = z
  .string()
  .refine((value) => streamEntryIdHelpers.is(value), {
    message: "Invalid corrective preference stream entry id",
  })
  .transform((value) => value as StreamEntryId);

const slotNegationSchema = z
  .object({
    subject_entity_id: correctivePreferenceEntityIdSchema,
    slot_key: z.string().min(1),
    rejected_value: z.string().min(1).nullable(),
    source_stream_entry_ids: z.array(correctivePreferenceStreamEntryIdSchema).min(1),
    confidence: z.number().min(0).max(1),
  })
  .strict();

const correctivePreferenceSchema = z
  .object({
    classification: z
      .enum(["corrective_preference", "none"])
      .describe(
        "Use corrective_preference only when the user is asking Borg to change durable future response behavior; use none for ordinary conversation, venting, task requests, or one-turn remarks.",
      ),
    type: commitmentTypeSchema
      .exclude(["promise"])
      .nullable()
      .describe(
        "Classify the durable response-behavior change as preference, rule, or boundary. Use null when classification is none.",
      ),
    directive: z
      .string()
      .nullable()
      .describe(
        "A concise first-person operational directive Borg can enforce when drafting or revising responses. Use null when classification is none.",
      ),
    directive_family: z
      .string()
      .min(1)
      .max(64)
      .nullable()
      .describe(
        "Short canonical snake_case slug for the directive family, such as no_terminal_valediction, no_signoff, or respond_substantively. Use null when classification is none.",
      ),
    closure_pressure_relevance: closurePressureRelevanceSchema
      .nullable()
      .describe(
        "Set no_closure when the durable correction asks Borg not to add endings, signoffs, wrap-ups, terminal valedictions, or closure pressure; set closure_seeking when it asks Borg to provide those; otherwise set neutral. Use null when classification is none.",
      ),
    priority: z
      .number()
      .int()
      .nullable()
      .describe(
        "Relative enforcement priority. Use higher values for explicit prohibitions or boundaries, lower values for softer style preferences. Use null when classification is none.",
      ),
    reason: z
      .string()
      .min(1)
      .describe("Brief semantic reason for the classification, grounded in the current user turn."),
    confidence: z
      .number()
      .min(0)
      .max(1)
      .describe("Confidence that the current user turn is making a durable correction."),
    supersedes_commitment_id: commitmentIdSchema
      .nullable()
      .optional()
      .describe(
        "Existing commitment id this correction replaces or tightens, if one was clearly selected from the supplied active commitments.",
      ),
    slot_negations: z
      .array(slotNegationSchema)
      .default([])
      .describe(
        "Relational slot values the current user turn rejects. Emit only when the user rejects a supplied relational slot, and cite the current user stream entry id.",
      ),
  })
  .strict();

const CORRECTIVE_PREFERENCE_TOOL = {
  name: CORRECTIVE_PREFERENCE_TOOL_NAME,
  description:
    "Classify whether the current user turn creates a durable correction to Borg's future response behavior.",
  inputSchema: toToolInputSchema(correctivePreferenceSchema),
} satisfies LLMToolDefinition;

const CORRECTIVE_PREFERENCE_SYSTEM_PROMPT = [
  "Classify whether the user is making a durable correction to Borg's future response behavior.",
  "Return corrective_preference only when the user is directing Borg to change how it should answer in future turns, such as a recurring style, boundary, interaction rule, or response pattern.",
  "Return none for ordinary task requests, emotional disclosure, venting, disagreement, one-turn instructions, or discussion about a behavior without asking Borg to adopt a lasting change.",
  "Separately, fill slot_negations when the user rejects a supplied relational slot value, even if classification is none.",
  "For slot_negations, select subject_entity_id and slot_key only from supplied relational_slots and cite only the current_user_stream_entry_id.",
  "Judge semantic intent across languages. Do not rely on wording, punctuation, capitalization, or phrase shapes.",
  "Emit directive_family as a short snake_case semantic family slug chosen by meaning, not by surface wording.",
  'Emit closure_pressure_relevance as "no_closure" for durable no-wrap-up/no-signoff/no-closure corrections, "closure_seeking" for durable requests to add closure, and "neutral" otherwise.',
  "When uncertain, return none. The directive must be enforceable by a later response checker without needing to remember the current phrasing.",
].join("\n");

type CorrectivePreferenceToolInput = z.infer<typeof correctivePreferenceSchema>;

class MissingCorrectivePreferenceToolCallError extends Error {}

export type CorrectivePreferenceCandidate = {
  type: Exclude<z.infer<typeof commitmentTypeSchema>, "promise">;
  directive: string;
  directive_family: string;
  closure_pressure_relevance: z.infer<typeof closurePressureRelevanceSchema>;
  priority: number;
  reason: string;
  confidence: number;
  supersedes_commitment_id?: CommitmentId | null;
};

export type CorrectivePreferenceSlotNegation = {
  subject_entity_id: EntityId;
  slot_key: string;
  rejected_value: string | null;
  source_stream_entry_ids: StreamEntryId[];
  confidence: number;
};

export type CorrectivePreferenceExtractionResult = {
  preference: CorrectivePreferenceCandidate | null;
  slot_negations: CorrectivePreferenceSlotNegation[];
};

export type CorrectivePreferenceExtractorDegradedReason =
  | "llm_unavailable"
  | "llm_failed"
  | "missing_tool_call"
  | "invalid_payload";

export type CorrectivePreferenceExtractorOptions = {
  llmClient?: LLMClient;
  model?: string;
  tracer?: TurnTracer;
  turnId?: string;
  onDegraded?: (
    reason: CorrectivePreferenceExtractorDegradedReason,
    error?: unknown,
  ) => Promise<void> | void;
};

export type ExtractCorrectivePreferenceInput = {
  userMessage: string;
  currentUserStreamEntryId?: StreamEntryId | null;
  recentHistory: readonly RecencyMessage[];
  audienceEntityId: EntityId | null;
  activeCommitments: readonly {
    id: CommitmentId;
    type: string;
    directive: string;
    directive_family?: string | null;
    closure_pressure_relevance?: z.infer<typeof closurePressureRelevanceSchema> | null;
    priority: number;
  }[];
  relationalSlots?: readonly {
    subject_entity_id: EntityId;
    slot_key: string;
    value: string;
    state: string;
    alternate_values: readonly { value: string }[];
  }[];
};

function toCandidate(input: CorrectivePreferenceToolInput): CorrectivePreferenceCandidate | null {
  if (input.classification !== "corrective_preference" || input.confidence < CONFIDENCE_THRESHOLD) {
    return null;
  }

  if (
    input.type === null ||
    input.directive === null ||
    input.directive_family === null ||
    input.closure_pressure_relevance === null ||
    input.priority === null
  ) {
    return null;
  }

  const directive = input.directive.trim();
  const directiveFamily = normalizeDirectiveFamily(input.directive_family);
  const reason = input.reason.trim();

  if (directive.length === 0 || directiveFamily.length === 0 || reason.length === 0) {
    return null;
  }

  return {
    type: input.type,
    directive,
    directive_family: directiveFamily,
    closure_pressure_relevance: input.closure_pressure_relevance,
    priority: input.priority,
    reason,
    confidence: input.confidence,
    supersedes_commitment_id: input.supersedes_commitment_id ?? null,
  };
}

function slotNegationsFromInput(
  input: CorrectivePreferenceToolInput,
): CorrectivePreferenceSlotNegation[] {
  const slotNegations: CorrectivePreferenceSlotNegation[] = [];

  for (const negation of input.slot_negations) {
    if (negation.confidence < CONFIDENCE_THRESHOLD) {
      continue;
    }

    slotNegations.push({
      subject_entity_id: negation.subject_entity_id,
      slot_key: negation.slot_key.trim(),
      rejected_value: negation.rejected_value === null ? null : negation.rejected_value.trim(),
      source_stream_entry_ids: [...negation.source_stream_entry_ids],
      confidence: negation.confidence,
    });
  }

  return slotNegations;
}

function toExtractionResult(
  input: CorrectivePreferenceToolInput,
): CorrectivePreferenceExtractionResult {
  return {
    preference: toCandidate(input),
    slot_negations: slotNegationsFromInput(input),
  };
}

function parseResponse(result: LLMCompleteResult): CorrectivePreferenceExtractionResult {
  const call = result.tool_calls.find(
    (toolCall) => toolCall.name === CORRECTIVE_PREFERENCE_TOOL_NAME,
  );

  if (call === undefined) {
    throw new MissingCorrectivePreferenceToolCallError(
      `Corrective preference extractor did not emit ${CORRECTIVE_PREFERENCE_TOOL_NAME}`,
    );
  }

  const parsed = correctivePreferenceSchema.safeParse(call.input);

  if (!parsed.success) {
    throw parsed.error;
  }

  return toExtractionResult(parsed.data);
}

function buildCorrectivePreferenceMessages(input: ExtractCorrectivePreferenceInput): LLMMessage[] {
  return [
    {
      role: "user",
      content: JSON.stringify({
        current_user_message: input.userMessage,
        current_user_stream_entry_id: input.currentUserStreamEntryId ?? null,
        recent_history: input.recentHistory.slice(-8).map((message) => ({
          role: message.role,
          content: message.content,
        })),
        audience_entity_id: input.audienceEntityId,
        active_commitments: input.activeCommitments.map((commitment) => ({
          id: commitment.id,
          type: commitment.type,
          directive_family: commitment.directive_family ?? null,
          closure_pressure_relevance: commitment.closure_pressure_relevance ?? null,
          directive: commitment.directive,
          priority: commitment.priority,
        })),
        relational_slots: (input.relationalSlots ?? []).map((slot) => ({
          subject_entity_id: slot.subject_entity_id,
          slot_key: slot.slot_key,
          value: slot.value,
          state: slot.state,
          alternate_values: slot.alternate_values.map((alternate) => ({
            value: alternate.value,
          })),
        })),
      }),
    },
  ];
}

function summarizeCorrectivePreferenceResponseShape(response: LLMCompleteResult): JsonValue {
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
      label: "corrective_preference_extractor",
      model: options.model,
      promptCharCount: countCompletePromptChars(
        CORRECTIVE_PREFERENCE_SYSTEM_PROMPT,
        options.messages,
      ),
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
      label: "corrective_preference_extractor",
      responseShape: summarizeCorrectivePreferenceResponseShape(options.response),
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
      label: "corrective_preference_extractor",
      responseShape: {
        error: options.error instanceof Error ? options.error.message : String(options.error),
      },
      stopReason: null,
      usage: null,
    });
  }
}

export class CorrectivePreferenceExtractor {
  constructor(private readonly options: CorrectivePreferenceExtractorOptions = {}) {}

  private async degraded(
    reason: CorrectivePreferenceExtractorDegradedReason,
    error?: unknown,
  ): Promise<null> {
    try {
      await this.options.onDegraded?.(reason, error);
    } catch {
      // Best-effort degraded-mode logging only.
    }

    return null;
  }

  async extractWithSlotNegations(
    input: ExtractCorrectivePreferenceInput,
  ): Promise<CorrectivePreferenceExtractionResult> {
    if (this.options.llmClient === undefined || this.options.model === undefined) {
      return (
        (await this.degraded("llm_unavailable")) ?? {
          preference: null,
          slot_negations: [],
        }
      );
    }

    const messages = buildCorrectivePreferenceMessages(input);
    const tools = [CORRECTIVE_PREFERENCE_TOOL];

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
        system: CORRECTIVE_PREFERENCE_SYSTEM_PROMPT,
        messages,
        tools,
        tool_choice: { type: "tool", name: CORRECTIVE_PREFERENCE_TOOL_NAME },
        max_tokens: 512,
        budget: "corrective-preference-extractor",
      });
    } catch (error) {
      traceLlmCallError({
        tracer: this.options.tracer,
        turnId: this.options.turnId,
        error,
      });

      return (
        (await this.degraded("llm_failed", error)) ?? {
          preference: null,
          slot_negations: [],
        }
      );
    }

    traceLlmCallResponse({
      tracer: this.options.tracer,
      turnId: this.options.turnId,
      response,
    });

    try {
      return parseResponse(response);
    } catch (error) {
      await this.degraded(
        error instanceof MissingCorrectivePreferenceToolCallError
          ? "missing_tool_call"
          : error instanceof z.ZodError
            ? "invalid_payload"
            : "llm_failed",
        error,
      );
      return {
        preference: null,
        slot_negations: [],
      };
    }
  }

  async extract(
    input: ExtractCorrectivePreferenceInput,
  ): Promise<CorrectivePreferenceCandidate | null> {
    const result = await this.extractWithSlotNegations(input);

    return result.preference;
  }
}

export { CORRECTIVE_PREFERENCE_TOOL_NAME };
