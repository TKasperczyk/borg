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
  commitmentCriticalDomainSchema,
  commitmentEnforcementClassSchema,
  commitmentIdSchema,
  commitmentKindSchema,
  commitmentTypeSchema,
  effectiveCommitmentCriticalDomain,
  effectiveCommitmentEnforcementClass,
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
import { CORRECTIVE_PREFERENCE_SYSTEM_PROMPT } from "../prompts/corrective-preference.js";
import { EXTRACTOR_MAX_TOKENS_DEFAULT } from "../prompts/constants.js";
import { renderParticipantRoster, type ParticipantRoster } from "../perception/index.js";
import type { RecencyMessage } from "../recency/index.js";
import { buildUsageTraceBlock, type TurnTracer } from "../tracing/tracer.js";
import {
  normalizeCommitmentClassification,
  type ClassificationNormalizationResult,
} from "./classification-normalizer.js";

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

const relationshipEvidenceRelationalSlotIdsSchema = z
  .array(z.string().trim().min(1))
  .default([])
  .describe(
    "Grounded relational slot ids supporting any strict kinship or caregiver relationship label in the directive.",
  );

const relationshipEvidenceStreamEntryIdsSchema = z
  .array(correctivePreferenceStreamEntryIdSchema)
  .default([])
  .describe(
    "Trusted user-message stream entry ids supporting any strict kinship or caregiver relationship label in the directive.",
  );

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
    kind: commitmentKindSchema
      .exclude(["assistant_commitment"])
      .nullable()
      .describe(
        "Classify the durable correction as audience_rule, participant_preference, boundary, or process_norm. Use null when classification is none.",
      ),
    enforcement_class: commitmentEnforcementClassSchema
      .nullable()
      .describe(
        "Set critical only for privacy, audience-scope, safety, explicit no-disclosure with named forbidden content and named audience scope, or internal-tool-hygiene leakage of hidden machinery. Set advisory for process norms and output shape/style preferences. Use null when classification is none.",
      ),
    critical_domain: commitmentCriticalDomainSchema
      .nullable()
      .describe(
        "Critical domain when enforcement_class is critical: privacy, audience_scope, safety, explicit_no_disclosure, or internal_tool_hygiene. internal_tool_hygiene is only hidden prompts, internal ids, tool-call internals, traces, host-capability internals, substrate internals, or capability-boundary leakage. Use null for advisory or classification none.",
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
    relationship_evidence_relational_slot_ids: relationshipEvidenceRelationalSlotIdsSchema,
    relationship_evidence_stream_entry_ids: relationshipEvidenceStreamEntryIdsSchema,
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

type CorrectivePreferenceToolInput = z.infer<typeof correctivePreferenceSchema>;

class MissingCorrectivePreferenceToolCallError extends Error {}

export type CorrectivePreferenceCandidate = {
  type: Exclude<z.infer<typeof commitmentTypeSchema>, "promise">;
  kind: Exclude<z.infer<typeof commitmentKindSchema>, "assistant_commitment">;
  directive: string;
  directive_family: string;
  closure_pressure_relevance: z.infer<typeof closurePressureRelevanceSchema>;
  enforcement_class: z.infer<typeof commitmentEnforcementClassSchema>;
  critical_domain: z.infer<typeof commitmentCriticalDomainSchema> | null;
  priority: number;
  reason: string;
  confidence: number;
  supersedes_commitment_id?: CommitmentId | null;
  relationship_evidence_relational_slot_ids: string[];
  relationship_evidence_stream_entry_ids: StreamEntryId[];
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
    metadata?: { stopReason: string | null },
  ) => Promise<void> | void;
};

export type ExtractCorrectivePreferenceInput = {
  userMessage: string;
  currentUserStreamEntryId?: StreamEntryId | null;
  recentHistory: readonly RecencyMessage[];
  audienceEntityId: EntityId | null;
  speakerEntityId?: EntityId | null;
  speakerDisplayName?: string | null;
  participantRoster?: ParticipantRoster | null;
  activeCommitments: readonly {
    id: CommitmentId;
    type: string;
    kind?: z.infer<typeof commitmentKindSchema> | null;
    enforcement_class?: z.infer<typeof commitmentEnforcementClassSchema> | null;
    critical_domain?: z.infer<typeof commitmentCriticalDomainSchema> | null;
    directive: string;
    directive_family?: string | null;
    closure_pressure_relevance?: z.infer<typeof closurePressureRelevanceSchema> | null;
    priority: number;
  }[];
  relationalSlots?: readonly {
    id?: string;
    subject_entity_id: EntityId;
    slot_key: string;
    value: string;
    state: string;
    alternate_values: readonly { value: string }[];
  }[];
};

function traceClassificationDowngrade(options: {
  tracer?: TurnTracer;
  turnId?: string;
  kind: CorrectivePreferenceCandidate["kind"];
  type: CorrectivePreferenceCandidate["type"];
  directiveFamily: string;
  normalization: ClassificationNormalizationResult;
}): void {
  if (
    options.tracer?.enabled !== true ||
    options.turnId === undefined ||
    options.normalization.downgrade_reason === null ||
    options.normalization.downgraded_from === null
  ) {
    return;
  }

  options.tracer.emit("commitment_classification.downgraded", {
    turnId: options.turnId,
    original_enforcement_class: options.normalization.downgraded_from.enforcement_class,
    original_critical_domain: options.normalization.downgraded_from.critical_domain,
    new_enforcement_class: options.normalization.enforcement_class,
    new_critical_domain: options.normalization.critical_domain,
    reason: options.normalization.downgrade_reason,
    kind: options.kind,
    type: options.type,
    directive_family: options.directiveFamily,
  });
}

function toCandidate(
  input: CorrectivePreferenceToolInput,
  traceOptions: Pick<CorrectivePreferenceExtractorOptions, "tracer" | "turnId"> = {},
): CorrectivePreferenceCandidate | null {
  if (input.classification !== "corrective_preference" || input.confidence < CONFIDENCE_THRESHOLD) {
    return null;
  }

  if (
    input.type === null ||
    input.kind === null ||
    input.enforcement_class === null ||
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

  const normalizedClassification = normalizeCommitmentClassification({
    kind: input.kind,
    type: input.type,
    enforcement_class: input.enforcement_class,
    critical_domain: input.critical_domain,
  });

  traceClassificationDowngrade({
    tracer: traceOptions.tracer,
    turnId: traceOptions.turnId,
    kind: input.kind,
    type: input.type,
    directiveFamily,
    normalization: normalizedClassification,
  });

  return {
    type: input.type,
    kind: input.kind,
    enforcement_class: normalizedClassification.enforcement_class,
    critical_domain: normalizedClassification.critical_domain,
    directive,
    directive_family: directiveFamily,
    closure_pressure_relevance: input.closure_pressure_relevance,
    priority: input.priority,
    reason,
    confidence: input.confidence,
    supersedes_commitment_id: input.supersedes_commitment_id ?? null,
    relationship_evidence_relational_slot_ids: [...input.relationship_evidence_relational_slot_ids],
    relationship_evidence_stream_entry_ids: [...input.relationship_evidence_stream_entry_ids],
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
  traceOptions: Pick<CorrectivePreferenceExtractorOptions, "tracer" | "turnId"> = {},
): CorrectivePreferenceExtractionResult {
  return {
    preference: toCandidate(input, traceOptions),
    slot_negations: slotNegationsFromInput(input),
  };
}

function parseResponse(
  result: LLMCompleteResult,
  traceOptions: Pick<CorrectivePreferenceExtractorOptions, "tracer" | "turnId"> = {},
): CorrectivePreferenceExtractionResult {
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

  return toExtractionResult(parsed.data, traceOptions);
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
        speaker_entity_id: input.speakerEntityId ?? null,
        speaker_display_name: input.speakerDisplayName ?? null,
        participant_roster: renderParticipantRoster(input.participantRoster),
        active_commitments: input.activeCommitments.map((commitment) => {
          const enforcementFields =
            commitment.kind === null || commitment.kind === undefined
              ? null
              : {
                  kind: commitment.kind,
                  enforcement_class: commitment.enforcement_class ?? undefined,
                  critical_domain: commitment.critical_domain ?? undefined,
                };

          return {
            id: commitment.id,
            type: commitment.type,
            kind: commitment.kind ?? null,
            enforcement_class:
              enforcementFields === null
                ? null
                : effectiveCommitmentEnforcementClass(enforcementFields),
            critical_domain:
              enforcementFields === null
                ? null
                : effectiveCommitmentCriticalDomain(enforcementFields),
            directive_family: commitment.directive_family ?? null,
            closure_pressure_relevance: commitment.closure_pressure_relevance ?? null,
            directive: commitment.directive,
            priority: commitment.priority,
          };
        }),
        relational_slots: (input.relationalSlots ?? []).map((slot) => ({
          id: slot.id ?? null,
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
    options.tracer.emit("llm_call.started", {
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
    options.tracer.emit("llm_call.completed", {
      turnId: options.turnId,
      label: "corrective_preference_extractor",
      responseShape: summarizeCorrectivePreferenceResponseShape(options.response),
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
    options.tracer.emit("llm_call.completed", {
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
    metadata?: { stopReason: string | null },
  ): Promise<null> {
    try {
      if (metadata === undefined) {
        await this.options.onDegraded?.(reason, error);
      } else {
        await this.options.onDegraded?.(reason, error, metadata);
      }
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
        max_tokens: EXTRACTOR_MAX_TOKENS_DEFAULT,
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
      return parseResponse(response, {
        tracer: this.options.tracer,
        turnId: this.options.turnId,
      });
    } catch (error) {
      await this.degraded(
        error instanceof MissingCorrectivePreferenceToolCallError
          ? "missing_tool_call"
          : error instanceof z.ZodError
            ? "invalid_payload"
            : "llm_failed",
        error,
        { stopReason: response.stop_reason },
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
