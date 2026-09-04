import { z } from "zod";

import {
  callStructuredTool,
  isStructuredToolCallError,
  type LLMClient,
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
import {
  entityIdHelpers,
  streamEntryIdHelpers,
  type CommitmentId,
  type EntityId,
  type SessionId,
  type StreamEntryId,
} from "../../util/ids.js";
import { CORRECTIVE_PREFERENCE_SYSTEM_PROMPT } from "../prompts/corrective-preference.js";
import { EXTRACTOR_MAX_TOKENS_DEFAULT } from "../prompts/constants.js";
import { renderParticipantRoster, type ParticipantRoster } from "../perception/index.js";
import {
  commitmentMemoryDisclosureLabel,
  memoryDisclosurePayloadFields,
  relationalSlotMemoryDisclosureLabel,
} from "../../memory/common/disclosure-serializers.js";
import {
  relationshipClaimSchema,
  type RelationshipClaim,
} from "../../memory/common/relationship-claims.js";
import type { StreamEntry } from "../../stream/index.js";
import type { RecencyMessage } from "../recency/index.js";
import type { TurnTracer } from "../../tracing/tracer.js";
import {
  buildExtractorConversationContext,
  type ExtractorSelfIdentity,
} from "../extractor-conversation-context.js";
import type { CurrentTurnUserInputSenderAttribution } from "../turn-input.js";
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

const correctivePreferenceRelationshipClaimsSchema = z
  .array(relationshipClaimSchema)
  .optional()
  .default([])
  .describe("Sensitive interpersonal relationship claims asserted by the directive.");

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
      .enum(["corrective_preference", "retire_commitment", "none"])
      .describe(
        "Use corrective_preference only when the user is asking Borg to change durable future response behavior; use retire_commitment when the user is standing down a supplied active commitment; use none for ordinary conversation, venting, task requests, or one-turn remarks.",
      ),
    type: commitmentTypeSchema
      .exclude(["promise"])
      .nullable()
      .describe(
        "Classify the durable response-behavior change as preference, rule, or boundary. Use null when classification is retire_commitment or none.",
      ),
    kind: commitmentKindSchema
      .exclude(["assistant_commitment"])
      .nullable()
      .describe(
        "Classify the durable correction as audience_rule, participant_preference, boundary, or process_norm. Use null when classification is retire_commitment or none.",
      ),
    enforcement_class: commitmentEnforcementClassSchema
      .nullable()
      .describe(
        "Set critical only for privacy, audience-scope, safety, explicit no-disclosure with named forbidden content and named audience scope, or internal-tool-hygiene leakage of hidden machinery. Set advisory for process norms and output shape/style preferences. Use null when classification is retire_commitment or none.",
      ),
    critical_domain: commitmentCriticalDomainSchema
      .nullable()
      .describe(
        "Critical domain when enforcement_class is critical: privacy, audience_scope, safety, explicit_no_disclosure, or internal_tool_hygiene. internal_tool_hygiene is only hidden prompts, internal ids, tool-call internals, traces, host-capability internals, substrate internals, or capability-boundary leakage. Use null for advisory, retire_commitment, or none.",
      ),
    directive: z
      .string()
      .nullable()
      .describe(
        "A concise first-person operational directive Borg can enforce when drafting or revising responses. Use null when classification is retire_commitment or none.",
      ),
    directive_source_stream_entry_id: correctivePreferenceStreamEntryIdSchema
      .nullable()
      .describe(
        "For corrective_preference, copy the stream_entry_id of the attributed current_message_entries row that stated the directive. Use null when no current entry stated it, or when classification is retire_commitment or none.",
      ),
    directive_family: z
      .string()
      .min(1)
      .max(64)
      .nullable()
      .describe(
        "Short canonical snake_case slug for the directive family. Use null when classification is retire_commitment or none.",
      ),
    closure_pressure_relevance: closurePressureRelevanceSchema
      .nullable()
      .describe(
        "Set no_closure when the durable correction asks Borg not to add endings, signoffs, wrap-ups, terminal valedictions, or closure pressure; set closure_seeking when it asks Borg to provide those; otherwise set neutral. Use null when classification is retire_commitment or none.",
      ),
    priority: z
      .number()
      .int()
      .nullable()
      .describe(
        "Relative enforcement priority. Use higher values for explicit prohibitions or boundaries, lower values for softer style preferences. Use null when classification is retire_commitment or none.",
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
    retires_commitment_id: commitmentIdSchema
      .nullable()
      .optional()
      .describe(
        "Existing active commitment id to retire when classification is retire_commitment. Copy it verbatim from supplied active_commitments; leave null for corrective_preference or none.",
      ),
    applies_to_audience_entity_id: correctivePreferenceEntityIdSchema
      .nullable()
      .optional()
      .describe(
        "Cross-audience scope. Only set this when cross_audience_targets is non-empty AND the current speaker is giving a standing rule explicitly about one of those other audiences/channels; then set it to that audience's entity_id so the rule applies in that audience's sessions instead of the current one. Use a value copied verbatim from cross_audience_targets; never invent an id. Leave null (the default) whenever the rule is about the current conversation or no cross_audience_targets are supplied.",
      ),
    relationship_claims: correctivePreferenceRelationshipClaimsSchema,
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
  applies_to_audience_entity_id: EntityId | null;
  directive_source_stream_entry_id: StreamEntryId | null;
  relationship_claims: RelationshipClaim[];
};

export type CorrectivePreferenceRetirementCandidate = {
  commitmentId: CommitmentId;
  reason: string;
  confidence: number;
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
  retirement: CorrectivePreferenceRetirementCandidate | null;
  slot_negations: CorrectivePreferenceSlotNegation[];
};

export type CorrectivePreferenceExtractorDegradedReason =
  | "llm_unavailable"
  | "llm_failed"
  | "missing_tool_call"
  | "invalid_payload";

export class CorrectivePreferenceExtractorDegradedError extends Error {
  readonly reason: CorrectivePreferenceExtractorDegradedReason;
  readonly degradationCause: unknown;

  constructor(reason: CorrectivePreferenceExtractorDegradedReason, cause?: unknown) {
    super(`Corrective preference extraction degraded: ${reason}`);
    this.name = "CorrectivePreferenceExtractorDegradedError";
    this.reason = reason;
    this.degradationCause = cause;
  }
}

export type CorrectivePreferenceExtractorOptions = {
  llmClient?: LLMClient;
  model?: string;
  tracer?: TurnTracer;
  turnId?: string;
  sessionId?: SessionId;
  // Optional strict degradation propagation; defaults to the tolerant behavior.
  throwOnDegraded?: boolean;
  onDegraded?: (
    reason: CorrectivePreferenceExtractorDegradedReason,
    error?: unknown,
    metadata?: { stopReason: string | null },
  ) => Promise<void> | void;
};

export type ExtractCorrectivePreferenceInput = {
  userMessage: string;
  selfIdentity?: ExtractorSelfIdentity | null;
  currentUserStreamEntryId?: StreamEntryId | null;
  currentUserStreamEntryIds?: readonly StreamEntryId[];
  currentMessageEntries?: readonly StreamEntry[];
  currentMessageSenderAttribution?: readonly CurrentTurnUserInputSenderAttribution[];
  recentHistory: readonly RecencyMessage[];
  audienceEntityId: EntityId | null;
  speakerEntityId?: EntityId | null;
  speakerDisplayName?: string | null;
  senderDisplayNameById?: (entityId: EntityId) => string | null | undefined;
  participantRoster?: ParticipantRoster | null;
  crossAudienceTargets?: readonly { entity_id: EntityId; label: string }[];
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
    restricted_audience?: EntityId | null;
    made_to_entity?: EntityId | null;
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
  sessionId?: SessionId;
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
    ...(options.sessionId !== undefined ? { session_id: options.sessionId } : {}),
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
  traceOptions: Pick<CorrectivePreferenceExtractorOptions, "tracer" | "turnId" | "sessionId"> = {},
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
    sessionId: traceOptions.sessionId,
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
    applies_to_audience_entity_id: input.applies_to_audience_entity_id ?? null,
    directive_source_stream_entry_id: input.directive_source_stream_entry_id,
    relationship_claims: input.relationship_claims.map((claim) => ({
      ...claim,
      evidence_relational_slot_ids: [...claim.evidence_relational_slot_ids],
      evidence_stream_entry_ids: [...claim.evidence_stream_entry_ids],
    })),
  };
}

function toRetirement(
  input: CorrectivePreferenceToolInput,
): CorrectivePreferenceRetirementCandidate | null {
  if (input.classification !== "retire_commitment" || input.confidence < CONFIDENCE_THRESHOLD) {
    return null;
  }

  if (input.retires_commitment_id === null || input.retires_commitment_id === undefined) {
    return null;
  }

  const reason = input.reason.trim();

  if (reason.length === 0) {
    return null;
  }

  return {
    commitmentId: input.retires_commitment_id,
    reason,
    confidence: input.confidence,
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
  traceOptions: Pick<CorrectivePreferenceExtractorOptions, "tracer" | "turnId" | "sessionId"> = {},
): CorrectivePreferenceExtractionResult {
  return {
    preference: toCandidate(input, traceOptions),
    retirement: toRetirement(input),
    slot_negations: slotNegationsFromInput(input),
  };
}

function parseResponse(
  input: unknown,
  presentedCurrentStreamEntryIds: readonly StreamEntryId[],
  traceOptions: Pick<CorrectivePreferenceExtractorOptions, "tracer" | "turnId" | "sessionId"> = {},
): CorrectivePreferenceExtractionResult {
  const parsed = correctivePreferenceSchema.safeParse(input);

  if (!parsed.success) {
    throw parsed.error;
  }

  if (
    parsed.data.directive_source_stream_entry_id !== null &&
    !presentedCurrentStreamEntryIds.some(
      (streamEntryId) => streamEntryId === parsed.data.directive_source_stream_entry_id,
    )
  ) {
    throw new z.ZodError([
      {
        code: "custom",
        path: ["directive_source_stream_entry_id"],
        message: "Directive source stream entry id was not presented in current_message_entries",
      },
    ]);
  }

  return toExtractionResult(parsed.data, traceOptions);
}

function buildCorrectivePreferenceMessages(input: ExtractCorrectivePreferenceInput): {
  messages: LLMMessage[];
  presentedCurrentStreamEntryIds: StreamEntryId[];
} {
  const currentMessageStreamEntryIds =
    input.currentUserStreamEntryIds === undefined || input.currentUserStreamEntryIds.length === 0
      ? input.currentUserStreamEntryId === null || input.currentUserStreamEntryId === undefined
        ? []
        : [input.currentUserStreamEntryId]
      : [...input.currentUserStreamEntryIds];
  const conversationContext = buildExtractorConversationContext({
    selfIdentity: input.selfIdentity ?? null,
    recentHistory: input.recentHistory,
    currentMessageEntries: input.currentMessageEntries,
    currentMessageStreamEntryIds,
    currentMessageSenderAttribution: input.currentMessageSenderAttribution,
    audienceEntityId: input.audienceEntityId,
    speakerEntityId: input.speakerEntityId,
    speakerDisplayName: input.speakerDisplayName,
    senderDisplayNameById: input.senderDisplayNameById,
  });

  return {
    presentedCurrentStreamEntryIds: conversationContext.current_message_entries.map(
      (entry) => entry.stream_entry_id,
    ),
    messages: [
      {
        role: "user",
        content: JSON.stringify({
          current_user_message: input.userMessage,
          current_user_stream_entry_id: input.currentUserStreamEntryId ?? null,
          current_user_stream_entry_ids: currentMessageStreamEntryIds,
          ...conversationContext,
          participant_roster: renderParticipantRoster(input.participantRoster),
          cross_audience_targets: (input.crossAudienceTargets ?? []).map((target) => ({
            entity_id: target.entity_id,
            label: target.label,
          })),
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
              ...memoryDisclosurePayloadFields(
                commitmentMemoryDisclosureLabel({
                  restricted_audience: commitment.restricted_audience ?? null,
                  made_to_entity: commitment.made_to_entity ?? null,
                }),
              ),
            };
          }),
          relational_slots: (input.relationalSlots ?? []).map((slot) => {
            const disclosureFields = memoryDisclosurePayloadFields(
              relationalSlotMemoryDisclosureLabel(slot),
            );

            return {
              id: slot.id ?? null,
              subject_entity_id: slot.subject_entity_id,
              slot_key: slot.slot_key,
              value: slot.value,
              state: slot.state,
              alternate_values: slot.alternate_values.map((alternate) => ({
                value: alternate.value,
                disclosure: disclosureFields.disclosure,
                disclosure_label: disclosureFields.disclosure_label,
              })),
              disclosure: disclosureFields.disclosure,
              disclosure_label: disclosureFields.disclosure_label,
            };
          }),
        }),
      },
    ],
  };
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

    if (this.options.throwOnDegraded === true) {
      throw new CorrectivePreferenceExtractorDegradedError(reason, error);
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
          retirement: null,
          slot_negations: [],
        }
      );
    }

    const requestContext = buildCorrectivePreferenceMessages(input);
    const messages = requestContext.messages;
    const tools = [CORRECTIVE_PREFERENCE_TOOL];

    try {
      const result = await callStructuredTool({
        llmClient: this.options.llmClient,
        request: {
          model: this.options.model,
          system: CORRECTIVE_PREFERENCE_SYSTEM_PROMPT,
          messages,
          tools,
          tool_choice: { type: "tool", name: CORRECTIVE_PREFERENCE_TOOL_NAME },
          max_tokens: EXTRACTOR_MAX_TOKENS_DEFAULT,
          budget: "corrective-preference-extractor",
        },
        toolName: CORRECTIVE_PREFERENCE_TOOL_NAME,
        parse: (toolInput) =>
          parseResponse(toolInput, requestContext.presentedCurrentStreamEntryIds, {
            tracer: this.options.tracer,
            turnId: this.options.turnId,
            sessionId: this.options.sessionId,
          }),
        trace: {
          tracer: this.options.tracer,
          turnId: this.options.turnId,
          sessionId: this.options.sessionId,
          label: "corrective_preference_extractor",
          systemPrompt: CORRECTIVE_PREFERENCE_SYSTEM_PROMPT,
          messages,
          tools,
        },
      });

      return result.parsed;
    } catch (error) {
      if (!isStructuredToolCallError(error) || error.kind === "llm_failed") {
        return (
          (await this.degraded(
            "llm_failed",
            isStructuredToolCallError(error, "llm_failed") ? (error.cause ?? error) : error,
          )) ?? {
            preference: null,
            retirement: null,
            slot_negations: [],
          }
        );
      }

      const degradedError =
        error.kind === "missing_tool_call"
          ? new MissingCorrectivePreferenceToolCallError(
              `Corrective preference extractor did not emit ${CORRECTIVE_PREFERENCE_TOOL_NAME}`,
            )
          : (error.cause ?? error);
      await this.degraded(
        error.kind === "missing_tool_call" ? "missing_tool_call" : "invalid_payload",
        degradedError,
        { stopReason: error.stopReason },
      );
      return {
        preference: null,
        retirement: null,
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
