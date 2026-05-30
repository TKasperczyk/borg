import { z } from "zod";

import {
  type LLMClient,
  type LLMCompleteResult,
  type LLMMessage,
  type LLMToolDefinition,
  toToolInputSchema,
} from "../../llm/index.js";
import {
  creatorDirectiveContentScopeSchema,
  creatorDirectiveDeniedAudienceBehaviorSchema,
  creatorDirectiveEntityIdSchema,
  creatorDirectiveKindSchema,
  creatorDirectiveMentionPolicySchema,
  creatorDirectiveSemanticSlotSchema,
  creatorDirectiveSubjectKindSchema,
  creatorDirectiveTopicTagSchema,
  type CreatorDirectiveContentScope,
  type CreatorDirectiveDeniedAudienceBehavior,
  type CreatorDirectiveKind,
  type CreatorDirectiveMentionPolicy,
  type CreatorDirectiveSemanticSlot,
  type CreatorDirectiveSubjectKind,
} from "../../memory/creator-directives/index.js";
import type { BorgRole } from "../../memory/commitments/index.js";
import type { SessionAudienceRole } from "../../sessions/index.js";
import type { JsonValue } from "../../util/json-value.js";
import type { EntityId, SessionId, StreamEntryId } from "../../util/ids.js";
import { renderParticipantRoster, type ParticipantRoster } from "../perception/index.js";
import { EXTRACTOR_MAX_TOKENS_DEFAULT } from "../prompts/constants.js";
import { CREATOR_DIRECTIVE_SYSTEM_PROMPT } from "../prompts/creator-directive.js";
import type { RecencyMessage } from "../recency/index.js";
import {
  traceLlmCallError,
  traceLlmCallResponse,
  traceLlmCallStarted,
} from "../tracing/llm-call-trace.js";
import type { TurnTracer } from "../tracing/tracer.js";

export const CREATOR_DIRECTIVE_TOOL_NAME = "EmitCreatorDirectives";

const entityLabelSchema = z.string().trim().min(1).max(256);

const creatorDirectiveExtractorDisclosurePolicySchema = z
  .object({
    content_scope: creatorDirectiveContentScopeSchema.describe(
      "Visibility scope. Use operator_only when durable visibility is ambiguous.",
    ),
    allowed_entity_ids: z.array(creatorDirectiveEntityIdSchema).default([]),
    allowed_entity_labels: z.array(entityLabelSchema).default([]),
    excluded_entity_ids: z.array(creatorDirectiveEntityIdSchema).default([]),
    excluded_entity_labels: z.array(entityLabelSchema).default([]),
    subject_may_know: z.boolean().nullable(),
    mention_policy: creatorDirectiveMentionPolicySchema,
    denied_audience_behavior: creatorDirectiveDeniedAudienceBehaviorSchema,
    boundary_prompt: z.string().trim().min(1).nullable(),
    topic_tags: z.array(creatorDirectiveTopicTagSchema).max(32).default([]),
  })
  .strict();

const creatorDirectiveCandidateSchema = z
  .object({
    kind: creatorDirectiveKindSchema,
    subject_kind: creatorDirectiveSubjectKindSchema,
    subject_entity_id: creatorDirectiveEntityIdSchema.nullable().optional(),
    subject_label: entityLabelSchema.nullable().optional(),
    semantic_slot: creatorDirectiveSemanticSlotSchema.nullable(),
    semantic_value: z.string().trim().min(1).nullable(),
    canonical_fact: z.string().trim().min(1).nullable(),
    operational_directive: z.string().trim().min(1),
    disclosure_policy: creatorDirectiveExtractorDisclosurePolicySchema,
    priority: z.number().int(),
    confidence: z.number().min(0).max(1),
    reason: z.string().trim().min(1),
  })
  .strict()
  .superRefine((value, ctx) => {
    if (value.semantic_slot === null && value.semantic_value !== null) {
      ctx.addIssue({
        code: "custom",
        path: ["semantic_value"],
        message: "semantic_value requires semantic_slot",
      });
    }

    if (value.semantic_slot !== null && value.semantic_value === null) {
      ctx.addIssue({
        code: "custom",
        path: ["semantic_value"],
        message: "semantic_slot requires semantic_value",
      });
    }
  });

export const creatorDirectiveExtractionOutputSchema = z
  .object({
    decision: z.enum(["creator_directive", "none"]),
    reason: z.string().trim().min(1),
    candidates: z.array(creatorDirectiveCandidateSchema).max(5).default([]),
  })
  .strict();

const CREATOR_DIRECTIVE_TOOL = {
  name: CREATOR_DIRECTIVE_TOOL_NAME,
  description: "Extract explicit durable creator disclosure directives from the current user turn.",
  inputSchema: toToolInputSchema(creatorDirectiveExtractionOutputSchema),
} satisfies LLMToolDefinition;

type CreatorDirectiveToolInput = z.infer<typeof creatorDirectiveExtractionOutputSchema>;

class MissingCreatorDirectiveToolCallError extends Error {}

export type CreatorDirectiveExtractorDisclosurePolicy = {
  content_scope: CreatorDirectiveContentScope;
  allowed_entity_ids: EntityId[];
  allowed_entity_labels: string[];
  excluded_entity_ids: EntityId[];
  excluded_entity_labels: string[];
  subject_may_know: boolean | null;
  mention_policy: CreatorDirectiveMentionPolicy;
  denied_audience_behavior: CreatorDirectiveDeniedAudienceBehavior;
  boundary_prompt: string | null;
  topic_tags: string[];
};

export type CreatorDirectiveCandidate = {
  kind: CreatorDirectiveKind;
  subject_kind: CreatorDirectiveSubjectKind;
  subject_entity_id: EntityId | null;
  subject_label: string | null;
  semantic_slot: CreatorDirectiveSemanticSlot | null;
  semantic_value: string | null;
  canonical_fact: string | null;
  operational_directive: string;
  disclosure_policy: CreatorDirectiveExtractorDisclosurePolicy;
  priority: number;
  confidence: number;
  reason: string;
};

export type CreatorDirectiveExtractorDegradedReason =
  | "llm_unavailable"
  | "llm_failed"
  | "missing_tool_call"
  | "invalid_payload";

export type KnownCreatorDirectiveEntity = {
  entity_id: EntityId;
  display_name: string;
  role?: string | null;
};

export type CreatorDirectiveExtractorOptions = {
  llmClient?: LLMClient;
  model?: string;
  tracer?: TurnTracer;
  turnId?: string;
  sessionId?: SessionId;
  onDegraded?: (
    reason: CreatorDirectiveExtractorDegradedReason,
    error?: unknown,
    metadata?: { stopReason: string | null },
  ) => Promise<void> | void;
};

export type ExtractCreatorDirectivesInput = {
  userMessage: string;
  currentUserStreamEntryId?: StreamEntryId | null;
  currentUserStreamEntryIds?: readonly StreamEntryId[];
  recentHistory: readonly RecencyMessage[];
  audienceEntityId: EntityId | null;
  currentSenderEntityId: EntityId | null;
  currentSenderDisplayName?: string | null;
  currentSenderBorgRole: BorgRole | null;
  sessionAudienceRole: SessionAudienceRole;
  participantRoster?: ParticipantRoster | null;
  knownEntities?: readonly KnownCreatorDirectiveEntity[];
};

function toCandidate(
  input: z.infer<typeof creatorDirectiveCandidateSchema>,
): CreatorDirectiveCandidate {
  return {
    kind: input.kind,
    subject_kind: input.subject_kind,
    subject_entity_id: input.subject_entity_id ?? null,
    subject_label: input.subject_label ?? null,
    semantic_slot: input.semantic_slot,
    semantic_value: input.semantic_value,
    canonical_fact: input.canonical_fact,
    operational_directive: input.operational_directive.trim(),
    disclosure_policy: {
      content_scope: input.disclosure_policy.content_scope,
      allowed_entity_ids: [...input.disclosure_policy.allowed_entity_ids],
      allowed_entity_labels: [...input.disclosure_policy.allowed_entity_labels],
      excluded_entity_ids: [...input.disclosure_policy.excluded_entity_ids],
      excluded_entity_labels: [...input.disclosure_policy.excluded_entity_labels],
      subject_may_know: input.disclosure_policy.subject_may_know,
      mention_policy: input.disclosure_policy.mention_policy,
      denied_audience_behavior: input.disclosure_policy.denied_audience_behavior,
      boundary_prompt: input.disclosure_policy.boundary_prompt,
      topic_tags: [...input.disclosure_policy.topic_tags],
    },
    priority: input.priority,
    confidence: input.confidence,
    reason: input.reason.trim(),
  };
}

function toCandidates(input: CreatorDirectiveToolInput): CreatorDirectiveCandidate[] {
  if (input.decision !== "creator_directive") {
    return [];
  }

  return input.candidates.map((candidate) => toCandidate(candidate));
}

function parseResponse(result: LLMCompleteResult): CreatorDirectiveCandidate[] {
  const call = result.tool_calls.find((toolCall) => toolCall.name === CREATOR_DIRECTIVE_TOOL_NAME);

  if (call === undefined) {
    throw new MissingCreatorDirectiveToolCallError(
      `Creator directive extractor did not emit ${CREATOR_DIRECTIVE_TOOL_NAME}`,
    );
  }

  const parsed = creatorDirectiveExtractionOutputSchema.safeParse(call.input);

  if (!parsed.success) {
    throw parsed.error;
  }

  return toCandidates(parsed.data);
}

function buildCreatorDirectiveMessages(input: ExtractCreatorDirectivesInput): LLMMessage[] {
  return [
    {
      role: "user",
      content: JSON.stringify({
        current_user_message: input.userMessage,
        current_user_stream_entry_id: input.currentUserStreamEntryId ?? null,
        current_user_stream_entry_ids: [...(input.currentUserStreamEntryIds ?? [])],
        recent_history: input.recentHistory.slice(-8).map((message) => ({
          role: message.role,
          content: message.content,
        })),
        audience_entity_id: input.audienceEntityId,
        current_sender_entity_id: input.currentSenderEntityId,
        current_sender_display_name: input.currentSenderDisplayName ?? null,
        current_sender_borg_role: input.currentSenderBorgRole,
        session_audience_role: input.sessionAudienceRole,
        participant_roster: renderParticipantRoster(input.participantRoster),
        known_entities: (input.knownEntities ?? []).map((entity) => ({
          entity_id: entity.entity_id,
          display_name: entity.display_name,
          role: entity.role ?? null,
        })),
      }),
    },
  ];
}

function summarizeCreatorDirectiveResponseShape(response: LLMCompleteResult): JsonValue {
  return {
    textLength: response.text.length,
    toolUseBlocks: response.tool_calls.map((call) => ({
      id: call.id,
      name: call.name,
    })),
  };
}

export class CreatorDirectiveExtractor {
  constructor(private readonly options: CreatorDirectiveExtractorOptions = {}) {}

  private async degraded(
    reason: CreatorDirectiveExtractorDegradedReason,
    error?: unknown,
    metadata?: { stopReason: string | null },
  ): Promise<CreatorDirectiveCandidate[]> {
    try {
      if (metadata === undefined) {
        await this.options.onDegraded?.(reason, error);
      } else {
        await this.options.onDegraded?.(reason, error, metadata);
      }
    } catch {
      // Best-effort degraded-mode logging only.
    }

    return [];
  }

  async extract(input: ExtractCreatorDirectivesInput): Promise<CreatorDirectiveCandidate[]> {
    if (this.options.llmClient === undefined || this.options.model === undefined) {
      return this.degraded("llm_unavailable");
    }

    const messages = buildCreatorDirectiveMessages(input);
    const tools = [CREATOR_DIRECTIVE_TOOL];

    traceLlmCallStarted({
      tracer: this.options.tracer,
      turnId: this.options.turnId,
      sessionId: this.options.sessionId,
      label: "creator_directive_extractor",
      model: this.options.model,
      systemPrompt: CREATOR_DIRECTIVE_SYSTEM_PROMPT,
      messages,
      tools,
    });

    let response: LLMCompleteResult;

    try {
      response = await this.options.llmClient.complete({
        model: this.options.model,
        system: CREATOR_DIRECTIVE_SYSTEM_PROMPT,
        messages,
        tools,
        tool_choice: { type: "tool", name: CREATOR_DIRECTIVE_TOOL_NAME },
        max_tokens: EXTRACTOR_MAX_TOKENS_DEFAULT,
        budget: "creator-directive-extractor",
      });
    } catch (error) {
      traceLlmCallError({
        tracer: this.options.tracer,
        turnId: this.options.turnId,
        sessionId: this.options.sessionId,
        label: "creator_directive_extractor",
        error,
      });

      return this.degraded("llm_failed", error);
    }

    traceLlmCallResponse({
      tracer: this.options.tracer,
      turnId: this.options.turnId,
      sessionId: this.options.sessionId,
      label: "creator_directive_extractor",
      response,
      responseShape: summarizeCreatorDirectiveResponseShape(response),
    });

    try {
      return parseResponse(response);
    } catch (error) {
      await this.degraded(
        error instanceof MissingCreatorDirectiveToolCallError
          ? "missing_tool_call"
          : error instanceof z.ZodError
            ? "invalid_payload"
            : "llm_failed",
        error,
        { stopReason: response.stop_reason },
      );
      return [];
    }
  }
}
