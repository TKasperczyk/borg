import { type SimilarityConfigSource } from "../../config/similarity.js";
import { z } from "zod";

import {
  callStructuredTool,
  isStructuredToolCallError,
  type LLMClient,
  type LLMMessage,
  type LLMToolDefinition,
  toToolInputSchema,
} from "../../llm/index.js";
import type { EmbeddingClient } from "../../embeddings/index.js";
import {
  ACTION_STATE_METADATA,
  ACTION_STATES,
  actionIdSchema,
  actionEntityIdSchema,
  actionActorSchema,
  actionSessionScopeSchema,
  type ActionRecord,
  type ActionSelectionAudit,
  type ActionRepository,
  type ActionSessionScope,
  type ActionState,
  type ActionStateTimestampField,
} from "../../memory/actions/index.js";
import type { SharedStateEntry } from "../../memory/shared-state/index.js";
import { StorageError } from "../../util/errors.js";
import { goalIdSchema, type GoalRecord } from "../../memory/self/index.js";
import { presentActionCandidates, presentActionGoals } from "./action-candidates.js";
import { SystemClock, type Clock } from "../../util/clock.js";
import {
  createActionId,
  type ActionId,
  type EntityId,
  type GoalId,
  type OpenQuestionId,
  type SessionId,
  type StreamEntryId,
} from "../../util/ids.js";
import { ACTION_STATE_SYSTEM_PROMPT } from "../prompts/action-extraction.js";
import { EXTRACTOR_MAX_TOKENS_DEFAULT } from "../prompts/constants.js";
import {
  memoryDisclosurePayloadFields,
  sharedStateMemoryDisclosureLabel,
} from "../../memory/common/disclosure-serializers.js";
import type { RecencyMessage } from "../recency/index.js";
import type { TurnTracer } from "../../tracing/tracer.js";

const ACTION_STATE_TOOL_NAME = "EmitActionStates";

export const ACTION_CANDIDATE_CLASSIFICATIONS = [
  "concrete_action",
  "conversational_acknowledgment",
  "decision_or_preference",
  "already_represented",
  "outside_borg_capability",
  "none",
] as const;

export const actionCandidateClassificationSchema = z.enum(ACTION_CANDIDATE_CLASSIFICATIONS);
export type ActionCandidateClassification = z.infer<typeof actionCandidateClassificationSchema>;

const ACTION_CANDIDATE_CLASSIFICATION_COUNT_KEYS = [
  ...ACTION_CANDIDATE_CLASSIFICATIONS,
  "invalid_classification",
] as const;

type ActionCandidateClassificationCountKey =
  (typeof ACTION_CANDIDATE_CLASSIFICATION_COUNT_KEYS)[number];

const extractedActionStateSchema = z.enum([
  "considering",
  "committed_to_do",
  "scheduled",
  "completed",
  "not_done",
  "declined",
]);

const actionStateCandidateSchema = z
  .object({
    classification: actionCandidateClassificationSchema.describe(
      "Classify the candidate before persistence. Only concrete_action can become an ActionRecord.",
    ),
    description: z.string().trim().min(1),
    actor: actionActorSchema.describe(
      "Who performs the action. For anything the author of current_user_message did, is doing, or will do, use their speaker_entity_id -- including when that author is another agent or automated system reporting on its own work, and including when the work resembles work Borg does. Use borg only for actions this message asks Borg to perform or reports Borg as having performed. Use user only when no speaker entity is resolvable.",
    ),
    state: extractedActionStateSchema,
    audience_entity_id: actionEntityIdSchema.nullable().optional(),
    session_scope: actionSessionScopeSchema.nullable().optional(),
    matched_existing_action_id: actionIdSchema.nullable().optional(),
    goal_id: goalIdSchema.nullable().optional(),
    evidence_stream_entry_ids: z.array(z.string().min(1)),
    confidence: z.number().min(0).max(1),
  })
  .strict();

const actionStateOutputSchema = z
  .object({
    referenced_action_ids: z
      .array(actionIdSchema)
      .default([])
      .describe(
        "Supplied active action ids explicitly referenced by the current user message or post-turn structural evidence, even when no state transition is asserted.",
      ),
    action_states: z
      .array(actionStateCandidateSchema)
      .describe(
        "Action-state assertions in the current user message. Emit an empty array when none are present.",
      ),
  })
  .strict();

const ACTION_STATE_TOOL = {
  name: ACTION_STATE_TOOL_NAME,
  description:
    "Extract action states asserted by the current user message, citing the current user stream entry.",
  inputSchema: toToolInputSchema(actionStateOutputSchema),
} satisfies LLMToolDefinition;

type ActionStateEnvelopeInput = z.infer<typeof actionStateOutputSchema>;
type ParsedActionStateCandidate = z.infer<typeof actionStateCandidateSchema>;
type ParsedActionStateCandidateWithIndex = {
  candidateIndex: number;
  candidate: ParsedActionStateCandidate;
};
type ActionStateSkippedReason =
  | "missing_current_user_evidence"
  | "repository_failed"
  | "invalid_classification"
  | "invalid_candidate"
  | "non_concrete_classification";
type ActionStateSkippedCandidate = {
  candidate_index: number;
  reason: ActionStateSkippedReason;
};
type ActionCandidateRejectedReason = "non_concrete_classification";
type ActionCandidateRejected = {
  candidate_index: number;
  classification: ActionCandidateClassification;
  description_excerpt: string;
  reason: ActionCandidateRejectedReason;
};
type ActionStateParseResult = {
  referencedActionIds: ActionId[];
  candidates: ParsedActionStateCandidateWithIndex[];
  candidatesEmitted: number;
  validCandidateCount: number;
  skippedCandidates: ActionStateSkippedCandidate[];
  classificationCounts: Record<ActionCandidateClassificationCountKey, number>;
  rejectedCandidates: ActionCandidateRejected[];
};
class MissingActionStateToolCallError extends Error {}

export type ActionStateExtractorDegradedReason =
  | "llm_unavailable"
  | "repository_unavailable"
  | "llm_failed"
  | "missing_tool_call"
  | "invalid_payload"
  | "repository_failed";

export type ActionStateExtractorOptions = {
  /** @deprecated Ignored: similarity retrieves candidates, never determines persistence. */
  similarityConfig?: SimilarityConfigSource;
  llmClient?: LLMClient;
  model?: string;
  actionRepository?: Pick<ActionRepository, "add"> &
    Partial<
      Pick<
        ActionRepository,
        "get" | "list" | "update" | "appendExtractionUpdate" | "applyExtractionDecision"
      >
    >;
  /** @deprecated Ignored: candidates are retrieved by the calling service. */
  embeddingClient?: EmbeddingClient;
  clock?: Clock;
  tracer?: TurnTracer;
  turnId?: string;
  sessionId?: SessionId;
  onDegraded?: (
    reason: ActionStateExtractorDegradedReason,
    error?: unknown,
  ) => Promise<void> | void;
};

export type ExtractActionStatesInput = {
  userMessage: string;
  currentUserStreamEntryId: StreamEntryId;
  currentUserStreamEntryIds?: readonly StreamEntryId[];
  currentAgentStreamEntryId?: StreamEntryId;
  recentHistory: readonly RecencyMessage[];
  audienceEntityId: EntityId | null;
  sessionId?: SessionId | null;
  speakerEntityId?: EntityId | null;
  speakerDisplayName?: string | null;
  senderDisplayNameById?: (entityId: EntityId) => string | null | undefined;
  senderAttribution?: readonly {
    entryId: StreamEntryId;
    senderEntityId: EntityId | null;
    senderDisplayName?: string;
  }[];
  /** @deprecated Ignored: a turn focus is not an action goal. */
  goalId?: GoalId | null;
  suppliedGoals?: readonly GoalRecord[];
  openQuestionId?: OpenQuestionId | null;
  turnCounter?: number | null;
  activeActionsForReference?: readonly ActionRecord[];
  actionCandidateTotal?: number;
  postTurnSelfPerformance?: {
    activeBorgActions: readonly ActionRecord[];
    currentTurnSharedStateEntries: readonly SharedStateEntry[];
    agentResponse: string;
  };
};

function compactSharedStateEntryForPrompt(entry: SharedStateEntry): Record<string, unknown> {
  return {
    id: entry.id,
    kind: entry.kind,
    text: entry.text,
    owner_entity_id: entry.owner_entity_id,
    last_updated_stream_entry_ids: entry.last_updated_stream_entry_ids,
    canonicalizes: entry.canonicalizes,
    ...memoryDisclosurePayloadFields(sharedStateMemoryDisclosureLabel(entry)),
  };
}

function buildActionStateMessages(
  input: ExtractActionStatesInput,
  candidates: ReturnType<typeof presentActionCandidates>,
  goals: ReturnType<typeof presentActionGoals>,
): LLMMessage[] {
  return [
    {
      role: "user",
      content: JSON.stringify({
        current_user_message: input.userMessage,
        current_user_stream_entry_id: input.currentUserStreamEntryId,
        current_user_stream_entry_ids: [
          ...(input.currentUserStreamEntryIds ?? [input.currentUserStreamEntryId]),
        ],
        recent_history_context: input.recentHistory.slice(-8).map((message) => ({
          context_stream_entry_id: message.stream_entry_id,
          role: message.role,
          kind: message.kind ?? null,
          sender_entity_id: message.sender_entity_id,
          sender_display_name:
            message.sender_entity_id === null
              ? null
              : (input.senderDisplayNameById?.(message.sender_entity_id) ?? null),
          content: message.content,
        })),
        audience_entity_id: input.audienceEntityId,
        current_session_id: input.sessionId ?? null,
        speaker_entity_id: input.speakerEntityId ?? null,
        speaker_display_name: input.speakerDisplayName ?? null,
        sender_attribution: (input.senderAttribution ?? []).map((item) => ({
          stream_entry_id: item.entryId,
          sender_entity_id: item.senderEntityId,
          sender_display_name: item.senderDisplayName ?? null,
        })),
        current_agent_stream_entry_id: input.currentAgentStreamEntryId ?? null,
        active_actions_for_reference: candidates.rows,
        action_candidate_omitted_count: candidates.omitted,
        supplied_goals: goals.rows,
        goal_candidate_omitted_count: goals.omitted,
        post_turn_self_performance:
          input.postTurnSelfPerformance === undefined
            ? null
            : {
                active_borg_action_ids: candidates.shown
                  .filter((row) => row.actor === "borg")
                  .map((row) => row.id),
                current_turn_shared_state_entries:
                  input.postTurnSelfPerformance.currentTurnSharedStateEntries.map(
                    compactSharedStateEntryForPrompt,
                  ),
                agent_response: input.postTurnSelfPerformance.agentResponse,
              },
      }),
    },
  ];
}

function zeroClassificationCounts(): Record<ActionCandidateClassificationCountKey, number> {
  return {
    concrete_action: 0,
    conversational_acknowledgment: 0,
    decision_or_preference: 0,
    already_represented: 0,
    outside_borg_capability: 0,
    none: 0,
    invalid_classification: 0,
  };
}

function incrementClassificationCount(
  counts: Record<ActionCandidateClassificationCountKey, number>,
  key: ActionCandidateClassificationCountKey,
): void {
  counts[key] += 1;
}

function descriptionExcerpt(description: string): string {
  return description.trim().slice(0, 60);
}

function rejectedCandidate(input: {
  candidateIndex: number;
  classification: ActionCandidateClassification;
  description: string;
  reason: ActionCandidateRejectedReason;
}): ActionCandidateRejected {
  return {
    candidate_index: input.candidateIndex,
    classification: input.classification,
    description_excerpt: descriptionExcerpt(input.description),
    reason: input.reason,
  };
}

function parseCandidates(envelope: ActionStateEnvelopeInput): {
  candidates: ParsedActionStateCandidateWithIndex[];
  skippedCandidates: ActionStateSkippedCandidate[];
  classificationCounts: Record<ActionCandidateClassificationCountKey, number>;
  rejectedCandidates: ActionCandidateRejected[];
} {
  const candidates: ParsedActionStateCandidateWithIndex[] = [];
  const skippedCandidates: ActionStateSkippedCandidate[] = [];
  const classificationCounts = zeroClassificationCounts();
  const rejectedCandidates: ActionCandidateRejected[] = [];

  for (const [candidateIndex, candidate] of envelope.action_states.entries()) {
    incrementClassificationCount(classificationCounts, candidate.classification);

    if (candidate.classification !== "concrete_action") {
      rejectedCandidates.push(
        rejectedCandidate({
          candidateIndex,
          classification: candidate.classification,
          description: candidate.description,
          reason: "non_concrete_classification",
        }),
      );
      continue;
    }

    candidates.push({
      candidateIndex,
      candidate,
    });
  }

  return {
    candidates,
    skippedCandidates,
    classificationCounts,
    rejectedCandidates,
  };
}

function parseResponse(input: unknown): ActionStateParseResult {
  const parsed = actionStateOutputSchema.safeParse(input);

  if (!parsed.success) {
    throw parsed.error;
  }

  const candidates = parseCandidates(parsed.data);
  const referencedActionIds = parsed.data.referenced_action_ids;

  return {
    referencedActionIds,
    candidates: candidates.candidates,
    candidatesEmitted: parsed.data.action_states.length,
    validCandidateCount: candidates.candidates.length + candidates.rejectedCandidates.length,
    skippedCandidates: candidates.skippedCandidates,
    classificationCounts: candidates.classificationCounts,
    rejectedCandidates: candidates.rejectedCandidates,
  };
}

function degradedReasonForStructuredToolError(error: unknown): ActionStateExtractorDegradedReason {
  if (isStructuredToolCallError(error, "missing_tool_call")) {
    return "missing_tool_call";
  }

  if (isStructuredToolCallError(error, "invalid_payload")) {
    return "invalid_payload";
  }

  return "llm_failed";
}

function hasCurrentUserEvidence(
  candidate: ParsedActionStateCandidate,
  currentUserStreamEntryIds: readonly StreamEntryId[],
): boolean {
  const currentIds = new Set<string>(currentUserStreamEntryIds);

  return candidate.evidence_stream_entry_ids.some((entryId) => currentIds.has(entryId));
}

function hasOnlyAllowedEvidence(
  candidate: ParsedActionStateCandidate,
  allowedEvidenceStreamEntryIds: ReadonlySet<string>,
): boolean {
  return (
    candidate.evidence_stream_entry_ids.length > 0 &&
    candidate.evidence_stream_entry_ids.every((entryId) =>
      allowedEvidenceStreamEntryIds.has(entryId),
    )
  );
}

function allowedEvidenceStreamEntryIds(input: ExtractActionStatesInput): Set<string> {
  return new Set([
    ...(input.currentUserStreamEntryIds ?? [input.currentUserStreamEntryId]),
    ...(input.currentAgentStreamEntryId === undefined ? [] : [input.currentAgentStreamEntryId]),
  ]);
}

function hasAllowedEvidence(
  candidate: ParsedActionStateCandidate,
  input: ExtractActionStatesInput,
): boolean {
  if (input.postTurnSelfPerformance === undefined) {
    const currentUserStreamEntryIds = input.currentUserStreamEntryIds ?? [
      input.currentUserStreamEntryId,
    ];

    return (
      hasCurrentUserEvidence(candidate, currentUserStreamEntryIds) &&
      hasOnlyAllowedEvidence(candidate, new Set(currentUserStreamEntryIds))
    );
  }

  const allowed = allowedEvidenceStreamEntryIds(input);

  return hasOnlyAllowedEvidence(candidate, allowed);
}

function allowedCandidateEvidenceStreamEntryIds(
  candidate: ParsedActionStateCandidate,
  input: ExtractActionStatesInput,
): StreamEntryId[] {
  const allowed = allowedEvidenceStreamEntryIds(input);

  return candidate.evidence_stream_entry_ids.filter((entryId): entryId is StreamEntryId =>
    allowed.has(entryId),
  );
}

function stateTimestampPatch(
  state: ActionState,
  timestamp: number,
): Partial<Record<ActionStateTimestampField, number>> {
  const timestampField = ACTION_STATE_METADATA[state].timestamp_field;

  return { [timestampField]: timestamp };
}

function speakerEntityIdForCandidateEvidence(input: {
  candidate: ParsedActionStateCandidate;
  senderAttribution?: ExtractActionStatesInput["senderAttribution"];
  fallbackSpeakerEntityId: EntityId | null;
}): EntityId | null {
  const evidenceIds = new Set(input.candidate.evidence_stream_entry_ids);
  const senderIds = [
    ...new Set(
      (input.senderAttribution ?? []).flatMap((item) =>
        evidenceIds.has(item.entryId) && item.senderEntityId !== null ? [item.senderEntityId] : [],
      ),
    ),
  ];

  return senderIds.length === 1 ? (senderIds[0] ?? null) : input.fallbackSpeakerEntityId;
}

function toActionRecord(input: {
  candidate: ParsedActionStateCandidate;
  currentUserStreamEntryIds: readonly StreamEntryId[];
  audienceEntityId: EntityId | null;
  sessionId: SessionId | null;
  speakerEntityId: EntityId | null;
  senderAttribution?: ExtractActionStatesInput["senderAttribution"];
  goalId: GoalId | null;
  openQuestionId: OpenQuestionId | null;
  nowMs: number;
  turnCounter: number | null;
}): ActionRecord {
  const candidateEvidenceIds = new Set(input.candidate.evidence_stream_entry_ids);
  const sourceStreamEntryIds = input.currentUserStreamEntryIds.filter((entryId) =>
    candidateEvidenceIds.has(entryId),
  );
  const provenanceStreamEntryIds =
    sourceStreamEntryIds.length === 0 ? [...input.currentUserStreamEntryIds] : sourceStreamEntryIds;
  const actorEntityId =
    input.candidate.actor === "user"
      ? speakerEntityIdForCandidateEvidence({
          candidate: input.candidate,
          senderAttribution: input.senderAttribution,
          fallbackSpeakerEntityId: input.speakerEntityId,
        })
      : null;

  return {
    id: createActionId(),
    description: input.candidate.description,
    actor: input.candidate.actor === "user" ? (actorEntityId ?? "user") : input.candidate.actor,
    audience_entity_id: input.candidate.audience_entity_id ?? input.audienceEntityId,
    goal_id: input.goalId,
    goal_link_basis: "extractor",
    open_question_id: input.openQuestionId,
    state: input.candidate.state,
    confidence: input.candidate.confidence,
    provenance_episode_ids: [],
    provenance_stream_entry_ids: provenanceStreamEntryIds,
    origin_stream_entry_ids: provenanceStreamEntryIds,
    created_at: input.nowMs,
    updated_at: input.nowMs,
    considering_at: null,
    committed_at: null,
    scheduled_at: null,
    completed_at: null,
    not_done_at: null,
    declined_at: null,
    expired_at: null,
    archived_at: null,
    unknown_at: null,
    canonicalized_by_artifact_entry_id: null,
    session_scope: input.candidate.session_scope ?? null,
    session_anchor_id: input.candidate.session_scope == null ? null : input.sessionId,
    last_referenced_at_ms: input.nowMs,
    last_referenced_turn_counter: input.turnCounter ?? null,
    last_referenced_turn_global: input.turnCounter ?? null,
    ...stateTimestampPatch(input.candidate.state, input.nowMs),
  };
}

function zeroActionStateCounts(): Record<ActionState, number> {
  return Object.fromEntries(ACTION_STATES.map((state) => [state, 0])) as Record<
    ActionState,
    number
  >;
}

function isTerminalEmissionState(
  state: ActionState,
): state is "completed" | "not_done" | "declined" {
  return ACTION_STATE_METADATA[state].terminal;
}

function traceExtractorCompleted(options: {
  tracer?: TurnTracer;
  turnId?: string;
  sessionId?: SessionId;
  candidatesEmitted: number;
  validCandidateCount?: number;
  persisted: readonly ActionRecord[];
  skippedReasons: ReadonlyMap<ActionStateSkippedReason, number>;
  skippedCandidates?: readonly ActionStateSkippedCandidate[];
  classificationCounts?: Record<ActionCandidateClassificationCountKey, number>;
  rejectedCandidates?: readonly ActionCandidateRejected[];
  terminalEmissionClosures?: number;
  selfPerformanceClosures?: number;
  degraded: boolean;
  fatalReason?: ActionStateExtractorDegradedReason;
}): void {
  if (options.tracer?.enabled !== true || options.turnId === undefined) {
    return;
  }

  const persistedByState = zeroActionStateCounts();

  for (const record of options.persisted) {
    persistedByState[record.state] += 1;
  }

  const skippedReasons = [...options.skippedReasons.entries()].map(([reason, count]) => ({
    reason,
    count,
  }));
  const rejectedByClassification = zeroClassificationCounts();
  const classificationCounts = options.classificationCounts ?? zeroClassificationCounts();
  const rejectedCandidates = options.rejectedCandidates ?? [];

  for (const rejection of rejectedCandidates) {
    if (rejection.reason === "non_concrete_classification") {
      incrementClassificationCount(rejectedByClassification, rejection.classification);
    }

    options.tracer.emit("extraction.actions.rejected", {
      turnId: options.turnId,
      ...(options.sessionId !== undefined ? { session_id: options.sessionId } : {}),
      classification: rejection.classification,
      description_excerpt: rejection.description_excerpt,
      reason: rejection.reason,
    });
  }

  options.tracer.emit("extraction.actions.completed", {
    turnId: options.turnId,
    ...(options.sessionId !== undefined ? { session_id: options.sessionId } : {}),
    candidates_emitted: options.candidatesEmitted,
    valid_candidate_count: options.validCandidateCount ?? 0,
    persisted_count: options.persisted.length,
    skipped_count: skippedReasons.reduce((sum, reason) => sum + reason.count, 0),
    skipped_reasons: skippedReasons,
    skipped_candidates: options.skippedCandidates?.map((candidate) => ({ ...candidate })) ?? [],
    persisted_by_state: persistedByState,
    classification_counts: classificationCounts,
    rejected_by_classification: rejectedByClassification,
    rejected_invalid_enum: classificationCounts.invalid_classification,
    // Preserve existing telemetry keys; similarity-based persistence no longer runs.
    action_persistence_dedup_skipped_embedding: 0,
    action_persistence_dedup_degraded: 0,
    actions_closed_by_terminal_emission: options.terminalEmissionClosures ?? 0,
    actions_closed_by_borg_self_performance: options.selfPerformanceClosures ?? 0,
    degraded: options.degraded,
    ...(options.fatalReason === undefined ? {} : { fatal_reason: options.fatalReason }),
  });
}

function incrementSkippedReason(
  reasons: Map<ActionStateSkippedReason, number>,
  reason: ActionStateSkippedReason,
): void {
  reasons.set(reason, (reasons.get(reason) ?? 0) + 1);
}

function incrementSkippedCandidate(
  skippedCandidates: ActionStateSkippedCandidate[],
  candidateIndex: number,
  reason: ActionStateSkippedReason,
): void {
  skippedCandidates.push({
    candidate_index: candidateIndex,
    reason,
  });
}

export class ActionStateExtractor {
  private readonly clock: Clock;

  constructor(private readonly options: ActionStateExtractorOptions = {}) {
    this.clock = options.clock ?? new SystemClock();
  }

  private async degraded(
    reason: ActionStateExtractorDegradedReason,
    error?: unknown,
  ): Promise<ActionRecord[]> {
    try {
      await this.options.onDegraded?.(reason, error);
    } catch {
      // Best-effort degraded-mode logging only.
    }

    return [];
  }

  private async degradedWithTrace(
    reason: ActionStateExtractorDegradedReason,
    error?: unknown,
    sessionId?: SessionId,
  ): Promise<ActionRecord[]> {
    const result = await this.degraded(reason, error);
    if (reason === "llm_unavailable" || reason === "repository_unavailable")
      this.traceSelection(0, 0, 0, 0, []);

    traceExtractorCompleted({
      tracer: this.options.tracer,
      turnId: this.options.turnId,
      sessionId,
      candidatesEmitted: 0,
      persisted: result,
      skippedReasons: new Map(),
      degraded: true,
    });

    return result;
  }

  private traceSelection(
    shown: number,
    omitted: number,
    matched: number,
    inserted: number,
    decisions: ActionSelectionAudit[],
  ): void {
    if (this.options.tracer?.enabled !== true || this.options.turnId === undefined) return;
    const rejected: Record<string, number> = {};
    for (const decision of decisions)
      if (decision.rejection_reason !== null)
        rejected[decision.rejection_reason] = (rejected[decision.rejection_reason] ?? 0) + 1;
    this.options.tracer.emit("extraction.actions.selection_summary", {
      turnId: this.options.turnId,
      candidates_shown: shown,
      candidates_omitted: omitted,
      matched,
      inserted,
      rejected_by_reason: rejected,
    });
  }

  async extract(input: ExtractActionStatesInput): Promise<ActionRecord[]> {
    const sessionId = input.sessionId ?? this.options.sessionId;

    if (this.options.llmClient === undefined || this.options.model === undefined) {
      return this.degradedWithTrace("llm_unavailable", undefined, sessionId);
    }

    if (this.options.actionRepository === undefined) {
      return this.degradedWithTrace("repository_unavailable", undefined, sessionId);
    }

    const candidateRecords = [
      ...new Map(
        [
          ...(input.activeActionsForReference ?? []),
          ...(input.postTurnSelfPerformance?.activeBorgActions ?? []),
        ].map((record) => [record.id, record]),
      ).values(),
    ];
    const candidates = presentActionCandidates(
      candidateRecords,
      input.actionCandidateTotal ?? candidateRecords.length,
    );
    const goals = presentActionGoals(input.suppliedGoals ?? []);
    const messages = buildActionStateMessages(input, candidates, goals);
    const tools = [ACTION_STATE_TOOL];

    let parsed: ActionStateParseResult;

    try {
      parsed = (
        await callStructuredTool({
          llmClient: this.options.llmClient,
          request: {
            model: this.options.model,
            system: ACTION_STATE_SYSTEM_PROMPT,
            messages,
            tools,
            tool_choice: { type: "tool", name: ACTION_STATE_TOOL_NAME },
            max_tokens: EXTRACTOR_MAX_TOKENS_DEFAULT,
            budget: "action-state-extractor",
          },
          toolName: ACTION_STATE_TOOL_NAME,
          parse: parseResponse,
          trace: {
            tracer: this.options.tracer,
            turnId: this.options.turnId,
            sessionId,
            label: "action_state_extractor",
            systemPrompt: ACTION_STATE_SYSTEM_PROMPT,
            messages,
            tools,
          },
        })
      ).parsed;
    } catch (error) {
      this.traceSelection(candidates.shown.length, candidates.omitted, 0, 0, []);
      const reason = degradedReasonForStructuredToolError(error);

      if (reason === "llm_failed") {
        return this.degradedWithTrace(
          "llm_failed",
          isStructuredToolCallError(error, "llm_failed") ? (error.cause ?? error) : error,
          sessionId,
        );
      }

      const degradedError = isStructuredToolCallError(error, "missing_tool_call")
        ? new MissingActionStateToolCallError(
            `Action state extractor did not emit ${ACTION_STATE_TOOL_NAME}`,
          )
        : isStructuredToolCallError(error, "invalid_payload")
          ? (error.cause ?? error)
          : error;
      const result = await this.degraded(reason, degradedError);

      traceExtractorCompleted({
        tracer: this.options.tracer,
        turnId: this.options.turnId,
        sessionId,
        candidatesEmitted: 0,
        persisted: result,
        skippedReasons: new Map(),
        degraded: true,
        fatalReason: reason,
      });

      return result;
    }

    const persisted: ActionRecord[] = [];
    const skippedReasons = new Map<ActionStateSkippedReason, number>();
    for (const skippedCandidate of parsed.skippedCandidates) {
      incrementSkippedReason(skippedReasons, skippedCandidate.reason);
    }
    const skippedCandidates = [...parsed.skippedCandidates];
    const rejectedCandidates = [...parsed.rejectedCandidates];
    for (const rejected of rejectedCandidates) {
      incrementSkippedReason(skippedReasons, rejected.reason);
    }
    let degraded = false;
    let terminalEmissionClosures = 0;
    let selfPerformanceClosures = 0;
    const nowMs = this.clock.now();
    const turnCounter = input.turnCounter ?? null;
    const offeredIds = new Set(candidates.shown.map((record) => record.id));
    const suppliedGoalIds = new Set(goals.shown.map((goal) => goal.id));

    const decisions: ActionRecord[] = [];
    for (const { candidate, candidateIndex } of parsed.candidates) {
      if (!hasAllowedEvidence(candidate, input)) {
        incrementSkippedReason(skippedReasons, "missing_current_user_evidence");
        incrementSkippedCandidate(
          skippedCandidates,
          candidateIndex,
          "missing_current_user_evidence",
        );
        continue;
      }
      const chosenId = candidate.matched_existing_action_id ?? null;
      const chosenGoalId = candidate.goal_id ?? null;
      const goalWasOffered = chosenGoalId === null || suppliedGoalIds.has(chosenGoalId);
      const audit: ActionSelectionAudit = {
        turn_id: this.options.turnId ?? null,
        candidate_index: candidateIndex,
        candidate_ids: [...offeredIds],
        candidate_omitted_count: candidates.omitted,
        chosen_action_id: chosenId,
        rejection_reason: chosenId !== null && !offeredIds.has(chosenId) ? "not_offered" : null,
        supplied_goal_ids: [...suppliedGoalIds],
        chosen_goal_id: chosenGoalId,
        goal_rejection_reason: goalWasOffered ? null : "not_offered",
      };
      const record = toActionRecord({
        candidate,
        currentUserStreamEntryIds: allowedCandidateEvidenceStreamEntryIds(candidate, input),
        audienceEntityId: input.audienceEntityId,
        sessionId: input.sessionId ?? null,
        speakerEntityId: input.speakerEntityId ?? null,
        senderAttribution: input.senderAttribution,
        goalId: goalWasOffered ? chosenGoalId : null,
        openQuestionId: input.openQuestionId ?? null,
        nowMs,
        turnCounter,
      });
      record.selection_audit = audit;
      decisions.push(record);
    }

    let matchedCount = 0;
    let committedDecisions: ActionSelectionAudit[] = [];
    try {
      if (this.options.actionRepository.applyExtractionDecision === undefined) {
        throw new StorageError("Atomic action decision repository unavailable", {
          code: "ACTION_DECISION_REPOSITORY_UNAVAILABLE",
        });
      }
      const result = this.options.actionRepository.applyExtractionDecision({
        records: decisions,
        audit: {
          turn_id: this.options.turnId ?? null,
          candidate_ids: [...offeredIds],
          candidate_omitted_count: candidates.omitted,
          referenced_action_ids: parsed.referencedActionIds,
          applied_reference_ids: [],
          ignored_reference_ids: [],
          decisions: [],
        },
        nowMs,
        turnCounter,
      });
      persisted.push(...result.inserted);
      matchedCount = result.matched.length;
      committedDecisions = result.audit.decisions;
      for (const failure of result.completionFailures) {
        if (this.options.tracer?.enabled === true && this.options.turnId !== undefined)
          this.options.tracer.emit("action_state.completion_hook_failed", {
            turnId: this.options.turnId,
            action_id: failure.actionId,
            error: failure.error instanceof Error ? failure.error.message : String(failure.error),
          });
      }
      for (const { decision } of result.matched) {
        if (isTerminalEmissionState(decision.state)) {
          terminalEmissionClosures += 1;
          if (input.postTurnSelfPerformance !== undefined) selfPerformanceClosures += 1;
        }
        if (this.options.tracer?.enabled === true && this.options.turnId !== undefined) {
          this.options.tracer.emit("action_state.transitioned", {
            turnId: this.options.turnId,
            action_id: decision.selection_audit!.chosen_action_id,
            candidate_index: decision.selection_audit!.candidate_index,
            terminal_state: decision.state,
            description_excerpt: descriptionExcerpt(decision.description),
          });
          if (input.postTurnSelfPerformance !== undefined)
            this.options.tracer.emit("action_state.borg_self_performance.completed", {
              turnId: this.options.turnId,
              action_id: decision.selection_audit!.chosen_action_id,
              terminal_state: decision.state,
            });
        }
      }
      for (const audit of committedDecisions) {
        if (
          (audit.rejection_reason !== null || audit.goal_rejection_reason !== null) &&
          this.options.tracer?.enabled === true &&
          this.options.turnId !== undefined
        ) {
          this.options.tracer.emit("extraction.actions.selection_rejected", {
            turnId: this.options.turnId,
            candidate_index: audit.candidate_index,
            chosen_action_id: audit.chosen_action_id,
            reason: audit.rejection_reason,
            chosen_goal_id: audit.chosen_goal_id,
            goal_reason: audit.goal_rejection_reason,
          });
        }
      }
    } catch (error) {
      degraded = true;
      incrementSkippedReason(skippedReasons, "repository_failed");
      await this.degraded("repository_failed", error);
    }
    this.traceSelection(
      candidates.shown.length,
      candidates.omitted,
      matchedCount,
      persisted.length,
      committedDecisions,
    );

    traceExtractorCompleted({
      tracer: this.options.tracer,
      turnId: this.options.turnId,
      sessionId,
      candidatesEmitted: parsed.candidatesEmitted,
      validCandidateCount: parsed.validCandidateCount,
      persisted,
      skippedReasons,
      skippedCandidates,
      classificationCounts: parsed.classificationCounts,
      rejectedCandidates,
      terminalEmissionClosures,
      selfPerformanceClosures,
      degraded,
    });

    return persisted;
  }
}

export { ACTION_STATE_TOOL_NAME };
