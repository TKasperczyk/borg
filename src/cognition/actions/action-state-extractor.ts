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
  ACTIVE_ACTION_STATES,
  ACTION_STATE_METADATA,
  ACTION_STATES,
  actionIdSchema,
  actionEntityIdSchema,
  actionActorSchema,
  actionSessionScopeSchema,
  type ActionRecord,
  type ActionRecordPatch,
  type ActionRepository,
  type ActionSessionScope,
  type ActionState,
  type ActionStateTimestampField,
} from "../../memory/actions/index.js";
import {
  completeAction,
  markActionNotDone,
  type LifecycleOperationResult,
} from "../../memory/lifecycle-ops/index.js";
import type { SharedStateEntry } from "../../memory/shared-state/index.js";
import { cosineSimilarity } from "../../retrieval/embedding-similarity.js";
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
  actionMemoryDisclosureLabel,
  memoryDisclosurePayloadFields,
  sharedStateMemoryDisclosureLabel,
} from "../../memory/common/disclosure-serializers.js";
import type { RecencyMessage } from "../recency/index.js";
import type { TurnTracer } from "../../tracing/tracer.js";

const ACTION_STATE_TOOL_NAME = "EmitActionStates";
const ACTION_PERSISTENCE_DUPLICATE_SIMILARITY_THRESHOLD = 0.85;

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
]);

const actionStateCandidateSchema = z
  .object({
    classification: actionCandidateClassificationSchema.describe(
      "Classify the candidate before persistence. Only concrete_action can become an ActionRecord.",
    ),
    description: z.string().trim().min(1),
    actor: actionActorSchema,
    state: extractedActionStateSchema,
    audience_entity_id: actionEntityIdSchema.nullable().optional(),
    session_scope: actionSessionScopeSchema.nullable().optional(),
    matched_existing_action_id: actionIdSchema.nullable().optional(),
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

const actionStateEnvelopeSchema = z
  .object({
    referenced_action_ids: z.array(z.unknown()).optional(),
    action_states: z.array(z.unknown()),
  })
  .strict();

const ACTION_STATE_TOOL = {
  name: ACTION_STATE_TOOL_NAME,
  description:
    "Extract action states asserted by the current user message, citing the current user stream entry.",
  inputSchema: toToolInputSchema(actionStateOutputSchema),
} satisfies LLMToolDefinition;

type ActionStateEnvelopeInput = z.infer<typeof actionStateEnvelopeSchema>;
type ParsedActionStateCandidate = z.infer<typeof actionStateCandidateSchema>;
type ParsedActionStateCandidateWithIndex = {
  candidateIndex: number;
  candidate: ParsedActionStateCandidate;
};
type ActionStateSkippedReason =
  | "missing_current_user_evidence"
  | "repository_failed"
  | "lifecycle_no_op"
  | "lifecycle_conflict"
  | "invalid_classification"
  | "invalid_candidate"
  | "non_concrete_classification"
  | "embedding_dedup";
type ActionStateSkippedCandidate = {
  candidate_index: number;
  reason: ActionStateSkippedReason;
};
type ActionCandidateRejectedReason = "non_concrete_classification" | "embedding_dedup";
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
type EmbeddedActionVector = {
  actionId: ActionId;
  vector: Float32Array;
  record: ActionRecord;
};
type ActionEmbeddingDedupState = {
  activeVectors: EmbeddedActionVector[];
  acceptedVectors: EmbeddedActionVector[];
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
  llmClient?: LLMClient;
  model?: string;
  actionRepository?: Pick<ActionRepository, "add"> &
    Partial<Pick<ActionRepository, "get" | "list" | "update">>;
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
  senderAttribution?: readonly {
    entryId: StreamEntryId;
    senderEntityId: EntityId | null;
    senderDisplayName?: string;
  }[];
  goalId?: GoalId | null;
  openQuestionId?: OpenQuestionId | null;
  turnCounter?: number | null;
  activeActionsForReference?: readonly ActionRecord[];
  postTurnSelfPerformance?: {
    activeBorgActions: readonly ActionRecord[];
    currentTurnSharedStateEntries: readonly SharedStateEntry[];
    agentResponse: string;
  };
  persistNewActions?: boolean;
};

function compactActionForPrompt(action: ActionRecord): Record<string, unknown> {
  return {
    id: action.id,
    description: action.description,
    actor: action.actor,
    state: action.state,
    audience_entity_id: action.audience_entity_id,
    session_scope: action.session_scope,
    session_anchor_id: action.session_anchor_id,
    last_referenced_turn_counter: action.last_referenced_turn_counter,
    last_referenced_turn_global: action.last_referenced_turn_global ?? null,
    ...memoryDisclosurePayloadFields(actionMemoryDisclosureLabel(action)),
  };
}

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

function buildActionStateMessages(input: ExtractActionStatesInput): LLMMessage[] {
  return [
    {
      role: "user",
      content: JSON.stringify({
        current_user_message: input.userMessage,
        current_user_stream_entry_id: input.currentUserStreamEntryId,
        current_user_stream_entry_ids: [
          ...(input.currentUserStreamEntryIds ?? [input.currentUserStreamEntryId]),
        ],
        recent_history: input.recentHistory.slice(-8).map((message) => ({
          role: message.role,
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
        active_actions_for_reference: (input.activeActionsForReference ?? []).map(
          compactActionForPrompt,
        ),
        post_turn_self_performance:
          input.postTurnSelfPerformance === undefined
            ? null
            : {
                active_borg_actions:
                  input.postTurnSelfPerformance.activeBorgActions.map(compactActionForPrompt),
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

function skippedReasonFromIssue(issue: {
  path: readonly (string | number | symbol)[];
}): ActionStateSkippedReason {
  return issue.path[0] === "classification" ? "invalid_classification" : "invalid_candidate";
}

function skippedReasonFromError(error: z.ZodError): ActionStateSkippedReason {
  return skippedReasonFromIssue(error.issues[0] ?? { path: [] });
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

  for (const [candidateIndex, rawCandidate] of envelope.action_states.entries()) {
    const parsed = actionStateCandidateSchema.safeParse(rawCandidate);

    if (!parsed.success) {
      const reason = skippedReasonFromError(parsed.error);

      if (reason === "invalid_classification") {
        incrementClassificationCount(classificationCounts, "invalid_classification");
      }

      skippedCandidates.push({
        candidate_index: candidateIndex,
        reason,
      });
      continue;
    }

    incrementClassificationCount(classificationCounts, parsed.data.classification);

    if (parsed.data.classification !== "concrete_action") {
      rejectedCandidates.push(
        rejectedCandidate({
          candidateIndex,
          classification: parsed.data.classification,
          description: parsed.data.description,
          reason: "non_concrete_classification",
        }),
      );
      continue;
    }

    candidates.push({
      candidateIndex,
      candidate: parsed.data,
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
  const parsed = actionStateEnvelopeSchema.safeParse(input);

  if (!parsed.success) {
    throw parsed.error;
  }

  const candidates = parseCandidates(parsed.data);
  const referencedActionIds = (parsed.data.referenced_action_ids ?? []).flatMap((rawId) => {
    const parsedId = actionIdSchema.safeParse(rawId);

    return parsedId.success ? [parsedId.data] : [];
  });

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
    return hasCurrentUserEvidence(
      candidate,
      input.currentUserStreamEntryIds ?? [input.currentUserStreamEntryId],
    );
  }

  const allowed = allowedEvidenceStreamEntryIds(input);

  return candidate.evidence_stream_entry_ids.some((entryId) => allowed.has(entryId));
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
    open_question_id: input.openQuestionId,
    state: input.candidate.state,
    confidence: input.candidate.confidence,
    provenance_episode_ids: [],
    provenance_stream_entry_ids: provenanceStreamEntryIds,
    created_at: input.nowMs,
    updated_at: input.nowMs,
    considering_at: null,
    committed_at: null,
    scheduled_at: null,
    completed_at: null,
    not_done_at: null,
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

function isTerminalEmissionState(state: ActionState): state is "completed" | "not_done" {
  return ACTION_STATE_METADATA[state].terminal;
}

function isActiveTerminalTransitionTarget(state: ActionState): boolean {
  return ACTION_STATE_METADATA[state].active;
}

function mergeUniqueIds<T extends string>(left: readonly T[], right: readonly T[]): T[] {
  return [...new Set([...left, ...right])];
}

type TerminalClosureState = "completed" | "not_done";
type TerminalClosurePatch = Omit<ActionRecordPatch, "state">;
type TerminalClosureRepository = Pick<ActionRepository, "update"> &
  Partial<Pick<ActionRepository, "get">>;
type TerminalClosureLifecycleResult = LifecycleOperationResult<{
  actionId: ActionId;
  previous: ActionRecord | null;
}>;

function closeActionThroughLifecycle(input: {
  actionId: ActionId;
  state: TerminalClosureState;
  repository: TerminalClosureRepository;
  patch: TerminalClosurePatch;
}): TerminalClosureLifecycleResult {
  return input.state === "completed"
    ? completeAction({
        actionId: input.actionId,
        repository: input.repository,
        patch: input.patch,
      })
    : markActionNotDone({
        actionId: input.actionId,
        repository: input.repository,
        patch: input.patch,
      });
}

function skippedReasonFromLifecycleResult(
  result: Exclude<TerminalClosureLifecycleResult, { status: "success" }>,
): ActionStateSkippedReason {
  return result.status === "conflict" ? "lifecycle_conflict" : "lifecycle_no_op";
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
  dedupSkippedEmbedding?: number;
  dedupDegraded?: number;
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
    action_persistence_dedup_skipped_embedding: options.dedupSkippedEmbedding ?? 0,
    action_persistence_dedup_degraded: options.dedupDegraded ?? 0,
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

type ActionDedupAxis = Pick<
  ActionRecord,
  "actor" | "audience_entity_id" | "goal_id" | "open_question_id"
>;

function actionAxisKey(record: ActionDedupAxis): string {
  return JSON.stringify([
    record.actor,
    record.audience_entity_id,
    record.goal_id,
    record.open_question_id,
  ]);
}

function sameActionDedupAxis(left: ActionDedupAxis, right: ActionDedupAxis): boolean {
  return (
    left.actor === right.actor &&
    left.audience_entity_id === right.audience_entity_id &&
    left.goal_id === right.goal_id &&
    left.open_question_id === right.open_question_id
  );
}

function traceDedupSkippedEmbedding(options: {
  tracer?: TurnTracer;
  turnId?: string;
  sessionId?: SessionId;
  candidate: ActionCandidateRejected;
  matchedActionId?: ActionId;
  similarity: number;
}): void {
  if (options.tracer?.enabled !== true || options.turnId === undefined) {
    return;
  }

  options.tracer.emit("action_persistence.dedup.skipped", {
    turnId: options.turnId,
    ...(options.sessionId !== undefined ? { session_id: options.sessionId } : {}),
    classification: options.candidate.classification,
    description_excerpt: options.candidate.description_excerpt,
    reason: options.candidate.reason,
    similarity: options.similarity,
    ...(options.matchedActionId === undefined
      ? {}
      : { matched_action_id: options.matchedActionId }),
  });
}

function traceDedupDegraded(options: {
  tracer?: TurnTracer;
  turnId?: string;
  sessionId?: SessionId;
  reason: string;
  error: unknown;
  candidateDescription?: string;
}): void {
  if (options.tracer?.enabled !== true || options.turnId === undefined) {
    return;
  }

  options.tracer.emit("action_persistence.dedup.degraded", {
    turnId: options.turnId,
    ...(options.sessionId !== undefined ? { session_id: options.sessionId } : {}),
    reason: options.reason,
    error: options.error instanceof Error ? options.error.message : String(options.error),
    ...(options.candidateDescription === undefined
      ? {}
      : { description_excerpt: descriptionExcerpt(options.candidateDescription) }),
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

  async extract(input: ExtractActionStatesInput): Promise<ActionRecord[]> {
    const sessionId = input.sessionId ?? this.options.sessionId;

    if (this.options.llmClient === undefined || this.options.model === undefined) {
      return this.degradedWithTrace("llm_unavailable", undefined, sessionId);
    }

    if (this.options.actionRepository === undefined) {
      return this.degradedWithTrace("repository_unavailable", undefined, sessionId);
    }

    const messages = buildActionStateMessages(input);
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
      const reason = degradedReasonForStructuredToolError(error);

      if (reason === "llm_failed") {
        return this.degradedWithTrace(
          "llm_failed",
          isStructuredToolCallError(error, "llm_failed") ? (error.cause ?? error) : error,
          sessionId,
        );
      }

      const degradedError =
        isStructuredToolCallError(error, "missing_tool_call")
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
      incrementSkippedReason(
        skippedReasons,
        rejected.reason === "embedding_dedup" ? "embedding_dedup" : "non_concrete_classification",
      );
    }
    let degraded = false;
    let dedupSkippedEmbedding = 0;
    let dedupDegraded = 0;
    let terminalEmissionClosures = 0;
    let selfPerformanceClosures = 0;
    const nowMs = this.clock.now();
    const turnCounter = input.turnCounter ?? null;
    const dedupStates = new Map<string, ActionEmbeddingDedupState | null>();
    const allowedReferencedActionIds = new Set(
      [
        ...(input.activeActionsForReference ?? []),
        ...(input.postTurnSelfPerformance?.activeBorgActions ?? []),
      ].map((action) => action.id),
    );

    if (this.options.actionRepository.update !== undefined && allowedReferencedActionIds.size > 0) {
      for (const actionId of parsed.referencedActionIds) {
        if (!allowedReferencedActionIds.has(actionId)) {
          continue;
        }

        try {
          this.options.actionRepository.update(actionId, {
            last_referenced_at_ms: nowMs,
            last_referenced_turn_counter: turnCounter,
            last_referenced_turn_global: turnCounter,
          });
        } catch (error) {
          degraded = true;
          await this.degraded("repository_failed", error);
        }
      }
    }

    const getEmbeddingDedupState = async (
      record: ActionRecord,
    ): Promise<ActionEmbeddingDedupState | null> => {
      if (
        this.options.embeddingClient === undefined ||
        this.options.actionRepository?.list === undefined
      ) {
        return null;
      }

      const axisKey = actionAxisKey(record);

      if (dedupStates.has(axisKey)) {
        return dedupStates.get(axisKey) ?? null;
      }

      try {
        const activeActions = this.options.actionRepository
          .list({
            states: ACTIVE_ACTION_STATES,
            actor: record.actor,
            audienceEntityId: record.audience_entity_id,
          })
          .filter((action) => sameActionDedupAxis(action, record));
        const embeddings =
          activeActions.length === 0
            ? []
            : await this.options.embeddingClient.embedBatch(
                activeActions.map((action) => action.description),
              );
        const state = {
          activeVectors: activeActions.flatMap((action, index) => {
            const vector = embeddings[index];
            return vector === undefined ? [] : [{ actionId: action.id, vector, record: action }];
          }),
          acceptedVectors: [],
        } satisfies ActionEmbeddingDedupState;

        dedupStates.set(axisKey, state);
        return state;
      } catch (error) {
        degraded = true;
        dedupDegraded += 1;
        traceDedupDegraded({
          tracer: this.options.tracer,
          turnId: this.options.turnId,
          sessionId,
          reason: "active_action_embedding_failed",
          error,
        });
        dedupStates.set(axisKey, null);
        return null;
      }
    };

    for (const parsedCandidate of parsed.candidates) {
      const candidate = parsedCandidate.candidate;
      if (!hasAllowedEvidence(candidate, input)) {
        incrementSkippedReason(skippedReasons, "missing_current_user_evidence");
        incrementSkippedCandidate(
          skippedCandidates,
          parsedCandidate.candidateIndex,
          "missing_current_user_evidence",
        );
        continue;
      }

      const matchedExistingActionId = candidate.matched_existing_action_id ?? null;
      if (
        matchedExistingActionId !== null &&
        input.postTurnSelfPerformance !== undefined &&
        isTerminalEmissionState(candidate.state)
      ) {
        const activeBorgAction = input.postTurnSelfPerformance.activeBorgActions.find(
          (action) => action.id === matchedExistingActionId,
        );

        if (activeBorgAction !== undefined && this.options.actionRepository.update !== undefined) {
          const patch = {
            confidence: Math.max(activeBorgAction.confidence, candidate.confidence),
            provenance_stream_entry_ids: mergeUniqueIds(
              activeBorgAction.provenance_stream_entry_ids,
              allowedCandidateEvidenceStreamEntryIds(candidate, input),
            ),
            updated_at: nowMs,
            last_referenced_at_ms: nowMs,
            last_referenced_turn_counter: turnCounter,
            last_referenced_turn_global: turnCounter,
            ...stateTimestampPatch(candidate.state, nowMs),
          } satisfies TerminalClosurePatch;

          try {
            const lifecycleResult = closeActionThroughLifecycle({
              actionId: matchedExistingActionId,
              state: candidate.state,
              repository: this.options.actionRepository as TerminalClosureRepository,
              patch,
            });

            if (lifecycleResult.status !== "success") {
              const reason = skippedReasonFromLifecycleResult(lifecycleResult);

              incrementSkippedReason(skippedReasons, reason);
              incrementSkippedCandidate(skippedCandidates, parsedCandidate.candidateIndex, reason);
              continue;
            }

            selfPerformanceClosures += 1;
            terminalEmissionClosures += 1;
            traceBorgSelfPerformanceClosure({
              tracer: this.options.tracer,
              turnId: this.options.turnId,
              sessionId,
              candidateIndex: parsedCandidate.candidateIndex,
              matchedActionId: matchedExistingActionId,
              terminalState: candidate.state,
              description: candidate.description,
            });
            continue;
          } catch (error) {
            degraded = true;
            incrementSkippedReason(skippedReasons, "repository_failed");
            incrementSkippedCandidate(
              skippedCandidates,
              parsedCandidate.candidateIndex,
              "repository_failed",
            );
            await this.degraded("repository_failed", error);
            continue;
          }
        }

        if (input.persistNewActions === false) {
          continue;
        }
      }

      if (input.persistNewActions === false) {
        continue;
      }

      const record = toActionRecord({
        candidate,
        currentUserStreamEntryIds: input.currentUserStreamEntryIds ?? [
          input.currentUserStreamEntryId,
        ],
        audienceEntityId: input.audienceEntityId,
        sessionId: input.sessionId ?? null,
        speakerEntityId: input.speakerEntityId ?? null,
        senderAttribution: input.senderAttribution,
        goalId: input.goalId ?? null,
        openQuestionId: input.openQuestionId ?? null,
        nowMs,
        turnCounter,
      });
      const dedupState = await getEmbeddingDedupState(record);
      let candidateVector: Float32Array | null = null;

      if (dedupState !== null && this.options.embeddingClient !== undefined) {
        try {
          candidateVector = await this.options.embeddingClient.embed(record.description);
        } catch (error) {
          degraded = true;
          dedupDegraded += 1;
          traceDedupDegraded({
            tracer: this.options.tracer,
            turnId: this.options.turnId,
            sessionId,
            reason: "candidate_embedding_failed",
            error,
            candidateDescription: record.description,
          });
        }

        if (candidateVector !== null) {
          let bestMatch: { actionId: ActionId; similarity: number; record: ActionRecord } | null =
            null;

          for (const existing of [...dedupState.activeVectors, ...dedupState.acceptedVectors]) {
            const similarity = cosineSimilarity(candidateVector, existing.vector);

            if (
              similarity >= ACTION_PERSISTENCE_DUPLICATE_SIMILARITY_THRESHOLD &&
              (bestMatch === null || similarity > bestMatch.similarity)
            ) {
              bestMatch = {
                actionId: existing.actionId,
                similarity,
                record: existing.record,
              };
            }
          }

          if (bestMatch !== null) {
            let persistTerminalWithoutUpdate = false;
            if (
              isTerminalEmissionState(record.state) &&
              isActiveTerminalTransitionTarget(bestMatch.record.state)
            ) {
              if (this.options.actionRepository.update === undefined) {
                // Repository fakes that only support add/list cannot retire the predecessor;
                // fall through so the terminal evidence is still persisted.
                persistTerminalWithoutUpdate = true;
              } else {
                const patch = {
                  confidence: Math.max(bestMatch.record.confidence, record.confidence),
                  provenance_episode_ids: mergeUniqueIds(
                    bestMatch.record.provenance_episode_ids,
                    record.provenance_episode_ids,
                  ),
                  provenance_stream_entry_ids: mergeUniqueIds(
                    bestMatch.record.provenance_stream_entry_ids,
                    record.provenance_stream_entry_ids,
                  ),
                  updated_at: nowMs,
                  last_referenced_at_ms: nowMs,
                  last_referenced_turn_counter: turnCounter,
                  last_referenced_turn_global: turnCounter,
                  ...stateTimestampPatch(record.state, nowMs),
                } satisfies TerminalClosurePatch;

                try {
                  const lifecycleResult = closeActionThroughLifecycle({
                    actionId: bestMatch.actionId,
                    state: record.state,
                    repository: this.options.actionRepository as TerminalClosureRepository,
                    patch,
                  });

                  if (lifecycleResult.status !== "success") {
                    const reason = skippedReasonFromLifecycleResult(lifecycleResult);

                    incrementSkippedReason(skippedReasons, reason);
                    incrementSkippedCandidate(
                      skippedCandidates,
                      parsedCandidate.candidateIndex,
                      reason,
                    );
                    continue;
                  }

                  Object.assign(bestMatch.record, patch, { state: record.state });
                  terminalEmissionClosures += 1;
                  traceTerminalEmissionClosure({
                    tracer: this.options.tracer,
                    turnId: this.options.turnId,
                    sessionId,
                    candidateIndex: parsedCandidate.candidateIndex,
                    matchedActionId: bestMatch.actionId,
                    terminalState: record.state,
                    description: record.description,
                    similarity: bestMatch.similarity,
                  });
                  continue;
                } catch (error) {
                  degraded = true;
                  incrementSkippedReason(skippedReasons, "repository_failed");
                  incrementSkippedCandidate(
                    skippedCandidates,
                    parsedCandidate.candidateIndex,
                    "repository_failed",
                  );
                  await this.degraded("repository_failed", error);
                  continue;
                }
              }
            }

            if (!persistTerminalWithoutUpdate && bestMatch.record.state !== "unknown") {
              if (this.options.actionRepository.update !== undefined) {
                try {
                  const referencePatch = {
                    last_referenced_at_ms: nowMs,
                    last_referenced_turn_counter: turnCounter,
                    last_referenced_turn_global: turnCounter,
                  } satisfies ActionRecordPatch;

                  this.options.actionRepository.update(bestMatch.actionId, referencePatch);
                  Object.assign(bestMatch.record, referencePatch, { updated_at: nowMs });
                } catch (error) {
                  degraded = true;
                  traceDedupDegraded({
                    tracer: this.options.tracer,
                    turnId: this.options.turnId,
                    sessionId,
                    reason: "reference_touch_failed",
                    error,
                    candidateDescription: record.description,
                  });
                }
              }

              const rejection = rejectedCandidate({
                candidateIndex: parsedCandidate.candidateIndex,
                classification: "concrete_action",
                description: record.description,
                reason: "embedding_dedup",
              });

              dedupSkippedEmbedding += 1;
              incrementSkippedReason(skippedReasons, "embedding_dedup");
              incrementSkippedCandidate(
                skippedCandidates,
                parsedCandidate.candidateIndex,
                "embedding_dedup",
              );
              rejectedCandidates.push(rejection);
              traceDedupSkippedEmbedding({
                tracer: this.options.tracer,
                turnId: this.options.turnId,
                sessionId,
                candidate: rejection,
                matchedActionId: bestMatch.actionId,
                similarity: bestMatch.similarity,
              });
              continue;
            }
          }
        }
      }

      try {
        this.options.actionRepository.add(record, { creationSource: "extractor" });
        persisted.push(record);
        if (dedupState !== null && candidateVector !== null) {
          dedupState.acceptedVectors.push({
            actionId: record.id,
            vector: candidateVector,
            record,
          });
        }
      } catch (error) {
        degraded = true;
        incrementSkippedReason(skippedReasons, "repository_failed");
        incrementSkippedCandidate(
          skippedCandidates,
          parsedCandidate.candidateIndex,
          "repository_failed",
        );
        await this.degraded("repository_failed", error);
      }
    }

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
      dedupSkippedEmbedding,
      dedupDegraded,
      terminalEmissionClosures,
      selfPerformanceClosures,
      degraded,
    });

    return persisted;
  }
}

function traceTerminalEmissionClosure(options: {
  tracer?: TurnTracer;
  turnId?: string;
  sessionId?: SessionId;
  candidateIndex: number;
  matchedActionId: ActionId;
  terminalState: "completed" | "not_done";
  description: string;
  similarity: number;
}): void {
  if (options.tracer?.enabled !== true || options.turnId === undefined) {
    return;
  }

  options.tracer.emit("action_state.transitioned", {
    turnId: options.turnId,
    ...(options.sessionId !== undefined ? { session_id: options.sessionId } : {}),
    action_id: options.matchedActionId,
    candidate_index: options.candidateIndex,
    terminal_state: options.terminalState,
    description_excerpt: descriptionExcerpt(options.description),
    similarity: options.similarity,
  });
}

function traceBorgSelfPerformanceClosure(options: {
  tracer?: TurnTracer;
  turnId?: string;
  sessionId?: SessionId;
  candidateIndex: number;
  matchedActionId: ActionId;
  terminalState: "completed" | "not_done";
  description: string;
}): void {
  if (options.tracer?.enabled !== true || options.turnId === undefined) {
    return;
  }

  options.tracer.emit("action_state.borg_self_performance.completed", {
    turnId: options.turnId,
    ...(options.sessionId !== undefined ? { session_id: options.sessionId } : {}),
    action_id: options.matchedActionId,
    candidate_index: options.candidateIndex,
    terminal_state: options.terminalState,
    description_excerpt: descriptionExcerpt(options.description),
  });
}

export { ACTION_STATE_TOOL_NAME };
