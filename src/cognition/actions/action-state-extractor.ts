import { z } from "zod";

import {
  type LLMClient,
  type LLMCompleteResult,
  type LLMMessage,
  type LLMToolDefinition,
  toToolInputSchema,
} from "../../llm/index.js";
import type { EmbeddingClient } from "../../embeddings/index.js";
import {
  ACTION_STATES,
  actionEntityIdSchema,
  actionActorSchema,
  type ActionRecord,
  type ActionRepository,
  type ActionState,
} from "../../memory/actions/index.js";
import { cosineSimilarity } from "../../retrieval/embedding-similarity.js";
import { SystemClock, type Clock } from "../../util/clock.js";
import type { JsonValue } from "../../util/json-value.js";
import {
  createActionId,
  type ActionId,
  type EntityId,
  type GoalId,
  type OpenQuestionId,
  type StreamEntryId,
} from "../../util/ids.js";
import { BORG_HOST_CAPABILITY_BOUNDARY_PROMPT } from "../host-capabilities.js";
import type { RecencyMessage } from "../recency/index.js";
import { buildUsageTraceBlock, type TurnTracer } from "../tracing/tracer.js";

const ACTION_STATE_TOOL_NAME = "EmitActionStates";
const ACTION_PERSISTENCE_DUPLICATE_SIMILARITY_THRESHOLD = 0.85;
// v62 P3 action classification fields made the old 768 cap truncate in v63 runs.
const ACTION_STATE_EXTRACTOR_MAX_TOKENS = 1536;
const ACTIVE_ACTION_STATES: readonly ActionState[] = [
  "considering",
  "committed_to_do",
  "scheduled",
];

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
    evidence_stream_entry_ids: z.array(z.string().min(1)),
    confidence: z.number().min(0).max(1),
  })
  .strict();

const actionStateOutputSchema = z
  .object({
    action_states: z
      .array(actionStateCandidateSchema)
      .describe(
        "Action-state assertions in the current user message. Emit an empty array when none are present.",
      ),
  })
  .strict();

const actionStateEnvelopeSchema = z
  .object({
    action_states: z.array(z.unknown()),
  })
  .strict();

const ACTION_STATE_TOOL = {
  name: ACTION_STATE_TOOL_NAME,
  description:
    "Extract action states asserted by the current user message, citing the current user stream entry.",
  inputSchema: toToolInputSchema(actionStateOutputSchema),
} satisfies LLMToolDefinition;

const ACTION_STATE_SYSTEM_PROMPT = [
  "Extract action-state assertions from the current user message.",
  "Use recent_history only to understand elliptical references. The evidence must be in current_user_message, and every emitted item must cite current_user_stream_entry_id.",
  "Emit an empty action_states array when the current user message contains no action-state assertion.",
  "Do NOT emit action records for messages about the conversation frame, roleplay, system prompt, or the agent's own prior behavior. Action records are for user-world actions only.",
  "Judge semantic intent across languages. Do not rely on wording, punctuation, capitalization, or phrase shapes.",
  "When speaker_entity_id is supplied and the current speaker asserts a first-person action, set actor to that speaker_entity_id. Use actor=user only when no speaker entity is supplied. Use actor=borg only for actions Borg is responsible for.",
  "Borg-owned actions must stay inside the host capability boundary below.",
  BORG_HOST_CAPABILITY_BOUNDARY_PROMPT,
  "In group chat, first-person user actions belong to the current sender, not the group, unless the message explicitly says the group is acting.",
  "Set audience_entity_id only when the current message clearly scopes the action to a supplied audience; otherwise use null so Borg can default it to the current audience.",
  "",
  "Classifications:",
  "- concrete_action: a discrete task someone (Borg, the user, a participant, or a third party) will do, has done, is doing, or is considering doing. Has a clear actor and a clear thing being done.",
  '- conversational_acknowledgment: a remark about state, mood, or transition that is not a task, such as "going to sleep", "heading back", "got it", or "thanks". Not memory-worthy as an action.',
  '- decision_or_preference: a settled decision or preference belongs to the decision artifact or commitments, not as a standalone action, such as "lock the service as the anchor", "avoid one-off handoffs", or "we prefer evenings".',
  "- already_represented: covered by an existing active action, commitment, or goal already in memory.",
  '- outside_borg_capability: a Borg-owned action that would require external document editing, production monitoring, scheduled future work, proactive outbound messaging, unwired tool execution, physical action, payment, or real-world attendance, such as "I\'ll seed the postmortem doc by morning", "I\'ll monitor p95", or "I\'ll send the reminder later".',
  "- none: not memory-worthy at all.",
  "",
  "States:",
  "- considering: the user is weighing or contemplating an action, not committing.",
  "- committed_to_do: the user says they will do something or intends to do it.",
  "- scheduled: the action is arranged for a time, appointment, or calendar-like slot.",
  "- completed: the user says the action was done, booked, sent, finished, or carried out.",
  "- not_done: the user says the action has not happened, was abandoned, or was not completed.",
  "",
  "Examples:",
  '- "I sent the review note" -> completed.',
  '- "I\'ll review the pull request this weekend" -> committed_to_do.',
  '- "Design review Tuesday 7pm" -> scheduled when the message arranges a future slot.',
  '- "The release checklist is done" -> completed.',
  '- "Yeah, I haven\'t gotten to it" -> not_done.',
  '- "Maybe I should send the card" -> considering.',
  "Return only the required tool call.",
].join("\n");

type ActionStateEnvelopeInput = z.infer<typeof actionStateEnvelopeSchema>;
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
  actionRepository?: Pick<ActionRepository, "add"> & Partial<Pick<ActionRepository, "list">>;
  embeddingClient?: EmbeddingClient;
  clock?: Clock;
  tracer?: TurnTracer;
  turnId?: string;
  onDegraded?: (
    reason: ActionStateExtractorDegradedReason,
    error?: unknown,
  ) => Promise<void> | void;
};

export type ExtractActionStatesInput = {
  userMessage: string;
  currentUserStreamEntryId: StreamEntryId;
  recentHistory: readonly RecencyMessage[];
  audienceEntityId: EntityId | null;
  speakerEntityId?: EntityId | null;
  speakerDisplayName?: string | null;
  goalId?: GoalId | null;
  openQuestionId?: OpenQuestionId | null;
};

function buildActionStateMessages(input: ExtractActionStatesInput): LLMMessage[] {
  return [
    {
      role: "user",
      content: JSON.stringify({
        current_user_message: input.userMessage,
        current_user_stream_entry_id: input.currentUserStreamEntryId,
        recent_history: input.recentHistory.slice(-8).map((message) => ({
          role: message.role,
          content: message.content,
        })),
        audience_entity_id: input.audienceEntityId,
        speaker_entity_id: input.speakerEntityId ?? null,
        speaker_display_name: input.speakerDisplayName ?? null,
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

function parseResponse(result: LLMCompleteResult): ActionStateParseResult {
  const call = result.tool_calls.find((toolCall) => toolCall.name === ACTION_STATE_TOOL_NAME);

  if (call === undefined) {
    throw new MissingActionStateToolCallError(
      `Action state extractor did not emit ${ACTION_STATE_TOOL_NAME}`,
    );
  }

  const parsed = actionStateEnvelopeSchema.safeParse(call.input);

  if (!parsed.success) {
    throw parsed.error;
  }

  const candidates = parseCandidates(parsed.data);

  return {
    candidates: candidates.candidates,
    candidatesEmitted: parsed.data.action_states.length,
    validCandidateCount: candidates.candidates.length + candidates.rejectedCandidates.length,
    skippedCandidates: candidates.skippedCandidates,
    classificationCounts: candidates.classificationCounts,
    rejectedCandidates: candidates.rejectedCandidates,
  };
}

function hasCurrentUserEvidence(
  candidate: ParsedActionStateCandidate,
  currentUserStreamEntryId: StreamEntryId,
): boolean {
  return candidate.evidence_stream_entry_ids.some(
    (entryId) => entryId === currentUserStreamEntryId,
  );
}

function stateTimestampPatch(
  state: ActionState,
  timestamp: number,
): Partial<
  Pick<
    ActionRecord,
    | "considering_at"
    | "committed_at"
    | "scheduled_at"
    | "completed_at"
    | "not_done_at"
    | "unknown_at"
  >
> {
  switch (state) {
    case "considering":
      return { considering_at: timestamp };
    case "committed_to_do":
      return { committed_at: timestamp };
    case "scheduled":
      return { scheduled_at: timestamp };
    case "completed":
      return { completed_at: timestamp };
    case "not_done":
      return { not_done_at: timestamp };
    case "unknown":
      return { unknown_at: timestamp };
  }
}

function toActionRecord(input: {
  candidate: ParsedActionStateCandidate;
  currentUserStreamEntryId: StreamEntryId;
  audienceEntityId: EntityId | null;
  speakerEntityId: EntityId | null;
  goalId: GoalId | null;
  openQuestionId: OpenQuestionId | null;
  nowMs: number;
}): ActionRecord {
  return {
    id: createActionId(),
    description: input.candidate.description,
    actor:
      input.candidate.actor === "user" && input.speakerEntityId !== null
        ? input.speakerEntityId
        : input.candidate.actor,
    audience_entity_id: input.candidate.audience_entity_id ?? input.audienceEntityId,
    goal_id: input.goalId,
    open_question_id: input.openQuestionId,
    state: input.candidate.state,
    confidence: input.candidate.confidence,
    provenance_episode_ids: [],
    provenance_stream_entry_ids: [input.currentUserStreamEntryId],
    created_at: input.nowMs,
    updated_at: input.nowMs,
    considering_at: null,
    committed_at: null,
    scheduled_at: null,
    completed_at: null,
    not_done_at: null,
    unknown_at: null,
    canonicalized_by_artifact_entry_id: null,
    ...stateTimestampPatch(input.candidate.state, input.nowMs),
  };
}

function summarizeActionStateResponseShape(response: LLMCompleteResult): JsonValue {
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
      label: "action_state_extractor",
      model: options.model,
      promptCharCount: countCompletePromptChars(ACTION_STATE_SYSTEM_PROMPT, options.messages),
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
      label: "action_state_extractor",
      responseShape: summarizeActionStateResponseShape(options.response),
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
      label: "action_state_extractor",
      responseShape: {
        error: options.error instanceof Error ? options.error.message : String(options.error),
      },
      stopReason: null,
      usage: null,
    });
  }
}

function zeroActionStateCounts(): Record<ActionState, number> {
  return Object.fromEntries(ACTION_STATES.map((state) => [state, 0])) as Record<
    ActionState,
    number
  >;
}

function traceExtractorCompleted(options: {
  tracer?: TurnTracer;
  turnId?: string;
  candidatesEmitted: number;
  validCandidateCount?: number;
  persisted: readonly ActionRecord[];
  skippedReasons: ReadonlyMap<ActionStateSkippedReason, number>;
  skippedCandidates?: readonly ActionStateSkippedCandidate[];
  classificationCounts?: Record<ActionCandidateClassificationCountKey, number>;
  rejectedCandidates?: readonly ActionCandidateRejected[];
  dedupSkippedEmbedding?: number;
  dedupDegraded?: number;
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

    options.tracer.emit("action_candidate_classification_rejected", {
      turnId: options.turnId,
      classification: rejection.classification,
      description_excerpt: rejection.description_excerpt,
      reason: rejection.reason,
    });
  }

  options.tracer.emit("action_state_extractor_completed", {
    turnId: options.turnId,
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
  candidate: ActionCandidateRejected;
  matchedActionId?: ActionId;
  similarity: number;
}): void {
  if (options.tracer?.enabled !== true || options.turnId === undefined) {
    return;
  }

  options.tracer.emit("action_persistence_dedup_skipped_embedding", {
    turnId: options.turnId,
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
  reason: string;
  error: unknown;
  candidateDescription?: string;
}): void {
  if (options.tracer?.enabled !== true || options.turnId === undefined) {
    return;
  }

  options.tracer.emit("action_persistence_dedup_degraded", {
    turnId: options.turnId,
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
  ): Promise<ActionRecord[]> {
    const result = await this.degraded(reason, error);

    traceExtractorCompleted({
      tracer: this.options.tracer,
      turnId: this.options.turnId,
      candidatesEmitted: 0,
      persisted: result,
      skippedReasons: new Map(),
      degraded: true,
    });

    return result;
  }

  async extract(input: ExtractActionStatesInput): Promise<ActionRecord[]> {
    if (this.options.llmClient === undefined || this.options.model === undefined) {
      return this.degradedWithTrace("llm_unavailable");
    }

    if (this.options.actionRepository === undefined) {
      return this.degradedWithTrace("repository_unavailable");
    }

    const messages = buildActionStateMessages(input);
    const tools = [ACTION_STATE_TOOL];

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
        system: ACTION_STATE_SYSTEM_PROMPT,
        messages,
        tools,
        tool_choice: { type: "tool", name: ACTION_STATE_TOOL_NAME },
        max_tokens: ACTION_STATE_EXTRACTOR_MAX_TOKENS,
        budget: "action-state-extractor",
      });
    } catch (error) {
      traceLlmCallError({
        tracer: this.options.tracer,
        turnId: this.options.turnId,
        error,
      });

      return this.degradedWithTrace("llm_failed", error);
    }

    traceLlmCallResponse({
      tracer: this.options.tracer,
      turnId: this.options.turnId,
      response,
    });

    let parsed: ActionStateParseResult;

    try {
      parsed = parseResponse(response);
    } catch (error) {
      const reason =
        error instanceof MissingActionStateToolCallError
          ? "missing_tool_call"
          : error instanceof z.ZodError
            ? "invalid_payload"
            : "llm_failed";
      const result = await this.degraded(reason, error);

      traceExtractorCompleted({
        tracer: this.options.tracer,
        turnId: this.options.turnId,
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
    const nowMs = this.clock.now();
    const dedupStates = new Map<string, ActionEmbeddingDedupState | null>();

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
            return vector === undefined ? [] : [{ actionId: action.id, vector }];
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
          reason: "active_action_embedding_failed",
          error,
        });
        dedupStates.set(axisKey, null);
        return null;
      }
    };

    for (const parsedCandidate of parsed.candidates) {
      const candidate = parsedCandidate.candidate;
      if (!hasCurrentUserEvidence(candidate, input.currentUserStreamEntryId)) {
        incrementSkippedReason(skippedReasons, "missing_current_user_evidence");
        incrementSkippedCandidate(
          skippedCandidates,
          parsedCandidate.candidateIndex,
          "missing_current_user_evidence",
        );
        continue;
      }

      const record = toActionRecord({
        candidate,
        currentUserStreamEntryId: input.currentUserStreamEntryId,
        audienceEntityId: input.audienceEntityId,
        speakerEntityId: input.speakerEntityId ?? null,
        goalId: input.goalId ?? null,
        openQuestionId: input.openQuestionId ?? null,
        nowMs,
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
            reason: "candidate_embedding_failed",
            error,
            candidateDescription: record.description,
          });
        }

        if (candidateVector !== null) {
          let bestMatch: { actionId: ActionId; similarity: number } | null = null;

          for (const existing of [...dedupState.activeVectors, ...dedupState.acceptedVectors]) {
            const similarity = cosineSimilarity(candidateVector, existing.vector);

            if (
              similarity >= ACTION_PERSISTENCE_DUPLICATE_SIMILARITY_THRESHOLD &&
              (bestMatch === null || similarity > bestMatch.similarity)
            ) {
              bestMatch = {
                actionId: existing.actionId,
                similarity,
              };
            }
          }

          if (bestMatch !== null) {
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
              candidate: rejection,
              matchedActionId: bestMatch.actionId,
              similarity: bestMatch.similarity,
            });
            continue;
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
      candidatesEmitted: parsed.candidatesEmitted,
      validCandidateCount: parsed.validCandidateCount,
      persisted,
      skippedReasons,
      skippedCandidates,
      classificationCounts: parsed.classificationCounts,
      rejectedCandidates,
      dedupSkippedEmbedding,
      dedupDegraded,
      degraded,
    });

    return persisted;
  }
}

export { ACTION_STATE_TOOL_NAME };
