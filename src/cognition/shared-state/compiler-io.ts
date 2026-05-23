import type { LLMCompleteResult, LLMMessage, LLMToolDefinition } from "../../llm/index.js";
import type {
  SharedStateArtifact,
  SharedStateOperation,
} from "../../memory/decision-artifacts/index.js";
import type { EntityId, StreamEntryId } from "../../util/ids.js";
import type { JsonValue } from "../../util/json-value.js";
import { SHARED_STATE_SYSTEM_PROMPT } from "../prompts/shared-state.js";
import { buildUsageTraceBlock, toTraceJsonValue, type TurnTracer } from "../tracing/tracer.js";
import { summarizeSharedStateArtifactRender, type SharedStateRenderOptions } from "./render.js";
import { similarStateKeyClusterCount } from "./state-key.js";
import type {
  SharedStateReconciliationResult,
  SharedStateUnsettledReconciliationSummary,
} from "./reconciliation.js";
import { type SharedStateArtifactPromptBudget } from "./compiler-prompt.js";
import {
  MissingSharedStateArtifactToolCallError,
  SHARED_STATE_PROMPT_WARNING_TOKEN_THRESHOLD,
  SHARED_STATE_ACCEPTED_TOOL_NAMES,
  SHARED_STATE_TOOL_NAME,
  sharedStatePatchSchema,
  type CanonicalizationDuplicateDrop,
  type EmitSharedStatePatch,
  type NonLockedCanonicalizesDrop,
  type PatchRejection,
  type SharedStateLedgerMode,
} from "./schema.js";

export function parseResponse(result: LLMCompleteResult): EmitSharedStatePatch {
  const acceptedToolNames = new Set<string>(SHARED_STATE_ACCEPTED_TOOL_NAMES);
  const call = result.tool_calls.find((toolCall) => acceptedToolNames.has(toolCall.name));

  if (call === undefined) {
    throw new MissingSharedStateArtifactToolCallError(
      `Shared state compiler did not emit ${SHARED_STATE_TOOL_NAME}`,
    );
  }

  const parsed = sharedStatePatchSchema.safeParse(call.input);

  if (!parsed.success) {
    throw parsed.error;
  }

  return parsed.data;
}

function summarizeSharedStateArtifactResponseShape(response: LLMCompleteResult): JsonValue {
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

export function traceLlmCallStarted(options: {
  tracer?: TurnTracer;
  turnId?: string;
  model: string;
  messages: readonly LLMMessage[];
  tools: readonly LLMToolDefinition[];
}): void {
  if (options.tracer?.enabled === true && options.turnId !== undefined) {
    options.tracer.emit("llm_call.started", {
      turnId: options.turnId,
      label: "decision_artifact_compiler",
      model: options.model,
      promptCharCount: countCompletePromptChars(SHARED_STATE_SYSTEM_PROMPT, options.messages),
      toolSchemas: summarizeToolSchemas(options.tools),
    });
  }
}

export function traceLlmCallResponse(options: {
  tracer?: TurnTracer;
  turnId?: string;
  response: LLMCompleteResult;
}): void {
  if (options.tracer?.enabled === true && options.turnId !== undefined) {
    options.tracer.emit("llm_call.completed", {
      turnId: options.turnId,
      label: "decision_artifact_compiler",
      responseShape: summarizeSharedStateArtifactResponseShape(options.response),
      stopReason: options.response.stop_reason,
      usage: buildUsageTraceBlock(options.response),
    });
  }
}

export function traceLlmCallError(options: {
  tracer?: TurnTracer;
  turnId?: string;
  error: unknown;
}): void {
  if (options.tracer?.enabled === true && options.turnId !== undefined) {
    options.tracer.emit("llm_call.completed", {
      turnId: options.turnId,
      label: "decision_artifact_compiler",
      responseShape: {
        error: options.error instanceof Error ? options.error.message : String(options.error),
      },
      stopReason: null,
      usage: null,
    });
  }
}

export function traceCompileCompleted(options: {
  tracer?: TurnTracer;
  turnId?: string;
  audienceEntityId: EntityId;
  previousEntryCount: number;
  operationCount: number;
  rejected: readonly PatchRejection[];
  applied: boolean;
  artifact: SharedStateArtifact | null;
  renderOptions?: SharedStateRenderOptions;
  currentTurnCounter?: number;
  currentUserStreamEntryId?: StreamEntryId;
  maxActiveEntries?: number;
  prunedEntryCountThisTurn: number;
  supersededEntryCountThisTurn: number;
  operationCountsByKind?: Record<SharedStateOperation["type"], number>;
  operationCountsByStateKey?: Record<string, Record<SharedStateOperation["type"], number>>;
  newStateKeys?: readonly string[];
  ledgerMode: SharedStateLedgerMode;
  promptBudget: SharedStateArtifactPromptBudget;
  nonLockedCanonicalizesDrops?: readonly NonLockedCanonicalizesDrop[];
}): void {
  const renderOptions =
    options.currentTurnCounter === undefined || options.currentUserStreamEntryId === undefined
      ? options.renderOptions
      : {
          ...(options.renderOptions ?? {}),
          currentTurnCounter:
            options.renderOptions?.currentTurnCounter ?? options.currentTurnCounter,
          currentUserStreamEntryId:
            options.renderOptions?.currentUserStreamEntryId ?? options.currentUserStreamEntryId,
          lastUpdatedTurnByStreamEntryId: {
            ...(options.renderOptions?.lastUpdatedTurnByStreamEntryId ?? {}),
            [options.currentUserStreamEntryId]: options.currentTurnCounter,
          },
        };
  const artifactSummary = summarizeSharedStateArtifactRender(options.artifact, renderOptions);
  const activeEntryCountsByKey = artifactSummary.activeEntriesByKey;
  const keysWithSingleEntryOnly = Object.values(activeEntryCountsByKey).filter(
    (count) => count === 1,
  ).length;
  const similarKeyClusterCount = similarStateKeyClusterCount(Object.keys(activeEntryCountsByKey));

  if (options.tracer?.enabled === true && options.turnId !== undefined) {
    options.tracer.emit("shared_state.compile.completed", {
      turnId: options.turnId,
      audienceEntityId: options.audienceEntityId,
      previousEntryCount: options.previousEntryCount,
      operationCount: options.operationCount,
      rejectedCount: options.rejected.length,
      rejectionReasons: options.rejected.map((rejection) => rejection.reason),
      source_trust_rejections: toTraceJsonValue(
        options.rejected
          .filter((rejection) => rejection.sourceTrustReason !== undefined)
          .map((rejection) => ({
            operation_index: rejection.operationIndex,
            operation_type: rejection.operationType,
            source_stream_entry_id: rejection.sourceStreamEntryId ?? null,
            source_trust_reason: rejection.sourceTrustReason ?? "unknown",
          })),
      ),
      applied: options.applied,
      recordVersion: options.artifact?.record_version ?? null,
      artifactEntryCount: artifactSummary.renderedEntryCount,
      artifactRenderedTokenEstimate: artifactSummary.estimatedTokens,
      artifact_total_entry_count: artifactSummary.totalEntryCount,
      artifact_active_entry_count: artifactSummary.activeEntryCount,
      artifact_max_active_entries: options.maxActiveEntries ?? null,
      artifact_omitted_entry_count: artifactSummary.omittedEntryCount,
      omitted_live_recent_operational: artifactSummary.omittedLiveRecentOperational,
      omitted_live_recent_low_salience: artifactSummary.omittedLiveRecentLowSalience,
      omitted_live_old: artifactSummary.omittedLiveOld,
      omitted_locked: artifactSummary.omittedLocked,
      omitted_pending: artifactSummary.omittedPending,
      all_active_keys_indexed: artifactSummary.allActiveKeysIndexed,
      newest_entries_reserved: artifactSummary.newestReservedEntryCount,
      live_starvation_with_reserved:
        artifactSummary.omittedByKind.live > 0 && artifactSummary.renderedByKind.locked > 0,
      artifact_pruned_entry_count_this_turn: options.prunedEntryCountThisTurn,
      artifact_superseded_count_this_turn: options.supersededEntryCountThisTurn,
      operation_counts_by_kind: toTraceJsonValue(
        options.operationCountsByKind ?? {
          add: 0,
          update: 0,
          supersede: 0,
          prune: 0,
        },
      ),
      operation_counts_by_state_key: toTraceJsonValue(options.operationCountsByStateKey ?? {}),
      new_state_key_count: options.newStateKeys?.length ?? 0,
      new_state_keys: toTraceJsonValue(options.newStateKeys ?? []),
      keys_with_single_entry_only: keysWithSingleEntryOnly,
      similar_key_cluster_count: similarKeyClusterCount,
      rendered_by_kind: toTraceJsonValue(artifactSummary.renderedByKind),
      omitted_by_kind: toTraceJsonValue(artifactSummary.omittedByKind),
      shared_state_entries_by_key: toTraceJsonValue(artifactSummary.activeEntriesByKey),
      shared_state_top_keys_by_entry_count: toTraceJsonValue(artifactSummary.topKeysByEntryCount),
      ledger_mode: options.ledgerMode,
      input_token_estimate: options.promptBudget.inputTokenEstimate,
      input_token_breakdown: toTraceJsonValue(options.promptBudget.breakdown),
      canonicalizes_rejected_non_locked: toTraceJsonValue(
        options.nonLockedCanonicalizesDrops ?? [],
      ),
    });
  }
}

export function traceAddRejectedCapExceeded(options: {
  tracer?: TurnTracer;
  turnId?: string;
  audienceEntityId: EntityId;
  rejection: PatchRejection;
}): void {
  if (options.tracer?.enabled !== true || options.turnId === undefined) {
    return;
  }

  options.tracer.emit("shared_state.compile.add_rejected_cap_exceeded", {
    turnId: options.turnId,
    audienceEntityId: options.audienceEntityId,
    operation_index: options.rejection.operationIndex,
    operation_type: options.rejection.operationType,
    state_key: options.rejection.stateKey ?? null,
    current_count: options.rejection.currentCount ?? null,
    proposed_count: options.rejection.proposedCount ?? null,
    max_live_entries_per_key: options.rejection.maxLiveEntriesPerKey ?? null,
    target_entry_id: options.rejection.targetEntryId ?? null,
  });
}

export function traceAddRejectedNearDuplicateStateKey(options: {
  tracer?: TurnTracer;
  turnId?: string;
  audienceEntityId: EntityId;
  rejection: PatchRejection;
}): void {
  if (options.tracer?.enabled !== true || options.turnId === undefined) {
    return;
  }

  options.tracer.emit("shared_state.compile.add_rejected_near_duplicate_state_key", {
    turnId: options.turnId,
    audienceEntityId: options.audienceEntityId,
    operation_index: options.rejection.operationIndex,
    operation_type: options.rejection.operationType,
    state_key: options.rejection.stateKey ?? null,
    similar_state_keys: options.rejection.similarStateKeys ?? [],
    shared_state_key_tokens: options.rejection.sharedStateKeyTokens ?? [],
  });
}

export function traceAddRejectedMissingNewKeyReason(options: {
  tracer?: TurnTracer;
  turnId?: string;
  audienceEntityId: EntityId;
  rejection: PatchRejection;
}): void {
  if (options.tracer?.enabled !== true || options.turnId === undefined) {
    return;
  }

  options.tracer.emit("shared_state.compile.add_rejected_missing_new_key_reason", {
    turnId: options.turnId,
    audienceEntityId: options.audienceEntityId,
    operation_index: options.rejection.operationIndex,
    operation_type: options.rejection.operationType,
    state_key: options.rejection.stateKey ?? null,
  });
}

export function traceLabelUngrounded(options: {
  tracer?: TurnTracer;
  turnId?: string;
  audienceEntityId: EntityId;
  rejection: PatchRejection;
}): void {
  if (options.tracer?.enabled !== true || options.turnId === undefined) {
    return;
  }

  options.tracer.emit("shared_state.compile.label_ungrounded", {
    turnId: options.turnId,
    audienceEntityId: options.audienceEntityId,
    operation_index: options.rejection.operationIndex,
    operation_type: options.rejection.operationType,
    operation_id:
      options.rejection.targetEntryId ?? `operation:${options.rejection.operationIndex}`,
    protected_relationship_labels: options.rejection.protectedRelationshipLabels ?? [],
    relationship_evidence_relational_slot_ids:
      options.rejection.relationshipEvidenceRelationalSlotIds ?? [],
    relationship_evidence_stream_entry_ids:
      options.rejection.relationshipEvidenceStreamEntryIds ?? [],
    rejected_relationship_evidence_relational_slot_ids:
      options.rejection.rejectedRelationshipEvidenceRelationalSlotIds ?? [],
    rejected_relationship_evidence_stream_entry_ids:
      options.rejection.rejectedRelationshipEvidenceStreamEntryIds ?? [],
  });
}

export function traceCompileDegraded(options: {
  tracer?: TurnTracer;
  turnId?: string;
  audienceEntityId: EntityId;
  reason: string;
  error?: unknown;
}): void {
  if (options.tracer?.enabled !== true || options.turnId === undefined) {
    return;
  }

  options.tracer.emit("shared_state.compile.degraded", {
    turnId: options.turnId,
    audienceEntityId: options.audienceEntityId,
    reason: options.reason,
    ...(options.error === undefined
      ? {}
      : { error: options.error instanceof Error ? options.error.message : String(options.error) }),
  });
}

function traceRepairEvent(
  event:
    | "shared_state.compile.repair_attempted"
    | "shared_state.compile.repair_succeeded"
    | "shared_state.compile.repair_failed",
  options: {
    tracer?: TurnTracer;
    turnId?: string;
    audienceEntityId: EntityId;
    error?: unknown;
  },
): void {
  if (options.tracer?.enabled !== true || options.turnId === undefined) {
    return;
  }

  options.tracer.emit(event, {
    turnId: options.turnId,
    audienceEntityId: options.audienceEntityId,
    ...(options.error === undefined
      ? {}
      : { error: options.error instanceof Error ? options.error.message : String(options.error) }),
  });
}

export function traceCompileRepairAttempted(options: {
  tracer?: TurnTracer;
  turnId?: string;
  audienceEntityId: EntityId;
  error: unknown;
}): void {
  traceRepairEvent("shared_state.compile.repair_attempted", options);
}

export function traceCompileRepairSucceeded(options: {
  tracer?: TurnTracer;
  turnId?: string;
  audienceEntityId: EntityId;
}): void {
  traceRepairEvent("shared_state.compile.repair_succeeded", options);
}

export function traceCompileRepairFailed(options: {
  tracer?: TurnTracer;
  turnId?: string;
  audienceEntityId: EntityId;
  error: unknown;
}): void {
  traceRepairEvent("shared_state.compile.repair_failed", options);
}

export function traceCompileOverBudget(options: {
  tracer?: TurnTracer;
  turnId?: string;
  audienceEntityId: EntityId;
  ledgerMode: SharedStateLedgerMode;
  promptBudget: SharedStateArtifactPromptBudget;
}): void {
  if (
    options.promptBudget.inputTokenEstimate <= SHARED_STATE_PROMPT_WARNING_TOKEN_THRESHOLD ||
    options.tracer?.enabled !== true ||
    options.turnId === undefined
  ) {
    return;
  }

  options.tracer.emit("shared_state.compile.degraded", {
    turnId: options.turnId,
    audienceEntityId: options.audienceEntityId,
    ledger_mode: options.ledgerMode,
    input_token_estimate: options.promptBudget.inputTokenEstimate,
    input_token_budget: SHARED_STATE_PROMPT_WARNING_TOKEN_THRESHOLD,
    breakdown: toTraceJsonValue(options.promptBudget.breakdown),
  });
}

export function traceReconciliationCompleted(options: {
  tracer?: TurnTracer;
  turnId?: string;
  result: SharedStateReconciliationResult;
  canonicalizationDuplicateDrops?: readonly CanonicalizationDuplicateDrop[];
  currentOperationCanonicalizationCount?: number;
  retriedStrandedCanonicalizationCount?: number;
  retrySummary?: SharedStateUnsettledReconciliationSummary | null;
}): void {
  if (options.tracer?.enabled === true && options.turnId !== undefined) {
    options.tracer.emit("shared_state.reconcile.completed", {
      turnId: options.turnId,
      mode: "primary",
      goals_retired: options.result.goals_retired,
      commitments_retired: options.result.commitments_retired,
      actions_retired: options.result.actions_retired,
      open_questions_retired: options.result.open_questions_retired,
      goals_canonicalized_attempted: options.result.goals_canonicalized_attempted,
      goals_canonicalized_succeeded: options.result.goals_canonicalized_succeeded,
      goals_canonicalized_skipped: options.result.goals_canonicalized_skipped,
      commitments_revoked_attempted: options.result.commitments_revoked_attempted,
      commitments_revoked_succeeded: options.result.commitments_revoked_succeeded,
      commitments_revoked_skipped: options.result.commitments_revoked_skipped,
      actions_completed_attempted: options.result.actions_completed_attempted,
      actions_completed_succeeded: options.result.actions_completed_succeeded,
      actions_completed_skipped: options.result.actions_completed_skipped,
      actions_closed_by_borg_self_performance:
        options.result.actions_closed_by_borg_self_performance,
      open_questions_resolved_attempted: options.result.open_questions_resolved_attempted,
      open_questions_resolved_succeeded: options.result.open_questions_resolved_succeeded,
      open_questions_resolved_skipped: options.result.open_questions_resolved_skipped,
      semantic_nodes_reviewed_attempted: options.result.semantic_nodes_reviewed_attempted,
      semantic_nodes_marked_superseded: options.result.semantic_nodes_marked_superseded,
      semantic_nodes_marked_contradicted: options.result.semantic_nodes_marked_contradicted,
      semantic_nodes_skipped: options.result.semantic_nodes_skipped,
      unknown_ids: toTraceJsonValue(options.result.unknown_ids),
      canonicalization_duplicates_dropped: toTraceJsonValue(
        options.canonicalizationDuplicateDrops ?? [],
      ),
      current_operation_canonicalization_count: options.currentOperationCanonicalizationCount ?? 0,
      retried_stranded_canonicalization_count: options.retriedStrandedCanonicalizationCount ?? 0,
      retry_unsettled_summary: toTraceJsonValue(options.retrySummary ?? null),
      skipped_commitments: toTraceJsonValue(options.result.skipped_commitments),
      errors: toTraceJsonValue(options.result.errors),
    });
  }
}
