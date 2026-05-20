import type { LLMCompleteResult, LLMMessage, LLMToolDefinition } from "../../llm/index.js";
import type { SharedStateArtifact } from "../../memory/decision-artifacts/index.js";
import type { EntityId } from "../../util/ids.js";
import type { JsonValue } from "../../util/json-value.js";
import { SHARED_STATE_SYSTEM_PROMPT } from "../prompts/shared-state.js";
import { buildUsageTraceBlock, toTraceJsonValue, type TurnTracer } from "../tracing/tracer.js";
import { summarizeSharedStateArtifactRender, type SharedStateRenderOptions } from "./render.js";
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
  maxActiveEntries?: number;
  prunedEntryCountThisTurn: number;
  supersededEntryCountThisTurn: number;
  ledgerMode: SharedStateLedgerMode;
  promptBudget: SharedStateArtifactPromptBudget;
  nonLockedCanonicalizesDrops?: readonly NonLockedCanonicalizesDrop[];
}): void {
  const artifactSummary = summarizeSharedStateArtifactRender(
    options.artifact,
    options.renderOptions,
  );

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
      artifact_pruned_entry_count_this_turn: options.prunedEntryCountThisTurn,
      artifact_superseded_count_this_turn: options.supersededEntryCountThisTurn,
      rendered_by_kind: toTraceJsonValue(artifactSummary.renderedByKind),
      omitted_by_kind: toTraceJsonValue(artifactSummary.omittedByKind),
      ledger_mode: options.ledgerMode,
      input_token_estimate: options.promptBudget.inputTokenEstimate,
      input_token_breakdown: toTraceJsonValue(options.promptBudget.breakdown),
      canonicalizes_rejected_non_locked: toTraceJsonValue(
        options.nonLockedCanonicalizesDrops ?? [],
      ),
    });
  }
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
