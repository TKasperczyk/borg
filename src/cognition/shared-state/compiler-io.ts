import type {
  SharedStateArtifact,
  SharedStateOperation,
} from "../../memory/shared-state/index.js";
import type { EntityId, StreamEntryId } from "../../util/ids.js";
import { toTraceJsonValue, type TurnTracer } from "../../tracing/tracer.js";
import { summarizeSharedStateArtifactRender, type SharedStateRenderOptions } from "./render.js";
import { similarStateKeyClusterCount } from "./state-key.js";
import type {
  SharedStateReconciliationResult,
  SharedStateUnsettledReconciliationSummary,
} from "./reconciliation.js";
import { type SharedStateArtifactPromptBudget } from "./compiler-prompt.js";
import {
  SHARED_STATE_PROMPT_WARNING_TOKEN_THRESHOLD,
  SHARED_STATE_TOOL_NAME,
} from "./constants.js";
import {
  sharedStatePatchSchema,
  type CanonicalizationDuplicateDrop,
  type EmptyUpdateDrop,
  type EmitSharedStatePatch,
  type NonLockedCanonicalizesDrop,
  type PatchRejection,
  type SharedStateLedgerMode,
} from "./types.js";
import type {
  LifecycleAgingBlockedSampleEntry,
  LifecycleAgingBlockerCounts,
  LifecycleAgingUnknownAgeSampleEntry,
  SharedStateLifecycleTransition,
} from "./lifecycle-aging.js";

type PublicSharedStateOperation = Exclude<SharedStateOperation, { type: "transition_kind" }>;

export function parseResponse(input: unknown): EmitSharedStatePatch {
  const parsed = sharedStatePatchSchema.safeParse(input);

  if (!parsed.success) {
    throw parsed.error;
  }

  return parsed.data;
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
  operationCountsByKind?: Record<PublicSharedStateOperation["type"], number>;
  operationCountsByStateKey?: Record<string, Record<PublicSharedStateOperation["type"], number>>;
  newStateKeys?: readonly string[];
  ledgerMode: SharedStateLedgerMode;
  promptBudget: SharedStateArtifactPromptBudget;
  nonLockedCanonicalizesDrops?: readonly NonLockedCanonicalizesDrop[];
  emptyUpdateAttemptedCount?: number;
  emptyUpdateDroppedCount?: number;
  emptyUpdateRepairedCount?: number;
  addRejectedCapExceededCount?: number;
  lifecycleTransitions?: readonly SharedStateLifecycleTransition[];
  lifecycleAgingBlockerCountsLiveToLowSalience?: LifecycleAgingBlockerCounts;
  lifecycleAgingBlockerCountsLowSalienceToDormant?: LifecycleAgingBlockerCounts;
  lifecycleAgingBlockedSample?: readonly LifecycleAgingBlockedSampleEntry[];
  lifecycleAgingUnknownAgeSample?: readonly LifecycleAgingUnknownAgeSampleEntry[];
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
  const renderedEntryIds = new Set(artifactSummary.renderedEntryIds);
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
      omitted_live_unknown_age: artifactSummary.omittedLiveUnknownAge,
      omitted_locked: artifactSummary.omittedLocked,
      omitted_locked_recent_final_compile: artifactSummary.omittedLockedRecent,
      omitted_locked_old_final_compile: artifactSummary.omittedLockedOld,
      omitted_locked_unknown_age_final_compile: artifactSummary.omittedLockedUnknownAge,
      omitted_locked_with_active_critical_commitment_final_compile:
        artifactSummary.omittedLockedWithActiveCriticalCommitment,
      omitted_locked_with_operational_canonicalizer_final_compile:
        artifactSummary.omittedLockedWithOperationalCanonicalizer,
      omitted_locked_indexed_only_final_compile: artifactSummary.omittedLockedIndexedOnly,
      omitted_pending: artifactSummary.omittedPending,
      omitted_low_salience_live: artifactSummary.omittedLowSalienceLive,
      omitted_dormant_live: artifactSummary.omittedDormantLive,
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
      active_by_kind: toTraceJsonValue(artifactSummary.activeByKind),
      shared_state_entries_by_key: toTraceJsonValue(artifactSummary.activeEntriesByKey),
      shared_state_top_keys_by_entry_count: toTraceJsonValue(artifactSummary.topKeysByEntryCount),
      ledger_mode: options.ledgerMode,
      input_token_estimate: options.promptBudget.inputTokenEstimate,
      input_token_breakdown: toTraceJsonValue(options.promptBudget.breakdown),
      canonicalizes_rejected_non_locked: toTraceJsonValue(
        options.nonLockedCanonicalizesDrops ?? [],
      ),
      update_checked_for_empty_count: options.emptyUpdateAttemptedCount ?? 0,
      empty_update_attempted_count: options.emptyUpdateAttemptedCount ?? 0,
      empty_update_dropped_count: options.emptyUpdateDroppedCount ?? 0,
      empty_update_repaired_count: options.emptyUpdateRepairedCount ?? 0,
      add_rejected_cap_exceeded_count: options.addRejectedCapExceededCount ?? 0,
      lifecycle_aging_blocker_counts_live_to_low_salience: toTraceJsonValue(
        options.lifecycleAgingBlockerCountsLiveToLowSalience ?? null,
      ),
      lifecycle_aging_blocker_counts_low_salience_to_dormant: toTraceJsonValue(
        options.lifecycleAgingBlockerCountsLowSalienceToDormant ?? null,
      ),
      lifecycle_aging_blocked_sample: toTraceJsonValue(
        (options.lifecycleAgingBlockedSample ?? []).map((entry) => ({
          ...entry,
          rendered: renderedEntryIds.has(entry.entry_id),
        })),
      ),
      lifecycle_aging_unknown_age_sample: toTraceJsonValue(
        (options.lifecycleAgingUnknownAgeSample ?? []).map((entry) => ({
          ...entry,
          rendered: renderedEntryIds.has(entry.entry_id),
        })),
      ),
      lifecycle_demoted_live_to_low_salience_count: (options.lifecycleTransitions ?? []).filter(
        (transition) => transition.fromKind === "live" && transition.toKind === "low_salience_live",
      ).length,
      lifecycle_demoted_low_salience_to_dormant_count: (options.lifecycleTransitions ?? []).filter(
        (transition) =>
          transition.fromKind === "low_salience_live" && transition.toKind === "dormant_live",
      ).length,
      lifecycle_reactivated_low_salience_live_count: (options.lifecycleTransitions ?? []).filter(
        (transition) => transition.fromKind === "low_salience_live" && transition.toKind === "live",
      ).length,
      lifecycle_reactivated_dormant_live_count: (options.lifecycleTransitions ?? []).filter(
        (transition) => transition.fromKind === "dormant_live" && transition.toKind === "live",
      ).length,
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

export function traceClaimUngrounded(options: {
  tracer?: TurnTracer;
  turnId?: string;
  audienceEntityId: EntityId;
  rejection: PatchRejection;
}): void {
  if (options.tracer?.enabled !== true || options.turnId === undefined) {
    return;
  }

  options.tracer.emit("shared_state.compile.claim_ungrounded", {
    turnId: options.turnId,
    audienceEntityId: options.audienceEntityId,
    operation_index: options.rejection.operationIndex,
    operation_type: options.rejection.operationType,
    operation_id:
      options.rejection.targetEntryId ?? `operation:${options.rejection.operationIndex}`,
    relationship_claim_label_families: [
      ...new Set(
        (options.rejection.ungroundedRelationshipClaims ?? []).map((claim) => claim.label_family),
      ),
    ],
    relationship_claims: options.rejection.relationshipClaims ?? [],
    ungrounded_relationship_claims: options.rejection.ungroundedRelationshipClaims ?? [],
    rejected_relationship_claim_evidence_relational_slot_ids:
      options.rejection.rejectedRelationshipClaimEvidenceRelationalSlotIds ?? [],
    rejected_relationship_claim_evidence_stream_entry_ids:
      options.rejection.rejectedRelationshipClaimEvidenceStreamEntryIds ?? [],
  });
}

export function traceEmptyUpdateDropped(options: {
  tracer?: TurnTracer;
  turnId?: string;
  audienceEntityId: EntityId;
  drop: EmptyUpdateDrop;
}): void {
  if (options.tracer?.enabled !== true || options.turnId === undefined) {
    return;
  }

  options.tracer.emit("shared_state.compile.empty_update_dropped", {
    turnId: options.turnId,
    audienceEntityId: options.audienceEntityId,
    operation_index: options.drop.operationIndex,
    operation_id: options.drop.operationId,
    state_key: options.drop.stateKey,
    field_presence: toTraceJsonValue(options.drop.fieldPresence),
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
    reason: "prompt_over_budget",
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
