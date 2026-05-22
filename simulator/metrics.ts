import { performance } from "node:perf_hooks";
import { readFileSync } from "node:fs";

import {
  QUARANTINED_USER_ENTRY_EVENT,
  ACTION_CANDIDATE_CLASSIFICATIONS,
  ACTION_STATES,
  COMMITMENT_CRITICAL_DOMAINS,
  COMMITMENT_ENFORCEMENT_CLASSES,
  COMMITMENT_KINDS,
  COMMITMENT_TYPES,
  GOAL_PROMOTION_CLASSIFICATIONS,
  RELATIONAL_SLOT_STATES,
  REVIEW_KINDS,
  SEMANTIC_NODE_STATUSES,
  OPEN_QUESTION_STATUSES,
  effectiveCommitmentCriticalDomain,
  effectiveCommitmentEnforcementClass,
  type ActionCandidateClassification,
  type ActionRecord,
  type ActionRecordCreationSource,
  type ActionState,
  type Borg,
  type CommitmentCriticalDomain,
  type CommitmentEnforcementClass,
  type GoalPromotionClassification,
  type CommitmentKind,
  type CommitmentRecord,
  type CommitmentType,
  type OpenQuestion,
  type OpenQuestionSource,
  type OpenQuestionStatus,
  type RelationalSlotState,
  type ReviewKind,
  type SemanticNode,
  type SemanticNodeStatus,
  type SessionId,
} from "../src/index.js";
import { CLASSIFICATION_DOWNGRADE_REASONS } from "../src/cognition/commitments/classification-normalizer.js";
import type { ClassificationDowngradeReason } from "../src/cognition/commitments/classification-normalizer.js";
import {
  DEFAULT_ACTION_THREAD_SIMILARITY_THRESHOLD,
  DEFAULT_ACTION_THREAD_SOURCE_RECORD_LIMIT,
  actionSalienceClass,
  buildActionThreads,
  summarizeActionPromptSalience,
  type ActionPromptSalienceSummary,
} from "../src/cognition/evidence-ledger/action-threads.js";
import {
  ACTION_ARCHIVE_ACTIVE_STATES,
  ACTION_ARCHIVE_SCAN_LIMIT,
  classifyActionArchiveCandidate,
  lastReferencedActionLifecycleTurn,
} from "../src/memory/actions/index.js";
import type { ActionLedgerRepository } from "../src/cognition/evidence-ledger/builder-types.js";
import type { ScopeResolver } from "../src/cognition/evidence-ledger/scope-resolver.js";
import { sharedStateSemanticRevisionVerdictCacheSize } from "../src/cognition/shared-state/reconciliation.js";
import { filterActiveStreamEntries } from "../src/stream/index.js";
import type { ActionId } from "../src/util/ids.js";
import { readTraceEvents } from "../assessor/trace-reader.js";
import type { TraceRecord } from "../assessor/types.js";
import { canonicalTraceEventName } from "../src/cognition/tracing/taxonomy.js";
import { isExtractorMaxTokenLlmLabel } from "../src/cognition/tracing/extractor-labels.js";
import { writeFileAtomic } from "../src/util/atomic-write.js";

import { appendJsonlLine } from "./jsonl.js";
import { simulatorHealthWarningsForRows } from "./health-warnings.js";
import type {
  ActionCandidateClassificationMetricKey,
  GoalPromotionClassificationMetricKey,
  MetricsRow,
  SimulatorHealthWarning,
  SimulatorHealthWarningKind,
} from "./types.js";

const LARGE_COUNT_LIMIT = 1_000_000;
const TURN_METRICS_EVENT = "turn_metrics";
const ABORTED_TURN_EVENT = "aborted_turn";
const ABORTED_ATTEMPT_EVENT = "aborted_attempt";
const OPEN_QUESTION_RECORD_TYPE = "open_question";
const RESOLVED_STATUS = "resolved";
const ACTION_CREATION_SOURCES = ["extractor", "reflector", "api", "unknown"] as const;
const ACTION_CANDIDATE_CLASSIFICATION_METRIC_KEYS = [
  ...ACTION_CANDIDATE_CLASSIFICATIONS,
  "invalid_classification",
] as const satisfies readonly ActionCandidateClassificationMetricKey[];
const ACTIVE_ACTION_STATES: readonly ActionState[] = [
  "considering",
  "committed_to_do",
  "scheduled",
  "unknown",
];
const TERMINAL_ACTION_STATES: readonly ActionState[] = [
  "completed",
  "not_done",
  "expired",
  "archived",
];
const SHARED_STATE_RECENT_RENDER_KINDS = ["live", "pending", "invalidated"] as const;
const DEFAULT_SHARED_STATE_MAX_ACTIVE_ENTRIES = 40;
const ACTION_DUPLICATE_PRESSURE_CHECK_EVERY_TURNS = 10;
const ACTION_DORMANT_AFTER_INACTIVE_TURNS = 15;
const ACTION_ARCHIVE_AFTER_INACTIVE_TURNS = 20;
const ACTION_INACTIVE_TURN_BUCKETS = ["0-15", "15-20", "20-30", "30+"] as const;
const SHARED_STATE_COMPILER_LABEL = "decision_artifact_compiler";
const SHARED_STATE_SEMANTIC_REVISION_LABEL = "decision_artifact_semantic_revision";
const SHARED_STATE_COMPILER_OPERATION_KINDS = ["add", "update", "supersede", "prune"] as const;
const GOAL_PROMOTION_CLASSIFICATION_METRIC_KEYS = [
  ...GOAL_PROMOTION_CLASSIFICATIONS,
  "invalid_classification",
] as const satisfies readonly GoalPromotionClassificationMetricKey[];

type GoalTreeNodeLike = {
  children?: GoalTreeNodeLike[];
};

type MemoryBandMetricCounts = Pick<
  MetricsRow,
  | "action_record_count_total"
  | "action_record_count_by_state"
  | "action_record_count_committed_to_do"
  | "action_record_count_canonicalized"
  | "action_record_count_active"
  | "borg_owned_active_actions"
  | "participant_owned_active_actions"
  | "group_owned_active_actions"
  | "prompt_salient_actions_total"
  | "borg_owned_salient_active_actions"
  | "participant_owned_salient_active_actions"
  | "dormant_actions_total"
  | "dormant_not_archive_eligible_count"
  | "dormant_archive_eligible_count"
  | "archive_oldest_inactive_turns"
  | "archive_inactive_turn_distribution"
  | "archive_archivable_count"
  | "archive_skipped_borg_owned"
  | "archive_skipped_due_date"
  | "archive_skipped_below_threshold"
  | "archive_skipped_other"
  | "archive_oldest_archivable_inactive_turns"
  | "stale_actions_omitted_from_prompt"
  | "actions_per_turn"
  | "salient_actions_per_turn"
  | "action_retirement_ratio"
  | "borg_owned_action_count"
  | "stale_action_count"
  | "action_record_creation_source_per_turn"
  | "action_record_creation_count_this_turn"
  | "actions_dormant_count"
  | "actions_archived_count"
  | "recent_completed_action_count"
  | "commitment_count_active"
  | "commitment_count_active_by_kind"
  | "commitments_by_enforcement_class"
  | "critical_commitments_by_kind_type_domain"
  | "commitments_advisory_count"
  | "commitments_critical_count"
  | "commitment_count_superseded"
  | "commitment_count_revoked"
  | "commitment_count_expired"
  | "commitment_count_canonicalized"
  | "pending_action_count"
  | "pending_action_merge_count"
  | "relational_slot_count_by_state"
  | "review_queue_open_count_by_type"
  | "open_question_resolved_count"
>;

export type MetricsCaptureOptions = {
  tracePath?: string;
  semanticRevisionVerdictCacheSize?: () => number;
  scenarioKey?: string;
};

export type MetricsCaptureContext = {
  sessionId: SessionId;
  sessionIds: readonly SessionId[];
  transportChatAttempts: number;
  overseerDueOnSuppressedTurn?: boolean;
  simulatorPersonaFailures?: number;
  borgHardAbortedTurns?: number;
  borgIntentionalSuppressions?: number;
  borgIntentionalSuppressionsByReason?: Record<string, number>;
  /**
   * Deprecated compatibility input; use borgHardAbortedTurns.
   */
  borgAbortedTurns?: number;
};

type FrameAnomalyMetricCounts = Pick<
  MetricsRow,
  | "frame_anomaly_classifier_calls"
  | "frame_anomaly_classified_normal_count"
  | "frame_anomaly_actual_anomaly_count"
  | "frame_anomaly_degraded_count"
  | "frame_anomaly_degraded_fallback_match_count"
  | "quarantined_user_entry_count"
  | "early_extractors_skipped_frame_anomaly_count"
>;

type GoalPromotionMetricCounts = Pick<
  MetricsRow,
  | "goal_promotion_salvaged_promotions"
  | "goal_promotion_skipped_promotions"
  | "goal_promotion_initial_step_downgraded"
  | "goal_promotion_dedup_skipped_extractor_signal"
  | "goal_promotion_dedup_skipped_embedding"
  | "goal_promotion_dedup_degraded"
  | "goal_promotion_classifications_per_turn"
  | "goal_promotion_rejected_classification"
  | "goal_promotion_cap_rejections"
>;

type SharedStateArtifactSemanticRevisionMetricCounts = Pick<
  MetricsRow,
  | "decision_artifact_semantic_revisions_attempted"
  | "decision_artifact_semantic_revisions_completed_succeeded"
  | "decision_artifact_semantic_nodes_marked_superseded"
  | "decision_artifact_semantic_nodes_marked_contradicted"
  | "decision_artifact_semantic_revision_cache_hits"
>;

type CommitmentRegenerationMetricCounts = Pick<
  MetricsRow,
  | "commitment_regeneration_attempted_count"
  | "commitment_regeneration_succeeded_count"
  | "commitment_regeneration_failed_count"
  | "commitment_regeneration_attempted_total"
  | "commitment_regeneration_succeeded_total"
  | "commitment_regeneration_failed_total"
  | "commitment_guard_advisory_violations_total"
  | "commitment_guard_advisory_violations_by_class"
>;

type CommitmentClassificationDowngradeMetricCounts = Pick<
  MetricsRow,
  | "commitments_critical_classification_downgraded_total"
  | "commitments_critical_classification_downgraded_by_reason"
  | "commitments_critical_classification_downgraded_by_kind_type_from_domain"
>;

type SemanticRevisionErrorMetricCounts = Pick<
  MetricsRow,
  | "semantic_revision_error_count"
  | "semantic_revision_skipped_due_to_error"
  | "semantic_revision_error_total_by_reason"
>;

type SemanticRevisionCumulativeMetricCounts = Pick<
  MetricsRow,
  | "semantic_revision_calls_total"
  | "semantic_revision_candidates_reviewed_total"
  | "semantic_revision_superseded_total"
  | "semantic_revision_contradicted_total"
  | "semantic_revision_degraded_total"
  | "semantic_revision_skipped_over_cap_total"
>;

type SharedStateCapPressureMetricCounts = Pick<
  MetricsRow,
  | "shared_state_at_cap_turns"
  | "shared_state_compile_evaluated_turns"
  | "shared_state_omitted_recent_entries"
  | "shared_state_live_entry_starvation"
  | "shared_state_newest_entries_reserved"
  | "shared_state_live_starvation_with_reserved"
  | "shared_state_live_starvation_ever"
  | "shared_state_live_starvation_final"
>;

type SharedStateCompilerHealthMetricCounts = Pick<
  MetricsRow,
  | "shared_state_compiler_max_tokens_total"
  | "shared_state_compiler_degraded_total"
  | "shared_state_compiler_repair_attempted_total"
  | "shared_state_compiler_repair_succeeded_total"
  | "shared_state_compiler_repair_failed_total"
  | "shared_state_compiler_repair_failed_by_rejection_reason"
  | "shared_state_compiler_operations_total_by_kind"
  | "shared_state_add_to_update_ratio"
  | "shared_state_entries_by_key"
  | "shared_state_add_to_update_ratio_by_key"
  | "shared_state_top_keys_by_entry_count"
  | "shared_state_add_rejected_cap_exceeded_total"
  | "shared_state_new_keys_per_compile"
  | "shared_state_new_keys_per_turn"
  | "shared_state_keys_with_single_entry_only"
  | "shared_state_similar_key_cluster_count"
  | "shared_state_add_rejected_near_duplicate_state_key_total"
  | "shared_state_add_rejected_missing_new_key_reason_total"
>;

type SessionReentryContinuityMetricCounts = Pick<
  MetricsRow,
  | "session_reentry_card_rendered_total"
  | "session_reentry_card_rendered_by_audience"
  | "session_reentry_first_turn_with_existing_state_total"
  | "session_reentry_first_turn_blank_audience_total"
>;

type ReviewResolverMetricCounts = {
  review_resolver_attempted: number;
  review_resolver_accepted: number;
  review_resolver_dismissed: number;
  review_resolver_rejected: number;
  review_resolver_needs_manual: number;
  review_queue_enqueued_this_turn: number;
  review_queue_resolved_this_turn: number;
  review_queue_drain_rate: number | null;
};

type ActionCandidateMetricCounts = Pick<
  MetricsRow,
  | "action_candidate_classifications_per_turn"
  | "action_candidate_rejected_classification"
  | "action_persistence_dedup_skipped_embedding"
  | "action_persistence_dedup_degraded"
  | "actions_closed_by_terminal_emission"
  | "actions_closed_by_borg_self_performance"
  | "actions_rejected_capability"
>;

type ActionSessionLifecycleMetricCounts = Pick<MetricsRow, "actions_expired_at_session_close">;

type SharedStateActionLifecycleMetricCounts = Pick<
  MetricsRow,
  "actions_canonicalized" | "actions_completed_via_canonicalization"
>;

type ExtractorHealthMetricCounts = Pick<
  MetricsRow,
  | "closure_loop_completed_count"
  | "closure_loop_degraded_count"
  | "corrective_preference_completed_count"
  | "corrective_preference_degraded_count"
  | "extractor_max_tokens_stop_count"
  | "extractor_max_tokens_total_by_label"
  | "extractor_degraded_total_by_label"
>;

type ClosurePressureMetricCounts = Pick<
  MetricsRow,
  | "closure_pressure_mixed_observed_total"
  | "closure_pressure_closure_only_suppressed_total"
  | "closure_pressure_mixed_passed_no_active_preference_total"
  | "closure_pressure_mixed_by_span_kind"
>;

type SemanticMemoryWriteGateMetricCounts = Pick<
  MetricsRow,
  | "semantic_nodes_rejected_ungrounded_label_count"
  | "semantic_nodes_rejected_ungrounded_label_total"
  | "semantic_nodes_rejected_ungrounded_label_by_label"
  | "shared_state_operations_rejected_ungrounded_label_total"
  | "shared_state_operations_rejected_ungrounded_label_by_label"
  | "commitment_candidates_rejected_ungrounded_label_total"
  | "commitment_candidates_rejected_ungrounded_label_by_label"
>;

function flattenGoalCount(nodes: readonly GoalTreeNodeLike[]): number {
  let count = 0;
  const stack = [...nodes];

  while (stack.length > 0) {
    const next = stack.shift();

    if (next === undefined) {
      continue;
    }

    count += 1;
    stack.push(...(next.children ?? []));
  }

  return count;
}

function latencyBetween(
  records: readonly TraceRecord[],
  startEvent: string,
  endEvent: string,
): number | null {
  const start = records.find((record) => record.event === startEvent);
  const end = [...records].reverse().find((record) => record.event === endEvent);

  if (start === undefined || end === undefined) {
    return null;
  }

  // Tracer records both `ts` (logical clock, can be a ManualClock that
  // shares values across all events in a turn) and `wallMs`
  // (performance.now monotonic real time). Latency is meaningful only
  // off real time; fall back to ts only if wallMs is missing for some
  // reason (older records).
  const startWall = typeof start.wallMs === "number" ? start.wallMs : null;
  const endWall = typeof end.wallMs === "number" ? end.wallMs : null;

  if (startWall !== null && endWall !== null && endWall >= startWall) {
    return Math.round(endWall - startWall);
  }

  if (end.ts < start.ts) {
    return null;
  }

  return end.ts - start.ts;
}

function usageForTurn(records: readonly TraceRecord[]): {
  inputTokens: number;
  outputTokens: number;
} {
  let inputTokens = 0;
  let outputTokens = 0;

  for (const record of records) {
    if (record.event !== "llm_call.completed") {
      continue;
    }

    const usage = record.usage;

    if (usage === null || typeof usage !== "object" || Array.isArray(usage)) {
      continue;
    }

    const input = (usage as { inputTokens?: unknown }).inputTokens;
    const output = (usage as { outputTokens?: unknown }).outputTokens;

    inputTokens += typeof input === "number" && Number.isFinite(input) ? input : 0;
    outputTokens += typeof output === "number" && Number.isFinite(output) ? output : 0;
  }

  return { inputTokens, outputTokens };
}

function generationSuppressionCount(borg: Borg, sessionIds: readonly SessionId[]): number {
  return [...new Set(sessionIds)]
    .flatMap((session) =>
      filterActiveStreamEntries(borg.stream.tail(LARGE_COUNT_LIMIT, { session })),
    )
    .filter((entry) => entry.kind === "agent_suppressed").length;
}

function traceLabel(record: TraceRecord): string | null {
  return typeof record.label === "string" ? record.label : null;
}

function traceStatus(record: TraceRecord): string | null {
  return typeof record.status === "string" ? record.status : null;
}

function traceKind(record: TraceRecord): string | null {
  return typeof record.kind === "string" ? record.kind : null;
}

function traceReason(record: TraceRecord): string | null {
  return typeof record.reason === "string" ? record.reason : null;
}

function traceStopReason(record: TraceRecord): string | null {
  const value = record.stopReason ?? record.stop_reason;

  return typeof value === "string" ? value : null;
}

function traceNumber(record: TraceRecord, key: string): number {
  const value = record[key];

  return typeof value === "number" && Number.isFinite(value) ? value : 0;
}

function traceObjectNumber(record: TraceRecord, objectKey: string, numberKey: string): number {
  const value = record[objectKey];

  if (value === null || typeof value !== "object" || Array.isArray(value)) {
    return 0;
  }

  const nested = (value as Record<string, unknown>)[numberKey];

  return typeof nested === "number" && Number.isFinite(nested) ? nested : 0;
}

function traceObjectNumberEntries(record: TraceRecord, objectKey: string): Record<string, number> {
  const value = record[objectKey];

  if (value === null || typeof value !== "object" || Array.isArray(value)) {
    return {};
  }

  const entries: Record<string, number> = {};

  for (const [key, nested] of Object.entries(value as Record<string, unknown>)) {
    if (typeof nested === "number" && Number.isFinite(nested)) {
      entries[key] = nested;
    }
  }

  return entries;
}

function traceOptionalNumber(record: TraceRecord, key: string): number | null {
  const value = record[key];

  return typeof value === "number" && Number.isFinite(value) ? value : null;
}

function canonicalizeTraceRecords(records: readonly TraceRecord[]): TraceRecord[] {
  return records.map((record) => {
    const event = canonicalTraceEventName(record.event);

    return event === record.event ? record : { ...record, event };
  });
}

function archiveAfterInactiveTurnsFromTrace(records: readonly TraceRecord[]): number {
  for (const record of [...records].reverse()) {
    if (record.event !== "action_archive_scan.completed") {
      continue;
    }

    const archiveAfterTurns = traceOptionalNumber(record, "archive_after_turns");

    if (archiveAfterTurns !== null && archiveAfterTurns >= 0) {
      return Math.floor(archiveAfterTurns);
    }
  }

  return ACTION_ARCHIVE_AFTER_INACTIVE_TURNS;
}

function streamContentEvent(content: unknown): string | null {
  if (content === null || typeof content !== "object" || Array.isArray(content)) {
    return null;
  }

  const event = (content as { event?: unknown }).event;
  return typeof event === "string" ? event : null;
}

function streamContentTurnId(content: unknown): string | null {
  if (content === null || typeof content !== "object" || Array.isArray(content)) {
    return null;
  }

  const turnId = (content as { turn_id?: unknown }).turn_id;
  return typeof turnId === "string" ? turnId : null;
}

function quarantinedUserEntryCount(
  borg: Borg,
  sessionIds: readonly SessionId[],
  turnId: string,
): number {
  return [...new Set(sessionIds)]
    .flatMap((session) => borg.stream.tail(LARGE_COUNT_LIMIT, { session }))
    .filter(
      (entry) =>
        entry.kind === "internal_event" &&
        streamContentEvent(entry.content) === QUARANTINED_USER_ENTRY_EVENT &&
        streamContentTurnId(entry.content) === turnId,
    ).length;
}

function frameAnomalyMetrics(input: {
  traceRecords: readonly TraceRecord[];
  borg: Borg;
  sessionIds: readonly SessionId[];
  turnId: string;
}): FrameAnomalyMetricCounts {
  const frameClassified = input.traceRecords.filter(
    (record) => record.event === "frame_anomaly.completed",
  );
  const actualAnomalyCount = frameClassified.filter(
    (record) => traceStatus(record) === "ok" && traceKind(record) !== "normal",
  ).length;
  const fallbackMatchCount = input.traceRecords.filter(
    (record) => record.event === "frame_anomaly.fallback.completed" && record.matched === true,
  ).length;

  return {
    frame_anomaly_classifier_calls: input.traceRecords.filter(
      (record) =>
        record.event === "llm_call.started" && traceLabel(record) === "frame_anomaly_classifier",
    ).length,
    frame_anomaly_classified_normal_count: frameClassified.filter(
      (record) => traceStatus(record) === "ok" && traceKind(record) === "normal",
    ).length,
    frame_anomaly_actual_anomaly_count: actualAnomalyCount,
    frame_anomaly_degraded_count: frameClassified.filter(
      (record) => traceStatus(record) === "degraded",
    ).length,
    frame_anomaly_degraded_fallback_match_count: fallbackMatchCount,
    quarantined_user_entry_count: quarantinedUserEntryCount(
      input.borg,
      input.sessionIds,
      input.turnId,
    ),
    early_extractors_skipped_frame_anomaly_count: actualAnomalyCount + fallbackMatchCount,
  };
}

function actionCandidateMetrics(traceRecords: readonly TraceRecord[]): ActionCandidateMetricCounts {
  const completed = traceRecords.filter(
    (record) => record.event === "extraction.actions.completed",
  );
  const classificationRejected = traceRecords.filter(
    (record) => record.event === "extraction.actions.rejected",
  );
  const classificationsPerTurn = zeroActionCandidateClassificationCounts();

  for (const record of completed) {
    addActionCandidateClassificationCounts(classificationsPerTurn, record);
  }

  return {
    action_candidate_classifications_per_turn: classificationsPerTurn,
    action_candidate_rejected_classification: classificationRejected.filter(
      (record) => traceReason(record) === "non_concrete_classification",
    ).length,
    action_persistence_dedup_skipped_embedding: traceRecords.filter(
      (record) => record.event === "action_persistence.dedup.skipped",
    ).length,
    action_persistence_dedup_degraded: traceRecords.filter(
      (record) => record.event === "action_persistence.dedup.degraded",
    ).length,
    actions_closed_by_terminal_emission: traceRecords.filter(
      (record) => record.event === "action_state.transitioned",
    ).length,
    actions_closed_by_borg_self_performance: traceRecords.filter(
      (record) => record.event === "action_state.borg_self_performance.completed",
    ).length,
    actions_rejected_capability: classificationsPerTurn.outside_borg_capability,
  };
}

function actionSessionLifecycleMetrics(
  traceRecords: readonly TraceRecord[],
): ActionSessionLifecycleMetricCounts {
  return {
    actions_expired_at_session_close: traceRecords
      .filter((record) => record.event === "action_session_scope.expired")
      .reduce((sum, record) => sum + traceNumber(record, "actions_expired_at_session_close"), 0),
  };
}

function sharedStateActionLifecycleMetrics(
  traceRecords: readonly TraceRecord[],
): SharedStateActionLifecycleMetricCounts {
  const completed = traceRecords.filter(
    (record) => record.event === "shared_state.reconcile.completed" && record.mode !== "retry_only",
  );
  const retryOnly = traceRecords.filter(
    (record) => record.event === "shared_state.reconcile.completed" && record.mode === "retry_only",
  );
  const actionsCanonicalized =
    completed.reduce((sum, record) => sum + traceNumber(record, "actions_retired"), 0) +
    retryOnly.reduce(
      (sum, record) => sum + traceObjectNumber(record, "outcome_counts", "actions_retired"),
      0,
    );
  const actionsCompletedViaCanonicalization =
    completed.reduce((sum, record) => sum + traceNumber(record, "actions_completed_succeeded"), 0) +
    retryOnly.reduce(
      (sum, record) =>
        sum + traceObjectNumber(record, "outcome_counts", "actions_completed_succeeded"),
      0,
    );

  return {
    actions_canonicalized: actionsCanonicalized,
    actions_completed_via_canonicalization: actionsCompletedViaCanonicalization,
  };
}

function goalPromotionMetrics(traceRecords: readonly TraceRecord[]): GoalPromotionMetricCounts {
  const completed = traceRecords.filter((record) => record.event === "extraction.goals.completed");
  const skippedAsDuplicate = traceRecords.filter(
    (record) => record.event === "extraction.goals.skipped",
  );
  const classificationRejected = traceRecords.filter(
    (record) => record.event === "extraction.goals.rejected",
  );
  const classificationsPerTurn = zeroGoalPromotionClassificationCounts();

  for (const record of completed) {
    addGoalPromotionClassificationCounts(classificationsPerTurn, record);
  }

  return {
    goal_promotion_salvaged_promotions: completed.reduce(
      (sum, record) => sum + traceNumber(record, "salvaged_promotion_count"),
      0,
    ),
    goal_promotion_skipped_promotions: completed.reduce(
      (sum, record) => sum + traceNumber(record, "skipped_promotion_count"),
      0,
    ),
    goal_promotion_initial_step_downgraded: traceRecords.filter(
      (record) => record.event === "extraction.goals.transitioned",
    ).length,
    goal_promotion_dedup_skipped_extractor_signal: skippedAsDuplicate.filter(
      (record) => traceReason(record) === "extractor_signal",
    ).length,
    goal_promotion_dedup_skipped_embedding: skippedAsDuplicate.filter(
      (record) => traceReason(record) === "embedding",
    ).length,
    goal_promotion_dedup_degraded: traceRecords.filter(
      (record) => record.event === "extraction.goals.dedup.degraded",
    ).length,
    goal_promotion_classifications_per_turn: classificationsPerTurn,
    goal_promotion_rejected_classification: classificationRejected.filter(
      (record) => traceReason(record) === "non_durable_classification",
    ).length,
    goal_promotion_cap_rejections: classificationRejected.filter(
      (record) => traceReason(record) === "cap_exceeded",
    ).length,
  };
}

function sharedStateSemanticRevisionMetrics(
  traceRecords: readonly TraceRecord[],
): SharedStateArtifactSemanticRevisionMetricCounts {
  const completed = traceRecords.filter((record) => record.event === "semantic_revision.completed");
  const degraded = traceRecords.filter((record) => record.event === "semantic_revision.degraded");
  const cacheHits = traceRecords.filter(
    (record) => record.event === "semantic_revision.cache.completed",
  );
  const attemptedArtifactEntryIds = new Set(
    [...completed, ...degraded].map((record, index) =>
      typeof record.artifact_entry_id === "string"
        ? record.artifact_entry_id
        : `unidentified:${index}`,
    ),
  );

  return {
    decision_artifact_semantic_revisions_attempted: attemptedArtifactEntryIds.size,
    decision_artifact_semantic_revisions_completed_succeeded: completed.length,
    decision_artifact_semantic_nodes_marked_superseded: completed.reduce(
      (sum, record) => sum + traceNumber(record, "superseded_count"),
      0,
    ),
    decision_artifact_semantic_nodes_marked_contradicted: completed.reduce(
      (sum, record) => sum + traceNumber(record, "contradicted_count"),
      0,
    ),
    decision_artifact_semantic_revision_cache_hits: cacheHits.length,
  };
}

function commitmentRegenerationMetrics(input: {
  traceRecords: readonly TraceRecord[];
  cumulativeTraceRecords: readonly TraceRecord[];
}): CommitmentRegenerationMetricCounts {
  const attempted = input.traceRecords.filter(
    (record) => record.event === "commitment_guard.regeneration_requested",
  ).length;
  const succeeded = input.traceRecords.filter(
    (record) => record.event === "commitment_guard.regeneration_succeeded",
  ).length;
  const failed = input.traceRecords.filter(
    (record) => record.event === "commitment_guard.regeneration_failed",
  ).length;
  const attemptedTotal = input.cumulativeTraceRecords.filter(
    (record) => record.event === "commitment_guard.regeneration_requested",
  ).length;
  const succeededTotal = input.cumulativeTraceRecords.filter(
    (record) => record.event === "commitment_guard.regeneration_succeeded",
  ).length;
  const failedTotal = input.cumulativeTraceRecords.filter(
    (record) => record.event === "commitment_guard.regeneration_failed",
  ).length;
  const advisoryViolations = input.cumulativeTraceRecords.filter(
    (record) => record.event === "commitment_guard.advisory_violation_observed",
  );
  const advisoryViolationsByClass = zeroCounts(COMMITMENT_ENFORCEMENT_CLASSES);
  let advisoryViolationTotal = 0;

  for (const record of advisoryViolations) {
    const violationCount = Math.max(1, traceNumber(record, "violationCount"));
    const classes = traceStringArray(record, "commitmentEnforcementClasses").filter(
      (value): value is CommitmentEnforcementClass =>
        COMMITMENT_ENFORCEMENT_CLASSES.includes(value as CommitmentEnforcementClass),
    );
    const countedClasses = classes.length === 0 ? ["advisory" as const] : classes;

    advisoryViolationTotal += violationCount;
    for (const enforcementClass of countedClasses) {
      advisoryViolationsByClass[enforcementClass] += violationCount;
    }
  }

  return {
    commitment_regeneration_attempted_count: attempted,
    commitment_regeneration_succeeded_count: succeeded,
    commitment_regeneration_failed_count: failed,
    commitment_regeneration_attempted_total: attemptedTotal,
    commitment_regeneration_succeeded_total: succeededTotal,
    commitment_regeneration_failed_total: failedTotal,
    commitment_guard_advisory_violations_total: advisoryViolationTotal,
    commitment_guard_advisory_violations_by_class: advisoryViolationsByClass,
  };
}

function classificationDowngradeReason(value: string | null): ClassificationDowngradeReason | null {
  for (const reason of CLASSIFICATION_DOWNGRADE_REASONS) {
    if (value === reason) {
      return reason;
    }
  }

  return null;
}

function commitmentClassificationDowngradeMetrics(
  cumulativeTraceRecords: readonly TraceRecord[],
): CommitmentClassificationDowngradeMetricCounts {
  const downgraded = cumulativeTraceRecords.filter(
    (record) => record.event === "commitment_classification.downgraded",
  );
  const byReason = zeroCounts(CLASSIFICATION_DOWNGRADE_REASONS);
  const byKindTypeFromDomain = new Map<string, number>();

  for (const record of downgraded) {
    const reason = classificationDowngradeReason(traceReason(record));

    if (reason !== null) {
      byReason[reason] += 1;
    }

    const kind = traceString(record, "kind") ?? "unknown";
    const type = traceString(record, "type") ?? "unknown";
    const fromDomain = traceString(record, "original_critical_domain") ?? "none";
    const key = `${kind}/${type}/${fromDomain}`;

    byKindTypeFromDomain.set(key, (byKindTypeFromDomain.get(key) ?? 0) + 1);
  }

  return {
    commitments_critical_classification_downgraded_total: downgraded.length,
    commitments_critical_classification_downgraded_by_reason: byReason,
    commitments_critical_classification_downgraded_by_kind_type_from_domain:
      sortedNumberRecord(byKindTypeFromDomain),
  };
}

function semanticRevisionErrorMetrics(input: {
  traceRecords: readonly TraceRecord[];
  cumulativeTraceRecords: readonly TraceRecord[];
}): SemanticRevisionErrorMetricCounts {
  const degraded = input.traceRecords.filter(
    (record) => record.event === "shared_state.semantic_revision.degraded",
  );
  const cumulativeDegraded = input.cumulativeTraceRecords.filter(
    (record) => record.event === "shared_state.semantic_revision.degraded",
  );
  const totalsByReason = new Map<string, number>();

  for (const record of cumulativeDegraded) {
    incrementReasonCount(totalsByReason, traceReason(record) ?? "unknown");
  }

  return {
    semantic_revision_error_count: degraded.length,
    semantic_revision_skipped_due_to_error: degraded.reduce(
      (sum, record) => sum + Math.max(1, traceNumber(record, "skipped_due_to_error")),
      0,
    ),
    semantic_revision_error_total_by_reason: sortedNumberRecord(totalsByReason),
  };
}

function isSemanticRevisionDegradedEvent(record: TraceRecord): boolean {
  const event = canonicalTraceEventName(record.event);

  return (
    event === "semantic_revision.degraded" || event === "shared_state.semantic_revision.degraded"
  );
}

function semanticRevisionCumulativeMetrics(
  cumulativeTraceRecords: readonly TraceRecord[],
): SemanticRevisionCumulativeMetricCounts {
  const completed = cumulativeTraceRecords.filter(
    (record) => record.event === "semantic_revision.completed",
  );
  const degraded = cumulativeTraceRecords.filter(isSemanticRevisionDegradedEvent);
  const skippedOverCap = cumulativeTraceRecords.filter(
    (record) =>
      record.event === "semantic_revision.degraded" && traceReason(record) === "skipped_over_cap",
  );

  return {
    semantic_revision_calls_total: llmCompletedCount(
      cumulativeTraceRecords,
      SHARED_STATE_SEMANTIC_REVISION_LABEL,
    ),
    semantic_revision_candidates_reviewed_total: completed.reduce(
      (sum, record) => sum + traceNumber(record, "candidates_enumerated"),
      0,
    ),
    semantic_revision_superseded_total: completed.reduce(
      (sum, record) => sum + traceNumber(record, "superseded_count"),
      0,
    ),
    semantic_revision_contradicted_total: completed.reduce(
      (sum, record) => sum + traceNumber(record, "contradicted_count"),
      0,
    ),
    semantic_revision_degraded_total: degraded.length,
    semantic_revision_skipped_over_cap_total: skippedOverCap.length,
  };
}

function sharedStateCapPressureMetrics(
  traceRecords: readonly TraceRecord[],
): SharedStateCapPressureMetricCounts {
  const completed = traceRecords.filter(
    (record) => record.event === "shared_state.compile.completed",
  );
  const compileEvaluatedTurnIds = new Set(completed.map((record) => record.turnId));
  const atCapTurnIds = new Set<string>();
  let omittedRecentEntries = 0;
  let liveEntryStarvation = false;
  let newestEntriesReserved = 0;
  let liveStarvationWithReserved = false;
  let liveStarvationFinal = false;

  for (const record of completed) {
    const activeEntryCount = traceOptionalNumber(record, "artifact_active_entry_count");
    const maxActiveEntries =
      traceOptionalNumber(record, "artifact_max_active_entries") ??
      DEFAULT_SHARED_STATE_MAX_ACTIVE_ENTRIES;

    if (activeEntryCount !== null && activeEntryCount >= maxActiveEntries) {
      atCapTurnIds.add(record.turnId);
    }

    for (const kind of SHARED_STATE_RECENT_RENDER_KINDS) {
      omittedRecentEntries += traceObjectNumber(record, "omitted_by_kind", kind);
    }

    newestEntriesReserved += traceNumber(record, "newest_entries_reserved");

    const liveStarvedThisRecord =
      traceObjectNumber(record, "omitted_by_kind", "live") > 0 &&
      traceObjectNumber(record, "rendered_by_kind", "locked") > 0;

    if (liveStarvedThisRecord) {
      liveEntryStarvation = true;
    }

    const starvedWithReservedThisRecord =
      record.live_starvation_with_reserved === true || liveStarvedThisRecord;

    if (starvedWithReservedThisRecord) {
      liveStarvationWithReserved = true;
    }

    liveStarvationFinal = starvedWithReservedThisRecord;
  }

  return {
    shared_state_at_cap_turns: atCapTurnIds.size,
    shared_state_compile_evaluated_turns: compileEvaluatedTurnIds.size,
    shared_state_omitted_recent_entries: omittedRecentEntries,
    shared_state_live_entry_starvation: liveEntryStarvation,
    shared_state_newest_entries_reserved: newestEntriesReserved,
    shared_state_live_starvation_with_reserved: liveStarvationWithReserved,
    shared_state_live_starvation_ever: liveStarvationWithReserved,
    shared_state_live_starvation_final: liveStarvationFinal,
  };
}

function zeroSharedStateCompilerOperationCounts(): Record<
  (typeof SHARED_STATE_COMPILER_OPERATION_KINDS)[number],
  number
> {
  return zeroCounts(SHARED_STATE_COMPILER_OPERATION_KINDS);
}

function sharedStateCompilerOperationCountsByKind(
  traceRecords: readonly TraceRecord[],
): Record<string, number> {
  const counts = zeroSharedStateCompilerOperationCounts();

  for (const record of traceRecords) {
    if (record.event !== "shared_state.compile.completed" || record.applied !== true) {
      continue;
    }

    for (const kind of SHARED_STATE_COMPILER_OPERATION_KINDS) {
      counts[kind] += traceObjectNumber(record, "operation_counts_by_kind", kind);
    }
  }

  return counts;
}

function sharedStateAddToUpdateRatio(counts: Record<string, number>): number {
  const addCount = counts.add ?? 0;
  const consolidationCount = (counts.update ?? 0) + (counts.supersede ?? 0);

  if (addCount <= 0) {
    return 0;
  }

  return addCount / Math.max(1, consolidationCount);
}

function traceNestedOperationCounts(
  record: TraceRecord,
  objectKey: string,
): Record<string, Record<string, number>> {
  const value = record[objectKey];

  if (value === null || typeof value !== "object" || Array.isArray(value)) {
    return {};
  }

  const counts: Record<string, Record<string, number>> = {};

  for (const [key, nested] of Object.entries(value as Record<string, unknown>)) {
    if (nested === null || typeof nested !== "object" || Array.isArray(nested)) {
      continue;
    }

    const nestedCounts: Record<string, number> = {};
    for (const [operationKind, count] of Object.entries(nested as Record<string, unknown>)) {
      if (typeof count === "number" && Number.isFinite(count)) {
        nestedCounts[operationKind] = count;
      }
    }

    counts[key] = nestedCounts;
  }

  return counts;
}

function mergeNestedOperationCounts(
  left: Record<string, Record<string, number>>,
  right: Record<string, Record<string, number>>,
): Record<string, Record<string, number>> {
  const merged: Record<string, Record<string, number>> = { ...left };

  for (const [key, counts] of Object.entries(right)) {
    merged[key] = { ...(merged[key] ?? {}) };

    for (const [operationKind, count] of Object.entries(counts)) {
      merged[key][operationKind] = (merged[key][operationKind] ?? 0) + count;
    }
  }

  return Object.fromEntries(
    Object.entries(merged).sort(([leftKey], [rightKey]) => leftKey.localeCompare(rightKey)),
  );
}

function sharedStateOperationCountsByKey(
  traceRecords: readonly TraceRecord[],
): Record<string, Record<string, number>> {
  let counts: Record<string, Record<string, number>> = {};

  for (const record of traceRecords) {
    if (record.event !== "shared_state.compile.completed" || record.applied !== true) {
      continue;
    }

    counts = mergeNestedOperationCounts(
      counts,
      traceNestedOperationCounts(record, "operation_counts_by_state_key"),
    );
  }

  return counts;
}

function sharedStateAddToUpdateRatioByKey(
  countsByKey: Record<string, Record<string, number>>,
): Record<string, number> {
  return Object.fromEntries(
    Object.entries(countsByKey)
      .map(([key, counts]) => [key, sharedStateAddToUpdateRatio(counts)] as const)
      .sort(([leftKey], [rightKey]) => leftKey.localeCompare(rightKey)),
  );
}

function latestSharedStateEntriesByKey(
  traceRecords: readonly TraceRecord[],
): Record<string, number> {
  const latest = [...traceRecords]
    .reverse()
    .find((record) => record.event === "shared_state.compile.completed");

  return latest === undefined
    ? {}
    : traceObjectNumberEntries(latest, "shared_state_entries_by_key");
}

function latestSharedStateTopKeysByEntryCount(
  traceRecords: readonly TraceRecord[],
): Record<string, number> {
  const latest = [...traceRecords]
    .reverse()
    .find((record) => record.event === "shared_state.compile.completed");

  return latest === undefined
    ? {}
    : traceObjectNumberEntries(latest, "shared_state_top_keys_by_entry_count");
}

function sharedStateNewKeysPerCompile(
  traceRecords: readonly TraceRecord[],
): Record<string, number> {
  const counts = new Map<string, number>();

  for (const record of traceRecords) {
    if (record.event !== "shared_state.compile.completed") {
      continue;
    }

    const newKeyCount = traceNumber(record, "new_state_key_count");
    const bucket = String(newKeyCount);
    counts.set(bucket, (counts.get(bucket) ?? 0) + 1);
  }

  return sortedNumberRecord(counts);
}

function sharedStateNewKeysForTurn(traceRecords: readonly TraceRecord[], turnId: string): number {
  return traceRecords
    .filter(
      (record) => record.event === "shared_state.compile.completed" && record.turnId === turnId,
    )
    .reduce((sum, record) => sum + traceNumber(record, "new_state_key_count"), 0);
}

function latestSharedStateTraceNumber(traceRecords: readonly TraceRecord[], key: string): number {
  const latest = [...traceRecords]
    .reverse()
    .find((record) => record.event === "shared_state.compile.completed");

  return latest === undefined ? 0 : traceNumber(latest, key);
}

function sharedStateCompilerRepairFailedReasons(
  traceRecords: readonly TraceRecord[],
): Record<string, number> {
  const failedTurnIds = new Set(
    traceRecords
      .filter((record) => record.event === "shared_state.compile.repair_failed")
      .map((record) => traceString(record, "turnId"))
      .filter((turnId): turnId is string => turnId !== null),
  );
  const counts: Record<string, number> = {};

  for (const record of traceRecords) {
    if (
      record.event !== "shared_state.compile.completed" ||
      record.applied !== false ||
      failedTurnIds.size === 0
    ) {
      continue;
    }

    const turnId = traceString(record, "turnId");

    if (turnId === null || !failedTurnIds.has(turnId)) {
      continue;
    }

    const reasons = traceStringArray(record, "rejectionReasons");

    for (const reason of reasons.length === 0 ? ["unknown"] : reasons) {
      counts[reason] = (counts[reason] ?? 0) + 1;
    }
  }

  return Object.fromEntries(
    Object.entries(counts).sort(([left], [right]) => left.localeCompare(right)),
  );
}

function sharedStateCompilerHealthMetrics(
  traceRecords: readonly TraceRecord[],
  turnId: string,
): SharedStateCompilerHealthMetricCounts {
  const operationsTotalByKind = sharedStateCompilerOperationCountsByKind(traceRecords);
  const operationsByKey = sharedStateOperationCountsByKey(traceRecords);

  return {
    shared_state_compiler_max_tokens_total: traceRecords.filter(
      (record) =>
        record.event === "llm_call.completed" &&
        traceLabel(record) === SHARED_STATE_COMPILER_LABEL &&
        traceStopReason(record) === "max_tokens",
    ).length,
    shared_state_compiler_degraded_total: traceRecords.filter(
      (record) => record.event === "shared_state.compile.degraded",
    ).length,
    shared_state_compiler_repair_attempted_total: traceRecords.filter(
      (record) => record.event === "shared_state.compile.repair_attempted",
    ).length,
    shared_state_compiler_repair_succeeded_total: traceRecords.filter(
      (record) => record.event === "shared_state.compile.repair_succeeded",
    ).length,
    shared_state_compiler_repair_failed_total: traceRecords.filter(
      (record) => record.event === "shared_state.compile.repair_failed",
    ).length,
    shared_state_compiler_repair_failed_by_rejection_reason:
      sharedStateCompilerRepairFailedReasons(traceRecords),
    shared_state_compiler_operations_total_by_kind: operationsTotalByKind,
    shared_state_add_to_update_ratio: sharedStateAddToUpdateRatio(operationsTotalByKind),
    shared_state_entries_by_key: latestSharedStateEntriesByKey(traceRecords),
    shared_state_add_to_update_ratio_by_key: sharedStateAddToUpdateRatioByKey(operationsByKey),
    shared_state_top_keys_by_entry_count: latestSharedStateTopKeysByEntryCount(traceRecords),
    shared_state_add_rejected_cap_exceeded_total: traceRecords.filter(
      (record) => record.event === "shared_state.compile.add_rejected_cap_exceeded",
    ).length,
    shared_state_new_keys_per_compile: sharedStateNewKeysPerCompile(traceRecords),
    shared_state_new_keys_per_turn: sharedStateNewKeysForTurn(traceRecords, turnId),
    shared_state_keys_with_single_entry_only: latestSharedStateTraceNumber(
      traceRecords,
      "keys_with_single_entry_only",
    ),
    shared_state_similar_key_cluster_count: latestSharedStateTraceNumber(
      traceRecords,
      "similar_key_cluster_count",
    ),
    shared_state_add_rejected_near_duplicate_state_key_total: traceRecords.filter(
      (record) => record.event === "shared_state.compile.add_rejected_near_duplicate_state_key",
    ).length,
    shared_state_add_rejected_missing_new_key_reason_total: traceRecords.filter(
      (record) => record.event === "shared_state.compile.add_rejected_missing_new_key_reason",
    ).length,
  };
}

function traceString(record: TraceRecord, key: string): string | null {
  const value = record[key];

  return typeof value === "string" ? value : null;
}

function traceStringArray(record: TraceRecord, key: string): string[] {
  const value = record[key];

  if (!Array.isArray(value)) {
    return [];
  }

  return value.filter((item): item is string => typeof item === "string");
}

function traceClosureSpanKinds(record: TraceRecord): string[] {
  const spans = record.spans;

  if (!Array.isArray(spans)) {
    return [];
  }

  return spans.flatMap((span) => {
    if (span === null || typeof span !== "object" || Array.isArray(span)) {
      return [];
    }

    const kind = (span as { kind?: unknown }).kind;
    return typeof kind === "string" ? [kind] : [];
  });
}

function sessionReentryContinuityMetrics(
  traceRecords: readonly TraceRecord[],
): SessionReentryContinuityMetricCounts {
  const rendered = traceRecords.filter(
    (record) => record.event === "session_reentry.continuity.rendered",
  );
  const evaluated = traceRecords.filter(
    (record) => record.event === "session_reentry.continuity.evaluated",
  );
  const renderedByAudience = new Map<string, number>();

  for (const record of rendered) {
    const audienceEntityId = traceString(record, "audience_entity_id") ?? "unknown";
    renderedByAudience.set(audienceEntityId, (renderedByAudience.get(audienceEntityId) ?? 0) + 1);
  }

  return {
    session_reentry_card_rendered_total: rendered.length,
    session_reentry_card_rendered_by_audience: sortedNumberRecord(renderedByAudience),
    session_reentry_first_turn_with_existing_state_total: evaluated.filter(
      (record) => traceString(record, "status") === "rendered",
    ).length,
    session_reentry_first_turn_blank_audience_total: evaluated.filter(
      (record) => traceString(record, "status") === "blank_audience",
    ).length,
  };
}

function reviewResolverMetrics(traceRecords: readonly TraceRecord[]): ReviewResolverMetricCounts {
  const completed = traceRecords.filter((record) => record.event === "review_resolver.completed");
  const reviewQueueDecisions = traceRecords.filter(
    (record) => record.event === "review_queue.completed",
  );
  const enqueued = reviewQueueDecisions.filter((record) => record.decision === "enqueued").length;
  const resolved = reviewQueueDecisions.filter(
    (record) =>
      record.decision === "auto_accepted" ||
      record.decision === "manually_accepted" ||
      record.decision === "rejected",
  ).length;

  return {
    review_resolver_attempted: completed.reduce(
      (sum, record) => sum + traceNumber(record, "processed"),
      0,
    ),
    review_resolver_accepted: completed.reduce(
      (sum, record) => sum + traceNumber(record, "accepted"),
      0,
    ),
    review_resolver_dismissed: completed.reduce(
      (sum, record) => sum + traceNumber(record, "dismissed"),
      0,
    ),
    review_resolver_rejected: completed.reduce(
      (sum, record) => sum + traceNumber(record, "rejected"),
      0,
    ),
    review_resolver_needs_manual: completed.reduce(
      (sum, record) => sum + traceNumber(record, "needs_manual"),
      0,
    ),
    review_queue_enqueued_this_turn: enqueued,
    review_queue_resolved_this_turn: resolved,
    review_queue_drain_rate: enqueued === 0 ? null : resolved / enqueued,
  };
}

function llmCompletedCount(traceRecords: readonly TraceRecord[], label: string): number {
  return traceRecords.filter(
    (record) => record.event === "llm_call.completed" && traceLabel(record) === label,
  ).length;
}

function sortedNumberRecord(counts: ReadonlyMap<string, number>): Record<string, number> {
  return Object.fromEntries(
    [...counts.entries()].sort(([left], [right]) => left.localeCompare(right)),
  );
}

function incrementLabelCount(counts: Map<string, number>, label: string): void {
  counts.set(label, (counts.get(label) ?? 0) + 1);
}

function incrementReasonCount(counts: Map<string, number>, reason: string): void {
  counts.set(reason, (counts.get(reason) ?? 0) + 1);
}

function sortedContextCounts(counts: Record<string, number> | undefined): Record<string, number> {
  if (counts === undefined) {
    return {};
  }

  return Object.fromEntries(
    Object.entries(counts)
      .filter(([, value]) => Number.isFinite(value) && value > 0)
      .sort(([left], [right]) => left.localeCompare(right)),
  );
}

function isExtractorMaxTokensStop(record: TraceRecord): boolean {
  const label = traceLabel(record);

  return (
    record.event === "llm_call.completed" &&
    traceStopReason(record) === "max_tokens" &&
    label !== null &&
    isExtractorMaxTokenLlmLabel(label)
  );
}

const EXTRACTOR_DEGRADED_EVENT_LABELS: Readonly<Record<string, string>> = {
  "closure_loop.degraded": "closure_loop_classifier",
  "extraction.actions.degraded": "action_state_extractor",
  "extraction.commitments.degraded": "corrective_preference_extractor",
  "extraction.goals.degraded": "goal_promotion_extractor",
  "frame_anomaly.degraded": "frame_anomaly_classifier",
  "semantic_extractor.degraded": "semantic_extractor",
};

function extractorDegradedLabel(record: TraceRecord): string | null {
  const fallbackLabel = EXTRACTOR_DEGRADED_EVENT_LABELS[record.event];

  if (fallbackLabel === undefined) {
    return null;
  }

  return traceLabel(record) ?? fallbackLabel;
}

function extractorMaxTokensTotalsByLabel(
  traceRecords: readonly TraceRecord[],
): Record<string, number> {
  const counts = new Map<string, number>();

  for (const record of traceRecords) {
    if (!isExtractorMaxTokensStop(record)) {
      continue;
    }

    const label = traceLabel(record);

    if (label === null) {
      continue;
    }

    incrementLabelCount(counts, label);
  }

  return sortedNumberRecord(counts);
}

function extractorDegradedTotalsByLabel(
  traceRecords: readonly TraceRecord[],
): Record<string, number> {
  const counts = new Map<string, number>();

  for (const record of traceRecords) {
    const label = extractorDegradedLabel(record);

    if (label === null) {
      continue;
    }

    incrementLabelCount(counts, label);
  }

  return sortedNumberRecord(counts);
}

function extractorHealthMetrics(input: {
  traceRecords: readonly TraceRecord[];
  cumulativeTraceRecords: readonly TraceRecord[];
}): ExtractorHealthMetricCounts {
  return {
    closure_loop_completed_count: llmCompletedCount(input.traceRecords, "closure_loop_classifier"),
    closure_loop_degraded_count: input.traceRecords.filter(
      (record) => record.event === "closure_loop.degraded",
    ).length,
    corrective_preference_completed_count: llmCompletedCount(
      input.traceRecords,
      "corrective_preference_extractor",
    ),
    corrective_preference_degraded_count: input.traceRecords.filter(
      (record) => record.event === "extraction.commitments.degraded",
    ).length,
    extractor_max_tokens_stop_count: input.traceRecords.filter(isExtractorMaxTokensStop).length,
    extractor_max_tokens_total_by_label: extractorMaxTokensTotalsByLabel(
      input.cumulativeTraceRecords,
    ),
    extractor_degraded_total_by_label: extractorDegradedTotalsByLabel(input.cumulativeTraceRecords),
  };
}

function closurePressureMetrics(
  traceRecords: readonly TraceRecord[],
): ClosurePressureMetricCounts {
  const completed = traceRecords.filter(
    (record) => record.event === "closure_response_guard.completed",
  );
  const mixedBySpanKind = new Map<string, number>();

  for (const record of completed) {
    if (traceString(record, "response_shape") !== "mixed") {
      continue;
    }

    for (const kind of traceClosureSpanKinds(record)) {
      incrementLabelCount(mixedBySpanKind, kind);
    }
  }

  return {
    closure_pressure_mixed_observed_total: completed.filter(
      (record) => traceReason(record) === "mixed_closure_observed",
    ).length,
    closure_pressure_closure_only_suppressed_total: completed.filter(
      (record) =>
        traceString(record, "mode") === "enforce" &&
        traceString(record, "verdict") === "suppressed" &&
        traceString(record, "response_shape") === "closure_only" &&
        traceReason(record) === "closure_pressure_only",
    ).length,
    closure_pressure_mixed_passed_no_active_preference_total: completed.filter(
      (record) =>
        traceString(record, "response_shape") === "mixed" &&
        traceReason(record) === "no_active_closure_preference",
    ).length,
    closure_pressure_mixed_by_span_kind: sortedNumberRecord(mixedBySpanKind),
  };
}

function semanticRelationshipLabelRejectionRecords(
  traceRecords: readonly TraceRecord[],
): TraceRecord[] {
  return traceRecords.filter(
    (record) =>
      record.event === "semantic_insert.skipped" &&
      traceKind(record) === "node" &&
      traceReason(record) === "relationship_label_ungrounded",
  );
}

function semanticRelationshipLabelRejectionsByLabel(
  traceRecords: readonly TraceRecord[],
): Record<string, number> {
  const counts: Record<string, number> = {};

  for (const record of semanticRelationshipLabelRejectionRecords(traceRecords)) {
    const labels = Array.isArray(record.protected_relationship_labels)
      ? record.protected_relationship_labels.filter(
          (label): label is string => typeof label === "string",
        )
      : [];

    for (const label of labels.length === 0 ? ["unknown"] : labels) {
      counts[label] = (counts[label] ?? 0) + 1;
    }
  }

  return counts;
}

function relationshipLabelRejectionRecords(
  traceRecords: readonly TraceRecord[],
  event: string,
): TraceRecord[] {
  return traceRecords.filter((record) => record.event === event);
}

function relationshipLabelRejectionsByLabel(
  traceRecords: readonly TraceRecord[],
  event: string,
): Record<string, number> {
  const counts: Record<string, number> = {};

  for (const record of relationshipLabelRejectionRecords(traceRecords, event)) {
    const labels = Array.isArray(record.protected_relationship_labels)
      ? record.protected_relationship_labels.filter(
          (label): label is string => typeof label === "string",
        )
      : [];

    for (const label of labels.length === 0 ? ["unknown"] : labels) {
      counts[label] = (counts[label] ?? 0) + 1;
    }
  }

  return counts;
}

function semanticMemoryWriteGateMetrics(input: {
  traceRecords: readonly TraceRecord[];
  cumulativeTraceRecords: readonly TraceRecord[];
}): SemanticMemoryWriteGateMetricCounts {
  const intervalRejections = semanticRelationshipLabelRejectionRecords(input.traceRecords);
  const cumulativeRejections = semanticRelationshipLabelRejectionRecords(
    input.cumulativeTraceRecords,
  );
  const sharedStateCumulativeRejections = relationshipLabelRejectionRecords(
    input.cumulativeTraceRecords,
    "shared_state.compile.label_ungrounded",
  );
  const commitmentCumulativeRejections = relationshipLabelRejectionRecords(
    input.cumulativeTraceRecords,
    "corrective_preference.candidate_rejected_ungrounded",
  );

  return {
    semantic_nodes_rejected_ungrounded_label_count: intervalRejections.length,
    semantic_nodes_rejected_ungrounded_label_total: cumulativeRejections.length,
    semantic_nodes_rejected_ungrounded_label_by_label: semanticRelationshipLabelRejectionsByLabel(
      input.cumulativeTraceRecords,
    ),
    shared_state_operations_rejected_ungrounded_label_total: sharedStateCumulativeRejections.length,
    shared_state_operations_rejected_ungrounded_label_by_label: relationshipLabelRejectionsByLabel(
      input.cumulativeTraceRecords,
      "shared_state.compile.label_ungrounded",
    ),
    commitment_candidates_rejected_ungrounded_label_total: commitmentCumulativeRejections.length,
    commitment_candidates_rejected_ungrounded_label_by_label: relationshipLabelRejectionsByLabel(
      input.cumulativeTraceRecords,
      "corrective_preference.candidate_rejected_ungrounded",
    ),
  };
}

function zeroCounts<K extends string>(keys: readonly K[]): Record<K, number> {
  return Object.fromEntries(keys.map((key) => [key, 0])) as Record<K, number>;
}

function zeroCriticalCommitmentsByKindTypeDomain(): Record<
  CommitmentKind,
  Record<CommitmentType, Record<CommitmentCriticalDomain, number>>
> {
  return Object.fromEntries(
    COMMITMENT_KINDS.map((kind) => [
      kind,
      Object.fromEntries(
        COMMITMENT_TYPES.map((type) => [type, zeroCounts(COMMITMENT_CRITICAL_DOMAINS)]),
      ),
    ]),
  ) as Record<CommitmentKind, Record<CommitmentType, Record<CommitmentCriticalDomain, number>>>;
}

function criticalCommitmentsByKindTypeDomain(
  commitments: readonly CommitmentRecord[],
): Record<CommitmentKind, Record<CommitmentType, Record<CommitmentCriticalDomain, number>>> {
  const counts = zeroCriticalCommitmentsByKindTypeDomain();

  for (const commitment of commitments) {
    if (effectiveCommitmentEnforcementClass(commitment) !== "critical") {
      continue;
    }

    const criticalDomain = effectiveCommitmentCriticalDomain(commitment);

    if (criticalDomain === null) {
      continue;
    }

    counts[commitment.kind][commitment.type][criticalDomain] += 1;
  }

  return counts;
}

function zeroActionCreationCounts(): Record<ActionRecordCreationSource, number> {
  return zeroCounts(ACTION_CREATION_SOURCES);
}

function zeroActionCandidateClassificationCounts(): Record<
  ActionCandidateClassificationMetricKey,
  number
> {
  return zeroCounts(ACTION_CANDIDATE_CLASSIFICATION_METRIC_KEYS);
}

function actionCandidateClassificationMetricValue(
  value: string,
): ActionCandidateClassificationMetricKey | null {
  if (value === "invalid_classification") {
    return value;
  }

  for (const classification of ACTION_CANDIDATE_CLASSIFICATIONS) {
    if (value === classification) {
      return classification as ActionCandidateClassification;
    }
  }

  return null;
}

function addActionCandidateClassificationCounts(
  target: Record<ActionCandidateClassificationMetricKey, number>,
  record: TraceRecord,
): void {
  const counts = record.classification_counts;

  if (counts === null || typeof counts !== "object" || Array.isArray(counts)) {
    return;
  }

  for (const [rawKey, rawValue] of Object.entries(counts)) {
    const key = actionCandidateClassificationMetricValue(rawKey);

    if (key === null || typeof rawValue !== "number" || !Number.isFinite(rawValue)) {
      continue;
    }

    target[key] += rawValue;
  }
}

function zeroGoalPromotionClassificationCounts(): Record<
  GoalPromotionClassificationMetricKey,
  number
> {
  return zeroCounts(GOAL_PROMOTION_CLASSIFICATION_METRIC_KEYS);
}

function goalPromotionClassificationMetricValue(
  value: string,
): GoalPromotionClassificationMetricKey | null {
  if (value === "invalid_classification") {
    return value;
  }

  for (const classification of GOAL_PROMOTION_CLASSIFICATIONS) {
    if (value === classification) {
      return classification as GoalPromotionClassification;
    }
  }

  return null;
}

function addGoalPromotionClassificationCounts(
  target: Record<GoalPromotionClassificationMetricKey, number>,
  record: TraceRecord,
): void {
  const counts = record.classification_counts;

  if (counts === null || typeof counts !== "object" || Array.isArray(counts)) {
    return;
  }

  for (const [rawKey, rawValue] of Object.entries(counts)) {
    const key = goalPromotionClassificationMetricValue(rawKey);

    if (key === null || typeof rawValue !== "number" || !Number.isFinite(rawValue)) {
      continue;
    }

    target[key] += rawValue;
  }
}

function actionCreationCountsFromRepository(
  borg: Borg,
): Record<ActionRecordCreationSource, number> {
  const source = borg.actions as unknown as {
    getCreationCountsBySource?: () => Partial<Record<ActionRecordCreationSource, number>>;
  };
  const counts = source.getCreationCountsBySource?.() ?? {};

  return {
    ...zeroActionCreationCounts(),
    ...counts,
  };
}

function diffActionCreationCounts(input: {
  previous: Record<ActionRecordCreationSource, number>;
  current: Record<ActionRecordCreationSource, number>;
}): Record<ActionRecordCreationSource, number> {
  const diff = zeroActionCreationCounts();

  for (const source of ACTION_CREATION_SOURCES) {
    diff[source] = Math.max(0, (input.current[source] ?? 0) - (input.previous[source] ?? 0));
  }

  return diff;
}

function actionCreationCountTotal(counts: Record<ActionRecordCreationSource, number>): number {
  return ACTION_CREATION_SOURCES.reduce((sum, source) => sum + counts[source], 0);
}

function actionCountFromRepository(borg: Borg, method: string): number | null {
  const actions = borg.actions as unknown as Record<string, unknown>;
  const candidate = actions[method];

  if (typeof candidate !== "function") {
    return null;
  }

  const value = candidate.call(actions);

  return typeof value === "number" && Number.isFinite(value) ? value : null;
}

function activeActionCountFromStateCounts(counts: Record<ActionState, number>): number {
  return ACTIVE_ACTION_STATES.reduce((sum, state) => sum + (counts[state] ?? 0), 0);
}

function activeActionActorSplit(
  actions: readonly Pick<ActionRecord, "actor" | "audience_entity_id">[],
): Pick<
  MetricsRow,
  "borg_owned_active_actions" | "participant_owned_active_actions" | "group_owned_active_actions"
> {
  let borgOwned = 0;
  let participantOwned = 0;
  let groupOwned = 0;

  for (const action of actions) {
    if (action.actor === "borg") {
      borgOwned += 1;
      continue;
    }

    if (action.actor !== "user" && action.actor === action.audience_entity_id) {
      groupOwned += 1;
      continue;
    }

    participantOwned += 1;
  }

  return {
    borg_owned_active_actions: borgOwned,
    participant_owned_active_actions: participantOwned,
    group_owned_active_actions: groupOwned,
  };
}

function actionRecordCountForStates(
  counts: Record<ActionState, number>,
  states: readonly ActionState[],
): number {
  return states.reduce((sum, state) => sum + (counts[state] ?? 0), 0);
}

function actionRatio(numerator: number, denominator: number): number {
  return denominator <= 0 ? 0 : numerator / denominator;
}

function borgOwnedActionCount(actions: readonly Pick<ActionRecord, "actor">[]): number {
  return actions.filter((action) => action.actor === "borg").length;
}

function dormantActionCount(
  actions: readonly Pick<
    ActionRecord,
    | "actor"
    | "audience_entity_id"
    | "state"
    | "scheduled_at"
    | "last_referenced_turn_counter"
    | "last_referenced_turn_global"
  >[],
  turnCounter: number,
): number {
  return actions.filter(
    (action) =>
      action.actor !== "borg" &&
      !(action.actor !== "user" && action.actor === action.audience_entity_id) &&
      action.state !== "scheduled" &&
      action.scheduled_at === null &&
      lastReferencedActionLifecycleTurn(action) !== null &&
      turnCounter - (lastReferencedActionLifecycleTurn(action) ?? turnCounter) >=
        ACTION_DORMANT_AFTER_INACTIVE_TURNS,
  ).length;
}

function archiveInactiveTurnBucket(
  inactiveTurns: number,
): (typeof ACTION_INACTIVE_TURN_BUCKETS)[number] {
  if (inactiveTurns < 15) {
    return "0-15";
  }

  if (inactiveTurns < 20) {
    return "15-20";
  }

  if (inactiveTurns < 30) {
    return "20-30";
  }

  return "30+";
}

function emptyArchiveInactiveTurnDistribution(): Record<
  (typeof ACTION_INACTIVE_TURN_BUCKETS)[number],
  number
> {
  return zeroCounts(ACTION_INACTIVE_TURN_BUCKETS);
}

function actionArchiveVisibilityMetrics(
  actions: readonly Pick<
    ActionRecord,
    | "actor"
    | "audience_entity_id"
    | "state"
    | "scheduled_at"
    | "last_referenced_turn_counter"
    | "last_referenced_turn_global"
  >[],
  turnCounter: number,
  archiveAfterInactiveTurns: number,
  traceRecords: readonly TraceRecord[],
): Pick<
  MemoryBandMetricCounts,
  | "dormant_not_archive_eligible_count"
  | "dormant_archive_eligible_count"
  | "archive_oldest_inactive_turns"
  | "archive_inactive_turn_distribution"
  | "archive_archivable_count"
  | "archive_skipped_borg_owned"
  | "archive_skipped_due_date"
  | "archive_skipped_below_threshold"
  | "archive_skipped_other"
  | "archive_oldest_archivable_inactive_turns"
> {
  const distribution = emptyArchiveInactiveTurnDistribution();
  let oldestInactiveTurns = 0;
  let archiveOldestArchivableInactiveTurns = 0;
  let archiveArchivableCount = 0;
  let archiveSkippedBorgOwned = 0;
  let archiveSkippedDueDate = 0;
  let archiveSkippedBelowThreshold = 0;
  let archiveSkippedOther = 0;
  const scanRecords = traceRecords.filter(
    (record) => record.event === "action_archive_scan.completed",
  );

  for (const record of scanRecords) {
    archiveArchivableCount += traceNumber(record, "eligible_count");
    archiveSkippedBorgOwned += traceObjectNumber(record, "skipped_by_reason", "borg_owned");
    archiveSkippedDueDate += traceObjectNumber(record, "skipped_by_reason", "scheduled_or_due");
    archiveSkippedBelowThreshold += traceObjectNumber(
      record,
      "skipped_by_reason",
      "below_inactive_threshold",
    );
    archiveOldestArchivableInactiveTurns = Math.max(
      archiveOldestArchivableInactiveTurns,
      traceNumber(record, "oldest_eligible_inactive_turns"),
    );
    oldestInactiveTurns = Math.max(
      oldestInactiveTurns,
      traceNumber(record, "oldest_inactive_turns"),
    );

    const skippedByReason = traceObjectNumberEntries(record, "skipped_by_reason");
    for (const [reason, count] of Object.entries(skippedByReason)) {
      if (
        reason === "borg_owned" ||
        reason === "scheduled_or_due" ||
        reason === "below_inactive_threshold"
      ) {
        continue;
      }

      archiveSkippedOther += count;
    }
  }

  for (const action of actions) {
    const classification = classifyActionArchiveCandidate(action, {
      turnCounter,
      archiveAfterTurns: archiveAfterInactiveTurns,
    });

    if (classification.inactiveTurns === undefined) {
      if (scanRecords.length > 0) {
        continue;
      }

      if (classification.status === "skipped") {
        switch (classification.reason) {
          case "borg_owned":
            archiveSkippedBorgOwned += 1;
            break;
          case "scheduled_or_due":
            archiveSkippedDueDate += 1;
            break;
          case "below_inactive_threshold":
            archiveSkippedBelowThreshold += 1;
            break;
          default:
            archiveSkippedOther += 1;
            break;
        }
      }

      continue;
    }

    const inactiveTurns = Math.max(0, classification.inactiveTurns);
    distribution[archiveInactiveTurnBucket(inactiveTurns)] += 1;

    if (scanRecords.length === 0) {
      oldestInactiveTurns = Math.max(oldestInactiveTurns, inactiveTurns);

      if (classification.status === "eligible") {
        archiveArchivableCount += 1;
        archiveOldestArchivableInactiveTurns = Math.max(
          archiveOldestArchivableInactiveTurns,
          inactiveTurns,
        );
      } else if (classification.reason === "below_inactive_threshold") {
        archiveSkippedBelowThreshold += 1;
      }
    }
  }

  return {
    dormant_not_archive_eligible_count: archiveSkippedBelowThreshold,
    dormant_archive_eligible_count: archiveArchivableCount,
    archive_oldest_inactive_turns: oldestInactiveTurns,
    archive_inactive_turn_distribution: distribution,
    archive_archivable_count: archiveArchivableCount,
    archive_skipped_borg_owned: archiveSkippedBorgOwned,
    archive_skipped_due_date: archiveSkippedDueDate,
    archive_skipped_below_threshold: archiveSkippedBelowThreshold,
    archive_skipped_other: archiveSkippedOther,
    archive_oldest_archivable_inactive_turns: archiveOldestArchivableInactiveTurns,
  };
}

function emptyActionPromptSalienceSummary(): ActionPromptSalienceSummary {
  return {
    promptSalientActionsTotal: 0,
    borgOwnedSalientActiveActions: 0,
    participantOwnedSalientActiveActions: 0,
    staleActionsOmittedFromPrompt: 0,
  };
}

function metricsScopeResolver(sessionId: SessionId): ScopeResolver {
  return {
    currentSessionId: sessionId,
    streamEntriesById: new Map(),
    streamOrderById: new Map(),
    episodeScopesById: new Map(),
    episodeSourceStreamIdsById: new Map(),
  };
}

async function actionPromptSalienceSummary(input: {
  actions: readonly ActionRecord[];
  borg: Borg;
  sessionId: SessionId;
  turnCounter: number;
}): Promise<ActionPromptSalienceSummary> {
  if (input.actions.length === 0) {
    return emptyActionPromptSalienceSummary();
  }

  const actionRepository = input.borg.actions as unknown as ActionLedgerRepository;
  const threads = await buildActionThreads({
    records: input.actions,
    repository: actionRepository,
    resolver: metricsScopeResolver(input.sessionId),
    similarityThreshold: DEFAULT_ACTION_THREAD_SIMILARITY_THRESHOLD,
  });
  const threadsWithSalience = threads.flatMap((thread) => {
    const salienceClass = actionSalienceClass({
      thread,
      currentTurnCounter: input.turnCounter,
    });

    return salienceClass === null ? [] : [{ ...thread, salienceClass }];
  });

  return summarizeActionPromptSalience(threadsWithSalience);
}

function reviewQueueOpenCountByType(borg: Borg): Record<ReviewKind, number> {
  const counts = zeroCounts(REVIEW_KINDS);

  for (const item of borg.review.list({ openOnly: true })) {
    counts[item.kind] += 1;
  }

  return counts;
}

function semanticNodeStatusCounts(
  nodes: readonly Pick<SemanticNode, "status">[],
): Record<SemanticNodeStatus, number> {
  const counts = zeroCounts<SemanticNodeStatus>(SEMANTIC_NODE_STATUSES);
  const knownStatuses = new Set<string>(SEMANTIC_NODE_STATUSES);

  for (const node of nodes) {
    if (!knownStatuses.has(node.status)) {
      continue;
    }

    counts[node.status] += 1;
  }

  return counts;
}

function openQuestionResolvedCount(borg: Borg): number {
  return borg.identity
    .listEvents({
      recordType: OPEN_QUESTION_RECORD_TYPE,
      limit: LARGE_COUNT_LIMIT,
    })
    .filter(
      (event) =>
        identityValueStatus(event.old_value) !== RESOLVED_STATUS &&
        identityValueStatus(event.new_value) === RESOLVED_STATUS,
    ).length;
}

const OPEN_QUESTION_SOURCE_BUCKETS = [
  "user_question",
  "borg_inferred",
  "review_promoted",
  "unknown",
] as const;
const OPEN_QUESTION_STATUS_AGE_BUCKETS = [
  "<3_turns",
  "3-10_turns",
  "10-30_turns",
  "30+_turns",
] as const;

function openQuestionSourceBucket(
  source: OpenQuestionSource | undefined,
): (typeof OPEN_QUESTION_SOURCE_BUCKETS)[number] {
  if (source === "user") {
    return "user_question";
  }

  if (source === "contradiction" || source === "overseer") {
    return "review_promoted";
  }

  if (source === undefined) {
    return "unknown";
  }

  return "borg_inferred";
}

function openQuestionsBySource(
  questions: readonly Partial<OpenQuestion>[],
): Record<string, number> {
  const counts = zeroCounts(OPEN_QUESTION_SOURCE_BUCKETS);

  for (const question of questions) {
    counts[openQuestionSourceBucket(question.source)] += 1;
  }

  return counts;
}

function openQuestionAgeBucket(ageTurns: number): string {
  if (ageTurns < 3) {
    return "<3_turns";
  }

  if (ageTurns <= 10) {
    return "3-10_turns";
  }

  if (ageTurns <= 30) {
    return "10-30_turns";
  }

  return "30+_turns";
}

function openQuestionCreatedTurn(input: {
  createdAt: number;
  previousRows: readonly MetricsRow[];
  currentTurnCounter: number;
  currentTs: number;
}): number {
  const timeline = [
    ...input.previousRows.map((row) => ({
      ts: row.ts,
      turnCounter: row.turn_counter,
    })),
    {
      ts: input.currentTs,
      turnCounter: input.currentTurnCounter,
    },
  ].sort((left, right) => left.ts - right.ts || left.turnCounter - right.turnCounter);
  const firstObserved = timeline.find((row) => row.ts >= input.createdAt);

  return firstObserved?.turnCounter ?? input.currentTurnCounter;
}

function openQuestionsByStatusAge(input: {
  questions: readonly Partial<OpenQuestion>[];
  previousRows: readonly MetricsRow[];
  currentTurnCounter: number;
  currentTs: number;
}): Record<string, number> {
  const counts: Record<string, number> = {};

  for (const status of OPEN_QUESTION_STATUSES) {
    for (const bucket of OPEN_QUESTION_STATUS_AGE_BUCKETS) {
      counts[`${status}:${bucket}`] = 0;
    }
  }

  for (const question of input.questions) {
    const status = OPEN_QUESTION_STATUSES.includes(question.status as OpenQuestionStatus)
      ? (question.status as OpenQuestionStatus)
      : "open";
    const createdAt =
      typeof question.created_at === "number" && Number.isFinite(question.created_at)
        ? question.created_at
        : input.currentTs;
    const createdTurn = openQuestionCreatedTurn({
      createdAt,
      previousRows: input.previousRows,
      currentTurnCounter: input.currentTurnCounter,
      currentTs: input.currentTs,
    });
    const ageTurns = Math.max(0, input.currentTurnCounter - createdTurn);
    const key = `${status}:${openQuestionAgeBucket(ageTurns)}`;
    counts[key] = (counts[key] ?? 0) + 1;
  }

  return counts;
}

function openQuestionsPromotedFromReviewItems(questions: readonly Partial<OpenQuestion>[]): number {
  return questions.filter(
    (question) => question.source === "contradiction" || question.source === "overseer",
  ).length;
}

function openQuestionsRenderedToFinalizer(traceRecords: readonly TraceRecord[]): number {
  return traceRecords
    .filter((record) => record.event === "evidence_ledger.completed")
    .reduce((sum, record) => sum + traceObjectNumber(record, "entry_counts", "open_questions"), 0);
}

function identityValueStatus(value: unknown): unknown {
  if (value === null || typeof value !== "object" || Array.isArray(value)) {
    return undefined;
  }

  return (value as { status?: unknown }).status;
}

export class MetricsCapture {
  private readonly filepath: string;
  private readonly tracePath?: string;
  private readonly scenarioKey?: string;
  private readonly semanticRevisionVerdictCacheSize: () => number;
  private previousSemanticNodeCount?: number;
  private previousSemanticEdgeCount?: number;
  private previousTraceRecordCount = 0;
  private previousActionCreationCountsBySource: Record<ActionRecordCreationSource, number> =
    zeroActionCreationCounts();
  private readonly completedActionIdsSeen = new Set<ActionId>();
  private readonly capturedRows: MetricsRow[] = [];
  private readonly healthWarnings: SimulatorHealthWarning[] = [];
  private readonly activeHealthWarningKinds = new Set<SimulatorHealthWarningKind>();

  constructor(filepath: string, options: MetricsCaptureOptions = {}) {
    this.filepath = filepath;
    this.tracePath = options.tracePath;
    this.scenarioKey = options.scenarioKey;
    this.semanticRevisionVerdictCacheSize =
      options.semanticRevisionVerdictCacheSize ?? sharedStateSemanticRevisionVerdictCacheSize;
  }

  listHealthWarnings(): SimulatorHealthWarning[] {
    return this.healthWarnings.map((warning) => ({ ...warning }));
  }

  listRows(): MetricsRow[] {
    return this.capturedRows.map((row) => ({ ...row }));
  }

  finalizeLastRow(patch: Partial<MetricsRow>): MetricsRow {
    const previous = this.capturedRows.at(-1);

    if (previous === undefined) {
      throw new Error("Cannot finalize simulator metrics before a row is captured");
    }

    const updated: MetricsRow = {
      ...previous,
      ...patch,
    };

    this.capturedRows[this.capturedRows.length - 1] = updated;
    const lines = readFileSync(this.filepath, "utf8").trimEnd().split(/\r?\n/);
    lines[lines.length - 1] = JSON.stringify(updated);
    writeFileAtomic(this.filepath, `${lines.join("\n")}\n`);
    return { ...updated };
  }

  private recordHealthWarnings(row: MetricsRow): void {
    this.capturedRows.push(row);
    const warnings = simulatorHealthWarningsForRows(this.capturedRows, {
      scenarioKey: this.scenarioKey,
    });
    const currentKinds = new Set(warnings.map((warning) => warning.kind));
    const risingWarnings = warnings.filter(
      (warning) => !this.activeHealthWarningKinds.has(warning.kind),
    );

    for (const kind of [...this.activeHealthWarningKinds]) {
      if (!currentKinds.has(kind)) {
        this.activeHealthWarningKinds.delete(kind);
      }
    }

    for (const kind of currentKinds) {
      this.activeHealthWarningKinds.add(kind);
    }

    if (risingWarnings.length === 0) {
      return;
    }

    this.healthWarnings.push(...risingWarnings);

    if (this.tracePath === undefined) {
      return;
    }

    for (const warning of risingWarnings) {
      appendJsonlLine(
        this.tracePath,
        `${JSON.stringify({
          ts: Date.now(),
          wallMs: performance.now(),
          turnId: warning.turnId,
          event: "simulator_health.degraded",
          artifact: "simulator",
          warning_kind: warning.kind,
          turn_counter: warning.turn_counter,
          threshold: warning.threshold,
          observed_value: warning.observed_value,
          ...(warning.label === undefined ? {} : { label: warning.label }),
          ...(warning.window_start_turn === undefined
            ? {}
            : { window_start_turn: warning.window_start_turn }),
          ...(warning.window_turns === undefined ? {} : { window_turns: warning.window_turns }),
        })}\n`,
      );
    }
  }

  private async captureMemoryBandMetrics(
    borg: Borg,
    sessionId: SessionId,
    turnCounter: number,
    archiveAfterInactiveTurns: number,
    archiveTraceRecords: readonly TraceRecord[],
  ): Promise<MemoryBandMetricCounts> {
    const actionRecordCountByState = borg.actions.countByState();
    const allActions = borg.actions.list({ limit: LARGE_COUNT_LIMIT });
    const promptSourceActions = borg.actions.list({
      limit: DEFAULT_ACTION_THREAD_SOURCE_RECORD_LIMIT,
    });
    const activeActions = borg.actions.list({
      states: ACTIVE_ACTION_STATES,
      limit: LARGE_COUNT_LIMIT,
    });
    const archiveScanCandidateActions = borg.actions.list({
      states: ACTION_ARCHIVE_ACTIVE_STATES,
      limit: ACTION_ARCHIVE_SCAN_LIMIT,
    });
    const activeActorSplit = activeActionActorSplit(activeActions);
    const promptSalienceSummary = await actionPromptSalienceSummary({
      actions: promptSourceActions,
      borg,
      sessionId,
      turnCounter,
    });
    const currentActionCreationCounts = actionCreationCountsFromRepository(borg);
    const actionCreationSourcePerTurn = diffActionCreationCounts({
      previous: this.previousActionCreationCountsBySource,
      current: currentActionCreationCounts,
    });
    const completedActionIds = borg.actions.listCompletedIds();
    const recentCompletedActionCount = completedActionIds.filter(
      (id) => !this.completedActionIdsSeen.has(id),
    ).length;
    const workingMemory = borg.workmem.load(sessionId);

    for (const id of completedActionIds) {
      this.completedActionIdsSeen.add(id);
    }

    const totalActions = borg.actions.count();
    const dormantActionsTotal = dormantActionCount(activeActions, turnCounter);
    const archiveVisibilityMetrics = actionArchiveVisibilityMetrics(
      archiveScanCandidateActions,
      turnCounter,
      archiveAfterInactiveTurns,
      archiveTraceRecords,
    );
    const terminalActionCount = actionRecordCountForStates(
      actionRecordCountByState,
      TERMINAL_ACTION_STATES,
    );
    const activeCommitments = borg.commitments.list({ activeOnly: true });
    const commitmentsByEnforcementClass = {
      ...zeroCounts(COMMITMENT_ENFORCEMENT_CLASSES),
      ...borg.commitments.countActiveByEnforcementClass(),
    };

    return {
      action_record_count_total: totalActions,
      action_record_count_by_state: {
        ...zeroCounts<ActionState>(ACTION_STATES),
        ...actionRecordCountByState,
      },
      action_record_count_committed_to_do: actionRecordCountByState.committed_to_do ?? 0,
      action_record_count_canonicalized: actionCountFromRepository(borg, "countCanonicalized") ?? 0,
      action_record_count_active:
        actionCountFromRepository(borg, "countActive") ??
        activeActionCountFromStateCounts(actionRecordCountByState),
      borg_owned_active_actions: activeActorSplit.borg_owned_active_actions,
      participant_owned_active_actions: activeActorSplit.participant_owned_active_actions,
      group_owned_active_actions: activeActorSplit.group_owned_active_actions,
      prompt_salient_actions_total: promptSalienceSummary.promptSalientActionsTotal,
      borg_owned_salient_active_actions: promptSalienceSummary.borgOwnedSalientActiveActions,
      participant_owned_salient_active_actions:
        promptSalienceSummary.participantOwnedSalientActiveActions,
      dormant_actions_total: dormantActionsTotal,
      dormant_not_archive_eligible_count:
        archiveVisibilityMetrics.dormant_not_archive_eligible_count,
      dormant_archive_eligible_count: archiveVisibilityMetrics.dormant_archive_eligible_count,
      archive_oldest_inactive_turns: archiveVisibilityMetrics.archive_oldest_inactive_turns,
      archive_inactive_turn_distribution:
        archiveVisibilityMetrics.archive_inactive_turn_distribution,
      archive_archivable_count: archiveVisibilityMetrics.archive_archivable_count,
      archive_skipped_borg_owned: archiveVisibilityMetrics.archive_skipped_borg_owned,
      archive_skipped_due_date: archiveVisibilityMetrics.archive_skipped_due_date,
      archive_skipped_below_threshold: archiveVisibilityMetrics.archive_skipped_below_threshold,
      archive_skipped_other: archiveVisibilityMetrics.archive_skipped_other,
      archive_oldest_archivable_inactive_turns:
        archiveVisibilityMetrics.archive_oldest_archivable_inactive_turns,
      stale_actions_omitted_from_prompt: promptSalienceSummary.staleActionsOmittedFromPrompt,
      actions_per_turn: actionRatio(totalActions, turnCounter),
      salient_actions_per_turn: actionRatio(
        promptSalienceSummary.promptSalientActionsTotal,
        turnCounter,
      ),
      action_retirement_ratio: actionRatio(terminalActionCount, totalActions),
      borg_owned_action_count: borgOwnedActionCount(allActions),
      stale_action_count: dormantActionsTotal + (actionRecordCountByState.expired ?? 0),
      action_record_creation_source_per_turn: actionCreationSourcePerTurn,
      action_record_creation_count_this_turn: actionCreationCountTotal(actionCreationSourcePerTurn),
      actions_dormant_count: dormantActionsTotal,
      actions_archived_count: actionRecordCountByState.archived ?? 0,
      recent_completed_action_count: recentCompletedActionCount,
      commitment_count_active: borg.commitments.countActive(),
      commitment_count_active_by_kind: {
        ...zeroCounts(COMMITMENT_KINDS),
        ...borg.commitments.countActiveByKind(),
      },
      commitments_by_enforcement_class: {
        ...commitmentsByEnforcementClass,
      },
      critical_commitments_by_kind_type_domain:
        criticalCommitmentsByKindTypeDomain(activeCommitments),
      commitments_advisory_count: commitmentsByEnforcementClass.advisory,
      commitments_critical_count: commitmentsByEnforcementClass.critical,
      commitment_count_superseded: borg.commitments.countSuperseded(),
      commitment_count_revoked: borg.commitments.countRevoked(),
      commitment_count_expired: borg.commitments.countExpired(),
      commitment_count_canonicalized: borg.commitments.countCanonicalized(),
      pending_action_count: workingMemory.pending_actions.length,
      pending_action_merge_count: borg.workmem.getPendingActionMergeCount(),
      relational_slot_count_by_state: {
        ...zeroCounts<RelationalSlotState>(RELATIONAL_SLOT_STATES),
        ...borg.relationalSlots.countByState(),
      },
      review_queue_open_count_by_type: reviewQueueOpenCountByType(borg),
      open_question_resolved_count: openQuestionResolvedCount(borg),
    };
  }

  private async emitActionDuplicatePressureTrace(input: {
    borg: Borg;
    turnId: string;
    turnCounter: number;
  }): Promise<void> {
    if (
      this.tracePath === undefined ||
      input.turnCounter % ACTION_DUPLICATE_PRESSURE_CHECK_EVERY_TURNS !== 0 ||
      typeof input.borg.actions.findSimilarDescriptionPairs !== "function"
    ) {
      return;
    }

    const activeActions = input.borg.actions.list({
      states: ACTIVE_ACTION_STATES,
      limit: DEFAULT_ACTION_THREAD_SOURCE_RECORD_LIMIT,
    });
    const threshold = DEFAULT_ACTION_THREAD_SIMILARITY_THRESHOLD;
    const pairs = await input.borg.actions.findSimilarDescriptionPairs(activeActions, threshold);
    const parents = new Map<string, string>();
    const find = (id: string): string => {
      const parent = parents.get(id);

      if (parent === undefined || parent === id) {
        parents.set(id, id);
        return id;
      }

      const root = find(parent);
      parents.set(id, root);
      return root;
    };
    const union = (leftId: string, rightId: string): void => {
      const left = find(leftId);
      const right = find(rightId);

      if (left !== right) {
        parents.set(right, left);
      }
    };

    for (const action of activeActions) {
      parents.set(action.id, action.id);
    }

    for (const pair of pairs) {
      union(pair.leftId, pair.rightId);
    }

    const clusterSizes = new Map<string, number>();

    for (const action of activeActions) {
      if (!pairs.some((pair) => pair.leftId === action.id || pair.rightId === action.id)) {
        continue;
      }

      const root = find(action.id);
      clusterSizes.set(root, (clusterSizes.get(root) ?? 0) + 1);
    }

    const sizes = [...clusterSizes.values()].filter((size) => size > 1);

    appendJsonlLine(
      this.tracePath,
      `${JSON.stringify({
        ts: Date.now(),
        wallMs: performance.now(),
        turnId: input.turnId,
        event: "action_duplicate_pressure.completed",
        cluster_count: sizes.length,
        max_cluster_size: sizes.length === 0 ? 0 : Math.max(...sizes),
        total_actions_in_clusters: sizes.reduce((sum, size) => sum + size, 0),
        threshold_used: threshold,
      })}\n`,
    );
  }

  async capture(
    borg: Borg,
    turnId: string,
    turnCounter: number,
    context: MetricsCaptureContext,
  ): Promise<MetricsRow> {
    const allTraceRecords =
      this.tracePath === undefined ? [] : canonicalizeTraceRecords(readTraceEvents(this.tracePath));
    const traceRecords = allTraceRecords.filter((record) => record.turnId === turnId);
    const traceRecordsSinceLastCapture = allTraceRecords.slice(this.previousTraceRecordCount);
    const usage = usageForTurn(traceRecords);
    const mood = borg.mood.current(context.sessionId);
    const episodeResult = await borg.episodic.list({ limit: LARGE_COUNT_LIMIT });
    const semanticNodes = await borg.semantic.nodes.list({ limit: LARGE_COUNT_LIMIT });
    const semanticEdges = borg.semantic.edges.list({ includeInvalid: true });
    const semanticNodesAdded =
      this.previousSemanticNodeCount === undefined
        ? 0
        : Math.max(0, semanticNodes.length - this.previousSemanticNodeCount);
    const semanticEdgesAdded =
      this.previousSemanticEdgeCount === undefined
        ? 0
        : Math.max(0, semanticEdges.length - this.previousSemanticEdgeCount);
    const allOpenQuestions = borg.self.openQuestions.list({
      limit: LARGE_COUNT_LIMIT,
    });
    const openQuestions = allOpenQuestions.filter(
      (question) => question.status === undefined || question.status === "open",
    );
    const activeGoals = borg.self.goals.list({ status: "active" });
    const generationSuppressions = generationSuppressionCount(borg, context.sessionIds);
    const archiveAfterInactiveTurns = archiveAfterInactiveTurnsFromTrace(allTraceRecords);
    const memoryBandMetrics = await this.captureMemoryBandMetrics(
      borg,
      context.sessionId,
      turnCounter,
      archiveAfterInactiveTurns,
      traceRecordsSinceLastCapture,
    );
    const frameAnomalyMetricCounts = frameAnomalyMetrics({
      traceRecords,
      borg,
      sessionIds: context.sessionIds,
      turnId,
    });
    const actionCandidateMetricCounts = actionCandidateMetrics(traceRecords);
    const actionSessionLifecycleMetricCounts = actionSessionLifecycleMetrics(
      traceRecordsSinceLastCapture,
    );
    const goalPromotionMetricCounts = goalPromotionMetrics(traceRecords);
    const sharedStateActionLifecycleMetricCounts = sharedStateActionLifecycleMetrics(traceRecords);
    const sharedStateSemanticRevisionMetricCounts =
      sharedStateSemanticRevisionMetrics(traceRecords);
    const commitmentRegenerationMetricCounts = commitmentRegenerationMetrics({
      traceRecords,
      cumulativeTraceRecords: allTraceRecords,
    });
    const commitmentClassificationDowngradeMetricCounts =
      commitmentClassificationDowngradeMetrics(allTraceRecords);
    const semanticRevisionErrorMetricCounts = semanticRevisionErrorMetrics({
      traceRecords,
      cumulativeTraceRecords: allTraceRecords,
    });
    const semanticRevisionCumulativeMetricCounts =
      semanticRevisionCumulativeMetrics(allTraceRecords);
    const sharedStateCapPressureMetricCounts = sharedStateCapPressureMetrics(allTraceRecords);
    const sharedStateCompilerHealthMetricCounts = sharedStateCompilerHealthMetrics(
      allTraceRecords,
      turnId,
    );
    const sessionReentryContinuityMetricCounts = sessionReentryContinuityMetrics(allTraceRecords);
    const reviewResolverMetricCounts = reviewResolverMetrics(traceRecordsSinceLastCapture);
    const semanticMemoryWriteGateMetricCounts = semanticMemoryWriteGateMetrics({
      traceRecords: traceRecordsSinceLastCapture,
      cumulativeTraceRecords: allTraceRecords,
    });
    const extractorHealthMetricCounts = extractorHealthMetrics({
      traceRecords,
      cumulativeTraceRecords: allTraceRecords,
    });
    const closurePressureMetricCounts = closurePressureMetrics(allTraceRecords);
    await this.emitActionDuplicatePressureTrace({
      borg,
      turnId,
      turnCounter,
    });
    const rowTs = Date.now();
    const row: MetricsRow = {
      event: TURN_METRICS_EVENT,
      ts: rowTs,
      turn_counter: turnCounter,
      turnId,
      transport_chat_attempts: context.transportChatAttempts,
      episode_count: episodeResult.items.length,
      semantic_node_count: semanticNodes.length,
      semantic_node_count_by_status: semanticNodeStatusCounts(semanticNodes),
      semantic_edge_count: semanticEdges.length,
      semantic_nodes_added_since_last_check: semanticNodesAdded,
      semantic_edges_added_since_last_check: semanticEdgesAdded,
      semantic_nodes_rejected_ungrounded_label_count:
        semanticMemoryWriteGateMetricCounts.semantic_nodes_rejected_ungrounded_label_count,
      semantic_nodes_rejected_ungrounded_label_total:
        semanticMemoryWriteGateMetricCounts.semantic_nodes_rejected_ungrounded_label_total,
      semantic_nodes_rejected_ungrounded_label_by_label:
        semanticMemoryWriteGateMetricCounts.semantic_nodes_rejected_ungrounded_label_by_label,
      shared_state_operations_rejected_ungrounded_label_total:
        semanticMemoryWriteGateMetricCounts.shared_state_operations_rejected_ungrounded_label_total,
      shared_state_operations_rejected_ungrounded_label_by_label:
        semanticMemoryWriteGateMetricCounts.shared_state_operations_rejected_ungrounded_label_by_label,
      commitment_candidates_rejected_ungrounded_label_total:
        semanticMemoryWriteGateMetricCounts.commitment_candidates_rejected_ungrounded_label_total,
      commitment_candidates_rejected_ungrounded_label_by_label:
        semanticMemoryWriteGateMetricCounts.commitment_candidates_rejected_ungrounded_label_by_label,
      open_question_count: openQuestions.length,
      active_goal_count: flattenGoalCount(activeGoals),
      generation_suppression_count: generationSuppressions,
      mood_valence: mood.valence,
      mood_arousal: mood.arousal,
      retrieval_latency_ms: latencyBetween(
        traceRecords,
        "retrieval.started",
        "retrieval.completed",
      ),
      deliberation_latency_ms: latencyBetween(
        traceRecords,
        "llm_call.started",
        "llm_call.completed",
      ),
      borg_input_tokens: usage.inputTokens,
      borg_output_tokens: usage.outputTokens,
      open_question_resolved_count: memoryBandMetrics.open_question_resolved_count,
      open_questions_by_source: openQuestionsBySource(allOpenQuestions),
      open_questions_by_status_age: openQuestionsByStatusAge({
        questions: allOpenQuestions,
        previousRows: this.capturedRows,
        currentTurnCounter: turnCounter,
        currentTs: rowTs,
      }),
      open_questions_resolved_this_run: memoryBandMetrics.open_question_resolved_count,
      open_questions_rendered_to_finalizer_this_turn:
        openQuestionsRenderedToFinalizer(traceRecords),
      open_questions_promoted_from_review_items:
        openQuestionsPromotedFromReviewItems(allOpenQuestions),
      action_record_count_total: memoryBandMetrics.action_record_count_total,
      action_record_count_by_state: memoryBandMetrics.action_record_count_by_state,
      action_record_count_committed_to_do: memoryBandMetrics.action_record_count_committed_to_do,
      action_record_count_canonicalized: memoryBandMetrics.action_record_count_canonicalized,
      action_record_count_active: memoryBandMetrics.action_record_count_active,
      borg_owned_active_actions: memoryBandMetrics.borg_owned_active_actions,
      participant_owned_active_actions: memoryBandMetrics.participant_owned_active_actions,
      group_owned_active_actions: memoryBandMetrics.group_owned_active_actions,
      prompt_salient_actions_total: memoryBandMetrics.prompt_salient_actions_total,
      borg_owned_salient_active_actions: memoryBandMetrics.borg_owned_salient_active_actions,
      participant_owned_salient_active_actions:
        memoryBandMetrics.participant_owned_salient_active_actions,
      dormant_actions_total: memoryBandMetrics.dormant_actions_total,
      dormant_not_archive_eligible_count: memoryBandMetrics.dormant_not_archive_eligible_count,
      dormant_archive_eligible_count: memoryBandMetrics.dormant_archive_eligible_count,
      archive_oldest_inactive_turns: memoryBandMetrics.archive_oldest_inactive_turns,
      archive_inactive_turn_distribution: memoryBandMetrics.archive_inactive_turn_distribution,
      archive_archivable_count: memoryBandMetrics.archive_archivable_count,
      archive_skipped_borg_owned: memoryBandMetrics.archive_skipped_borg_owned,
      archive_skipped_due_date: memoryBandMetrics.archive_skipped_due_date,
      archive_skipped_below_threshold: memoryBandMetrics.archive_skipped_below_threshold,
      archive_skipped_other: memoryBandMetrics.archive_skipped_other,
      archive_oldest_archivable_inactive_turns:
        memoryBandMetrics.archive_oldest_archivable_inactive_turns,
      stale_actions_omitted_from_prompt: memoryBandMetrics.stale_actions_omitted_from_prompt,
      actions_per_turn: memoryBandMetrics.actions_per_turn,
      salient_actions_per_turn: memoryBandMetrics.salient_actions_per_turn,
      action_retirement_ratio: memoryBandMetrics.action_retirement_ratio,
      borg_owned_action_count: memoryBandMetrics.borg_owned_action_count,
      stale_action_count: memoryBandMetrics.stale_action_count,
      action_record_creation_source_per_turn:
        memoryBandMetrics.action_record_creation_source_per_turn,
      action_record_creation_count_this_turn:
        memoryBandMetrics.action_record_creation_count_this_turn,
      action_candidate_classifications_per_turn:
        actionCandidateMetricCounts.action_candidate_classifications_per_turn,
      action_candidate_rejected_classification:
        actionCandidateMetricCounts.action_candidate_rejected_classification,
      action_persistence_dedup_skipped_embedding:
        actionCandidateMetricCounts.action_persistence_dedup_skipped_embedding,
      action_persistence_dedup_degraded:
        actionCandidateMetricCounts.action_persistence_dedup_degraded,
      actions_closed_by_terminal_emission:
        actionCandidateMetricCounts.actions_closed_by_terminal_emission,
      actions_closed_by_borg_self_performance:
        actionCandidateMetricCounts.actions_closed_by_borg_self_performance,
      actions_expired_at_session_close:
        actionSessionLifecycleMetricCounts.actions_expired_at_session_close,
      actions_rejected_capability: actionCandidateMetricCounts.actions_rejected_capability,
      actions_canonicalized: sharedStateActionLifecycleMetricCounts.actions_canonicalized,
      actions_completed_via_canonicalization:
        sharedStateActionLifecycleMetricCounts.actions_completed_via_canonicalization,
      actions_dormant_count: memoryBandMetrics.actions_dormant_count,
      actions_archived_count: memoryBandMetrics.actions_archived_count,
      recent_completed_action_count: memoryBandMetrics.recent_completed_action_count,
      commitment_count_active: memoryBandMetrics.commitment_count_active,
      commitment_count_active_by_kind: memoryBandMetrics.commitment_count_active_by_kind,
      commitments_by_enforcement_class: memoryBandMetrics.commitments_by_enforcement_class,
      critical_commitments_by_kind_type_domain:
        memoryBandMetrics.critical_commitments_by_kind_type_domain,
      commitments_advisory_count: memoryBandMetrics.commitments_advisory_count,
      commitments_critical_count: memoryBandMetrics.commitments_critical_count,
      commitments_critical_classification_downgraded_total:
        commitmentClassificationDowngradeMetricCounts.commitments_critical_classification_downgraded_total,
      commitments_critical_classification_downgraded_by_reason:
        commitmentClassificationDowngradeMetricCounts.commitments_critical_classification_downgraded_by_reason,
      commitments_critical_classification_downgraded_by_kind_type_from_domain:
        commitmentClassificationDowngradeMetricCounts.commitments_critical_classification_downgraded_by_kind_type_from_domain,
      commitment_count_superseded: memoryBandMetrics.commitment_count_superseded,
      commitment_count_revoked: memoryBandMetrics.commitment_count_revoked,
      commitment_count_expired: memoryBandMetrics.commitment_count_expired,
      commitment_count_canonicalized: memoryBandMetrics.commitment_count_canonicalized,
      commitment_regeneration_attempted_count:
        commitmentRegenerationMetricCounts.commitment_regeneration_attempted_count,
      commitment_regeneration_succeeded_count:
        commitmentRegenerationMetricCounts.commitment_regeneration_succeeded_count,
      commitment_regeneration_failed_count:
        commitmentRegenerationMetricCounts.commitment_regeneration_failed_count,
      commitment_regeneration_attempted_total:
        commitmentRegenerationMetricCounts.commitment_regeneration_attempted_total,
      commitment_regeneration_succeeded_total:
        commitmentRegenerationMetricCounts.commitment_regeneration_succeeded_total,
      commitment_regeneration_failed_total:
        commitmentRegenerationMetricCounts.commitment_regeneration_failed_total,
      commitment_guard_advisory_violations_total:
        commitmentRegenerationMetricCounts.commitment_guard_advisory_violations_total,
      commitment_guard_advisory_violations_by_class:
        commitmentRegenerationMetricCounts.commitment_guard_advisory_violations_by_class,
      pending_action_count: memoryBandMetrics.pending_action_count,
      pending_action_merge_count: memoryBandMetrics.pending_action_merge_count,
      relational_slot_count_by_state: memoryBandMetrics.relational_slot_count_by_state,
      review_queue_open_count_by_type: memoryBandMetrics.review_queue_open_count_by_type,
      review_resolver_attempted: reviewResolverMetricCounts.review_resolver_attempted,
      review_resolver_accepted: reviewResolverMetricCounts.review_resolver_accepted,
      review_resolver_dismissed: reviewResolverMetricCounts.review_resolver_dismissed,
      review_resolver_rejected: reviewResolverMetricCounts.review_resolver_rejected,
      review_resolver_needs_manual: reviewResolverMetricCounts.review_resolver_needs_manual,
      review_queue_enqueued_this_turn: reviewResolverMetricCounts.review_queue_enqueued_this_turn,
      review_queue_resolved_this_turn: reviewResolverMetricCounts.review_queue_resolved_this_turn,
      review_queue_drain_rate: reviewResolverMetricCounts.review_queue_drain_rate,
      frame_anomaly_classifier_calls: frameAnomalyMetricCounts.frame_anomaly_classifier_calls,
      frame_anomaly_classified_normal_count:
        frameAnomalyMetricCounts.frame_anomaly_classified_normal_count,
      frame_anomaly_actual_anomaly_count:
        frameAnomalyMetricCounts.frame_anomaly_actual_anomaly_count,
      frame_anomaly_degraded_count: frameAnomalyMetricCounts.frame_anomaly_degraded_count,
      frame_anomaly_degraded_fallback_match_count:
        frameAnomalyMetricCounts.frame_anomaly_degraded_fallback_match_count,
      quarantined_user_entry_count: frameAnomalyMetricCounts.quarantined_user_entry_count,
      early_extractors_skipped_frame_anomaly_count:
        frameAnomalyMetricCounts.early_extractors_skipped_frame_anomaly_count,
      goal_promotion_salvaged_promotions:
        goalPromotionMetricCounts.goal_promotion_salvaged_promotions,
      goal_promotion_skipped_promotions:
        goalPromotionMetricCounts.goal_promotion_skipped_promotions,
      goal_promotion_initial_step_downgraded:
        goalPromotionMetricCounts.goal_promotion_initial_step_downgraded,
      goal_promotion_dedup_skipped_extractor_signal:
        goalPromotionMetricCounts.goal_promotion_dedup_skipped_extractor_signal,
      goal_promotion_dedup_skipped_embedding:
        goalPromotionMetricCounts.goal_promotion_dedup_skipped_embedding,
      goal_promotion_dedup_degraded: goalPromotionMetricCounts.goal_promotion_dedup_degraded,
      goal_promotion_classifications_per_turn:
        goalPromotionMetricCounts.goal_promotion_classifications_per_turn,
      goal_promotion_rejected_classification:
        goalPromotionMetricCounts.goal_promotion_rejected_classification,
      goal_promotion_cap_rejections: goalPromotionMetricCounts.goal_promotion_cap_rejections,
      decision_artifact_semantic_revisions_attempted:
        sharedStateSemanticRevisionMetricCounts.decision_artifact_semantic_revisions_attempted,
      decision_artifact_semantic_revisions_completed_succeeded:
        sharedStateSemanticRevisionMetricCounts.decision_artifact_semantic_revisions_completed_succeeded,
      decision_artifact_semantic_nodes_marked_superseded:
        sharedStateSemanticRevisionMetricCounts.decision_artifact_semantic_nodes_marked_superseded,
      decision_artifact_semantic_nodes_marked_contradicted:
        sharedStateSemanticRevisionMetricCounts.decision_artifact_semantic_nodes_marked_contradicted,
      decision_artifact_semantic_revision_cache_hits:
        sharedStateSemanticRevisionMetricCounts.decision_artifact_semantic_revision_cache_hits,
      decision_artifact_semantic_revision_cache_size: this.semanticRevisionVerdictCacheSize(),
      semantic_revision_error_count:
        semanticRevisionErrorMetricCounts.semantic_revision_error_count,
      semantic_revision_skipped_due_to_error:
        semanticRevisionErrorMetricCounts.semantic_revision_skipped_due_to_error,
      semantic_revision_error_total_by_reason:
        semanticRevisionErrorMetricCounts.semantic_revision_error_total_by_reason,
      semantic_revision_calls_total:
        semanticRevisionCumulativeMetricCounts.semantic_revision_calls_total,
      semantic_revision_candidates_reviewed_total:
        semanticRevisionCumulativeMetricCounts.semantic_revision_candidates_reviewed_total,
      semantic_revision_superseded_total:
        semanticRevisionCumulativeMetricCounts.semantic_revision_superseded_total,
      semantic_revision_contradicted_total:
        semanticRevisionCumulativeMetricCounts.semantic_revision_contradicted_total,
      semantic_revision_degraded_total:
        semanticRevisionCumulativeMetricCounts.semantic_revision_degraded_total,
      semantic_revision_skipped_over_cap_total:
        semanticRevisionCumulativeMetricCounts.semantic_revision_skipped_over_cap_total,
      overseer_due_on_suppressed_turn: context.overseerDueOnSuppressedTurn ?? false,
      closure_loop_completed_count: extractorHealthMetricCounts.closure_loop_completed_count,
      closure_loop_degraded_count: extractorHealthMetricCounts.closure_loop_degraded_count,
      closure_pressure_mixed_observed_total:
        closurePressureMetricCounts.closure_pressure_mixed_observed_total,
      closure_pressure_closure_only_suppressed_total:
        closurePressureMetricCounts.closure_pressure_closure_only_suppressed_total,
      closure_pressure_mixed_passed_no_active_preference_total:
        closurePressureMetricCounts.closure_pressure_mixed_passed_no_active_preference_total,
      closure_pressure_mixed_by_span_kind:
        closurePressureMetricCounts.closure_pressure_mixed_by_span_kind,
      corrective_preference_completed_count:
        extractorHealthMetricCounts.corrective_preference_completed_count,
      corrective_preference_degraded_count:
        extractorHealthMetricCounts.corrective_preference_degraded_count,
      extractor_max_tokens_stop_count: extractorHealthMetricCounts.extractor_max_tokens_stop_count,
      extractor_max_tokens_total_by_label:
        extractorHealthMetricCounts.extractor_max_tokens_total_by_label,
      extractor_degraded_total_by_label:
        extractorHealthMetricCounts.extractor_degraded_total_by_label,
      shared_state_compiler_max_tokens_total:
        sharedStateCompilerHealthMetricCounts.shared_state_compiler_max_tokens_total,
      shared_state_compiler_degraded_total:
        sharedStateCompilerHealthMetricCounts.shared_state_compiler_degraded_total,
      shared_state_compiler_repair_attempted_total:
        sharedStateCompilerHealthMetricCounts.shared_state_compiler_repair_attempted_total,
      shared_state_compiler_repair_succeeded_total:
        sharedStateCompilerHealthMetricCounts.shared_state_compiler_repair_succeeded_total,
      shared_state_compiler_repair_failed_total:
        sharedStateCompilerHealthMetricCounts.shared_state_compiler_repair_failed_total,
      shared_state_compiler_repair_failed_by_rejection_reason:
        sharedStateCompilerHealthMetricCounts.shared_state_compiler_repair_failed_by_rejection_reason,
      capability_overclaim_count: 0,
      capability_ambiguity_count: 0,
      capability_boundary_refusal_count: 0,
      shared_state_at_cap_turns: sharedStateCapPressureMetricCounts.shared_state_at_cap_turns,
      shared_state_compile_evaluated_turns:
        sharedStateCapPressureMetricCounts.shared_state_compile_evaluated_turns,
      shared_state_omitted_recent_entries:
        sharedStateCapPressureMetricCounts.shared_state_omitted_recent_entries,
      shared_state_live_entry_starvation:
        sharedStateCapPressureMetricCounts.shared_state_live_entry_starvation,
      shared_state_newest_entries_reserved:
        sharedStateCapPressureMetricCounts.shared_state_newest_entries_reserved,
      shared_state_live_starvation_with_reserved:
        sharedStateCapPressureMetricCounts.shared_state_live_starvation_with_reserved,
      shared_state_live_starvation_ever:
        sharedStateCapPressureMetricCounts.shared_state_live_starvation_ever,
      shared_state_live_starvation_final:
        sharedStateCapPressureMetricCounts.shared_state_live_starvation_final,
      shared_state_compiler_operations_total_by_kind:
        sharedStateCompilerHealthMetricCounts.shared_state_compiler_operations_total_by_kind,
      shared_state_add_to_update_ratio:
        sharedStateCompilerHealthMetricCounts.shared_state_add_to_update_ratio,
      shared_state_entries_by_key:
        sharedStateCompilerHealthMetricCounts.shared_state_entries_by_key,
      shared_state_add_to_update_ratio_by_key:
        sharedStateCompilerHealthMetricCounts.shared_state_add_to_update_ratio_by_key,
      shared_state_top_keys_by_entry_count:
        sharedStateCompilerHealthMetricCounts.shared_state_top_keys_by_entry_count,
      shared_state_add_rejected_cap_exceeded_total:
        sharedStateCompilerHealthMetricCounts.shared_state_add_rejected_cap_exceeded_total,
      shared_state_new_keys_per_compile:
        sharedStateCompilerHealthMetricCounts.shared_state_new_keys_per_compile,
      shared_state_new_keys_per_turn:
        sharedStateCompilerHealthMetricCounts.shared_state_new_keys_per_turn,
      shared_state_keys_with_single_entry_only:
        sharedStateCompilerHealthMetricCounts.shared_state_keys_with_single_entry_only,
      shared_state_similar_key_cluster_count:
        sharedStateCompilerHealthMetricCounts.shared_state_similar_key_cluster_count,
      shared_state_add_rejected_near_duplicate_state_key_total:
        sharedStateCompilerHealthMetricCounts.shared_state_add_rejected_near_duplicate_state_key_total,
      shared_state_add_rejected_missing_new_key_reason_total:
        sharedStateCompilerHealthMetricCounts.shared_state_add_rejected_missing_new_key_reason_total,
      session_reentry_card_rendered_total:
        sessionReentryContinuityMetricCounts.session_reentry_card_rendered_total,
      session_reentry_card_rendered_by_audience:
        sessionReentryContinuityMetricCounts.session_reentry_card_rendered_by_audience,
      session_reentry_first_turn_with_existing_state_total:
        sessionReentryContinuityMetricCounts.session_reentry_first_turn_with_existing_state_total,
      session_reentry_first_turn_blank_audience_total:
        sessionReentryContinuityMetricCounts.session_reentry_first_turn_blank_audience_total,
      simulator_persona_failures: context.simulatorPersonaFailures ?? 0,
      borg_hard_aborted_turns: context.borgHardAbortedTurns ?? context.borgAbortedTurns ?? 0,
      borg_intentional_suppressions: context.borgIntentionalSuppressions ?? 0,
      borg_intentional_suppressions_by_reason: sortedContextCounts(
        context.borgIntentionalSuppressionsByReason,
      ),
      borg_aborted_turns: context.borgHardAbortedTurns ?? context.borgAbortedTurns ?? 0,
    };

    this.previousSemanticNodeCount = semanticNodes.length;
    this.previousSemanticEdgeCount = semanticEdges.length;
    this.previousTraceRecordCount = allTraceRecords.length;
    this.previousActionCreationCountsBySource = actionCreationCountsFromRepository(borg);
    appendJsonlLine(this.filepath, `${JSON.stringify(row)}\n`);
    this.recordHealthWarnings(row);
    return row;
  }

  async captureAborted(
    borg: Borg,
    turnCounter: number,
    context: MetricsCaptureContext & {
      failureReason: string;
      turnId?: string;
      event?: typeof ABORTED_TURN_EVENT | typeof ABORTED_ATTEMPT_EVENT;
    },
  ): Promise<MetricsRow> {
    const event = context.event ?? ABORTED_TURN_EVENT;
    const allTraceRecords =
      this.tracePath === undefined ? [] : canonicalizeTraceRecords(readTraceEvents(this.tracePath));
    const mood = borg.mood.current(context.sessionId);
    const episodeResult = await borg.episodic.list({ limit: LARGE_COUNT_LIMIT });
    const semanticNodes = await borg.semantic.nodes.list({ limit: LARGE_COUNT_LIMIT });
    const semanticEdges = borg.semantic.edges.list({ includeInvalid: true });
    const allOpenQuestions = borg.self.openQuestions.list({
      limit: LARGE_COUNT_LIMIT,
    });
    const openQuestions = allOpenQuestions.filter(
      (question) => question.status === undefined || question.status === "open",
    );
    const activeGoals = borg.self.goals.list({ status: "active" });
    const generationSuppressions = generationSuppressionCount(borg, context.sessionIds);
    const archiveAfterInactiveTurns = archiveAfterInactiveTurnsFromTrace(allTraceRecords);
    const memoryBandMetrics = await this.captureMemoryBandMetrics(
      borg,
      context.sessionId,
      turnCounter,
      archiveAfterInactiveTurns,
      [],
    );
    const turnId = context.turnId ?? `${event}_${turnCounter}`;
    const traceRecords = allTraceRecords.filter((record) => record.turnId === turnId);
    const frameAnomalyMetricCounts = frameAnomalyMetrics({
      traceRecords: [],
      borg,
      sessionIds: context.sessionIds,
      turnId,
    });
    const actionCandidateMetricCounts = actionCandidateMetrics([]);
    const actionSessionLifecycleMetricCounts = actionSessionLifecycleMetrics([]);
    const goalPromotionMetricCounts = goalPromotionMetrics([]);
    const sharedStateActionLifecycleMetricCounts = sharedStateActionLifecycleMetrics([]);
    const sharedStateSemanticRevisionMetricCounts = sharedStateSemanticRevisionMetrics([]);
    const commitmentRegenerationMetricCounts = commitmentRegenerationMetrics({
      traceRecords,
      cumulativeTraceRecords: allTraceRecords,
    });
    const commitmentClassificationDowngradeMetricCounts =
      commitmentClassificationDowngradeMetrics(allTraceRecords);
    const semanticRevisionErrorMetricCounts = semanticRevisionErrorMetrics({
      traceRecords,
      cumulativeTraceRecords: allTraceRecords,
    });
    const semanticRevisionCumulativeMetricCounts =
      semanticRevisionCumulativeMetrics(allTraceRecords);
    const sharedStateCapPressureMetricCounts = sharedStateCapPressureMetrics(allTraceRecords);
    const sharedStateCompilerHealthMetricCounts = sharedStateCompilerHealthMetrics(
      allTraceRecords,
      turnId,
    );
    const sessionReentryContinuityMetricCounts = sessionReentryContinuityMetrics(allTraceRecords);
    const reviewResolverMetricCounts = reviewResolverMetrics([]);
    const extractorHealthMetricCounts = extractorHealthMetrics({
      traceRecords: [],
      cumulativeTraceRecords: allTraceRecords,
    });
    const semanticMemoryWriteGateMetricCounts = semanticMemoryWriteGateMetrics({
      traceRecords: [],
      cumulativeTraceRecords: allTraceRecords,
    });
    const closurePressureMetricCounts = closurePressureMetrics(allTraceRecords);
    const rowTs = Date.now();
    const row: MetricsRow = {
      event,
      ts: rowTs,
      turn_counter: turnCounter,
      turnId,
      transport_chat_attempts: context.transportChatAttempts,
      failure_reason: context.failureReason,
      episode_count: episodeResult.items.length,
      semantic_node_count: semanticNodes.length,
      semantic_node_count_by_status: semanticNodeStatusCounts(semanticNodes),
      semantic_edge_count: semanticEdges.length,
      semantic_nodes_added_since_last_check: 0,
      semantic_edges_added_since_last_check: 0,
      semantic_nodes_rejected_ungrounded_label_count: 0,
      semantic_nodes_rejected_ungrounded_label_total:
        semanticMemoryWriteGateMetricCounts.semantic_nodes_rejected_ungrounded_label_total,
      semantic_nodes_rejected_ungrounded_label_by_label:
        semanticMemoryWriteGateMetricCounts.semantic_nodes_rejected_ungrounded_label_by_label,
      shared_state_operations_rejected_ungrounded_label_total:
        semanticMemoryWriteGateMetricCounts.shared_state_operations_rejected_ungrounded_label_total,
      shared_state_operations_rejected_ungrounded_label_by_label:
        semanticMemoryWriteGateMetricCounts.shared_state_operations_rejected_ungrounded_label_by_label,
      commitment_candidates_rejected_ungrounded_label_total:
        semanticMemoryWriteGateMetricCounts.commitment_candidates_rejected_ungrounded_label_total,
      commitment_candidates_rejected_ungrounded_label_by_label:
        semanticMemoryWriteGateMetricCounts.commitment_candidates_rejected_ungrounded_label_by_label,
      open_question_count: openQuestions.length,
      active_goal_count: flattenGoalCount(activeGoals),
      generation_suppression_count: generationSuppressions,
      mood_valence: mood.valence,
      mood_arousal: mood.arousal,
      retrieval_latency_ms: null,
      deliberation_latency_ms: null,
      borg_input_tokens: 0,
      borg_output_tokens: 0,
      open_question_resolved_count: memoryBandMetrics.open_question_resolved_count,
      open_questions_by_source: openQuestionsBySource(allOpenQuestions),
      open_questions_by_status_age: openQuestionsByStatusAge({
        questions: allOpenQuestions,
        previousRows: this.capturedRows,
        currentTurnCounter: turnCounter,
        currentTs: rowTs,
      }),
      open_questions_resolved_this_run: memoryBandMetrics.open_question_resolved_count,
      open_questions_rendered_to_finalizer_this_turn:
        openQuestionsRenderedToFinalizer(traceRecords),
      open_questions_promoted_from_review_items:
        openQuestionsPromotedFromReviewItems(allOpenQuestions),
      action_record_count_total: memoryBandMetrics.action_record_count_total,
      action_record_count_by_state: memoryBandMetrics.action_record_count_by_state,
      action_record_count_committed_to_do: memoryBandMetrics.action_record_count_committed_to_do,
      action_record_count_canonicalized: memoryBandMetrics.action_record_count_canonicalized,
      action_record_count_active: memoryBandMetrics.action_record_count_active,
      borg_owned_active_actions: memoryBandMetrics.borg_owned_active_actions,
      participant_owned_active_actions: memoryBandMetrics.participant_owned_active_actions,
      group_owned_active_actions: memoryBandMetrics.group_owned_active_actions,
      prompt_salient_actions_total: memoryBandMetrics.prompt_salient_actions_total,
      borg_owned_salient_active_actions: memoryBandMetrics.borg_owned_salient_active_actions,
      participant_owned_salient_active_actions:
        memoryBandMetrics.participant_owned_salient_active_actions,
      dormant_actions_total: memoryBandMetrics.dormant_actions_total,
      dormant_not_archive_eligible_count: memoryBandMetrics.dormant_not_archive_eligible_count,
      dormant_archive_eligible_count: memoryBandMetrics.dormant_archive_eligible_count,
      archive_oldest_inactive_turns: memoryBandMetrics.archive_oldest_inactive_turns,
      archive_inactive_turn_distribution: memoryBandMetrics.archive_inactive_turn_distribution,
      archive_archivable_count: memoryBandMetrics.archive_archivable_count,
      archive_skipped_borg_owned: memoryBandMetrics.archive_skipped_borg_owned,
      archive_skipped_due_date: memoryBandMetrics.archive_skipped_due_date,
      archive_skipped_below_threshold: memoryBandMetrics.archive_skipped_below_threshold,
      archive_skipped_other: memoryBandMetrics.archive_skipped_other,
      archive_oldest_archivable_inactive_turns:
        memoryBandMetrics.archive_oldest_archivable_inactive_turns,
      stale_actions_omitted_from_prompt: memoryBandMetrics.stale_actions_omitted_from_prompt,
      actions_per_turn: memoryBandMetrics.actions_per_turn,
      salient_actions_per_turn: memoryBandMetrics.salient_actions_per_turn,
      action_retirement_ratio: memoryBandMetrics.action_retirement_ratio,
      borg_owned_action_count: memoryBandMetrics.borg_owned_action_count,
      stale_action_count: memoryBandMetrics.stale_action_count,
      action_record_creation_source_per_turn:
        memoryBandMetrics.action_record_creation_source_per_turn,
      action_record_creation_count_this_turn:
        memoryBandMetrics.action_record_creation_count_this_turn,
      action_candidate_classifications_per_turn:
        actionCandidateMetricCounts.action_candidate_classifications_per_turn,
      action_candidate_rejected_classification:
        actionCandidateMetricCounts.action_candidate_rejected_classification,
      action_persistence_dedup_skipped_embedding:
        actionCandidateMetricCounts.action_persistence_dedup_skipped_embedding,
      action_persistence_dedup_degraded:
        actionCandidateMetricCounts.action_persistence_dedup_degraded,
      actions_closed_by_terminal_emission:
        actionCandidateMetricCounts.actions_closed_by_terminal_emission,
      actions_closed_by_borg_self_performance:
        actionCandidateMetricCounts.actions_closed_by_borg_self_performance,
      actions_expired_at_session_close:
        actionSessionLifecycleMetricCounts.actions_expired_at_session_close,
      actions_rejected_capability: actionCandidateMetricCounts.actions_rejected_capability,
      actions_canonicalized: sharedStateActionLifecycleMetricCounts.actions_canonicalized,
      actions_completed_via_canonicalization:
        sharedStateActionLifecycleMetricCounts.actions_completed_via_canonicalization,
      actions_dormant_count: memoryBandMetrics.actions_dormant_count,
      actions_archived_count: memoryBandMetrics.actions_archived_count,
      recent_completed_action_count: memoryBandMetrics.recent_completed_action_count,
      commitment_count_active: memoryBandMetrics.commitment_count_active,
      commitment_count_active_by_kind: memoryBandMetrics.commitment_count_active_by_kind,
      commitments_by_enforcement_class: memoryBandMetrics.commitments_by_enforcement_class,
      critical_commitments_by_kind_type_domain:
        memoryBandMetrics.critical_commitments_by_kind_type_domain,
      commitments_advisory_count: memoryBandMetrics.commitments_advisory_count,
      commitments_critical_count: memoryBandMetrics.commitments_critical_count,
      commitments_critical_classification_downgraded_total:
        commitmentClassificationDowngradeMetricCounts.commitments_critical_classification_downgraded_total,
      commitments_critical_classification_downgraded_by_reason:
        commitmentClassificationDowngradeMetricCounts.commitments_critical_classification_downgraded_by_reason,
      commitments_critical_classification_downgraded_by_kind_type_from_domain:
        commitmentClassificationDowngradeMetricCounts.commitments_critical_classification_downgraded_by_kind_type_from_domain,
      commitment_count_superseded: memoryBandMetrics.commitment_count_superseded,
      commitment_count_revoked: memoryBandMetrics.commitment_count_revoked,
      commitment_count_expired: memoryBandMetrics.commitment_count_expired,
      commitment_count_canonicalized: memoryBandMetrics.commitment_count_canonicalized,
      commitment_regeneration_attempted_count:
        commitmentRegenerationMetricCounts.commitment_regeneration_attempted_count,
      commitment_regeneration_succeeded_count:
        commitmentRegenerationMetricCounts.commitment_regeneration_succeeded_count,
      commitment_regeneration_failed_count:
        commitmentRegenerationMetricCounts.commitment_regeneration_failed_count,
      commitment_regeneration_attempted_total:
        commitmentRegenerationMetricCounts.commitment_regeneration_attempted_total,
      commitment_regeneration_succeeded_total:
        commitmentRegenerationMetricCounts.commitment_regeneration_succeeded_total,
      commitment_regeneration_failed_total:
        commitmentRegenerationMetricCounts.commitment_regeneration_failed_total,
      commitment_guard_advisory_violations_total:
        commitmentRegenerationMetricCounts.commitment_guard_advisory_violations_total,
      commitment_guard_advisory_violations_by_class:
        commitmentRegenerationMetricCounts.commitment_guard_advisory_violations_by_class,
      pending_action_count: memoryBandMetrics.pending_action_count,
      pending_action_merge_count: memoryBandMetrics.pending_action_merge_count,
      relational_slot_count_by_state: memoryBandMetrics.relational_slot_count_by_state,
      review_queue_open_count_by_type: memoryBandMetrics.review_queue_open_count_by_type,
      review_resolver_attempted: reviewResolverMetricCounts.review_resolver_attempted,
      review_resolver_accepted: reviewResolverMetricCounts.review_resolver_accepted,
      review_resolver_dismissed: reviewResolverMetricCounts.review_resolver_dismissed,
      review_resolver_rejected: reviewResolverMetricCounts.review_resolver_rejected,
      review_resolver_needs_manual: reviewResolverMetricCounts.review_resolver_needs_manual,
      review_queue_enqueued_this_turn: reviewResolverMetricCounts.review_queue_enqueued_this_turn,
      review_queue_resolved_this_turn: reviewResolverMetricCounts.review_queue_resolved_this_turn,
      review_queue_drain_rate: reviewResolverMetricCounts.review_queue_drain_rate,
      frame_anomaly_classifier_calls: frameAnomalyMetricCounts.frame_anomaly_classifier_calls,
      frame_anomaly_classified_normal_count:
        frameAnomalyMetricCounts.frame_anomaly_classified_normal_count,
      frame_anomaly_actual_anomaly_count:
        frameAnomalyMetricCounts.frame_anomaly_actual_anomaly_count,
      frame_anomaly_degraded_count: frameAnomalyMetricCounts.frame_anomaly_degraded_count,
      frame_anomaly_degraded_fallback_match_count:
        frameAnomalyMetricCounts.frame_anomaly_degraded_fallback_match_count,
      quarantined_user_entry_count: frameAnomalyMetricCounts.quarantined_user_entry_count,
      early_extractors_skipped_frame_anomaly_count:
        frameAnomalyMetricCounts.early_extractors_skipped_frame_anomaly_count,
      goal_promotion_salvaged_promotions:
        goalPromotionMetricCounts.goal_promotion_salvaged_promotions,
      goal_promotion_skipped_promotions:
        goalPromotionMetricCounts.goal_promotion_skipped_promotions,
      goal_promotion_initial_step_downgraded:
        goalPromotionMetricCounts.goal_promotion_initial_step_downgraded,
      goal_promotion_dedup_skipped_extractor_signal:
        goalPromotionMetricCounts.goal_promotion_dedup_skipped_extractor_signal,
      goal_promotion_dedup_skipped_embedding:
        goalPromotionMetricCounts.goal_promotion_dedup_skipped_embedding,
      goal_promotion_dedup_degraded: goalPromotionMetricCounts.goal_promotion_dedup_degraded,
      goal_promotion_classifications_per_turn:
        goalPromotionMetricCounts.goal_promotion_classifications_per_turn,
      goal_promotion_rejected_classification:
        goalPromotionMetricCounts.goal_promotion_rejected_classification,
      goal_promotion_cap_rejections: goalPromotionMetricCounts.goal_promotion_cap_rejections,
      decision_artifact_semantic_revisions_attempted:
        sharedStateSemanticRevisionMetricCounts.decision_artifact_semantic_revisions_attempted,
      decision_artifact_semantic_revisions_completed_succeeded:
        sharedStateSemanticRevisionMetricCounts.decision_artifact_semantic_revisions_completed_succeeded,
      decision_artifact_semantic_nodes_marked_superseded:
        sharedStateSemanticRevisionMetricCounts.decision_artifact_semantic_nodes_marked_superseded,
      decision_artifact_semantic_nodes_marked_contradicted:
        sharedStateSemanticRevisionMetricCounts.decision_artifact_semantic_nodes_marked_contradicted,
      decision_artifact_semantic_revision_cache_hits:
        sharedStateSemanticRevisionMetricCounts.decision_artifact_semantic_revision_cache_hits,
      decision_artifact_semantic_revision_cache_size: this.semanticRevisionVerdictCacheSize(),
      semantic_revision_error_count:
        semanticRevisionErrorMetricCounts.semantic_revision_error_count,
      semantic_revision_skipped_due_to_error:
        semanticRevisionErrorMetricCounts.semantic_revision_skipped_due_to_error,
      semantic_revision_error_total_by_reason:
        semanticRevisionErrorMetricCounts.semantic_revision_error_total_by_reason,
      semantic_revision_calls_total:
        semanticRevisionCumulativeMetricCounts.semantic_revision_calls_total,
      semantic_revision_candidates_reviewed_total:
        semanticRevisionCumulativeMetricCounts.semantic_revision_candidates_reviewed_total,
      semantic_revision_superseded_total:
        semanticRevisionCumulativeMetricCounts.semantic_revision_superseded_total,
      semantic_revision_contradicted_total:
        semanticRevisionCumulativeMetricCounts.semantic_revision_contradicted_total,
      semantic_revision_degraded_total:
        semanticRevisionCumulativeMetricCounts.semantic_revision_degraded_total,
      semantic_revision_skipped_over_cap_total:
        semanticRevisionCumulativeMetricCounts.semantic_revision_skipped_over_cap_total,
      overseer_due_on_suppressed_turn: context.overseerDueOnSuppressedTurn ?? false,
      closure_loop_completed_count: extractorHealthMetricCounts.closure_loop_completed_count,
      closure_loop_degraded_count: extractorHealthMetricCounts.closure_loop_degraded_count,
      closure_pressure_mixed_observed_total:
        closurePressureMetricCounts.closure_pressure_mixed_observed_total,
      closure_pressure_closure_only_suppressed_total:
        closurePressureMetricCounts.closure_pressure_closure_only_suppressed_total,
      closure_pressure_mixed_passed_no_active_preference_total:
        closurePressureMetricCounts.closure_pressure_mixed_passed_no_active_preference_total,
      closure_pressure_mixed_by_span_kind:
        closurePressureMetricCounts.closure_pressure_mixed_by_span_kind,
      corrective_preference_completed_count:
        extractorHealthMetricCounts.corrective_preference_completed_count,
      corrective_preference_degraded_count:
        extractorHealthMetricCounts.corrective_preference_degraded_count,
      extractor_max_tokens_stop_count: extractorHealthMetricCounts.extractor_max_tokens_stop_count,
      extractor_max_tokens_total_by_label:
        extractorHealthMetricCounts.extractor_max_tokens_total_by_label,
      extractor_degraded_total_by_label:
        extractorHealthMetricCounts.extractor_degraded_total_by_label,
      shared_state_compiler_max_tokens_total:
        sharedStateCompilerHealthMetricCounts.shared_state_compiler_max_tokens_total,
      shared_state_compiler_degraded_total:
        sharedStateCompilerHealthMetricCounts.shared_state_compiler_degraded_total,
      shared_state_compiler_repair_attempted_total:
        sharedStateCompilerHealthMetricCounts.shared_state_compiler_repair_attempted_total,
      shared_state_compiler_repair_succeeded_total:
        sharedStateCompilerHealthMetricCounts.shared_state_compiler_repair_succeeded_total,
      shared_state_compiler_repair_failed_total:
        sharedStateCompilerHealthMetricCounts.shared_state_compiler_repair_failed_total,
      shared_state_compiler_repair_failed_by_rejection_reason:
        sharedStateCompilerHealthMetricCounts.shared_state_compiler_repair_failed_by_rejection_reason,
      capability_overclaim_count: 0,
      capability_ambiguity_count: 0,
      capability_boundary_refusal_count: 0,
      shared_state_at_cap_turns: sharedStateCapPressureMetricCounts.shared_state_at_cap_turns,
      shared_state_compile_evaluated_turns:
        sharedStateCapPressureMetricCounts.shared_state_compile_evaluated_turns,
      shared_state_omitted_recent_entries:
        sharedStateCapPressureMetricCounts.shared_state_omitted_recent_entries,
      shared_state_live_entry_starvation:
        sharedStateCapPressureMetricCounts.shared_state_live_entry_starvation,
      shared_state_newest_entries_reserved:
        sharedStateCapPressureMetricCounts.shared_state_newest_entries_reserved,
      shared_state_live_starvation_with_reserved:
        sharedStateCapPressureMetricCounts.shared_state_live_starvation_with_reserved,
      shared_state_live_starvation_ever:
        sharedStateCapPressureMetricCounts.shared_state_live_starvation_ever,
      shared_state_live_starvation_final:
        sharedStateCapPressureMetricCounts.shared_state_live_starvation_final,
      shared_state_compiler_operations_total_by_kind:
        sharedStateCompilerHealthMetricCounts.shared_state_compiler_operations_total_by_kind,
      shared_state_add_to_update_ratio:
        sharedStateCompilerHealthMetricCounts.shared_state_add_to_update_ratio,
      shared_state_entries_by_key:
        sharedStateCompilerHealthMetricCounts.shared_state_entries_by_key,
      shared_state_add_to_update_ratio_by_key:
        sharedStateCompilerHealthMetricCounts.shared_state_add_to_update_ratio_by_key,
      shared_state_top_keys_by_entry_count:
        sharedStateCompilerHealthMetricCounts.shared_state_top_keys_by_entry_count,
      shared_state_add_rejected_cap_exceeded_total:
        sharedStateCompilerHealthMetricCounts.shared_state_add_rejected_cap_exceeded_total,
      shared_state_new_keys_per_compile:
        sharedStateCompilerHealthMetricCounts.shared_state_new_keys_per_compile,
      shared_state_new_keys_per_turn:
        sharedStateCompilerHealthMetricCounts.shared_state_new_keys_per_turn,
      shared_state_keys_with_single_entry_only:
        sharedStateCompilerHealthMetricCounts.shared_state_keys_with_single_entry_only,
      shared_state_similar_key_cluster_count:
        sharedStateCompilerHealthMetricCounts.shared_state_similar_key_cluster_count,
      shared_state_add_rejected_near_duplicate_state_key_total:
        sharedStateCompilerHealthMetricCounts.shared_state_add_rejected_near_duplicate_state_key_total,
      shared_state_add_rejected_missing_new_key_reason_total:
        sharedStateCompilerHealthMetricCounts.shared_state_add_rejected_missing_new_key_reason_total,
      session_reentry_card_rendered_total:
        sessionReentryContinuityMetricCounts.session_reentry_card_rendered_total,
      session_reentry_card_rendered_by_audience:
        sessionReentryContinuityMetricCounts.session_reentry_card_rendered_by_audience,
      session_reentry_first_turn_with_existing_state_total:
        sessionReentryContinuityMetricCounts.session_reentry_first_turn_with_existing_state_total,
      session_reentry_first_turn_blank_audience_total:
        sessionReentryContinuityMetricCounts.session_reentry_first_turn_blank_audience_total,
      simulator_persona_failures: context.simulatorPersonaFailures ?? 0,
      borg_hard_aborted_turns: context.borgHardAbortedTurns ?? context.borgAbortedTurns ?? 0,
      borg_intentional_suppressions: context.borgIntentionalSuppressions ?? 0,
      borg_intentional_suppressions_by_reason: sortedContextCounts(
        context.borgIntentionalSuppressionsByReason,
      ),
      borg_aborted_turns: context.borgHardAbortedTurns ?? context.borgAbortedTurns ?? 0,
    };

    this.previousActionCreationCountsBySource = actionCreationCountsFromRepository(borg);
    appendJsonlLine(this.filepath, `${JSON.stringify(row)}\n`);
    this.recordHealthWarnings(row);
    return row;
  }

  close(): void {
    // Metrics rows are fsynced on each append.
  }
}
