import { performance } from "node:perf_hooks";

import {
  QUARANTINED_USER_ENTRY_EVENT,
  ACTION_CANDIDATE_CLASSIFICATIONS,
  ACTION_STATES,
  GOAL_PROMOTION_CLASSIFICATIONS,
  RELATIONAL_SLOT_STATES,
  REVIEW_KINDS,
  SEMANTIC_NODE_STATUSES,
  type ActionCandidateClassification,
  type ActionRecordCreationSource,
  type ActionState,
  type Borg,
  type GoalPromotionClassification,
  type RelationalSlotState,
  type ReviewKind,
  type SemanticNode,
  type SemanticNodeStatus,
  type SessionId,
} from "../src/index.js";
import {
  DEFAULT_ACTION_THREAD_SIMILARITY_THRESHOLD,
  DEFAULT_ACTION_THREAD_SOURCE_RECORD_LIMIT,
} from "../src/cognition/evidence-ledger/action-threads.js";
import { decisionArtifactSemanticRevisionVerdictCacheSize } from "../src/cognition/decision-artifact/reconciliation.js";
import { filterActiveStreamEntries } from "../src/stream/index.js";
import type { ActionId } from "../src/util/ids.js";
import { readTraceEvents } from "../assessor/trace-reader.js";
import type { TraceRecord } from "../assessor/types.js";

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
const ACTION_DUPLICATE_PRESSURE_CHECK_EVERY_TURNS = 10;
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
  | "action_record_creation_source_per_turn"
  | "action_record_creation_count_this_turn"
  | "recent_completed_action_count"
  | "commitment_count_active"
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

type DecisionArtifactSemanticRevisionMetricCounts = Pick<
  MetricsRow,
  | "decision_artifact_semantic_revisions_attempted"
  | "decision_artifact_semantic_revisions_completed_succeeded"
  | "decision_artifact_semantic_nodes_marked_superseded"
  | "decision_artifact_semantic_nodes_marked_contradicted"
  | "decision_artifact_semantic_revision_cache_hits"
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
    if (record.event !== "llm_call_response") {
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

function traceNumber(record: TraceRecord, key: string): number {
  const value = record[key];

  return typeof value === "number" && Number.isFinite(value) ? value : 0;
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
    (record) => record.event === "frame_anomaly_classified",
  );
  const actualAnomalyCount = frameClassified.filter(
    (record) => traceStatus(record) === "ok" && traceKind(record) !== "normal",
  ).length;
  const fallbackMatchCount = input.traceRecords.filter(
    (record) => record.event === "frame_anomaly_degraded_fallback_match",
  ).length;

  return {
    frame_anomaly_classifier_calls: input.traceRecords.filter(
      (record) =>
        record.event === "llm_call_started" && traceLabel(record) === "frame_anomaly_classifier",
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
    (record) => record.event === "action_state_extractor_completed",
  );
  const classificationRejected = traceRecords.filter(
    (record) => record.event === "action_candidate_classification_rejected",
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
      (record) => record.event === "action_persistence_dedup_skipped_embedding",
    ).length,
    action_persistence_dedup_degraded: traceRecords.filter(
      (record) => record.event === "action_persistence_dedup_degraded",
    ).length,
  };
}

function goalPromotionMetrics(traceRecords: readonly TraceRecord[]): GoalPromotionMetricCounts {
  const completed = traceRecords.filter(
    (record) => record.event === "goal_promotion_extractor_completed",
  );
  const skippedAsDuplicate = traceRecords.filter(
    (record) => record.event === "goal_promotion_skipped_as_duplicate",
  );
  const classificationRejected = traceRecords.filter(
    (record) => record.event === "goal_promotion_classification_rejected",
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
      (record) => record.event === "goal_promotion_initial_step_downgraded",
    ).length,
    goal_promotion_dedup_skipped_extractor_signal: skippedAsDuplicate.filter(
      (record) => traceReason(record) === "extractor_signal",
    ).length,
    goal_promotion_dedup_skipped_embedding: skippedAsDuplicate.filter(
      (record) => traceReason(record) === "embedding",
    ).length,
    goal_promotion_dedup_degraded: traceRecords.filter(
      (record) => record.event === "goal_promotion_dedup_degraded",
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

function decisionArtifactSemanticRevisionMetrics(
  traceRecords: readonly TraceRecord[],
): DecisionArtifactSemanticRevisionMetricCounts {
  const completed = traceRecords.filter(
    (record) => record.event === "decision_artifact_semantic_revision_completed",
  );
  const degraded = traceRecords.filter(
    (record) => record.event === "decision_artifact_semantic_revision_degraded",
  );
  const cacheHits = traceRecords.filter(
    (record) => record.event === "decision_artifact_semantic_revision_cache_hit",
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

function reviewResolverMetrics(traceRecords: readonly TraceRecord[]): ReviewResolverMetricCounts {
  const completed = traceRecords.filter(
    (record) => record.event === "review_resolver_pass_completed",
  );
  const reviewQueueDecisions = traceRecords.filter(
    (record) => record.event === "review_queue_decision",
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

function zeroCounts<K extends string>(keys: readonly K[]): Record<K, number> {
  return Object.fromEntries(keys.map((key) => [key, 0])) as Record<K, number>;
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
      options.semanticRevisionVerdictCacheSize ?? decisionArtifactSemanticRevisionVerdictCacheSize;
  }

  listHealthWarnings(): SimulatorHealthWarning[] {
    return this.healthWarnings.map((warning) => ({ ...warning }));
  }

  listRows(): MetricsRow[] {
    return this.capturedRows.map((row) => ({ ...row }));
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
          event: "simulator_health_warning",
          artifact: "simulator",
          warning_kind: warning.kind,
          turn_counter: warning.turn_counter,
          threshold: warning.threshold,
          observed_value: warning.observed_value,
          ...(warning.window_start_turn === undefined
            ? {}
            : { window_start_turn: warning.window_start_turn }),
          ...(warning.window_turns === undefined ? {} : { window_turns: warning.window_turns }),
        })}\n`,
      );
    }
  }

  private captureMemoryBandMetrics(borg: Borg, sessionId: SessionId): MemoryBandMetricCounts {
    const actionRecordCountByState = borg.actions.countByState();
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

    return {
      action_record_count_total: borg.actions.count(),
      action_record_count_by_state: {
        ...zeroCounts<ActionState>(ACTION_STATES),
        ...actionRecordCountByState,
      },
      action_record_count_committed_to_do: actionRecordCountByState.committed_to_do ?? 0,
      action_record_count_canonicalized: actionCountFromRepository(borg, "countCanonicalized") ?? 0,
      action_record_count_active:
        actionCountFromRepository(borg, "countActive") ??
        activeActionCountFromStateCounts(actionRecordCountByState),
      action_record_creation_source_per_turn: actionCreationSourcePerTurn,
      action_record_creation_count_this_turn: actionCreationCountTotal(actionCreationSourcePerTurn),
      recent_completed_action_count: recentCompletedActionCount,
      commitment_count_active: borg.commitments.countActive(),
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
        event: "action_duplicate_pressure_observed",
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
    const allTraceRecords = this.tracePath === undefined ? [] : readTraceEvents(this.tracePath);
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
    const openQuestions = borg.self.openQuestions.list({
      status: "open",
      limit: LARGE_COUNT_LIMIT,
    });
    const activeGoals = borg.self.goals.list({ status: "active" });
    const generationSuppressions = generationSuppressionCount(borg, context.sessionIds);
    const memoryBandMetrics = this.captureMemoryBandMetrics(borg, context.sessionId);
    const frameAnomalyMetricCounts = frameAnomalyMetrics({
      traceRecords,
      borg,
      sessionIds: context.sessionIds,
      turnId,
    });
    const actionCandidateMetricCounts = actionCandidateMetrics(traceRecords);
    const goalPromotionMetricCounts = goalPromotionMetrics(traceRecords);
    const decisionArtifactSemanticRevisionMetricCounts =
      decisionArtifactSemanticRevisionMetrics(traceRecords);
    const reviewResolverMetricCounts = reviewResolverMetrics(traceRecordsSinceLastCapture);
    await this.emitActionDuplicatePressureTrace({
      borg,
      turnId,
      turnCounter,
    });
    const row: MetricsRow = {
      event: TURN_METRICS_EVENT,
      ts: Date.now(),
      turn_counter: turnCounter,
      turnId,
      transport_chat_attempts: context.transportChatAttempts,
      episode_count: episodeResult.items.length,
      semantic_node_count: semanticNodes.length,
      semantic_node_count_by_status: semanticNodeStatusCounts(semanticNodes),
      semantic_edge_count: semanticEdges.length,
      semantic_nodes_added_since_last_check: semanticNodesAdded,
      semantic_edges_added_since_last_check: semanticEdgesAdded,
      open_question_count: openQuestions.length,
      active_goal_count: flattenGoalCount(activeGoals),
      generation_suppression_count: generationSuppressions,
      mood_valence: mood.valence,
      mood_arousal: mood.arousal,
      retrieval_latency_ms: latencyBetween(
        traceRecords,
        "retrieval_started",
        "retrieval_completed",
      ),
      deliberation_latency_ms: latencyBetween(
        traceRecords,
        "llm_call_started",
        "llm_call_response",
      ),
      borg_input_tokens: usage.inputTokens,
      borg_output_tokens: usage.outputTokens,
      open_question_resolved_count: memoryBandMetrics.open_question_resolved_count,
      action_record_count_total: memoryBandMetrics.action_record_count_total,
      action_record_count_by_state: memoryBandMetrics.action_record_count_by_state,
      action_record_count_committed_to_do: memoryBandMetrics.action_record_count_committed_to_do,
      action_record_count_canonicalized: memoryBandMetrics.action_record_count_canonicalized,
      action_record_count_active: memoryBandMetrics.action_record_count_active,
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
      recent_completed_action_count: memoryBandMetrics.recent_completed_action_count,
      commitment_count_active: memoryBandMetrics.commitment_count_active,
      commitment_count_superseded: memoryBandMetrics.commitment_count_superseded,
      commitment_count_revoked: memoryBandMetrics.commitment_count_revoked,
      commitment_count_expired: memoryBandMetrics.commitment_count_expired,
      commitment_count_canonicalized: memoryBandMetrics.commitment_count_canonicalized,
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
        decisionArtifactSemanticRevisionMetricCounts.decision_artifact_semantic_revisions_attempted,
      decision_artifact_semantic_revisions_completed_succeeded:
        decisionArtifactSemanticRevisionMetricCounts.decision_artifact_semantic_revisions_completed_succeeded,
      decision_artifact_semantic_nodes_marked_superseded:
        decisionArtifactSemanticRevisionMetricCounts.decision_artifact_semantic_nodes_marked_superseded,
      decision_artifact_semantic_nodes_marked_contradicted:
        decisionArtifactSemanticRevisionMetricCounts.decision_artifact_semantic_nodes_marked_contradicted,
      decision_artifact_semantic_revision_cache_hits:
        decisionArtifactSemanticRevisionMetricCounts.decision_artifact_semantic_revision_cache_hits,
      decision_artifact_semantic_revision_cache_size: this.semanticRevisionVerdictCacheSize(),
      overseer_due_on_suppressed_turn: context.overseerDueOnSuppressedTurn ?? false,
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
    const mood = borg.mood.current(context.sessionId);
    const episodeResult = await borg.episodic.list({ limit: LARGE_COUNT_LIMIT });
    const semanticNodes = await borg.semantic.nodes.list({ limit: LARGE_COUNT_LIMIT });
    const semanticEdges = borg.semantic.edges.list({ includeInvalid: true });
    const openQuestions = borg.self.openQuestions.list({
      status: "open",
      limit: LARGE_COUNT_LIMIT,
    });
    const activeGoals = borg.self.goals.list({ status: "active" });
    const generationSuppressions = generationSuppressionCount(borg, context.sessionIds);
    const memoryBandMetrics = this.captureMemoryBandMetrics(borg, context.sessionId);
    const turnId = context.turnId ?? `${event}_${turnCounter}`;
    const frameAnomalyMetricCounts = frameAnomalyMetrics({
      traceRecords: [],
      borg,
      sessionIds: context.sessionIds,
      turnId,
    });
    const actionCandidateMetricCounts = actionCandidateMetrics([]);
    const goalPromotionMetricCounts = goalPromotionMetrics([]);
    const decisionArtifactSemanticRevisionMetricCounts = decisionArtifactSemanticRevisionMetrics(
      [],
    );
    const reviewResolverMetricCounts = reviewResolverMetrics([]);
    const row: MetricsRow = {
      event,
      ts: Date.now(),
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
      action_record_count_total: memoryBandMetrics.action_record_count_total,
      action_record_count_by_state: memoryBandMetrics.action_record_count_by_state,
      action_record_count_committed_to_do: memoryBandMetrics.action_record_count_committed_to_do,
      action_record_count_canonicalized: memoryBandMetrics.action_record_count_canonicalized,
      action_record_count_active: memoryBandMetrics.action_record_count_active,
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
      recent_completed_action_count: memoryBandMetrics.recent_completed_action_count,
      commitment_count_active: memoryBandMetrics.commitment_count_active,
      commitment_count_superseded: memoryBandMetrics.commitment_count_superseded,
      commitment_count_revoked: memoryBandMetrics.commitment_count_revoked,
      commitment_count_expired: memoryBandMetrics.commitment_count_expired,
      commitment_count_canonicalized: memoryBandMetrics.commitment_count_canonicalized,
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
        decisionArtifactSemanticRevisionMetricCounts.decision_artifact_semantic_revisions_attempted,
      decision_artifact_semantic_revisions_completed_succeeded:
        decisionArtifactSemanticRevisionMetricCounts.decision_artifact_semantic_revisions_completed_succeeded,
      decision_artifact_semantic_nodes_marked_superseded:
        decisionArtifactSemanticRevisionMetricCounts.decision_artifact_semantic_nodes_marked_superseded,
      decision_artifact_semantic_nodes_marked_contradicted:
        decisionArtifactSemanticRevisionMetricCounts.decision_artifact_semantic_nodes_marked_contradicted,
      decision_artifact_semantic_revision_cache_hits:
        decisionArtifactSemanticRevisionMetricCounts.decision_artifact_semantic_revision_cache_hits,
      decision_artifact_semantic_revision_cache_size: this.semanticRevisionVerdictCacheSize(),
      overseer_due_on_suppressed_turn: context.overseerDueOnSuppressedTurn ?? false,
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
