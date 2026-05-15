import { performance } from "node:perf_hooks";

import {
  QUARANTINED_USER_ENTRY_EVENT,
  ACTION_STATES,
  RELATIONAL_SLOT_STATES,
  REVIEW_KINDS,
  SEMANTIC_NODE_STATUSES,
  type ActionRecordCreationSource,
  type ActionState,
  type Borg,
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
import { filterActiveStreamEntries } from "../src/stream/index.js";
import type { ActionId } from "../src/util/ids.js";
import { readTraceEvents } from "../assessor/trace-reader.js";
import type { TraceRecord } from "../assessor/types.js";

import { appendJsonlLine } from "./jsonl.js";
import type { MetricsRow } from "./types.js";

const LARGE_COUNT_LIMIT = 1_000_000;
const TURN_METRICS_EVENT = "turn_metrics";
const ABORTED_TURN_EVENT = "aborted_turn";
const ABORTED_ATTEMPT_EVENT = "aborted_attempt";
const OPEN_QUESTION_RECORD_TYPE = "open_question";
const RESOLVED_STATUS = "resolved";
const ACTION_CREATION_SOURCES = ["extractor", "reflector", "api", "unknown"] as const;
const ACTIVE_ACTION_STATES: readonly ActionState[] = [
  "considering",
  "committed_to_do",
  "scheduled",
  "unknown",
];
const ACTION_DUPLICATE_PRESSURE_CHECK_EVERY_TURNS = 10;

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

function zeroCounts<K extends string>(keys: readonly K[]): Record<K, number> {
  return Object.fromEntries(keys.map((key) => [key, 0])) as Record<K, number>;
}

function zeroActionCreationCounts(): Record<ActionRecordCreationSource, number> {
  return zeroCounts(ACTION_CREATION_SOURCES);
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
  private previousSemanticNodeCount?: number;
  private previousSemanticEdgeCount?: number;
  private previousActionCreationCountsBySource: Record<ActionRecordCreationSource, number> =
    zeroActionCreationCounts();
  private readonly completedActionIdsSeen = new Set<ActionId>();

  constructor(filepath: string, options: MetricsCaptureOptions = {}) {
    this.filepath = filepath;
    this.tracePath = options.tracePath;
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
    const traceRecords =
      this.tracePath === undefined
        ? []
        : readTraceEvents(this.tracePath).filter((record) => record.turnId === turnId);
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
      overseer_due_on_suppressed_turn: context.overseerDueOnSuppressedTurn ?? false,
    };

    this.previousSemanticNodeCount = semanticNodes.length;
    this.previousSemanticEdgeCount = semanticEdges.length;
    this.previousActionCreationCountsBySource = actionCreationCountsFromRepository(borg);
    appendJsonlLine(this.filepath, `${JSON.stringify(row)}\n`);
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
      overseer_due_on_suppressed_turn: context.overseerDueOnSuppressedTurn ?? false,
    };

    this.previousActionCreationCountsBySource = actionCreationCountsFromRepository(borg);
    appendJsonlLine(this.filepath, `${JSON.stringify(row)}\n`);
    return row;
  }

  close(): void {
    // Metrics rows are fsynced on each append.
  }
}
