import { closeSync, fsyncSync, mkdirSync, openSync, writeFileSync } from "node:fs";
import { dirname } from "node:path";

import {
  ACTION_STATES,
  RELATIONAL_SLOT_STATES,
  REVIEW_KINDS,
  type ActionState,
  type Borg,
  type RelationalSlotState,
  type ReviewKind,
  type SessionId,
} from "../src/index.js";
import { filterActiveStreamEntries } from "../src/stream/index.js";
import type { ActionId } from "../src/util/ids.js";
import { readTraceEvents } from "../assessor/trace-reader.js";
import type { TraceRecord } from "../assessor/types.js";

import type { MetricsRow } from "./types.js";

const LARGE_COUNT_LIMIT = 1_000_000;
const TURN_METRICS_EVENT = "turn_metrics";
const ABORTED_TURN_EVENT = "aborted_turn";
const ABORTED_ATTEMPT_EVENT = "aborted_attempt";
const OPEN_QUESTION_RECORD_TYPE = "open_question";
const RESOLVED_STATUS = "resolved";

type GoalTreeNodeLike = {
  children?: GoalTreeNodeLike[];
};

type MemoryBandMetricCounts = Pick<
  MetricsRow,
  | "action_record_count_total"
  | "action_record_count_by_state"
  | "recent_completed_action_count"
  | "commitment_count_active"
  | "commitment_count_superseded"
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
};

function appendJsonlLine(filePath: string, line: string): void {
  mkdirSync(dirname(filePath), { recursive: true });

  let fileDescriptor: number | undefined;

  try {
    fileDescriptor = openSync(filePath, "a");
    writeFileSync(fileDescriptor, line);
    fsyncSync(fileDescriptor);
  } finally {
    if (fileDescriptor !== undefined) {
      closeSync(fileDescriptor);
    }
  }
}

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

function zeroCounts<K extends string>(keys: readonly K[]): Record<K, number> {
  return Object.fromEntries(keys.map((key) => [key, 0])) as Record<K, number>;
}

function reviewQueueOpenCountByType(borg: Borg): Record<ReviewKind, number> {
  const counts = zeroCounts(REVIEW_KINDS);

  for (const item of borg.review.list({ openOnly: true })) {
    counts[item.kind] += 1;
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
  private readonly completedActionIdsSeen = new Set<ActionId>();

  constructor(filepath: string, options: MetricsCaptureOptions = {}) {
    this.filepath = filepath;
    this.tracePath = options.tracePath;
  }

  private captureMemoryBandMetrics(borg: Borg, sessionId: SessionId): MemoryBandMetricCounts {
    const actionRecordCountByState = borg.actions.countByState();
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
      recent_completed_action_count: recentCompletedActionCount,
      commitment_count_active: borg.commitments.countActive(),
      commitment_count_superseded: borg.commitments.countSuperseded(),
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
    const row: MetricsRow = {
      event: TURN_METRICS_EVENT,
      ts: Date.now(),
      turn_counter: turnCounter,
      turnId,
      transport_chat_attempts: context.transportChatAttempts,
      episode_count: episodeResult.items.length,
      semantic_node_count: semanticNodes.length,
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
      recent_completed_action_count: memoryBandMetrics.recent_completed_action_count,
      commitment_count_active: memoryBandMetrics.commitment_count_active,
      commitment_count_superseded: memoryBandMetrics.commitment_count_superseded,
      pending_action_count: memoryBandMetrics.pending_action_count,
      pending_action_merge_count: memoryBandMetrics.pending_action_merge_count,
      relational_slot_count_by_state: memoryBandMetrics.relational_slot_count_by_state,
      review_queue_open_count_by_type: memoryBandMetrics.review_queue_open_count_by_type,
    };

    this.previousSemanticNodeCount = semanticNodes.length;
    this.previousSemanticEdgeCount = semanticEdges.length;
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
    const row: MetricsRow = {
      event,
      ts: Date.now(),
      turn_counter: turnCounter,
      turnId: context.turnId ?? `${event}_${turnCounter}`,
      transport_chat_attempts: context.transportChatAttempts,
      failure_reason: context.failureReason,
      episode_count: episodeResult.items.length,
      semantic_node_count: semanticNodes.length,
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
      recent_completed_action_count: memoryBandMetrics.recent_completed_action_count,
      commitment_count_active: memoryBandMetrics.commitment_count_active,
      commitment_count_superseded: memoryBandMetrics.commitment_count_superseded,
      pending_action_count: memoryBandMetrics.pending_action_count,
      pending_action_merge_count: memoryBandMetrics.pending_action_merge_count,
      relational_slot_count_by_state: memoryBandMetrics.relational_slot_count_by_state,
      review_queue_open_count_by_type: memoryBandMetrics.review_queue_open_count_by_type,
    };

    appendJsonlLine(this.filepath, `${JSON.stringify(row)}\n`);
    return row;
  }

  close(): void {
    // Metrics rows are fsynced on each append.
  }
}
