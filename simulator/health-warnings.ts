import type { MetricsRow, OverseerVerdict, SimulatorHealthWarning } from "./types.js";

// Existing goal thresholds: high enough to catch runaway promotion without
// flagging ordinary long-running planning scenarios.
export const ACTIVE_GOAL_HIGH_THRESHOLD = 25;
export const ACTIVE_GOAL_GROWTH_START_TURN = 20;
export const ACTIVE_GOAL_GROWTH_WINDOW_ROWS = 10;
export const ACTIVE_GOAL_GROWTH_THRESHOLD_PER_TURN = 0.5;

// Action thresholds: incident scenarios can legitimately create denser action
// state, so only sustained/final accumulation is flagged.
export const ACTIVE_ACTION_FINAL_HIGH_THRESHOLD = 30;
export const COMMITTED_TO_DO_ACTION_FINAL_HIGH_THRESHOLD = 18;
export const ACTIONS_PER_TURN_HIGH_THRESHOLD = 2;
export const SALIENT_ACTIONS_PER_TURN_HIGH_THRESHOLD = 0.8;
export const ACTION_RETIREMENT_RATIO_LOW_THRESHOLD = 0.3;
export const ACTION_RETIREMENT_RATIO_MIN_TOTAL_ACTIONS = 10;
export const DORMANT_ARCHIVE_ELIGIBLE_COUNT_HIGH_THRESHOLD = 0;

// Canonicalization is expected to remain sparse, but a large action set with
// almost no canonicalization is a stabilization signal.
export const ACTION_CANONICALIZATION_MIN_TOTAL_ACTIONS = 30;
export const ACTION_CANONICALIZATION_RATE_LOW_THRESHOLD = 0.05;

// Latency thresholds are deliberately above ordinary local variance.
export const RETRIEVAL_LATENCY_MAX_HIGH_THRESHOLD_MS = 30_000;
export const DELIBERATION_LATENCY_MAX_HIGH_THRESHOLD_MS = 120_000;

// Semantic revision thresholds target runaway LLM churn and low state-change
// yield over a full simulator run, not one-off misses.
export const SEMANTIC_REVISION_LLM_CALLS_HIGH_THRESHOLD = 40;
export const SEMANTIC_REVISION_TRANSITION_YIELD_MIN_CALLS = 6;
export const SEMANTIC_REVISION_TRANSITION_YIELD_LOW_THRESHOLD = 0.25;
export const SEMANTIC_REVISION_DEGRADED_HIGH_THRESHOLD = 3;

// Classifier degradation is noisy in tiny samples; wait for a meaningful call
// count before flagging.
export const CLASSIFIER_DEGRADED_RATE_MIN_CALLS = 10;
export const CLASSIFIER_DEGRADED_RATE_HIGH_THRESHOLD = 0.2;

// Any validated unsupported/contradicted capability overclaim is worth surfacing.
export const CAPABILITY_OVERCLAIM_COUNT_HIGH_THRESHOLD = 1;
export const CAPABILITY_AMBIGUITY_COUNT_HIGH_THRESHOLD = 3;

export const CLOSURE_LOOP_DEGRADED_RATE_HIGH_THRESHOLD = 0.1;
export const CORRECTIVE_PREFERENCE_DEGRADED_RATE_HIGH_THRESHOLD = 0.1;
export const EXTRACTOR_MAX_TOKENS_HIGH_THRESHOLD = 1;
export const EXTRACTOR_MAX_TOKENS_SEVERE_THRESHOLD = 3;

// Review backlog needs to be large enough to indicate drain trouble, not just
// a few expected review items.
export const REVIEW_QUEUE_BACKLOG_HIGH_THRESHOLD = 50;
export const SHARED_STATE_CAP_SATURATION_HIGH_THRESHOLD = 0.5;
export const SHARED_STATE_STARVATION_HIGH_THRESHOLD = 1;
export const SHARED_STATE_STARVATION_PERSISTENT_THRESHOLD = 1;
export const SHARED_STATE_COMPILER_ADD_DOMINANT_THRESHOLD = 2;
export const SHARED_STATE_COMPILER_MAX_TOKENS_HIGH_THRESHOLD = 1;

export type SimulatorHealthWarningOptions = {
  scenarioKey?: string;
  overseerCheckpoints?: readonly OverseerVerdict[];
};

export type CapabilityFindingMetrics = Pick<
  MetricsRow,
  "capability_overclaim_count" | "capability_ambiguity_count" | "capability_boundary_refusal_count"
>;

function total(values: Record<string, number>): number {
  return Object.values(values).reduce((sum, value) => sum + value, 0);
}

function latestWarning(input: {
  row: MetricsRow;
  kind: SimulatorHealthWarning["kind"];
  threshold: number;
  observedValue: number;
  label?: string;
}): SimulatorHealthWarning {
  return {
    kind: input.kind,
    turn_counter: input.row.turn_counter,
    turnId: input.row.turnId,
    threshold: input.threshold,
    observed_value: input.observedValue,
    ...(input.label === undefined ? {} : { label: input.label }),
  };
}

function maxNumberRow(
  rows: readonly MetricsRow[],
  value: (row: MetricsRow) => number | null,
): { row: MetricsRow; value: number } | null {
  let max: { row: MetricsRow; value: number } | null = null;

  for (const row of rows) {
    const rowValue = value(row);

    if (rowValue === null) {
      continue;
    }

    if (max === null || rowValue > max.value) {
      max = { row, value: rowValue };
    }
  }

  return max;
}

function isActiveCapabilityFinding(finding: OverseerVerdict["findings"][number]): boolean {
  return finding.category === "K" && finding.carryover_demoted !== true;
}

function maxLabelCount(counts: Record<string, number>): { label: string; count: number } | null {
  let max: { label: string; count: number } | null = null;

  for (const [label, count] of Object.entries(counts)) {
    if (!Number.isFinite(count)) {
      continue;
    }

    if (max === null || count > max.count) {
      max = { label, count };
    }
  }

  return max;
}

export function capabilityFindingMetrics(
  overseerCheckpoints: readonly OverseerVerdict[] | undefined,
): CapabilityFindingMetrics {
  if (overseerCheckpoints === undefined) {
    return {
      capability_overclaim_count: 0,
      capability_ambiguity_count: 0,
      capability_boundary_refusal_count: 0,
    };
  }

  const metrics: CapabilityFindingMetrics = {
    capability_overclaim_count: 0,
    capability_ambiguity_count: 0,
    capability_boundary_refusal_count: 0,
  };

  for (const checkpoint of overseerCheckpoints) {
    for (const finding of checkpoint.findings) {
      if (!isActiveCapabilityFinding(finding)) {
        continue;
      }

      if (
        (finding.claim_status === "unsupported" || finding.claim_status === "contradicted") &&
        finding.status_impact !== "none"
      ) {
        metrics.capability_overclaim_count += 1;
        continue;
      }

      if (finding.claim_status === "unclear" && finding.status_impact !== "none") {
        metrics.capability_ambiguity_count += 1;
        continue;
      }

      if (finding.claim_status === "grounded" && finding.status_impact === "none") {
        metrics.capability_boundary_refusal_count += 1;
      }
    }
  }

  return metrics;
}

function degradedRate(input: { completed: number; degraded: number }): number | null {
  if (input.completed === 0) {
    return input.degraded > 0 ? 1 : null;
  }

  return input.degraded / input.completed;
}

export function simulatorHealthWarningsForRows(
  rows: readonly MetricsRow[],
  options: SimulatorHealthWarningOptions = {},
): SimulatorHealthWarning[] {
  const latest = rows.at(-1);

  if (latest === undefined) {
    return [];
  }

  const warnings: SimulatorHealthWarning[] = [];

  if (latest.active_goal_count > ACTIVE_GOAL_HIGH_THRESHOLD) {
    warnings.push({
      kind: "active_goals_high",
      turn_counter: latest.turn_counter,
      turnId: latest.turnId,
      threshold: ACTIVE_GOAL_HIGH_THRESHOLD,
      observed_value: latest.active_goal_count,
    });
  }

  if (latest.turn_counter > ACTIVE_GOAL_GROWTH_START_TURN) {
    const postStartRows = rows.filter((row) => row.turn_counter > ACTIVE_GOAL_GROWTH_START_TURN);
    const windowRows = postStartRows.slice(-ACTIVE_GOAL_GROWTH_WINDOW_ROWS);
    const first = windowRows[0];
    const last = windowRows.at(-1);

    if (
      windowRows.length === ACTIVE_GOAL_GROWTH_WINDOW_ROWS &&
      first !== undefined &&
      last !== undefined &&
      last.turn_counter > first.turn_counter
    ) {
      const slope =
        (last.active_goal_count - first.active_goal_count) /
        (last.turn_counter - first.turn_counter);

      if (slope > ACTIVE_GOAL_GROWTH_THRESHOLD_PER_TURN) {
        warnings.push({
          kind: "active_goals_growth_high",
          turn_counter: latest.turn_counter,
          turnId: latest.turnId,
          threshold: ACTIVE_GOAL_GROWTH_THRESHOLD_PER_TURN,
          observed_value: slope,
          window_start_turn: first.turn_counter,
          window_turns: last.turn_counter - first.turn_counter,
        });
      }
    }
  }

  if (latest.action_record_count_active > ACTIVE_ACTION_FINAL_HIGH_THRESHOLD) {
    warnings.push(
      latestWarning({
        row: latest,
        kind: "active_actions_final_high",
        threshold: ACTIVE_ACTION_FINAL_HIGH_THRESHOLD,
        observedValue: latest.action_record_count_active,
      }),
    );
  }

  if (latest.action_record_count_committed_to_do > COMMITTED_TO_DO_ACTION_FINAL_HIGH_THRESHOLD) {
    warnings.push(
      latestWarning({
        row: latest,
        kind: "committed_to_do_actions_final_high",
        threshold: COMMITTED_TO_DO_ACTION_FINAL_HIGH_THRESHOLD,
        observedValue: latest.action_record_count_committed_to_do,
      }),
    );
  }

  if (latest.actions_per_turn > ACTIONS_PER_TURN_HIGH_THRESHOLD) {
    warnings.push(
      latestWarning({
        row: latest,
        kind: "actions_per_turn_high",
        threshold: ACTIONS_PER_TURN_HIGH_THRESHOLD,
        observedValue: latest.actions_per_turn,
      }),
    );
  }

  if (latest.salient_actions_per_turn > SALIENT_ACTIONS_PER_TURN_HIGH_THRESHOLD) {
    warnings.push(
      latestWarning({
        row: latest,
        kind: "salient_actions_per_turn_high",
        threshold: SALIENT_ACTIONS_PER_TURN_HIGH_THRESHOLD,
        observedValue: latest.salient_actions_per_turn,
      }),
    );
  }

  if (
    latest.action_record_count_total >= ACTION_RETIREMENT_RATIO_MIN_TOTAL_ACTIONS &&
    latest.action_retirement_ratio < ACTION_RETIREMENT_RATIO_LOW_THRESHOLD
  ) {
    warnings.push(
      latestWarning({
        row: latest,
        kind: "action_retirement_ratio_low",
        threshold: ACTION_RETIREMENT_RATIO_LOW_THRESHOLD,
        observedValue: latest.action_retirement_ratio,
      }),
    );
  }

  if (latest.action_record_count_total >= ACTION_CANONICALIZATION_MIN_TOTAL_ACTIONS) {
    const canonicalizationRate =
      latest.action_record_count_canonicalized / latest.action_record_count_total;

    if (canonicalizationRate < ACTION_CANONICALIZATION_RATE_LOW_THRESHOLD) {
      warnings.push(
        latestWarning({
          row: latest,
          kind: "action_canonicalization_rate_low",
          threshold: ACTION_CANONICALIZATION_RATE_LOW_THRESHOLD,
          observedValue: canonicalizationRate,
        }),
      );
    }
  }

  if (latest.dormant_archive_eligible_count > DORMANT_ARCHIVE_ELIGIBLE_COUNT_HIGH_THRESHOLD) {
    warnings.push(
      latestWarning({
        row: latest,
        kind: "dormant_archive_eligible_count_high",
        threshold: DORMANT_ARCHIVE_ELIGIBLE_COUNT_HIGH_THRESHOLD,
        observedValue: latest.dormant_archive_eligible_count,
      }),
    );
  }

  const maxRetrievalLatency = maxNumberRow(rows, (row) => row.retrieval_latency_ms);

  if (
    maxRetrievalLatency !== null &&
    maxRetrievalLatency.value > RETRIEVAL_LATENCY_MAX_HIGH_THRESHOLD_MS
  ) {
    warnings.push(
      latestWarning({
        row: maxRetrievalLatency.row,
        kind: "retrieval_latency_max_high",
        threshold: RETRIEVAL_LATENCY_MAX_HIGH_THRESHOLD_MS,
        observedValue: maxRetrievalLatency.value,
      }),
    );
  }

  const maxDeliberationLatency = maxNumberRow(rows, (row) => row.deliberation_latency_ms);

  if (
    maxDeliberationLatency !== null &&
    maxDeliberationLatency.value > DELIBERATION_LATENCY_MAX_HIGH_THRESHOLD_MS
  ) {
    warnings.push(
      latestWarning({
        row: maxDeliberationLatency.row,
        kind: "deliberation_latency_max_high",
        threshold: DELIBERATION_LATENCY_MAX_HIGH_THRESHOLD_MS,
        observedValue: maxDeliberationLatency.value,
      }),
    );
  }

  const semanticRevisionCalls = rows.reduce(
    (sum, row) => sum + row.decision_artifact_semantic_revisions_attempted,
    0,
  );

  if (semanticRevisionCalls > SEMANTIC_REVISION_LLM_CALLS_HIGH_THRESHOLD) {
    warnings.push(
      latestWarning({
        row: latest,
        kind: "semantic_revision_llm_calls_high",
        threshold: SEMANTIC_REVISION_LLM_CALLS_HIGH_THRESHOLD,
        observedValue: semanticRevisionCalls,
      }),
    );
  }

  if (semanticRevisionCalls >= SEMANTIC_REVISION_TRANSITION_YIELD_MIN_CALLS) {
    const semanticTransitions = rows.reduce(
      (sum, row) =>
        sum +
        row.decision_artifact_semantic_nodes_marked_superseded +
        row.decision_artifact_semantic_nodes_marked_contradicted,
      0,
    );
    const transitionYield = semanticTransitions / semanticRevisionCalls;

    if (transitionYield < SEMANTIC_REVISION_TRANSITION_YIELD_LOW_THRESHOLD) {
      warnings.push(
        latestWarning({
          row: latest,
          kind: "semantic_revision_transition_yield_low",
          threshold: SEMANTIC_REVISION_TRANSITION_YIELD_LOW_THRESHOLD,
          observedValue: transitionYield,
        }),
      );
    }
  }

  const semanticRevisionErrors = rows.reduce(
    (sum, row) => sum + row.semantic_revision_error_count,
    0,
  );

  if (semanticRevisionErrors >= SEMANTIC_REVISION_DEGRADED_HIGH_THRESHOLD) {
    warnings.push(
      latestWarning({
        row: latest,
        kind: "semantic_revision_degraded_high",
        threshold: SEMANTIC_REVISION_DEGRADED_HIGH_THRESHOLD,
        observedValue: semanticRevisionErrors,
      }),
    );
  }

  const classifierCalls = rows.reduce((sum, row) => sum + row.frame_anomaly_classifier_calls, 0);

  if (classifierCalls >= CLASSIFIER_DEGRADED_RATE_MIN_CALLS) {
    const classifierDegraded = rows.reduce((sum, row) => sum + row.frame_anomaly_degraded_count, 0);
    const degradedRate = classifierDegraded / classifierCalls;

    if (degradedRate > CLASSIFIER_DEGRADED_RATE_HIGH_THRESHOLD) {
      warnings.push(
        latestWarning({
          row: latest,
          kind: "classifier_degraded_rate_high",
          threshold: CLASSIFIER_DEGRADED_RATE_HIGH_THRESHOLD,
          observedValue: degradedRate,
        }),
      );
    }
  }

  const closureLoopCompleted = rows.reduce((sum, row) => sum + row.closure_loop_completed_count, 0);

  const closureLoopDegraded = rows.reduce((sum, row) => sum + row.closure_loop_degraded_count, 0);
  const closureLoopDegradedRate = degradedRate({
    completed: closureLoopCompleted,
    degraded: closureLoopDegraded,
  });

  if (
    closureLoopDegradedRate !== null &&
    closureLoopDegradedRate > CLOSURE_LOOP_DEGRADED_RATE_HIGH_THRESHOLD
  ) {
    warnings.push(
      latestWarning({
        row: latest,
        kind: "closure_loop_degraded_rate_high",
        threshold: CLOSURE_LOOP_DEGRADED_RATE_HIGH_THRESHOLD,
        observedValue: closureLoopDegradedRate,
      }),
    );
  }

  const correctivePreferenceCompleted = rows.reduce(
    (sum, row) => sum + row.corrective_preference_completed_count,
    0,
  );

  const correctivePreferenceDegraded = rows.reduce(
    (sum, row) => sum + row.corrective_preference_degraded_count,
    0,
  );
  const correctivePreferenceDegradedRate = degradedRate({
    completed: correctivePreferenceCompleted,
    degraded: correctivePreferenceDegraded,
  });

  if (
    correctivePreferenceDegradedRate !== null &&
    correctivePreferenceDegradedRate > CORRECTIVE_PREFERENCE_DEGRADED_RATE_HIGH_THRESHOLD
  ) {
    warnings.push(
      latestWarning({
        row: latest,
        kind: "corrective_preference_degraded_rate_high",
        threshold: CORRECTIVE_PREFERENCE_DEGRADED_RATE_HIGH_THRESHOLD,
        observedValue: correctivePreferenceDegradedRate,
      }),
    );
  }

  const extractorMaxTokensStops = maxLabelCount(latest.extractor_max_tokens_total_by_label);

  if (
    extractorMaxTokensStops !== null &&
    extractorMaxTokensStops.count >= EXTRACTOR_MAX_TOKENS_HIGH_THRESHOLD
  ) {
    warnings.push(
      latestWarning({
        row: latest,
        kind: "extractor_max_tokens_high",
        threshold: EXTRACTOR_MAX_TOKENS_HIGH_THRESHOLD,
        observedValue: extractorMaxTokensStops.count,
        label: extractorMaxTokensStops.label,
      }),
    );
  }

  if (
    latest.shared_state_compiler_max_tokens_total >= SHARED_STATE_COMPILER_MAX_TOKENS_HIGH_THRESHOLD
  ) {
    warnings.push(
      latestWarning({
        row: latest,
        kind: "shared_state_compiler_max_tokens_high",
        threshold: SHARED_STATE_COMPILER_MAX_TOKENS_HIGH_THRESHOLD,
        observedValue: latest.shared_state_compiler_max_tokens_total,
        label: "decision_artifact_compiler",
      }),
    );
  }

  if (
    extractorMaxTokensStops !== null &&
    extractorMaxTokensStops.count >= EXTRACTOR_MAX_TOKENS_SEVERE_THRESHOLD
  ) {
    warnings.push(
      latestWarning({
        row: latest,
        kind: "extractor_max_tokens_severe",
        threshold: EXTRACTOR_MAX_TOKENS_SEVERE_THRESHOLD,
        observedValue: extractorMaxTokensStops.count,
        label: extractorMaxTokensStops.label,
      }),
    );
  }

  const capabilityMetrics = capabilityFindingMetrics(options.overseerCheckpoints);
  const overclaimCount = capabilityMetrics.capability_overclaim_count;

  if (overclaimCount >= CAPABILITY_OVERCLAIM_COUNT_HIGH_THRESHOLD) {
    warnings.push(
      latestWarning({
        row: latest,
        kind: "capability_overclaim_count_high",
        threshold: CAPABILITY_OVERCLAIM_COUNT_HIGH_THRESHOLD,
        observedValue: overclaimCount,
      }),
    );
  }

  const ambiguityCount = capabilityMetrics.capability_ambiguity_count;

  if (ambiguityCount >= CAPABILITY_AMBIGUITY_COUNT_HIGH_THRESHOLD) {
    warnings.push(
      latestWarning({
        row: latest,
        kind: "capability_ambiguity_count_high",
        threshold: CAPABILITY_AMBIGUITY_COUNT_HIGH_THRESHOLD,
        observedValue: ambiguityCount,
      }),
    );
  }

  const reviewQueueBacklog = total(latest.review_queue_open_count_by_type);

  if (reviewQueueBacklog > REVIEW_QUEUE_BACKLOG_HIGH_THRESHOLD) {
    warnings.push(
      latestWarning({
        row: latest,
        kind: "review_queue_backlog_high",
        threshold: REVIEW_QUEUE_BACKLOG_HIGH_THRESHOLD,
        observedValue: reviewQueueBacklog,
      }),
    );
  }

  const sharedStateCapSaturation =
    latest.shared_state_compile_evaluated_turns <= 0
      ? 0
      : latest.shared_state_at_cap_turns / latest.shared_state_compile_evaluated_turns;

  if (sharedStateCapSaturation > SHARED_STATE_CAP_SATURATION_HIGH_THRESHOLD) {
    warnings.push(
      latestWarning({
        row: latest,
        kind: "shared_state_cap_saturation_high",
        threshold: SHARED_STATE_CAP_SATURATION_HIGH_THRESHOLD,
        observedValue: sharedStateCapSaturation,
      }),
    );
  }

  if (latest.shared_state_live_starvation_ever) {
    warnings.push(
      latestWarning({
        row: latest,
        kind: "shared_state_starvation_high",
        threshold: SHARED_STATE_STARVATION_HIGH_THRESHOLD,
        observedValue: 1,
      }),
    );
  }

  if (latest.shared_state_live_starvation_final) {
    warnings.push(
      latestWarning({
        row: latest,
        kind: "shared_state_starvation_persistent",
        threshold: SHARED_STATE_STARVATION_PERSISTENT_THRESHOLD,
        observedValue: 1,
      }),
    );
  }

  if (latest.shared_state_add_to_update_ratio > SHARED_STATE_COMPILER_ADD_DOMINANT_THRESHOLD) {
    warnings.push(
      latestWarning({
        row: latest,
        kind: "shared_state_compiler_add_dominant",
        threshold: SHARED_STATE_COMPILER_ADD_DOMINANT_THRESHOLD,
        observedValue: latest.shared_state_add_to_update_ratio,
      }),
    );
  }

  return warnings;
}
