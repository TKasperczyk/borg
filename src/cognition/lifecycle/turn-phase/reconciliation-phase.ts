import {
  reconcileSharedStateCanonicalizations,
  type SharedStateReconciliationRepositories,
  type SharedStateReconciliationResult,
  type SharedStateUnsettledReconciliation,
} from "../../shared-state/index.js";
import type { SharedStateSourceTrustValidator } from "../../../memory/decision-artifacts/index.js";
import { toTraceJsonValue, type TurnTracer } from "../../tracing/tracer.js";

function sharedStateReconciliationOutcomeCounts(
  result: SharedStateReconciliationResult,
): Record<string, number> {
  return {
    goals_retired: result.goals_retired,
    commitments_retired: result.commitments_retired,
    actions_retired: result.actions_retired,
    open_questions_retired: result.open_questions_retired,
    goals_canonicalized_attempted: result.goals_canonicalized_attempted,
    goals_canonicalized_succeeded: result.goals_canonicalized_succeeded,
    goals_canonicalized_skipped: result.goals_canonicalized_skipped,
    commitments_revoked_attempted: result.commitments_revoked_attempted,
    commitments_revoked_succeeded: result.commitments_revoked_succeeded,
    commitments_revoked_skipped: result.commitments_revoked_skipped,
    actions_completed_attempted: result.actions_completed_attempted,
    actions_completed_succeeded: result.actions_completed_succeeded,
    actions_completed_skipped: result.actions_completed_skipped,
    actions_closed_by_borg_self_performance: result.actions_closed_by_borg_self_performance,
    open_questions_resolved_attempted: result.open_questions_resolved_attempted,
    open_questions_resolved_succeeded: result.open_questions_resolved_succeeded,
    open_questions_resolved_skipped: result.open_questions_resolved_skipped,
    semantic_nodes_reviewed_attempted: result.semantic_nodes_reviewed_attempted,
    semantic_nodes_marked_superseded: result.semantic_nodes_marked_superseded,
    semantic_nodes_marked_contradicted: result.semantic_nodes_marked_contradicted,
    semantic_nodes_skipped: result.semantic_nodes_skipped,
    unknown_id_count: result.unknown_ids.length,
    skipped_commitment_count: result.skipped_commitments.length,
    error_count: result.errors.length,
  };
}

export function runSharedStateArtifactRetryOnlyReconciliation(input: {
  unsettledReconciliation: SharedStateUnsettledReconciliation;
  repositories: SharedStateReconciliationRepositories;
  sourceTrustValidator: SharedStateSourceTrustValidator;
  nowMs: number;
  tracer: TurnTracer;
  turnId?: string;
}): void {
  const result = reconcileSharedStateCanonicalizations({
    entries: input.unsettledReconciliation.entries,
    repositories: input.repositories,
    nowMs: input.nowMs,
    sourceTrustValidator: input.sourceTrustValidator,
    tracer: input.tracer,
    turnId: input.turnId,
  });

  if (input.tracer.enabled && input.turnId !== undefined) {
    input.tracer.emit("shared_state.reconcile.completed", {
      turnId: input.turnId,
      mode: "retry_only",
      unsettled_entry_count: input.unsettledReconciliation.entries.length,
      outcome_counts: toTraceJsonValue(sharedStateReconciliationOutcomeCounts(result)),
    });
  }
}
