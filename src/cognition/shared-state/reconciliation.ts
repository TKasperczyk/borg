import type {
  SharedStateArtifact,
  SharedStateEntry,
} from "../../memory/decision-artifacts/index.js";
import {
  isSharedStateArtifactCanonicalizableCommitmentType,
  isTerminalActionState,
  isTerminalCommitment,
  isTerminalGoalStatus,
  isTerminalOpenQuestionStatus,
} from "../../memory/lifecycle-ops/index.js";
import type { ActionId, CommitmentId, GoalId, OpenQuestionId } from "../../util/ids.js";
import {
  activeLockedEntries,
  contaminatedSharedStateArtifactSources,
  recordCanonicalizationSkipsForEntry,
  recordUnknownIdSkips,
  traceContaminatedSharedStateEntrySkip,
  type ReconcileSharedStateCanonicalizationsInput,
  type SharedStateReconciliationLookupRepositories,
  type SharedStateReconciliationRepositories,
  type SharedStateReconciliationResult,
  type SharedStateUnsettledReconciliation,
} from "./reconciliation-summary.js";
import { emptyCanonicalizes } from "./patch-validation.js";
import { reconcileActionCanonicalizations } from "./reconcile-actions.js";
import { reconcileCommitmentCanonicalizations } from "./reconcile-commitments.js";
import { reconcileGoalCanonicalizations } from "./reconcile-goals.js";
import { reconcileOpenQuestionCanonicalizations } from "./reconcile-open-questions.js";
export { reconcileSemanticBeliefRevision } from "./semantic-revision.js";
export { mergeSemanticBeliefRevisionResult } from "./reconciliation-summary.js";
export {
  SEMANTIC_REVISION_VERDICT_CACHE_MAX_ENTRIES,
  SemanticRevisionVerdictCache,
  clearSharedStateSemanticRevisionVerdictCache,
  sharedStateSemanticRevisionVerdictCacheSize,
  type SemanticRevisionCachedVerdict,
} from "./semantic-revision-cache.js";
export type {
  ReconcileSemanticBeliefRevisionInput,
  SharedStateReconciliationError,
  SharedStateReconciliationLookupRepositories,
  SharedStateReconciliationRepositories,
  SharedStateReconciliationResult,
  SharedStateSemanticBeliefRevisionDependencies,
  SharedStateSkippedCommitmentCanonicalization,
  SharedStateUnsettledReconciliation,
  SharedStateUnsettledReconciliationSummary,
  ReconcileSharedStateCanonicalizationsInput,
} from "./reconciliation-summary.js";

function retryEntry(
  entriesById: Map<SharedStateEntry["id"], SharedStateEntry>,
  entry: SharedStateEntry,
): SharedStateEntry {
  const existing = entriesById.get(entry.id);

  if (existing !== undefined) {
    return existing;
  }

  const next = {
    ...entry,
    canonicalizes: emptyCanonicalizes(),
  };

  entriesById.set(entry.id, next);
  return next;
}

export function findUnsettledSharedStateReconciliation(input: {
  previousArtifact: SharedStateArtifact | null | undefined;
  repositories?: SharedStateReconciliationLookupRepositories;
  nowMs?: number;
}): SharedStateUnsettledReconciliation | null {
  const entries = activeLockedEntries(input.previousArtifact?.entries ?? []);

  if (entries.length === 0) {
    return null;
  }

  const nowMs = input.nowMs ?? Date.now();
  const goalsRepository = input.repositories?.goalsRepository;
  const commitmentRepository = input.repositories?.commitmentRepository;
  const actionRepository = input.repositories?.actionRepository;
  const openQuestionsRepository = input.repositories?.openQuestionsRepository;
  const goalIds = new Set(entries.flatMap((entry) => entry.canonicalizes.goal_ids));
  const commitmentIds = new Set(entries.flatMap((entry) => entry.canonicalizes.commitment_ids));
  const actionIds = new Set(entries.flatMap((entry) => entry.canonicalizes.action_ids));
  const openQuestionIds = new Set(
    entries.flatMap((entry) => entry.canonicalizes.open_question_ids),
  );
  const retryEntriesById = new Map<SharedStateEntry["id"], SharedStateEntry>();
  let unsettledGoalCount = 0;
  let unsettledCommitmentCount = 0;
  let unsettledActionCount = 0;
  let unsettledOpenQuestionCount = 0;

  for (const entry of entries) {
    for (const goalId of entry.canonicalizes.goal_ids) {
      const goal = goalsRepository?.get?.(goalId) ?? null;

      if (goal !== null && !isTerminalGoalStatus(goal.status)) {
        retryEntry(retryEntriesById, entry).canonicalizes.goal_ids.push(goalId);
        unsettledGoalCount += 1;
      }
    }

    for (const commitmentId of entry.canonicalizes.commitment_ids) {
      const commitment = commitmentRepository?.get?.(commitmentId) ?? null;

      if (
        commitment !== null &&
        !isTerminalCommitment(commitment, nowMs) &&
        isSharedStateArtifactCanonicalizableCommitmentType(commitment.type)
      ) {
        retryEntry(retryEntriesById, entry).canonicalizes.commitment_ids.push(commitmentId);
        unsettledCommitmentCount += 1;
      }
    }

    for (const actionId of entry.canonicalizes.action_ids) {
      const action = actionRepository?.get?.(actionId) ?? null;

      if (action !== null && !isTerminalActionState(action.state)) {
        retryEntry(retryEntriesById, entry).canonicalizes.action_ids.push(actionId);
        unsettledActionCount += 1;
      }
    }

    for (const openQuestionId of entry.canonicalizes.open_question_ids) {
      const openQuestion = openQuestionsRepository?.get?.(openQuestionId) ?? null;

      if (openQuestion !== null && !isTerminalOpenQuestionStatus(openQuestion.status)) {
        retryEntry(retryEntriesById, entry).canonicalizes.open_question_ids.push(openQuestionId);
        unsettledOpenQuestionCount += 1;
      }
    }
  }

  const unsettledTotalCount =
    unsettledGoalCount +
    unsettledCommitmentCount +
    unsettledActionCount +
    unsettledOpenQuestionCount;

  if (unsettledTotalCount === 0) {
    return null;
  }

  return {
    summary: {
      active_locked_canonicalizing_entry_count: entries.length,
      referenced_goal_count: goalIds.size,
      referenced_commitment_count: commitmentIds.size,
      referenced_action_count: actionIds.size,
      referenced_open_question_count: openQuestionIds.size,
      unsettled_goal_count: unsettledGoalCount,
      unsettled_commitment_count: unsettledCommitmentCount,
      unsettled_action_count: unsettledActionCount,
      unsettled_open_question_count: unsettledOpenQuestionCount,
      unsettled_total_count: unsettledTotalCount,
    },
    entries: [...retryEntriesById.values()],
  };
}

export function reconcileSharedStateCanonicalizations(
  input: ReconcileSharedStateCanonicalizationsInput,
): SharedStateReconciliationResult {
  const result: SharedStateReconciliationResult = {
    goals_retired: 0,
    commitments_retired: 0,
    actions_retired: 0,
    open_questions_retired: 0,
    goals_canonicalized_attempted: 0,
    goals_canonicalized_succeeded: 0,
    goals_canonicalized_skipped: 0,
    commitments_revoked_attempted: 0,
    commitments_revoked_succeeded: 0,
    commitments_revoked_skipped: 0,
    actions_completed_attempted: 0,
    actions_completed_succeeded: 0,
    actions_completed_skipped: 0,
    actions_closed_by_borg_self_performance: 0,
    open_questions_resolved_attempted: 0,
    open_questions_resolved_succeeded: 0,
    open_questions_resolved_skipped: 0,
    semantic_nodes_reviewed_attempted: 0,
    semantic_nodes_marked_superseded: 0,
    semantic_nodes_marked_contradicted: 0,
    semantic_nodes_skipped: 0,
    unknown_ids: input.unknownIds ?? [],
    skipped_commitments: [],
    errors: [],
  };
  recordUnknownIdSkips(result, result.unknown_ids);
  const nowMs = input.nowMs ?? Date.now();
  const entries = activeLockedEntries(input.entries);
  const goalsRepository = input.repositories?.goalsRepository;
  const commitmentRepository = input.repositories?.commitmentRepository;
  const actionRepository = input.repositories?.actionRepository;
  const openQuestionsRepository = input.repositories?.openQuestionsRepository;
  const retiredGoals = new Set<GoalId>();
  const retiredCommitments = new Set<CommitmentId>();
  const retiredActions = new Set<ActionId>();
  const retiredOpenQuestions = new Set<OpenQuestionId>();

  for (const entry of entries) {
    const contaminatedSources = contaminatedSharedStateArtifactSources(
      entry,
      input.sourceTrustValidator,
    );

    if (contaminatedSources.length > 0) {
      recordCanonicalizationSkipsForEntry(result, entry);
      traceContaminatedSharedStateEntrySkip({
        tracer: input.tracer,
        turnId: input.turnId,
        entry,
        contaminatedSources,
      });
      continue;
    }

    reconcileGoalCanonicalizations({
      entry,
      goalIds: entry.canonicalizes.goal_ids,
      repository: goalsRepository,
      retiredGoals,
      result,
      tracer: input.tracer,
      turnId: input.turnId,
    });

    reconcileCommitmentCanonicalizations({
      entry,
      commitmentIds: entry.canonicalizes.commitment_ids,
      repository: commitmentRepository,
      retiredCommitments,
      result,
      nowMs,
      tracer: input.tracer,
      turnId: input.turnId,
    });

    reconcileActionCanonicalizations({
      entry,
      actionIds: entry.canonicalizes.action_ids,
      repository: actionRepository,
      retiredActions,
      result,
      nowMs,
      turnCounter: input.turnCounter ?? null,
      tracer: input.tracer,
      turnId: input.turnId,
    });

    reconcileOpenQuestionCanonicalizations({
      entry,
      openQuestionIds: entry.canonicalizes.open_question_ids,
      repository: openQuestionsRepository,
      retiredOpenQuestions,
      result,
      tracer: input.tracer,
      turnId: input.turnId,
    });
  }

  return result;
}
