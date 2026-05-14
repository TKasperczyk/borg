import type { ActionRepository } from "../../memory/actions/index.js";
import type { CommitmentRepository } from "../../memory/commitments/index.js";
import type {
  DecisionArtifact,
  DecisionArtifactCanonicalizes,
  DecisionArtifactEntry,
} from "../../memory/decision-artifacts/index.js";
import type { GoalsRepository, OpenQuestionsRepository } from "../../memory/self/index.js";
import type { ActionId, CommitmentId, GoalId, OpenQuestionId } from "../../util/ids.js";
import type { Provenance } from "../../memory/common/provenance.js";
import type { DroppedCanonicalizeId } from "./compiler.js";

const RECONCILIATION_PROVENANCE = {
  kind: "online",
  process: "decision_artifact_reconciliation",
} as const satisfies Provenance;

export type DecisionArtifactReconciliationRepositories = {
  goalsRepository?: Pick<GoalsRepository, "updateStatus"> & Partial<Pick<GoalsRepository, "get">>;
  commitmentRepository?: Pick<CommitmentRepository, "revoke"> &
    Partial<Pick<CommitmentRepository, "get">>;
  actionRepository?: Pick<ActionRepository, "update"> & Partial<Pick<ActionRepository, "get">>;
  openQuestionsRepository?: Pick<OpenQuestionsRepository, "resolve"> &
    Partial<Pick<OpenQuestionsRepository, "get">>;
};

export type DecisionArtifactReconciliationLookupRepositories = {
  goalsRepository?: Partial<Pick<GoalsRepository, "get">>;
  commitmentRepository?: Partial<Pick<CommitmentRepository, "get">>;
  actionRepository?: Partial<Pick<ActionRepository, "get">>;
  openQuestionsRepository?: Partial<Pick<OpenQuestionsRepository, "get">>;
};

export type DecisionArtifactUnsettledReconciliationSummary = {
  active_locked_canonicalizing_entry_count: number;
  referenced_goal_count: number;
  referenced_commitment_count: number;
  referenced_action_count: number;
  referenced_open_question_count: number;
  unsettled_goal_count: number;
  unsettled_commitment_count: number;
  unsettled_action_count: number;
  unsettled_open_question_count: number;
  unsettled_total_count: number;
};

export type DecisionArtifactUnsettledReconciliation = {
  summary: DecisionArtifactUnsettledReconciliationSummary;
  entries: DecisionArtifactEntry[];
};

export type DecisionArtifactReconciliationError = {
  channel: "goal" | "commitment" | "action" | "open_question";
  id: string;
  artifactEntryId: string;
  message: string;
};

export type DecisionArtifactReconciliationResult = {
  goals_retired: number;
  commitments_retired: number;
  actions_retired: number;
  open_questions_retired: number;
  unknown_ids: readonly DroppedCanonicalizeId[];
  errors: DecisionArtifactReconciliationError[];
};

export type ReconcileDecisionArtifactCanonicalizationsInput = {
  entries: readonly DecisionArtifactEntry[];
  repositories?: DecisionArtifactReconciliationRepositories;
  unknownIds?: readonly DroppedCanonicalizeId[];
};

function errorMessage(error: unknown): string {
  return error instanceof Error ? error.message : String(error);
}

function hasCanonicalizedIds(entry: DecisionArtifactEntry): boolean {
  return (
    entry.canonicalizes.goal_ids.length > 0 ||
    entry.canonicalizes.commitment_ids.length > 0 ||
    entry.canonicalizes.action_ids.length > 0 ||
    entry.canonicalizes.open_question_ids.length > 0
  );
}

function emptyCanonicalizes(): DecisionArtifactCanonicalizes {
  return {
    goal_ids: [],
    commitment_ids: [],
    action_ids: [],
    open_question_ids: [],
  };
}

function activeLockedEntries(
  entries: readonly DecisionArtifactEntry[],
): readonly DecisionArtifactEntry[] {
  return entries.filter(
    (entry) =>
      entry.kind === "locked" && entry.superseded_by_id === null && hasCanonicalizedIds(entry),
  );
}

function isTerminalGoalStatus(status: string): boolean {
  return status === "done" || status === "abandoned" || status === "superseded";
}

function isTerminalCommitment(
  commitment: NonNullable<ReturnType<CommitmentRepository["get"]>>,
): boolean {
  return commitment.revoked_at !== null || commitment.superseded_by !== null;
}

function isTerminalActionState(state: string): boolean {
  return state === "completed" || state === "not_done" || state === "superseded";
}

function isTerminalOpenQuestionStatus(status: string): boolean {
  return status === "resolved" || status === "abandoned";
}

function retryEntry(
  entriesById: Map<DecisionArtifactEntry["id"], DecisionArtifactEntry>,
  entry: DecisionArtifactEntry,
): DecisionArtifactEntry {
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

export function findUnsettledDecisionArtifactReconciliation(input: {
  previousArtifact: DecisionArtifact | null | undefined;
  repositories?: DecisionArtifactReconciliationLookupRepositories;
}): DecisionArtifactUnsettledReconciliation | null {
  const entries = activeLockedEntries(input.previousArtifact?.entries ?? []);

  if (entries.length === 0) {
    return null;
  }

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
  const retryEntriesById = new Map<DecisionArtifactEntry["id"], DecisionArtifactEntry>();
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

      if (commitment !== null && !isTerminalCommitment(commitment)) {
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

export function reconcileDecisionArtifactCanonicalizations(
  input: ReconcileDecisionArtifactCanonicalizationsInput,
): DecisionArtifactReconciliationResult {
  const result: DecisionArtifactReconciliationResult = {
    goals_retired: 0,
    commitments_retired: 0,
    actions_retired: 0,
    open_questions_retired: 0,
    unknown_ids: input.unknownIds ?? [],
    errors: [],
  };
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
    for (const goalId of entry.canonicalizes.goal_ids) {
      if (retiredGoals.has(goalId)) {
        continue;
      }

      if (goalsRepository === undefined) {
        continue;
      }

      try {
        goalsRepository.updateStatus(goalId, "done", RECONCILIATION_PROVENANCE, {
          canonicalizedByArtifactEntryId: entry.id,
        });
        retiredGoals.add(goalId);
        result.goals_retired += 1;
      } catch (error) {
        result.errors.push({
          channel: "goal",
          id: goalId,
          artifactEntryId: entry.id,
          message: errorMessage(error),
        });
      }
    }

    for (const commitmentId of entry.canonicalizes.commitment_ids) {
      if (retiredCommitments.has(commitmentId)) {
        continue;
      }

      if (commitmentRepository === undefined) {
        continue;
      }

      try {
        const retired = commitmentRepository.revoke(
          commitmentId,
          `canonicalized_by_artifact_entry_id=${entry.id}`,
          RECONCILIATION_PROVENANCE,
          undefined,
          {
            canonicalizedByArtifactEntryId: entry.id,
          },
        );

        if (retired === null) {
          result.errors.push({
            channel: "commitment",
            id: commitmentId,
            artifactEntryId: entry.id,
            message: `Unknown commitment id: ${commitmentId}`,
          });
          continue;
        }

        retiredCommitments.add(commitmentId);
        result.commitments_retired += 1;
      } catch (error) {
        result.errors.push({
          channel: "commitment",
          id: commitmentId,
          artifactEntryId: entry.id,
          message: errorMessage(error),
        });
      }
    }

    for (const actionId of entry.canonicalizes.action_ids) {
      if (retiredActions.has(actionId)) {
        continue;
      }

      if (actionRepository === undefined) {
        continue;
      }

      try {
        actionRepository.update(
          actionId,
          {
            state: "completed",
            canonicalized_by_artifact_entry_id: entry.id,
          },
          {
            skipSideEffects: true,
          },
        );
        retiredActions.add(actionId);
        result.actions_retired += 1;
      } catch (error) {
        result.errors.push({
          channel: "action",
          id: actionId,
          artifactEntryId: entry.id,
          message: errorMessage(error),
        });
      }
    }

    for (const openQuestionId of entry.canonicalizes.open_question_ids) {
      if (retiredOpenQuestions.has(openQuestionId)) {
        continue;
      }

      if (openQuestionsRepository === undefined) {
        continue;
      }

      try {
        openQuestionsRepository.resolve(
          openQuestionId,
          {
            resolution_evidence_stream_entry_ids: entry.last_updated_stream_entry_ids,
            resolution_note: `resolved_by_artifact_entry_id=${entry.id}`,
          },
          {
            resolvedByArtifactEntryId: entry.id,
          },
        );
        retiredOpenQuestions.add(openQuestionId);
        result.open_questions_retired += 1;
      } catch (error) {
        result.errors.push({
          channel: "open_question",
          id: openQuestionId,
          artifactEntryId: entry.id,
          message: errorMessage(error),
        });
      }
    }
  }

  return result;
}
