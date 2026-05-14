import type { ActionRepository } from "../../memory/actions/index.js";
import type { CommitmentRepository } from "../../memory/commitments/index.js";
import type { DecisionArtifactEntry } from "../../memory/decision-artifacts/index.js";
import type { GoalsRepository, OpenQuestionsRepository } from "../../memory/self/index.js";
import type { ActionId, CommitmentId, GoalId, OpenQuestionId } from "../../util/ids.js";
import type { Provenance } from "../../memory/common/provenance.js";
import type { DroppedCanonicalizeId } from "./compiler.js";

const RECONCILIATION_PROVENANCE = {
  kind: "online",
  process: "decision_artifact_reconciliation",
} as const satisfies Provenance;

export type DecisionArtifactReconciliationRepositories = {
  goalsRepository?: Pick<GoalsRepository, "updateStatus">;
  commitmentRepository?: Pick<CommitmentRepository, "revoke">;
  actionRepository?: Pick<ActionRepository, "update">;
  openQuestionsRepository?: Pick<OpenQuestionsRepository, "resolve">;
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

function activeLockedEntries(
  entries: readonly DecisionArtifactEntry[],
): readonly DecisionArtifactEntry[] {
  return entries.filter(
    (entry) =>
      entry.kind === "locked" && entry.superseded_by_id === null && hasCanonicalizedIds(entry),
  );
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
