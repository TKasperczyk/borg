import type { GoalId } from "../../util/ids.js";
import type { SharedStateEntry } from "../../memory/decision-artifacts/index.js";
import type { GoalsRepository } from "../../memory/self/index.js";
import {
  RECONCILIATION_PROVENANCE,
  errorMessage,
  type SharedStateReconciliationResult,
} from "./reconciliation-summary.js";

export function isTerminalGoalStatus(status: string): boolean {
  return status === "done" || status === "abandoned" || status === "superseded";
}

export function reconcileGoalCanonicalizations(input: {
  entry: SharedStateEntry;
  goalIds: readonly GoalId[];
  repository: ReconcileGoalsRepository | undefined;
  retiredGoals: Set<GoalId>;
  result: SharedStateReconciliationResult;
}): void {
  for (const goalId of input.goalIds) {
    input.result.goals_canonicalized_attempted += 1;

    if (input.retiredGoals.has(goalId)) {
      input.result.goals_canonicalized_skipped += 1;
      continue;
    }

    if (input.repository === undefined) {
      input.result.goals_canonicalized_skipped += 1;
      continue;
    }

    try {
      const goal = input.repository.get?.(goalId) ?? null;

      if (goal !== null && isTerminalGoalStatus(goal.status)) {
        input.result.goals_canonicalized_skipped += 1;
        continue;
      }

      input.repository.updateStatus(goalId, "done", RECONCILIATION_PROVENANCE, {
        canonicalizedByArtifactEntryId: input.entry.id,
      });
      input.retiredGoals.add(goalId);
      input.result.goals_retired += 1;
      input.result.goals_canonicalized_succeeded += 1;
    } catch (error) {
      input.result.errors.push({
        channel: "goal",
        id: goalId,
        artifactEntryId: input.entry.id,
        message: errorMessage(error),
      });
    }
  }
}

type ReconcileGoalsRepository = Pick<GoalsRepository, "updateStatus"> &
  Partial<Pick<GoalsRepository, "get">>;
