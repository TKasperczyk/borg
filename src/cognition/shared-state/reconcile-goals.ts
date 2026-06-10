import type { GoalId } from "../../util/ids.js";
import type { SharedStateEntry } from "../../memory/shared-state/index.js";
import type { GoalsRepository } from "../../memory/self/index.js";
import {
  canonicalizeGoalWithSharedStateEntry,
  type LifecycleTracer,
} from "../../memory/lifecycle-ops/index.js";
import { errorMessage, type SharedStateReconciliationResult } from "./reconciliation-summary.js";

export { isTerminalGoalStatus } from "../../memory/lifecycle-ops/index.js";

export function reconcileGoalCanonicalizations(input: {
  entry: SharedStateEntry;
  goalIds: readonly GoalId[];
  repository: ReconcileGoalsRepository | undefined;
  retiredGoals: Set<GoalId>;
  result: SharedStateReconciliationResult;
  tracer?: LifecycleTracer;
  turnId?: string;
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
      const result = canonicalizeGoalWithSharedStateEntry({
        goalId,
        entry: input.entry,
        repository: input.repository,
        tracer: input.tracer,
        turnId: input.turnId,
      });

      if (result.status === "no_op" && result.reason === "missing") {
        input.result.goals_canonicalized_skipped += 1;
        input.result.errors.push({
          channel: "goal",
          id: goalId,
          artifactEntryId: input.entry.id,
          message: `Unknown goal id: ${goalId}`,
        });
        continue;
      }

      if (result.status === "no_op") {
        input.result.goals_canonicalized_skipped += 1;
        continue;
      }

      if (result.status === "conflict") {
        input.result.goals_canonicalized_skipped += 1;
        input.result.errors.push({
          channel: "goal",
          id: goalId,
          artifactEntryId: input.entry.id,
          message: errorMessage(result.error),
        });
        continue;
      }

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
