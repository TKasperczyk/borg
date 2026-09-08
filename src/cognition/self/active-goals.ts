import type { GoalRecord, GoalTreeNode, GoalsRepository } from "../../memory/self/index.js";
import { flattenGoalTree } from "../../memory/self/goal-tree.js";

export { flattenGoalTree } from "../../memory/self/goal-tree.js";

export function listUnfinishedGoalsForCognition(
  goalsRepository: Pick<GoalsRepository, "list">,
): GoalRecord[] {
  return flattenGoalTree(goalsRepository.list({ statuses: ["active", "blocked"] }));
}

export function listActiveGoalsForCognition(
  goalsRepository: Pick<GoalsRepository, "list">,
): GoalRecord[] {
  return flattenGoalTree(
    goalsRepository.list({
      status: "active",
    }) as readonly GoalTreeNode[],
  );
}
