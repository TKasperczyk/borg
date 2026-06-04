import type { GoalRecord, GoalTreeNode, GoalsRepository } from "../../memory/self/index.js";

type GoalTreeNodeLike = GoalRecord & {
  children?: readonly GoalTreeNodeLike[];
};

export function flattenGoalTree(goals: ReadonlyArray<GoalTreeNodeLike>): GoalRecord[] {
  const flattened: GoalRecord[] = [];
  const stack = [...goals];

  while (stack.length > 0) {
    const next = stack.shift();

    if (next === undefined) {
      continue;
    }

    flattened.push(next);

    if (Array.isArray(next.children)) {
      stack.push(...next.children);
    }
  }

  return flattened;
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
