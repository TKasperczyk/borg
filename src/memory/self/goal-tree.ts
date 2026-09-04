import type { GoalRecord } from "./types.js";

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
