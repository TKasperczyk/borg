import type { GoalRecord } from "./types.js";

export const DEFAULT_GOAL_MAX_INACTIVITY_MS = 2_592_000_000;

/** Factual activity, independent of presentation and paused scheduling clocks. */
export function goalActivityAnchor(goal: GoalRecord): number {
  return Math.max(
    goal.created_at,
    goal.last_progress_ts ?? goal.created_at,
    goal.last_interaction_at ?? goal.created_at,
  );
}

export function isGoalInactive(
  goal: GoalRecord,
  now: number,
  maxInactivityMs = DEFAULT_GOAL_MAX_INACTIVITY_MS,
): boolean {
  return now - goalActivityAnchor(goal) > maxInactivityMs;
}
