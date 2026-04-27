import type { GoalRecord } from "../memory/self/index.js";

export type ExecutiveGoalScoreComponents = {
  priority: number;
  deadline_pressure: number;
  context_fit: number;
  progress_debt: number;
};

export type ExecutiveGoalScore = {
  goal_id: GoalRecord["id"];
  goal: GoalRecord;
  score: number;
  components: ExecutiveGoalScoreComponents;
  reason: string;
};

export type ExecutiveFocus = {
  selected_goal: GoalRecord | null;
  selected_score: ExecutiveGoalScore | null;
  candidates: ExecutiveGoalScore[];
  threshold: number;
};
