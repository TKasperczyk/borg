import type { GoalRecord } from "../../memory/self/index.js";
import type { CognitiveMode, AttentionWeights } from "../types.js";

export type AttentionState = {
  currentGoals: readonly GoalRecord[];
  hasTemporalCue: boolean;
};

const DEFAULT_WEIGHTS: AttentionWeights = {
  semantic: 0.7,
  goal_relevance: 0,
  mood: 0,
  time: 0,
  social: 0,
  heat: 0.1,
  suppression_penalty: 0.5,
};

export function computeWeights(mode: CognitiveMode, state: AttentionState): AttentionWeights {
  const hasGoals = state.currentGoals.length > 0;
  const hasTemporalCue = state.hasTemporalCue;

  if (mode === "problem_solving") {
    return {
      ...DEFAULT_WEIGHTS,
      semantic: 0.8,
      goal_relevance: hasGoals ? 0.1 : 0,
      time: hasTemporalCue ? 0.1 : 0,
      heat: 0.15,
    };
  }

  if (mode === "relational") {
    return {
      ...DEFAULT_WEIGHTS,
      semantic: 0.65,
      social: 0.2,
      goal_relevance: hasGoals ? 0.05 : 0,
      time: hasTemporalCue ? 0.05 : 0,
    };
  }

  if (mode === "reflective") {
    return {
      ...DEFAULT_WEIGHTS,
      semantic: 0.65,
      goal_relevance: hasGoals ? 0.2 : 0,
      time: hasTemporalCue ? 0.1 : 0,
      heat: 0.05,
    };
  }

  return {
    ...DEFAULT_WEIGHTS,
    semantic: 0.55,
    goal_relevance: hasGoals ? 0.05 : 0,
    time: hasTemporalCue ? 0.05 : 0,
    heat: 0.05,
  };
}

export function computeRetrievalLimit(mode: CognitiveMode): number {
  if (mode === "idle") {
    return 1;
  }

  if (mode === "problem_solving") {
    return 6;
  }

  if (mode === "reflective") {
    return 5;
  }

  return 4;
}
