import type { GoalRecord } from "../../memory/self/index.js";
import type { CognitiveMode, AttentionWeights } from "../types.js";

export type AttentionState = {
  currentGoals: readonly GoalRecord[];
  hasActiveValues: boolean;
  hasTemporalCue: boolean;
  moodActive?: boolean;
  audienceTrust?: number | null;
};

const DEFAULT_WEIGHTS: AttentionWeights = {
  semantic: 0.7,
  goal_relevance: 0,
  value_alignment: 0,
  mood: 0,
  time: 0,
  social: 0,
  entity: 0.2,
  heat: 0.1,
  suppression_penalty: 0.5,
};

export function computeWeights(mode: CognitiveMode, state: AttentionState): AttentionWeights {
  const hasGoals = state.currentGoals.length > 0;
  const valueAlignmentWeight = state.hasActiveValues ? 0.15 : 0;
  const hasTemporalCue = state.hasTemporalCue;
  const moodWeight = state.moodActive === true ? 0.2 : 0;
  const socialBase = state.audienceTrust !== undefined && state.audienceTrust !== null ? 0.1 : 0;

  if (mode === "problem_solving") {
    return {
      ...DEFAULT_WEIGHTS,
      semantic: 0.8,
      goal_relevance: hasGoals ? 0.1 : 0,
      value_alignment: state.hasActiveValues ? 0.1 : 0,
      mood: moodWeight,
      time: hasTemporalCue ? 0.1 : 0,
      entity: 0.25,
      heat: 0.15,
    };
  }

  if (mode === "relational") {
    return {
      ...DEFAULT_WEIGHTS,
      semantic: 0.65,
      mood: moodWeight,
      social: Math.max(0.2, socialBase),
      goal_relevance: hasGoals ? 0.05 : 0,
      value_alignment: state.hasActiveValues ? 0.2 : 0,
      time: hasTemporalCue ? 0.05 : 0,
      entity: 0.25,
    };
  }

  if (mode === "reflective") {
    return {
      ...DEFAULT_WEIGHTS,
      semantic: 0.65,
      goal_relevance: hasGoals ? 0.2 : 0,
      value_alignment: state.hasActiveValues ? 0.2 : 0,
      mood: moodWeight,
      time: hasTemporalCue ? 0.1 : 0,
      entity: 0.2,
      heat: 0.05,
    };
  }

  return {
    ...DEFAULT_WEIGHTS,
    semantic: 0.55,
    goal_relevance: hasGoals ? 0.05 : 0,
    value_alignment: valueAlignmentWeight,
    mood: moodWeight,
    social: socialBase,
    time: hasTemporalCue ? 0.05 : 0,
    entity: 0.1,
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
