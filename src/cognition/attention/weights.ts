import type { GoalRecord } from "../../memory/self/index.js";
import type { CognitiveMode, AttentionWeights } from "../types.js";

export type AttentionState = {
  currentGoals: readonly GoalRecord[];
  hasActiveValues: boolean;
  hasTemporalCue: boolean;
  moodActive?: boolean;
  audienceTrust?: number | null;
};

// Tunes the fallback retrieval blend before mode-specific overrides are applied.
const DEFAULT_ATTENTION_WEIGHTS: AttentionWeights = {
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

// Tunes value alignment when active values are present in idle/default retrieval.
const DEFAULT_ACTIVE_VALUE_ALIGNMENT_WEIGHT = 0.15;

// Tunes mood-congruent ranking when the mood signal is active.
const ACTIVE_MOOD_WEIGHT = 0.2;

// Tunes the baseline social ranking weight when audience trust is known.
const KNOWN_AUDIENCE_SOCIAL_BASE_WEIGHT = 0.1;

// Tunes the problem-solving mode's static ranking preset.
const PROBLEM_SOLVING_ATTENTION_PRESET = {
  semantic: 0.8,
  entity: 0.25,
  heat: 0.15,
} as const satisfies Pick<AttentionWeights, "semantic" | "entity" | "heat">;

// Tunes dynamic problem-solving ranking terms when their signals are present.
const PROBLEM_SOLVING_CONTEXT_WEIGHT = 0.1;

// Tunes the relational mode's static ranking preset.
const RELATIONAL_ATTENTION_PRESET = {
  semantic: 0.65,
  entity: 0.25,
} as const satisfies Pick<AttentionWeights, "semantic" | "entity">;

// Tunes the minimum social weight in relational mode.
const RELATIONAL_SOCIAL_MIN_WEIGHT = 0.2;

// Tunes relational goal relevance when active goals are present.
const RELATIONAL_GOAL_RELEVANCE_WEIGHT = 0.05;

// Tunes relational value alignment when active values are present.
const RELATIONAL_VALUE_ALIGNMENT_WEIGHT = 0.2;

// Tunes relational time relevance when a temporal cue is present.
const RELATIONAL_TIME_WEIGHT = 0.05;

// Tunes the reflective mode's static ranking preset.
const REFLECTIVE_ATTENTION_PRESET = {
  semantic: 0.65,
  entity: 0.2,
  heat: 0.05,
} as const satisfies Pick<AttentionWeights, "semantic" | "entity" | "heat">;

// Tunes reflective goal relevance when active goals are present.
const REFLECTIVE_GOAL_RELEVANCE_WEIGHT = 0.2;

// Tunes reflective value alignment when active values are present.
const REFLECTIVE_VALUE_ALIGNMENT_WEIGHT = 0.2;

// Tunes reflective time relevance when a temporal cue is present.
const REFLECTIVE_TIME_WEIGHT = 0.1;

// Tunes the idle/default mode's static ranking preset.
const IDLE_ATTENTION_PRESET = {
  semantic: 0.55,
  entity: 0.1,
  heat: 0.05,
} as const satisfies Pick<AttentionWeights, "semantic" | "entity" | "heat">;

// Tunes idle/default goal relevance when active goals are present.
const IDLE_GOAL_RELEVANCE_WEIGHT = 0.05;

// Tunes idle/default time relevance when a temporal cue is present.
const IDLE_TIME_WEIGHT = 0.05;

export function computeWeights(mode: CognitiveMode, state: AttentionState): AttentionWeights {
  const hasGoals = state.currentGoals.length > 0;
  const valueAlignmentWeight = state.hasActiveValues ? DEFAULT_ACTIVE_VALUE_ALIGNMENT_WEIGHT : 0;
  const hasTemporalCue = state.hasTemporalCue;
  const moodWeight = state.moodActive === true ? ACTIVE_MOOD_WEIGHT : 0;
  const socialBase =
    state.audienceTrust !== undefined && state.audienceTrust !== null
      ? KNOWN_AUDIENCE_SOCIAL_BASE_WEIGHT
      : 0;

  if (mode === "problem_solving") {
    return {
      ...DEFAULT_ATTENTION_WEIGHTS,
      ...PROBLEM_SOLVING_ATTENTION_PRESET,
      goal_relevance: hasGoals ? PROBLEM_SOLVING_CONTEXT_WEIGHT : 0,
      value_alignment: state.hasActiveValues ? PROBLEM_SOLVING_CONTEXT_WEIGHT : 0,
      mood: moodWeight,
      time: hasTemporalCue ? PROBLEM_SOLVING_CONTEXT_WEIGHT : 0,
    };
  }

  if (mode === "relational") {
    return {
      ...DEFAULT_ATTENTION_WEIGHTS,
      ...RELATIONAL_ATTENTION_PRESET,
      mood: moodWeight,
      social: Math.max(RELATIONAL_SOCIAL_MIN_WEIGHT, socialBase),
      goal_relevance: hasGoals ? RELATIONAL_GOAL_RELEVANCE_WEIGHT : 0,
      value_alignment: state.hasActiveValues ? RELATIONAL_VALUE_ALIGNMENT_WEIGHT : 0,
      time: hasTemporalCue ? RELATIONAL_TIME_WEIGHT : 0,
    };
  }

  if (mode === "reflective") {
    return {
      ...DEFAULT_ATTENTION_WEIGHTS,
      ...REFLECTIVE_ATTENTION_PRESET,
      goal_relevance: hasGoals ? REFLECTIVE_GOAL_RELEVANCE_WEIGHT : 0,
      value_alignment: state.hasActiveValues ? REFLECTIVE_VALUE_ALIGNMENT_WEIGHT : 0,
      mood: moodWeight,
      time: hasTemporalCue ? REFLECTIVE_TIME_WEIGHT : 0,
    };
  }

  return {
    ...DEFAULT_ATTENTION_WEIGHTS,
    ...IDLE_ATTENTION_PRESET,
    goal_relevance: hasGoals ? IDLE_GOAL_RELEVANCE_WEIGHT : 0,
    value_alignment: valueAlignmentWeight,
    mood: moodWeight,
    social: socialBase,
    time: hasTemporalCue ? IDLE_TIME_WEIGHT : 0,
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
