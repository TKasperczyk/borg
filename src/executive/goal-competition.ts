import type { GoalRecord } from "../memory/self/index.js";
import { goalSchedulingTimes } from "../memory/self/goal-blocks.js";

import type {
  ExecutiveFocus,
  ExecutiveGoalScore,
  ExecutiveGoalScoreComponents,
  ExecutiveGoalScoreContext,
} from "./types.js";
import type { ExecutiveContextFitByGoalId } from "./context-fit.js";

export const DEFAULT_EXECUTIVE_GOAL_FOCUS_THRESHOLD = 0.45;

const PRIORITY_WEIGHT = 0.35;
const DEADLINE_PRESSURE_WEIGHT = 0.3;
const CONTEXT_FIT_WEIGHT = 0.2;
const PROGRESS_DEBT_WEIGHT = 0.15;

export type SelectExecutiveFocusInput = {
  goals: readonly GoalRecord[];
  cognitionInput: string;
  perceptionEntities?: readonly string[];
  autonomyPayload?: Record<string, unknown> | null;
  nowMs: number;
  threshold?: number;
  deadlineLookaheadMs: number;
  staleMs: number;
  scoreContext?: ExecutiveGoalScoreContext;
  contextFitByGoalId?: ExecutiveContextFitByGoalId;
};

function clamp01(value: number): number {
  return Math.min(1, Math.max(0, value));
}

function normalizePriority(goal: GoalRecord, maxPriority: number): number {
  if (!Number.isFinite(goal.priority) || goal.priority <= 0 || maxPriority <= 0) {
    return 0;
  }

  return clamp01(goal.priority / maxPriority);
}

function computeDeadlinePressure(goal: GoalRecord, nowMs: number, lookaheadMs: number): number {
  const targetAt = goalSchedulingTimes(goal).targetAt;
  if (targetAt === null || !Number.isFinite(targetAt) || lookaheadMs <= 0) {
    return 0;
  }

  const remainingMs = targetAt - nowMs;

  if (remainingMs <= 0) {
    return 1;
  }

  if (remainingMs >= lookaheadMs) {
    return 0;
  }

  return clamp01(1 - remainingMs / lookaheadMs);
}

// staleMs is supplied per scoring context, not by this module. Turn selection and
// both wake-time selectors all receive autonomy.triggers.goalFollowupDue.staleMs,
// so the progress-debt divisor agrees everywhere. The executive-focus trigger's
// shorter stalenessSec remains a due-cadence threshold only. The divisor is
// rendered for the model in score_basis; keep that surface in step if it moves.
function computeProgressDebt(goal: GoalRecord, nowMs: number, staleMs: number): number {
  if (staleMs <= 0) {
    return 0;
  }

  const progressAnchor = goalSchedulingTimes(goal).progressAnchor;

  if (!Number.isFinite(progressAnchor)) {
    return 0;
  }

  return clamp01((nowMs - progressAnchor) / staleMs);
}

function computeScore(components: ExecutiveGoalScoreComponents): number {
  return clamp01(
    PRIORITY_WEIGHT * components.priority +
      DEADLINE_PRESSURE_WEIGHT * components.deadline_pressure +
      CONTEXT_FIT_WEIGHT * components.context_fit +
      PROGRESS_DEBT_WEIGHT * components.progress_debt,
  );
}

function summarizeScore(components: ExecutiveGoalScoreComponents): string {
  return [
    `priority=${components.priority.toFixed(2)}`,
    `deadline=${components.deadline_pressure.toFixed(2)}`,
    `context=${components.context_fit.toFixed(2)}`,
    `progress_debt=${components.progress_debt.toFixed(2)}`,
  ].join(", ");
}

function compareGoalScores(left: ExecutiveGoalScore, right: ExecutiveGoalScore): number {
  return (
    right.score - left.score ||
    (left.goal.target_at ?? Number.POSITIVE_INFINITY) -
      (right.goal.target_at ?? Number.POSITIVE_INFINITY) ||
    right.goal.priority - left.goal.priority ||
    left.goal.created_at - right.goal.created_at ||
    left.goal.id.localeCompare(right.goal.id)
  );
}

export function topExecutiveCandidateGoalIds(input: {
  candidates: readonly ExecutiveGoalScore[];
  eligibleGoalIds: ReadonlySet<GoalRecord["id"]>;
  limit: number;
}): GoalRecord["id"][] {
  const limit = Number.isFinite(input.limit) ? Math.max(0, Math.floor(input.limit)) : 0;
  const seenGoalIds = new Set<GoalRecord["id"]>();

  return [...input.candidates]
    .sort((left, right) => right.score - left.score || left.goal_id.localeCompare(right.goal_id))
    .map((candidate) => candidate.goal_id)
    .filter((goalId) => {
      if (seenGoalIds.has(goalId) || !input.eligibleGoalIds.has(goalId)) {
        return false;
      }
      seenGoalIds.add(goalId);
      return true;
    })
    .slice(0, limit);
}

export function selectExecutiveFocus(input: SelectExecutiveFocusInput): ExecutiveFocus {
  const threshold = input.threshold ?? DEFAULT_EXECUTIVE_GOAL_FOCUS_THRESHOLD;
  const scoreBasis = {
    score_context: input.scoreContext ?? "turn_selection",
    deadline_lookahead_ms: input.deadlineLookaheadMs,
    progress_debt_stale_ms: input.staleMs,
  } as const;
  const activeGoals = input.goals.filter((goal) => goal.status === "active");

  if (activeGoals.length === 0) {
    return {
      selected_goal: null,
      selected_score: null,
      candidates: [],
      threshold,
      score_basis: scoreBasis,
    };
  }

  const maxPriority = Math.max(1, ...activeGoals.map((goal) => Math.max(0, goal.priority)));
  const candidates = activeGoals
    .map((goal): ExecutiveGoalScore => {
      const components = {
        priority: normalizePriority(goal, maxPriority),
        deadline_pressure: computeDeadlinePressure(goal, input.nowMs, input.deadlineLookaheadMs),
        context_fit: input.contextFitByGoalId?.get(goal.id) ?? 0,
        progress_debt: computeProgressDebt(goal, input.nowMs, input.staleMs),
      };
      const score = computeScore(components);

      return {
        goal_id: goal.id,
        goal,
        score,
        components,
        reason: summarizeScore(components),
      };
    })
    .sort(compareGoalScores);
  const selected = candidates[0] ?? null;

  if (selected === null || selected.score < threshold) {
    return {
      selected_goal: null,
      selected_score: null,
      candidates,
      threshold,
      score_basis: scoreBasis,
    };
  }

  return {
    selected_goal: selected.goal,
    selected_score: selected,
    candidates,
    threshold,
    score_basis: scoreBasis,
  };
}
