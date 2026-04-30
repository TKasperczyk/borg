import type { GoalRecord, GoalTreeNode, GoalsRepository } from "../../memory/self/index.js";
import type { StreamWatermarkRepository } from "../../stream/index.js";
import { SystemClock, type Clock } from "../../util/clock.js";
import { DEFAULT_SESSION_ID, type SessionId } from "../../util/ids.js";
import type { AutonomyTrigger, DueEvent } from "../types.js";

const TRIGGER_NAME = "goal_followup_due" as const;
const WATERMARK_PREFIX = "autonomy:goal-followup-due";
const DAY_MS = 24 * 60 * 60 * 1_000;

export type GoalFollowupDuePayload = {
  goal_id: GoalRecord["id"];
  description: string;
  priority: number;
  target_at: number | null;
  last_progress_ts: number | null;
  days_stale: number;
  reason: "deadline" | "stale" | "both";
};

export type GoalFollowupDueTriggerOptions = {
  goalsRepository: GoalsRepository;
  watermarkRepository: StreamWatermarkRepository;
  lookaheadMs: number;
  staleMs: number;
  clock?: Clock;
  sessionId?: SessionId;
};

function flattenGoals(goals: readonly GoalTreeNode[]): GoalRecord[] {
  const flattened: GoalRecord[] = [];
  const stack = [...goals];

  while (stack.length > 0) {
    const next = stack.shift();

    if (next === undefined) {
      continue;
    }

    flattened.push(next);
    stack.push(...next.children);
  }

  return flattened;
}

export function createGoalFollowupDueTrigger(
  options: GoalFollowupDueTriggerOptions,
): AutonomyTrigger<GoalFollowupDuePayload> {
  const clock = options.clock ?? new SystemClock();
  const sessionId = options.sessionId ?? DEFAULT_SESSION_ID;

  return {
    name: TRIGGER_NAME,
    type: "trigger",
    async scan() {
      const nowMs = clock.now();
      const goals = flattenGoals(options.goalsRepository.list({ status: "active" }));
      const dueEvents = goals
        .map<DueEvent<GoalFollowupDuePayload> | null>((goal) => {
          const baseProgressTs = goal.last_progress_ts ?? goal.created_at;
          const deadlineDue =
            goal.target_at !== null && goal.target_at - nowMs < options.lookaheadMs;
          const staleDue = baseProgressTs + options.staleMs < nowMs;

          if (!deadlineDue && !staleDue) {
            return null;
          }

          const targetAtKey = goal.target_at ?? "no-target";
          const progressKey = goal.last_progress_ts ?? goal.created_at;
          const watermarkProcessName = `${WATERMARK_PREFIX}:${goal.id}:${targetAtKey}:${progressKey}`;

          if (options.watermarkRepository.get(watermarkProcessName, sessionId) !== null) {
            return null;
          }

          const reason = deadlineDue && staleDue ? "both" : deadlineDue ? "deadline" : "stale";
          const staleSinceMs = Math.max(0, nowMs - baseProgressTs);
          const sortTs =
            goal.target_at === null
              ? baseProgressTs + options.staleMs
              : Math.min(goal.target_at, baseProgressTs + options.staleMs);

          return {
            id: `${goal.id}:${targetAtKey}:${progressKey}`,
            sourceName: TRIGGER_NAME,
            sourceType: "trigger",
            watermarkProcessName,
            sortTs,
            payload: {
              goal_id: goal.id,
              description: goal.description,
              priority: goal.priority,
              target_at: goal.target_at,
              last_progress_ts: goal.last_progress_ts,
              days_stale: Math.floor(staleSinceMs / DAY_MS),
              reason,
            },
          };
        })
        .filter((event): event is DueEvent<GoalFollowupDuePayload> => event !== null);

      return dueEvents.sort(
        (left, right) =>
          left.sortTs - right.sortTs || right.payload.priority - left.payload.priority,
      );
    },
    buildTurn(event) {
      return {
        audience: "self",
        stakes: "low",
        userMessage: "",
        autonomyTrigger: {
          source_name: event.sourceName,
          source_type: event.sourceType,
          event_id: event.id,
          sort_ts: event.sortTs,
          payload: event.payload,
        },
      };
    },
  };
}
