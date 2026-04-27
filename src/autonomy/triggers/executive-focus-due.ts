import {
  DEFAULT_EXECUTIVE_GOAL_FOCUS_THRESHOLD,
  selectExecutiveFocus,
} from "../../executive/index.js";
import type { ExecutiveStep, ExecutiveStepsRepository } from "../../executive/index.js";
import {
  isEpisodeVisibleToAudience,
  type EpisodicRepository,
} from "../../memory/episodic/index.js";
import type { GoalRecord, GoalTreeNode, GoalsRepository } from "../../memory/self/index.js";
import type { StreamWatermarkRepository } from "../../stream/index.js";
import { SystemClock, type Clock } from "../../util/clock.js";
import { DEFAULT_SESSION_ID, type EpisodeId, type SessionId } from "../../util/ids.js";
import { AUTONOMOUS_WAKE_USER_MESSAGE } from "../../cognition/autonomy-trigger.js";
import type { AutonomyTrigger, DueEvent } from "../types.js";

const TRIGGER_NAME = "executive_focus_due" as const;
const WATERMARK_PREFIX = "autonomy:executive-focus-due";

type ExecutiveFocusDueReason = "step_due" | "goal_stale";

type ExecutiveFocusDueStepPayload = Pick<
  ExecutiveStep,
  "id" | "goal_id" | "description" | "status" | "kind" | "due_at" | "last_attempt_ts"
>;

export type ExecutiveFocusDuePayload = {
  reason: ExecutiveFocusDueReason;
  selected_goal_id: GoalRecord["id"];
  selected_goal: {
    goal_id: GoalRecord["id"];
    description: string;
    priority: number;
    target_at: number | null;
    last_progress_ts: number | null;
  };
  selected_score: {
    score: number;
    components: {
      priority: number;
      deadline_pressure: number;
      context_fit: number;
      progress_debt: number;
    };
    reason: string;
    threshold: number;
  };
  top_open_step: ExecutiveFocusDueStepPayload | null;
  due_step?: ExecutiveFocusDueStepPayload;
};

export type ExecutiveFocusDueTriggerOptions = {
  enabled: boolean;
  goalsRepository: GoalsRepository;
  executiveStepsRepository: ExecutiveStepsRepository;
  episodicRepository: EpisodicRepository;
  watermarkRepository: StreamWatermarkRepository;
  threshold?: number;
  stalenessMs: number;
  dueLeadMs: number;
  deadlineLookaheadMs: number;
  goalFollowupDue?: {
    enabled: boolean;
    lookaheadMs: number;
    staleMs: number;
  };
  clock?: Clock;
  sessionId?: SessionId;
};

type ProvenanceScopedSelfRecord = {
  provenance?: {
    kind: string;
    episode_ids?: readonly EpisodeId[];
  } | null;
  evidence_episode_ids?: readonly EpisodeId[] | null;
  key_episode_ids?: readonly EpisodeId[] | null;
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

function getSelfRecordEvidenceEpisodeIds(record: ProvenanceScopedSelfRecord): EpisodeId[] {
  if (record.provenance?.kind !== "episodes") {
    return [];
  }

  const hasExplicitEvidence =
    record.evidence_episode_ids !== undefined || record.key_episode_ids !== undefined;
  const explicitEpisodeIds = [
    ...(record.evidence_episode_ids ?? []),
    ...(record.key_episode_ids ?? []),
  ];

  if (hasExplicitEvidence && explicitEpisodeIds.length === 0) {
    return [];
  }

  return [...new Set([...(record.provenance.episode_ids ?? []), ...explicitEpisodeIds])];
}

function isSelfRecordVisible(
  record: ProvenanceScopedSelfRecord,
  visibleEpisodeIds: ReadonlySet<EpisodeId>,
): boolean {
  const episodeIds = getSelfRecordEvidenceEpisodeIds(record);

  if (episodeIds.length === 0) {
    return true;
  }

  return episodeIds.some((episodeId) => visibleEpisodeIds.has(episodeId));
}

async function listSelfVisibleActiveGoals(options: {
  goalsRepository: GoalsRepository;
  episodicRepository: EpisodicRepository;
}): Promise<GoalRecord[]> {
  const goals = flattenGoals(options.goalsRepository.list({ status: "active" }));
  const evidenceEpisodeIds = [
    ...new Set(goals.flatMap((goal) => getSelfRecordEvidenceEpisodeIds(goal))),
  ];

  if (evidenceEpisodeIds.length === 0) {
    return goals;
  }

  const evidenceEpisodes = await options.episodicRepository.getMany(evidenceEpisodeIds);
  const visibleEpisodeIds = new Set(
    evidenceEpisodes
      .filter((episode) => isEpisodeVisibleToAudience(episode, null))
      .map((episode) => episode.id),
  );

  return goals.filter((goal) => isSelfRecordVisible(goal, visibleEpisodeIds));
}

function serializeStep(step: ExecutiveStep): ExecutiveFocusDueStepPayload {
  return {
    id: step.id,
    goal_id: step.goal_id,
    description: step.description,
    status: step.status,
    kind: step.kind,
    due_at: step.due_at,
    last_attempt_ts: step.last_attempt_ts,
  };
}

function buildScorePayload(input: {
  goal: GoalRecord;
  score: NonNullable<ReturnType<typeof selectExecutiveFocus>["selected_score"]>;
  threshold: number;
  topOpenStep: ExecutiveStep | null;
  reason: ExecutiveFocusDueReason;
  dueStep?: ExecutiveStep;
}): ExecutiveFocusDuePayload {
  return {
    reason: input.reason,
    selected_goal_id: input.goal.id,
    selected_goal: {
      goal_id: input.goal.id,
      description: input.goal.description,
      priority: input.goal.priority,
      target_at: input.goal.target_at,
      last_progress_ts: input.goal.last_progress_ts,
    },
    selected_score: {
      score: input.score.score,
      components: input.score.components,
      reason: input.score.reason,
      threshold: input.threshold,
    },
    top_open_step: input.topOpenStep === null ? null : serializeStep(input.topOpenStep),
    ...(input.dueStep === undefined ? {} : { due_step: serializeStep(input.dueStep) }),
  };
}

function isGoalFollowupDueMatch(input: {
  goal: GoalRecord;
  nowMs: number;
  lookaheadMs: number;
  staleMs: number;
}): boolean {
  const baseProgressTs = input.goal.last_progress_ts ?? input.goal.created_at;
  const deadlineDue =
    input.goal.target_at !== null && input.goal.target_at - input.nowMs < input.lookaheadMs;
  const staleDue = baseProgressTs + input.staleMs < input.nowMs;

  return deadlineDue || staleDue;
}

export function createExecutiveFocusDueTrigger(
  options: ExecutiveFocusDueTriggerOptions,
): AutonomyTrigger<ExecutiveFocusDuePayload> {
  const clock = options.clock ?? new SystemClock();
  const sessionId = options.sessionId ?? DEFAULT_SESSION_ID;
  const threshold = options.threshold ?? DEFAULT_EXECUTIVE_GOAL_FOCUS_THRESHOLD;

  function shouldDeferToGoalFollowup(goal: GoalRecord, nowMs: number): boolean {
    if (options.goalFollowupDue?.enabled !== true) {
      return false;
    }

    return isGoalFollowupDueMatch({
      goal,
      nowMs,
      lookaheadMs: options.goalFollowupDue.lookaheadMs,
      staleMs: options.goalFollowupDue.staleMs,
    });
  }

  function scoreGoals(input: {
    goals: readonly GoalRecord[];
    nowMs: number;
    autonomyPayload: Record<string, unknown>;
  }) {
    return selectExecutiveFocus({
      goals: input.goals,
      cognitionInput: AUTONOMOUS_WAKE_USER_MESSAGE,
      autonomyPayload: input.autonomyPayload,
      nowMs: input.nowMs,
      threshold,
      deadlineLookaheadMs: options.deadlineLookaheadMs,
      staleMs: options.stalenessMs,
    });
  }

  return {
    name: TRIGGER_NAME,
    type: "trigger",
    async scan() {
      if (!options.enabled) {
        return [];
      }

      const nowMs = clock.now();
      const goals = await listSelfVisibleActiveGoals({
        goalsRepository: options.goalsRepository,
        episodicRepository: options.episodicRepository,
      });
      const goalsById = new Map(goals.map((goal) => [goal.id, goal]));
      const events: DueEvent<ExecutiveFocusDuePayload>[] = [];
      const eventGoalIds = new Set<GoalRecord["id"]>();

      for (const goal of goals) {
        const dueStep = options.executiveStepsRepository
          .listOpen(goal.id)
          .filter((step) => step.due_at !== null && step.due_at <= nowMs + options.dueLeadMs)
          .sort(
            (left, right) =>
              (left.due_at ?? Number.POSITIVE_INFINITY) -
                (right.due_at ?? Number.POSITIVE_INFINITY) ||
              left.created_at - right.created_at ||
              left.id.localeCompare(right.id),
          )[0];

        if (dueStep === undefined) {
          continue;
        }

        const topOpenStep = options.executiveStepsRepository.topOpen(goal.id);
        const focus = scoreGoals({
          goals,
          nowMs,
          autonomyPayload: {
            trigger: TRIGGER_NAME,
            reason: "step_due",
            selected_goal_id: goal.id,
            selected_goal_description: goal.description,
            due_step_description: dueStep.description,
            top_open_step_description: topOpenStep?.description ?? null,
          },
        });
        const score = focus.candidates.find((candidate) => candidate.goal_id === goal.id);

        if (score === undefined) {
          continue;
        }

        const dueAt = dueStep.due_at ?? nowMs;
        const attemptKey = dueStep.last_attempt_ts ?? dueStep.created_at;
        const watermarkProcessName = `${WATERMARK_PREFIX}:step:${dueStep.id}:${dueAt}:${dueStep.status}:${attemptKey}`;

        if (options.watermarkRepository.get(watermarkProcessName, sessionId) !== null) {
          continue;
        }

        events.push({
          id: `step:${dueStep.id}:${dueAt}:${dueStep.status}:${attemptKey}`,
          sourceName: TRIGGER_NAME,
          sourceType: "trigger",
          watermarkProcessName,
          sortTs: dueAt,
          payload: buildScorePayload({
            goal,
            score,
            threshold,
            topOpenStep,
            reason: "step_due",
            dueStep,
          }),
        });
        eventGoalIds.add(goal.id);
      }

      const focus = scoreGoals({
        goals,
        nowMs,
        autonomyPayload: {
          trigger: TRIGGER_NAME,
          reason: "goal_stale",
        },
      });
      const selectedScore = focus.selected_score;
      const selectedGoal =
        focus.selected_goal === null ? null : (goalsById.get(focus.selected_goal.id) ?? null);

      if (
        selectedGoal !== null &&
        selectedScore !== null &&
        !eventGoalIds.has(selectedGoal.id) &&
        !shouldDeferToGoalFollowup(selectedGoal, nowMs)
      ) {
        const progressAnchor = selectedGoal.last_progress_ts ?? selectedGoal.created_at;
        const staleDue = progressAnchor + options.stalenessMs <= nowMs;
        const watermarkProcessName = `${WATERMARK_PREFIX}:goal:${selectedGoal.id}:${progressAnchor}`;

        if (staleDue && options.watermarkRepository.get(watermarkProcessName, sessionId) === null) {
          events.push({
            id: `goal:${selectedGoal.id}:${progressAnchor}`,
            sourceName: TRIGGER_NAME,
            sourceType: "trigger",
            watermarkProcessName,
            sortTs: progressAnchor + options.stalenessMs,
            payload: buildScorePayload({
              goal: selectedGoal,
              score: selectedScore,
              threshold,
              topOpenStep: options.executiveStepsRepository.topOpen(selectedGoal.id),
              reason: "goal_stale",
            }),
          });
        }
      }

      return events.sort(
        (left, right) => left.sortTs - right.sortTs || left.id.localeCompare(right.id),
      );
    },
    buildTurn(event) {
      return {
        audience: "self",
        stakes: "low",
        userMessage: AUTONOMOUS_WAKE_USER_MESSAGE,
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
