import { computeExecutiveContextFits, selectExecutiveFocus } from "../../executive/index.js";
import type { EmbeddingClient } from "../../embeddings/index.js";
import { flattenGoalTree, type GoalRecord, type GoalsRepository } from "../../memory/self/index.js";
import {
  goalMemoryDisclosureLabel,
  memoryDisclosurePayloadFields,
} from "../../memory/common/disclosure-serializers.js";
import type { SourceStreamAudienceDisclosureResolver } from "../../memory/common/index.js";
import type { StreamWatermarkRepository } from "../../stream/index.js";
import type { TurnTracer } from "../../tracing/tracer.js";
import { SystemClock, type Clock } from "../../util/clock.js";
import { DEFAULT_SESSION_ID, type SessionId } from "../../util/ids.js";
import { buildSelfScoringFeatureSet } from "../../retrieval/scoring-features.js";
import {
  getExecutiveFocusGoalStaleBackoffProcessName,
  goalStaleBackoffState,
  type GoalStaleBackoffState,
} from "../executive-focus-stale-backoff.js";
import type { AutonomyTrigger, DueEvent } from "../types.js";

const TRIGGER_NAME = "goal_followup_due" as const;
const WATERMARK_PREFIX = "autonomy:goal-followup-due";
const DAY_MS = 24 * 60 * 60 * 1_000;
const NEXT_DUE_CANDIDATE_LIMIT = 512;

export type GoalFollowupDuePayload = {
  goal_id: GoalRecord["id"];
  selected_goal_id: GoalRecord["id"];
  selected_goal: GoalRecord & ReturnType<typeof memoryDisclosurePayloadFields>;
  description: string;
  priority: number;
  target_at: number | null;
  last_progress_ts: number | null;
  days_stale: number;
  reason: "deadline" | "stale" | "both";
} & ReturnType<typeof memoryDisclosurePayloadFields>;

export type GoalFollowupDueTriggerOptions = {
  goalsRepository: GoalsRepository;
  sourceStreamAudienceDisclosureResolver?: SourceStreamAudienceDisclosureResolver;
  watermarkRepository: StreamWatermarkRepository;
  lookaheadMs: number;
  staleMs: number;
  staleBackoff: {
    baseCooldownMs: number;
    multiplier: number;
    maxCooldownMs: number;
    dormancyCount: number;
  };
  respectStaleBackoff: boolean;
  executiveScoring?: {
    embeddingClient: EmbeddingClient;
    threshold: number;
    deadlineLookaheadMs: number;
    staleMs: number;
    tracer?: TurnTracer;
  };
  clock?: Clock;
  sessionId?: SessionId;
  goalStaleBackoffActionAvailabilityKey?: () => string | null;
};

function dueAfterStrictThreshold(thresholdTs: number): number {
  return thresholdTs + 1;
}

type GoalFollowupPhase = "deadline" | "stale";

function legacyLatchProcessName(goal: GoalRecord): string {
  const targetAtKey = goal.target_at ?? "no-target";
  const progressKey = goal.last_progress_ts ?? goal.created_at;

  return `${WATERMARK_PREFIX}:${goal.id}:${targetAtKey}:${progressKey}`;
}

function phaseLatchProcessName(goal: GoalRecord, phase: GoalFollowupPhase): string {
  return `${legacyLatchProcessName(goal)}:${phase}`;
}

export function createGoalFollowupDueTrigger(
  options: GoalFollowupDueTriggerOptions,
): AutonomyTrigger<GoalFollowupDuePayload> {
  const clock = options.clock ?? new SystemClock();
  const sessionId = options.sessionId ?? DEFAULT_SESSION_ID;

  function staleBackoff(
    goal: GoalRecord,
    actionAvailabilityKey: string | null,
  ): GoalStaleBackoffState {
    if (!options.respectStaleBackoff) {
      return { endMs: null, actionAvailabilityChanged: false };
    }

    return goalStaleBackoffState({
      watermark: options.watermarkRepository.get(
        getExecutiveFocusGoalStaleBackoffProcessName(goal.id),
        sessionId,
      ),
      lastProgressTs: goal.last_progress_ts,
      actionAvailabilityKey,
      ...options.staleBackoff,
    });
  }

  async function attachExecutiveScores(
    goals: readonly GoalRecord[],
    dueEvents: readonly DueEvent<GoalFollowupDuePayload>[],
    nowMs: number,
  ): Promise<DueEvent<GoalFollowupDuePayload>[]> {
    const scoring = options.executiveScoring;

    if (scoring === undefined || dueEvents.length === 0) {
      return [...dueEvents];
    }

    let goalVectors: Awaited<ReturnType<typeof buildSelfScoringFeatureSet>>["goalVectors"] = [];

    try {
      goalVectors = (
        await buildSelfScoringFeatureSet({
          embeddingClient: scoring.embeddingClient,
          goals,
          activeValues: [],
        })
      ).goalVectors;
    } catch (error) {
      if (scoring.tracer?.enabled === true) {
        scoring.tracer.emit("retrieval.degraded", {
          turnId: "autonomy:goal_followup_due",
          subsystem: "scoring_features",
          reason: error instanceof Error ? error.message : String(error),
        });
      }
    }

    // Use the same wake-time scoring context as today's executive stale-goal
    // selection so a batch's primary is the goal that path would have named.
    const autonomyPayload = {
      trigger: "executive_focus_due",
      reason: "goal_stale",
    };
    const contextText = JSON.stringify(autonomyPayload);
    let contextFitByGoalId: Awaited<ReturnType<typeof computeExecutiveContextFits>> = new Map();

    try {
      contextFitByGoalId = await computeExecutiveContextFits({
        embeddingClient: scoring.embeddingClient,
        goalVectors,
        contextText,
      });
    } catch (error) {
      if (scoring.tracer?.enabled === true) {
        scoring.tracer.emit("retrieval.degraded", {
          turnId: "autonomy:goal_followup_due",
          subsystem: "executive_context_fit",
          reason: error instanceof Error ? error.message : String(error),
        });
      }
    }

    const focus = selectExecutiveFocus({
      goals,
      cognitionInput: contextText,
      autonomyPayload,
      nowMs,
      threshold: scoring.threshold,
      deadlineLookaheadMs: scoring.deadlineLookaheadMs,
      staleMs: scoring.staleMs,
      scoreContext: "wake_time_trigger_selection",
      contextFitByGoalId,
    });
    const scoreByGoalId = new Map(
      focus.candidates.map((candidate, rank) => [candidate.goal_id, { candidate, rank }]),
    );

    return dueEvents.map((event) => {
      const ranked = scoreByGoalId.get(event.payload.goal_id);

      return ranked === undefined
        ? event
        : {
            ...event,
            executiveGoalScore: ranked.candidate,
            executiveGoalRank: ranked.rank,
          };
    });
  }

  return {
    name: TRIGGER_NAME,
    type: "trigger",
    sourceCategory: "operational",
    async scan() {
      const nowMs = clock.now();
      const actionAvailabilityKey = options.goalStaleBackoffActionAvailabilityKey?.() ?? null;
      const rawGoals = flattenGoalTree(options.goalsRepository.list({ status: "active" }));
      const goals =
        options.sourceStreamAudienceDisclosureResolver?.resolve({ goals: rawGoals }).goals ??
        rawGoals;
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
          const phase: GoalFollowupPhase = deadlineDue ? "deadline" : "stale";
          const legacyProcessName = legacyLatchProcessName(goal);
          const watermarkProcessName = phaseLatchProcessName(goal, phase);
          const backoff = staleBackoff(goal, actionAvailabilityKey);

          // A phase latch identifies the goal/progress threshold, so its key
          // cannot express the newly executable topology. Let only the durable
          // dormant-key mismatch pierce it; scheduler bookkeeping re-stamps the
          // key after an empty wake and closes this exception again.
          if (
            !backoff.actionAvailabilityChanged &&
            (options.watermarkRepository.get(legacyProcessName, sessionId) !== null ||
              options.watermarkRepository.get(watermarkProcessName, sessionId) !== null)
          ) {
            return null;
          }

          if (!deadlineDue) {
            if (backoff.endMs !== null && backoff.endMs > nowMs) {
              return null;
            }
          }

          const reason = deadlineDue && staleDue ? "both" : deadlineDue ? "deadline" : "stale";
          const staleSinceMs = Math.max(0, nowMs - baseProgressTs);
          const sortTs =
            goal.target_at === null
              ? baseProgressTs + options.staleMs
              : Math.min(goal.target_at, baseProgressTs + options.staleMs);

          return {
            id: `${goal.id}:${targetAtKey}:${progressKey}:${phase}`,
            sourceName: TRIGGER_NAME,
            sourceType: "trigger",
            watermarkProcessName,
            sortTs,
            stateTs: baseProgressTs,
            ...(actionAvailabilityKey === null
              ? {}
              : { goalStaleBackoffActionAvailabilityKey: actionAvailabilityKey }),
            payload: {
              goal_id: goal.id,
              selected_goal_id: goal.id,
              selected_goal: {
                ...goal,
                ...memoryDisclosurePayloadFields(goalMemoryDisclosureLabel(goal)),
              },
              description: goal.description,
              priority: goal.priority,
              target_at: goal.target_at,
              last_progress_ts: goal.last_progress_ts,
              days_stale: Math.floor(staleSinceMs / DAY_MS),
              reason,
              ...memoryDisclosurePayloadFields(goalMemoryDisclosureLabel(goal)),
            },
          };
        })
        .filter((event): event is DueEvent<GoalFollowupDuePayload> => event !== null);

      const scoredEvents = await attachExecutiveScores(goals, dueEvents, nowMs);

      return scoredEvents.sort(
        (left, right) =>
          left.sortTs - right.sortTs || right.payload.priority - left.payload.priority,
      );
    },
    async nextDueAt() {
      const nowMs = clock.now();
      const actionAvailabilityKey = options.goalStaleBackoffActionAvailabilityKey?.() ?? null;
      const candidates = options.goalsRepository.listActiveFollowupDueCandidatesReadOnly({
        lookaheadMs: options.lookaheadMs,
        staleMs: options.staleMs,
        limit: NEXT_DUE_CANDIDATE_LIMIT + 1,
      });

      if (candidates.length > NEXT_DUE_CANDIDATE_LIMIT) {
        return null;
      }

      let nextCandidateAt: number | null = null;

      for (const candidate of candidates) {
        const goal = candidate.goal;
        const baseProgressTs = goal.last_progress_ts ?? goal.created_at;
        const deadlineDue = goal.target_at !== null && goal.target_at - nowMs < options.lookaheadMs;
        const legacyProcessName = legacyLatchProcessName(goal);
        const backoff = staleBackoff(goal, actionAvailabilityKey);

        if (
          !backoff.actionAvailabilityChanged &&
          options.watermarkRepository.get(legacyProcessName, sessionId) !== null
        ) {
          continue;
        }

        const deadlineProcessName = phaseLatchProcessName(goal, "deadline");

        if (deadlineDue) {
          if (
            backoff.actionAvailabilityChanged ||
            options.watermarkRepository.get(deadlineProcessName, sessionId) === null
          ) {
            return nowMs;
          }

          continue;
        }

        const staleDueAt = dueAfterStrictThreshold(baseProgressTs + options.staleMs);
        const deadlineDueAt =
          goal.target_at === null
            ? Number.POSITIVE_INFINITY
            : dueAfterStrictThreshold(goal.target_at - options.lookaheadMs);
        const staleCandidateAt =
          (!backoff.actionAvailabilityChanged &&
            options.watermarkRepository.get(phaseLatchProcessName(goal, "stale"), sessionId) !==
              null) ||
          backoff.endMs === Number.POSITIVE_INFINITY
            ? Number.POSITIVE_INFINITY
            : Math.max(staleDueAt, backoff.endMs ?? Number.NEGATIVE_INFINITY, nowMs);
        // Deadline piercing begins at the structural lookahead boundary, even
        // when that boundary lies in the future at describe-time.
        const deadlineCandidateAt =
          backoff.actionAvailabilityChanged ||
          options.watermarkRepository.get(deadlineProcessName, sessionId) === null
            ? Math.max(deadlineDueAt, nowMs)
            : Number.POSITIVE_INFINITY;
        const effectiveDueAt = Math.min(staleCandidateAt, deadlineCandidateAt);

        if (effectiveDueAt === Number.POSITIVE_INFINITY) {
          continue;
        }

        nextCandidateAt =
          nextCandidateAt === null ? effectiveDueAt : Math.min(nextCandidateAt, effectiveDueAt);
      }

      return nextCandidateAt;
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
