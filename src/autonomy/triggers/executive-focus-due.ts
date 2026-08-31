import {
  DEFAULT_EXECUTIVE_GOAL_FOCUS_THRESHOLD,
  computeExecutiveContextFits,
  selectExecutiveFocus,
} from "../../executive/index.js";
import type { ExecutiveStep, ExecutiveStepsRepository } from "../../executive/index.js";
import type { EmbeddingClient } from "../../embeddings/index.js";
import type { EpisodicRepository } from "../../memory/episodic/index.js";
import type { GoalRecord, GoalsRepository } from "../../memory/self/index.js";
import { goalMemoryDisclosureLabel } from "../../memory/common/disclosure-serializers.js";
import { listActiveGoalsForCognition } from "../../cognition/self/active-goals.js";
import {
  combineMemoryDisclosureLabels,
  memoryDisclosureLabelFromEpisodeAccess,
  memoryDisclosureLabelMetadata,
  renderMemoryDisclosureLabelForModel,
  type MemoryDisclosureLabel,
  unknownMemoryDisclosureLabel,
} from "../../retrieval/index.js";
import {
  buildSelfScoringFeatureSet,
  type GoalScoringVector,
} from "../../retrieval/scoring-features.js";
import type { StreamWatermarkRepository } from "../../stream/index.js";
import type { TurnTracer } from "../../tracing/tracer.js";
import { SystemClock, type Clock } from "../../util/clock.js";
import { DEFAULT_SESSION_ID, type EpisodeId, type SessionId } from "../../util/ids.js";
import type { AutonomyTrigger, DueEvent } from "../types.js";
import {
  getExecutiveFocusGoalStaleBackoffProcessName,
  goalStaleBackoffEndMs,
} from "../executive-focus-stale-backoff.js";

const TRIGGER_NAME = "executive_focus_due" as const;
const WATERMARK_PREFIX = "autonomy:executive-focus-due";
const NEXT_DUE_CANDIDATE_LIMIT = 512;

type ExecutiveFocusDueReason = "step_due" | "goal_stale";
type ExecutiveFocusDisclosureLabelMetadata = ReturnType<typeof memoryDisclosureLabelMetadata>;

type ExecutiveFocusDisclosurePayload = {
  disclosure: string;
  disclosure_label: ExecutiveFocusDisclosureLabelMetadata;
};

type ExecutiveFocusGoalDisclosurePayload = ExecutiveFocusDisclosurePayload & {
  goal_disclosure: string;
  goal_disclosure_label: ExecutiveFocusDisclosureLabelMetadata;
  source_disclosure?: string;
  source_disclosure_label?: ExecutiveFocusDisclosureLabelMetadata;
};

type ExecutiveFocusDueStepPayload = Pick<
  ExecutiveStep,
  "id" | "goal_id" | "description" | "status" | "kind" | "due_at" | "last_attempt_ts"
> &
  ExecutiveFocusDisclosurePayload;

export type ExecutiveFocusDuePayload = {
  reason: ExecutiveFocusDueReason;
  selected_goal_id: GoalRecord["id"];
  force_executive_focus_goal_id?: GoalRecord["id"];
  selected_goal: {
    goal_id: GoalRecord["id"];
    description: string;
    priority: number;
    target_at: number | null;
    last_progress_ts: number | null;
  } & ExecutiveFocusGoalDisclosurePayload;
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
  embeddingClient: EmbeddingClient;
  watermarkRepository: StreamWatermarkRepository;
  threshold?: number;
  stalenessMs: number;
  dueLeadMs: number;
  wakeCooldownMs: number;
  wakeCooldownBackoffMultiplier: number;
  wakeCooldownMaxMs: number;
  wakeEmptyDormancyCount: number;
  deadlineLookaheadMs: number;
  goalFollowupDue?: {
    enabled: boolean;
    lookaheadMs: number;
    staleMs: number;
  };
  clock?: Clock;
  tracer?: TurnTracer;
  sessionId?: SessionId;
  goalStaleBackoffActionAvailabilityKey?: () => string | null;
};

type ProvenanceScopedSelfRecord = {
  provenance?: {
    kind: string;
    episode_ids?: readonly EpisodeId[];
  } | null;
  evidence_episode_ids?: readonly EpisodeId[] | null;
  key_episode_ids?: readonly EpisodeId[] | null;
};

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

function disclosurePayload(label: MemoryDisclosureLabel): ExecutiveFocusDisclosurePayload {
  return {
    disclosure: renderMemoryDisclosureLabelForModel(label),
    disclosure_label: memoryDisclosureLabelMetadata(label),
  };
}

function goalDisclosurePayload(input: {
  goalLabel: MemoryDisclosureLabel;
  sourceLabel: MemoryDisclosureLabel | null;
}): ExecutiveFocusGoalDisclosurePayload {
  const combinedLabel = combineMemoryDisclosureLabels(
    input.sourceLabel === null ? [input.goalLabel] : [input.goalLabel, input.sourceLabel],
  );
  const combined = disclosurePayload(combinedLabel);
  const goalOnly = disclosurePayload(input.goalLabel);
  const sourceOnly = input.sourceLabel === null ? null : disclosurePayload(input.sourceLabel);

  return {
    disclosure: combined.disclosure,
    disclosure_label: combined.disclosure_label,
    goal_disclosure: goalOnly.disclosure,
    goal_disclosure_label: goalOnly.disclosure_label,
    ...(sourceOnly === null
      ? {}
      : {
          source_disclosure: sourceOnly.disclosure,
          source_disclosure_label: sourceOnly.disclosure_label,
        }),
  };
}

async function buildGoalDisclosurePayloads(options: {
  goals: readonly GoalRecord[];
  episodicRepository: EpisodicRepository;
}): Promise<Map<GoalRecord["id"], ExecutiveFocusGoalDisclosurePayload>> {
  const evidenceEpisodeIds = [
    ...new Set(options.goals.flatMap((goal) => getSelfRecordEvidenceEpisodeIds(goal))),
  ];
  const evidenceEpisodes =
    evidenceEpisodeIds.length === 0
      ? []
      : await options.episodicRepository.getMany(evidenceEpisodeIds);
  const episodesById = new Map(evidenceEpisodes.map((episode) => [episode.id, episode]));
  const disclosureByGoalId = new Map<GoalRecord["id"], ExecutiveFocusGoalDisclosurePayload>();

  for (const goal of options.goals) {
    const goalLabel = goalMemoryDisclosureLabel(goal);
    const sourceEpisodeIds = getSelfRecordEvidenceEpisodeIds(goal);
    const sourceLabel =
      sourceEpisodeIds.length === 0
        ? null
        : combineMemoryDisclosureLabels(
            sourceEpisodeIds.map((episodeId) => {
              const episode = episodesById.get(episodeId);
              return episode === undefined
                ? unknownMemoryDisclosureLabel()
                : memoryDisclosureLabelFromEpisodeAccess(episode);
            }),
          );
    disclosureByGoalId.set(
      goal.id,
      goalDisclosurePayload({
        goalLabel,
        sourceLabel,
      }),
    );
  }

  return disclosureByGoalId;
}

function serializeStep(
  step: ExecutiveStep,
  disclosure: ExecutiveFocusDisclosurePayload,
): ExecutiveFocusDueStepPayload {
  return {
    id: step.id,
    goal_id: step.goal_id,
    description: step.description,
    status: step.status,
    kind: step.kind,
    due_at: step.due_at,
    last_attempt_ts: step.last_attempt_ts,
    disclosure: disclosure.disclosure,
    disclosure_label: disclosure.disclosure_label,
  };
}

function buildScorePayload(input: {
  goal: GoalRecord;
  score: NonNullable<ReturnType<typeof selectExecutiveFocus>["selected_score"]>;
  threshold: number;
  topOpenStep: ExecutiveStep | null;
  reason: ExecutiveFocusDueReason;
  disclosure: ExecutiveFocusGoalDisclosurePayload;
  dueStep?: ExecutiveStep;
}): ExecutiveFocusDuePayload {
  return {
    reason: input.reason,
    selected_goal_id: input.goal.id,
    ...(input.reason === "step_due" ? { force_executive_focus_goal_id: input.goal.id } : {}),
    selected_goal: {
      goal_id: input.goal.id,
      description: input.goal.description,
      priority: input.goal.priority,
      target_at: input.goal.target_at,
      last_progress_ts: input.goal.last_progress_ts,
      disclosure: input.disclosure.disclosure,
      disclosure_label: input.disclosure.disclosure_label,
      goal_disclosure: input.disclosure.goal_disclosure,
      goal_disclosure_label: input.disclosure.goal_disclosure_label,
      ...(input.disclosure.source_disclosure === undefined ||
      input.disclosure.source_disclosure_label === undefined
        ? {}
        : {
            source_disclosure: input.disclosure.source_disclosure,
            source_disclosure_label: input.disclosure.source_disclosure_label,
          }),
    },
    selected_score: {
      score: input.score.score,
      components: input.score.components,
      reason: input.score.reason,
      threshold: input.threshold,
    },
    top_open_step:
      input.topOpenStep === null ? null : serializeStep(input.topOpenStep, input.disclosure),
    ...(input.dueStep === undefined
      ? {}
      : { due_step: serializeStep(input.dueStep, input.disclosure) }),
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

  function getGoalCooldownProcessName(goalId: GoalRecord["id"]): string {
    return `${WATERMARK_PREFIX}:cooldown:${goalId}`;
  }

  function getGoalCooldownEnd(input: {
    goal_id: GoalRecord["id"];
    last_progress_ts: number | null;
  }): number | null {
    const cooldown = options.watermarkRepository.get(
      getGoalCooldownProcessName(input.goal_id),
      sessionId,
    );

    if (cooldown === null) {
      return null;
    }

    if (input.last_progress_ts !== null && input.last_progress_ts >= cooldown.updatedAt) {
      return null;
    }

    return cooldown.updatedAt + options.wakeCooldownMs;
  }

  function isGoalCoolingDown(goal: GoalRecord, nowMs: number): boolean {
    const cooldownEnd = getGoalCooldownEnd({
      goal_id: goal.id,
      last_progress_ts: goal.last_progress_ts,
    });

    return cooldownEnd !== null && cooldownEnd > nowMs;
  }

  function getGoalStaleBackoffEnd(
    input: {
      goal_id: GoalRecord["id"];
      last_progress_ts: number | null;
    },
    actionAvailabilityKey: string | null,
  ): number | null {
    return goalStaleBackoffEndMs({
      watermark: options.watermarkRepository.get(
        getExecutiveFocusGoalStaleBackoffProcessName(input.goal_id),
        sessionId,
      ),
      lastProgressTs: input.last_progress_ts,
      baseCooldownMs: options.wakeCooldownMs,
      multiplier: options.wakeCooldownBackoffMultiplier,
      maxCooldownMs: options.wakeCooldownMaxMs,
      dormancyCount: options.wakeEmptyDormancyCount,
      actionAvailabilityKey,
    });
  }

  function isGoalStaleBackedOff(
    goal: GoalRecord,
    nowMs: number,
    actionAvailabilityKey: string | null,
  ): boolean {
    const backoffEnd = getGoalStaleBackoffEnd(
      {
        goal_id: goal.id,
        last_progress_ts: goal.last_progress_ts,
      },
      actionAvailabilityKey,
    );

    return backoffEnd !== null && backoffEnd > nowMs;
  }

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

  async function scoreGoals(input: {
    goals: readonly GoalRecord[];
    goalVectors: readonly GoalScoringVector[];
    nowMs: number;
    autonomyPayload: Record<string, unknown>;
  }) {
    const executiveContextText = JSON.stringify(input.autonomyPayload);
    let contextFitByGoalId: Awaited<ReturnType<typeof computeExecutiveContextFits>> = new Map();

    try {
      contextFitByGoalId = await computeExecutiveContextFits({
        embeddingClient: options.embeddingClient,
        goalVectors: input.goalVectors,
        contextText: executiveContextText,
      });
    } catch (error) {
      if (options.tracer?.enabled === true) {
        options.tracer.emit("retrieval.degraded", {
          turnId: "autonomy:executive_focus_due",
          subsystem: "executive_context_fit",
          reason: error instanceof Error ? error.message : String(error),
        });
      }
    }

    return selectExecutiveFocus({
      goals: input.goals,
      cognitionInput: executiveContextText,
      autonomyPayload: input.autonomyPayload,
      nowMs: input.nowMs,
      threshold,
      deadlineLookaheadMs: options.deadlineLookaheadMs,
      staleMs: options.stalenessMs,
      scoreContext: "wake_time_trigger_selection",
      contextFitByGoalId,
    });
  }

  return {
    name: TRIGGER_NAME,
    type: "trigger",
    sourceCategory: "operational",
    async scan() {
      if (!options.enabled) {
        return [];
      }

      const nowMs = clock.now();
      const actionAvailabilityKey = options.goalStaleBackoffActionAvailabilityKey?.() ?? null;
      const goals = listActiveGoalsForCognition(options.goalsRepository);
      const goalDisclosureById = await buildGoalDisclosurePayloads({
        goals,
        episodicRepository: options.episodicRepository,
      });
      let goalVectors: GoalScoringVector[] = [];

      try {
        goalVectors = [
          ...(
            await buildSelfScoringFeatureSet({
              embeddingClient: options.embeddingClient,
              goals,
              activeValues: [],
            })
          ).goalVectors,
        ];
      } catch (error) {
        if (options.tracer?.enabled === true) {
          options.tracer.emit("retrieval.degraded", {
            turnId: "autonomy:executive_focus_due",
            subsystem: "scoring_features",
            reason: error instanceof Error ? error.message : String(error),
          });
        }
      }
      const goalsById = new Map(goals.map((goal) => [goal.id, goal]));
      const events: DueEvent<ExecutiveFocusDuePayload>[] = [];
      const eventGoalIds = new Set<GoalRecord["id"]>();

      for (const goal of goals) {
        if (isGoalCoolingDown(goal, nowMs)) {
          continue;
        }

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
        const focus = await scoreGoals({
          goals,
          goalVectors,
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

        events.push({
          id: `step:${dueStep.id}:${dueAt}:${dueStep.status}:${attemptKey}`,
          sourceName: TRIGGER_NAME,
          sourceType: "trigger",
          watermarkProcessName: getGoalCooldownProcessName(goal.id),
          sortTs: dueAt,
          stateTs: dueStep.updated_at,
          executiveGoalScore: score,
          executiveGoalRank: focus.candidates.findIndex(
            (candidate) => candidate.goal_id === goal.id,
          ),
          payload: buildScorePayload({
            goal,
            score,
            threshold,
            topOpenStep,
            reason: "step_due",
            disclosure:
              goalDisclosureById.get(goal.id) ??
              goalDisclosurePayload({
                goalLabel: goalMemoryDisclosureLabel(goal),
                sourceLabel: null,
              }),
            dueStep,
          }),
        });
        eventGoalIds.add(goal.id);
      }

      const focus = await scoreGoals({
        goals,
        goalVectors,
        nowMs,
        autonomyPayload: {
          trigger: TRIGGER_NAME,
          reason: "goal_stale",
        },
      });
      const topEligibleCandidate = focus.candidates.find((candidate) => {
        const candidateGoal = goalsById.get(candidate.goal_id) ?? candidate.goal;

        return (
          candidate.score >= threshold &&
          !eventGoalIds.has(candidateGoal.id) &&
          !isGoalCoolingDown(candidateGoal, nowMs) &&
          !isGoalStaleBackedOff(candidateGoal, nowMs, actionAvailabilityKey) &&
          !shouldDeferToGoalFollowup(candidateGoal, nowMs)
        );
      });

      if (topEligibleCandidate !== undefined) {
        const selectedGoal =
          goalsById.get(topEligibleCandidate.goal_id) ?? topEligibleCandidate.goal;
        const progressAnchor = selectedGoal.last_progress_ts ?? selectedGoal.created_at;

        // Keep staleness out of the skip predicate: a fresh top-eligible goal
        // must suppress goal_stale rather than reaching past it to a lower stale one.
        if (progressAnchor + options.stalenessMs <= nowMs) {
          events.push({
            id: `goal:${selectedGoal.id}:${progressAnchor}`,
            sourceName: TRIGGER_NAME,
            sourceType: "trigger",
            watermarkProcessName: getGoalCooldownProcessName(selectedGoal.id),
            sortTs: progressAnchor + options.stalenessMs,
            stateTs: progressAnchor,
            executiveGoalScore: topEligibleCandidate,
            executiveGoalRank: focus.candidates.findIndex(
              (candidate) => candidate.goal_id === selectedGoal.id,
            ),
            ...(actionAvailabilityKey === null
              ? {}
              : { goalStaleBackoffActionAvailabilityKey: actionAvailabilityKey }),
            payload: buildScorePayload({
              goal: selectedGoal,
              score: topEligibleCandidate,
              threshold,
              topOpenStep: options.executiveStepsRepository.topOpen(selectedGoal.id),
              reason: "goal_stale",
              disclosure:
                goalDisclosureById.get(selectedGoal.id) ??
                goalDisclosurePayload({
                  goalLabel: goalMemoryDisclosureLabel(selectedGoal),
                  sourceLabel: null,
                }),
            }),
          });
        }
      }

      return events.sort(
        (left, right) => left.sortTs - right.sortTs || left.id.localeCompare(right.id),
      );
    },
    async nextDueAt() {
      if (!options.enabled) {
        return null;
      }

      // The step-due path is pure schedule data. The goal-stale path depends on
      // executive scoring/context fit in scan(), so this read-only surface does
      // not predict it rather than approximating a fire decision.
      const nowMs = clock.now();
      const candidates = options.executiveStepsRepository.listDueStepWakeCandidatesReadOnly({
        dueLeadMs: options.dueLeadMs,
        limit: NEXT_DUE_CANDIDATE_LIMIT + 1,
      });

      if (candidates.length > NEXT_DUE_CANDIDATE_LIMIT) {
        return null;
      }

      for (const candidate of candidates) {
        const cooldownEnd = getGoalCooldownEnd({
          goal_id: candidate.goal_id,
          last_progress_ts: candidate.goal_last_progress_ts,
        });
        const dueAt = Math.max(
          candidate.due_at,
          cooldownEnd === null ? Number.NEGATIVE_INFINITY : cooldownEnd,
          nowMs,
        );

        if (Number.isFinite(dueAt)) {
          return dueAt;
        }
      }

      return null;
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
          presentation: {
            score_basis: {
              score_context: "wake_time_trigger_selection",
              deadline_lookahead_ms: options.deadlineLookaheadMs,
              progress_debt_stale_ms: options.stalenessMs,
            },
          },
        },
      };
    },
  };
}
