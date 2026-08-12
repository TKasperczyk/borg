import { computeExecutiveContextFits, selectExecutiveFocus } from "../../executive/index.js";
import type { ExecutiveFocus, ExecutiveStepsRepository } from "../../executive/index.js";
import type { EmbeddingClient } from "../../embeddings/index.js";
import type {
  AutobiographicalRepository,
  GoalRecord,
  GoalsRepository,
  GrowthMarkersRepository,
  TraitsRepository,
  ValuesRepository,
} from "../../memory/self/index.js";
import {
  buildSelfScoringFeatureSet,
  selectActiveScoringValues,
  toRetrievalScoringFeatures,
  type RetrievalScoringFeatures,
  type SelfScoringFeatureSet,
} from "../../retrieval/scoring-features.js";
import type { MemoryDisclosureLabel } from "../../retrieval/recall-context.js";
import type { Clock } from "../../util/clock.js";
import { goalIdHelpers, type EntityId, type GoalId, type SessionId } from "../../util/ids.js";
import type { AutonomyTriggerContext } from "../autonomy-trigger.js";
import {
  memoryDisclosureLabelFromMetadata,
  memoryDisclosurePayloadFields,
} from "../../memory/common/disclosure-serializers.js";
import type { SelfSnapshot } from "../deliberation/deliberator.js";
import type { TurnTracer } from "../../tracing/tracer.js";
import type { PerceptionResult } from "../types.js";
import { listActiveGoalsForCognition as listActiveGoalRecordsForCognition } from "./active-goals.js";

export type TurnSelfContextOptions = {
  embeddingClient: EmbeddingClient;
  valuesRepository: Pick<ValuesRepository, "list">;
  goalsRepository: Pick<GoalsRepository, "list">;
  traitsRepository: Pick<TraitsRepository, "list">;
  autobiographicalRepository?: Pick<AutobiographicalRepository, "currentPeriod">;
  growthMarkersRepository?: Pick<GrowthMarkersRepository, "list">;
  executiveStepsRepository: Pick<ExecutiveStepsRepository, "topOpen">;
  clock: Clock;
  tracer: TurnTracer;
  goalFocusThreshold: number;
  goalFollowupLookaheadMs: number;
  goalFollowupStaleMs: number;
};

export type TurnSelfContextInput = {
  turnId: string;
  sessionId?: SessionId;
  cognitionInput: string;
  perception: PerceptionResult;
  autonomyTrigger?: AutonomyTriggerContext | null;
  audienceEntityId: EntityId | null;
};

export type TurnSelfContext = {
  selfSnapshot: SelfSnapshot;
  activeScoringValues: ReturnType<typeof selectActiveScoringValues>;
  selfScoringFeatures: SelfScoringFeatureSet;
  retrievalScoringFeatures: RetrievalScoringFeatures;
  executiveFocus: ExecutiveFocus;
};

function getForcedExecutiveFocusGoalId(
  autonomyTrigger: AutonomyTriggerContext | null | undefined,
): GoalId | null {
  if (autonomyTrigger?.source_name === "executive_focus_due") {
    if (autonomyTrigger.payload.reason !== "step_due") {
      return null;
    }

    const candidate = autonomyTrigger.payload.force_executive_focus_goal_id;

    return typeof candidate === "string" && goalIdHelpers.is(candidate) ? candidate : null;
  }

  if (autonomyTrigger?.source_name !== "goal_followup_due") {
    return null;
  }

  const candidate = autonomyTrigger.payload.selected_goal_id;
  const selectedGoal = autonomyTrigger.payload.selected_goal;

  return typeof candidate === "string" &&
    goalIdHelpers.is(candidate) &&
    isRecord(selectedGoal) &&
    selectedGoal.id === candidate
    ? candidate
    : null;
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return value !== null && typeof value === "object" && !Array.isArray(value);
}

function executiveFocusGoalDisclosureLabelsFromTrigger(
  autonomyTrigger: AutonomyTriggerContext | null | undefined,
): ReadonlyMap<GoalId, MemoryDisclosureLabel> {
  if (
    autonomyTrigger?.source_name !== "executive_focus_due" &&
    autonomyTrigger?.source_name !== "goal_followup_due"
  ) {
    return new Map();
  }

  const selectedGoalId = autonomyTrigger.payload.selected_goal_id;
  const selectedGoalPayload = autonomyTrigger.payload.selected_goal;

  if (typeof selectedGoalId !== "string" || !goalIdHelpers.is(selectedGoalId)) {
    return new Map();
  }

  if (!isRecord(selectedGoalPayload)) {
    return new Map();
  }

  const disclosureLabel = memoryDisclosureLabelFromMetadata(selectedGoalPayload.disclosure_label);

  return disclosureLabel === null ? new Map() : new Map([[selectedGoalId, disclosureLabel]]);
}

function annotateGoalsWithDisclosure(
  goals: readonly GoalRecord[],
  disclosureLabelsByGoalId: ReadonlyMap<GoalId, MemoryDisclosureLabel>,
): SelfSnapshot["goals"] {
  return goals.map((goal) => {
    const disclosureLabel = disclosureLabelsByGoalId.get(goal.id);

    return disclosureLabel === undefined
      ? goal
      : {
          ...goal,
          ...memoryDisclosurePayloadFields(disclosureLabel),
        };
  });
}

function applyForcedExecutiveFocus(
  focus: ExecutiveFocus,
  forcedGoalId: GoalId | null,
): ExecutiveFocus {
  if (forcedGoalId === null) {
    return focus;
  }

  const forcedScore = focus.candidates.find((candidate) => candidate.goal_id === forcedGoalId);

  if (forcedScore === undefined) {
    return focus;
  }

  return {
    ...focus,
    selected_goal: forcedScore.goal,
    selected_score: forcedScore,
  };
}

export class TurnSelfContextBuilder {
  constructor(private readonly options: TurnSelfContextOptions) {}

  async buildSelfSnapshot(_audienceEntityId: EntityId | null): Promise<SelfSnapshot> {
    const goals = listActiveGoalRecordsForCognition(this.options.goalsRepository);
    const values = this.options.valuesRepository.list();
    const traits = this.options.traitsRepository.list();
    const currentPeriod = this.options.autobiographicalRepository?.currentPeriod() ?? null;
    const recentGrowthMarkers = this.options.growthMarkersRepository?.list({ limit: 3 }) ?? [];

    return {
      values,
      goals,
      traits,
      currentPeriod,
      recentGrowthMarkers,
    };
  }

  async listActiveGoalsForCognition(_audienceEntityId: EntityId | null): Promise<GoalRecord[]> {
    return listActiveGoalRecordsForCognition(this.options.goalsRepository);
  }

  async build(input: TurnSelfContextInput): Promise<TurnSelfContext> {
    const baseSelfSnapshot = await this.buildSelfSnapshot(input.audienceEntityId);
    const disclosureLabelsByGoalId = executiveFocusGoalDisclosureLabelsFromTrigger(
      input.autonomyTrigger,
    );
    const selfSnapshot: SelfSnapshot = {
      ...baseSelfSnapshot,
      goals: annotateGoalsWithDisclosure(baseSelfSnapshot.goals, disclosureLabelsByGoalId),
    };
    const executiveContextText = [
      input.cognitionInput,
      ...input.perception.entities,
      input.autonomyTrigger === null || input.autonomyTrigger === undefined
        ? ""
        : JSON.stringify(input.autonomyTrigger.payload),
    ]
      .join(" ")
      .trim();
    const activeScoringValues = selectActiveScoringValues(selfSnapshot.values);
    let selfScoringFeatures: SelfScoringFeatureSet = {
      goalVectors: [],
      valueVectors: [],
    };
    let contextFitByGoalId: Awaited<ReturnType<typeof computeExecutiveContextFits>> = new Map();

    try {
      selfScoringFeatures = await buildSelfScoringFeatureSet({
        embeddingClient: this.options.embeddingClient,
        goals: selfSnapshot.goals,
        activeValues: activeScoringValues,
      });
    } catch (error) {
      if (this.options.tracer.enabled) {
        this.options.tracer.emit("retrieval.degraded", {
          turnId: input.turnId,
          ...(input.sessionId !== undefined ? { session_id: input.sessionId } : {}),
          subsystem: "scoring_features",
          reason: error instanceof Error ? error.message : String(error),
        });
      }
    }

    try {
      contextFitByGoalId = await computeExecutiveContextFits({
        embeddingClient: this.options.embeddingClient,
        goalVectors: selfScoringFeatures.goalVectors,
        contextText: executiveContextText,
      });
    } catch (error) {
      if (this.options.tracer.enabled) {
        this.options.tracer.emit("retrieval.degraded", {
          turnId: input.turnId,
          ...(input.sessionId !== undefined ? { session_id: input.sessionId } : {}),
          subsystem: "executive_context_fit",
          reason: error instanceof Error ? error.message : String(error),
        });
      }
    }

    const executiveFocus = applyForcedExecutiveFocus(
      selectExecutiveFocus({
        goals: selfSnapshot.goals,
        cognitionInput: input.cognitionInput,
        perceptionEntities: input.perception.entities,
        autonomyPayload: input.autonomyTrigger?.payload ?? null,
        nowMs: this.options.clock.now(),
        threshold: this.options.goalFocusThreshold,
        deadlineLookaheadMs: this.options.goalFollowupLookaheadMs,
        staleMs: this.options.goalFollowupStaleMs,
        scoreContext: "turn_selection",
        contextFitByGoalId,
      }),
      getForcedExecutiveFocusGoalId(input.autonomyTrigger),
    );
    const retrievalScoringFeatures = toRetrievalScoringFeatures({
      selfFeatures: selfScoringFeatures,
      primaryGoalId: executiveFocus.selected_goal?.id ?? null,
    });
    const executiveFocusWithStep =
      executiveFocus.selected_goal === null
        ? executiveFocus
        : {
            ...executiveFocus,
            next_step: (() => {
              const step = this.options.executiveStepsRepository.topOpen(
                executiveFocus.selected_goal.id,
              );
              const disclosureLabel = disclosureLabelsByGoalId.get(executiveFocus.selected_goal.id);

              return step === null || disclosureLabel === undefined
                ? step
                : {
                    ...step,
                    ...memoryDisclosurePayloadFields(disclosureLabel),
                  };
            })(),
          };

    return {
      selfSnapshot,
      activeScoringValues,
      selfScoringFeatures,
      retrievalScoringFeatures,
      executiveFocus: executiveFocusWithStep,
    };
  }
}
