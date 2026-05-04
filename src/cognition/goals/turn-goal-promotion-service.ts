import type { LLMClient } from "../../llm/index.js";
import type { ExecutiveStepsRepository } from "../../executive/index.js";
import type { IdentityService } from "../../memory/identity/index.js";
import type { GoalRecord } from "../../memory/self/index.js";
import type { Clock } from "../../util/clock.js";
import type { EntityId, ExecutiveStepId, GoalId, StreamEntryId } from "../../util/ids.js";
import type { ExtractCorrectivePreferenceInput } from "../commitments/corrective-preference-extractor.js";
import type { TurnTracer } from "../tracing/tracer.js";
import type { TemporalCue } from "../types.js";
import { GoalPromotionExtractor, type GoalPromotionCandidate } from "./goal-promotion-extractor.js";

const GOAL_PROMOTION_PROVENANCE = {
  kind: "online" as const,
  process: "goal-promotion-extractor",
};

export type PersistedGoalPromotionIds = {
  goalIds: GoalId[];
  executiveStepIds: ExecutiveStepId[];
};

export type TurnGoalPromotionServiceOptions = {
  model: string;
  identityService: Pick<IdentityService, "addGoal">;
  executiveStepsRepository: Pick<ExecutiveStepsRepository, "add">;
  clock: Clock;
  tracer: TurnTracer;
};

export type ExtractTurnGoalPromotionsInput = {
  llmClient: LLMClient;
  turnId: string;
  isUserTurn: boolean;
  userMessage: string;
  recentHistory: ExtractCorrectivePreferenceInput["recentHistory"];
  audienceEntityId: EntityId | null;
  temporalCue: TemporalCue | null;
  activeGoals: readonly GoalRecord[];
  persistedUserEntryId?: StreamEntryId;
  onHookFailure: (hook: string, error: unknown, details?: Record<string, unknown>) => Promise<void>;
};

export class TurnGoalPromotionService {
  constructor(private readonly options: TurnGoalPromotionServiceOptions) {}

  async extractAndPersist(
    input: ExtractTurnGoalPromotionsInput,
  ): Promise<PersistedGoalPromotionIds> {
    if (!input.isUserTurn) {
      return {
        goalIds: [],
        executiveStepIds: [],
      };
    }

    const goalPromotionExtractor = new GoalPromotionExtractor({
      llmClient: input.llmClient,
      model: this.options.model,
      tracer: this.options.tracer,
      turnId: input.turnId,
      onDegraded: (reason, error) => {
        this.emitDegraded({
          turnId: input.turnId,
          reason,
          error,
        });
      },
    });
    const goalPromotionCandidates = await goalPromotionExtractor.extract({
      userMessage: input.userMessage,
      recentHistory: input.recentHistory,
      audienceEntityId: input.audienceEntityId,
      temporalCue: input.temporalCue,
      activeGoals: input.activeGoals.map((goal) => ({
        id: goal.id,
        description: goal.description,
        priority: goal.priority,
        target_at: goal.target_at,
      })),
    });

    if (goalPromotionCandidates.length === 0) {
      return {
        goalIds: [],
        executiveStepIds: [],
      };
    }

    return this.persistGoalPromotions({
      candidates: goalPromotionCandidates,
      audienceEntityId: input.audienceEntityId,
      persistedUserEntryId: input.persistedUserEntryId,
      turnId: input.turnId,
      onHookFailure: input.onHookFailure,
    });
  }

  private async persistGoalPromotions(input: {
    candidates: readonly GoalPromotionCandidate[];
    audienceEntityId: EntityId | null;
    persistedUserEntryId?: StreamEntryId;
    turnId: string;
    onHookFailure: (
      hook: string,
      error: unknown,
      details?: Record<string, unknown>,
    ) => Promise<void>;
  }): Promise<PersistedGoalPromotionIds> {
    const sourceStreamEntryIds =
      input.persistedUserEntryId === undefined ? undefined : [input.persistedUserEntryId];
    const persisted: PersistedGoalPromotionIds = {
      goalIds: [],
      executiveStepIds: [],
    };

    for (const candidate of input.candidates) {
      let goal: GoalRecord;

      try {
        goal = this.options.identityService.addGoal({
          description: candidate.description,
          priority: candidate.priority,
          status: "active",
          targetAt: candidate.target_at,
          audienceEntityId: input.audienceEntityId,
          provenance: GOAL_PROMOTION_PROVENANCE,
          sourceStreamEntryIds,
        });
      } catch (error) {
        this.emitDegraded({
          turnId: input.turnId,
          reason: "goal_persist_failed",
          error,
          details: {
            description: candidate.description,
          },
        });
        await input.onHookFailure("goal_promotion_goal_persist", error, {
          description: candidate.description,
        });
        continue;
      }

      persisted.goalIds.push(goal.id);

      if (candidate.initial_step === null) {
        continue;
      }

      try {
        const step = this.options.executiveStepsRepository.add({
          goalId: goal.id,
          description: candidate.initial_step.description,
          kind: candidate.initial_step.kind,
          dueAt: candidate.initial_step.due_at,
          provenance: GOAL_PROMOTION_PROVENANCE,
        });
        persisted.executiveStepIds.push(step.id);
      } catch (error) {
        this.emitDegraded({
          turnId: input.turnId,
          reason: "initial_step_persist_failed",
          error,
          details: {
            goalId: goal.id,
          },
        });
        await input.onHookFailure("goal_promotion_initial_step_persist", error, {
          goalId: goal.id,
        });
      }
    }

    return persisted;
  }

  private emitDegraded(input: {
    turnId: string;
    reason: string;
    error?: unknown;
    details?: Record<string, unknown>;
  }): void {
    if (!this.options.tracer.enabled) {
      return;
    }

    this.options.tracer.emit("goal_promotion_extractor_degraded", {
      turnId: input.turnId,
      reason: input.reason,
      ...(input.details ?? {}),
      ...(this.options.tracer.includePayloads && input.error !== undefined
        ? { error: input.error instanceof Error ? input.error.message : String(input.error) }
        : {}),
    });
  }
}
