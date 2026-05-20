import type { EmbeddingClient } from "../../embeddings/index.js";
import type { LLMClient } from "../../llm/index.js";
import type { ExecutiveStepsRepository } from "../../executive/index.js";
import type { IdentityService } from "../../memory/identity/index.js";
import type { GoalRecord, GoalsRepository } from "../../memory/self/index.js";
import { cosineSimilarity } from "../../retrieval/embedding-similarity.js";
import type { Clock } from "../../util/clock.js";
import type { EntityId, ExecutiveStepId, GoalId, StreamEntryId } from "../../util/ids.js";
import type { ExtractCorrectivePreferenceInput } from "../commitments/corrective-preference-extractor.js";
import type { TurnTracer } from "../tracing/tracer.js";
import type { TemporalCue } from "../types.js";
import {
  GoalPromotionExtractor,
  type GoalPromotionCandidate,
  type GoalPromotionInitialStep,
} from "./goal-promotion-extractor.js";

const GOAL_PROMOTION_PROVENANCE = {
  kind: "online" as const,
  process: "goal-promotion-extractor",
};
const GOAL_PROMOTION_DUPLICATE_SIMILARITY_THRESHOLD = 0.9;

export type PersistedGoalPromotionIds = {
  goalIds: GoalId[];
  executiveStepIds: ExecutiveStepId[];
};

export type TurnGoalPromotionServiceOptions = {
  model: string;
  identityService: Pick<IdentityService, "addGoal">;
  goalsRepository: Pick<GoalsRepository, "list">;
  executiveStepsRepository: Pick<ExecutiveStepsRepository, "add">;
  embeddingClient: EmbeddingClient;
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
  ownerEntityId?: EntityId | null;
  speakerDisplayName?: string | null;
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
      speakerEntityId: input.ownerEntityId ?? null,
      speakerDisplayName: input.speakerDisplayName ?? null,
      temporalCue: input.temporalCue,
      activeGoals: input.activeGoals.map((goal) => ({
        id: goal.id,
        description: goal.description,
        priority: goal.priority,
        target_at: goal.target_at,
        owner_entity_id: goal.owner_entity_id ?? null,
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
      ownerEntityId: input.ownerEntityId ?? null,
      persistedUserEntryId: input.persistedUserEntryId,
      turnId: input.turnId,
      onHookFailure: input.onHookFailure,
    });
  }

  private async persistGoalPromotions(input: {
    candidates: readonly GoalPromotionCandidate[];
    audienceEntityId: EntityId | null;
    ownerEntityId: EntityId | null;
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
    const activeSameAxisGoals = this.listActiveSameAxisGoals({
      audienceEntityId: input.audienceEntityId,
      ownerEntityId: input.ownerEntityId,
      turnId: input.turnId,
    });
    let embeddingDedupState:
      | {
          activeVectors: EmbeddedGoalVector[];
          acceptedVectors: EmbeddedGoalVector[];
        }
      | null
      | undefined;

    const getEmbeddingDedupState = async (): Promise<
      { activeVectors: EmbeddedGoalVector[]; acceptedVectors: EmbeddedGoalVector[] } | null
    > => {
      if (embeddingDedupState !== undefined) {
        return embeddingDedupState;
      }

      try {
        const embeddings =
          activeSameAxisGoals.length === 0
            ? []
            : await this.options.embeddingClient.embedBatch(
                activeSameAxisGoals.map((goal) => goal.description),
              );

        embeddingDedupState = {
          activeVectors: activeSameAxisGoals.flatMap((goal, index) => {
            const vector = embeddings[index];
            return vector === undefined ? [] : [{ goalId: goal.id, vector }];
          }),
          acceptedVectors: [],
        };
      } catch (error) {
        this.emitDedupDegraded({
          turnId: input.turnId,
          reason: "active_goal_embedding_failed",
          error,
        });
        embeddingDedupState = null;
      }

      return embeddingDedupState;
    };

    for (const candidate of input.candidates) {
      let goal: GoalRecord;
      let candidateVector: Float32Array | null = null;

      if (
        candidate.duplicate_of_goal_id !== null &&
        activeSameAxisGoals.some((goal) => goal.id === candidate.duplicate_of_goal_id)
      ) {
        this.emitSkippedAsDuplicate({
          turnId: input.turnId,
          candidateDescription: candidate.description,
          reason: "extractor_signal",
          duplicateOfGoalId: candidate.duplicate_of_goal_id,
        });
        continue;
      }

      const dedupState = await getEmbeddingDedupState();

      if (dedupState !== null) {
        const embeddingMatch = await this.findEmbeddingDuplicate({
          turnId: input.turnId,
          candidate,
          state: dedupState,
        });

        if (embeddingMatch.kind === "matched") {
          this.emitSkippedAsDuplicate({
            turnId: input.turnId,
            candidateDescription: candidate.description,
            reason: "embedding",
            matchedExistingId: embeddingMatch.goalId,
            similarity: embeddingMatch.similarity,
          });
          continue;
        }

        candidateVector = embeddingMatch.candidateVector;
      }

      try {
        goal = this.options.identityService.addGoal({
          description: candidate.description,
          priority: candidate.priority,
          status: "active",
          targetAt: candidate.target_at,
          audienceEntityId: input.audienceEntityId,
          ownerEntityId: input.ownerEntityId,
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

      if (candidateVector !== null && dedupState !== null) {
        dedupState.acceptedVectors.push({
          goalId: goal.id,
          vector: candidateVector,
        });
      }

      const initialStep = this.initialStepForPersistence({
        candidate,
        goal,
        turnId: input.turnId,
      });

      if (initialStep === null) {
        continue;
      }

      try {
        const step = this.options.executiveStepsRepository.add({
          goalId: goal.id,
          description: initialStep.description,
          kind: initialStep.kind,
          dueAt: initialStep.due_at,
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

  private listActiveSameAxisGoals(input: {
    audienceEntityId: EntityId | null;
    ownerEntityId: EntityId | null;
    turnId: string;
  }): GoalRecord[] {
    try {
      return flattenGoals(
        this.options.goalsRepository.list({
          status: "active",
          ownerEntityId: input.ownerEntityId,
        }),
      ).filter(
        (goal) =>
          goal.status === "active" &&
          goal.audience_entity_id === input.audienceEntityId &&
          (goal.owner_entity_id ?? null) === input.ownerEntityId,
      );
    } catch (error) {
      this.emitDedupDegraded({
        turnId: input.turnId,
        reason: "active_goal_lookup_failed",
        error,
      });
      return [];
    }
  }

  private async findEmbeddingDuplicate(input: {
    turnId: string;
    candidate: GoalPromotionCandidate;
    state: {
      activeVectors: EmbeddedGoalVector[];
      acceptedVectors: EmbeddedGoalVector[];
    };
  }): Promise<
    | {
        kind: "matched";
        goalId: GoalId;
        similarity: number;
      }
    | {
        kind: "clear";
        candidateVector: Float32Array | null;
      }
  > {
    let candidateVector: Float32Array;

    try {
      candidateVector = await this.options.embeddingClient.embed(input.candidate.description);
    } catch (error) {
      this.emitDedupDegraded({
        turnId: input.turnId,
        reason: "candidate_embedding_failed",
        error,
        candidateDescription: input.candidate.description,
      });
      return {
        kind: "clear",
        candidateVector: null,
      };
    }

    let bestMatch: { goalId: GoalId; similarity: number } | null = null;

    for (const existing of [...input.state.activeVectors, ...input.state.acceptedVectors]) {
      const similarity = cosineSimilarity(candidateVector, existing.vector);

      if (
        similarity >= GOAL_PROMOTION_DUPLICATE_SIMILARITY_THRESHOLD &&
        (bestMatch === null || similarity > bestMatch.similarity)
      ) {
        bestMatch = {
          goalId: existing.goalId,
          similarity,
        };
      }
    }

    if (bestMatch !== null) {
      return {
        kind: "matched",
        goalId: bestMatch.goalId,
        similarity: bestMatch.similarity,
      };
    }

    return {
      kind: "clear",
      candidateVector,
    };
  }

  private initialStepForPersistence(input: {
    candidate: GoalPromotionCandidate;
    goal: GoalRecord;
    turnId: string;
  }): GoalPromotionInitialStep | null {
    const initialStep = input.candidate.initial_step;

    if (initialStep === null) {
      return null;
    }

    if (initialStep.kind === "wait" && initialStep.due_at === null) {
      this.emitInitialStepDowngraded({
        turnId: input.turnId,
        goalId: input.goal.id,
        description: initialStep.description,
      });
      return null;
    }

    return initialStep;
  }

  private emitInitialStepDowngraded(input: {
    turnId: string;
    goalId: GoalId;
    description: string;
  }): void {
    if (!this.options.tracer.enabled) {
      return;
    }

    this.options.tracer.emit("extraction.goals.transitioned", {
      turnId: input.turnId,
      reason: "wait_without_due_at",
      goalId: input.goalId,
      ...(this.options.tracer.includePayloads ? { description: input.description } : {}),
    });
  }

  private emitSkippedAsDuplicate(input: {
    turnId: string;
    candidateDescription: string;
    reason: "extractor_signal" | "embedding";
    duplicateOfGoalId?: GoalId;
    matchedExistingId?: GoalId;
    similarity?: number;
  }): void {
    if (!this.options.tracer.enabled) {
      return;
    }

    this.options.tracer.emit("extraction.goals.skipped", {
      turnId: input.turnId,
      candidate_description: input.candidateDescription,
      reason: input.reason,
      ...(input.duplicateOfGoalId === undefined
        ? {}
        : { duplicate_of_goal_id: input.duplicateOfGoalId }),
      ...(input.matchedExistingId === undefined
        ? {}
        : { matched_existing_id: input.matchedExistingId }),
      ...(input.similarity === undefined ? {} : { similarity: input.similarity }),
    });
  }

  private emitDedupDegraded(input: {
    turnId: string;
    reason: string;
    error: unknown;
    candidateDescription?: string;
  }): void {
    if (!this.options.tracer.enabled) {
      return;
    }

    this.options.tracer.emit("extraction.goals.dedup.degraded", {
      turnId: input.turnId,
      reason: input.reason,
      error: input.error instanceof Error ? input.error.message : String(input.error),
      ...(input.candidateDescription === undefined
        ? {}
        : { candidate_description: input.candidateDescription }),
    });
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

    this.options.tracer.emit("extraction.goals.degraded", {
      turnId: input.turnId,
      reason: input.reason,
      ...(input.details ?? {}),
      ...(this.options.tracer.includePayloads && input.error !== undefined
        ? { error: input.error instanceof Error ? input.error.message : String(input.error) }
        : {}),
    });
  }
}

type GoalTreeNodeLike = GoalRecord & {
  children?: readonly GoalTreeNodeLike[];
};

type EmbeddedGoalVector = {
  goalId: GoalId;
  vector: Float32Array;
};

function flattenGoals(goals: readonly GoalTreeNodeLike[]): GoalRecord[] {
  const flattened: GoalRecord[] = [];
  const stack = [...goals];

  while (stack.length > 0) {
    const next = stack.shift();

    if (next === undefined) {
      continue;
    }

    flattened.push(next);
    stack.push(...(next.children ?? []));
  }

  return flattened;
}
