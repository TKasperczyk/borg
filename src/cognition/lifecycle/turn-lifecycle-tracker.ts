import type { ExecutiveStep, ExecutiveStepsRepository } from "../../executive/index.js";
import type { ActionRepository } from "../../memory/actions/index.js";
import type {
  EpisodicRepository,
  EpisodeStats,
  EpisodeStatsPatch,
} from "../../memory/episodic/index.js";
import type {
  RelationalSlot,
  RelationalSlotRepository,
} from "../../memory/relational-slots/index.js";
import {
  GOAL_TURN_ROLLBACK_REASON,
  type GoalRecord,
  type GoalsRepository,
  type OpenQuestion,
  type OpenQuestionsRepository,
} from "../../memory/self/index.js";
import type { WorkingMemory, WorkingMemoryStore } from "../../memory/working/index.js";
import type {
  ActionId,
  ExecutiveStepId,
  GoalId,
  OpenQuestionId,
  SessionId,
} from "../../util/ids.js";
import type { ReflectionEffects } from "../reflection/index.js";
import type { TurnTracer } from "../../tracing/tracer.js";

export type TurnLifecycleTrackerOptions = {
  workingMemoryStore: Pick<WorkingMemoryStore, "recordPendingActionMerges" | "load" | "save">;
  actionRepository: Pick<ActionRepository, "delete">;
  executiveStepsRepository: Pick<ExecutiveStepsRepository, "delete" | "restore">;
  goalsRepository: Pick<GoalsRepository, "remove" | "restore">;
  openQuestionsRepository: Pick<OpenQuestionsRepository, "delete" | "restore">;
  episodicRepository: Pick<EpisodicRepository, "updateStats">;
  relationalSlotRepository: Pick<RelationalSlotRepository, "restore">;
  tracer?: TurnTracer;
};

function episodeStatsRestorePatch(stats: EpisodeStats): EpisodeStatsPatch {
  return {
    retrieval_count: stats.retrieval_count,
    use_count: stats.use_count,
    last_retrieved: stats.last_retrieved,
    win_rate: stats.win_rate,
    tier: stats.tier,
    promoted_at: stats.promoted_at,
    promoted_from: stats.promoted_from,
    gist: stats.gist,
    gist_generated_at: stats.gist_generated_at,
    last_decayed_at: stats.last_decayed_at,
    heat_multiplier: stats.heat_multiplier,
    valence_mean: stats.valence_mean,
  };
}

export type AbortCleanupFailure = {
  operation: string;
  id: string;
  error: string;
};

function formatCleanupError(error: unknown): string {
  if (error instanceof Error) {
    return `${error.name}: ${error.message}`;
  }

  return String(error);
}

export class TurnLifecycleTracker {
  private initialWorkingMemory: WorkingMemory | null = null;
  private readonly createdGoalIds: GoalId[] = [];
  private readonly createdExecutiveStepIds: ExecutiveStepId[] = [];
  private readonly createdActionIds: ActionId[] = [];
  private readonly createdOpenQuestionIds: OpenQuestionId[] = [];
  private readonly updatedExecutiveSteps: ExecutiveStep[] = [];
  private readonly updatedGoals: GoalRecord[] = [];
  private readonly resolvedOpenQuestions: OpenQuestion[] = [];
  private readonly updatedEpisodeStats: EpisodeStats[] = [];
  private readonly appliedSlotNegations: RelationalSlot[] = [];
  private pendingActionMergeCount = 0;
  private turnCommitted = false;

  constructor(private readonly options: TurnLifecycleTrackerOptions) {}

  captureInitialWorkingMemory(workingMemory: WorkingMemory): void {
    this.initialWorkingMemory = structuredClone(workingMemory);
  }

  trackCreatedGoalIds(goalIds: readonly GoalId[]): void {
    this.createdGoalIds.push(...goalIds);
  }

  trackCreatedExecutiveStepIds(stepIds: readonly ExecutiveStepId[]): void {
    this.createdExecutiveStepIds.push(...stepIds);
  }

  trackCreatedActionIds(actionIds: readonly ActionId[]): void {
    this.createdActionIds.push(...actionIds);
  }

  trackAppliedSlotNegation(slot: RelationalSlot): void {
    this.appliedSlotNegations.push(slot);
  }

  trackPendingActionMerges(count: number): void {
    if (count <= 0) {
      return;
    }

    this.pendingActionMergeCount += Math.floor(count);
  }

  trackReflectionEffects(effects: ReflectionEffects): void {
    this.createdActionIds.push(...effects.createdActionIds);
    this.createdExecutiveStepIds.push(...effects.createdExecutiveStepIds);
    this.createdOpenQuestionIds.push(...effects.createdOpenQuestionIds);
    this.updatedExecutiveSteps.push(...effects.updatedExecutiveSteps);
    this.updatedGoals.push(...effects.updatedGoals);
    this.resolvedOpenQuestions.push(...effects.resolvedOpenQuestions);
    this.updatedEpisodeStats.push(...effects.updatedEpisodeStats);
  }

  private async bestEffort<T>(
    failures: AbortCleanupFailure[],
    operation: string,
    items: readonly T[],
    idForItem: (item: T) => string,
    run: (item: T) => unknown | Promise<unknown>,
  ): Promise<void> {
    for (const item of items) {
      try {
        await run(item);
      } catch (error) {
        failures.push({
          operation,
          id: idForItem(item),
          error: formatCleanupError(error),
        });
      }
    }
  }

  private traceAbortCleanupFailures(input: {
    turnId: string;
    sessionId?: SessionId;
    failures: readonly AbortCleanupFailure[];
  }): void {
    if (this.options.tracer?.enabled !== true || input.failures.length === 0) {
      return;
    }

    try {
      this.options.tracer.emit("turn.rollback_incomplete", {
        turnId: input.turnId,
        turn_id: input.turnId,
        ...(input.sessionId === undefined ? {} : { session_id: input.sessionId }),
        failure_count: input.failures.length,
        failures: input.failures.map((failure) => ({
          operation: failure.operation,
          id: failure.id,
          error: failure.error,
        })),
      });
    } catch {
      // Best-effort observability must never mask the original turn failure.
    }
  }

  async cleanupAbortedTurnState(input: {
    turnId: string;
    sessionId?: SessionId;
  }): Promise<AbortCleanupFailure[]> {
    const failures: AbortCleanupFailure[] = [];
    if (this.turnCommitted) return failures;
    const sessionId = input.sessionId ?? this.initialWorkingMemory?.session_id;
    const persisted: WorkingMemory[] = [];
    await this.bestEffort(
      failures,
      "read_working_memory",
      sessionId === undefined ? [] : [sessionId],
      (id) => id,
      (id) => persisted.push(this.options.workingMemoryStore.load(id)),
    );
    if (failures.length > 0) {
      this.traceAbortCleanupFailures({ ...input, failures });
      return failures;
    }
    if (
      persisted.some((state) =>
        state.pending_actions.some(
          (action) =>
            action.completing_action?.turn_id === input.turnId &&
            action.completing_action.phase === "committed",
        ),
      )
    )
      return failures;

    // Reflection retains local pending rows with durable completion markers.
    // Restore the rest of working memory (including pre-action merge/cap changes)
    // while switching this turn's markers to rollback before deleting actions.
    const restoredWorkingMemory: WorkingMemory[] = [];
    await this.bestEffort(
      failures,
      "restore_working_memory",
      this.initialWorkingMemory === null ? [] : [this.initialWorkingMemory],
      (workingMemory) => input.sessionId ?? workingMemory.session_id,
      (workingMemory) => {
        restoredWorkingMemory.push(
          this.options.workingMemoryStore.save(workingMemory, {
            pendingActionTurn: { turnId: input.turnId, outcome: "rollback" },
          }),
        );
      },
    );

    // A failed write leaves the forward marker and row intact for locked recovery.
    await this.bestEffort(
      failures,
      "delete_action",
      this.initialWorkingMemory === null || restoredWorkingMemory.length > 0
        ? this.createdActionIds
        : [],
      (actionId) => actionId,
      (actionId) => this.options.actionRepository.delete(actionId),
    );

    // Clear markers only after deletion; failed deletes keep their durable marker
    // for recovery under a subsequent holder of the origin session lock.
    await this.bestEffort(
      failures,
      "clear_pending_rollback_markers",
      this.createdActionIds.length === 0 ? [] : restoredWorkingMemory,
      (workingMemory) => workingMemory.session_id,
      (workingMemory) => this.options.workingMemoryStore.save(workingMemory),
    );

    await this.bestEffort(
      failures,
      "delete_executive_step",
      this.createdExecutiveStepIds,
      (stepId) => stepId,
      (stepId) => this.options.executiveStepsRepository.delete(stepId),
    );

    // Abort cleanup only touches goals created during the turn being rolled
    // back; no committed caller has observed a stable version to race here.
    await this.bestEffort(
      failures,
      "remove_goal",
      this.createdGoalIds,
      (goalId) => goalId,
      (goalId) =>
        this.options.goalsRepository.remove(goalId, {
          auditContext: {
            reason: GOAL_TURN_ROLLBACK_REASON,
            provenance: { kind: "system" },
          },
        }),
    );

    await this.bestEffort(
      failures,
      "delete_open_question",
      this.createdOpenQuestionIds,
      (openQuestionId) => openQuestionId,
      (openQuestionId) => this.options.openQuestionsRepository.delete(openQuestionId),
    );

    await this.bestEffort(
      failures,
      "restore_executive_step",
      [...this.updatedExecutiveSteps].reverse(),
      (step) => step.id,
      (step) => this.options.executiveStepsRepository.restore(step),
    );

    await this.bestEffort(
      failures,
      "restore_goal",
      [...this.updatedGoals].reverse(),
      (goal) => goal.id,
      (goal) => this.options.goalsRepository.restore(goal),
    );

    await this.bestEffort(
      failures,
      "restore_open_question",
      [...this.resolvedOpenQuestions].reverse(),
      (question) => question.id,
      (question) => this.options.openQuestionsRepository.restore(question),
    );

    await this.bestEffort(
      failures,
      "restore_episode_stats",
      [...this.updatedEpisodeStats].reverse(),
      (stats) => stats.episode_id,
      (stats) =>
        this.options.episodicRepository.updateStats(
          stats.episode_id,
          episodeStatsRestorePatch(stats),
        ),
    );

    await this.bestEffort(
      failures,
      "restore_relational_slot",
      [...this.appliedSlotNegations].reverse(),
      (slot) => slot.id,
      (slot) => this.options.relationalSlotRepository.restore(slot),
    );

    this.traceAbortCleanupFailures({
      turnId: input.turnId,
      sessionId: input.sessionId,
      failures,
    });

    return failures;
  }

  commitTurnState(input: { turnId: string; sessionId: SessionId }): void {
    const store = this.options.workingMemoryStore;
    const current = store.load(input.sessionId);
    const pending = current.pending_actions.filter(
      (action) => action.completing_action?.turn_id === input.turnId,
    );
    if (pending.length > 0) {
      let committed: WorkingMemory;
      try {
        // The durable decision precedes cleanup, so a crash cannot turn a
        // committed completion back into an orphaned provisional action.
        committed = store.save(current, {
          pendingActionTurn: { turnId: input.turnId, outcome: "commit" },
        });
      } catch (error) {
        // A writer/trace callback may throw after the atomic write succeeded.
        // Only the persisted decision can tell us whether rollback is still safe.
        committed = store.load(input.sessionId);
        if (
          !pending.every((action) =>
            committed.pending_actions.some(
              (candidate) =>
                candidate.pending_action_id === action.pending_action_id &&
                candidate.completing_action?.turn_id === input.turnId &&
                candidate.completing_action.phase === "committed",
            ),
          )
        )
          throw error;
      }
      this.turnCommitted = true;
      try {
        store.save(committed);
      } catch (error) {
        // Commit has happened. Retain the action and let the next locked write
        // finish row cleanup; this failure must not enter turn rollback.
        try {
          this.options.tracer?.emit("working_memory.degraded", {
            turnId: input.turnId,
            session_id: input.sessionId,
            operation: "pending_action_commit_cleanup",
            reason: "WORKING_MEMORY_SAVE_FAILED",
          });
        } catch {
          // Observability failure cannot undo a durable commit decision.
        }
      }
    }
    this.turnCommitted = true;
    this.options.workingMemoryStore.recordPendingActionMerges(this.pendingActionMergeCount);
  }
}
