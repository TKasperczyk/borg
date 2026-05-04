import type { ExecutiveStep, ExecutiveStepsRepository } from "../../executive/index.js";
import type { ActionRepository } from "../../memory/actions/index.js";
import type { EpisodicRepository, EpisodeStats } from "../../memory/episodic/index.js";
import type {
  RelationalSlot,
  RelationalSlotRepository,
} from "../../memory/relational-slots/index.js";
import type {
  GoalRecord,
  GoalsRepository,
  OpenQuestion,
  OpenQuestionsRepository,
} from "../../memory/self/index.js";
import type { WorkingMemory, WorkingMemoryStore } from "../../memory/working/index.js";
import type { ActionId, ExecutiveStepId, GoalId, OpenQuestionId } from "../../util/ids.js";
import type { ReflectionEffects } from "../reflection/index.js";

export type TurnLifecycleTrackerOptions = {
  workingMemoryStore: Pick<WorkingMemoryStore, "recordPendingActionMerges" | "save">;
  actionRepository: Pick<ActionRepository, "delete">;
  executiveStepsRepository: Pick<ExecutiveStepsRepository, "delete" | "restore">;
  goalsRepository: Pick<GoalsRepository, "remove" | "restore">;
  openQuestionsRepository: Pick<OpenQuestionsRepository, "delete" | "restore">;
  episodicRepository: Pick<EpisodicRepository, "updateStats">;
  relationalSlotRepository: Pick<RelationalSlotRepository, "restore">;
};

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

  async cleanupAbortedTurnState(): Promise<void> {
    if (this.initialWorkingMemory !== null) {
      this.options.workingMemoryStore.save(this.initialWorkingMemory);
    }

    for (const actionId of this.createdActionIds) {
      try {
        await this.options.actionRepository.delete(actionId);
      } catch {
        // Best effort; the abort marker still prevents stream-derived state reuse.
      }
    }

    for (const stepId of this.createdExecutiveStepIds) {
      try {
        this.options.executiveStepsRepository.delete(stepId);
      } catch {
        // Best effort.
      }
    }

    for (const goalId of this.createdGoalIds) {
      try {
        this.options.goalsRepository.remove(goalId);
      } catch {
        // Best effort.
      }
    }

    for (const openQuestionId of this.createdOpenQuestionIds) {
      try {
        await this.options.openQuestionsRepository.delete(openQuestionId);
      } catch {
        // Best effort.
      }
    }

    for (const step of [...this.updatedExecutiveSteps].reverse()) {
      try {
        this.options.executiveStepsRepository.restore(step);
      } catch {
        // Best effort.
      }
    }

    for (const goal of [...this.updatedGoals].reverse()) {
      try {
        this.options.goalsRepository.restore(goal);
      } catch {
        // Best effort.
      }
    }

    for (const question of [...this.resolvedOpenQuestions].reverse()) {
      try {
        this.options.openQuestionsRepository.restore(question);
      } catch {
        // Best effort.
      }
    }

    for (const stats of [...this.updatedEpisodeStats].reverse()) {
      try {
        this.options.episodicRepository.updateStats(stats.episode_id, stats);
      } catch {
        // Best effort.
      }
    }

    for (const slot of [...this.appliedSlotNegations].reverse()) {
      try {
        this.options.relationalSlotRepository.restore(slot);
      } catch {
        // Best effort.
      }
    }
  }

  commitTurnState(): void {
    this.options.workingMemoryStore.recordPendingActionMerges(this.pendingActionMergeCount);
  }
}
