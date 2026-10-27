import { z } from "zod";
import type { GoalRecord } from "../memory/self/types.js";
import type { StreamWatermarkRepository } from "../stream/watermark.js";
import { DEFAULT_SESSION_ID, type GoalId, type StreamEntryId } from "../util/ids.js";

const stateSchema = z.object({
  version: z.literal(1),
  inputGeneration: z.number().int().nonnegative(),
  lastProgressVersion: z.number().int().nonnegative().nullable(),
  presented: z
    .object({
      progressAnchor: z.number(),
      inputGeneration: z.number().int().nonnegative(),
      turnId: z.string(),
    })
    .nullable(),
});

export type GoalPresentation = {
  goalId: GoalId;
  progressAnchor: number;
  inputGeneration: number;
};

/** Global scheduling state, independent of recall/disclosure and wake session. */
export class GoalPresentationStore {
  constructor(private readonly watermarks: StreamWatermarkRepository) {}

  private read(goalId: GoalId): z.infer<typeof stateSchema> {
    const row = this.watermarks.get(`autonomy:goal-presentation:${goalId}`, DEFAULT_SESSION_ID);
    return row === null
      ? { version: 1, inputGeneration: 0, lastProgressVersion: null, presented: null }
      : stateSchema.parse(row.metadata);
  }

  private write(goalId: GoalId, state: z.infer<typeof stateSchema>): void {
    this.watermarks.set(`autonomy:goal-presentation:${goalId}`, DEFAULT_SESSION_ID, {
      lastTs: state.inputGeneration,
      lastEntryId: goalId,
      metadata: state,
    });
  }

  snapshot(goal: Pick<GoalRecord, "id" | "last_progress_ts" | "created_at">): GoalPresentation {
    return {
      goalId: goal.id,
      progressAnchor: goal.last_progress_ts ?? goal.created_at,
      inputGeneration: this.read(goal.id).inputGeneration,
    };
  }

  wasPresented(goal: GoalRecord): boolean {
    const current = this.snapshot(goal);
    const presented = this.read(goal.id).presented;
    return (
      presented !== null &&
      presented.inputGeneration === current.inputGeneration &&
      presented.progressAnchor === current.progressAnchor
    );
  }

  hasNewInput(goal: GoalRecord): boolean {
    const state = this.read(goal.id);
    // A new association also supersedes legacy timestamp/phase latches before
    // the first recorded presentation of this goal.
    return state.presented === null ? state.inputGeneration > 0 : !this.wasPresented(goal);
  }

  progressWritten(goalId: GoalId, recordVersion: number): void {
    const state = this.read(goalId);
    // The successful goal CAS and this increment share a transaction; failed
    // CAS attempts cannot advance the input generation.
    this.write(goalId, {
      ...state,
      inputGeneration: state.inputGeneration + 1,
      lastProgressVersion: recordVersion,
    });
  }

  associateInbound(goalId: GoalId, entryId: StreamEntryId): void {
    // One durable idempotency handle per explicit model association. Replays
    // of older turns cannot become new input after a later presentation.
    this.watermarks.runInTransaction(() => {
      const key = `autonomy:goal-associated-input:${goalId}:${entryId}`;
      if (this.watermarks.get(key, DEFAULT_SESSION_ID) !== null) return;
      const state = this.read(goalId);
      this.write(goalId, { ...state, inputGeneration: state.inputGeneration + 1 });
      this.watermarks.set(key, DEFAULT_SESSION_ID, {
        lastTs: state.inputGeneration + 1,
        lastEntryId: entryId,
      });
    });
  }

  presented(presentations: readonly GoalPresentation[], turnId: string): void {
    this.watermarks.runInTransaction(() => {
      for (const presentation of presentations) {
        const state = this.read(presentation.goalId);
        const prior = state.presented;
        if (
          prior !== null &&
          (prior.inputGeneration > presentation.inputGeneration ||
            (prior.inputGeneration === presentation.inputGeneration &&
              prior.progressAnchor >= presentation.progressAnchor))
        )
          continue;
        this.write(presentation.goalId, {
          ...state,
          presented: {
            progressAnchor: presentation.progressAnchor,
            inputGeneration: presentation.inputGeneration,
            turnId,
          },
        });
      }
    });
  }
}
