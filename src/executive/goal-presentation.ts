import { z } from "zod";
import type { GoalRecord } from "../memory/self/types.js";
import { goalBlockerSchema } from "../memory/self/goal-blocks.js";
import { memoryDisclosureLabelMetadataSchema } from "../memory/common/disclosure-label.js";
import type { StreamWatermarkRepository } from "../stream/watermark.js";
import { streamEntryIdSchema } from "../util/id-schemas.js";
import { DEFAULT_SESSION_ID, type GoalId, type StreamEntryId } from "../util/ids.js";

const INPUT_EVENT_LIMIT = 16;
const goalBlockEndSchema = z.discriminatedUnion("ended_by", [
  z.object({
    ended_by: z.literal("until_elapsed"),
    until: goalBlockerSchema.options[2].shape.until,
    blocked_at: z.number().finite(),
  }),
  z.object({
    ended_by: z.literal("entity_inbound"),
    entity_id: goalBlockerSchema.options[1].shape.entity_id,
    entry_id: streamEntryIdSchema,
  }),
  z.object({
    ended_by: z.literal("goal_terminal"),
    goal_id: goalBlockerSchema.options[0].shape.goal_id,
    status: z.enum(["done", "superseded", "abandoned"]),
  }),
  z.object({ ended_by: z.literal("manual") }),
]);
export type GoalBlockEnd = z.infer<typeof goalBlockEndSchema>;
const blockEndedFields = z.object({
  record_version: z.number().int().positive(),
  disclosure_label: memoryDisclosureLabelMetadataSchema,
});
const goalInputSchema = z.union([
  z.object({ kind: z.literal("associated_inbound"), entry_id: streamEntryIdSchema }),
  z.object({
    kind: z.literal("goal_write"),
    write: z.enum(["update", "update_status", "update_progress"]),
    record_version: z.number().int().positive(),
  }),
  blockEndedFields.extend({ kind: z.literal("block_ended") }).and(goalBlockEndSchema),
]);
type GoalInput = z.infer<typeof goalInputSchema>;
export type GoalWriteKind = Extract<GoalInput, { kind: "goal_write" }>["write"];
const goalInputEventSchema = goalInputSchema.and(
  z.object({ generation: z.number().int().positive() }),
);
export type GoalInputEvent = z.infer<typeof goalInputEventSchema>;

export type GoalAdmission = {
  basis: "first_presentation" | "new_input";
  new_inputs: GoalInputEvent[];
  unitemized_input_count: number;
  previous_progress_anchor?: number;
  current_progress_anchor?: number;
};

export type GoalPresentationEvaluation = {
  wasPresented: boolean;
  hasNewInput: boolean;
  admission: GoalAdmission;
};

const stateSchema = z.object({
  version: z.literal(1),
  inputGeneration: z.number().int().nonnegative(),
  lastProgressVersion: z.number().int().nonnegative().nullable(),
  inputEvents: z.array(goalInputEventSchema).max(INPUT_EVENT_LIMIT).default([]),
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
  constructor(
    private readonly watermarks: StreamWatermarkRepository,
    private readonly onAssociatedInput?: (goalId: GoalId) => void,
  ) {}

  private executiveKey(goalId: GoalId): string {
    return `autonomy:goal-presentation:executive:${goalId}`;
  }

  wasExecutivePresented(goal: GoalRecord): boolean {
    return this.evaluate(goal, "executive").wasPresented;
  }

  hasNewExecutiveInput(goal: GoalRecord): boolean {
    return this.evaluate(goal, "executive").hasNewInput;
  }

  private read(goalId: GoalId): z.infer<typeof stateSchema> {
    const row = this.watermarks.get(`autonomy:goal-presentation:${goalId}`, DEFAULT_SESSION_ID);
    return row === null
      ? {
          version: 1,
          inputGeneration: 0,
          lastProgressVersion: null,
          inputEvents: [],
          presented: null,
        }
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
    return this.evaluate(goal, "followup").wasPresented;
  }

  hasNewInput(goal: GoalRecord): boolean {
    return this.evaluate(goal, "followup").hasNewInput;
  }

  admission(
    goal: Pick<GoalRecord, "id" | "last_progress_ts" | "created_at">,
    lane: "executive" | "followup",
  ): GoalAdmission {
    return this.evaluate(goal, lane).admission;
  }

  /** The latch and its explanation share one captured presentation state. */
  evaluate(
    goal: Pick<GoalRecord, "id" | "last_progress_ts" | "created_at">,
    lane: "executive" | "followup",
  ): GoalPresentationEvaluation {
    const state = this.read(goal.id);
    const currentAnchor = goal.last_progress_ts ?? goal.created_at;
    const executiveRow =
      lane === "executive"
        ? this.watermarks.get(this.executiveKey(goal.id), DEFAULT_SESSION_ID)
        : null;
    const prior =
      lane === "followup"
        ? state.presented
        : executiveRow === null
          ? null
          : stateSchema.shape.presented.unwrap().parse(executiveRow.metadata);
    const previousGeneration = prior?.inputGeneration ?? 0;
    const newInputs = state.inputEvents.filter((event) => event.generation > previousGeneration);
    const wasPresented =
      prior !== null &&
      prior.inputGeneration === state.inputGeneration &&
      prior.progressAnchor === currentAnchor;
    return {
      wasPresented,
      hasNewInput: prior === null ? state.inputGeneration > 0 : !wasPresented,
      admission: {
        basis: prior === null && state.inputGeneration === 0 ? "first_presentation" : "new_input",
        new_inputs: newInputs,
        unitemized_input_count: Math.max(
          0,
          state.inputGeneration - previousGeneration - newInputs.length,
        ),
        ...(prior !== null && prior.progressAnchor !== currentAnchor
          ? {
              previous_progress_anchor: prior.progressAnchor,
              current_progress_anchor: currentAnchor,
            }
          : {}),
      },
    };
  }

  private recordInput(goalId: GoalId, input: GoalInput, recordVersion?: number): number {
    const state = this.read(goalId);
    const generation = state.inputGeneration + 1;
    // The successful goal CAS and this increment share a transaction; failed
    // CAS attempts cannot advance the input generation.
    this.write(goalId, {
      ...state,
      inputGeneration: generation,
      lastProgressVersion: recordVersion ?? state.lastProgressVersion,
      inputEvents: [...state.inputEvents, { ...input, generation }].slice(-INPUT_EVENT_LIMIT),
    });
    return generation;
  }

  goalWritten(goalId: GoalId, recordVersion: number, write: GoalWriteKind): void {
    this.recordInput(
      goalId,
      { kind: "goal_write", write, record_version: recordVersion },
      recordVersion,
    );
  }

  blockEnded(goalId: GoalId, input: GoalBlockEnd & z.infer<typeof blockEndedFields>): void {
    this.recordInput(goalId, { kind: "block_ended", ...input }, input.record_version);
  }

  associateInbound(goalId: GoalId, entryId: StreamEntryId): void {
    // One durable idempotency handle per explicit model association. Replays
    // of older turns cannot become new input after a later presentation.
    this.watermarks.runInTransaction(() => {
      const key = `autonomy:goal-associated-input:${goalId}:${entryId}`;
      if (this.watermarks.get(key, DEFAULT_SESSION_ID) !== null) return;
      this.onAssociatedInput?.(goalId);
      const generation = this.recordInput(goalId, {
        kind: "associated_inbound",
        entry_id: entryId,
      });
      this.watermarks.set(key, DEFAULT_SESSION_ID, {
        lastTs: generation,
        lastEntryId: entryId,
      });
    });
  }

  presented(
    presentations: readonly GoalPresentation[],
    turnId: string,
    executiveGoalIds: ReadonlySet<string> = new Set(),
  ): void {
    this.watermarks.runInTransaction(() => {
      for (const presentation of presentations) {
        const state = this.read(presentation.goalId);
        if (executiveGoalIds.has(presentation.goalId)) {
          const key = this.executiveKey(presentation.goalId);
          const row = this.watermarks.get(key, DEFAULT_SESSION_ID);
          const prior =
            row === null ? null : stateSchema.shape.presented.unwrap().parse(row.metadata);
          // S2 planner and finalizer acknowledge the same presentation once.
          if (
            prior?.turnId !== turnId &&
            (prior === null ||
              prior.inputGeneration < presentation.inputGeneration ||
              (prior.inputGeneration === presentation.inputGeneration &&
                prior.progressAnchor <= presentation.progressAnchor))
          ) {
            this.watermarks.set(key, DEFAULT_SESSION_ID, {
              lastTs: presentation.inputGeneration,
              lastEntryId: presentation.goalId,
              metadata: {
                progressAnchor: presentation.progressAnchor,
                inputGeneration: presentation.inputGeneration,
                turnId,
              },
            });
          }
        }
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
