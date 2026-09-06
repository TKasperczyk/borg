import { z } from "zod";

import { artifactReferenceSchema } from "../common/artifact-reference.js";
import { entityIdHelpers, goalIdHelpers, type EntityId, type GoalId } from "../../util/ids.js";
import type { GoalRecord } from "./types.js";

export const goalBlockerSchema = z.discriminatedUnion("kind", [
  z
    .object({
      kind: z.literal("goal"),
      goal_id: z
        .string()
        .refine(goalIdHelpers.is)
        .transform((id) => id as GoalId),
    })
    .strict(),
  z
    .object({
      kind: z.literal("entity"),
      entity_id: z
        .string()
        .refine(entityIdHelpers.is)
        .transform((id) => id as EntityId),
    })
    .strict(),
  z
    .object({
      kind: z.literal("until"),
      until: z.number().int().nonnegative().max(8_640_000_000_000_000),
    })
    .strict(),
]);

export const goalBlockInputSchema = z
  .object({
    blocker: goalBlockerSchema,
    attempt_status: z.literal("attempted_unavailable"),
    reason: z.string().trim().min(1),
    attempt_evidence: artifactReferenceSchema.optional(),
  })
  .strict();

export const goalBlockRecordSchema = goalBlockInputSchema.extend({
  blocked_at: z.number().finite(),
  unblocked_at: z.number().finite().nullable(),
  unblock_reason: z.string().nullable(),
});

export type GoalBlockInput = z.infer<typeof goalBlockInputSchema>;
export type GoalBlockRecord = z.infer<typeof goalBlockRecordSchema>;

export function currentGoalBlock(goal: Pick<GoalRecord, "block_history">): GoalBlockRecord | null {
  const last = goal.block_history?.at(-1);
  return last?.unblocked_at === null ? last : null;
}

/** Pause scheduling clocks without rewriting the dates of real progress or deadlines. */
export function goalSchedulingTimes(goal: GoalRecord): {
  progressAnchor: number;
  targetAt: number | null;
} {
  const anchor = goal.last_progress_ts ?? goal.created_at;
  let progressPause = 0;
  let deadlinePause = 0;
  for (const block of goal.block_history ?? []) {
    if (block.unblocked_at === null) continue;
    deadlinePause += Math.max(0, block.unblocked_at - block.blocked_at);
    progressPause += Math.max(0, block.unblocked_at - Math.max(anchor, block.blocked_at));
  }
  return {
    progressAnchor: anchor + progressPause,
    targetAt: goal.target_at === null ? null : goal.target_at + deadlinePause,
  };
}

export function goalBlockStateFields(goal: Pick<GoalRecord, "block_history">) {
  return { block: currentGoalBlock(goal), block_history: goal.block_history ?? [] };
}

export function goalBlockStateText(goal: Pick<GoalRecord, "block_history">): string {
  return JSON.stringify(goalBlockStateFields(goal));
}
