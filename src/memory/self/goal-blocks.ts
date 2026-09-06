import { z } from "zod";

import { artifactReferenceSchema } from "../common/artifact-reference.js";
import {
  memoryDisclosureLabelMetadata,
  memoryDisclosureLabelMetadataSchema,
  unknownMemoryDisclosureLabel,
} from "../common/disclosure-label.js";
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

const blockHistoryFields = {
  unblocked_at: z.number().finite().nullable(),
  unblock_reason: z.string().nullable(),
  disclosure_label: memoryDisclosureLabelMetadataSchema.default(() =>
    memoryDisclosureLabelMetadata(unknownMemoryDisclosureLabel()),
  ),
};

export const goalBlockRecordSchema = z.union([
  goalBlockInputSchema.extend({
    ...blockHistoryFields,
    blocked_at: z.number().finite(),
  }),
  z.object({
    ...blockHistoryFields,
    blocker: z.object({ kind: z.literal("legacy_unknown") }).strict(),
    attempt_status: z.literal("not_recorded"),
    reason: z.string(),
    attempt_evidence: artifactReferenceSchema.optional(),
    blocked_at: z.null(),
  }),
]);

export type GoalBlockInput = z.infer<typeof goalBlockInputSchema>;
export type GoalBlockRecord = z.infer<typeof goalBlockRecordSchema>;

export function legacyUnknownGoalBlock(): GoalBlockRecord {
  return goalBlockRecordSchema.parse({
    blocker: { kind: "legacy_unknown" },
    attempt_status: "not_recorded",
    reason: "legacy, blocker not recorded; attempt and block time not recorded",
    disclosure_label: memoryDisclosureLabelMetadata(unknownMemoryDisclosureLabel()),
    blocked_at: null,
    unblocked_at: null,
    unblock_reason: null,
  });
}

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
    if (block.unblocked_at === null || block.blocked_at === null) continue;
    if (goal.target_assigned_at != null) {
      deadlinePause += Math.max(
        0,
        block.unblocked_at - Math.max(goal.target_assigned_at, block.blocked_at),
      );
    }
    progressPause += Math.max(0, block.unblocked_at - Math.max(anchor, block.blocked_at));
  }
  return {
    progressAnchor: anchor + progressPause,
    targetAt: goal.target_at === null ? null : goal.target_at + deadlinePause,
  };
}

export function goalBlockStateFields(
  goal: Pick<GoalRecord, "block_history" | "target_assigned_at">,
) {
  // Older custom snapshots can predate persisted labels; missing source policy
  // is unknown, never inherited from the enclosing goal or used to filter recall.
  const history = (goal.block_history ?? []).map((block) => goalBlockRecordSchema.parse(block));
  return {
    block: currentGoalBlock({ block_history: history }),
    block_history: history,
    ...(history.length === 0 ? {} : { target_assigned_at: goal.target_assigned_at ?? null }),
  };
}

export function goalBlockStateText(
  goal: Pick<GoalRecord, "block_history" | "target_assigned_at">,
): string {
  return JSON.stringify(goalBlockStateFields(goal));
}
