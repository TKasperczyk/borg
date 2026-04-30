import { z } from "zod";

import { episodeIdSchema } from "../../episodic/index.js";
import { skillIdHelpers, type SkillId } from "../../../util/ids.js";
import {
  type ReviewQueueHandler,
  type ReviewQueueItem,
  type ReviewResolution,
} from "../review-queue.js";

const SKILL_SPLIT_REVIEW_RESOLUTIONS = new Set<ReviewResolution>(["accept", "reject"]);

const reviewSkillIdSchema = z
  .string()
  .refine((value) => skillIdHelpers.is(value), {
    message: "Invalid skill id",
  })
  .transform((value) => value as SkillId);

const skillSplitContextStatsSchema = z
  .object({
    skill_id: reviewSkillIdSchema,
    context_key: z.string().min(1),
    alpha: z.number().positive(),
    beta: z.number().positive(),
    attempts: z.number().int().nonnegative(),
    successes: z.number().int().nonnegative(),
    failures: z.number().int().nonnegative(),
    last_used: z.number().finite().nullable(),
    last_successful: z.number().finite().nullable(),
    updated_at: z.number().finite(),
  })
  .strict();

const skillSplitChildSpecSchema = z
  .object({
    label: z.string().min(1),
    problem: z.string().min(1),
    approach: z.string().min(1),
    context_stats: z.array(skillSplitContextStatsSchema).min(1),
  })
  .strict();

const skillSplitEvidenceBucketSchema = z
  .object({
    context_key: z.string().min(1),
    posterior_mean: z.number().min(0).max(1),
    alpha: z.number().positive(),
    beta: z.number().positive(),
    attempts: z.number().int().nonnegative(),
    successes: z.number().int().nonnegative(),
    failures: z.number().int().nonnegative(),
    last_used: z.number().finite().nullable(),
    last_successful: z.number().finite().nullable(),
  })
  .strict();

export const skillSplitReviewPayloadSchema = z
  .object({
    target_type: z.literal("skill"),
    target_id: reviewSkillIdSchema,
    original_skill_id: reviewSkillIdSchema,
    proposed_children: z.array(skillSplitChildSpecSchema).min(2),
    rationale: z.string().min(1),
    evidence_summary: z
      .object({
        source_episode_ids: z.array(episodeIdSchema).min(1),
        divergence: z.number().min(0).max(1),
        min_posterior_mean: z.number().min(0).max(1),
        max_posterior_mean: z.number().min(0).max(1),
        buckets: z.array(skillSplitEvidenceBucketSchema).min(2),
      })
      .strict(),
    cooldown: z
      .object({
        proposed_at: z.number().finite(),
        claimed_at: z.number().finite().nullable(),
        claim_expires_at: z.number().finite().nullable(),
        split_cooldown_days: z.number().positive(),
        split_claim_stale_sec: z.number().int().positive(),
        last_split_attempt_at: z.number().finite().nullable(),
        split_failure_count: z.number().int().nonnegative(),
        last_split_error: z.string().min(1).nullable(),
      })
      .strict(),
  })
  .strict();

export type SkillSplitReviewPayload = z.infer<typeof skillSplitReviewPayloadSchema>;
export type SkillSplitReviewApplyResult =
  | {
      status: "applied";
      newSkillIds: SkillId[];
    }
  | {
      status: "rejected";
      reason: string;
    };
export type SkillSplitReviewHandler = {
  accept: (
    item: ReviewQueueItem,
    payload: SkillSplitReviewPayload,
  ) => Promise<SkillSplitReviewApplyResult> | SkillSplitReviewApplyResult;
  reject: (
    item: ReviewQueueItem,
    payload: SkillSplitReviewPayload,
    reason: string,
  ) => Promise<void> | void;
};

export function createSkillSplitReviewQueueHandler(
  handler: SkillSplitReviewHandler,
): ReviewQueueHandler<"skill_split", SkillSplitReviewPayload> {
  return {
    kind: "skill_split",
    refsSchema: skillSplitReviewPayloadSchema,
    allowedResolutions: SKILL_SPLIT_REVIEW_RESOLUTIONS,
    transactionScope: () => "external",
    async apply({ item, refs, resolution, ctx }) {
      const resolvedAt = ctx.clock.now();

      if (resolution.decision === "accept") {
        const result = await handler.accept(item, refs);

        if (result.status === "applied") {
          return {
            refs: {
              ...item.refs,
              review_resolution: {
                decision: "accept",
                applied_at: resolvedAt,
                new_skill_ids: result.newSkillIds,
              },
            },
          };
        }

        return {
          finalResolution: {
            decision: "reject",
            reason: result.reason,
          },
          refs: {
            ...item.refs,
            review_resolution: {
              decision: "reject",
              rejected_at: resolvedAt,
              reason: result.reason,
              requested_decision: "accept",
            },
          },
        };
      }

      const reason = resolution.reason ?? "operator rejected skill split";
      await handler.reject(item, refs, reason);
      return {
        refs: {
          ...item.refs,
          review_resolution: {
            decision: "reject",
            rejected_at: resolvedAt,
            reason,
          },
        },
      };
    },
  };
}
