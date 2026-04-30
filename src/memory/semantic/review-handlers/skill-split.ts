import {
  skillSplitReviewPayloadSchema,
  type ReviewQueueHandler,
  type ReviewResolution,
  type SkillSplitReviewHandler,
  type SkillSplitReviewPayload,
} from "../review-queue.js";

const SKILL_SPLIT_REVIEW_RESOLUTIONS = new Set<ReviewResolution>(["accept", "reject"]);

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
