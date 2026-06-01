import { z } from "zod";

import {
  activationPolicySchema,
  creatorDirectiveEntityIdSchema,
  creatorDirectiveIdSchema,
  creatorDirectiveKindSchema,
  creatorDirectiveSubjectKindSchema,
  disclosurePolicySchema,
} from "../../creator-directives/index.js";
import type { ReviewQueueHandler, ReviewResolution } from "../review-queue.js";

export const CREATOR_DIRECTIVE_RECONCILIATION_REVIEW_KIND =
  "creator_directive_reconciliation" as const;
export const CREATOR_DIRECTIVE_RECONCILIATION_REVIEW_SUBKINDS = [
  "conflict",
  "same_content_different_scope",
  "low_confidence_redundancy",
  "disclosure_widening",
] as const;
export const CREATOR_DIRECTIVE_RECONCILIATION_VERDICTS = [
  "same_intent",
  "conflicting",
  "independent",
] as const;
export const CREATOR_DIRECTIVE_RECONCILIATION_RESOLUTIONS = [
  "supersede_to_survivor",
  "revoke_stale",
  "keep_independent",
  "escalate",
] as const;
export const CREATOR_DIRECTIVE_RECONCILIATION_CONFIDENCE_LEVELS = [
  "high",
  "medium",
  "low",
] as const;

const CREATOR_DIRECTIVE_RECONCILIATION_REVIEW_RESOLUTIONS = new Set<ReviewResolution>([
  "dismiss",
  "reject",
  "accept",
  "keep",
]);

export const creatorDirectiveReconciliationSubkindSchema = z.enum(
  CREATOR_DIRECTIVE_RECONCILIATION_REVIEW_SUBKINDS,
);
export const creatorDirectiveReconciliationVerdictSchema = z.enum(
  CREATOR_DIRECTIVE_RECONCILIATION_VERDICTS,
);
export const creatorDirectiveReconciliationResolutionSchema = z.enum(
  CREATOR_DIRECTIVE_RECONCILIATION_RESOLUTIONS,
);
export const creatorDirectiveReconciliationConfidenceSchema = z.enum(
  CREATOR_DIRECTIVE_RECONCILIATION_CONFIDENCE_LEVELS,
);

export const creatorDirectiveReconciliationFamilyKeySchema = z
  .object({
    kind: creatorDirectiveKindSchema,
    subject_kind: creatorDirectiveSubjectKindSchema,
    subject_entity_id: creatorDirectiveEntityIdSchema.nullable(),
  })
  .strict();

export const creatorDirectiveScopeEquivalenceSnapshotSchema = z
  .object({
    created_by_entity_id: creatorDirectiveEntityIdSchema,
    disclosure_policy: disclosurePolicySchema,
    activation_policy: activationPolicySchema,
  })
  .strict();

export const creatorDirectiveReconciliationJudgmentSchema = z
  .object({
    member_ids: z.array(creatorDirectiveIdSchema).min(2),
    verdict: creatorDirectiveReconciliationVerdictSchema,
    // Backward-compat: review items enqueued before the resolution field existed
    // (and any LLM output that omits it) parse as "escalate" -- the conservative
    // default that keeps a human in the loop. New reconciler runs always emit it.
    resolution: creatorDirectiveReconciliationResolutionSchema.default("escalate"),
    survivor_id: creatorDirectiveIdSchema.nullable().default(null),
    loser_ids: z.array(creatorDirectiveIdSchema).default([]),
    confidence: creatorDirectiveReconciliationConfidenceSchema,
    rationale: z.string().trim().min(1).max(1_000),
  })
  .strict()
  .superRefine((value, ctx) => {
    const memberIds = new Set(value.member_ids);
    const loserIds = new Set(value.loser_ids);
    const loserIdsAreUnique = loserIds.size === value.loser_ids.length;
    const survivorIsMember = value.survivor_id !== null && memberIds.has(value.survivor_id);
    const losersAreMembers = value.loser_ids.every((id) => memberIds.has(id));
    const survivorIsLoser = value.survivor_id !== null && loserIds.has(value.survivor_id);

    if (value.loser_ids.length !== loserIds.size) {
      ctx.addIssue({
        code: "custom",
        path: ["loser_ids"],
        message: "loser_ids must not contain duplicates",
      });
    }

    if (value.resolution === "supersede_to_survivor" || value.resolution === "revoke_stale") {
      if (value.survivor_id === null || !survivorIsMember) {
        ctx.addIssue({
          code: "custom",
          path: ["survivor_id"],
          message: `${value.resolution} requires survivor_id from member_ids`,
        });
      }

      if (value.loser_ids.length === 0) {
        ctx.addIssue({
          code: "custom",
          path: ["loser_ids"],
          message: `${value.resolution} requires at least one loser_id`,
        });
      }

      if (!losersAreMembers) {
        ctx.addIssue({
          code: "custom",
          path: ["loser_ids"],
          message: `${value.resolution} loser_ids must be from member_ids`,
        });
      }

      if (survivorIsLoser) {
        ctx.addIssue({
          code: "custom",
          path: ["loser_ids"],
          message: `${value.resolution} survivor_id must not be in loser_ids`,
        });
      }

      if (!loserIdsAreUnique) {
        ctx.addIssue({
          code: "custom",
          path: ["loser_ids"],
          message: `${value.resolution} loser_ids must be unique`,
        });
      }

      return;
    }

    if (value.survivor_id !== null) {
      ctx.addIssue({
        code: "custom",
        path: ["survivor_id"],
        message: `${value.resolution} requires survivor_id to be null`,
      });
    }

    if (value.loser_ids.length > 0) {
      ctx.addIssue({
        code: "custom",
        path: ["loser_ids"],
        message: `${value.resolution} requires loser_ids to be empty`,
      });
    }
  });

export const creatorDirectiveReconciliationReviewRefsSchema = z
  .object({
    target_type: z.literal(CREATOR_DIRECTIVE_RECONCILIATION_REVIEW_KIND),
    subkind: creatorDirectiveReconciliationSubkindSchema,
    directive_ids: z.array(creatorDirectiveIdSchema).min(2),
    family_key: creatorDirectiveReconciliationFamilyKeySchema,
    members: z
      .array(
        z
          .object({
            id: creatorDirectiveIdSchema,
            family_key: creatorDirectiveReconciliationFamilyKeySchema,
            scope_equivalence: creatorDirectiveScopeEquivalenceSnapshotSchema,
          })
          .strict(),
      )
      .min(2),
    judgment: creatorDirectiveReconciliationJudgmentSchema,
  })
  .strict();

export type CreatorDirectiveReconciliationReviewRefs = z.infer<
  typeof creatorDirectiveReconciliationReviewRefsSchema
>;
export type CreatorDirectiveReconciliationSubkind = z.infer<
  typeof creatorDirectiveReconciliationSubkindSchema
>;
export type CreatorDirectiveReconciliationResolution = z.infer<
  typeof creatorDirectiveReconciliationResolutionSchema
>;
export type CreatorDirectiveReconciliationJudgment = z.infer<
  typeof creatorDirectiveReconciliationJudgmentSchema
>;
export type CreatorDirectiveReconciliationFamilyKey = z.infer<
  typeof creatorDirectiveReconciliationFamilyKeySchema
>;
export type CreatorDirectiveScopeEquivalenceSnapshot = z.infer<
  typeof creatorDirectiveScopeEquivalenceSnapshotSchema
>;

export function createCreatorDirectiveReconciliationReviewQueueHandler(): ReviewQueueHandler<
  typeof CREATOR_DIRECTIVE_RECONCILIATION_REVIEW_KIND,
  CreatorDirectiveReconciliationReviewRefs
> {
  return {
    kind: CREATOR_DIRECTIVE_RECONCILIATION_REVIEW_KIND,
    refsSchema: creatorDirectiveReconciliationReviewRefsSchema,
    allowedResolutions: CREATOR_DIRECTIVE_RECONCILIATION_REVIEW_RESOLUTIONS,
    transactionScope: () => "sqlite",
    apply: () => undefined,
  };
}
