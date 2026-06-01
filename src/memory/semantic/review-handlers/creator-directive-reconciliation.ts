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
] as const;
export const CREATOR_DIRECTIVE_RECONCILIATION_VERDICTS = [
  "same_intent",
  "conflicting",
  "independent",
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
    confidence: creatorDirectiveReconciliationConfidenceSchema,
    rationale: z.string().trim().min(1).max(1_000),
  })
  .strict();

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
