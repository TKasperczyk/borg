import { z } from "zod";

import {
  commitmentIdSchema,
  commitmentKindSchema,
  commitmentTypeSchema,
  entityIdSchema,
  directiveFamilySchema,
  streamEntryIdSchema,
} from "../../commitments/index.js";
import { memoryDisclosureLabelSchema } from "../../common/disclosure-label.js";
import type { ReviewQueueHandler, ReviewResolution } from "../review-queue.js";

export const COMMITMENT_RECONCILIATION_REVIEW_KIND = "commitment_reconciliation" as const;
export const COMMITMENT_RECONCILIATION_REVIEW_SUBKINDS = [
  "conflict",
  "cross_scope_conflict",
  "cross_scope_redundancy",
] as const;
export const COMMITMENT_RECONCILIATION_RESOLUTIONS = [
  "supersede_to_survivor",
  "keep_independent",
  "conflict",
] as const;

const COMMITMENT_RECONCILIATION_REVIEW_RESOLUTIONS = new Set<ReviewResolution>([
  "dismiss",
  "reject",
  "accept",
  "keep",
]);

export const commitmentReconciliationSubkindSchema = z.enum(
  COMMITMENT_RECONCILIATION_REVIEW_SUBKINDS,
);
export const commitmentReconciliationResolutionSchema = z.enum(
  COMMITMENT_RECONCILIATION_RESOLUTIONS,
);

export const commitmentReconciliationScopeKeySchema = z
  .object({
    kind: commitmentKindSchema,
    restricted_audience: entityIdSchema.nullable(),
    made_to_entity: entityIdSchema.nullable(),
    about_entity: entityIdSchema.nullable(),
  })
  .strict();

export const commitmentReconciliationDetectionKeySchema = z
  .object({
    kind: commitmentKindSchema,
    about_entity: entityIdSchema.nullable(),
    directive_family: directiveFamilySchema,
  })
  .strict();

// What the model emits. superseded_commitment_ids is deliberately absent: for
// supersede_to_survivor it is fully determined as commitment_ids minus the
// survivor, so asking the model to restate it only creates a partition invariant
// it can fail. The harness derives it (deriveCommitmentReconciliationJudgment);
// the stored judgment below keeps the field for consumers and persisted refs.
const commitmentReconciliationJudgmentModelShape = {
  commitment_ids: z.array(commitmentIdSchema).min(2),
  resolution: commitmentReconciliationResolutionSchema,
  survivor_commitment_id: commitmentIdSchema.nullable().default(null),
  reason: z.string().trim().min(1).max(1_000),
};

export const commitmentReconciliationModelJudgmentSchema = z
  .object(commitmentReconciliationJudgmentModelShape)
  .strict()
  .superRefine((value, ctx) => {
    const commitmentIds = new Set(value.commitment_ids);

    if (commitmentIds.size !== value.commitment_ids.length) {
      ctx.addIssue({
        code: "custom",
        path: ["commitment_ids"],
        message: "commitment_ids must not contain duplicates",
      });
    }

    if (value.resolution === "supersede_to_survivor") {
      if (
        value.survivor_commitment_id === null ||
        !commitmentIds.has(value.survivor_commitment_id)
      ) {
        ctx.addIssue({
          code: "custom",
          path: ["survivor_commitment_id"],
          message: "supersede_to_survivor requires survivor_commitment_id from commitment_ids",
        });
      }

      return;
    }

    if (value.survivor_commitment_id !== null) {
      ctx.addIssue({
        code: "custom",
        path: ["survivor_commitment_id"],
        message: `${value.resolution} requires survivor_commitment_id to be null`,
      });
    }
  });

export type CommitmentReconciliationModelJudgment = z.infer<
  typeof commitmentReconciliationModelJudgmentSchema
>;

export const commitmentReconciliationJudgmentSchema = z
  .object({
    ...commitmentReconciliationJudgmentModelShape,
    superseded_commitment_ids: z.array(commitmentIdSchema).default([]),
  })
  .strict()
  .superRefine((value, ctx) => {
    const commitmentIds = new Set(value.commitment_ids);
    const supersededIds = new Set(value.superseded_commitment_ids);
    const survivorIsMember =
      value.survivor_commitment_id !== null && commitmentIds.has(value.survivor_commitment_id);
    const supersededAreMembers = value.superseded_commitment_ids.every((id) =>
      commitmentIds.has(id),
    );
    const survivorIsSuperseded =
      value.survivor_commitment_id !== null && supersededIds.has(value.survivor_commitment_id);

    if (commitmentIds.size !== value.commitment_ids.length) {
      ctx.addIssue({
        code: "custom",
        path: ["commitment_ids"],
        message: "commitment_ids must not contain duplicates",
      });
    }

    if (supersededIds.size !== value.superseded_commitment_ids.length) {
      ctx.addIssue({
        code: "custom",
        path: ["superseded_commitment_ids"],
        message: "superseded_commitment_ids must not contain duplicates",
      });
    }

    if (value.resolution === "supersede_to_survivor") {
      if (value.survivor_commitment_id === null || !survivorIsMember) {
        ctx.addIssue({
          code: "custom",
          path: ["survivor_commitment_id"],
          message: "supersede_to_survivor requires survivor_commitment_id from commitment_ids",
        });
      }

      if (value.superseded_commitment_ids.length === 0) {
        ctx.addIssue({
          code: "custom",
          path: ["superseded_commitment_ids"],
          message: "supersede_to_survivor requires at least one superseded_commitment_id",
        });
      }

      if (!supersededAreMembers) {
        ctx.addIssue({
          code: "custom",
          path: ["superseded_commitment_ids"],
          message: "superseded_commitment_ids must be from commitment_ids",
        });
      }

      if (survivorIsSuperseded) {
        ctx.addIssue({
          code: "custom",
          path: ["superseded_commitment_ids"],
          message: "survivor_commitment_id must not be superseded",
        });
      }

      if (
        value.survivor_commitment_id !== null &&
        new Set([value.survivor_commitment_id, ...value.superseded_commitment_ids]).size !==
          commitmentIds.size
      ) {
        ctx.addIssue({
          code: "custom",
          path: ["superseded_commitment_ids"],
          message: "supersede_to_survivor must partition commitment_ids",
        });
      }

      return;
    }

    if (value.survivor_commitment_id !== null) {
      ctx.addIssue({
        code: "custom",
        path: ["survivor_commitment_id"],
        message: `${value.resolution} requires survivor_commitment_id to be null`,
      });
    }

    if (value.superseded_commitment_ids.length > 0) {
      ctx.addIssue({
        code: "custom",
        path: ["superseded_commitment_ids"],
        message: `${value.resolution} requires superseded_commitment_ids to be empty`,
      });
    }
  });

export const commitmentReconciliationReviewRefsSchema = z
  .object({
    target_type: z.literal(COMMITMENT_RECONCILIATION_REVIEW_KIND),
    subkind: commitmentReconciliationSubkindSchema,
    commitment_ids: z.array(commitmentIdSchema).min(2),
    scope_key: commitmentReconciliationScopeKeySchema,
    reason: z.string().trim().min(1).max(1_000),
    members: z
      .array(
        z
          .object({
            id: commitmentIdSchema,
            kind: commitmentKindSchema,
            type: commitmentTypeSchema,
            directive_family: directiveFamilySchema,
            directive: z.string().min(1).optional(),
            scope_key: commitmentReconciliationScopeKeySchema.optional(),
            source_stream_entry_ids: z.array(streamEntryIdSchema).optional(),
            disclosure_label: memoryDisclosureLabelSchema.optional(),
          })
          .strict(),
      )
      .min(2),
    judgment: commitmentReconciliationJudgmentSchema,
    detection_key: commitmentReconciliationDetectionKeySchema.optional(),
    source_stream_entry_ids: z.array(streamEntryIdSchema).optional(),
    disclosure_label: memoryDisclosureLabelSchema.optional(),
  })
  .strict();

export type CommitmentReconciliationJudgment = z.infer<
  typeof commitmentReconciliationJudgmentSchema
>;

/**
 * Complete a model judgment into the stored shape. The superseded set is the
 * complement of the survivor within commitment_ids for supersede_to_survivor
 * and empty otherwise, so the partition invariant holds by construction.
 */
export function deriveCommitmentReconciliationJudgment(
  judgment: CommitmentReconciliationModelJudgment,
): CommitmentReconciliationJudgment {
  const superseded_commitment_ids =
    judgment.resolution === "supersede_to_survivor" && judgment.survivor_commitment_id !== null
      ? judgment.commitment_ids.filter((id) => id !== judgment.survivor_commitment_id)
      : [];

  return { ...judgment, superseded_commitment_ids };
}
export type CommitmentReconciliationReviewRefs = z.infer<
  typeof commitmentReconciliationReviewRefsSchema
>;
export type CommitmentReconciliationScopeKey = z.infer<
  typeof commitmentReconciliationScopeKeySchema
>;
export type CommitmentReconciliationDetectionKey = z.infer<
  typeof commitmentReconciliationDetectionKeySchema
>;
export type CommitmentReconciliationSubkind = z.infer<typeof commitmentReconciliationSubkindSchema>;

export function createCommitmentReconciliationReviewQueueHandler(): ReviewQueueHandler<
  typeof COMMITMENT_RECONCILIATION_REVIEW_KIND,
  CommitmentReconciliationReviewRefs
> {
  return {
    kind: COMMITMENT_RECONCILIATION_REVIEW_KIND,
    refsSchema: commitmentReconciliationReviewRefsSchema,
    allowedResolutions: COMMITMENT_RECONCILIATION_REVIEW_RESOLUTIONS,
    transactionScope: () => "sqlite",
    apply: () => undefined,
  };
}
