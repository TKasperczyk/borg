import { z } from "zod";

import { episodeIdSchema, episodePatchSchema } from "../../episodic/index.js";
import { commitmentIdSchema, entityIdSchema } from "../../commitments/index.js";
import {
  goalIdSchema,
  openQuestionIdSchema,
  traitIdSchema,
  valueIdSchema,
} from "../../self/index.js";
import { provenanceSchema } from "../../common/provenance.js";
import { SemanticError } from "../../../util/errors.js";
import { semanticEdgeIdSchema, semanticNodeIdSchema, semanticNodePatchSchema } from "../types.js";
import {
  reviewResolutionSchema,
  type ReviewQueueHandler,
  type ReviewQueueItem,
  type ReviewResolution,
} from "../review-queue.js";

const correctionPatchSchema = z.record(z.string(), z.unknown());
const correctionBaseShape = {
  patch: correctionPatchSchema,
  proposed_provenance: provenanceSchema.optional(),
  audience_entity_id: entityIdSchema.nullable().optional(),
  prompt_summary: z.string().min(1).optional(),
};

export const correctionReviewRefsSchema = z.discriminatedUnion("target_type", [
  z
    .object({
      target_type: z.literal("episode"),
      target_id: episodeIdSchema,
      ...correctionBaseShape,
      patch: episodePatchSchema,
    })
    .strict(),
  z
    .object({
      target_type: z.literal("semantic_node"),
      target_id: semanticNodeIdSchema,
      ...correctionBaseShape,
      patch: semanticNodePatchSchema,
    })
    .strict(),
  z
    .object({
      target_type: z.literal("semantic_edge"),
      target_id: semanticEdgeIdSchema,
      ...correctionBaseShape,
    })
    .strict(),
  z
    .object({
      target_type: z.literal("value"),
      target_id: valueIdSchema,
      ...correctionBaseShape,
    })
    .strict(),
  z
    .object({
      target_type: z.literal("goal"),
      target_id: goalIdSchema,
      ...correctionBaseShape,
    })
    .strict(),
  z
    .object({
      target_type: z.literal("trait"),
      target_id: traitIdSchema,
      ...correctionBaseShape,
    })
    .strict(),
  z
    .object({
      target_type: z.literal("commitment"),
      target_id: commitmentIdSchema,
      ...correctionBaseShape,
    })
    .strict(),
  z
    .object({
      target_type: z.literal("open_question"),
      target_id: openQuestionIdSchema,
      ...correctionBaseShape,
    })
    .strict(),
]);

const CORRECTION_REVIEW_RESOLUTIONS = new Set<ReviewResolution>(["accept", "reject"]);
const correctionApplyingStateSchema = z
  .object({
    decision: reviewResolutionSchema,
    started_at: z.number().finite(),
  })
  .strict();

export type CorrectionReviewRefs = z.infer<typeof correctionReviewRefsSchema>;

export type CorrectionReviewHandlerOptions = {
  applyCorrection: (item: ReviewQueueItem) => Promise<void> | void;
};

export function createCorrectionReviewHandler(
  options: CorrectionReviewHandlerOptions,
): ReviewQueueHandler<
  "correction",
  CorrectionReviewRefs,
  z.infer<typeof correctionApplyingStateSchema>
> {
  return {
    kind: "correction",
    refsSchema: correctionReviewRefsSchema,
    allowedResolutions: CORRECTION_REVIEW_RESOLUTIONS,
    transactionScope: ({ refs, resolution }) => {
      if (resolution.decision !== "accept") {
        return "sqlite";
      }

      return refs.target_type === "episode" || refs.target_type === "semantic_node"
        ? "cross_store_applying_state"
        : "sqlite";
    },
    applyingState: {
      schema: correctionApplyingStateSchema,
      prepare: ({ resolution, ctx }) => ({
        decision: resolution.decision,
        started_at: ctx.clock.now(),
      }),
      matches: (state, resolution) => state.decision === resolution.decision,
    },
    async apply({ item, refs, resolution }) {
      if (resolution.decision !== "accept") {
        return;
      }

      if (refs.target_type === "semantic_edge") {
        throw new SemanticError(
          `Semantic edge corrections are applied with semantic edge invalidate: ${refs.target_id}`,
          {
            code: "SEMANTIC_EDGE_CORRECTION_UNSUPPORTED",
          },
        );
      }

      await options.applyCorrection({
        ...item,
        refs,
      });
    },
  };
}
