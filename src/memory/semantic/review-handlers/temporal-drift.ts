import { z } from "zod";

import { provenanceSchema } from "../../common/provenance.js";
import { episodeIdSchema, episodePatchSchema } from "../../episodic/index.js";
import { SemanticError } from "../../../util/errors.js";
import { semanticEdgeIdSchema, semanticNodeIdSchema } from "../types.js";
import {
  reviewResolutionSchema,
  type ReviewQueueHandler,
  type ReviewResolution,
} from "../review-queue.js";
import { closeSemanticEdgeFromReview } from "./semantic-edge-closure.js";

const TEMPORAL_DRIFT_REVIEW_RESOLUTIONS = new Set<ReviewResolution>([
  "accept",
  "reject",
  "dismiss",
]);

const temporalDriftEpisodeRefsSchema = z
  .object({
    target_type: z.literal("episode"),
    target_id: episodeIdSchema,
    corrected_start_time: z.number().finite().optional(),
    corrected_end_time: z.number().finite().optional(),
    patch_description: z.string().min(1).optional(),
    proposed_provenance: provenanceSchema.optional(),
  })
  .strict()
  .refine(
    (refs) =>
      refs.corrected_start_time !== undefined ||
      refs.corrected_end_time !== undefined ||
      refs.patch_description !== undefined,
    {
      message: "Temporal drift episode refs must include at least one repair field",
    },
  );

export const temporalDriftReviewRefsSchema = z.discriminatedUnion("target_type", [
  temporalDriftEpisodeRefsSchema,
  z
    .object({
      target_type: z.literal("semantic_node"),
      target_id: semanticNodeIdSchema,
      patch_description: z.string().min(1),
      proposed_provenance: provenanceSchema.optional(),
    })
    .strict(),
  z
    .object({
      target_type: z.literal("semantic_edge"),
      target_kind: z.literal("semantic_edge"),
      target_id: semanticEdgeIdSchema,
      suggested_valid_to: z.number().finite().optional(),
      by_edge_id: semanticEdgeIdSchema.optional(),
      reason: z.string().min(1),
      proposed_provenance: provenanceSchema.optional(),
    })
    .strict(),
]);

export type TemporalDriftReviewRefs = z.infer<typeof temporalDriftReviewRefsSchema>;
const temporalDriftApplyingStateSchema = z
  .object({
    decision: reviewResolutionSchema,
    target_type: z.enum(["episode", "semantic_node", "semantic_edge"]),
    target_id: z.string().min(1),
    started_at: z.number().finite(),
  })
  .strict();

type TemporalDriftApplyingState = z.infer<typeof temporalDriftApplyingStateSchema>;

export function createTemporalDriftReviewQueueHandler(): ReviewQueueHandler<
  "temporal_drift",
  TemporalDriftReviewRefs,
  TemporalDriftApplyingState
> {
  return {
    kind: "temporal_drift",
    refsSchema: temporalDriftReviewRefsSchema,
    allowedResolutions: TEMPORAL_DRIFT_REVIEW_RESOLUTIONS,
    transactionScope: ({ refs, resolution }) => {
      if (resolution.decision !== "accept") {
        return "sqlite";
      }

      return refs.target_type === "semantic_edge" ? "sqlite" : "cross_store_applying_state";
    },
    applyingState: {
      schema: temporalDriftApplyingStateSchema,
      prepare: ({ refs, resolution, ctx }) => ({
        decision: resolution.decision,
        target_type: refs.target_type,
        target_id: refs.target_id,
        started_at: ctx.clock.now(),
      }),
      matches: (state, resolution) => state.decision === resolution.decision,
    },
    async apply({ item, refs, resolution, ctx }) {
      if (resolution.decision !== "accept") {
        return;
      }

      if (refs.target_type === "semantic_edge") {
        closeSemanticEdgeFromReview({
          item,
          repair: {
            edgeId: refs.target_id,
            validTo: refs.suggested_valid_to,
            byEdgeId: refs.by_edge_id,
            reason: refs.reason,
          },
          ctx,
        });
        return;
      }

      if (refs.target_type === "episode") {
        if (ctx.episodicRepository === undefined) {
          throw new SemanticError("Episode repository is required for temporal drift repair", {
            code: "REVIEW_QUEUE_REPAIR_UNSUPPORTED",
          });
        }

        const patch = episodePatchSchema.parse({
          ...(refs.corrected_start_time === undefined
            ? {}
            : { start_time: refs.corrected_start_time }),
          ...(refs.corrected_end_time === undefined ? {} : { end_time: refs.corrected_end_time }),
          ...(refs.patch_description === undefined ? {} : { narrative: refs.patch_description }),
        });
        const updated = await ctx.episodicRepository.update(refs.target_id, patch);

        if (updated === null) {
          throw new SemanticError(
            `Unknown episode id for temporal drift repair: ${refs.target_id}`,
            {
              code: "REVIEW_QUEUE_TARGET_NOT_FOUND",
            },
          );
        }
        return;
      }

      if (ctx.semanticNodeRepository === undefined) {
        throw new SemanticError("Semantic node repository is required for temporal drift repair", {
          code: "REVIEW_QUEUE_REPAIR_UNSUPPORTED",
        });
      }

      const updated = await ctx.semanticNodeRepository.update(refs.target_id, {
        description: refs.patch_description,
        last_verified_at: ctx.clock.now(),
      });

      if (updated === null) {
        throw new SemanticError(
          `Unknown semantic node id for temporal drift repair: ${refs.target_id}`,
          {
            code: "REVIEW_QUEUE_TARGET_NOT_FOUND",
          },
        );
      }
    },
  };
}
