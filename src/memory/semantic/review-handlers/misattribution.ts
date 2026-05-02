import { z } from "zod";

import { provenanceSchema } from "../../common/provenance.js";
import { episodeIdSchema, episodePatchSchema } from "../../episodic/index.js";
import { streamEntryIdSchema } from "../../../stream/index.js";
import { SemanticError } from "../../../util/errors.js";
import { semanticNodeIdSchema } from "../types.js";
import {
  reviewResolutionSchema,
  type ReviewQueueHandler,
  type ReviewResolution,
} from "../review-queue.js";

const MISATTRIBUTION_REVIEW_RESOLUTIONS = new Set<ReviewResolution>([
  "accept",
  "reject",
  "dismiss",
]);

const misattributionEpisodePatchSchema = episodePatchSchema
  .pick({
    participants: true,
    audience_entity_id: true,
    narrative: true,
    tags: true,
  })
  .strict()
  .refine((patch) => Object.keys(patch).length > 0, {
    message: "Misattribution episode patch must not be empty",
  });

const semanticNodeMisattributionPatchSchema = z
  .object({
    label: z.string().min(1).optional(),
    aliases: z.array(z.string().min(1)).optional(),
    description: z.string().min(1).optional(),
    source_episode_ids: z.array(episodeIdSchema).min(1).optional(),
  })
  .strict()
  .refine((patch) => Object.keys(patch).length > 0, {
    message: "Misattribution semantic node patch must not be empty",
  });

export const misattributionReviewRefsSchema = z.discriminatedUnion("target_type", [
  z
    .object({
      target_type: z.literal("episode"),
      target_id: episodeIdSchema,
      patch: misattributionEpisodePatchSchema,
      evidence_stream_ids: z.array(streamEntryIdSchema).optional(),
      proposed_provenance: provenanceSchema.optional(),
    })
    .strict(),
  z
    .object({
      target_type: z.literal("semantic_node"),
      target_id: semanticNodeIdSchema,
      patch: semanticNodeMisattributionPatchSchema,
      evidence_stream_ids: z.array(streamEntryIdSchema).optional(),
      proposed_provenance: provenanceSchema.optional(),
    })
    .strict(),
]);

export type MisattributionReviewRefs = z.infer<typeof misattributionReviewRefsSchema>;
const misattributionApplyingStateSchema = z
  .object({
    decision: reviewResolutionSchema,
    target_type: z.enum(["episode", "semantic_node"]),
    target_id: z.string().min(1),
    started_at: z.number().finite(),
  })
  .strict();

type MisattributionApplyingState = z.infer<typeof misattributionApplyingStateSchema>;

export function createMisattributionReviewQueueHandler(): ReviewQueueHandler<
  "misattribution",
  MisattributionReviewRefs,
  MisattributionApplyingState
> {
  return {
    kind: "misattribution",
    refsSchema: misattributionReviewRefsSchema,
    allowedResolutions: MISATTRIBUTION_REVIEW_RESOLUTIONS,
    transactionScope: ({ resolution }) =>
      resolution.decision === "accept" ? "cross_store_applying_state" : "sqlite",
    applyingState: {
      schema: misattributionApplyingStateSchema,
      prepare: ({ refs, resolution, ctx }) => ({
        decision: resolution.decision,
        target_type: refs.target_type,
        target_id: refs.target_id,
        started_at: ctx.clock.now(),
      }),
      matches: (state, resolution) => state.decision === resolution.decision,
    },
    async apply({ refs, resolution, ctx }) {
      if (resolution.decision !== "accept") {
        return;
      }

      if (refs.target_type === "episode") {
        if (ctx.episodicRepository === undefined) {
          throw new SemanticError("Episode repository is required for misattribution repair", {
            code: "REVIEW_QUEUE_REPAIR_UNSUPPORTED",
          });
        }

        const updated = await ctx.episodicRepository.update(refs.target_id, refs.patch);

        if (updated === null) {
          throw new SemanticError(
            `Unknown episode id for misattribution repair: ${refs.target_id}`,
            {
              code: "REVIEW_QUEUE_TARGET_NOT_FOUND",
            },
          );
        }
        return;
      }

      if (ctx.semanticNodeRepository === undefined) {
        throw new SemanticError("Semantic node repository is required for misattribution repair", {
          code: "REVIEW_QUEUE_REPAIR_UNSUPPORTED",
        });
      }

      const updated = await ctx.semanticNodeRepository.update(refs.target_id, {
        ...refs.patch,
        ...(refs.patch.aliases === undefined ? {} : { replace_aliases: true }),
        ...(refs.patch.source_episode_ids === undefined
          ? {}
          : { replace_source_episode_ids: true }),
      });

      if (updated === null) {
        throw new SemanticError(
          `Unknown semantic node id for misattribution repair: ${refs.target_id}`,
          {
            code: "REVIEW_QUEUE_TARGET_NOT_FOUND",
          },
        );
      }
    },
  };
}
