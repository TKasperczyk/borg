import { z } from "zod";

import { entityIdSchema } from "../../commitments/index.js";
import { episodeIdSchema } from "../../episodic/index.js";
import { semanticEdgeIdSchema, semanticNodeIdSchema } from "../types.js";
import type { ReviewQueueHandler, ReviewResolution } from "../review-queue.js";

const BELIEF_REVISION_REVIEW_RESOLUTIONS = new Set<ReviewResolution>(["dismiss"]);

const beliefRevisionClaimSchema = z
  .object({
    run_id: z.string().min(1),
    claimed_at: z.number().finite(),
  })
  .strict();

const beliefRevisionApplyingSchema = z
  .object({
    run_id: z.string().min(1),
    claimed_at: z.number().finite(),
    verdict: z.enum(["keep", "weaken", "archive_node"]),
    target_id: semanticNodeIdSchema,
    confidence: z.number().min(0).max(1).optional(),
    archived: z.boolean().optional(),
  })
  .strict();

const autoConfidenceDropSchema = z
  .object({
    previous_confidence: z.number().min(0).max(1),
    next_confidence: z.number().min(0).max(1),
    applied_at: z.number().finite(),
  })
  .strict();

const beliefRevisionLlmSchema = z
  .object({
    verdict: z.string().min(1),
    original_verdict: z.string().min(1),
    rationale: z.string().min(1),
    confidence_delta: z.number().min(-0.5).max(0).nullable(),
    applied_at: z.number().finite(),
  })
  .strict();

const beliefRevisionProcessStateSchema = z
  .object({
    __borg_belief_revision_claim: beliefRevisionClaimSchema.optional(),
    __borg_belief_revision_applying: beliefRevisionApplyingSchema.optional(),
    auto_confidence_drop: autoConfidenceDropSchema.optional(),
    belief_revision_llm: beliefRevisionLlmSchema.optional(),
    belief_revision_escalated_at: z.number().finite().optional(),
    belief_revision_failure_count: z.number().int().nonnegative().optional(),
    belief_revision_last_error: z.string().min(1).optional(),
  })
  .strict();

export const beliefRevisionReviewRefsSchema = z.discriminatedUnion("target_type", [
  z
    .object({
      target_type: z.literal("semantic_node"),
      target_id: semanticNodeIdSchema,
      invalidated_edge_id: semanticEdgeIdSchema,
      dependency_path_edge_ids: z.array(semanticEdgeIdSchema),
      surviving_support_edge_ids: z.array(semanticEdgeIdSchema),
      evidence_episode_ids: z.array(episodeIdSchema),
      audience_entity_id: entityIdSchema.nullable().optional(),
    })
    .merge(beliefRevisionProcessStateSchema)
    .strict(),
  z
    .object({
      target_type: z.literal("semantic_edge"),
      target_id: semanticEdgeIdSchema,
      invalidated_edge_id: semanticEdgeIdSchema,
      dependency_path_edge_ids: z.array(semanticEdgeIdSchema),
      surviving_support_edge_ids: z.array(semanticEdgeIdSchema),
      evidence_episode_ids: z.array(episodeIdSchema),
      audience_entity_id: entityIdSchema.nullable().optional(),
    })
    .merge(beliefRevisionProcessStateSchema)
    .strict(),
]);

export type BeliefRevisionReviewRefs = z.infer<typeof beliefRevisionReviewRefsSchema>;

export function createBeliefRevisionReviewQueueHandler(): ReviewQueueHandler<
  "belief_revision",
  BeliefRevisionReviewRefs
> {
  return {
    kind: "belief_revision",
    refsSchema: beliefRevisionReviewRefsSchema,
    allowedResolutions: BELIEF_REVISION_REVIEW_RESOLUTIONS,
    transactionScope: () => "sqlite",
    apply() {
      return;
    },
  };
}
