import { z } from "zod";

import { episodeIdSchema } from "../../episodic/index.js";
import { SemanticError } from "../../../util/errors.js";
import {
  semanticEdgeIdSchema,
  semanticNodeIdSchema,
  semanticNodeSchema,
  type SemanticNode,
} from "../types.js";
import {
  reviewResolutionSchema,
  type ReviewQueueHandler,
  type ReviewResolution,
} from "../review-queue.js";

const NEW_INSIGHT_REVIEW_RESOLUTIONS = new Set<ReviewResolution>([
  "accept",
  "invalidate",
  "dismiss",
]);

const reviewSemanticNodePayloadSchema = z
  .object({
    id: semanticNodeIdSchema,
    kind: z.enum(["concept", "entity", "proposition"]),
    label: z.string().min(1),
    description: z.string().min(1),
    domain: z.string().min(1).nullable().default(null),
    aliases: z.array(z.string().min(1)),
    confidence: z.number().min(0).max(1),
    source_episode_ids: z.array(episodeIdSchema).min(1),
    created_at: z.number().finite(),
    updated_at: z.number().finite(),
    last_verified_at: z.number().finite(),
    embedding: z.array(z.number().finite()),
    archived: z.boolean(),
    superseded_by: semanticNodeIdSchema.nullable(),
  })
  .strict();

const pendingReflectorTargetSchema = z.discriminatedUnion("mode", [
  z
    .object({
      mode: z.literal("insert"),
      node: reviewSemanticNodePayloadSchema,
    })
    .strict(),
  z
    .object({
      mode: z.literal("update"),
      node_id: semanticNodeIdSchema,
      patch: z
        .object({
          description: z.string().min(1),
          confidence: z.number().min(0).max(1),
          source_episode_ids: z.array(episodeIdSchema).min(1),
          last_verified_at: z.number().finite(),
          embedding: z.array(z.number().finite()),
          archived: z.boolean(),
        })
        .strict(),
    })
    .strict(),
]);

const pendingReflectorSupportEdgeSchema = z
  .object({
    id: semanticEdgeIdSchema,
    insight_node_id: semanticNodeIdSchema,
    target_node_id: semanticNodeIdSchema,
    source_episode_ids: z.array(episodeIdSchema).min(1),
    confidence: z.number().min(0).max(1),
  })
  .strict();

const pendingReflectorInsightSchema = z
  .object({
    target: pendingReflectorTargetSchema,
    candidate_support_edges: z.array(pendingReflectorSupportEdgeSchema).default([]),
    evidence_cluster: z
      .object({
        key: z.string().min(1),
        episode_ids: z.array(episodeIdSchema).min(1),
        size: z.number().int().positive(),
      })
      .strict(),
  })
  .strict();

export const newInsightReviewRefsSchema = z
  .object({
    node_ids: z.array(semanticNodeIdSchema).min(1),
    episode_ids: z.array(episodeIdSchema).min(1),
    evidence_cluster_key: z.string().min(1),
    evidence_cluster_size: z.number().int().positive(),
    reflector_pending_insight: pendingReflectorInsightSchema,
  })
  .strict();

const newInsightApplyingStateSchema = z
  .object({
    decision: reviewResolutionSchema,
    target_mode: z.enum(["insert", "update"]),
    target_node_id: semanticNodeIdSchema,
    started_at: z.number().finite(),
  })
  .strict();

export type NewInsightReviewRefs = z.infer<typeof newInsightReviewRefsSchema>;
type NewInsightApplyingState = z.infer<typeof newInsightApplyingStateSchema>;

function deserializeReviewSemanticNode(
  node: z.infer<typeof reviewSemanticNodePayloadSchema>,
): SemanticNode {
  return semanticNodeSchema.parse({
    ...node,
    embedding: Float32Array.from(node.embedding),
  });
}

function targetNodeId(refs: NewInsightReviewRefs): SemanticNode["id"] {
  const target = refs.reflector_pending_insight.target;
  return target.mode === "insert" ? target.node.id : target.node_id;
}

export function createNewInsightReviewQueueHandler(): ReviewQueueHandler<
  "new_insight",
  NewInsightReviewRefs,
  NewInsightApplyingState
> {
  return {
    kind: "new_insight",
    refsSchema: newInsightReviewRefsSchema,
    allowedResolutions: NEW_INSIGHT_REVIEW_RESOLUTIONS,
    transactionScope: ({ resolution }) =>
      resolution.decision === "accept" || resolution.decision === "invalidate"
        ? "cross_store_applying_state"
        : "sqlite",
    applyingState: {
      schema: newInsightApplyingStateSchema,
      prepare: ({ refs, resolution, ctx }) => ({
        decision: resolution.decision,
        target_mode: refs.reflector_pending_insight.target.mode,
        target_node_id: targetNodeId(refs),
        started_at: ctx.clock.now(),
      }),
      matches: (state, resolution) => state.decision === resolution.decision,
    },
    async apply({ item, refs, resolution, ctx }) {
      if (resolution.decision === "dismiss") {
        return;
      }

      if (ctx.semanticNodeRepository === undefined) {
        throw new SemanticError("Semantic node repository is required for pending insight review", {
          code: "REVIEW_QUEUE_REPAIR_UNSUPPORTED",
        });
      }

      const target = refs.reflector_pending_insight.target;

      if (resolution.decision === "invalidate") {
        const nodeId = targetNodeId(refs);
        const current = await ctx.semanticNodeRepository.get(nodeId);

        if (current === null && target.mode === "insert") {
          return;
        }

        const updated = await ctx.semanticNodeRepository.update(nodeId, {
          archived: true,
        });

        if (updated === null) {
          throw new SemanticError(`Unknown semantic node id for pending insight: ${nodeId}`, {
            code: "REVIEW_QUEUE_TARGET_NOT_FOUND",
          });
        }
        return;
      }

      const candidateSupportEdges = refs.reflector_pending_insight.candidate_support_edges;
      const insightNodeId = targetNodeId(refs);

      if (target.mode === "insert") {
        await ctx.semanticNodeRepository.insert(deserializeReviewSemanticNode(target.node));
      } else {
        const updated = await ctx.semanticNodeRepository.update(target.node_id, {
          description: target.patch.description,
          confidence: target.patch.confidence,
          source_episode_ids: target.patch.source_episode_ids,
          last_verified_at: target.patch.last_verified_at,
          embedding: Float32Array.from(target.patch.embedding),
          archived: target.patch.archived,
        });

        if (updated === null) {
          throw new SemanticError(
            `Unknown semantic node id for pending insight: ${target.node_id}`,
            {
              code: "REVIEW_QUEUE_TARGET_NOT_FOUND",
            },
          );
        }
      }

      if (candidateSupportEdges.length === 0) {
        return;
      }

      if (ctx.semanticEdgeRepository === undefined) {
        throw new SemanticError("Semantic edge repository is required for pending insight review", {
          code: "REVIEW_QUEUE_REPAIR_UNSUPPORTED",
        });
      }

      for (const edge of candidateSupportEdges) {
        if (edge.insight_node_id !== insightNodeId) {
          throw new SemanticError("Pending insight support edge points at the wrong insight node", {
            code: "REVIEW_QUEUE_TARGET_NOT_FOUND",
            cause: { itemId: item.id, insightNodeId, edgeInsightNodeId: edge.insight_node_id },
          });
        }

        const duplicate = ctx.semanticEdgeRepository.listEdges({
          fromId: edge.target_node_id,
          toId: edge.insight_node_id,
          relation: "supports",
        });
        if (duplicate.length > 0) {
          continue;
        }

        ctx.semanticEdgeRepository.addEdge({
          id: edge.id,
          from_node_id: edge.target_node_id,
          to_node_id: edge.insight_node_id,
          relation: "supports",
          confidence: edge.confidence,
          evidence_episode_ids: edge.source_episode_ids,
          created_at: ctx.clock.now(),
          last_verified_at: ctx.clock.now(),
        });
      }
    },
  };
}
