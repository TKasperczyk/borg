import { z } from "zod";

import { SemanticError } from "../../../util/errors.js";
import { semanticEdgeIdSchema, semanticNodeIdSchema, type SemanticNode } from "../types.js";
import {
  reviewResolutionSchema,
  type ReviewKind,
  type ReviewQueueHandler,
  type ReviewResolution,
} from "../review-queue.js";
import { closeSemanticEdgeFromReview } from "./semantic-edge-closure.js";

const SEMANTIC_PAIR_REVIEW_RESOLUTIONS = new Set<ReviewResolution>([
  "keep_both",
  "supersede",
  "invalidate",
  "dismiss",
]);

const semanticPairNodeRefsSchema = z
  .object({
    node_ids: z.tuple([semanticNodeIdSchema, semanticNodeIdSchema]),
    node_labels: z.tuple([z.string().min(1), z.string().min(1)]).optional(),
    edge_id: semanticEdgeIdSchema.optional(),
  })
  .strict();

const semanticPairEdgeClosureRefsSchema = z
  .object({
    loser_edge_id: semanticEdgeIdSchema,
    suggested_valid_to: z.number().finite().optional(),
    reason: z.string().min(1),
  })
  .strict();

export const semanticPairReviewRefsSchema = z.union([
  semanticPairNodeRefsSchema,
  semanticPairEdgeClosureRefsSchema,
]);

const semanticPairApplyingStateSchema = z
  .object({
    decision: reviewResolutionSchema,
    operation: z.enum(["node_supersede", "node_invalidate"]),
    winner_node_id: semanticNodeIdSchema,
    started_at: z.number().finite(),
  })
  .strict();

export type SemanticPairReviewKind = Extract<ReviewKind, "contradiction" | "duplicate">;
export type SemanticPairReviewRefs = z.infer<typeof semanticPairReviewRefsSchema>;
type SemanticPairApplyingState = z.infer<typeof semanticPairApplyingStateSchema>;

function isEdgeClosureRefs(
  refs: SemanticPairReviewRefs,
): refs is z.infer<typeof semanticPairEdgeClosureRefsSchema> {
  return "loser_edge_id" in refs;
}

function requireNodePairRefs(
  refs: SemanticPairReviewRefs,
  kind: SemanticPairReviewKind,
): z.infer<typeof semanticPairNodeRefsSchema> {
  if (isEdgeClosureRefs(refs)) {
    throw new SemanticError("semantic edge closure refs are not node-pair refs", {
      code: "REVIEW_QUEUE_MALFORMED_PAIR_REFS",
      cause: { kind },
    });
  }

  return refs;
}

function requireWinner(input: {
  refs: z.infer<typeof semanticPairNodeRefsSchema>;
  winnerNodeId: SemanticNode["id"] | undefined;
  itemId: number;
  kind: SemanticPairReviewKind;
}): SemanticNode["id"] {
  if (input.winnerNodeId === undefined) {
    throw new SemanticError("winner_node_id is required for supersede/invalidate", {
      code: "REVIEW_QUEUE_WINNER_REQUIRED",
      cause: { itemId: input.itemId, kind: input.kind },
    });
  }

  if (!input.refs.node_ids.includes(input.winnerNodeId)) {
    throw new SemanticError("winner_node_id must reference a node in the review item", {
      code: "REVIEW_QUEUE_WINNER_INVALID",
      cause: {
        itemId: input.itemId,
        kind: input.kind,
        winner_node_id: input.winnerNodeId,
      },
    });
  }

  return input.winnerNodeId;
}

export function createSemanticPairReviewQueueHandler(
  kind: SemanticPairReviewKind,
): ReviewQueueHandler<SemanticPairReviewKind, SemanticPairReviewRefs, SemanticPairApplyingState> {
  return {
    kind,
    refsSchema: semanticPairReviewRefsSchema,
    allowedResolutions: SEMANTIC_PAIR_REVIEW_RESOLUTIONS,
    transactionScope: ({ refs, resolution }) => {
      if (resolution.decision === "keep_both" || resolution.decision === "dismiss") {
        return "sqlite";
      }

      return isEdgeClosureRefs(refs) ? "sqlite" : "cross_store_applying_state";
    },
    applyingState: {
      schema: semanticPairApplyingStateSchema,
      prepare: ({ item, refs, resolution, ctx }) => {
        const nodeRefs = requireNodePairRefs(refs, kind);
        const winnerNodeId = requireWinner({
          refs: nodeRefs,
          winnerNodeId: resolution.winner_node_id,
          itemId: item.id,
          kind,
        });

        return {
          decision: resolution.decision,
          operation: resolution.decision === "supersede" ? "node_supersede" : "node_invalidate",
          winner_node_id: winnerNodeId,
          started_at: ctx.clock.now(),
        };
      },
      matches: (state, resolution) =>
        state.decision === resolution.decision &&
        state.winner_node_id === resolution.winner_node_id,
    },
    async apply({ item, refs, resolution, ctx }) {
      if (resolution.decision === "keep_both" || resolution.decision === "dismiss") {
        return;
      }

      if (isEdgeClosureRefs(refs)) {
        closeSemanticEdgeFromReview({
          item,
          repair: {
            edgeId: refs.loser_edge_id,
            validTo: refs.suggested_valid_to,
            reason: refs.reason,
          },
          ctx,
        });
        return;
      }

      if (ctx.semanticNodeRepository === undefined) {
        throw new SemanticError("Semantic node repository is required for pair review repair", {
          code: "REVIEW_QUEUE_REPAIR_UNSUPPORTED",
        });
      }

      const winnerNodeId = requireWinner({
        refs,
        winnerNodeId: resolution.winner_node_id,
        itemId: item.id,
        kind,
      });
      const nodes = await ctx.semanticNodeRepository.getMany(refs.node_ids, {
        includeArchived: true,
      });
      const first = nodes[0];
      const second = nodes[1];

      if (first === null || first === undefined || second === null || second === undefined) {
        throw new SemanticError(
          `Semantic pair targets missing for review item ${item.id}: ${refs.node_ids.join(", ")}`,
          {
            code: "REVIEW_QUEUE_TARGET_NOT_FOUND",
          },
        );
      }

      const winner = first.id === winnerNodeId ? first : second;
      const loser = winner.id === first.id ? second : first;

      if (resolution.decision === "supersede") {
        await ctx.semanticNodeRepository.update(loser.id, {
          superseded_by: winner.id,
          archived: true,
        });
        return;
      }

      await ctx.semanticNodeRepository.update(loser.id, {
        confidence: 0,
        archived: true,
      });
    },
  };
}
