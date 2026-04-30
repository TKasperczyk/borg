import { SemanticError } from "../../../util/errors.js";
import type { JsonValue } from "../../../util/json-value.js";
import type { ReviewHandlerContext, ReviewQueueItem } from "../review-queue.js";
import type { SemanticEdge } from "../types.js";

export type SemanticEdgeClosureRepair = {
  edgeId: SemanticEdge["id"];
  validTo?: number;
  byEdgeId?: SemanticEdge["id"];
  reason: string;
};

function recordSemanticEdgeInvalidationAudit(input: {
  item: ReviewQueueItem;
  previous: SemanticEdge;
  next: SemanticEdge;
  ctx: ReviewHandlerContext;
}): void {
  const identityEventRepository = input.ctx.identityEventRepository;

  if (identityEventRepository === undefined) {
    return;
  }

  const existing = identityEventRepository.findByReviewKey({
    reviewItemId: input.item.id,
    recordType: "semantic_edge",
    recordId: input.next.id,
    action: "edge_invalidate",
  });

  if (existing !== null) {
    return;
  }

  const auditShape = {
    edge_id: input.next.id,
    prior_valid_to: input.previous.valid_to,
    new_valid_to: input.next.valid_to,
    by_process: input.next.invalidated_by_process,
    by_review_id: input.next.invalidated_by_review_id,
    reason: input.next.invalidated_reason,
    by_edge_id: input.next.invalidated_by_edge_id,
  } satisfies JsonValue;

  identityEventRepository.record({
    record_type: "semantic_edge",
    record_id: input.next.id,
    action: "edge_invalidate",
    old_value: {
      edge_id: input.previous.id,
      prior_valid_to: input.previous.valid_to,
    },
    new_value: auditShape,
    reason: input.next.invalidated_reason,
    provenance: {
      kind: "manual",
    },
    review_item_id: input.item.id,
  });
}

export function closeSemanticEdgeFromReview(input: {
  item: ReviewQueueItem;
  repair: SemanticEdgeClosureRepair;
  ctx: ReviewHandlerContext;
}): void {
  const semanticEdgeRepository = input.ctx.semanticEdgeRepository;

  if (semanticEdgeRepository === undefined) {
    throw new SemanticError("Semantic edge repository is required for edge review repair", {
      code: "REVIEW_QUEUE_REPAIR_UNSUPPORTED",
    });
  }

  const current = semanticEdgeRepository.getEdge(input.repair.edgeId);

  if (current === null) {
    throw new SemanticError(`Unknown semantic edge id for review repair: ${input.repair.edgeId}`, {
      code: "REVIEW_QUEUE_TARGET_NOT_FOUND",
    });
  }

  const invalidated = semanticEdgeRepository.invalidateEdge(input.repair.edgeId, {
    at: input.repair.validTo ?? input.ctx.clock.now(),
    by_edge_id: input.repair.byEdgeId,
    by_process: "review",
    by_review_id: input.item.id,
    reason: input.repair.reason,
  });

  if (invalidated === null) {
    throw new SemanticError(`Unknown semantic edge id for review repair: ${input.repair.edgeId}`, {
      code: "REVIEW_QUEUE_TARGET_NOT_FOUND",
    });
  }

  if (current.valid_to === null) {
    recordSemanticEdgeInvalidationAudit({
      item: input.item,
      previous: current,
      next: invalidated,
      ctx: input.ctx,
    });
  }
}
