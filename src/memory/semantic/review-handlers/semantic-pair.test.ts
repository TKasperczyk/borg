import { describe, expect, it, vi } from "vitest";

import { FixedClock } from "../../../util/clock.js";
import type { ReviewHandlerContext, ReviewQueueItem } from "../review-queue.js";
import type { SemanticEdge, SemanticNode } from "../types.js";
import {
  createSemanticPairReviewQueueHandler,
  semanticPairReviewRefsSchema,
  type SemanticPairReviewKind,
  type SemanticPairReviewRefs,
} from "./semantic-pair.js";

const firstNode = {
  id: "semn_aaaaaaaaaaaaaaaa",
  label: "Atlas succeeds",
  archived: false,
  confidence: 0.8,
} as SemanticNode;
const secondNode = {
  id: "semn_bbbbbbbbbbbbbbbb",
  label: "Atlas fails",
  archived: false,
  confidence: 0.6,
} as SemanticNode;
const edge = {
  id: "seme_aaaaaaaaaaaaaaaa",
  from_node_id: firstNode.id,
  to_node_id: secondNode.id,
  relation: "supports",
  confidence: 0.7,
  evidence_episode_ids: ["ep_aaaaaaaaaaaaaaaa"],
  created_at: 1_000,
  last_verified_at: 1_000,
  valid_from: 1_000,
  valid_to: null,
  invalidated_at: null,
  invalidated_by_edge_id: null,
  invalidated_by_review_id: null,
  invalidated_by_process: null,
  invalidated_reason: null,
} as SemanticEdge;

function itemFor(kind: SemanticPairReviewKind, refs: SemanticPairReviewRefs): ReviewQueueItem {
  return {
    id: kind === "contradiction" ? 1 : 2,
    kind,
    refs,
    reason: "pair needs review",
    created_at: 1_000,
    resolved_at: null,
    resolution: null,
  };
}

function ctxWith(input: Partial<ReviewHandlerContext>): ReviewHandlerContext {
  return {
    clock: new FixedClock(5_000),
    ...input,
  } as ReviewHandlerContext;
}

describe("semantic pair review handler", () => {
  for (const kind of ["contradiction", "duplicate"] as const) {
    it(`supersedes node-pair losers for ${kind}`, async () => {
      const refs = semanticPairReviewRefsSchema.parse({
        node_ids: [firstNode.id, secondNode.id],
        node_labels: ["Atlas succeeds", "Atlas fails"],
      });
      const update = vi.fn(async () => ({}));
      const handler = createSemanticPairReviewQueueHandler(kind);
      const ctx = ctxWith({
        semanticNodeRepository: {
          getMany: vi.fn(async () => [firstNode, secondNode]),
          update,
        } as never,
      });

      expect(
        handler.transactionScope({
          item: itemFor(kind, refs),
          refs,
          resolution: {
            decision: "supersede",
            winner_node_id: firstNode.id,
          },
          ctx,
        }),
      ).toBe("cross_store_applying_state");

      await handler.apply({
        item: itemFor(kind, refs),
        refs,
        resolution: {
          decision: "supersede",
          winner_node_id: firstNode.id,
        },
        applyingState: {
          decision: "supersede",
          operation: "node_supersede",
          winner_node_id: firstNode.id,
          started_at: 5_000,
        },
        ctx,
      });

      expect(update).toHaveBeenCalledWith(secondNode.id, {
        superseded_by: firstNode.id,
        archived: true,
      });
    });

    it(`invalidates node-pair losers for ${kind}`, async () => {
      const refs = semanticPairReviewRefsSchema.parse({
        node_ids: [firstNode.id, secondNode.id],
      });
      const update = vi.fn(async () => ({}));
      const handler = createSemanticPairReviewQueueHandler(kind);

      await handler.apply({
        item: itemFor(kind, refs),
        refs,
        resolution: {
          decision: "invalidate",
          winner_node_id: secondNode.id,
        },
        applyingState: {
          decision: "invalidate",
          operation: "node_invalidate",
          winner_node_id: secondNode.id,
          started_at: 5_000,
        },
        ctx: ctxWith({
          semanticNodeRepository: {
            getMany: vi.fn(async () => [firstNode, secondNode]),
            update,
          } as never,
        }),
      });

      expect(update).toHaveBeenCalledWith(firstNode.id, {
        confidence: 0,
        archived: true,
      });
    });

    it(`closes loser semantic edges for ${kind}`, async () => {
      const refs = semanticPairReviewRefsSchema.parse({
        loser_edge_id: edge.id,
        suggested_valid_to: 4_000,
        reason: "support edge lost the review",
      });
      const invalidated = {
        ...edge,
        valid_to: 4_000,
        invalidated_at: 5_000,
        invalidated_by_review_id: kind === "contradiction" ? 1 : 2,
        invalidated_by_process: "review",
        invalidated_reason: "support edge lost the review",
      } as SemanticEdge;
      const record = vi.fn();
      const handler = createSemanticPairReviewQueueHandler(kind);
      const ctx = ctxWith({
        semanticEdgeRepository: {
          getEdge: vi.fn(() => edge),
          invalidateEdge: vi.fn(() => invalidated),
        } as never,
        identityEventRepository: {
          findByReviewKey: vi.fn(() => null),
          record,
        } as never,
      });

      expect(
        handler.transactionScope({
          item: itemFor(kind, refs),
          refs,
          resolution: { decision: "invalidate" },
          ctx,
        }),
      ).toBe("sqlite");

      await handler.apply({
        item: itemFor(kind, refs),
        refs,
        resolution: { decision: "invalidate" },
        applyingState: null,
        ctx,
      });

      expect(record).toHaveBeenCalledWith(
        expect.objectContaining({
          record_type: "semantic_edge",
          record_id: edge.id,
          action: "edge_invalidate",
        }),
      );
    });
  }

  it("requires winners to be members of the node pair", async () => {
    const refs = semanticPairReviewRefsSchema.parse({
      node_ids: [firstNode.id, secondNode.id],
    });
    const handler = createSemanticPairReviewQueueHandler("duplicate");

    await expect(
      handler.apply({
        item: itemFor("duplicate", refs),
        refs,
        resolution: {
          decision: "invalidate",
          winner_node_id: "semn_cccccccccccccccc" as never,
        },
        applyingState: null,
        ctx: ctxWith({
          semanticNodeRepository: {
            getMany: vi.fn(),
            update: vi.fn(),
          } as never,
        }),
      }),
    ).rejects.toMatchObject({
      code: "REVIEW_QUEUE_WINNER_INVALID",
    });
  });

  it("rejects malformed pair refs", () => {
    expect(() =>
      semanticPairReviewRefsSchema.parse({
        node_ids: ["semn_aaaaaaaaaaaaaaaa"],
      }),
    ).toThrow();
    expect(() =>
      semanticPairReviewRefsSchema.parse({
        loser_edge_id: "seme_aaaaaaaaaaaaaaaa",
      }),
    ).toThrow();
  });
});
