import { describe, expect, it, vi } from "vitest";

import { FixedClock } from "../../../util/clock.js";
import type { ReviewHandlerContext, ReviewQueueItem } from "../review-queue.js";
import type { SemanticEdge } from "../types.js";
import {
  createTemporalDriftReviewQueueHandler,
  temporalDriftReviewRefsSchema,
  type TemporalDriftReviewRefs,
} from "./temporal-drift.js";

function itemFor(refs: TemporalDriftReviewRefs): ReviewQueueItem {
  return {
    id: 2,
    kind: "temporal_drift",
    refs,
    reason: "time semantics drifted",
    created_at: 1_000,
    resolved_at: null,
    resolution: null,
  };
}

function ctxWith(input: Partial<ReviewHandlerContext>): ReviewHandlerContext {
  return {
    clock: new FixedClock(6_000),
    ...input,
  } as ReviewHandlerContext;
}

const edge = {
  id: "seme_aaaaaaaaaaaaaaaa",
  from_node_id: "semn_aaaaaaaaaaaaaaaa",
  to_node_id: "semn_bbbbbbbbbbbbbbbb",
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

describe("temporal drift review handler", () => {
  it("accepts episode timestamp and narrative repairs", async () => {
    const refs = temporalDriftReviewRefsSchema.parse({
      target_type: "episode",
      target_id: "ep_aaaaaaaaaaaaaaaa",
      corrected_start_time: 2_000,
      corrected_end_time: 3_000,
      patch_description: "The incident happened after the rollback.",
    });
    const update = vi.fn(async () => ({}));
    const handler = createTemporalDriftReviewQueueHandler();

    await handler.apply({
      item: itemFor(refs),
      refs,
      resolution: { decision: "accept" },
      applyingState: null,
      ctx: ctxWith({
        episodicRepository: {
          update,
        } as never,
      }),
    });

    expect(update).toHaveBeenCalledWith(
      "ep_aaaaaaaaaaaaaaaa",
      expect.objectContaining({
        start_time: 2_000,
        end_time: 3_000,
        narrative: "The incident happened after the rollback.",
      }),
    );
  });

  it("accepts semantic-node description repairs", async () => {
    const refs = temporalDriftReviewRefsSchema.parse({
      target_type: "semantic_node",
      target_id: "semn_aaaaaaaaaaaaaaaa",
      patch_description: "This happened after the rollback.",
    });
    const update = vi.fn(async () => ({}));
    const handler = createTemporalDriftReviewQueueHandler();
    const ctx = ctxWith({
      semanticNodeRepository: {
        update,
      } as never,
    });

    expect(
      handler.transactionScope({
        item: itemFor(refs),
        refs,
        resolution: { decision: "accept" },
        ctx,
      }),
    ).toBe("cross_store_applying_state");

    await handler.apply({
      item: itemFor(refs),
      refs,
      resolution: { decision: "accept" },
      applyingState: null,
      ctx,
    });

    expect(update).toHaveBeenCalledWith("semn_aaaaaaaaaaaaaaaa", {
      description: "This happened after the rollback.",
      last_verified_at: 6_000,
    });
  });

  it("closes semantic edges and records the identity audit", async () => {
    const refs = temporalDriftReviewRefsSchema.parse({
      target_type: "semantic_edge",
      target_kind: "semantic_edge",
      target_id: "seme_aaaaaaaaaaaaaaaa",
      suggested_valid_to: 4_000,
      by_edge_id: "seme_bbbbbbbbbbbbbbbb",
      reason: "support interval should close",
    });
    const invalidated = {
      ...edge,
      valid_to: 4_000,
      invalidated_at: 6_000,
      invalidated_by_edge_id: "seme_bbbbbbbbbbbbbbbb",
      invalidated_by_review_id: 2,
      invalidated_by_process: "review",
      invalidated_reason: "support interval should close",
    } as SemanticEdge;
    const record = vi.fn();
    const handler = createTemporalDriftReviewQueueHandler();
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
        item: itemFor(refs),
        refs,
        resolution: { decision: "accept" },
        ctx,
      }),
    ).toBe("sqlite");

    await handler.apply({
      item: itemFor(refs),
      refs,
      resolution: { decision: "accept" },
      applyingState: null,
      ctx,
    });

    expect(record).toHaveBeenCalledWith(
      expect.objectContaining({
        record_type: "semantic_edge",
        record_id: "seme_aaaaaaaaaaaaaaaa",
        action: "edge_invalidate",
        review_item_id: 2,
      }),
    );
  });

  it("rejects malformed refs", () => {
    expect(() =>
      temporalDriftReviewRefsSchema.parse({
        target_type: "episode",
        target_id: "ep_aaaaaaaaaaaaaaaa",
      }),
    ).toThrow();
    expect(() =>
      temporalDriftReviewRefsSchema.parse({
        target_type: "semantic_edge",
        target_id: "seme_aaaaaaaaaaaaaaaa",
        reason: "missing target kind",
      }),
    ).toThrow();
  });
});
