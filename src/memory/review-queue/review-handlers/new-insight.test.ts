import { describe, expect, it, vi } from "vitest";

import { FixedClock } from "../../../util/clock.js";
import type { ReviewHandlerContext, ReviewQueueItem } from "../review-queue.js";
import {
  createNewInsightReviewQueueHandler,
  newInsightReviewRefsSchema,
  type NewInsightReviewRefs,
} from "./new-insight.js";

function insertRefs(): NewInsightReviewRefs {
  return newInsightReviewRefsSchema.parse({
    node_ids: ["semn_aaaaaaaaaaaaaaaa"],
    episode_ids: ["ep_aaaaaaaaaaaaaaaa"],
    evidence_cluster_key: "cluster:alpha",
    evidence_cluster_size: 1,
    reflector_pending_insight: {
      target: {
        mode: "insert",
        node: {
          id: "semn_aaaaaaaaaaaaaaaa",
          kind: "proposition",
          label: "Deploys stabilize",
          description: "Deploys stabilize when rollback plans are documented.",
          aliases: [],
          confidence: 0.7,
          source_episode_ids: ["ep_aaaaaaaaaaaaaaaa"],
          created_at: 1_000,
          updated_at: 1_000,
          last_verified_at: 1_000,
          embedding: [1, 0, 0, 0],
          archived: false,
          superseded_by: null,
        },
      },
      candidate_support_edges: [
        {
          id: "seme_aaaaaaaaaaaaaaaa",
          insight_node_id: "semn_aaaaaaaaaaaaaaaa",
          target_node_id: "semn_bbbbbbbbbbbbbbbb",
          source_episode_ids: ["ep_aaaaaaaaaaaaaaaa"],
          confidence: 0.65,
        },
      ],
      evidence_cluster: {
        key: "cluster:alpha",
        episode_ids: ["ep_aaaaaaaaaaaaaaaa"],
        size: 1,
      },
    },
  });
}

function updateRefs(): NewInsightReviewRefs {
  return newInsightReviewRefsSchema.parse({
    node_ids: ["semn_aaaaaaaaaaaaaaaa"],
    episode_ids: ["ep_aaaaaaaaaaaaaaaa"],
    evidence_cluster_key: "cluster:alpha",
    evidence_cluster_size: 1,
    reflector_pending_insight: {
      target: {
        mode: "update",
        node_id: "semn_aaaaaaaaaaaaaaaa",
        patch: {
          description: "Deploys stabilize after rollback plans are documented.",
          confidence: 0.75,
          source_episode_ids: ["ep_aaaaaaaaaaaaaaaa"],
          last_verified_at: 2_000,
          embedding: [0, 1, 0, 0],
          archived: false,
        },
      },
      candidate_support_edges: [],
      evidence_cluster: {
        key: "cluster:alpha",
        episode_ids: ["ep_aaaaaaaaaaaaaaaa"],
        size: 1,
      },
    },
  });
}

function itemFor(refs: NewInsightReviewRefs): ReviewQueueItem {
  return {
    id: 1,
    kind: "new_insight",
    refs,
    reason: "reflector proposed an insight",
    created_at: 500,
    resolved_at: null,
    resolution: null,
  };
}

function ctxWith(input: Partial<ReviewHandlerContext>): ReviewHandlerContext {
  return {
    clock: new FixedClock(3_000),
    ...input,
  } as ReviewHandlerContext;
}

describe("new insight review handler", () => {
  it("accepts insert-mode insights and adds non-duplicate support edges", async () => {
    const refs = insertRefs();
    const insert = vi.fn();
    const addEdge = vi.fn();
    const handler = createNewInsightReviewQueueHandler();
    const ctx = ctxWith({
      semanticNodeRepository: {
        insert,
      } as never,
      semanticEdgeRepository: {
        listEdges: vi.fn(() => []),
        addEdge,
      } as never,
    });

    expect(handler.refsSchema.parse(refs)).toEqual(refs);
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
      applyingState: {
        decision: "accept",
        target_mode: "insert",
        target_node_id: refs.node_ids[0]!,
        started_at: 3_000,
      },
      ctx,
    });

    expect(insert).toHaveBeenCalledWith(
      expect.objectContaining({
        id: "semn_aaaaaaaaaaaaaaaa",
        embedding: expect.any(Float32Array),
      }),
    );
    expect(addEdge).toHaveBeenCalledWith(
      expect.objectContaining({
        from_node_id: "semn_bbbbbbbbbbbbbbbb",
        to_node_id: "semn_aaaaaaaaaaaaaaaa",
        relation: "supports",
      }),
    );
  });

  it("accepts update-mode insights by applying the pending patch", async () => {
    const refs = updateRefs();
    const update = vi.fn(async () => ({}));
    const handler = createNewInsightReviewQueueHandler();
    const ctx = ctxWith({
      semanticNodeRepository: {
        update,
      } as never,
    });

    await handler.apply({
      item: itemFor(refs),
      refs,
      resolution: { decision: "accept" },
      applyingState: {
        decision: "accept",
        target_mode: "update",
        target_node_id: refs.node_ids[0]!,
        started_at: 3_000,
      },
      ctx,
    });

    expect(update).toHaveBeenCalledWith("semn_aaaaaaaaaaaaaaaa", {
      description: "Deploys stabilize after rollback plans are documented.",
      confidence: 0.75,
      source_episode_ids: ["ep_aaaaaaaaaaaaaaaa"],
      last_verified_at: 2_000,
      embedding: expect.any(Float32Array),
      archived: false,
    });
  });

  it("archives the target on invalidate", async () => {
    const refs = updateRefs();
    const update = vi.fn(async () => ({}));
    const handler = createNewInsightReviewQueueHandler();
    const ctx = ctxWith({
      semanticNodeRepository: {
        get: vi.fn(async () => ({})),
        update,
      } as never,
    });

    await handler.apply({
      item: itemFor(refs),
      refs,
      resolution: { decision: "invalidate" },
      applyingState: {
        decision: "invalidate",
        target_mode: "update",
        target_node_id: refs.node_ids[0]!,
        started_at: 3_000,
      },
      ctx,
    });

    expect(update).toHaveBeenCalledWith("semn_aaaaaaaaaaaaaaaa", {
      archived: true,
    });
  });

  it("does no apply work on dismiss", async () => {
    const refs = insertRefs();
    const insert = vi.fn();
    const handler = createNewInsightReviewQueueHandler();

    await handler.apply({
      item: itemFor(refs),
      refs,
      resolution: { decision: "dismiss" },
      applyingState: null,
      ctx: ctxWith({
        semanticNodeRepository: {
          insert,
        } as never,
      }),
    });

    expect(insert).not.toHaveBeenCalled();
  });

  it("rejects malformed legacy refs", () => {
    expect(() =>
      newInsightReviewRefsSchema.parse({
        node_ids: ["semn_aaaaaaaaaaaaaaaa"],
      }),
    ).toThrow();
  });
});
