import { describe, expect, it, vi } from "vitest";

import { FixedClock } from "../../../util/clock.js";
import type { ReviewHandlerContext, ReviewQueueItem } from "../review-queue.js";
import type { SemanticEdge } from "../types.js";
import {
  createIdentityInconsistencyReviewQueueHandler,
  identityInconsistencyReviewRefsSchema,
  type IdentityInconsistencyReviewRefs,
} from "./identity-inconsistency.js";

function itemFor(refs: IdentityInconsistencyReviewRefs): ReviewQueueItem {
  return {
    id: 7,
    kind: "identity_inconsistency",
    refs,
    reason: "identity repair proposed",
    created_at: 1_000,
    resolved_at: null,
    resolution: null,
  };
}

function ctxWith(input: Partial<ReviewHandlerContext>): ReviewHandlerContext {
  return {
    clock: new FixedClock(7_000),
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

describe("identity inconsistency review handler", () => {
  it("reinforces values from evidence episodes", () => {
    const refs = identityInconsistencyReviewRefsSchema.parse({
      target_type: "value",
      target_id: "val_aaaaaaaaaaaaaaaa",
      repair_op: "reinforce",
      evidence_episode_ids: ["ep_aaaaaaaaaaaaaaaa"],
      proposed_provenance: {
        kind: "offline",
        process: "overseer",
      },
    });
    const reinforce = vi.fn();
    const handler = createIdentityInconsistencyReviewQueueHandler();

    handler.apply({
      item: itemFor(refs),
      refs,
      resolution: { decision: "accept" },
      applyingState: null,
      ctx: ctxWith({
        valuesRepository: {
          reinforce,
        } as never,
      }),
    });

    expect(reinforce).toHaveBeenCalledWith(
      "val_aaaaaaaaaaaaaaaa",
      {
        kind: "episodes",
        episode_ids: ["ep_aaaaaaaaaaaaaaaa"],
      },
      7_000,
    );
  });

  it("records trait contradictions by current label", () => {
    const refs = identityInconsistencyReviewRefsSchema.parse({
      target_type: "trait",
      target_id: "trt_aaaaaaaaaaaaaaaa",
      repair_op: "contradict",
      evidence_episode_ids: ["ep_aaaaaaaaaaaaaaaa"],
    });
    const recordContradiction = vi.fn();
    const handler = createIdentityInconsistencyReviewQueueHandler();

    handler.apply({
      item: itemFor(refs),
      refs,
      resolution: { decision: "accept" },
      applyingState: null,
      ctx: ctxWith({
        traitsRepository: {
          get: vi.fn(() => ({ label: "patient" })),
          recordContradiction,
        } as never,
      }),
    });

    expect(recordContradiction).toHaveBeenCalledWith({
      label: "patient",
      provenance: {
        kind: "episodes",
        episode_ids: ["ep_aaaaaaaaaaaaaaaa"],
      },
      timestamp: 7_000,
    });
  });

  it("applies goal patches through the identity service", () => {
    const refs = identityInconsistencyReviewRefsSchema.parse({
      target_type: "goal",
      target_id: "goal_aaaaaaaaaaaaaaaa",
      repair_op: "patch",
      patch: {
        progress_notes: "Reviewed progress.",
        last_progress_ts: 7_000,
      },
      proposed_provenance: {
        kind: "online",
        process: "reflector",
      },
    });
    const updateGoal = vi.fn(() => ({ status: "applied", record: {} }));
    const handler = createIdentityInconsistencyReviewQueueHandler();

    handler.apply({
      item: itemFor(refs),
      refs,
      resolution: { decision: "accept" },
      applyingState: null,
      ctx: ctxWith({
        identityService: {
          updateGoal,
        } as never,
      }),
    });

    expect(updateGoal).toHaveBeenCalledWith(
      "goal_aaaaaaaaaaaaaaaa",
      {
        progress_notes: "Reviewed progress.",
        last_progress_ts: 7_000,
      },
      {
        kind: "online",
        process: "reflector",
      },
      {
        throughReview: true,
        reason: "identity repair proposed",
        reviewItemId: 7,
      },
    );
  });

  it("wraps period rollovers in the autobiographical repository transaction", () => {
    const refs = identityInconsistencyReviewRefsSchema.parse({
      target_type: "autobiographical_period",
      target_id: "abp_aaaaaaaaaaaaaaaa",
      repair_op: "patch",
      patch: {
        end_ts: 7_000,
      },
      proposed_provenance: {
        kind: "offline",
        process: "self-narrator",
      },
      next_period_open_payload: {
        id: "abp_bbbbbbbbbbbbbbbb",
        label: "2026-Q3",
        start_ts: 7_000,
        end_ts: null,
        narrative: "Next period.",
        key_episode_ids: ["ep_aaaaaaaaaaaaaaaa"],
        themes: ["rollover"],
        provenance: {
          kind: "offline",
          process: "self-narrator",
        },
        created_at: 7_000,
        last_updated: 7_000,
      },
    });
    const runInTransaction = vi.fn((callback: () => void) => callback());
    const updatePeriod = vi.fn(() => ({ status: "applied", record: {} }));
    const addPeriod = vi.fn();
    const handler = createIdentityInconsistencyReviewQueueHandler();

    handler.apply({
      item: itemFor(refs),
      refs,
      resolution: { decision: "accept" },
      applyingState: null,
      ctx: ctxWith({
        autobiographicalRepository: {
          runInTransaction,
        } as never,
        identityService: {
          updatePeriod,
          addPeriod,
        } as never,
      }),
    });

    expect(runInTransaction).toHaveBeenCalledOnce();
    expect(updatePeriod).toHaveBeenCalledWith(
      "abp_aaaaaaaaaaaaaaaa",
      { end_ts: 7_000 },
      {
        kind: "offline",
        process: "self-narrator",
      },
      expect.objectContaining({
        throughReview: true,
        reviewItemId: 7,
      }),
    );
    expect(addPeriod).toHaveBeenCalledWith(
      expect.objectContaining({
        id: "abp_bbbbbbbbbbbbbbbb",
      }),
    );
  });

  it("closes semantic-edge inconsistencies", () => {
    const refs = identityInconsistencyReviewRefsSchema.parse({
      target_type: "semantic_edge",
      target_kind: "semantic_edge",
      target_id: edge.id,
      suggested_valid_to: 4_000,
      by_edge_id: "seme_bbbbbbbbbbbbbbbb",
      reason: "edge should be closed",
      source_target_type: "semantic_edge",
      source_target_id: edge.id,
    });
    const invalidated = {
      ...edge,
      valid_to: 4_000,
      invalidated_at: 7_000,
      invalidated_by_edge_id: "seme_bbbbbbbbbbbbbbbb",
      invalidated_by_review_id: 7,
      invalidated_by_process: "review",
      invalidated_reason: "edge should be closed",
    } as SemanticEdge;
    const record = vi.fn();
    const handler = createIdentityInconsistencyReviewQueueHandler();

    handler.apply({
      item: itemFor(refs),
      refs,
      resolution: { decision: "accept" },
      applyingState: null,
      ctx: ctxWith({
        semanticEdgeRepository: {
          getEdge: vi.fn(() => edge),
          invalidateEdge: vi.fn(() => invalidated),
        } as never,
        identityEventRepository: {
          findByReviewKey: vi.fn(() => null),
          record,
        } as never,
      }),
    });

    expect(record).toHaveBeenCalledWith(
      expect.objectContaining({
        record_type: "semantic_edge",
        record_id: edge.id,
        action: "edge_invalidate",
        review_item_id: 7,
      }),
    );
  });

  it("rejects malformed refs", () => {
    expect(() =>
      identityInconsistencyReviewRefsSchema.parse({
        target_type: "goal",
        target_id: "goal_aaaaaaaaaaaaaaaa",
        repair_op: "patch",
      }),
    ).toThrow();
    expect(() =>
      identityInconsistencyReviewRefsSchema.parse({
        target_type: "value",
        target_id: "val_aaaaaaaaaaaaaaaa",
        repair_op: "reinforce",
      }),
    ).toThrow();
  });
});
