import { describe, expect, it } from "vitest";

import { FixedClock } from "../../../util/clock.js";
import type { ReviewHandlerContext, ReviewQueueItem } from "../review-queue.js";
import {
  beliefRevisionReviewRefsSchema,
  createBeliefRevisionReviewQueueHandler,
  type BeliefRevisionReviewRefs,
} from "./belief-revision.js";

const refs = beliefRevisionReviewRefsSchema.parse({
  target_type: "semantic_node",
  target_id: "semn_aaaaaaaaaaaaaaaa",
  invalidated_edge_id: "seme_aaaaaaaaaaaaaaaa",
  dependency_path_edge_ids: ["seme_aaaaaaaaaaaaaaaa"],
  surviving_support_edge_ids: [],
  evidence_episode_ids: ["ep_aaaaaaaaaaaaaaaa"],
  audience_entity_id: null,
});

function itemFor(inputRefs: BeliefRevisionReviewRefs): ReviewQueueItem {
  return {
    id: 1,
    kind: "belief_revision",
    refs: inputRefs,
    reason: "support chain collapsed",
    created_at: 1_000,
    resolved_at: null,
    resolution: null,
  };
}

const ctx = {
  clock: new FixedClock(2_000),
} as unknown as ReviewHandlerContext;

describe("belief revision review handler", () => {
  it("keeps manual resolution dismiss-only and sqlite scoped", () => {
    const handler = createBeliefRevisionReviewQueueHandler();

    expect([...handler.allowedResolutions]).toEqual(["dismiss"]);
    expect(
      handler.transactionScope({
        item: itemFor(refs),
        refs,
        resolution: { decision: "dismiss" },
        ctx,
      }),
    ).toBe("sqlite");
    expect(
      handler.apply({
        item: itemFor(refs),
        refs,
        resolution: { decision: "dismiss" },
        applyingState: null,
        ctx,
      }),
    ).toBeUndefined();
  });

  it("parses known process-private state strictly", () => {
    expect(
      beliefRevisionReviewRefsSchema.parse({
        ...refs,
        __borg_belief_revision_claim: {
          run_id: "run-1",
          claimed_at: 2_000,
        },
        belief_revision_failure_count: 1,
        belief_revision_last_error: "parse failed",
      }),
    ).toEqual(
      expect.objectContaining({
        belief_revision_failure_count: 1,
      }),
    );
    expect(() =>
      beliefRevisionReviewRefsSchema.parse({
        ...refs,
        unexpected: true,
      }),
    ).toThrow();
  });
});
