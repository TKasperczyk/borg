import { describe, expect, it, vi } from "vitest";

import { FixedClock } from "../../../util/clock.js";
import { createStreamEntryId } from "../../../util/ids.js";
import type { ReviewHandlerContext, ReviewQueueItem } from "../review-queue.js";
import {
  createMisattributionReviewQueueHandler,
  misattributionReviewRefsSchema,
  type MisattributionReviewRefs,
} from "./misattribution.js";

function itemFor(refs: MisattributionReviewRefs): ReviewQueueItem {
  return {
    id: 1,
    kind: "misattribution",
    refs,
    reason: "attribution should be repaired",
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

describe("misattribution review handler", () => {
  it("accepts episode attribution patches", async () => {
    const evidenceStreamId = createStreamEntryId();
    const refs = misattributionReviewRefsSchema.parse({
      target_type: "episode",
      target_id: "ep_aaaaaaaaaaaaaaaa",
      patch: {
        participants: ["team", "Alex"],
        narrative: "Alex led the review.",
        tags: ["review", "alex"],
      },
      evidence_stream_ids: [evidenceStreamId],
      proposed_provenance: {
        kind: "offline",
        process: "overseer",
      },
    });
    const update = vi.fn(async () => ({}));
    const handler = createMisattributionReviewQueueHandler();
    const ctx = ctxWith({
      episodicRepository: {
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

    expect(update).toHaveBeenCalledWith("ep_aaaaaaaaaaaaaaaa", {
      participants: ["team", "Alex"],
      narrative: "Alex led the review.",
      tags: ["review", "alex"],
    });
  });

  it("accepts semantic-node patches with replacement flags", async () => {
    const refs = misattributionReviewRefsSchema.parse({
      target_type: "semantic_node",
      target_id: "semn_aaaaaaaaaaaaaaaa",
      patch: {
        aliases: ["Atlas review"],
        source_episode_ids: ["ep_aaaaaaaaaaaaaaaa"],
      },
    });
    const update = vi.fn(async () => ({}));
    const handler = createMisattributionReviewQueueHandler();

    await handler.apply({
      item: itemFor(refs),
      refs,
      resolution: { decision: "accept" },
      applyingState: null,
      ctx: ctxWith({
        semanticNodeRepository: {
          update,
        } as never,
      }),
    });

    expect(update).toHaveBeenCalledWith("semn_aaaaaaaaaaaaaaaa", {
      aliases: ["Atlas review"],
      source_episode_ids: ["ep_aaaaaaaaaaaaaaaa"],
      replace_aliases: true,
      replace_source_episode_ids: true,
    });
  });

  it("keeps reject and dismiss sqlite scoped", () => {
    const refs = misattributionReviewRefsSchema.parse({
      target_type: "semantic_node",
      target_id: "semn_aaaaaaaaaaaaaaaa",
      patch: {
        label: "Correct label",
      },
    });
    const handler = createMisattributionReviewQueueHandler();

    expect(
      handler.transactionScope({
        item: itemFor(refs),
        refs,
        resolution: { decision: "reject" },
        ctx: ctxWith({}),
      }),
    ).toBe("sqlite");
    expect(
      handler.transactionScope({
        item: itemFor(refs),
        refs,
        resolution: { decision: "dismiss" },
        ctx: ctxWith({}),
      }),
    ).toBe("sqlite");
  });

  it("rejects malformed refs", () => {
    expect(() =>
      misattributionReviewRefsSchema.parse({
        target_type: "episode",
        target_id: "ep_aaaaaaaaaaaaaaaa",
      }),
    ).toThrow();
    expect(() =>
      misattributionReviewRefsSchema.parse({
        target_type: "semantic_node",
        target_id: "semn_aaaaaaaaaaaaaaaa",
        patch: {},
      }),
    ).toThrow();
  });
});
