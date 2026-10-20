import { describe, expect, it, vi } from "vitest";

import { openDatabase } from "../../../storage/sqlite/index.js";
import { FixedClock } from "../../../util/clock.js";
import { createStreamEntryId } from "../../../util/ids.js";
import { identityMigrations } from "../../identity/migrations.js";
import { IdentityEventRepository } from "../../identity/repository.js";
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

  it("keeps the overwritten episode fields in one identity event per review item", async () => {
    const refs = misattributionReviewRefsSchema.parse({
      target_type: "episode",
      target_id: "ep_aaaaaaaaaaaaaaaa",
      patch: {
        participants: ["team", "Alex"],
        narrative: "Alex led the review.",
      },
      proposed_provenance: {
        kind: "offline",
        process: "overseer",
      },
    });
    const before = {
      id: "ep_aaaaaaaaaaaaaaaa",
      narrative: "Sam led the review.",
      participants: ["team", "Sam"],
      tags: ["review"],
      source_stream_ids: ["strm_aaaaaaaaaaaaaaaa"],
      embedding: new Float32Array([0.25, 0.5]),
      updated_at: 1_000,
    };
    const after = {
      ...before,
      narrative: "Alex led the review.",
      participants: ["team", "Alex"],
      updated_at: 5_000,
    };
    const get = vi.fn(async () => before);
    const update = vi.fn(async () => after);
    const db = openDatabase(":memory:", { migrations: identityMigrations });
    const identityEventRepository = new IdentityEventRepository({
      db,
      clock: new FixedClock(5_000),
    });
    const handler = createMisattributionReviewQueueHandler();
    const ctx = ctxWith({
      episodicRepository: { get, update } as never,
      identityEventRepository,
    });

    try {
      for (let attempt = 0; attempt < 2; attempt += 1) {
        await handler.apply({
          item: itemFor(refs),
          refs,
          resolution: { decision: "accept" },
          applyingState: null,
          ctx,
        });
      }

      expect(get).toHaveBeenCalledWith("ep_aaaaaaaaaaaaaaaa", { includeArchived: true });
      expect(identityEventRepository.list({ recordType: "episode" })).toEqual([
        expect.objectContaining({
          record_id: "ep_aaaaaaaaaaaaaaaa",
          action: "misattribution_apply",
          review_item_id: 1,
          reason: "attribution should be repaired",
          provenance: { kind: "offline", process: "overseer" },
          old_value: {
            narrative: "Sam led the review.",
            participants: ["team", "Sam"],
            updated_at: 1_000,
          },
          new_value: {
            narrative: "Alex led the review.",
            participants: ["team", "Alex"],
            updated_at: 5_000,
          },
        }),
      ]);
    } finally {
      db.close();
    }
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
