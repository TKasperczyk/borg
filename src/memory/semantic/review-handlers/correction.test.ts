import { describe, expect, it, vi } from "vitest";

import { FixedClock } from "../../../util/clock.js";
import {
  type CorrectionReviewRefs,
  correctionReviewRefsSchema,
  createCorrectionReviewHandler,
} from "./correction.js";
import type { ReviewHandlerContext, ReviewQueueItem } from "../review-queue.js";

const ctx = {
  clock: new FixedClock(5_000),
} as unknown as ReviewHandlerContext;

function itemFor(refs: CorrectionReviewRefs): ReviewQueueItem {
  return {
    id: 1,
    kind: "correction",
    refs,
    reason: "user corrected the record",
    created_at: 1_000,
    resolved_at: null,
    resolution: null,
  };
}

function parseRefs(raw: Record<string, unknown>): CorrectionReviewRefs {
  return correctionReviewRefsSchema.parse(raw);
}

describe("correction review handler", () => {
  it("uses target_type to route episode corrections through applying state", async () => {
    const applier = vi.fn();
    const handler = createCorrectionReviewHandler({ applyCorrection: applier });
    const refs = parseRefs({
      target_type: "episode",
      target_id: "ep_aaaaaaaaaaaaaaaa",
      patch: {
        narrative: "Corrected episode narrative.",
      },
    });
    const item = itemFor(refs);

    expect(
      handler.transactionScope({
        item,
        refs,
        resolution: { decision: "accept" },
        ctx,
      }),
    ).toBe("cross_store_applying_state");
    expect(
      handler.transactionScope({
        item,
        refs,
        resolution: { decision: "reject" },
        ctx,
      }),
    ).toBe("sqlite");

    await handler.apply({
      item,
      refs,
      resolution: { decision: "accept" },
      applyingState: { decision: "accept", started_at: 5_000 },
      ctx,
    });

    expect(applier).toHaveBeenCalledWith(expect.objectContaining({ refs }));
  });

  it("uses target_type to route semantic-node corrections through applying state", () => {
    const handler = createCorrectionReviewHandler({ applyCorrection: vi.fn() });
    const refs = parseRefs({
      target_type: "semantic_node",
      target_id: "semn_aaaaaaaaaaaaaaaa",
      patch: {
        description: "Corrected semantic node description.",
      },
    });

    expect(
      handler.transactionScope({
        item: itemFor(refs),
        refs,
        resolution: { decision: "accept" },
        ctx,
      }),
    ).toBe("cross_store_applying_state");
  });

  it("keeps identity and open-question corrections sqlite scoped", () => {
    const handler = createCorrectionReviewHandler({ applyCorrection: vi.fn() });
    const valueRefs = parseRefs({
      target_type: "value",
      target_id: "val_aaaaaaaaaaaaaaaa",
      patch: {
        description: "Prefer grounded claims.",
      },
    });
    const openQuestionRefs = parseRefs({
      target_type: "open_question",
      target_id: "oq_aaaaaaaaaaaaaaaa",
      patch: {
        status: "abandoned",
      },
    });

    expect(
      handler.transactionScope({
        item: itemFor(valueRefs),
        refs: valueRefs,
        resolution: { decision: "accept" },
        ctx,
      }),
    ).toBe("sqlite");
    expect(
      handler.transactionScope({
        item: itemFor(openQuestionRefs),
        refs: openQuestionRefs,
        resolution: { decision: "accept" },
        ctx,
      }),
    ).toBe("sqlite");
  });

  it("rejects semantic-edge correction rows", async () => {
    const handler = createCorrectionReviewHandler({ applyCorrection: vi.fn() });
    const refs = parseRefs({
      target_type: "semantic_edge",
      target_id: "seme_aaaaaaaaaaaaaaaa",
      patch: {},
    });

    await expect(
      handler.apply({
        item: itemFor(refs),
        refs,
        resolution: { decision: "accept" },
        applyingState: null,
        ctx,
      }),
    ).rejects.toMatchObject({
      code: "SEMANTIC_EDGE_CORRECTION_UNSUPPORTED",
    });
  });

  it("fails loudly when target_type and target_id disagree", () => {
    expect(() =>
      correctionReviewRefsSchema.parse({
        target_type: "episode",
        target_id: "semn_aaaaaaaaaaaaaaaa",
        patch: {
          narrative: "Wrong id family.",
        },
      }),
    ).toThrow();
  });
});
