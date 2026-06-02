import { describe, expect, it } from "vitest";

import { createCommitmentId, createEntityId } from "../../../util/ids.js";
import {
  COMMITMENT_RECONCILIATION_REVIEW_SUBKINDS,
  commitmentReconciliationReviewRefsSchema,
  commitmentReconciliationSubkindSchema,
  createCommitmentReconciliationReviewQueueHandler,
} from "./commitment-reconciliation.js";

describe("commitment reconciliation review handler", () => {
  it("validates conflict refs and exposes a manual no-op handler", () => {
    const firstId = createCommitmentId();
    const secondId = createCommitmentId();
    const audienceId = createEntityId();
    const refs = commitmentReconciliationReviewRefsSchema.parse({
      target_type: "commitment_reconciliation",
      subkind: "conflict",
      commitment_ids: [firstId, secondId],
      scope_key: {
        kind: "participant_preference",
        restricted_audience: audienceId,
        made_to_entity: null,
        about_entity: null,
      },
      reason: "The commitments conflict.",
      members: [
        {
          id: firstId,
          kind: "participant_preference",
          type: "preference",
          directive_family: "reply_style_a",
        },
        {
          id: secondId,
          kind: "participant_preference",
          type: "preference",
          directive_family: "reply_style_b",
        },
      ],
      judgment: {
        commitment_ids: [firstId, secondId],
        resolution: "conflict",
        survivor_commitment_id: null,
        superseded_commitment_ids: [],
        reason: "The commitments conflict.",
      },
    });
    const handler = createCommitmentReconciliationReviewQueueHandler();

    expect(COMMITMENT_RECONCILIATION_REVIEW_SUBKINDS).toEqual(["conflict"]);
    expect(commitmentReconciliationSubkindSchema.safeParse("conflict").success).toBe(true);
    expect(refs.commitment_ids).toEqual([firstId, secondId]);
    expect(handler.kind).toBe("commitment_reconciliation");
    expect(handler.allowedResolutions.has("accept")).toBe(true);
    expect(handler.allowedResolutions.has("keep")).toBe(true);
    expect(handler.transactionScope({} as never)).toBe("sqlite");
    expect(handler.apply({} as never)).toBeUndefined();
  });
});
