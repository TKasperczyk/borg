import { describe, expect, it } from "vitest";

import { createCommitmentId, createEntityId, createStreamEntryId } from "../../../util/ids.js";
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

    expect(COMMITMENT_RECONCILIATION_REVIEW_SUBKINDS).toEqual([
      "conflict",
      "cross_scope_conflict",
      "cross_scope_redundancy",
    ]);
    expect(commitmentReconciliationSubkindSchema.safeParse("conflict").success).toBe(true);
    expect(refs.commitment_ids).toEqual([firstId, secondId]);
    expect(handler.kind).toBe("commitment_reconciliation");
    expect(handler.allowedResolutions.has("accept")).toBe(true);
    expect(handler.allowedResolutions.has("keep")).toBe(true);
    expect(handler.transactionScope({} as never)).toBe("sqlite");
    expect(handler.apply({} as never)).toBeUndefined();
  });

  it("validates enriched cross-scope awareness refs with disclosure labels", () => {
    const firstId = createCommitmentId();
    const secondId = createCommitmentId();
    const firstAudienceId = createEntityId();
    const secondAudienceId = createEntityId();
    const firstEntryId = createStreamEntryId();
    const secondEntryId = createStreamEntryId();
    const sortedAudienceIds = [firstAudienceId, secondAudienceId].sort();

    const refs = commitmentReconciliationReviewRefsSchema.parse({
      target_type: "commitment_reconciliation",
      subkind: "cross_scope_conflict",
      commitment_ids: [firstId, secondId],
      scope_key: {
        kind: "participant_preference",
        restricted_audience: null,
        made_to_entity: null,
        about_entity: null,
      },
      detection_key: {
        kind: "participant_preference",
        about_entity: null,
        directive_family: "reply_style",
      },
      reason: "The cross-scope commitments conflict.",
      members: [
        {
          id: firstId,
          kind: "participant_preference",
          type: "preference",
          directive_family: "reply_style",
          directive: "Keep Alice replies short.",
          scope_key: {
            kind: "participant_preference",
            restricted_audience: firstAudienceId,
            made_to_entity: null,
            about_entity: null,
          },
          source_stream_entry_ids: [firstEntryId],
          disclosure_label: {
            disclosureClass: "relationship_private",
            originAudienceEntityIds: [firstAudienceId],
            privateToEntityIds: [firstAudienceId],
            publicToEntityIds: [],
          },
        },
        {
          id: secondId,
          kind: "participant_preference",
          type: "preference",
          directive_family: "reply_style",
          directive: "Give Bob extensive replies.",
          scope_key: {
            kind: "participant_preference",
            restricted_audience: secondAudienceId,
            made_to_entity: null,
            about_entity: null,
          },
          source_stream_entry_ids: [secondEntryId],
          disclosure_label: {
            disclosureClass: "relationship_private",
            originAudienceEntityIds: [secondAudienceId],
            privateToEntityIds: [secondAudienceId],
            publicToEntityIds: [],
          },
        },
      ],
      judgment: {
        commitment_ids: [firstId, secondId],
        resolution: "conflict",
        survivor_commitment_id: null,
        superseded_commitment_ids: [],
        reason: "The cross-scope commitments conflict.",
      },
      source_stream_entry_ids: [firstEntryId, secondEntryId],
      disclosure_label: {
        disclosureClass: "relationship_private",
        originAudienceEntityIds: sortedAudienceIds,
        privateToEntityIds: sortedAudienceIds,
        publicToEntityIds: [],
      },
    });

    expect(refs.subkind).toBe("cross_scope_conflict");
    expect(refs.source_stream_entry_ids).toEqual([firstEntryId, secondEntryId]);
    expect(refs.disclosure_label?.originAudienceEntityIds).toEqual(sortedAudienceIds);
  });
});
