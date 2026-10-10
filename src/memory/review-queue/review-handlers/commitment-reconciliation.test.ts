import { describe, expect, it } from "vitest";

import { createCommitmentId, createEntityId, createStreamEntryId } from "../../../util/ids.js";
import {
  COMMITMENT_RECONCILIATION_REVIEW_SUBKINDS,
  commitmentReconciliationJudgmentSchema,
  commitmentReconciliationModelJudgmentSchema,
  commitmentReconciliationReviewRefsSchema,
  commitmentReconciliationSubkindSchema,
  createCommitmentReconciliationReviewQueueHandler,
  deriveCommitmentReconciliationJudgment,
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
    const originAudienceIds = [firstAudienceId, secondAudienceId];
    const authorizationAudienceIds = [...originAudienceIds].sort();

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
        // Origins keep member chronology; authorization IDs use lexical set order.
        originAudienceEntityIds: originAudienceIds,
        privateToEntityIds: authorizationAudienceIds,
        publicToEntityIds: [],
      },
    });

    expect(refs.subkind).toBe("cross_scope_conflict");
    expect(refs.source_stream_entry_ids).toEqual([firstEntryId, secondEntryId]);
    expect(refs.disclosure_label).toEqual({
      disclosureClass: "relationship_private",
      originAudienceEntityIds: originAudienceIds,
      privateToEntityIds: authorizationAudienceIds,
      publicToEntityIds: [],
    });
  });
});

describe("commitment reconciliation model judgment", () => {
  it("derives the superseded set as the complement of the survivor", () => {
    const [first, second, third] = [
      createCommitmentId(),
      createCommitmentId(),
      createCommitmentId(),
    ];
    const judgment = deriveCommitmentReconciliationJudgment(
      commitmentReconciliationModelJudgmentSchema.parse({
        commitment_ids: [first, second, third],
        resolution: "supersede_to_survivor",
        survivor_commitment_id: second,
        reason: "Redundant restatements of one commitment.",
      }),
    );

    expect(judgment.superseded_commitment_ids).toEqual([first, third]);
    // The derived judgment satisfies the stored schema's partition invariant.
    expect(commitmentReconciliationJudgmentSchema.safeParse(judgment).success).toBe(true);
  });

  it("derives an empty superseded set when nothing is superseded", () => {
    const ids = [createCommitmentId(), createCommitmentId()];

    for (const resolution of ["keep_independent", "conflict"] as const) {
      const judgment = deriveCommitmentReconciliationJudgment(
        commitmentReconciliationModelJudgmentSchema.parse({
          commitment_ids: ids,
          resolution,
          survivor_commitment_id: null,
          reason: "Distinct commitments.",
        }),
      );

      expect(judgment.superseded_commitment_ids).toEqual([]);
      expect(commitmentReconciliationJudgmentSchema.safeParse(judgment).success).toBe(true);
    }
  });

  it("rejects a model payload that restates the derived superseded set", () => {
    const [first, second] = [createCommitmentId(), createCommitmentId()];

    expect(
      commitmentReconciliationModelJudgmentSchema.safeParse({
        commitment_ids: [first, second],
        resolution: "supersede_to_survivor",
        survivor_commitment_id: second,
        superseded_commitment_ids: [first],
        reason: "Redundant.",
      }).success,
    ).toBe(false);
  });

  it("still requires the survivor to be one of the commitment ids", () => {
    const [first, second] = [createCommitmentId(), createCommitmentId()];

    expect(
      commitmentReconciliationModelJudgmentSchema.safeParse({
        commitment_ids: [first, second],
        resolution: "supersede_to_survivor",
        survivor_commitment_id: createCommitmentId(),
        reason: "Redundant.",
      }).success,
    ).toBe(false);
  });
});
