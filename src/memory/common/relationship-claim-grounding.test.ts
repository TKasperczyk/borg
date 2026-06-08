import { describe, expect, it } from "vitest";

import { createEntityId, createRelationalSlotId, createStreamEntryId } from "../../util/ids.js";
import { checkRelationshipClaimGrounding } from "./relationship-claim-grounding.js";
import type { RelationshipClaim } from "./relationship-claims.js";

function relationshipClaim(overrides: Partial<RelationshipClaim> = {}): RelationshipClaim {
  return {
    label_family: "kinship",
    subject_entity_id: null,
    object_entity_id: null,
    object_text: "mi hermana",
    requires_grounding: true,
    evidence_relational_slot_ids: [],
    evidence_stream_entry_ids: [],
    ...overrides,
  };
}

describe("checkRelationshipClaimGrounding", () => {
  it("passes writes with no declared relationship claims", () => {
    const result = checkRelationshipClaimGrounding({});

    expect(result.grounded).toBe(true);
    expect(result.claims).toEqual([]);
    expect(result.ungroundedClaims).toEqual([]);
  });

  it("rejects a required relationship claim without accepted evidence", () => {
    const claim = relationshipClaim();
    const result = checkRelationshipClaimGrounding({
      claims: [claim],
    });

    expect(result.grounded).toBe(false);
    expect(result.ungroundedClaims).toEqual([claim]);
  });

  it("does not interpret non-English object_text when deciding grounding", () => {
    const streamEntryId = createStreamEntryId();
    const ungrounded = relationshipClaim({ object_text: "我的妻子" });
    const grounded = relationshipClaim({
      object_text: "我的妻子",
      evidence_stream_entry_ids: [streamEntryId],
    });

    expect(checkRelationshipClaimGrounding({ claims: [ungrounded] }).grounded).toBe(false);
    expect(
      checkRelationshipClaimGrounding({
        claims: [grounded],
        relationshipEvidenceStreamEntryTrust: (id) =>
          id === streamEntryId ? { allowed: true } : { allowed: false, reason: "missing" },
      }).grounded,
    ).toBe(true);
  });

  it("accepts required relationship claims grounded by established roster slot evidence", () => {
    const slotId = createRelationalSlotId();
    const claim = relationshipClaim({
      evidence_relational_slot_ids: [slotId],
    });
    const result = checkRelationshipClaimGrounding({
      claims: [claim],
      participantRoster: {
        participants: [
          {
            entity_id: createEntityId(),
            display_name: "<person A>",
            known_relationships: ["family:<person B>"],
            audience_role: "speaker",
            relationship_source: `relational_slot:${slotId}`,
          },
        ],
        non_chat_subjects: [],
        unknown_or_uncertain: [],
      },
    });

    expect(result.grounded).toBe(true);
    expect(result.acceptedRelationalSlotIds).toEqual([slotId]);
  });

  it("accepts required relationship claims grounded by trusted user stream evidence", () => {
    const streamEntryId = createStreamEntryId();
    const claim = relationshipClaim({
      evidence_stream_entry_ids: [streamEntryId],
    });
    const result = checkRelationshipClaimGrounding({
      claims: [claim],
      relationshipEvidenceStreamEntryTrust: (id) =>
        id === streamEntryId ? { allowed: true } : { allowed: false, reason: "missing" },
    });

    expect(result.grounded).toBe(true);
    expect(result.acceptedStreamEntryIds).toEqual([streamEntryId]);
  });

  it("rejects required relationship claims grounded only by contested or quarantined roster slots", () => {
    const contestedSlotId = createRelationalSlotId();
    const quarantinedSlotId = createRelationalSlotId();
    const result = checkRelationshipClaimGrounding({
      claims: [
        relationshipClaim({
          evidence_relational_slot_ids: [contestedSlotId, quarantinedSlotId],
        }),
      ],
      participantRoster: {
        participants: [],
        non_chat_subjects: [],
        unknown_or_uncertain: [
          {
            entity_id: null,
            display_name: "uncertain relation",
            known_relationships: ["family:uncertain"],
            reason: "relational_slot_state:contested",
            relationship_source: `relational_slot:${contestedSlotId}`,
            relationship_sources: [`relational_slot:${contestedSlotId}`],
          },
          {
            entity_id: null,
            display_name: "quarantined relation",
            known_relationships: ["family:quarantined"],
            reason: "relational_slot_state:quarantined",
            relationship_source: `relational_slot:${quarantinedSlotId}`,
            relationship_sources: [`relational_slot:${quarantinedSlotId}`],
          },
        ],
      },
    });

    expect(result.grounded).toBe(false);
    expect(result.rejectedRelationalSlotIds).toEqual([contestedSlotId, quarantinedSlotId]);
  });

  it("rejects assistant stream evidence", () => {
    const assistantStreamEntryId = createStreamEntryId();
    const result = checkRelationshipClaimGrounding({
      claims: [
        relationshipClaim({
          evidence_stream_entry_ids: [assistantStreamEntryId],
        }),
      ],
      relationshipEvidenceStreamEntryTrust: () => ({
        allowed: false,
        reason: "not_user_msg",
      }),
    });

    expect(result.grounded).toBe(false);
    expect(result.rejectedStreamEntryIds).toEqual([
      {
        id: assistantStreamEntryId,
        reason: "not_user_msg",
      },
    ]);
  });

  it("rejects stream evidence outside the allowed source bundle", () => {
    const sourceBundleEntryId = createStreamEntryId();
    const outsideEntryId = createStreamEntryId();
    const trust = () => ({
      allowed: true,
    });
    const result = checkRelationshipClaimGrounding({
      claims: [
        relationshipClaim({
          evidence_stream_entry_ids: [outsideEntryId],
        }),
      ],
      allowedRelationshipEvidenceStreamEntryIds: new Set([sourceBundleEntryId]),
      relationshipEvidenceStreamEntryTrust: trust,
    });

    expect(result.grounded).toBe(false);
    expect(result.rejectedStreamEntryIds).toEqual([
      {
        id: outsideEntryId,
        reason: "not_in_source_bundle",
      },
    ]);
  });
});
