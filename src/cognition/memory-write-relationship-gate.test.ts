import { describe, expect, it } from "vitest";

import { createEntityId, createRelationalSlotId, createStreamEntryId } from "../util/ids.js";
import { checkRelationshipLabelGrounding } from "./memory-write-relationship-gate.js";

describe("checkRelationshipLabelGrounding", () => {
  it("requires grounding for strict relationship labels", () => {
    const result = checkRelationshipLabelGrounding({
      text: "Nora's sibling is Priya.",
    });

    expect(result.grounded).toBe(false);
    expect(result.protectedLabels).toEqual(["sibling"]);
  });

  it("rejects placeholder sibling assignments without relationship evidence", () => {
    const result = checkRelationshipLabelGrounding({
      text: "<person A> is <person B>'s sibling.",
    });

    expect(result.grounded).toBe(false);
    expect(result.protectedLabels).toEqual(["sibling"]);
  });

  it("accepts placeholder sibling assignments grounded by relational slot evidence", () => {
    const slotId = createRelationalSlotId();
    const result = checkRelationshipLabelGrounding({
      text: "<person A> is <person B>'s sibling.",
      participantRoster: {
        participants: [
          {
            entity_id: createEntityId(),
            display_name: "<person A>",
            known_relationships: ["sibling:<person B>"],
            audience_role: "speaker",
            relationship_source: `relational_slot:${slotId}`,
          },
        ],
        non_chat_subjects: [],
        unknown_or_uncertain: [],
      },
      relationshipEvidenceRelationalSlotIds: [slotId],
    });

    expect(result.grounded).toBe(true);
    expect(result.protectedLabels).toEqual(["sibling"]);
    expect(result.acceptedRelationalSlotIds).toEqual([slotId]);
  });

  it("accepts placeholder sibling assignments grounded by trusted user stream evidence", () => {
    const streamEntryId = createStreamEntryId();
    const result = checkRelationshipLabelGrounding({
      text: "<person A> is <person B>'s sibling.",
      relationshipEvidenceStreamEntryIds: [streamEntryId],
      relationshipEvidenceStreamEntryTrust: (id) =>
        id === streamEntryId ? { allowed: true } : { allowed: false, reason: "missing" },
    });

    expect(result.grounded).toBe(true);
    expect(result.protectedLabels).toEqual(["sibling"]);
    expect(result.acceptedStreamEntryIds).toEqual([streamEntryId]);
  });

  it("does not require grounding for medical context nouns", () => {
    const result = checkRelationshipLabelGrounding({
      text: "The doctor appointment is pending, and the patient portal is down.",
    });

    expect(result.grounded).toBe(true);
    expect(result.protectedLabels).toEqual([]);
  });

  it("does not require grounding for adjacent professional appointment nouns", () => {
    const result = checkRelationshipLabelGrounding({
      text: "I haven't booked the dentist appointment yet.",
    });

    expect(result.grounded).toBe(true);
    expect(result.protectedLabels).toEqual([]);
  });

  it("does not hard-gate contextual role labels", () => {
    const result = checkRelationshipLabelGrounding({
      text: "The design partner and rollout owner are tracked for this project.",
    });

    expect(result.grounded).toBe(true);
    expect(result.protectedLabels).toEqual([]);
  });
});
