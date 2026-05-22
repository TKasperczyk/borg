import { describe, expect, it } from "vitest";

import {
  CONTEXTUAL_RELATIONSHIP_LABELS,
  PROTECTED_RELATIONSHIP_LABELS,
  STRICT_RELATIONSHIP_LABELS,
  protectedRelationshipLabelsInText,
} from "./relationship-labels.js";

describe("relationship label tiers", () => {
  it("uses strict labels as the protected hard-gated labels", () => {
    expect(PROTECTED_RELATIONSHIP_LABELS).toEqual(STRICT_RELATIONSHIP_LABELS);
    expect(PROTECTED_RELATIONSHIP_LABELS).toEqual([
      "sibling",
      "spouse",
      "parent",
      "child",
      "caregiver",
    ]);
  });

  it("matches strict labels in memory-write text", () => {
    expect(protectedRelationshipLabelsInText("Nora's sibling is Priya.")).toEqual(["sibling"]);
    expect(protectedRelationshipLabelsInText("Mom's spouse is asleep.")).toEqual(["spouse"]);
    expect(protectedRelationshipLabelsInText("My parent lives upstate.")).toEqual(["parent"]);
    expect(protectedRelationshipLabelsInText("My child is at school.")).toEqual(["child"]);
  });

  it("does not hard-gate contextual role labels", () => {
    expect(CONTEXTUAL_RELATIONSHIP_LABELS).toContain("partner");
    expect(protectedRelationshipLabelsInText("The design partner owns rollout notes.")).toEqual([]);
  });

  it("does not hard-gate medical or professional context nouns", () => {
    expect(protectedRelationshipLabelsInText("The doctor appointment is pending.")).toEqual([]);
    expect(protectedRelationshipLabelsInText("Mom's doctor scheduled the appointment.")).toEqual(
      [],
    );
    expect(protectedRelationshipLabelsInText("No doctor calls this week.")).toEqual([]);
    expect(protectedRelationshipLabelsInText("The patient portal is down.")).toEqual([]);
    expect(protectedRelationshipLabelsInText("Patient portal is broken.")).toEqual([]);
    expect(protectedRelationshipLabelsInText("Patient paperwork is waiting.")).toEqual([]);
    expect(
      protectedRelationshipLabelsInText("I haven't booked the dentist appointment yet."),
    ).toEqual([]);
  });
});
