import { describe, expect, it } from "vitest";

import { checkRelationshipLabelGrounding } from "./memory-write-relationship-gate.js";

describe("checkRelationshipLabelGrounding", () => {
  it("requires grounding for strict relationship labels", () => {
    const result = checkRelationshipLabelGrounding({
      text: "Nora's sibling is Priya.",
    });

    expect(result.grounded).toBe(false);
    expect(result.protectedLabels).toEqual(["sibling"]);
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
