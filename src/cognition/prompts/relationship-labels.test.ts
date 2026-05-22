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
      "doctor",
      "patient",
    ]);
  });

  it("matches strict labels in memory-write text", () => {
    expect(protectedRelationshipLabelsInText("The sibling handoff is confirmed.")).toEqual([
      "sibling",
    ]);
  });

  it("does not hard-gate contextual role labels", () => {
    expect(CONTEXTUAL_RELATIONSHIP_LABELS).toContain("partner");
    expect(protectedRelationshipLabelsInText("The design partner owns rollout notes.")).toEqual([]);
  });
});
