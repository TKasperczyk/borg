import { describe, expect, it } from "vitest";

import { checkRelationshipLabelGrounding } from "./memory-write-relationship-gate.js";

describe("checkRelationshipLabelGrounding", () => {
  it("requires grounding for strict relationship labels", () => {
    const result = checkRelationshipLabelGrounding({
      text: "Use the sibling context for planning.",
    });

    expect(result.grounded).toBe(false);
    expect(result.protectedLabels).toEqual(["sibling"]);
  });

  it("does not hard-gate contextual role labels", () => {
    const result = checkRelationshipLabelGrounding({
      text: "The design partner and rollout owner are tracked for this project.",
    });

    expect(result.grounded).toBe(true);
    expect(result.protectedLabels).toEqual([]);
  });
});
