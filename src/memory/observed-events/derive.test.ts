import { describe, expect, it } from "vitest";

import { deriveObservedEventDimensions, type ObservedEventDerivationInput } from "./derive.js";

describe("deriveObservedEventDimensions", () => {
  it("maps only structural disposition and classification kind enums to observed-event dimensions", () => {
    const cases: Array<{
      input: ObservedEventDerivationInput;
      expected: ReturnType<typeof deriveObservedEventDimensions>;
    }> = [
      {
        input: {
          disposition: "quarantine",
          classificationKind: "frame_assignment_claim",
        },
        expected: {
          stance: "rejected_frame",
          taint: "quarantined",
          beliefEffect: "unchanged",
          classificationKind: "frame_assignment_claim",
        },
      },
      {
        input: {
          disposition: "trusted_operator_control",
          classificationKind: "system_prompt_claim",
        },
        expected: {
          stance: "accepted_frame",
          taint: "none",
          beliefEffect: "updated",
          classificationKind: "system_prompt_claim",
        },
      },
      {
        input: {
          disposition: "none",
          classificationKind: "normal",
        },
        expected: {
          stance: "noted_frame",
          taint: "none",
          beliefEffect: "unchanged",
          classificationKind: "normal",
        },
      },
    ];

    for (const { input, expected } of cases) {
      expect(Object.keys(input)).toEqual(["disposition", "classificationKind"]);
      expect(deriveObservedEventDimensions(input)).toEqual(expected);
    }
  });
});
