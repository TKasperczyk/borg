import { describe, expect, it } from "vitest";

import {
  HEADCOUNT_SET_GROUNDING_PROMPT,
  RELATIONSHIP_LABELS_PROMPT,
  RELATIONSHIP_LABEL_JUSTIFICATION_PROMPT,
  RELATIONSHIP_LABEL_WRITE_GROUNDING_PROMPT,
} from "./relationship-labels.js";

describe("relationship claim prompt guidance", () => {
  it("describes structured language-agnostic relationship claims", () => {
    const prompt = [
      RELATIONSHIP_LABELS_PROMPT,
      RELATIONSHIP_LABEL_JUSTIFICATION_PROMPT,
      RELATIONSHIP_LABEL_WRITE_GROUNDING_PROMPT,
      HEADCOUNT_SET_GROUNDING_PROMPT,
    ].join("\n");

    expect(prompt).toContain("relationship_claim");
    expect(prompt).toContain("label_family");
    expect(prompt).toContain("requires_grounding=true");
    expect(prompt).toContain("evidence_relational_slot_ids");
    expect(prompt).toContain("evidence_stream_entry_ids");
    expect(prompt).toContain("any language");
  });

  it("does not contain the removed English relationship word list", () => {
    const prompt = [
      RELATIONSHIP_LABELS_PROMPT,
      RELATIONSHIP_LABEL_JUSTIFICATION_PROMPT,
      RELATIONSHIP_LABEL_WRITE_GROUNDING_PROMPT,
      HEADCOUNT_SET_GROUNDING_PROMPT,
    ]
      .join("\n")
      .toLowerCase();

    for (const word of [
      "sibling",
      "siblings",
      "spouse",
      "spouses",
      "parent",
      "parents",
      "child",
      "children",
      "wife",
      "mother",
      "father",
    ]) {
      expect(prompt).not.toContain(word);
    }
  });
});
