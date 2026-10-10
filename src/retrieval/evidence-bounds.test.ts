import { describe, expect, it } from "vitest";

import { MAX_RECALLED_EVIDENCE_TEXT_CHARS, clipRecalledEvidenceText } from "./evidence-bounds.js";

describe("clipRecalledEvidenceText", () => {
  it("returns short text unchanged", () => {
    expect(clipRecalledEvidenceText("krótki tekst")).toBe("krótki tekst");
  });

  it("never ends an excerpt on the high half of an emoji", () => {
    // Prod shape (2026-09-09): "### 🆕" with the emoji straddling code units 179/180.
    const head = "x".repeat(MAX_RECALLED_EVIDENCE_TEXT_CHARS - 1);
    const text = `${head}🆕 dalej`;
    const clipped = clipRecalledEvidenceText(text);

    expect(clipped).toBe(head);
    expect(clipped).not.toMatch(/[\uD800-\uDFFF]$/u);
    expect(() => new TextEncoder().encode(JSON.stringify(clipped))).not.toThrow();
  });

  it("keeps a whole emoji when the cut lands after it", () => {
    const head = "x".repeat(MAX_RECALLED_EVIDENCE_TEXT_CHARS - 2);
    const clipped = clipRecalledEvidenceText(`${head}🆕 dalej`);

    expect(clipped).toBe(`${head}🆕`);
    expect(clipped.length).toBe(MAX_RECALLED_EVIDENCE_TEXT_CHARS);
  });
});
