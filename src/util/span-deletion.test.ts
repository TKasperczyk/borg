import { describe, expect, it } from "vitest";

import { deleteSpans, deleteSpansSentenceAware } from "./span-deletion.js";

describe("deleteSpansSentenceAware", () => {
  it("removes the enclosing sentence when a span is embedded in prose", () => {
    const result = deleteSpansSentenceAware(
      "Keep this. On the walk, Otto was a sand-deranged dog because Marta had been distracted. Keep that.",
      ["a sand-deranged dog"],
    );

    expect(result).toEqual({
      rewrittenText: "Keep this. Keep that.",
      outcome: "clean",
      removedSpans: ["a sand-deranged dog"],
    });
  });

  it("removes every sentence touched by a cross-sentence span", () => {
    const result = deleteSpansSentenceAware("Keep. Bad first. Bad second. End.", [
      "Bad first. Bad second.",
    ]);

    expect(result).toMatchObject({
      rewrittenText: "Keep. End.",
      outcome: "clean",
    });
  });

  it("does not split sentence boundaries at abbreviations", () => {
    const result = deleteSpansSentenceAware("Keep this. Dr. Luis will call. End.", ["Luis"]);

    expect(result).toMatchObject({
      rewrittenText: "Keep this. End.",
      outcome: "clean",
    });
  });

  it("merges overlapping sentence-expanded ranges inside the same sentence", () => {
    const result = deleteSpansSentenceAware("Keep. The answer names Luis and Ana today. End.", [
      "Luis",
      "Ana",
    ]);

    expect(result).toMatchObject({
      rewrittenText: "Keep. End.",
      outcome: "clean",
      removedSpans: ["Luis", "Ana"],
    });
  });

  it("falls back to paragraph deletion when sentence deletion leaves malformed residue", () => {
    const result = deleteSpansSentenceAware("Quick note: .\nRemove Luis.\n\nKeep this.", ["Luis"]);

    expect(result).toMatchObject({
      rewrittenText: "Keep this.",
      outcome: "clean",
      removedSpans: ["Luis"],
    });
  });

  it("handles smart quotes around a span inside an em-dash clause", () => {
    const result = deleteSpansSentenceAware("Keep. “Luis should go — now.” End.", [
      '"Luis should go — now."',
    ]);

    expect(result).toMatchObject({
      rewrittenText: "Keep. End.",
      outcome: "clean",
    });
  });

  it("removes only the paragraph sentence containing the failed span", () => {
    const result = deleteSpansSentenceAware(
      "First paragraph stays.\n\nSecond paragraph says Luis will call.\n\nThird paragraph stays.",
      ["Luis"],
    );

    expect(result).toMatchObject({
      rewrittenText: "First paragraph stays.\n\nThird paragraph stays.",
      outcome: "clean",
    });
  });

  it("flags malformed residue left after expanded deletion", () => {
    const result = deleteSpansSentenceAware("Quick note: .\n\nRemove Luis.", ["Luis"]);

    expect(result).toMatchObject({
      rewrittenText: "Quick note: .",
      outcome: "malformed",
    });
  });

  it("does not treat space-before-period setup residue as clean", () => {
    const result = deleteSpansSentenceAware("The point is . Remove Luis.", ["Luis"]);

    expect(result).toMatchObject({
      rewrittenText: "",
      outcome: "empty",
    });
  });

  it.each(["Quote: ''. Remove Luis.", "Quote: ‘’. Remove Luis."])(
    "does not treat empty quote-back residue as clean: %s",
    (text) => {
      const result = deleteSpansSentenceAware(text, ["Luis"]);

      expect(result).toMatchObject({
        rewrittenText: "",
        outcome: "empty",
      });
    },
  );

  it("does not treat name-initial chain residue as clean", () => {
    const result = deleteSpansSentenceAware("Keep this. J. R. Tolkien arrived. End.", ["Tolkien"]);

    expect(result).toMatchObject({
      rewrittenText: "",
      outcome: "empty",
    });
  });

  it("preserves indentation on non-deleted lines", () => {
    const result = deleteSpansSentenceAware(
      "Keep this.\n  - bullet 1.\n  - bullet 2.\nRemove Luis.",
      ["Luis"],
    );

    expect(result).toMatchObject({
      rewrittenText: "Keep this.\n  - bullet 1.\n  - bullet 2.",
      outcome: "clean",
    });
  });

  it("preserves indentation on the first surviving line when a leading sentence is deleted", () => {
    const result = deleteSpansSentenceAware(
      "Bad sentence about Luis.\n  - bullet 1\n  - bullet 2",
      ["Luis"],
    );

    expect(result).toMatchObject({
      rewrittenText: "  - bullet 1\n  - bullet 2",
      outcome: "clean",
    });
  });

  it("reports empty when expanded deletion leaves no substantive text", () => {
    const result = deleteSpansSentenceAware("Quick note: a sand-deranged dog.", [
      "a sand-deranged dog",
    ]);

    expect(result).toMatchObject({
      rewrittenText: "",
      outcome: "empty",
    });
  });
});

describe("deleteSpans", () => {
  it("removes spans at the start and trims trailing removal junk", () => {
    expect(deleteSpans("Remove me, keep this.", ["Remove me,"]).result).toBe("keep this.");
  });

  it("removes spans at the end", () => {
    expect(deleteSpans("Keep this. Remove me.", ["Remove me."]).result).toBe("Keep this.");
  });

  it("removes spans with surrounding punctuation when the span includes it", () => {
    expect(deleteSpans("Keep this, remove me; done.", ["remove me;"]).result).toBe(
      "Keep this,  done.",
    );
  });

  it("does not remove duplicate spans ambiguously", () => {
    const result = deleteSpans("Repeat claim. Repeat claim.", ["Repeat claim."]);

    expect(result.allRemoved).toBe(false);
    expect(result.result).toBe("Repeat claim. Repeat claim.");
  });

  it("does not remove overlapping spans", () => {
    const result = deleteSpans("Keep the bad span here.", ["bad span", "span"]);

    expect(result.allRemoved).toBe(false);
    expect(result.result).toBe("Keep the bad span here.");
  });
});
