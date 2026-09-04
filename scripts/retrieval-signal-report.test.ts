import { describe, expect, it } from "vitest";

import { buildReport } from "./retrieval-signal-report.js";

function intent(kind: string, candidates: { id: string; score: number; sim: number }[]): string {
  return JSON.stringify({
    event: "retrieval.intent_candidates",
    intent_kind: kind,
    candidates: candidates.map((c) => ({
      episode_id: c.id,
      score: c.score,
      vector_score: c.sim,
    })),
  });
}

describe("retrieval signal report", () => {
  it("counts every intent but only analyses lanes that carry similarity", () => {
    const lines = [
      intent("raw_text", [
        { id: "a", score: 0.9, sim: 0.7 },
        { id: "b", score: 0.5, sim: 0.4 },
        { id: "c", score: 0.4, sim: 0.3 },
      ]),
      // known_term/time/recent score similarity 0 by construction; including
      // them would drag every similarity statistic toward zero.
      intent("known_term", [
        { id: "d", score: 0.8, sim: 0 },
        { id: "e", score: 0.7, sim: 0 },
      ]),
    ];

    const r = buildReport(lines);

    expect(r.intentsTotal).toBe(2);
    expect(r.intentsSemantic).toBe(1);
    expect(r.similarityMedian).toBeCloseTo(0.4, 5);
  });

  it("calls a wide-separation corpus similarity-forward", () => {
    const lines = Array.from({ length: 5 }, (_, i) =>
      intent("semantic_query", [
        { id: `a${i}`, score: 0.9, sim: 0.9 },
        { id: `b${i}`, score: 0.5, sim: 0.6 },
        { id: `c${i}`, score: 0.4, sim: 0.3 },
      ]),
    );

    const r = buildReport(lines);

    expect(r.gapTop1ToTop3Median).toBeCloseTo(0.6, 5);
    expect(r.verdict).toContain("WIDE");
    expect(r.flatQueriesPct).toBe(0);
  });

  it("warns against a similarity-forward weight on a narrow corpus", () => {
    const lines = Array.from({ length: 5 }, (_, i) =>
      intent("semantic_query", [
        { id: `a${i}`, score: 0.9, sim: 0.601 },
        { id: `b${i}`, score: 0.5, sim: 0.6 },
        { id: `c${i}`, score: 0.4, sim: 0.599 },
      ]),
    );

    const r = buildReport(lines);

    expect(r.gapTop1ToTop3Median).toBeLessThan(0.02);
    expect(r.flatQueriesPct).toBe(100);
    expect(r.verdict).toContain("NARROW");
  });

  it("reports clamp saturation and top-1 concentration", () => {
    const lines = [
      intent("raw_text", [
        { id: "hit", score: 1, sim: 0.5 },
        { id: "x", score: 0.4, sim: 0.9 },
      ]),
      intent("raw_text", [
        { id: "hit", score: 1, sim: 0.5 },
        { id: "y", score: 0.3, sim: 0.8 },
      ]),
    ];

    const r = buildReport(lines);

    expect(r.clampSaturationPct).toBeCloseTo(50, 5);
    // the same episode took top-1 on both queries despite lower similarity
    expect(r.topOneConcentrationPct).toBeCloseTo(100, 5);
    expect(r.similarityDisplacementPct).toBeCloseTo(100, 5);
  });

  it("returns a non-crashing empty report when no semantic intents exist", () => {
    const r = buildReport([intent("recent", [{ id: "a", score: 0.5, sim: 0 }])]);

    expect(r.intentsSemantic).toBe(0);
    expect(r.verdict).toContain("nothing to judge");
  });
});
