import { describe, expect, it } from "vitest";

import type { RetrievalConfidence, RetrievedEpisode } from "../../retrieval/index.js";

import { chooseDeliberationPath } from "./path-selector.js";

function makeEpisode(score: number, tags: string[] = []): RetrievedEpisode {
  return {
    episode: {
      id: "epi_aaaaaaaaaaaaaaaa" as RetrievedEpisode["episode"]["id"],
      title: "title",
      narrative: "narrative",
      participants: ["team"],
      location: null,
      start_time: 0,
      end_time: 1,
      source_stream_ids: [
        "strm_aaaaaaaaaaaaaaaa" as RetrievedEpisode["episode"]["source_stream_ids"][number],
      ],
      significance: 0.8,
      tags,
      confidence: 0.8,
      lineage: {
        derived_from: [],
        supersedes: [],
      },
      emotional_arc: null,
      embedding: Float32Array.from([1, 0, 0, 0]),
      created_at: 0,
      updated_at: 0,
    },
    score,
    scoreBreakdown: {
      similarity: score,
      decayedSalience: 0.3,
      heat: 0,
      goalRelevance: 0,
      valueAlignment: 0,
      timeRelevance: 0,
      moodBoost: 0,
      socialRelevance: 0,
      entityRelevance: 0,
      suppressionPenalty: 0,
    },
    citationChain: [],
  };
}

function makeConfidence(overall: number, contradictionPresent = false): RetrievalConfidence {
  return {
    overall,
    evidenceStrength: overall,
    coverage: 1,
    sourceDiversity: 1,
    contradictionPresent,
    sampleSize: 5,
  };
}

describe("chooseDeliberationPath", () => {
  it("uses RetrievalConfidence.overall when provided, not the relevance-score average", () => {
    // Relevance score average is high, but epistemic confidence is low.
    // Should route to S2 because the epistemic signal is what matters.
    const highRelevance = [makeEpisode(0.9), makeEpisode(0.9)];

    const decision = chooseDeliberationPath(
      "problem_solving",
      "low",
      highRelevance,
      false,
      makeConfidence(0.2),
    );

    expect(decision.path).toBe("system_2");
    expect(decision.reason).toMatch(/low retrieval confidence/i);
  });

  it("routes to S1 when epistemic confidence is high, regardless of score", () => {
    // Low relevance-score average but high epistemic confidence.
    const lowRelevance = [makeEpisode(0.1), makeEpisode(0.2)];

    const decision = chooseDeliberationPath(
      "problem_solving",
      "low",
      lowRelevance,
      false,
      makeConfidence(0.9),
    );

    expect(decision.path).toBe("system_1");
  });

  it("routes from explicit low retrieval confidence without averaging scores", () => {
    const decision = chooseDeliberationPath(
      "problem_solving",
      "low",
      [makeEpisode(0.9)],
      false,
      makeConfidence(0.2),
    );

    expect(decision.path).toBe("system_2");
    expect(decision.reason).toMatch(/low retrieval confidence/i);
  });

  it("routes to S2 when reflective mode is active regardless of confidence", () => {
    const decision = chooseDeliberationPath(
      "reflective",
      "low",
      [makeEpisode(0.9)],
      false,
      makeConfidence(0.95),
    );

    expect(decision.path).toBe("system_2");
  });

  it("routes to S1 in idle mode regardless of confidence", () => {
    const decision = chooseDeliberationPath(
      "idle",
      "low",
      [makeEpisode(0.1)],
      false,
      makeConfidence(0.1),
    );

    expect(decision.path).toBe("system_1");
  });

  it("routes to S2 when contradiction flag is set even if confidence is high", () => {
    const decision = chooseDeliberationPath(
      "problem_solving",
      "low",
      [makeEpisode(0.9)],
      true,
      makeConfidence(0.9, true),
    );

    expect(decision.path).toBe("system_2");
    expect(decision.reason).toMatch(/contradiction/i);
  });

  it("routes to S2 when only confidence.contradictionPresent is set (boolean flag omitted)", () => {
    const decision = chooseDeliberationPath(
      "problem_solving",
      "low",
      [makeEpisode(0.9)],
      false,
      makeConfidence(0.9, true),
    );

    expect(decision.path).toBe("system_2");
    expect(decision.reason).toMatch(/contradiction/i);
  });

  it("ignores warning/recommended tags as contradiction cues", () => {
    const decision = chooseDeliberationPath(
      "problem_solving",
      "low",
      [makeEpisode(0.9, ["warning"]), makeEpisode(0.9, ["recommended"])],
      false,
      makeConfidence(0.9, false),
    );

    expect(decision.path).toBe("system_1");
  });

  it("routes to S2 for high stakes even with confident retrieval", () => {
    const decision = chooseDeliberationPath(
      "problem_solving",
      "high",
      [makeEpisode(0.9)],
      false,
      makeConfidence(0.95),
    );

    expect(decision.path).toBe("system_2");
    expect(decision.reason).toMatch(/high-stakes/i);
  });

  it("escalates idle mode to S2 when stakes are high", () => {
    // Sprint 53: idle was a hard early return that bypassed the high-stakes
    // and contradiction checks below it. A misclassified high-stakes idle
    // turn must still take the deeper path.
    const decision = chooseDeliberationPath(
      "idle",
      "high",
      [makeEpisode(0.9)],
      false,
      makeConfidence(0.95),
    );

    expect(decision.path).toBe("system_2");
    expect(decision.reason).toMatch(/high-stakes/i);
  });

  it("escalates idle mode to S2 when retrieved context contradicts", () => {
    const decision = chooseDeliberationPath(
      "idle",
      "low",
      [makeEpisode(0.9)],
      true,
      makeConfidence(0.9, true),
    );

    expect(decision.path).toBe("system_2");
    expect(decision.reason).toMatch(/contradiction/i);
  });
});
