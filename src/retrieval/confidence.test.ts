import { describe, expect, it } from "vitest";

import { ManualClock } from "../util/clock.js";
import { computeRetrievalConfidence } from "./confidence.js";
import type { RetrievedEpisode } from "./scoring.js";

const NOW_MS = 2_000;

function makeEpisode(overrides: {
  id: string;
  decayedSalience: number;
  participants?: string[];
}): RetrievedEpisode {
  return {
    episode: {
      id: overrides.id as RetrievedEpisode["episode"]["id"],
      title: `${overrides.id} title`,
      narrative: `${overrides.id} narrative`,
      participants: overrides.participants ?? ["team"],
      location: null,
      start_time: 0,
      end_time: 1,
      source_stream_ids: [
        "strm_aaaaaaaaaaaaaaaa" as RetrievedEpisode["episode"]["source_stream_ids"][number],
      ],
      significance: 0.8,
      tags: [],
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
    score: 0.5,
    scoreBreakdown: {
      similarity: 0.5,
      decayedSalience: overrides.decayedSalience,
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

describe("computeRetrievalConfidence", () => {
  it("returns zero overall when there are no episodes", () => {
    const confidence = computeRetrievalConfidence({
      episodes: [],
      contradictionPresent: false,
      nowMs: NOW_MS,
    });

    expect(confidence.overall).toBe(0);
    expect(confidence.evidenceStrength).toBe(0);
    expect(confidence.coverage).toBe(0);
    expect(confidence.sourceDiversity).toBe(0);
    expect(confidence.sampleSize).toBe(0);
    expect(confidence.contradictionPresent).toBe(false);
  });

  it("produces high confidence for strong, diverse, uncontested evidence", () => {
    const episodes = [
      makeEpisode({ id: "epi_aaaaaaaaaaaaaaaa", decayedSalience: 0.9, participants: ["alice"] }),
      makeEpisode({ id: "epi_bbbbbbbbbbbbbbbb", decayedSalience: 0.85, participants: ["bob"] }),
      makeEpisode({ id: "epi_cccccccccccccccc", decayedSalience: 0.8, participants: ["carol"] }),
      makeEpisode({ id: "epi_dddddddddddddddd", decayedSalience: 0.75, participants: ["dave"] }),
      makeEpisode({ id: "epi_eeeeeeeeeeeeeeee", decayedSalience: 0.7, participants: ["eve"] }),
    ];
    const confidence = computeRetrievalConfidence({
      episodes,
      contradictionPresent: false,
      nowMs: NOW_MS,
      expectedCount: 5,
    });

    expect(confidence.sampleSize).toBe(5);
    expect(confidence.evidenceStrength).toBeCloseTo(0.8, 2);
    expect(confidence.coverage).toBe(1);
    expect(confidence.sourceDiversity).toBe(1);
    // With full coverage and diversity, modulation = 1.0, so overall == evidenceStrength.
    expect(confidence.overall).toBeCloseTo(0.8, 2);
  });

  it("does not lift weak evidence above the S1 threshold via high coverage+diversity", () => {
    // Five episodes, all distinct participants, full coverage -- but each one is
    // barely established. The old formula (0.6*evidence + 0.25*coverage +
    // 0.15*diversity) produced 0.46 here, wrongly routing S1. The multiplicative
    // gate keeps overall pinned near evidenceStrength.
    const episodes = [
      makeEpisode({ id: "epi_aaaaaaaaaaaaaaaa", decayedSalience: 0.1, participants: ["a"] }),
      makeEpisode({ id: "epi_bbbbbbbbbbbbbbbb", decayedSalience: 0.1, participants: ["b"] }),
      makeEpisode({ id: "epi_cccccccccccccccc", decayedSalience: 0.1, participants: ["c"] }),
      makeEpisode({ id: "epi_dddddddddddddddd", decayedSalience: 0.1, participants: ["d"] }),
      makeEpisode({ id: "epi_eeeeeeeeeeeeeeee", decayedSalience: 0.1, participants: ["e"] }),
    ];
    const confidence = computeRetrievalConfidence({
      episodes,
      contradictionPresent: false,
      nowMs: NOW_MS,
      expectedCount: 5,
    });

    expect(confidence.evidenceStrength).toBeCloseTo(0.1, 2);
    expect(confidence.coverage).toBe(1);
    expect(confidence.sourceDiversity).toBe(1);
    expect(confidence.overall).toBeLessThan(0.45);
  });

  it("penalizes overall when contradictions are present", () => {
    const episodes = [
      makeEpisode({ id: "epi_aaaaaaaaaaaaaaaa", decayedSalience: 0.9, participants: ["alice"] }),
      makeEpisode({ id: "epi_bbbbbbbbbbbbbbbb", decayedSalience: 0.85, participants: ["bob"] }),
    ];
    const withoutContradiction = computeRetrievalConfidence({
      episodes,
      contradictionPresent: false,
      nowMs: NOW_MS,
      expectedCount: 5,
    });
    const withContradiction = computeRetrievalConfidence({
      episodes,
      contradictionPresent: true,
      nowMs: NOW_MS,
      expectedCount: 5,
    });

    expect(withContradiction.overall).toBeLessThan(withoutContradiction.overall);
    expect(withContradiction.contradictionPresent).toBe(true);
  });

  it("only penalizes contradiction edges valid at the query as-of", () => {
    const episodes = [
      makeEpisode({ id: "epi_aaaaaaaaaaaaaaaa", decayedSalience: 0.9, participants: ["alice"] }),
      makeEpisode({ id: "epi_bbbbbbbbbbbbbbbb", decayedSalience: 0.85, participants: ["bob"] }),
    ];
    const current = computeRetrievalConfidence({
      episodes,
      contradictionPresent: true,
      contradictionEdges: [
        {
          valid_from: 1_000,
          valid_to: 1_500,
        },
      ],
      nowMs: NOW_MS,
      asOf: 2_000,
      expectedCount: 5,
    });
    const historical = computeRetrievalConfidence({
      episodes,
      contradictionPresent: true,
      contradictionEdges: [
        {
          valid_from: 1_000,
          valid_to: 1_500,
        },
      ],
      nowMs: NOW_MS,
      asOf: 1_250,
      expectedCount: 5,
    });

    expect(current.contradictionPresent).toBe(false);
    expect(historical.contradictionPresent).toBe(true);
    expect(historical.overall).toBeLessThan(current.overall);
  });

  it("uses injected current time when checking contradiction edge validity", () => {
    const clock = new ManualClock(1_000);
    const episodes = [
      makeEpisode({ id: "epi_aaaaaaaaaaaaaaaa", decayedSalience: 0.9, participants: ["alice"] }),
    ];
    const confidence = computeRetrievalConfidence({
      episodes,
      contradictionPresent: true,
      contradictionEdges: [
        {
          valid_from: 500,
          valid_to: 1_500,
        },
      ],
      nowMs: clock.now(),
      expectedCount: 1,
    });

    expect(confidence.contradictionPresent).toBe(true);
  });

  it("drops coverage when fewer episodes than expected were found", () => {
    const episode = makeEpisode({
      id: "epi_aaaaaaaaaaaaaaaa",
      decayedSalience: 0.9,
      participants: ["alice"],
    });
    const confidence = computeRetrievalConfidence({
      episodes: [episode],
      contradictionPresent: false,
      nowMs: NOW_MS,
      expectedCount: 5,
    });

    expect(confidence.coverage).toBeCloseTo(0.2, 2);
    expect(confidence.evidenceStrength).toBeCloseTo(0.9, 2);
    // Low coverage pulls modulation down:
    // modulation = 0.7 + 0.2*0.2 + 0.1*1.0 = 0.84
    // overall = 0.9 * 0.84 = 0.756, still below full-coverage 0.9.
    expect(confidence.overall).toBeLessThan(0.8);
  });

  it("caps diversity when all episodes share the same participants", () => {
    const participants = ["alice", "bob"];
    const episodes = [
      makeEpisode({
        id: "epi_aaaaaaaaaaaaaaaa",
        decayedSalience: 0.9,
        participants,
      }),
      makeEpisode({
        id: "epi_bbbbbbbbbbbbbbbb",
        decayedSalience: 0.9,
        participants,
      }),
      makeEpisode({
        id: "epi_cccccccccccccccc",
        decayedSalience: 0.9,
        participants,
      }),
    ];
    const confidence = computeRetrievalConfidence({
      episodes,
      contradictionPresent: false,
      nowMs: NOW_MS,
      expectedCount: 3,
    });

    expect(confidence.sourceDiversity).toBeCloseTo(1 / 3, 5);
  });

  it("clamps evidence strength to [0, 1]", () => {
    const episodes = [
      makeEpisode({ id: "epi_aaaaaaaaaaaaaaaa", decayedSalience: 5 }),
      makeEpisode({ id: "epi_bbbbbbbbbbbbbbbb", decayedSalience: -1 }),
    ];
    const confidence = computeRetrievalConfidence({
      episodes,
      contradictionPresent: false,
      nowMs: NOW_MS,
    });

    expect(confidence.evidenceStrength).toBeGreaterThanOrEqual(0);
    expect(confidence.evidenceStrength).toBeLessThanOrEqual(1);
    expect(confidence.overall).toBeGreaterThanOrEqual(0);
    expect(confidence.overall).toBeLessThanOrEqual(1);
  });
});
