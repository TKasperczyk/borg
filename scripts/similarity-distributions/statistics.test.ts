import { describe, expect, it } from "vitest";
import { cosineSimilarity } from "../../src/retrieval/embedding-similarity.js";
import { THRESHOLDS } from "./inventory.js";
import { matchProposal } from "./report.js";
import {
  distribution,
  measureRows,
  percentileRank,
  quantile,
  samplePairOrdinals,
} from "./statistics.js";
import type { VectorRow } from "./bank.js";

function row(id: string, x: number, y: number, dimensions = 1024): VectorRow {
  const vector = new Float32Array(dimensions);
  vector[0] = x;
  vector[1] = y;
  return { id, title: `Title ${id}`, vector };
}

describe("percentile math", () => {
  it("interpolates percentiles, including negative cosines and exact endpoints", () => {
    const d = distribution([-1, 1, 0, 0.5]);
    expect(d.sorted_values).toEqual([-1, 0, 0.5, 1]);
    expect(d.percentiles).toEqual({
      min: -1,
      p50: 0.25,
      p90: expect.closeTo(0.85),
      p95: expect.closeTo(0.925),
      p99: expect.closeTo(0.985),
      max: 1,
    });
    expect(quantile([], 0.5)).toBeNull();
    expect(quantile([0.4], 0.99)).toBe(0.4);
    expect(() => quantile([1], NaN)).toThrow();
    expect(() => quantile([1], -0.1)).toThrow();
  });

  it("inverts the quantile curve, with midpoint ties and explicit endpoint clipping", () => {
    expect(percentileRank([0.1, 0.4, 0.4, 0.9], 0.4)).toBe(0.5);
    expect(percentileRank([0.1, 0.4, 0.7, 1], 0.55)).toBeCloseTo(0.5);
    expect(percentileRank([0.2, 0.5], -0.1)).toBe(0);
    expect(percentileRank([0.2, 0.5], 0.9)).toBe(1);
    expect(percentileRank([0.5], 0.5)).toBe(0.5);
    expect(percentileRank([], 0.5)).toBeNull();
  });

  it("maps the full measured distribution and converts cosine distance back", () => {
    const threshold = THRESHOLDS.find((item) => item.id === "action_persistence_duplicate")!;
    const proposal = matchProposal(threshold, [0.7, 0.8, 0.9, 1], [0.3, 0.4, 0.6, 0.8]);
    expect(proposal.qwen_percentile).toBeCloseTo(50);
    expect(proposal.proposed_value).toBeCloseTo(0.5);
    expect(proposal.support).toBe("within_observed_range");
    const diameter = THRESHOLDS.find((item) => item.id === "consolidation_diameter")!;
    const distance = matchProposal(diameter, [0.7, 0.8, 0.84, 0.9], [0.3, 0.4, 0.5, 0.6]);
    expect(distance.qwen_cosine).toBeCloseTo(0.82);
    expect(distance.proposed_value).toBeCloseTo(0.55);
    expect(matchProposal(threshold, [0.2, 0.5], [0.1, 0.3])).toMatchObject({
      support: "above_observed_range",
      proposed_value: 0.3,
    });
    expect(matchProposal(threshold, [0.9, 1], [0.4, 0.6])).toMatchObject({
      support: "below_observed_range",
      proposed_value: 0.4,
    });
    expect(matchProposal(threshold, [], [0.4, 0.6]).proposed_value).toBeNull();
    expect(matchProposal(threshold, [0.7, 0.9], []).proposed_value).toBeNull();
  });
});

describe("stored-vector measurements", () => {
  it("samples exactly n unique unordered pair ordinals with a seed", () => {
    const sampled = samplePairOrdinals(100, 20, "repeatable");
    expect(sampled).toHaveLength(20);
    expect(new Set(sampled).size).toBe(20);
    expect(sampled.every((value) => value >= 0 && value < 100)).toBe(true);
    expect(samplePairOrdinals(100, 20, "repeatable")).toEqual(sampled);
    expect(samplePairOrdinals(100, 20, "different")).not.toEqual(sampled);
    expect(samplePairOrdinals(3, 10, "seed")).toEqual([0, 1, 2]);
    expect(samplePairOrdinals(0, 10, "seed")).toEqual([]);
  });

  it.each([1024, 4096])(
    "computes exact nearest neighbors and sweep counts at %i dimensions",
    async (dimensions) => {
      const rows = [
        row("a", 2, 0, dimensions),
        row("b", 0.9, Math.sqrt(0.19), dimensions),
        row("c", -1, 0, dimensions),
        row("d", 0, -1, dimensions),
      ];
      const measured = await measureRows({ rows, sample: 100, seed: "fixture" });
      expect(measured.pair_count).toBe(6);
      expect(measured.nearest_neighbor_rows).toEqual([
        { id: "a", neighbor_id: "b", cosine: expect.closeTo(0.9) },
        { id: "b", neighbor_id: "a", cosine: expect.closeTo(0.9) },
        { id: "c", neighbor_id: "d", cosine: 0 },
        { id: "d", neighbor_id: "a", cosine: 0 },
      ]);
      const reference = rows
        .flatMap((left, i) =>
          rows.slice(i + 1).map((right) => cosineSimilarity(left.vector, right.vector)),
        )
        .sort((a, b) => a - b);
      expect(measured.random_pair.sorted_values).toEqual(
        reference.map((value) => expect.closeTo(value, 6)),
      );
      expect(measured.near_duplicates[0]).toMatchObject({
        cutoff: 0.8,
        count: 1,
        examples: [{ left_id: "a", right_id: "b", left_title: "Title a", right_title: "Title b" }],
      });
      expect(measured.near_duplicates.at(-1)?.count).toBe(0);
    },
  );

  it("counts all qualifying pairs once, including identical vectors with distinct IDs", async () => {
    const measured = await measureRows({
      rows: [row("a", 1, 0), row("b", 1, 0), row("c", 1, 0)],
      sample: 1,
      seed: "same",
    });
    expect(measured.near_duplicates.every((cutoff) => cutoff.count === 3)).toBe(true);
    expect(measured.random_pair).toMatchObject({ count: 1, population: 3, sampled: true });
    expect(
      measured.nearest_neighbor_rows.every((r) => r.cosine === 1 && r.id !== r.neighbor_id),
    ).toBe(true);
  });

  it("separates family strata and excludes unlabeled rows from both", async () => {
    const measured = await measureRows({
      rows: [row("a", 1, 0), row("b", 1, 0), row("c", 0, 1), row("d", 1, 0)],
      sample: 10,
      seed: "family",
      families: {
        source: "fixture",
        reason: null,
        byId: new Map([
          ["a", "one"],
          ["b", "one"],
          ["c", "two"],
        ]),
      },
    });
    expect(measured.families).toMatchObject({
      labeled_row_count: 3,
      groups: [
        { family_id: "one", member_ids: ["a", "b"] },
        { family_id: "two", member_ids: ["c"] },
      ],
      within_family: { population: 1, sorted_values: [1] },
      across_family: { population: 2, sorted_values: [0, 0] },
    });
  });

  it("handles empty, singleton, and entirely negative nearest-neighbor populations", async () => {
    const empty = await measureRows({ rows: [], sample: 2, seed: "edge" });
    expect(empty.nearest_neighbor.percentiles.p50).toBeNull();
    const singleton = await measureRows({ rows: [row("a", 1, 0)], sample: 2, seed: "edge" });
    expect(singleton.nearest_neighbor_rows).toEqual([{ id: "a", neighbor_id: null, cosine: null }]);
    const opposite = await measureRows({
      rows: [row("a", 1, 0), row("b", -1, 0)],
      sample: 2,
      seed: "edge",
    });
    expect(opposite.nearest_neighbor.sorted_values).toEqual([-1, -1]);
  });
});
