import { describe, expect, it } from "vitest";

import { applyMmr } from "./mmr.js";

describe("mmr", () => {
  it("diversifies near-duplicate candidates", () => {
    const candidates = [
      {
        item: "alpha",
        vector: Float32Array.from([1, 0]),
        relevanceScore: 0.95,
      },
      {
        item: "alpha-dup",
        vector: Float32Array.from([0.99, 0.01]),
        relevanceScore: 0.94,
      },
      {
        item: "beta",
        vector: Float32Array.from([0, 1]),
        relevanceScore: 0.7,
      },
    ];

    const selected = applyMmr(candidates, {
      limit: 2,
      lambda: 0.5,
    });

    expect(selected.map((candidate) => candidate.item)).toEqual(["alpha", "beta"]);
  });

  it("picks the most relevant candidate first even at lambda zero", () => {
    const candidates = [
      {
        item: "pool-first-low-relevance",
        vector: Float32Array.from([1, 0]),
        relevanceScore: 0.5,
      },
      {
        item: "high-relevance",
        vector: Float32Array.from([0, 1]),
        relevanceScore: 0.9,
      },
    ];

    // Lambda scales the relevance/diversity trade-off for later picks only;
    // the first pick has no redundancy term and must maximize relevance.
    // Before the fix, lambda 0 zeroed every first-pick score and degraded the
    // top selection to pool order.
    const selected = applyMmr(candidates, {
      limit: 2,
      lambda: 0,
    });

    expect(selected[0]?.item).toBe("high-relevance");
  });

  it("falls back to the default lambda when given a non-finite value", () => {
    const candidates = [
      {
        item: "alpha",
        vector: Float32Array.from([1, 0]),
        relevanceScore: 0.95,
      },
      {
        item: "alpha-dup",
        vector: Float32Array.from([0.99, 0.01]),
        relevanceScore: 0.94,
      },
      {
        item: "beta",
        vector: Float32Array.from([0, 1]),
        relevanceScore: 0.7,
      },
    ];

    const selected = applyMmr(candidates, {
      limit: 2,
      lambda: Number.NaN,
    });

    expect(selected.map((candidate) => candidate.item)).toEqual(["alpha", "beta"]);
  });

  it("returns an empty selection for zero limit", () => {
    expect(
      applyMmr(
        [
          {
            item: "alpha",
            vector: Float32Array.from([1, 0]),
            relevanceScore: 1,
          },
        ],
        {
          limit: 0,
        },
      ),
    ).toEqual([]);
  });

  it("uses canonical embedding similarity dimension checks", () => {
    expect(() =>
      applyMmr(
        [
          {
            item: "alpha",
            vector: Float32Array.from([1, 0]),
            relevanceScore: 1,
          },
          {
            item: "beta",
            vector: Float32Array.from([0]),
            relevanceScore: 0.5,
          },
        ],
        {
          limit: 2,
        },
      ),
    ).toThrow(/different dimensions/);
  });
});
