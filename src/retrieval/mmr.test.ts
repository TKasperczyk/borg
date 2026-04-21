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
});
