import { describe, expect, it } from "vitest";

import { commonEpisodeIds, summarizeRanks, topKOverlap } from "./ranking.js";

describe("embedding A/B ranking metrics", () => {
  it("counts missing source ranks as misses in recall and MRR", () => {
    expect(summarizeRanks([1, 3, 10, 11, null])).toEqual({
      question_count: 5,
      ranked_source_count: 4,
      recall_at_1: 0.2,
      recall_at_3: 0.4,
      recall_at_10: 0.6,
      mrr: (1 + 1 / 3 + 1 / 10 + 1 / 11) / 5,
    });
  });

  it("returns null rates for an empty gold set", () => {
    expect(summarizeRanks([])).toEqual({
      question_count: 0,
      ranked_source_count: 0,
      recall_at_1: null,
      recall_at_3: null,
      recall_at_10: null,
      mrr: null,
    });
  });

  it("normalizes top-k overlap by the available top-k when the corpus is tiny", () => {
    const ranked = (ids: readonly string[]) =>
      ids.map((episode_id, index) => ({
        episode_id,
        rank: index + 1,
        title: episode_id,
        cosine_similarity: 1 - index / 10,
      }));

    expect(topKOverlap(ranked(["a", "b", "c"]), ranked(["b", "c", "d"]), 10)).toEqual({
      count: 2,
      denominator: 3,
      ratio: 2 / 3,
    });
  });

  it("builds one corpus from episodes embedded by every participating model", () => {
    const vector = new Float32Array([1, 0]);
    const left = new Map([
      ["source", vector],
      ["failed-distractor", vector],
      ["shared-distractor", vector],
    ]);
    const right = new Map([
      ["source", vector],
      ["shared-distractor", vector],
    ]);

    expect(
      commonEpisodeIds(["source", "failed-distractor", "shared-distractor"], [left, right]),
    ).toEqual(["source", "shared-distractor"]);
  });
});
