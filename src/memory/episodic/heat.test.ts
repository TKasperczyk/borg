import { describe, expect, it } from "vitest";

import { computeEpisodeHeat } from "./heat.js";
import type { Episode, EpisodeStats } from "./types.js";

function createEpisode(overrides: Partial<Episode> = {}): Episode {
  return {
    id: "ep_aaaaaaaaaaaaaaaa" as Episode["id"],
    title: "Episode",
    narrative: "A short narrative.",
    participants: [],
    location: null,
    start_time: 0,
    end_time: 1,
    source_stream_ids: ["strm_aaaaaaaaaaaaaaaa" as Episode["source_stream_ids"][number]],
    significance: 0.6,
    tags: [],
    confidence: 0.8,
    lineage: {
      derived_from: [],
      supersedes: [],
    },
    embedding: Float32Array.from([1, 0, 0, 0]),
    created_at: 0,
    updated_at: 0,
    ...overrides,
  };
}

function createStats(overrides: Partial<EpisodeStats> = {}): EpisodeStats {
  return {
    episode_id: "ep_aaaaaaaaaaaaaaaa" as EpisodeStats["episode_id"],
    retrieval_count: 0,
    use_count: 0,
    last_retrieved: null,
    win_rate: 0,
    tier: "T1",
    promoted_at: 0,
    promoted_from: null,
    gist: null,
    gist_generated_at: null,
    last_decayed_at: null,
    ...overrides,
  };
}

describe("episodic heat", () => {
  it("rewards recent and successful episodes", () => {
    const nowMs = 10 * 24 * 3_600_000;
    const episode = createEpisode({
      updated_at: nowMs - 1_000,
    });
    const recent = computeEpisodeHeat(
      episode,
      createStats({
        retrieval_count: 3,
        win_rate: 0.8,
        last_retrieved: nowMs - 1_000,
      }),
      nowMs,
    );
    const stale = computeEpisodeHeat(
      episode,
      createStats({
        retrieval_count: 1,
        win_rate: 0.1,
        last_retrieved: nowMs - 20 * 24 * 3_600_000,
      }),
      nowMs,
    );

    expect(recent).toBeGreaterThan(stale);
  });
});
