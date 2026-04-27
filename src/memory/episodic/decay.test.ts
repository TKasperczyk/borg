import { describe, expect, it } from "vitest";

import { applyEpisodeDecay } from "./decay.js";
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
    significance: 1,
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
    emotional_arc: overrides.emotional_arc ?? null,
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
    heat_multiplier: overrides.heat_multiplier ?? 1,
    valence_mean: overrides.valence_mean ?? 0,
    archived: overrides.archived ?? false,
  };
}

describe("episodic decay", () => {
  it("decays never-retrieved episodes faster than high-win-rate episodes", () => {
    const episode = createEpisode();
    const nowMs = 24 * 3 * 3_600_000;
    const cold = applyEpisodeDecay(episode, createStats(), {
      nowMs,
      baseHalfLifeHours: 24 * 7,
    });
    const reinforced = applyEpisodeDecay(
      episode,
      createStats({
        retrieval_count: 5,
        use_count: 4,
        win_rate: 0.9,
      }),
      {
        nowMs,
        baseHalfLifeHours: 24 * 7,
      },
    );

    expect(cold.decayedSalience).toBeLessThan(reinforced.decayedSalience);
    expect(cold.effectiveHalfLifeHours).toBeLessThan(reinforced.effectiveHalfLifeHours);
  });

  it("honors tier-specific half-life overrides", () => {
    const episode = createEpisode();
    const stats = createStats({
      tier: "T4",
    });
    const result = applyEpisodeDecay(episode, stats, {
      nowMs: 24 * 10 * 3_600_000,
      baseHalfLifeHours: 24,
      halfLifeByTier: {
        T4: 24 * 30,
      },
    });

    expect(result.effectiveHalfLifeHours).toBe(24 * 30 * 0.5);
    expect(result.decayedSalience).toBeGreaterThan(0.5);
  });
});
