import { afterEach, describe, expect, it } from "vitest";

import { createEpisodeFixture, createOfflineTestHarness } from "../offline/test-support.js";

describe("RetrievalPipeline Sprint 7 scoring", () => {
  let harness: Awaited<ReturnType<typeof createOfflineTestHarness>> | undefined;

  afterEach(async () => {
    await harness?.cleanup();
    harness = undefined;
  });

  it("boosts mood-congruent episodes when mood is active", async () => {
    harness = await createOfflineTestHarness();
    const negativeEpisode = createEpisodeFixture({
      title: "Rust lifetime frustration",
      narrative: "A frustrating Rust debugging session stayed tense throughout.",
      tags: ["rust", "debugging"],
      emotional_arc: {
        start: { valence: -0.8, arousal: 0.7 },
        peak: { valence: -0.9, arousal: 0.8 },
        end: { valence: -0.4, arousal: 0.5 },
        dominant_emotion: "anger",
      },
    });
    const positiveEpisode = createEpisodeFixture({
      title: "Rust lifetime success",
      narrative: "The Rust issue resolved smoothly and felt satisfying.",
      tags: ["rust", "debugging"],
      emotional_arc: {
        start: { valence: 0.4, arousal: 0.2 },
        peak: { valence: 0.8, arousal: 0.4 },
        end: { valence: 0.7, arousal: 0.3 },
        dominant_emotion: "joy",
      },
    });
    await harness.episodicRepository.insert(negativeEpisode);
    await harness.episodicRepository.insert(positiveEpisode);

    const withoutMood = await harness.retrievalPipeline.search("Rust lifetime debugging", {
      limit: 2,
    });
    const withMood = await harness.retrievalPipeline.search("Rust lifetime debugging", {
      limit: 2,
      attentionWeights: {
        semantic: 0.7,
        goal_relevance: 0,
        mood: 0.2,
        time: 0,
        social: 0,
        heat: 0.1,
        suppression_penalty: 0.5,
      },
      moodState: {
        session_id: "default",
        valence: -0.7,
        arousal: 0.6,
        updated_at: 1_000_000,
        half_life_hours: 24,
        recent_triggers: [],
      },
    });

    expect(withMood[0]?.episode.id).toBe(negativeEpisode.id);
    expect(withMood[0]?.scoreBreakdown.moodBoost ?? 0).toBeGreaterThan(0);
    expect(withoutMood.map((item) => item.episode.id)).toContain(positiveEpisode.id);
  });

  it("boosts audience-relevant episodes when a trusted audience profile is present", async () => {
    harness = await createOfflineTestHarness();
    const audienceId = harness.entityRepository.resolve("Sam");
    harness.socialRepository.adjustTrust(audienceId, 0.3);
    const withAudience = createEpisodeFixture({
      title: "Sam architecture discussion",
      participants: ["Sam"],
      tags: ["architecture"],
    });
    const withoutAudience = createEpisodeFixture({
      title: "Background architecture note",
      participants: ["team"],
      tags: ["architecture"],
    });
    await harness.episodicRepository.insert(withAudience);
    await harness.episodicRepository.insert(withoutAudience);

    const results = await harness.retrievalPipeline.search("architecture", {
      limit: 2,
      attentionWeights: {
        semantic: 0.7,
        goal_relevance: 0,
        mood: 0,
        time: 0,
        social: 0.2,
        heat: 0.1,
        suppression_penalty: 0.5,
      },
      audienceProfile: harness.socialRepository.getProfile(audienceId),
      audienceTerms: ["Sam"],
    });

    expect(results[0]?.episode.id).toBe(withAudience.id);
    expect(results[0]?.scoreBreakdown.socialRelevance ?? 0).toBeGreaterThan(0);
  });
});
