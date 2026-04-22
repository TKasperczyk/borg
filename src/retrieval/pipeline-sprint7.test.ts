import { afterEach, describe, expect, it } from "vitest";

import {
  createEpisodeFixture,
  createOfflineTestHarness,
  createSemanticNodeFixture,
} from "../offline/test-support.js";

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
        entity: 0,
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
    harness.socialRepository.adjustTrust(audienceId, 0.3, { kind: "manual" });
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
        entity: 0,
        heat: 0.1,
        suppression_penalty: 0.5,
      },
      audienceProfile: harness.socialRepository.getProfile(audienceId),
      audienceTerms: ["Sam"],
    });

    expect(results[0]?.episode.id).toBe(withAudience.id);
    expect(results[0]?.scoreBreakdown.socialRelevance ?? 0).toBeGreaterThan(0);
  });

  it("hard-excludes audience-scoped episodes from other audiences", async () => {
    harness = await createOfflineTestHarness();
    const sam = harness.entityRepository.resolve("Sam");
    const alex = harness.entityRepository.resolve("Alex");
    const privateEpisode = createEpisodeFixture({
      title: "Sam-only architecture review",
      tags: ["architecture"],
      audience_entity_id: sam,
      shared: false,
    });
    await harness.episodicRepository.insert(privateEpisode);

    const results = await harness.retrievalPipeline.search("architecture", {
      limit: 3,
      audienceEntityId: alex,
    });

    expect(results).toEqual([]);
  });

  it("keeps public episodes visible for any audience", async () => {
    harness = await createOfflineTestHarness();
    const alex = harness.entityRepository.resolve("Alex");
    const publicEpisode = createEpisodeFixture({
      title: "Public architecture note",
      tags: ["architecture"],
      audience_entity_id: null,
      shared: true,
    });
    await harness.episodicRepository.insert(publicEpisode);

    const results = await harness.retrievalPipeline.search("architecture", {
      limit: 3,
      audienceEntityId: alex,
    });

    expect(results.map((result) => result.episode.id)).toContain(publicEpisode.id);
  });

  it("filters semantic nodes whose source episodes include hidden evidence", async () => {
    harness = await createOfflineTestHarness();
    const sam = harness.entityRepository.resolve("Sam");
    const alex = harness.entityRepository.resolve("Alex");
    const publicEpisode = createEpisodeFixture({
      title: "Atlas public note",
      tags: ["atlas"],
      audience_entity_id: null,
      shared: true,
    });
    const hiddenEpisode = createEpisodeFixture({
      title: "Atlas Sam-only note",
      tags: ["atlas"],
      audience_entity_id: sam,
      shared: false,
    });
    await harness.episodicRepository.insert(publicEpisode);
    await harness.episodicRepository.insert(hiddenEpisode);
    const mixedNode = createSemanticNodeFixture({
      kind: "entity",
      label: "Atlas Audience Scoped",
      description: "Atlas node backed by both public and hidden evidence.",
      source_episode_ids: [publicEpisode.id, hiddenEpisode.id],
    });
    await harness.semanticNodeRepository.insert(mixedNode);

    const result = await harness.retrievalPipeline.searchWithContext("Atlas Audience Scoped", {
      limit: 2,
      audienceEntityId: alex,
      graphWalkDepth: 1,
      maxGraphNodes: 4,
    });

    expect(result.semantic.matched_node_ids).not.toContain(mixedNode.id);
    expect(result.semantic.matched_nodes).toEqual([]);
    expect(result.semantic.supports).toEqual([]);
    expect(result.semantic.contradicts).toEqual([]);
    expect(result.semantic.categories).toEqual([]);
    expect(result.semantic.support_hits).toEqual([]);
    expect(result.semantic.contradiction_hits).toEqual([]);
    expect(result.semantic.category_hits).toEqual([]);
  });
});
