import { afterEach, describe, expect, it, vi } from "vitest";

import {
  createEpisodeFixture,
  createOfflineTestHarness,
  createSemanticNodeFixture,
  TestEmbeddingClient,
} from "../offline/test-support.js";
import { DEFAULT_SESSION_ID, type EntityId } from "../util/ids.js";
import { SELF_RECALL_SCOPE } from "./recall-context.js";

function socialAttentionWeights() {
  return {
    semantic: 0.7,
    goal_relevance: 0,
    value_alignment: 0,
    mood: 0,
    time: 0,
    social: 0.2,
    entity: 0,
    heat: 0.1,
    suppression_penalty: 0.5,
  };
}

function cognitionRecallOptions(currentAudienceEntityId: EntityId | null = null) {
  return {
    recallContext: {
      reader: SELF_RECALL_SCOPE,
      currentSessionId: DEFAULT_SESSION_ID,
      currentAudienceEntityId,
      currentParticipantEntityIds:
        currentAudienceEntityId === null ? [] : [currentAudienceEntityId],
    },
    rankingAudienceEntityId: currentAudienceEntityId,
    sessionId: DEFAULT_SESSION_ID,
  };
}

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
    await harness.episodicRepository.createEpisode(negativeEpisode);
    await harness.episodicRepository.createEpisode(positiveEpisode);

    const withoutMood = await harness.retrievalPipeline.searchEpisodesForDisclosure(
      "Rust lifetime debugging",
      {
        limit: 2,
      },
    );
    const withMood = await harness.retrievalPipeline.searchEpisodesForDisclosure(
      "Rust lifetime debugging",
      {
        limit: 2,
        attentionWeights: {
          semantic: 0.7,
          goal_relevance: 0,
          value_alignment: 0,
          mood: 0.2,
          time: 0,
          social: 0,
          entity: 0,
          heat: 0.1,
          suppression_penalty: 0.5,
        },
        moodState: {
          valence: -0.7,
          arousal: 0.6,
        },
      },
    );

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
    await harness.episodicRepository.createEpisode(withAudience);
    await harness.episodicRepository.createEpisode(withoutAudience);

    const results = await harness.retrievalPipeline.searchEpisodesForDisclosure("architecture", {
      limit: 2,
      attentionWeights: socialAttentionWeights(),
      audienceProfile: harness.socialRepository.getProfile(audienceId),
      audienceTerms: ["Sam"],
    });

    expect(results[0]?.episode.id).toBe(withAudience.id);
    expect(results[0]?.scoreBreakdown.socialRelevance ?? 0).toBeGreaterThan(0);
  });

  it("matches audience aliases through participant entity resolution", async () => {
    harness = await createOfflineTestHarness();
    const audienceId = harness.entityRepository.resolve("Tomasz");
    harness.entityRepository.addAlias(audienceId, "Tom");
    const withAudience = createEpisodeFixture(
      {
        title: "Tom architecture discussion",
        participants: ["Tom"],
        tags: ["architecture"],
      },
      [1, 0, 0, 0],
    );
    await harness.episodicRepository.createEpisode(withAudience);

    const results = await harness.retrievalPipeline.searchEpisodesForDisclosure("architecture", {
      limit: 1,
      attentionWeights: socialAttentionWeights(),
      audienceEntityId: audienceId,
      audienceTerms: ["Tomasz"],
    });

    expect(results[0]?.episode.id).toBe(withAudience.id);
    expect(results[0]?.scoreBreakdown.socialRelevance).toBe(0.2);
  });

  it("keeps string fallback for unresolved participant entities", async () => {
    harness = await createOfflineTestHarness();
    const audienceId = harness.entityRepository.resolve("Tomasz");
    const withFreeFormAudience = createEpisodeFixture(
      {
        title: "Visitor architecture discussion",
        participants: ["Visitor"],
        tags: ["architecture"],
      },
      [1, 0, 0, 0],
    );
    await harness.episodicRepository.createEpisode(withFreeFormAudience);

    const results = await harness.retrievalPipeline.searchEpisodesForDisclosure("architecture", {
      limit: 1,
      attentionWeights: socialAttentionWeights(),
      audienceEntityId: audienceId,
      audienceTerms: ["Visitor"],
    });

    expect(results[0]?.episode.id).toBe(withFreeFormAudience.id);
    expect(results[0]?.scoreBreakdown.socialRelevance).toBe(0.2);
  });

  it("does not match unknown participants without a string fallback hit", async () => {
    harness = await createOfflineTestHarness();
    const audienceId = harness.entityRepository.resolve("Tomasz");
    const unrelated = createEpisodeFixture(
      {
        title: "Visitor architecture discussion",
        participants: ["Visitor"],
        tags: ["architecture"],
      },
      [1, 0, 0, 0],
    );
    await harness.episodicRepository.createEpisode(unrelated);

    const results = await harness.retrievalPipeline.searchEpisodesForDisclosure("architecture", {
      limit: 1,
      attentionWeights: socialAttentionWeights(),
      audienceEntityId: audienceId,
      audienceTerms: ["Tomasz"],
    });

    expect(results[0]?.episode.id).toBe(unrelated.id);
    expect(results[0]?.scoreBreakdown.socialRelevance).toBe(0);
  });

  it("does not fall back to string matching for a participant resolved to another entity", async () => {
    harness = await createOfflineTestHarness();
    const project = harness.entityRepository.resolve("Alice Codename");
    harness.entityRepository.addAlias(project, "Alice");
    const audienceId = harness.entityRepository.resolve("Alice Person");
    const withOtherEntity = createEpisodeFixture(
      {
        title: "Alice codename architecture discussion",
        participants: ["Alice"],
        tags: ["architecture"],
      },
      [1, 0, 0, 0],
    );
    await harness.episodicRepository.createEpisode(withOtherEntity);

    const results = await harness.retrievalPipeline.searchEpisodesForDisclosure("architecture", {
      limit: 1,
      attentionWeights: socialAttentionWeights(),
      audienceEntityId: audienceId,
      audienceTerms: ["Alice"],
    });

    expect(results[0]?.episode.id).toBe(withOtherEntity.id);
    expect(results[0]?.scoreBreakdown.socialRelevance).toBe(0);
  });

  it("keeps trust gating after entity-based social matches", async () => {
    harness = await createOfflineTestHarness();
    const audienceId = harness.entityRepository.resolve("Tomasz");
    harness.entityRepository.addAlias(audienceId, "Tom");
    const episode = createEpisodeFixture(
      {
        title: "Tom trusted architecture discussion",
        participants: ["Tom"],
        tags: ["architecture"],
      },
      [1, 0, 0, 0],
    );
    await harness.episodicRepository.createEpisode(episode);

    const lowTrust = await harness.retrievalPipeline.searchEpisodesForDisclosure("architecture", {
      limit: 1,
      attentionWeights: socialAttentionWeights(),
      audienceEntityId: audienceId,
      audienceProfile: harness.socialRepository.upsertProfile(audienceId),
      audienceTerms: ["Tomasz"],
    });
    const highTrustProfile = harness.socialRepository.adjustTrust(audienceId, 0.3, {
      kind: "manual",
    });
    const highTrust = await harness.retrievalPipeline.searchEpisodesForDisclosure("architecture", {
      limit: 1,
      attentionWeights: socialAttentionWeights(),
      audienceEntityId: audienceId,
      audienceProfile: highTrustProfile,
      audienceTerms: ["Tomasz"],
    });

    expect(lowTrust[0]?.scoreBreakdown.socialRelevance).toBe(0.2);
    expect(highTrust[0]?.scoreBreakdown.socialRelevance).toBe(0.25);
  });

  it("caches participant entity resolution per retrieval call", async () => {
    harness = await createOfflineTestHarness();
    const audienceId = harness.entityRepository.resolve("Tomasz");
    harness.entityRepository.addAlias(audienceId, "Tom");
    const first = createEpisodeFixture(
      {
        title: "First Tom architecture discussion",
        participants: ["Tom"],
        tags: ["architecture"],
      },
      [1, 0, 0, 0],
    );
    const second = createEpisodeFixture(
      {
        title: "Second Tom architecture discussion",
        participants: ["Tom"],
        tags: ["architecture"],
      },
      [1, 0, 0, 0],
    );
    await harness.episodicRepository.createEpisode(first);
    await harness.episodicRepository.createEpisode(second);
    const findSpy = vi.spyOn(harness.entityRepository, "findByName");

    await harness.retrievalPipeline.searchEpisodesForDisclosure("architecture", {
      limit: 2,
      attentionWeights: socialAttentionWeights(),
      audienceEntityId: audienceId,
      audienceTerms: ["Tomasz"],
    });

    expect(findSpy.mock.calls.filter(([name]) => name === "Tom")).toHaveLength(1);

    findSpy.mockClear();

    await harness.retrievalPipeline.searchEpisodesForDisclosure("architecture", {
      limit: 2,
      attentionWeights: socialAttentionWeights(),
      audienceTerms: ["Tom"],
    });

    expect(findSpy).not.toHaveBeenCalled();
  });

  it("recalls audience-scoped episodes across audiences and labels them", async () => {
    harness = await createOfflineTestHarness();
    const sam = harness.entityRepository.resolve("Sam");
    const alex = harness.entityRepository.resolve("Alex");
    const privateEpisode = createEpisodeFixture({
      title: "Sam-only architecture review",
      tags: ["architecture"],
      audience_entity_id: sam,
      shared: false,
    });
    await harness.episodicRepository.createEpisode(privateEpisode);

    const results = await harness.retrievalPipeline.recallEpisodesForCognition("architecture", {
      ...cognitionRecallOptions(alex),
      limit: 3,
    });

    const recalled = results.episodes.find((result) => result.episode.id === privateEpisode.id);
    expect(recalled).toBeDefined();
    expect(recalled?.disclosureLabel).toMatchObject({
      disclosureClass: "relationship_private",
      privateToEntityIds: [sam],
    });
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
    await harness.episodicRepository.createEpisode(publicEpisode);

    const results = await harness.retrievalPipeline.searchEpisodesForDisclosure("architecture", {
      limit: 3,
      audienceEntityId: alex,
    });

    expect(results.map((result) => result.episode.id)).toContain(publicEpisode.id);
  });

  it("recalls self-scoped and other private episodes globally while preserving labels", async () => {
    harness = await createOfflineTestHarness();
    const selfEntityId = harness.entityRepository.add({
      canonicalName: "Sol",
      kind: "self",
    }).id;
    const otherEntityId = harness.entityRepository.resolve("Other");
    const audienceEntityId = harness.entityRepository.resolve("Audience");
    const selfScopedEpisode = createEpisodeFixture({
      title: "Self-scoped architecture memory",
      tags: ["architecture"],
      audience_entity_id: selfEntityId,
      shared: false,
    });
    const otherPrivateEpisode = createEpisodeFixture({
      title: "Other-private architecture memory",
      tags: ["architecture"],
      audience_entity_id: otherEntityId,
      shared: false,
    });
    await harness.episodicRepository.createEpisode(selfScopedEpisode);
    await harness.episodicRepository.createEpisode(otherPrivateEpisode);

    const cognition = await harness.retrievalPipeline.recallEpisodesForCognition("architecture", {
      ...cognitionRecallOptions(audienceEntityId),
      limit: 5,
      entityTerms: ["architecture"],
    });
    const disclosure = await harness.retrievalPipeline.searchEpisodesForDisclosure("architecture", {
      limit: 5,
      audienceEntityId: audienceEntityId,
      entityTerms: ["architecture"],
    });

    const cognitionById = new Map(cognition.episodes.map((result) => [result.episode.id, result]));
    expect(cognitionById.get(selfScopedEpisode.id)?.disclosureLabel).toMatchObject({
      disclosureClass: "relationship_private",
      privateToEntityIds: [selfEntityId],
    });
    expect(cognitionById.get(otherPrivateEpisode.id)?.disclosureLabel).toMatchObject({
      disclosureClass: "relationship_private",
      privateToEntityIds: [otherEntityId],
    });
    expect(disclosure.map((result) => result.episode.id)).not.toContain(selfScopedEpisode.id);
    expect(disclosure.map((result) => result.episode.id)).not.toContain(otherPrivateEpisode.id);
  });

  it("recalls private semantic sources globally during cognition and labels them", async () => {
    const query = "atlas semantic source visibility";
    harness = await createOfflineTestHarness({
      embeddingClient: new TestEmbeddingClient(new Map([[query, [1, 0, 0, 0]]])),
    });
    const currentAudience = harness.entityRepository.resolve("Current Audience");
    const otherAudience = harness.entityRepository.resolve("Other Audience");
    const currentPrivateEpisode = createEpisodeFixture({
      title: "Atlas current-audience semantic source",
      tags: ["atlas"],
      audience_entity_id: currentAudience,
      shared: false,
    });
    const otherPrivateEpisode = createEpisodeFixture({
      title: "Atlas other-audience semantic source",
      tags: ["atlas"],
      audience_entity_id: otherAudience,
      shared: false,
    });
    await harness.episodicRepository.createEpisode(currentPrivateEpisode);
    await harness.episodicRepository.createEpisode(otherPrivateEpisode);
    const currentNode = createSemanticNodeFixture(
      {
        kind: "entity",
        label: "Atlas Current Audience Private",
        description: "Atlas node backed by current-audience private evidence.",
        source_episode_ids: [currentPrivateEpisode.id],
      },
      [1, 0, 0, 0],
    );
    const otherNode = createSemanticNodeFixture(
      {
        kind: "entity",
        label: "Atlas Other Audience Private",
        description: "Atlas node backed by other-audience private evidence.",
        source_episode_ids: [otherPrivateEpisode.id],
      },
      [1, 0, 0, 0],
    );
    await harness.semanticNodeRepository.insert(currentNode);
    await harness.semanticNodeRepository.insert(otherNode);

    const result = await harness.retrievalPipeline.recallEpisodesForCognition(query, {
      ...cognitionRecallOptions(currentAudience),
      limit: 5,
      graphWalkDepth: 1,
      maxGraphNodes: 4,
    });
    const matchedNodesById = new Map(result.semantic.matched_nodes.map((node) => [node.id, node]));

    expect(matchedNodesById.get(currentNode.id)?.disclosureLabel).toEqual({
      disclosureClass: "relationship_private",
      originAudienceEntityIds: [currentAudience],
      privateToEntityIds: [currentAudience],
      publicToEntityIds: [],
    });
    expect(matchedNodesById.get(otherNode.id)?.disclosureLabel).toEqual({
      disclosureClass: "relationship_private",
      originAudienceEntityIds: [otherAudience],
      privateToEntityIds: [otherAudience],
      publicToEntityIds: [],
    });
  });

  it("labels semantic-edge evidence with the private target node source", async () => {
    const query = "atlas public root private support";
    harness = await createOfflineTestHarness({
      embeddingClient: new TestEmbeddingClient(new Map([[query, [1, 0, 0, 0]]])),
    });
    const alice = harness.entityRepository.resolve("Alice");
    const bob = harness.entityRepository.resolve("Bob");
    const publicEpisode = createEpisodeFixture({
      title: "Atlas public edge evidence",
      tags: ["atlas"],
      audience_entity_id: null,
      shared: true,
    });
    const privateEpisode = createEpisodeFixture({
      title: "Atlas Alice private semantic target",
      tags: ["atlas"],
      audience_entity_id: alice,
      shared: false,
    });
    await harness.episodicRepository.createEpisode(publicEpisode);
    await harness.episodicRepository.createEpisode(privateEpisode);
    const root = await harness.semanticNodeRepository.insert(
      createSemanticNodeFixture(
        {
          kind: "entity",
          label: "Atlas Public Root",
          description: "Atlas root backed by public evidence.",
          source_episode_ids: [publicEpisode.id],
        },
        [1, 0, 0, 0],
      ),
    );
    const privateSupport = await harness.semanticNodeRepository.insert(
      createSemanticNodeFixture(
        {
          kind: "proposition",
          label: "Alice Private Support",
          description: "Alice-private support must not be under-labeled by a public edge.",
          source_episode_ids: [privateEpisode.id],
        },
        [0, 1, 0, 0],
      ),
    );
    const edge = harness.semanticEdgeRepository.addEdge({
      from_node_id: root.id,
      to_node_id: privateSupport.id,
      relation: "supports",
      confidence: 0.8,
      evidence_episode_ids: [publicEpisode.id],
      created_at: 1_000_000,
      last_verified_at: 1_000_000,
    });

    const result = await harness.retrievalPipeline.recallEpisodesForCognition(query, {
      ...cognitionRecallOptions(bob),
      limit: 5,
      graphWalkDepth: 1,
      maxGraphNodes: 4,
    });
    const edgeEvidence = result.evidence.find(
      (item) => item.source === "semantic_edge" && item.provenance?.edgeId === edge.id,
    );

    expect(edgeEvidence?.text).toContain("Alice Private Support");
    expect(edgeEvidence?.source_episode_ids).toEqual([privateEpisode.id, publicEpisode.id]);
    expect(edgeEvidence?.disclosureLabel).toEqual({
      disclosureClass: "relationship_private",
      originAudienceEntityIds: [alice],
      privateToEntityIds: [alice],
      publicToEntityIds: [],
    });
  });

  it("keeps semantic nodes whose source episodes include visible evidence", async () => {
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
    await harness.episodicRepository.createEpisode(publicEpisode);
    await harness.episodicRepository.createEpisode(hiddenEpisode);
    const mixedNode = createSemanticNodeFixture({
      kind: "entity",
      label: "Atlas Audience Scoped",
      description: "Atlas node backed by both public and hidden evidence.",
      source_episode_ids: [publicEpisode.id, hiddenEpisode.id],
    });
    await harness.semanticNodeRepository.insert(mixedNode);

    const result = await harness.retrievalPipeline.searchWithContextForDisclosure(
      "Atlas Audience Scoped",
      {
        limit: 2,
        audienceEntityId: alex,
        graphWalkDepth: 1,
        maxGraphNodes: 4,
      },
    );

    expect(result.semantic.matched_node_ids).toContain(mixedNode.id);
    expect(result.semantic.matched_nodes).toEqual([
      expect.objectContaining({
        id: mixedNode.id,
        partial_source_visibility: true,
        source_visibility_fraction: 0.5,
        source_episode_ids: [publicEpisode.id],
        disclosureLabel: {
          disclosureClass: "relationship_private",
          originAudienceEntityIds: [sam],
          privateToEntityIds: [sam],
          publicToEntityIds: [],
        },
      }),
    ]);
    expect(result.semantic.supports).toEqual([]);
    expect(result.semantic.contradicts).toEqual([]);
    expect(result.semantic.categories).toEqual([]);
    expect(result.semantic.support_hits).toEqual([]);
    expect(result.semantic.contradiction_hits).toEqual([]);
    expect(result.semantic.category_hits).toEqual([]);
  });

  it("marks semantic-edge evidence partial when the target node sources were pruned", async () => {
    const query = "atlas public root mixed support";
    harness = await createOfflineTestHarness({
      embeddingClient: new TestEmbeddingClient(new Map([[query, [1, 0, 0, 0]]])),
    });
    const alice = harness.entityRepository.resolve("Alice");
    const bob = harness.entityRepository.resolve("Bob");
    const rootEpisode = createEpisodeFixture({
      title: "Atlas public root source",
      tags: ["atlas"],
      audience_entity_id: null,
      shared: true,
    });
    const visibleSupportEpisode = createEpisodeFixture({
      title: "Atlas public support source",
      tags: ["atlas"],
      audience_entity_id: null,
      shared: true,
    });
    const hiddenSupportEpisode = createEpisodeFixture({
      title: "Atlas Alice-private support source",
      tags: ["atlas"],
      audience_entity_id: alice,
      shared: false,
    });
    const edgeEpisode = createEpisodeFixture({
      title: "Atlas public edge source",
      tags: ["atlas"],
      audience_entity_id: null,
      shared: true,
    });
    await harness.episodicRepository.createEpisode(rootEpisode);
    await harness.episodicRepository.createEpisode(visibleSupportEpisode);
    await harness.episodicRepository.createEpisode(hiddenSupportEpisode);
    await harness.episodicRepository.createEpisode(edgeEpisode);
    const root = await harness.semanticNodeRepository.insert(
      createSemanticNodeFixture(
        {
          kind: "entity",
          label: "Atlas Public Root",
          description: "Atlas root backed by public evidence.",
          source_episode_ids: [rootEpisode.id],
        },
        [1, 0, 0, 0],
      ),
    );
    const mixedSupport = await harness.semanticNodeRepository.insert(
      createSemanticNodeFixture(
        {
          kind: "proposition",
          label: "Atlas Mixed Support",
          description: "Mixed support should keep the edge evidence partial flag.",
          source_episode_ids: [visibleSupportEpisode.id, hiddenSupportEpisode.id],
        },
        [0, 1, 0, 0],
      ),
    );
    const edge = harness.semanticEdgeRepository.addEdge({
      from_node_id: root.id,
      to_node_id: mixedSupport.id,
      relation: "supports",
      confidence: 0.8,
      evidence_episode_ids: [edgeEpisode.id],
      created_at: 1_000_000,
      last_verified_at: 1_000_000,
    });

    const result = await harness.retrievalPipeline.searchWithContextForDisclosure(query, {
      limit: 5,
      audienceEntityId: bob,
      graphWalkDepth: 1,
      maxGraphNodes: 4,
    });
    const edgeEvidence = result.evidence.find(
      (item) => item.source === "semantic_edge" && item.provenance?.edgeId === edge.id,
    );

    expect(edgeEvidence?.source_episode_ids).toEqual([visibleSupportEpisode.id, edgeEpisode.id]);
    expect(edgeEvidence?.partial_source_visibility).toBe(true);
    expect(edgeEvidence?.source_visibility_fraction).toBe(0.5);
    expect(edgeEvidence?.disclosureLabel).toEqual({
      disclosureClass: "relationship_private",
      originAudienceEntityIds: [alice],
      privateToEntityIds: [alice],
      publicToEntityIds: [],
    });
  });
});
