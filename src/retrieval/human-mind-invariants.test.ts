import { afterEach, describe, expect, it } from "vitest";

import {
  TestEmbeddingClient,
  createEpisodeFixture,
  createOfflineTestHarness,
  createSemanticNodeFixture,
} from "../offline/test-support.js";
import { DEFAULT_SESSION_ID } from "../util/ids.js";
import type { CognitionRecallContext, DisclosureContext } from "./recall-context.js";

const BOB_RECALL_QUERY = "human mind invariant bob recall";
const OPERATOR_RECALL_QUERY = "human mind invariant operator recall";
const MATCH_VECTOR = [1, 0, 0, 0];

function embeddingClient() {
  return new TestEmbeddingClient(
    new Map([
      [BOB_RECALL_QUERY, MATCH_VECTOR],
      [OPERATOR_RECALL_QUERY, MATCH_VECTOR],
    ]),
  );
}

describe("human-mind memory invariants", () => {
  let harness: Awaited<ReturnType<typeof createOfflineTestHarness>> | undefined;

  afterEach(async () => {
    await harness?.cleanup();
    harness = undefined;
  });

  it("recalls an Alice-private episode for Sol cognition during a Bob-audience turn", async () => {
    harness = await createOfflineTestHarness({ embeddingClient: embeddingClient() });
    const aliceId = harness.entityRepository.resolve("Alice");
    const bobId = harness.entityRepository.resolve("Bob");
    const episode = createEpisodeFixture(
      {
        title: "Alice private recall invariant",
        narrative: "A structurally private Alice memory exists for Sol cognition.",
        participants: ["Alice"],
        tags: ["human-mind-invariant"],
        audience_entity_id: aliceId,
        shared: false,
      },
      MATCH_VECTOR,
    );
    await harness.episodicRepository.insert(episode);

    const recallContext: CognitionRecallContext = {
      reader: "sol",
      currentSessionId: DEFAULT_SESSION_ID,
      currentAudienceEntityId: bobId,
      currentParticipantEntityIds: [bobId],
    };
    const disclosureContext: DisclosureContext = {
      currentSessionId: DEFAULT_SESSION_ID,
      currentAudienceEntityId: bobId,
      audienceRole: "participant",
      senderEntityId: bobId,
      senderRole: null,
      participantEntityIds: [bobId],
      isPrivateSelfCognition: false,
    };

    const result = await harness.retrievalPipeline.recallEpisodesForCognition(BOB_RECALL_QUERY, {
      limit: 3,
      recallContext,
      disclosureContext,
      rankingAudienceEntityId: bobId,
      recordRetrieval: false,
    });

    const recalled = result.episodes.find((item) => item.episode.id === episode.id);

    expect(recalled).toBeDefined();
    expect(recalled?.disclosureLabel).toEqual({
      disclosureClass: "relationship_private",
      originAudienceEntityIds: [aliceId],
      privateToEntityIds: [aliceId],
      publicToEntityIds: [],
    });
  });

  it("labels the recalled Alice-private episode as private to Alice", async () => {
    harness = await createOfflineTestHarness({ embeddingClient: embeddingClient() });
    const aliceId = harness.entityRepository.resolve("Alice");
    const bobId = harness.entityRepository.resolve("Bob");
    const episode = createEpisodeFixture(
      {
        title: "Alice private disclosure-label invariant",
        narrative: "A private Alice memory should be recalled with disclosure metadata.",
        participants: ["Alice"],
        tags: ["human-mind-invariant"],
        audience_entity_id: aliceId,
        shared: false,
      },
      MATCH_VECTOR,
    );
    await harness.episodicRepository.insert(episode);

    const result = await harness.retrievalPipeline.recallEpisodesForCognition(BOB_RECALL_QUERY, {
      limit: 3,
      recallContext: {
        reader: "sol",
        currentSessionId: DEFAULT_SESSION_ID,
        currentAudienceEntityId: bobId,
        currentParticipantEntityIds: [bobId],
      },
      disclosureContext: {
        currentSessionId: DEFAULT_SESSION_ID,
        currentAudienceEntityId: bobId,
        audienceRole: "participant",
        senderEntityId: bobId,
        senderRole: null,
        participantEntityIds: [bobId],
        isPrivateSelfCognition: false,
      },
      rankingAudienceEntityId: bobId,
      recordRetrieval: false,
    });
    const recalled = result.episodes.find((item) => item.episode.id === episode.id);

    expect(recalled?.disclosureLabel).toMatchObject({
      disclosureClass: "relationship_private",
      originAudienceEntityIds: [aliceId],
      privateToEntityIds: [aliceId],
      publicToEntityIds: [],
    });
  });

  it("recalls cross-audience prior activity for an operator with no participant present", async () => {
    harness = await createOfflineTestHarness({ embeddingClient: embeddingClient() });
    const aliceId = harness.entityRepository.resolve("Alice");
    const operatorId = harness.entityRepository.resolve("Operator");
    const episode = createEpisodeFixture(
      {
        title: "Alice prior activity invariant",
        narrative: "A prior cross-audience activity record should remain recallable to Sol.",
        participants: ["Alice"],
        tags: ["human-mind-invariant"],
        audience_entity_id: aliceId,
        shared: false,
      },
      MATCH_VECTOR,
    );
    await harness.episodicRepository.insert(episode);

    const result = await harness.retrievalPipeline.recallEpisodesForCognition(
      OPERATOR_RECALL_QUERY,
      {
        limit: 3,
        recallContext: {
          reader: "sol",
          currentSessionId: DEFAULT_SESSION_ID,
          currentAudienceEntityId: operatorId,
          currentParticipantEntityIds: [],
        },
        disclosureContext: {
          currentSessionId: DEFAULT_SESSION_ID,
          currentAudienceEntityId: operatorId,
          audienceRole: "operator",
          senderEntityId: operatorId,
          senderRole: "creator",
          participantEntityIds: [],
          isPrivateSelfCognition: false,
        },
        rankingAudienceEntityId: operatorId,
        recordRetrieval: false,
      },
    );

    const recalled = result.episodes.find((item) => item.episode.id === episode.id);

    expect(recalled).toBeDefined();
    expect(recalled?.disclosureLabel).toEqual({
      disclosureClass: "relationship_private",
      originAudienceEntityIds: [aliceId],
      privateToEntityIds: [aliceId],
      publicToEntityIds: [],
    });
  });

  it("recalls Alice-private semantic node and edge evidence during a Bob-audience turn with labels", async () => {
    harness = await createOfflineTestHarness({ embeddingClient: embeddingClient() });
    const aliceId = harness.entityRepository.resolve("Alice");
    const bobId = harness.entityRepository.resolve("Bob");
    const episode = createEpisodeFixture(
      {
        title: "Alice private semantic invariant",
        narrative: "A structurally private Alice memory backs semantic recall.",
        participants: ["Alice"],
        tags: ["human-mind-invariant"],
        audience_entity_id: aliceId,
        shared: false,
      },
      MATCH_VECTOR,
    );
    await harness.episodicRepository.insert(episode);
    const root = await harness.semanticNodeRepository.insert(
      createSemanticNodeFixture(
        {
          label: "Alice private semantic root",
          description: "Alice private semantic root should remain recallable to Sol cognition.",
          source_episode_ids: [episode.id],
        },
        MATCH_VECTOR,
      ),
    );
    const support = await harness.semanticNodeRepository.insert(
      createSemanticNodeFixture(
        {
          label: "Alice private semantic support",
          description: "Alice private semantic support should carry disclosure metadata.",
          source_episode_ids: [episode.id],
        },
        [0, 1, 0, 0],
      ),
    );
    const edge = harness.semanticEdgeRepository.addEdge({
      from_node_id: root.id,
      to_node_id: support.id,
      relation: "supports",
      confidence: 0.8,
      evidence_episode_ids: [episode.id],
      created_at: 1_000_000,
      last_verified_at: 1_000_000,
    });

    const result = await harness.retrievalPipeline.recallEpisodesForCognition(BOB_RECALL_QUERY, {
      limit: 3,
      graphWalkDepth: 1,
      maxGraphNodes: 4,
      recallContext: {
        reader: "sol",
        currentSessionId: DEFAULT_SESSION_ID,
        currentAudienceEntityId: bobId,
        currentParticipantEntityIds: [bobId],
      },
      disclosureContext: {
        currentSessionId: DEFAULT_SESSION_ID,
        currentAudienceEntityId: bobId,
        audienceRole: "participant",
        senderEntityId: bobId,
        senderRole: null,
        participantEntityIds: [bobId],
        isPrivateSelfCognition: false,
      },
      rankingAudienceEntityId: bobId,
      recordRetrieval: false,
    });
    const recalledNode = result.semantic.matched_nodes.find((node) => node.id === root.id);
    const recalledHit = result.semantic.support_hits.find(
      (hit) =>
        hit.node.id === support.id && hit.edgePath.some((candidate) => candidate.id === edge.id),
    );
    const recalledEdge = recalledHit?.edgePath.find((candidate) => candidate.id === edge.id);

    expect(recalledNode?.disclosureLabel).toEqual({
      disclosureClass: "relationship_private",
      originAudienceEntityIds: [aliceId],
      privateToEntityIds: [aliceId],
      publicToEntityIds: [],
    });
    expect(recalledEdge?.disclosureLabel).toEqual({
      disclosureClass: "relationship_private",
      originAudienceEntityIds: [aliceId],
      privateToEntityIds: [aliceId],
      publicToEntityIds: [],
    });
  });
});
