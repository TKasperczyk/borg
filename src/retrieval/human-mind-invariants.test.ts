import { afterEach, describe, expect, it } from "vitest";

import {
  TestEmbeddingClient,
  createEpisodeFixture,
  createOfflineTestHarness,
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

  it.fails(
    "recalls an Alice-private episode for Sol cognition during a Bob-audience turn",
    async () => {
      // Expected to flip in Sprint 2 when cognition recall stops using audience visibility gates.
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

      const result = await harness.retrievalPipeline.searchWithContext(BOB_RECALL_QUERY, {
        limit: 3,
        audienceEntityId: bobId,
        recallContext,
        disclosureContext,
        recordRetrieval: false,
      });

      expect(result.episodes.map((item) => item.episode.id)).toContain(episode.id);
    },
  );

  it.fails("labels the recalled Alice-private episode as private to Alice", async () => {
    // Expected to flip in Sprint 2; Sprint 1 adds the label type and attachment point.
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

    const result = await harness.retrievalPipeline.searchWithContext(BOB_RECALL_QUERY, {
      limit: 3,
      audienceEntityId: bobId,
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

  it.fails(
    "recalls cross-audience prior activity for an operator with no participant present",
    async () => {
      // Expected to flip in Sprint 2 when operator cognition recall is no longer audience-pruned.
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

      const result = await harness.retrievalPipeline.searchWithContext(OPERATOR_RECALL_QUERY, {
        limit: 3,
        audienceEntityId: operatorId,
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
        recordRetrieval: false,
      });

      expect(result.episodes.map((item) => item.episode.id)).toContain(episode.id);
    },
  );
});
