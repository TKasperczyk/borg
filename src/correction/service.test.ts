import { mkdtempSync, rmSync } from "node:fs";
import { join } from "node:path";
import { tmpdir } from "node:os";

import { afterEach, describe, expect, it, vi } from "vitest";

import { Borg } from "../borg.js";
import { DEFAULT_CONFIG } from "../config/index.js";
import type { EmbeddingClient } from "../embeddings/index.js";
import { FakeLLMClient } from "../llm/test-support/fake-client.js";
import {
  createEpisodeFixture,
  createOfflineTestHarness,
  createSemanticNodeFixture,
} from "../offline/test-support.js";
import { FixedClock } from "../util/clock.js";
import { createEntityId } from "../util/ids.js";
import { GOAL_TURN_ROLLBACK_REASON } from "../memory/self/index.js";
import { CorrectionService, type CorrectionServiceOptions } from "./service.js";

class TestEmbeddingClient implements EmbeddingClient {
  async embed(): Promise<Float32Array> {
    return Float32Array.from([1, 0, 0, 0]);
  }

  async embedBatch(texts: readonly string[]): Promise<Float32Array[]> {
    return texts.map(() => Float32Array.from([1, 0, 0, 0]));
  }
}

function createHarnessCorrectionService(
  harness: Awaited<ReturnType<typeof createOfflineTestHarness>>,
  overrides: Partial<Pick<CorrectionServiceOptions, "identityEventRepository">> = {},
): CorrectionService {
  return new CorrectionService({
    config: harness.config,
    db: harness.db,
    clock: harness.clock,
    retrievalPipeline: harness.retrievalPipeline,
    episodicRepository: harness.episodicRepository,
    semanticNodeRepository: harness.semanticNodeRepository,
    semanticEdgeRepository: harness.semanticEdgeRepository,
    semanticGraph: harness.semanticGraph,
    valuesRepository: harness.valuesRepository,
    goalsRepository: harness.goalsRepository,
    traitsRepository: harness.traitsRepository,
    openQuestionsRepository: harness.openQuestionsRepository,
    socialRepository: harness.socialRepository,
    entityRepository: harness.entityRepository,
    commitmentRepository: harness.commitmentRepository,
    reviewQueueRepository: harness.reviewQueueRepository,
    identityService: harness.identityService,
    identityEventRepository: overrides.identityEventRepository ?? harness.identityEventRepository,
  });
}

function expectNoWhyVectors(value: unknown): void {
  if (value === null || typeof value !== "object") {
    return;
  }

  expect(ArrayBuffer.isView(value)).toBe(false);

  if (Array.isArray(value)) {
    for (const entry of value) {
      expectNoWhyVectors(entry);
    }
    return;
  }

  for (const [key, entry] of Object.entries(value)) {
    expect(key).not.toBe("embedding");
    expect(key).not.toBe("vector");
    expectNoWhyVectors(entry);
  }
}

describe("correction service", () => {
  const tempDirs: string[] = [];

  afterEach(async () => {
    while (tempDirs.length > 0) {
      rmSync(tempDirs.pop() as string, { recursive: true, force: true });
    }
  });

  it("queues corrections and applies them through review resolution", async () => {
    const tempDir = mkdtempSync(join(tmpdir(), "borg-"));
    tempDirs.push(tempDir);
    const borg = await Borg.open({
      config: {
        ...DEFAULT_CONFIG,
        dataDir: tempDir,
        defaultUser: "Sam",
        embedding: {
          ...DEFAULT_CONFIG.embedding,
          dims: 4,
        },
        perception: {
          llmEnabled: false,
        },
        anthropic: {
          ...DEFAULT_CONFIG.anthropic,
          auth: "api-key",
          apiKey: "test",
          models: {
            cognition: "sonnet",
            background: "haiku",
            extraction: "haiku",
            recallExpansion: "haiku",
            creatorDirective: "sonnet",
            imagePerception: "haiku",
          },
        },
      },
      clock: new FixedClock(1_000),
      embeddingDimensions: 4,
      embeddingClient: new TestEmbeddingClient(),
      llmClient: new FakeLLMClient(),
    });

    try {
      const value = borg.self.values.add({
        label: "clarity",
        description: "Prefer explicit state.",
        priority: 5,
        provenance: {
          kind: "manual",
        },
      });

      const queued = await borg.correction.correct(value.id, {
        description: "Prefer explicit state and reviewable changes.",
      });

      expect(queued.kind).toBe("correction");
      expect(queued.reason).toContain(new Date(1_000).toISOString());

      const resolved = await borg.review.resolve(queued.id, "accept");

      expect(resolved?.resolution).toBe("accept");
      expect(borg.self.values.get(value.id)?.description).toBe(
        "Prefer explicit state and reviewable changes.",
      );
      expect(
        borg.correction.listIdentityEvents({
          recordType: "value",
          recordId: value.id,
        }),
      ).toEqual(
        expect.arrayContaining([
          expect.objectContaining({
            action: "correction_apply",
            review_item_id: queued.id,
          }),
        ]),
      );
    } finally {
      await borg.close();
    }
  });

  it("preserves proposer provenance when a reviewed correction is accepted", async () => {
    const tempDir = mkdtempSync(join(tmpdir(), "borg-"));
    tempDirs.push(tempDir);
    const borg = await Borg.open({
      config: {
        ...DEFAULT_CONFIG,
        dataDir: tempDir,
        defaultUser: "Sam",
        embedding: {
          ...DEFAULT_CONFIG.embedding,
          dims: 4,
        },
        perception: {
          llmEnabled: false,
        },
        anthropic: {
          ...DEFAULT_CONFIG.anthropic,
          auth: "api-key",
          apiKey: "test",
          models: {
            cognition: "sonnet",
            background: "haiku",
            extraction: "haiku",
            recallExpansion: "haiku",
            creatorDirective: "sonnet",
            imagePerception: "haiku",
          },
        },
      },
      clock: new FixedClock(1_500),
      embeddingDimensions: 4,
      embeddingClient: new TestEmbeddingClient(),
      llmClient: new FakeLLMClient(),
    });

    try {
      const value = borg.self.values.add({
        label: "groundedness",
        description: "Stay anchored to evidence.",
        priority: 6,
        provenance: {
          kind: "manual",
        },
      });

      const queued = await borg.correction.correct(
        value.id,
        {
          description: "Stay anchored to lived evidence.",
        },
        {
          kind: "offline",
          process: "reflector",
        },
      );

      await borg.review.resolve(queued.id, "accept");

      expect(borg.self.values.get(value.id)?.provenance).toEqual({
        kind: "offline",
        process: "reflector",
      });
      expect(
        borg.correction.listIdentityEvents({
          recordType: "value",
          recordId: value.id,
        }),
      ).toEqual(
        expect.arrayContaining([
          expect.objectContaining({
            action: "correction_apply",
            provenance: {
              kind: "offline",
              process: "reflector",
            },
          }),
        ]),
      );
    } finally {
      await borg.close();
    }
  });

  it("does not duplicate audit events when an episode correction is retried", async () => {
    const harness = await createOfflineTestHarness({
      clock: new FixedClock(1_000_500),
    });

    try {
      const correction = new CorrectionService({
        config: harness.config,
        db: harness.db,
        retrievalPipeline: harness.retrievalPipeline,
        episodicRepository: harness.episodicRepository,
        semanticNodeRepository: harness.semanticNodeRepository,
        semanticEdgeRepository: harness.semanticEdgeRepository,
        semanticGraph: harness.semanticGraph,
        valuesRepository: harness.valuesRepository,
        goalsRepository: harness.goalsRepository,
        traitsRepository: harness.traitsRepository,
        openQuestionsRepository: harness.openQuestionsRepository,
        socialRepository: harness.socialRepository,
        entityRepository: harness.entityRepository,
        commitmentRepository: harness.commitmentRepository,
        reviewQueueRepository: harness.reviewQueueRepository,
        identityService: harness.identityService,
        identityEventRepository: harness.identityEventRepository,
      });
      const episode = await harness.episodicRepository.createEpisode(
        createEpisodeFixture({
          title: "Planning sync",
          narrative: "Original narrative.",
        }),
      );
      const item = harness.reviewQueueRepository.enqueue({
        kind: "correction",
        refs: {
          target_id: episode.id,
          target_type: "episode",
          patch: {
            narrative: "Corrected narrative.",
          },
          proposed_provenance: {
            kind: "manual",
          },
        },
        reason: "user corrected the episode narrative",
      });

      await correction.applyCorrectionReview(item);
      await correction.applyCorrectionReview(item);

      const events = harness.identityEventRepository
        .list({
          recordType: "episode",
          recordId: episode.id,
          limit: 10,
        })
        .filter((event) => event.action === "correction_apply");

      expect((await harness.episodicRepository.get(episode.id))?.narrative).toBe(
        "Corrected narrative.",
      );
      expect(events).toHaveLength(1);
      expect(events[0]).toEqual(
        expect.objectContaining({
          review_item_id: item.id,
          old_value: expect.objectContaining({
            narrative: "Original narrative.",
          }),
          new_value: expect.objectContaining({
            narrative: "Corrected narrative.",
          }),
        }),
      );
      expect(events).not.toEqual(
        expect.arrayContaining([
          expect.objectContaining({
            old_value: expect.objectContaining({
              narrative: "Corrected narrative.",
            }),
          }),
        ]),
      );
    } finally {
      await harness.cleanup();
    }
  });

  it("strips vector data from why payload records and nested semantic nodes", async () => {
    const harness = await createOfflineTestHarness({
      clock: new FixedClock(3_000),
    });

    try {
      const correction = createHarnessCorrectionService(harness);
      const episode = await harness.episodicRepository.createEpisode(
        createEpisodeFixture(
          {
            title: "Vector payload fixture",
            narrative: "The why payload should not ship raw embeddings.",
          },
          [0.1, 0.2, 0.3, 0.4],
        ),
      );
      const first = await harness.semanticNodeRepository.insert(
        createSemanticNodeFixture(
          {
            label: "why sanitizer source",
            description: "Source node for why payload sanitizer coverage.",
            source_episode_ids: [episode.id],
          },
          [0.5, 0.6, 0.7, 0.8],
        ),
      );
      const second = await harness.semanticNodeRepository.insert(
        createSemanticNodeFixture(
          {
            label: "why sanitizer target",
            description: "Target node for nested semantic why payload coverage.",
            source_episode_ids: [episode.id],
          },
          [0.9, 1, 0.1, 0.2],
        ),
      );
      const edge = harness.semanticEdgeRepository.addEdge({
        from_node_id: first.id,
        to_node_id: second.id,
        relation: "supports",
        confidence: 0.8,
        evidence_episode_ids: [episode.id],
        created_at: 3_000,
        last_verified_at: 3_000,
      });

      const episodeWhy = await correction.why(episode.id);
      const nodeWhy = await correction.why(first.id);
      const edgeWhy = await correction.why(edge.id);

      expectNoWhyVectors(episodeWhy);
      expectNoWhyVectors(nodeWhy);
      expectNoWhyVectors(edgeWhy);
      expect(episodeWhy.record).not.toHaveProperty("embedding");
      expect(nodeWhy.record).not.toHaveProperty("embedding");
      expect((nodeWhy.walked_edges as Array<{ node?: unknown }>)[0]?.node).not.toHaveProperty(
        "embedding",
      );
      expect(edgeWhy.from_node).not.toHaveProperty("embedding");
      expect(edgeWhy.to_node).not.toHaveProperty("embedding");
    } finally {
      await harness.cleanup();
    }
  });

  it("labels open-question corrections from the question audience instead of falling back public", async () => {
    const harness = await createOfflineTestHarness({
      clock: new FixedClock(2_000),
    });

    try {
      const correction = createHarnessCorrectionService(harness);
      const alice = createEntityId();
      const question = harness.openQuestionsRepository.add({
        question: "Which Alice-scoped correction label should render?",
        urgency: 0.5,
        audience_entity_id: alice,
        provenance: { kind: "manual" },
        source: "reflection",
      });

      const queued = await correction.correct(question.id, {
        urgency: 0.75,
      });

      expect(queued.refs).toMatchObject({
        target_id: question.id,
        target_type: "open_question",
        audience_entity_id: alice,
        disclosure_label: {
          disclosure_class: "relationship_private",
          origin_audience_entity_ids: [alice],
          private_to_entity_ids: [alice],
          public_to_entity_ids: [],
        },
      });
    } finally {
      await harness.cleanup();
    }
  });

  it("invalidates semantic edges manually with explicit event time idempotently", async () => {
    const harness = await createOfflineTestHarness({
      clock: new FixedClock(5_000),
    });

    try {
      const correction = createHarnessCorrectionService(harness);
      const episodeId = createEpisodeFixture().id;
      const first = await harness.semanticNodeRepository.insert(
        createSemanticNodeFixture(
          {
            label: "Atlas manual revoke source",
            description: "Atlas was stable.",
            source_episode_ids: [episodeId],
          },
          [1, 0, 0, 0],
        ),
      );
      const second = await harness.semanticNodeRepository.insert(
        createSemanticNodeFixture(
          {
            label: "Atlas manual revoke target",
            description: "Rollback was complete.",
            source_episode_ids: [episodeId],
          },
          [0, 1, 0, 0],
        ),
      );
      const edge = harness.semanticEdgeRepository.addEdge({
        from_node_id: first.id,
        to_node_id: second.id,
        relation: "supports",
        confidence: 0.8,
        evidence_episode_ids: [episodeId],
        created_at: 4_000,
        last_verified_at: 4_000,
        valid_from: 4_000,
      });

      const invalidated = correction.invalidateSemanticEdge(edge.id, {
        at: 4_500,
        reason: "manual revoke",
      });
      const secondCall = correction.invalidateSemanticEdge(edge.id, {
        at: 4_900,
        reason: "second call should be idempotent",
      });
      const events = harness.identityEventRepository.list({
        recordType: "semantic_edge",
        recordId: edge.id,
      });

      expect(edge.id).toMatch(/^seme_/);
      expect(invalidated).toEqual(
        expect.objectContaining({
          id: edge.id,
          valid_to: 4_500,
          invalidated_at: 5_000,
          invalidated_by_process: "manual",
          invalidated_reason: "manual revoke",
        }),
      );
      expect(secondCall).toEqual(invalidated);
      expect(events).toHaveLength(1);
      expect(events[0]?.new_value).toEqual(
        expect.objectContaining({
          edge_id: edge.id,
          prior_valid_to: null,
          new_valid_to: 4_500,
          by_process: "manual",
          reason: "manual revoke",
        }),
      );
    } finally {
      await harness.cleanup();
    }
  });

  it("rolls back manual semantic edge invalidation when audit recording fails", async () => {
    const harness = await createOfflineTestHarness({
      clock: new FixedClock(6_000),
    });

    try {
      const originalRecord = harness.identityEventRepository.record.bind(
        harness.identityEventRepository,
      );
      let recordCalls = 0;
      const throwingIdentityEventRepository = Object.create(
        harness.identityEventRepository,
      ) as typeof harness.identityEventRepository;
      throwingIdentityEventRepository.record = ((
        input: Parameters<typeof harness.identityEventRepository.record>[0],
      ) => {
        recordCalls += 1;

        if (recordCalls === 1) {
          throw new Error("audit write failed");
        }

        return originalRecord(input);
      }) as typeof harness.identityEventRepository.record;
      const failingCorrection = createHarnessCorrectionService(harness, {
        identityEventRepository: throwingIdentityEventRepository,
      });
      const retryCorrection = createHarnessCorrectionService(harness);
      const episodeId = createEpisodeFixture().id;
      const first = await harness.semanticNodeRepository.insert(
        createSemanticNodeFixture(
          {
            label: "Atlas transaction source",
            description: "Atlas was stable before correction.",
            source_episode_ids: [episodeId],
          },
          [1, 0, 0, 0],
        ),
      );
      const second = await harness.semanticNodeRepository.insert(
        createSemanticNodeFixture(
          {
            label: "Atlas transaction target",
            description: "Rollback state depended on audit integrity.",
            source_episode_ids: [episodeId],
          },
          [0, 1, 0, 0],
        ),
      );
      const edge = harness.semanticEdgeRepository.addEdge({
        from_node_id: first.id,
        to_node_id: second.id,
        relation: "supports",
        confidence: 0.8,
        evidence_episode_ids: [episodeId],
        created_at: 5_000,
        last_verified_at: 5_000,
        valid_from: 5_000,
      });

      expect(() =>
        failingCorrection.invalidateSemanticEdge(edge.id, {
          at: 5_500,
          reason: "manual revoke",
        }),
      ).toThrow("audit write failed");
      expect(harness.semanticEdgeRepository.getEdge(edge.id)?.valid_to).toBeNull();
      expect(
        harness.identityEventRepository.list({
          recordType: "semantic_edge",
          recordId: edge.id,
        }),
      ).toEqual([]);

      const invalidated = retryCorrection.invalidateSemanticEdge(edge.id, {
        at: 5_500,
        reason: "manual revoke",
      });
      const events = harness.identityEventRepository.list({
        recordType: "semantic_edge",
        recordId: edge.id,
      });

      expect(invalidated.valid_to).toBe(5_500);
      expect(events).toEqual([
        expect.objectContaining({
          action: "edge_invalidate",
          record_id: edge.id,
        }),
      ]);
    } finally {
      await harness.cleanup();
    }
  });

  it("surfaces a clean error for nonexistent semantic edge invalidation", async () => {
    const harness = await createOfflineTestHarness();

    try {
      const correction = createHarnessCorrectionService(harness);

      let thrown: unknown;
      try {
        correction.invalidateSemanticEdge("seme_aaaaaaaaaaaaaaaa");
      } catch (error) {
        thrown = error;
      }

      expect(thrown).toBeInstanceOf(Error);
      expect(thrown).toMatchObject({
        code: "SEMANTIC_EDGE_NOT_FOUND",
        message: expect.stringContaining("Unknown semantic edge id"),
      });
    } finally {
      await harness.cleanup();
    }
  });

  it("supports forgetting records and remembering the default user", async () => {
    const tempDir = mkdtempSync(join(tmpdir(), "borg-"));
    tempDirs.push(tempDir);
    const borg = await Borg.open({
      config: {
        ...DEFAULT_CONFIG,
        dataDir: tempDir,
        defaultUser: "Sam",
        embedding: {
          ...DEFAULT_CONFIG.embedding,
          dims: 4,
        },
        perception: {
          llmEnabled: false,
        },
        anthropic: {
          ...DEFAULT_CONFIG.anthropic,
          auth: "api-key",
          apiKey: "test",
          models: {
            cognition: "sonnet",
            background: "haiku",
            extraction: "haiku",
            recallExpansion: "haiku",
            creatorDirective: "sonnet",
            imagePerception: "haiku",
          },
        },
      },
      clock: new FixedClock(2_000),
      embeddingDimensions: 4,
      embeddingClient: new TestEmbeddingClient(),
      llmClient: new FakeLLMClient(),
    });

    try {
      const value = borg.self.values.add({
        label: "memory",
        description: "Keep a usable trace.",
        priority: 2,
        provenance: {
          kind: "manual",
        },
      });
      borg.commitments.add({
        type: "boundary",
        directiveFamily: "sam_memory_changes",
        directive: "Keep Sam posted on memory changes",
        priority: 7,
        audience: "Sam",
        provenance: {
          kind: "manual",
        },
      });
      borg.social.recordInteraction("Sam", {
        provenance: {
          kind: "manual",
        },
        valence: 0.2,
      });

      const forgotten = await borg.correction.forget(value.id);
      const aboutMe = await borg.correction.rememberAboutMe();
      const why = await borg.correction.why(value.id).catch((error) => error);

      expect(forgotten).toEqual(
        expect.objectContaining({
          id: value.id,
          archived: true,
        }),
      );
      expect(borg.self.values.get(value.id)).toBeNull();
      expect(aboutMe.social_profile?.interaction_count).toBeGreaterThan(0);
      expect(aboutMe.active_commitments).toHaveLength(1);
      expect(why).toBeInstanceOf(Error);
      expect(
        borg.correction.listIdentityEvents({
          recordType: "value",
          recordId: value.id,
        }),
      ).toEqual(
        expect.arrayContaining([
          expect.objectContaining({
            action: "forget",
          }),
        ]),
      );
    } finally {
      await borg.close();
    }
  });

  it("forgets open questions through the identity service transaction", async () => {
    const harness = await createOfflineTestHarness({
      clock: new FixedClock(2_500),
    });

    try {
      const correction = createHarnessCorrectionService(harness);
      const question = harness.openQuestionsRepository.add({
        question: "Which forget path should be transactional?",
        urgency: 0.5,
        related_episode_ids: [createEpisodeFixture().id],
        source: "reflection",
      });

      await expect(correction.forget(question.id)).resolves.toEqual(
        expect.objectContaining({
          id: question.id,
          archived: true,
        }),
      );
      expect(harness.openQuestionsRepository.get(question.id)).toEqual(
        expect.objectContaining({
          status: "abandoned",
          abandoned_reason: "forgotten manually",
        }),
      );
      expect(
        harness.identityEventRepository.list({
          recordType: "open_question",
          recordId: question.id,
        }),
      ).toEqual(
        expect.arrayContaining([
          expect.objectContaining({
            action: "abandon",
            reason: "forgotten manually",
          }),
        ]),
      );

      const rolledBackQuestion = harness.openQuestionsRepository.add({
        question: "Will failed forget roll back?",
        urgency: 0.4,
        related_episode_ids: [createEpisodeFixture().id],
        source: "reflection",
      });
      const eventError = new Error("identity event insert failed");
      const recordSpy = vi
        .spyOn(harness.identityEventRepository, "record")
        .mockImplementation(() => {
          throw eventError;
        });

      try {
        await expect(correction.forget(rolledBackQuestion.id)).rejects.toThrow(eventError);
      } finally {
        recordSpy.mockRestore();
      }

      expect(harness.openQuestionsRepository.get(rolledBackQuestion.id)).toEqual(
        expect.objectContaining({
          status: "open",
          abandoned_reason: null,
        }),
      );
    } finally {
      await harness.cleanup();
    }
  });

  it("keeps the repository delete and correction forget audit events for goals", async () => {
    const harness = await createOfflineTestHarness({
      clock: new FixedClock(2_750),
    });

    try {
      const correction = createHarnessCorrectionService(harness);
      const goal = harness.goalsRepository.add({
        description: "Verify both intentional goal deletion audit layers",
        priority: 6,
        provenance: { kind: "manual" },
      });

      await expect(correction.forget(goal.id)).resolves.toMatchObject({
        id: goal.id,
        target_type: "goal",
        archived: true,
      });
      expect(harness.goalsRepository.get(goal.id)).toBeNull();

      const events = harness.identityEventRepository.list({
        recordType: "goal",
        recordId: goal.id,
        limit: 10,
      });
      expect(events.map((event) => event.action)).toEqual(["forget", "delete", "create"]);
      expect(events.find((event) => event.action === "delete")).toMatchObject({
        old_value: goal,
        new_value: null,
        reason: GOAL_TURN_ROLLBACK_REASON,
      });
      expect(events.find((event) => event.action === "forget")).toMatchObject({
        old_value: goal,
        new_value: null,
        reason: "forgotten manually",
      });
    } finally {
      await harness.cleanup();
    }
  });

  it("uses the injected clock when synthesizing a missing remember-about-me entity", async () => {
    const harness = await createOfflineTestHarness({
      clock: new FixedClock(7_000),
    });

    try {
      const correction = createHarnessCorrectionService(harness);
      const originalGet = harness.entityRepository.get.bind(harness.entityRepository);
      harness.entityRepository.get = (() => null) as typeof harness.entityRepository.get;
      const dateNow = vi.spyOn(Date, "now").mockImplementation(() => {
        throw new Error("wall clock used");
      });

      try {
        await expect(correction.rememberAboutMe({ entity: "Sam" })).resolves.toEqual(
          expect.objectContaining({
            entity: null,
          }),
        );
      } finally {
        dateNow.mockRestore();
        harness.entityRepository.get = originalGet;
      }
    } finally {
      await harness.cleanup();
    }
  });
});
