import { mkdtempSync, rmSync } from "node:fs";
import { join } from "node:path";
import { tmpdir } from "node:os";

import { afterEach, describe, expect, it } from "vitest";

import { Borg } from "../borg.js";
import { DEFAULT_CONFIG } from "../config/index.js";
import type { EmbeddingClient } from "../embeddings/index.js";
import { FakeLLMClient } from "../llm/index.js";
import {
  createEpisodeFixture,
  createOfflineTestHarness,
  createSemanticNodeFixture,
} from "../offline/test-support.js";
import { FixedClock } from "../util/clock.js";
import { CorrectionService } from "./service.js";

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
): CorrectionService {
  return new CorrectionService({
    config: harness.config,
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
    identityEventRepository: harness.identityEventRepository,
  });
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
          useLlmFallback: false,
          modeWhenLlmAbsent: "problem_solving",
        },
        anthropic: {
          auth: "api-key",
          apiKey: "test",
          models: {
            cognition: "sonnet",
            background: "haiku",
            extraction: "haiku",
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
          useLlmFallback: false,
          modeWhenLlmAbsent: "problem_solving",
        },
        anthropic: {
          auth: "api-key",
          apiKey: "test",
          models: {
            cognition: "sonnet",
            background: "haiku",
            extraction: "haiku",
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
      const episode = await harness.episodicRepository.insert(
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
          useLlmFallback: false,
          modeWhenLlmAbsent: "problem_solving",
        },
        anthropic: {
          auth: "api-key",
          apiKey: "test",
          models: {
            cognition: "sonnet",
            background: "haiku",
            extraction: "haiku",
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
});
