import { mkdtempSync, rmSync } from "node:fs";
import { join } from "node:path";
import { tmpdir } from "node:os";

import { afterEach, describe, expect, it, vi } from "vitest";

import { FakeLLMClient } from "../../llm/test-support/fake-client.js";
import {
  createEpisodeFixture,
  createOfflineTestHarness,
  createSemanticNodeFixture,
} from "../../offline/test-support.js";
import { resolveSemanticContext } from "../../retrieval/semantic-retrieval.js";
import { LanceDbStore } from "../../storage/lancedb/index.js";
import { openDatabase } from "../../storage/sqlite/index.js";
import { StreamReader, StreamWriter } from "../../stream/index.js";
import { FixedClock } from "../../util/clock.js";
import { createSemanticNodeId, type EpisodeId, type SemanticNodeId } from "../../util/ids.js";
import { semanticMigrations } from "./migrations.js";
import { SemanticNodeRepository, createSemanticNodesTableSchema } from "./repository.js";
import { SemanticReviewService } from "./review-service.js";

const CONTRADICTION_TOOL_NAME = "EmitContradictionJudgment";

async function createSemanticFixture() {
  const tempDir = mkdtempSync(join(tmpdir(), "borg-"));
  const store = new LanceDbStore({
    uri: join(tempDir, "lancedb"),
  });
  const db = openDatabase(join(tempDir, "borg.db"), {
    migrations: semanticMigrations,
  });
  const table = await store.openTable({
    name: "semantic_nodes",
    schema: createSemanticNodesTableSchema(4),
  });
  const clock = new FixedClock(1_000);
  const nodeRepository = new SemanticNodeRepository({
    table,
    db,
    clock,
  });

  return {
    tempDir,
    store,
    db,
    table,
    clock,
    nodeRepository,
  };
}

function buildProposition(id: SemanticNodeId, label: string) {
  return {
    id,
    kind: "proposition" as const,
    label,
    description: `${label} description`,
    aliases: [],
    confidence: 0.7,
    source_episode_ids: ["ep_aaaaaaaaaaaaaaaa" as EpisodeId],
    created_at: 1_000,
    updated_at: 1_000,
    last_verified_at: 1_000,
    embedding: Float32Array.from([1, 0, 0, 0]),
    archived: false,
    superseded_by: null,
  };
}

describe("semantic review service", () => {
  const cleanup: Array<() => Promise<void>> = [];

  afterEach(async () => {
    vi.restoreAllMocks();

    while (cleanup.length > 0) {
      await cleanup.pop()?.();
    }
  });

  it("keeps repository inserts inert until duplicate review is explicitly requested", async () => {
    const fixture = await createSemanticFixture();
    const llm = new FakeLLMClient({
      responses: [
        {
          text: "",
          input_tokens: 1,
          output_tokens: 1,
          stop_reason: "tool_use",
          tool_calls: [
            {
              id: "toolu_1",
              name: CONTRADICTION_TOOL_NAME,
              input: { contradicts: true, confidence: 0.9 },
            },
          ],
        },
      ],
    });
    const enqueueReview = vi.fn();
    const reviewService = new SemanticReviewService({
      nodeRepository: fixture.nodeRepository,
      llmClient: llm,
      contradictionJudgeModel: "haiku",
      enqueueReview,
    });

    cleanup.push(async () => {
      fixture.db.close();
      await fixture.store.close();
      rmSync(fixture.tempDir, { recursive: true, force: true });
    });

    await fixture.nodeRepository.insert(
      buildProposition(createSemanticNodeId(), "Atlas is stable"),
    );
    const inserted = await fixture.nodeRepository.insert(
      buildProposition(createSemanticNodeId(), "Atlas is unstable"),
    );

    expect(llm.requests).toHaveLength(0);
    expect(enqueueReview).not.toHaveBeenCalled();

    await reviewService.reviewDuplicateCandidate(inserted);

    expect(llm.requests).toHaveLength(1);
    expect(enqueueReview).toHaveBeenCalledWith(
      expect.objectContaining({
        kind: "duplicate",
        refs: expect.objectContaining({
          node_ids: expect.arrayContaining([inserted.id]),
        }),
      }),
    );
  });

  it("marks LLM-flagged duplicate contradictions non-active and ranks corrected beliefs first", async () => {
    const llm = new FakeLLMClient({
      responses: [
        {
          text: "",
          input_tokens: 1,
          output_tokens: 1,
          stop_reason: "tool_use",
          tool_calls: [
            {
              id: "toolu_python_runtime_contradiction",
              name: CONTRADICTION_TOOL_NAME,
              input: {
                contradicts: true,
                confidence: 0.92,
                reason: "The later runtime claim revises the earlier one.",
              },
            },
          ],
        },
      ],
    });
    const harness = await createOfflineTestHarness({
      clock: new FixedClock(2_000),
      llmClient: llm,
    });
    const reviewService = new SemanticReviewService({
      nodeRepository: harness.semanticNodeRepository,
      enqueueReview: (input) => harness.reviewQueueRepository.enqueue(input),
      llmClient: llm,
      contradictionJudgeModel: "haiku",
    });

    cleanup.push(harness.cleanup);

    const episode = await harness.episodicRepository.createEpisode(
      createEpisodeFixture({
        title: "Python runtime correction",
        narrative: "The project runtime note was corrected from Python 3.11 to Python 3.12.",
        tags: ["python", "runtime"],
        created_at: 2_000,
        updated_at: 2_000,
      }),
    );
    const stale = await harness.semanticNodeRepository.insert(
      createSemanticNodeFixture(
        {
          label: "Project runtime is Python 3.11",
          description: "The project runtime is Python 3.11.",
          source_episode_ids: [episode.id],
          created_at: 1_000,
          updated_at: 1_000,
          last_verified_at: 1_000,
        },
        [1, 0, 0, 0],
      ),
    );
    const corrected = await harness.semanticNodeRepository.insert(
      createSemanticNodeFixture(
        {
          label: "Project runtime is Python 3.12",
          description: "The project runtime is Python 3.12.",
          source_episode_ids: [episode.id],
          created_at: 2_000,
          updated_at: 2_000,
          last_verified_at: 2_000,
        },
        [0.98, 0.2, 0, 0],
      ),
    );

    await reviewService.reviewDuplicateCandidate(corrected);

    const [review] = harness.reviewQueueRepository.list({
      kind: "duplicate",
      openOnly: true,
    });

    expect(llm.requests, "LLM contradiction judge should be called once").toHaveLength(1);
    expect(review, "LLM contradiction verdict should enqueue a duplicate review").toBeDefined();

    if (review === undefined) {
      throw new Error("Expected LLM-mediated duplicate review to be enqueued");
    }

    expect(review.refs).toMatchObject({
      node_ids: expect.arrayContaining([corrected.id, stale.id]),
    });

    await harness.reviewQueueRepository.resolve(review.id, {
      decision: "invalidate",
      winner_node_id: corrected.id,
    });

    const staleAfter = await harness.semanticNodeRepository.get(stale.id);

    expect(staleAfter, "stale semantic node should still be retrievable").not.toBeNull();
    expect(staleAfter).toMatchObject({
      id: stale.id,
      archived: false,
      status: "contradicted",
      corrected_by: corrected.id,
      superseded_at: 2_000,
    });
    expect(
      harness.semanticNodeRepository.countByStatus(),
      "non-archived lifecycle counts should include the contradicted stale node",
    ).toEqual({
      active: 1,
      superseded: 0,
      contradicted: 1,
      quarantined: 0,
    });

    const retrieval = await resolveSemanticContext(
      "Project runtime Python",
      {
        graphWalkDepth: 1,
        maxGraphNodes: 4,
        queryVector: Float32Array.from([1, 0, 0, 0]),
      },
      {
        embeddingClient: harness.embeddingClient,
        episodicRepository: harness.episodicRepository,
        semanticNodeRepository: harness.semanticNodeRepository,
        semanticGraph: harness.semanticGraph,
        reviewQueueRepository: harness.reviewQueueRepository,
      },
    );
    const correctedMatch = retrieval.matchedNodes.find((node) => node.id === corrected.id);
    const staleMatch = retrieval.matchedNodes.find((node) => node.id === stale.id);

    expect(correctedMatch, "corrected active node should be retrieved").toBeDefined();
    expect(staleMatch, "contradicted stale node should remain visible to retrieval").toBeDefined();

    if (correctedMatch === undefined || staleMatch === undefined) {
      throw new Error("Expected both active and contradicted semantic nodes in retrieval results");
    }

    expect(
      staleMatch.base_retrieval_score ?? 0,
      "fixture should give the stale node the stronger raw vector score",
    ).toBeGreaterThan(correctedMatch.base_retrieval_score ?? 0);
    expect(staleMatch.status).toBe("contradicted");
    expect(staleMatch.status_retrieval_multiplier).toBe(0.3);
    expect(staleMatch.retrieval_score).toBeCloseTo((staleMatch.base_retrieval_score ?? 0) * 0.3);
    expect(correctedMatch.status).toBe("active");
    expect(correctedMatch.retrieval_score).toBe(correctedMatch.base_retrieval_score);
    expect(
      retrieval.matchedNodeIds[0],
      "status multiplier should rank the active corrected node above the stronger stale vector match",
    ).toBe(corrected.id);
  });

  it("logs duplicate-review background failures without an unhandled rejection", async () => {
    const fixture = await createSemanticFixture();
    const writer = new StreamWriter({
      dataDir: fixture.tempDir,
      clock: fixture.clock,
    });
    const logged: Promise<void>[] = [];
    const reviewService = new SemanticReviewService({
      nodeRepository: fixture.nodeRepository,
      enqueueReview: vi.fn(),
      llmClient: new FakeLLMClient(),
      contradictionJudgeModel: "haiku",
      onDuplicateReviewError: (error) => {
        const promise = writer
          .append({
            kind: "internal_event",
            content: {
              hook: "semantic_duplicate_review",
              error: error instanceof Error ? error.message : String(error),
            },
          })
          .then(() => undefined);
        logged.push(promise);
        return promise;
      },
    });

    cleanup.push(async () => {
      writer.close();
      fixture.db.close();
      await fixture.store.close();
      rmSync(fixture.tempDir, { recursive: true, force: true });
    });

    vi.spyOn(fixture.nodeRepository, "searchByVector").mockRejectedValue(
      new Error("vector exploded"),
    );

    const inserted = await fixture.nodeRepository.insert(
      buildProposition(createSemanticNodeId(), "Atlas is unstable"),
    );

    reviewService.queueDuplicateReview(inserted);
    await new Promise((resolve) => {
      setImmediate(resolve);
    });
    await Promise.all(logged);

    const [entry] = new StreamReader({
      dataDir: fixture.tempDir,
    }).tail(1);

    expect(entry?.kind).toBe("internal_event");
    expect(entry?.content).toMatchObject({
      hook: "semantic_duplicate_review",
      error: "vector exploded",
    });
  });
});
