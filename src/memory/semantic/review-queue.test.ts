import { mkdtempSync, rmSync } from "node:fs";
import { join } from "node:path";
import { tmpdir } from "node:os";

import { afterEach, describe, expect, it } from "vitest";

import { LanceDbStore } from "../../storage/lancedb/index.js";
import { openDatabase } from "../../storage/sqlite/index.js";
import { StreamReader, StreamWriter } from "../../stream/index.js";
import { FixedClock } from "../../util/clock.js";
import { createSemanticNodeId, type EpisodeId } from "../../util/ids.js";
import { OpenQuestionsRepository, selfMigrations } from "../self/index.js";
import { enqueueOpenQuestionForReview } from "../self/review-open-question-hook.js";
import { semanticMigrations } from "./migrations.js";
import { ReviewQueueRepository } from "./review-queue.js";
import {
  SemanticEdgeRepository,
  SemanticNodeRepository,
  createSemanticNodesTableSchema,
} from "./repository.js";

describe("review queue", () => {
  const cleanup: Array<() => Promise<void>> = [];

  afterEach(async () => {
    while (cleanup.length > 0) {
      await cleanup.pop()?.();
    }
  });

  it("enqueues contradiction reviews on conflicting support paths and resolves them", async () => {
    const tempDir = mkdtempSync(join(tmpdir(), "borg-"));
    const store = new LanceDbStore({
      uri: join(tempDir, "lancedb"),
    });
    const db = openDatabase(join(tempDir, "borg.db"), {
      migrations: [...semanticMigrations, ...selfMigrations],
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
    const openQuestionsRepository = new OpenQuestionsRepository({
      db,
      clock,
    });
    const reviewQueue = new ReviewQueueRepository({
      db,
      clock,
      semanticNodeRepository: nodeRepository,
      onEnqueue: (item) => enqueueOpenQuestionForReview(openQuestionsRepository, item),
    });
    const edgeRepository = new SemanticEdgeRepository({
      db,
      clock,
      enqueueReview: (input) => reviewQueue.enqueue(input),
    });

    cleanup.push(async () => {
      db.close();
      await store.close();
      rmSync(tempDir, { recursive: true, force: true });
    });

    const episodeIds = ["ep_aaaaaaaaaaaaaaaa" as EpisodeId];
    const first = await nodeRepository.insert({
      id: createSemanticNodeId(),
      kind: "proposition",
      label: "Atlas succeeds",
      description: "Atlas succeeds",
      aliases: [],
      confidence: 0.8,
      source_episode_ids: episodeIds,
      created_at: 1,
      updated_at: 1,
      last_verified_at: 1,
      embedding: Float32Array.from([1, 0, 0, 0]),
      archived: false,
      superseded_by: null,
    });
    const second = await nodeRepository.insert({
      id: createSemanticNodeId(),
      kind: "proposition",
      label: "Atlas fails",
      description: "Atlas does not succeed",
      aliases: [],
      confidence: 0.6,
      source_episode_ids: episodeIds,
      created_at: 1,
      updated_at: 1,
      last_verified_at: 1,
      embedding: Float32Array.from([1, 0, 0, 0]),
      archived: false,
      superseded_by: null,
    });

    edgeRepository.addEdge({
      from_node_id: first.id,
      to_node_id: second.id,
      relation: "supports",
      confidence: 0.7,
      evidence_episode_ids: episodeIds,
      created_at: 1_000,
      last_verified_at: 1_000,
    });
    edgeRepository.addEdge({
      from_node_id: first.id,
      to_node_id: second.id,
      relation: "contradicts",
      confidence: 0.7,
      evidence_episode_ids: episodeIds,
      created_at: 1_000,
      last_verified_at: 1_000,
    });

    const openItems = reviewQueue.getOpen();
    const openQuestions = openQuestionsRepository.list({ status: "open" });

    expect(openItems).toHaveLength(1);
    expect(openItems[0]?.kind).toBe("contradiction");
    expect(openItems[0]?.reason).toContain("conflicts_with_support_chain");
    expect(openQuestions[0]?.source).toBe("contradiction");
    expect(openQuestions[0]?.related_semantic_node_ids).toEqual([first.id, second.id]);

    const resolved = await reviewQueue.resolve(openItems[0]!.id, "invalidate");
    const updatedSecond = await nodeRepository.get(second.id);

    expect(resolved?.resolution).toBe("invalidate");
    expect(updatedSecond?.archived).toBe(true);
    expect(updatedSecond?.confidence).toBe(0);
  });

  it("logs hook failures without aborting edge insertion", async () => {
    const tempDir = mkdtempSync(join(tmpdir(), "borg-"));
    const store = new LanceDbStore({
      uri: join(tempDir, "lancedb"),
    });
    const db = openDatabase(join(tempDir, "borg.db"), {
      migrations: [...semanticMigrations, ...selfMigrations],
    });
    const table = await store.openTable({
      name: "semantic_nodes",
      schema: createSemanticNodesTableSchema(4),
    });
    const clock = new FixedClock(1_000);
    const writer = new StreamWriter({
      dataDir: tempDir,
      clock,
    });
    const nodeRepository = new SemanticNodeRepository({
      table,
      db,
      clock,
    });
    const reviewQueue = new ReviewQueueRepository({
      db,
      clock,
      semanticNodeRepository: nodeRepository,
      onEnqueue() {
        throw new Error("hook exploded");
      },
      onEnqueueError: (error) => {
        void writer.append({
          kind: "internal_event",
          content: {
            hook: "review_queue_open_question",
            error: error instanceof Error ? error.message : String(error),
          },
        });
      },
    });
    const edgeRepository = new SemanticEdgeRepository({
      db,
      clock,
      enqueueReview: (input) => reviewQueue.enqueue(input),
    });

    cleanup.push(async () => {
      writer.close();
      db.close();
      await store.close();
      rmSync(tempDir, { recursive: true, force: true });
    });

    const episodeIds = ["ep_cccccccccccccccc" as EpisodeId];
    const first = await nodeRepository.insert({
      id: createSemanticNodeId(),
      kind: "proposition",
      label: "Atlas works",
      description: "Atlas works",
      aliases: [],
      confidence: 0.8,
      source_episode_ids: episodeIds,
      created_at: 1,
      updated_at: 1,
      last_verified_at: 1,
      embedding: Float32Array.from([1, 0, 0, 0]),
      archived: false,
      superseded_by: null,
    });
    const second = await nodeRepository.insert({
      id: createSemanticNodeId(),
      kind: "proposition",
      label: "Atlas breaks",
      description: "Atlas breaks",
      aliases: [],
      confidence: 0.6,
      source_episode_ids: episodeIds,
      created_at: 1,
      updated_at: 1,
      last_verified_at: 1,
      embedding: Float32Array.from([1, 0, 0, 0]),
      archived: false,
      superseded_by: null,
    });

    const edge = edgeRepository.addEdge({
      from_node_id: first.id,
      to_node_id: second.id,
      relation: "contradicts",
      confidence: 0.7,
      evidence_episode_ids: episodeIds,
      created_at: 1_000,
      last_verified_at: 1_000,
    });

    await new Promise((resolve) => {
      setTimeout(resolve, 25);
    });

    const entries = new StreamReader({
      dataDir: tempDir,
    }).tail(1);

    expect(edge.relation).toBe("contradicts");
    expect(reviewQueue.getOpen()).toHaveLength(1);
    expect(entries[0]).toMatchObject({
      kind: "internal_event",
      content: {
        hook: "review_queue_open_question",
      },
    });
  });

  it("rejects incompatible review decisions and allows valid pairings", async () => {
    const db = openDatabase(":memory:", {
      migrations: [...semanticMigrations],
    });
    const reviewQueue = new ReviewQueueRepository({
      db,
      clock: new FixedClock(1_000),
    });

    try {
      const correction = reviewQueue.enqueue({
        kind: "correction",
        refs: {
          record_id: "val_aaaaaaaaaaaaaaaa",
          patch: {
            description: "Prefer grounded claims.",
          },
        },
        reason: "user corrected the record",
      });
      const contradiction = reviewQueue.enqueue({
        kind: "contradiction",
        refs: {
          node_ids: ["semn_aaaaaaaaaaaaaaaa", "semn_bbbbbbbbbbbbbbbb"],
        },
        reason: "conflicting support paths",
      });
      const stale = reviewQueue.enqueue({
        kind: "stale",
        refs: {
          node_id: "semn_cccccccccccccccc",
        },
        reason: "needs refresh",
      });

      await expect(reviewQueue.resolve(correction.id, "keep_both")).rejects.toMatchObject({
        name: "SemanticError",
        code: "REVIEW_QUEUE_RESOLUTION_INVALID",
      });
      await expect(reviewQueue.resolve(contradiction.id, "accept")).rejects.toMatchObject({
        name: "SemanticError",
        code: "REVIEW_QUEUE_RESOLUTION_INVALID",
      });

      const rejectedCorrection = await reviewQueue.resolve(correction.id, "reject");
      const dismissedContradiction = await reviewQueue.resolve(contradiction.id, "dismiss");
      const acceptedStale = await reviewQueue.resolve(stale.id, "accept");

      expect(rejectedCorrection?.resolution).toBe("reject");
      expect(dismissedContradiction?.resolution).toBe("dismiss");
      expect(acceptedStale?.resolution).toBe("accept");
    } finally {
      db.close();
    }
  });
});
