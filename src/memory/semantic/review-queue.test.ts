import { mkdtempSync, rmSync } from "node:fs";
import { join } from "node:path";
import { tmpdir } from "node:os";

import { afterEach, describe, expect, it } from "vitest";

import { LanceDbStore } from "../../storage/lancedb/index.js";
import { openDatabase } from "../../storage/sqlite/index.js";
import { FixedClock } from "../../util/clock.js";
import { createSemanticNodeId, type EpisodeId } from "../../util/ids.js";
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
    const reviewQueue = new ReviewQueueRepository({
      db,
      clock,
      semanticNodeRepository: nodeRepository,
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
    expect(openItems).toHaveLength(1);
    expect(openItems[0]?.kind).toBe("contradiction");
    expect(openItems[0]?.reason).toContain("conflicts_with_support_chain");

    const resolved = await reviewQueue.resolve(openItems[0]!.id, "invalidate");
    const updatedSecond = await nodeRepository.get(second.id);

    expect(resolved?.resolution).toBe("invalidate");
    expect(updatedSecond?.archived).toBe(true);
    expect(updatedSecond?.confidence).toBe(0);
  });
});
