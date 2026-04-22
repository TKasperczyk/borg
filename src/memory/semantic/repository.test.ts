import { mkdtempSync, rmSync } from "node:fs";
import { join } from "node:path";
import { tmpdir } from "node:os";

import { afterEach, describe, expect, it, vi } from "vitest";

import { LanceDbStore } from "../../storage/lancedb/index.js";
import { openDatabase } from "../../storage/sqlite/index.js";
import { FixedClock } from "../../util/clock.js";
import {
  createSemanticEdgeId,
  createSemanticNodeId,
  type EpisodeId,
  type SemanticNodeId,
} from "../../util/ids.js";
import { semanticMigrations } from "./migrations.js";
import {
  SemanticEdgeRepository,
  SemanticNodeRepository,
  createSemanticNodesTableSchema,
} from "./repository.js";

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

  return {
    tempDir,
    store,
    db,
    table,
    clock,
    nodeRepository: new SemanticNodeRepository({
      table,
      db,
      clock,
    }),
    edgeRepository: new SemanticEdgeRepository({
      db,
      clock,
    }),
  };
}

function buildNode(id: SemanticNodeId, label: string) {
  return {
    id,
    kind: "concept" as const,
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

describe("semantic repositories", () => {
  const cleanup: Array<() => Promise<void>> = [];

  afterEach(async () => {
    vi.restoreAllMocks();

    while (cleanup.length > 0) {
      await cleanup.pop()?.();
    }
  });

  it("supports semantic node CRUD and vector search", async () => {
    const fixture = await createSemanticFixture();

    cleanup.push(async () => {
      fixture.db.close();
      await fixture.store.close();
      rmSync(fixture.tempDir, { recursive: true, force: true });
    });

    const inserted = await fixture.nodeRepository.insert(
      buildNode(createSemanticNodeId(), "Atlas deploy"),
    );
    const fetched = await fixture.nodeRepository.get(inserted.id);
    const searched = await fixture.nodeRepository.searchByVector(Float32Array.from([1, 0, 0, 0]), {
      limit: 1,
    });
    const updated = await fixture.nodeRepository.update(inserted.id, {
      aliases: ["deploy issue"],
      confidence: 0.8,
    });
    const listed = await fixture.nodeRepository.list();
    const deleted = await fixture.nodeRepository.delete(inserted.id);

    expect(fetched?.id).toBe(inserted.id);
    expect(searched[0]?.node.id).toBe(inserted.id);
    expect(updated?.aliases).toContain("deploy issue");
    expect(listed).toHaveLength(1);
    expect(deleted).toBe(true);
    expect(await fixture.nodeRepository.get(inserted.id)).toBeNull();
  });

  it("excludes archived nodes from getMany and label lookup by default", async () => {
    const fixture = await createSemanticFixture();

    cleanup.push(async () => {
      fixture.db.close();
      await fixture.store.close();
      rmSync(fixture.tempDir, { recursive: true, force: true });
    });

    const archived = await fixture.nodeRepository.insert({
      ...buildNode(createSemanticNodeId(), "Legacy Atlas"),
      aliases: ["atlas legacy"],
      archived: true,
    });
    const active = await fixture.nodeRepository.insert({
      ...buildNode(createSemanticNodeId(), "Legacy Atlas"),
      aliases: ["atlas legacy"],
      archived: false,
    });

    expect(await fixture.nodeRepository.getMany([archived.id, active.id])).toEqual([null, active]);
    expect(await fixture.nodeRepository.findByLabelOrAlias("Legacy Atlas", 3)).toEqual([
      expect.objectContaining({ id: active.id }),
    ]);
    expect(
      await fixture.nodeRepository.findByLabelOrAlias("atlas legacy", 3, {
        includeArchived: true,
      }),
    ).toEqual(
      expect.arrayContaining([
        expect.objectContaining({ id: active.id }),
        expect.objectContaining({ id: archived.id }),
      ]),
    );
  });

  it("rolls sqlite back when LanceDB insert fails and removes Lance data when sqlite insert fails", async () => {
    const fixture = await createSemanticFixture();

    cleanup.push(async () => {
      fixture.db.close();
      await fixture.store.close();
      rmSync(fixture.tempDir, { recursive: true, force: true });
    });

    const lanceFailure = new Error("lance failed");
    const tableSpy = vi.spyOn(fixture.table, "upsert").mockRejectedValueOnce(lanceFailure);

    await expect(
      fixture.nodeRepository.insert(buildNode(createSemanticNodeId(), "Lance failure")),
    ).rejects.toMatchObject({
      code: "SEMANTIC_NODE_INSERT_FAILED",
      cause: lanceFailure,
    });
    expect(
      (
        fixture.db.prepare("SELECT COUNT(*) AS count FROM semantic_nodes").get() as {
          count: number;
        }
      ).count,
    ).toBe(0);

    tableSpy.mockRestore();

    const repoSpy = vi
      .spyOn(
        fixture.nodeRepository as unknown as { upsertSqlRow(node: unknown): void },
        "upsertSqlRow",
      )
      .mockImplementationOnce(() => {
        throw new Error("sqlite failed");
      });
    const failingId = createSemanticNodeId();

    await expect(
      fixture.nodeRepository.insert(buildNode(failingId, "Sqlite failure")),
    ).rejects.toMatchObject({
      code: "SEMANTIC_NODE_INSERT_FAILED",
    });
    expect(await fixture.nodeRepository.get(failingId)).toBeNull();

    repoSpy.mockRestore();
  });

  it("enforces semantic edge constraints and supports relation/direction queries", async () => {
    const fixture = await createSemanticFixture();

    cleanup.push(async () => {
      fixture.db.close();
      await fixture.store.close();
      rmSync(fixture.tempDir, { recursive: true, force: true });
    });

    const atlas = await fixture.nodeRepository.insert(buildNode(createSemanticNodeId(), "Atlas"));
    const deploy = await fixture.nodeRepository.insert(buildNode(createSemanticNodeId(), "Deploy"));
    const rollback = await fixture.nodeRepository.insert(
      buildNode(createSemanticNodeId(), "Rollback"),
    );

    const support = fixture.edgeRepository.addEdge({
      id: createSemanticEdgeId(),
      from_node_id: atlas.id,
      to_node_id: deploy.id,
      relation: "supports",
      confidence: 0.7,
      evidence_episode_ids: ["ep_aaaaaaaaaaaaaaaa" as EpisodeId],
      created_at: 1_000,
      last_verified_at: 1_000,
    });
    fixture.edgeRepository.addEdge({
      id: createSemanticEdgeId(),
      from_node_id: rollback.id,
      to_node_id: deploy.id,
      relation: "prevents",
      confidence: 0.7,
      evidence_episode_ids: ["ep_aaaaaaaaaaaaaaaa" as EpisodeId],
      created_at: 1_000,
      last_verified_at: 1_000,
    });

    expect(
      fixture.edgeRepository.listEdges({
        fromId: atlas.id,
        relation: "supports",
      }),
    ).toEqual([support]);
    expect(
      fixture.edgeRepository.listEdges({
        toId: deploy.id,
      }),
    ).toHaveLength(2);
    expect(() =>
      fixture.edgeRepository.addEdge({
        from_node_id: atlas.id,
        to_node_id: atlas.id,
        relation: "related_to",
        confidence: 0.5,
        evidence_episode_ids: ["ep_aaaaaaaaaaaaaaaa" as EpisodeId],
        created_at: 1_000,
        last_verified_at: 1_000,
      }),
    ).toThrow();
    expect(() =>
      fixture.edgeRepository.addEdge({
        from_node_id: atlas.id,
        to_node_id: deploy.id,
        relation: "supports",
        confidence: 0.6,
        evidence_episode_ids: ["ep_aaaaaaaaaaaaaaaa" as EpisodeId],
        created_at: 1_000,
        last_verified_at: 1_000,
      }),
    ).toThrow();
  });

  it("enqueues direct contradiction reviews and rejects dangling endpoints", async () => {
    const fixture = await createSemanticFixture();
    const enqueueReview = vi.fn();
    const edgeRepository = new SemanticEdgeRepository({
      db: fixture.db,
      clock: fixture.clock,
      enqueueReview,
    });

    cleanup.push(async () => {
      fixture.db.close();
      await fixture.store.close();
      rmSync(fixture.tempDir, { recursive: true, force: true });
    });

    const atlas = await fixture.nodeRepository.insert(buildNode(createSemanticNodeId(), "Atlas"));
    const rollback = await fixture.nodeRepository.insert(
      buildNode(createSemanticNodeId(), "Rollback"),
    );

    edgeRepository.addEdge({
      from_node_id: atlas.id,
      to_node_id: rollback.id,
      relation: "contradicts",
      confidence: 0.6,
      evidence_episode_ids: ["ep_aaaaaaaaaaaaaaaa" as EpisodeId],
      created_at: 1_000,
      last_verified_at: 1_000,
    });

    expect(enqueueReview).toHaveBeenCalledWith(
      expect.objectContaining({
        kind: "contradiction",
        refs: expect.objectContaining({
          node_ids: [atlas.id, rollback.id],
        }),
      }),
    );

    let danglingError: unknown;

    try {
      edgeRepository.addEdge({
        from_node_id: atlas.id,
        to_node_id: createSemanticNodeId(),
        relation: "supports",
        confidence: 0.7,
        evidence_episode_ids: ["ep_aaaaaaaaaaaaaaaa" as EpisodeId],
        created_at: 1_000,
        last_verified_at: 1_000,
      });
    } catch (error) {
      danglingError = error;
    }

    expect(danglingError).toMatchObject({
      code: "SEMANTIC_EDGE_DANGLING",
    });
  });
});
