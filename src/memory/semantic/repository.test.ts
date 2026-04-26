import { mkdtempSync, rmSync } from "node:fs";
import { join } from "node:path";
import { tmpdir } from "node:os";

import { afterEach, describe, expect, it, vi } from "vitest";

import { FakeLLMClient } from "../../llm/index.js";
import { LanceDbStore } from "../../storage/lancedb/index.js";
import { openDatabase } from "../../storage/sqlite/index.js";
import { StreamReader, StreamWriter } from "../../stream/index.js";
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
import type { SemanticRelation } from "./types.js";

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

type SemanticFixture = Awaited<ReturnType<typeof createSemanticFixture>>;

function buildEdgeInput(
  fromId: SemanticNodeId,
  toId: SemanticNodeId,
  relation: SemanticRelation = "supports",
) {
  return {
    from_node_id: fromId,
    to_node_id: toId,
    relation,
    confidence: 0.7,
    evidence_episode_ids: ["ep_aaaaaaaaaaaaaaaa" as EpisodeId],
    created_at: 500,
    last_verified_at: 500,
  };
}

async function insertEdgeEndpoints(fixture: SemanticFixture) {
  const atlas = await fixture.nodeRepository.insert(buildNode(createSemanticNodeId(), "Atlas"));
  const deploy = await fixture.nodeRepository.insert(buildNode(createSemanticNodeId(), "Deploy"));

  return { atlas, deploy };
}

describe("semantic repositories", () => {
  const cleanup: Array<() => Promise<void>> = [];

  afterEach(async () => {
    vi.restoreAllMocks();

    while (cleanup.length > 0) {
      await cleanup.pop()?.();
    }
  });

  it("migrates an empty semantic edge table with validity columns and indexes", () => {
    const tempDir = mkdtempSync(join(tmpdir(), "borg-"));
    let db: ReturnType<typeof openDatabase> | null = null;

    cleanup.push(async () => {
      db?.close();
      rmSync(tempDir, { recursive: true, force: true });
    });

    db = openDatabase(join(tempDir, "borg.db"), {
      migrations: semanticMigrations,
    });

    const columns = db.prepare("PRAGMA table_info(semantic_edges)").all() as Array<{
      name: string;
      notnull: number;
    }>;
    const columnNames = new Set(columns.map((column) => column.name));
    const validFrom = columns.find((column) => column.name === "valid_from");
    const indexes = db.prepare("PRAGMA index_list(semantic_edges)").all() as Array<{
      name: string;
      unique: number;
      partial: number;
    }>;

    expect(columnNames.has("valid_from")).toBe(true);
    expect(columnNames.has("valid_to")).toBe(true);
    expect(columnNames.has("invalidated_at")).toBe(true);
    expect(columnNames.has("invalidated_by_edge_id")).toBe(true);
    expect(columnNames.has("invalidated_by_review_id")).toBe(true);
    expect(columnNames.has("invalidated_by_process")).toBe(true);
    expect(columnNames.has("invalidated_reason")).toBe(true);
    expect(validFrom?.notnull).toBe(1);
    expect(indexes).toEqual(
      expect.arrayContaining([
        expect.objectContaining({
          name: "semantic_edges_open_unique_idx",
          unique: 1,
          partial: 1,
        }),
        expect.objectContaining({
          name: "semantic_edges_from_relation_validity_idx",
        }),
        expect.objectContaining({
          name: "semantic_edges_to_relation_validity_idx",
        }),
      ]),
    );
  });

  it("backfills pre-migration semantic edges with created_at as valid_from", () => {
    const tempDir = mkdtempSync(join(tmpdir(), "borg-"));
    const dbPath = join(tempDir, "borg.db");
    let db: ReturnType<typeof openDatabase> | null = null;

    cleanup.push(async () => {
      db?.close();
      rmSync(tempDir, { recursive: true, force: true });
    });

    db = openDatabase(dbPath, {
      migrations: semanticMigrations.filter((migration) => migration.id < 133),
    });
    const edgeId = createSemanticEdgeId();

    db.prepare(
      `
        INSERT INTO semantic_edges (
          id, from_node_id, to_node_id, relation, confidence, evidence_episode_ids, created_at, last_verified_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
      `,
    ).run(
      edgeId,
      createSemanticNodeId(),
      createSemanticNodeId(),
      "supports",
      0.7,
      JSON.stringify(["ep_aaaaaaaaaaaaaaaa"]),
      250,
      300,
    );
    db.close();
    db = null;

    db = openDatabase(dbPath, {
      migrations: semanticMigrations,
    });

    const row = db
      .prepare("SELECT valid_from, valid_to FROM semantic_edges WHERE id = ?")
      .get(edgeId) as { valid_from: number; valid_to: number | null };

    expect(row).toEqual({
      valid_from: 250,
      valid_to: null,
    });
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

  it("canonicalizes the final stored domain on update even when the patch omits domain", async () => {
    const fixture = await createSemanticFixture();

    cleanup.push(async () => {
      fixture.db.close();
      await fixture.store.close();
      rmSync(fixture.tempDir, { recursive: true, force: true });
    });

    const inserted = await fixture.nodeRepository.insert({
      ...buildNode(createSemanticNodeId(), "Atlas deploy"),
      domain: "tech",
    });

    await fixture.table.upsert(
      [
        {
          id: inserted.id,
          kind: inserted.kind,
          label: inserted.label,
          description: inserted.description,
          domain: "technology",
          aliases: JSON.stringify(inserted.aliases),
          confidence: inserted.confidence,
          source_episode_ids: JSON.stringify(inserted.source_episode_ids),
          created_at: inserted.created_at,
          updated_at: inserted.updated_at,
          last_verified_at: inserted.last_verified_at,
          embedding: Array.from(inserted.embedding),
          archived: inserted.archived ? 1 : 0,
          superseded_by: inserted.superseded_by,
        },
      ],
      {
        on: "id",
      },
    );
    fixture.db
      .prepare("UPDATE semantic_nodes SET domain = ? WHERE id = ?")
      .run("technology", inserted.id);

    const updated = await fixture.nodeRepository.update(inserted.id, {
      aliases: ["deploy issue"],
    });

    expect(updated).toMatchObject({
      id: inserted.id,
      domain: "tech",
    });
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

  it("writes semantic edge validity defaults on insert", async () => {
    const fixture = await createSemanticFixture();

    cleanup.push(async () => {
      fixture.db.close();
      await fixture.store.close();
      rmSync(fixture.tempDir, { recursive: true, force: true });
    });

    const { atlas, deploy } = await insertEdgeEndpoints(fixture);
    const edge = fixture.edgeRepository.addEdge(buildEdgeInput(atlas.id, deploy.id));

    expect(edge).toMatchObject({
      valid_from: 1_000,
      valid_to: null,
      invalidated_at: null,
      invalidated_by_edge_id: null,
      invalidated_by_review_id: null,
      invalidated_by_process: null,
      invalidated_reason: null,
    });
    expect(fixture.edgeRepository.getEdge(edge.id)).toEqual(edge);
  });

  it("rejects currently open duplicate semantic edges", async () => {
    const fixture = await createSemanticFixture();

    cleanup.push(async () => {
      fixture.db.close();
      await fixture.store.close();
      rmSync(fixture.tempDir, { recursive: true, force: true });
    });

    const { atlas, deploy } = await insertEdgeEndpoints(fixture);
    fixture.edgeRepository.addEdge(buildEdgeInput(atlas.id, deploy.id));

    expect(() => fixture.edgeRepository.addEdge(buildEdgeInput(atlas.id, deploy.id))).toThrow(
      "Duplicate semantic edge",
    );
  });

  it("allows historical duplicates after closing the previous semantic edge", async () => {
    const fixture = await createSemanticFixture();

    cleanup.push(async () => {
      fixture.db.close();
      await fixture.store.close();
      rmSync(fixture.tempDir, { recursive: true, force: true });
    });

    const { atlas, deploy } = await insertEdgeEndpoints(fixture);
    const closed = fixture.edgeRepository.addEdge(buildEdgeInput(atlas.id, deploy.id));

    fixture.edgeRepository.invalidateEdge(closed.id, {
      at: 1_000,
      by_process: "manual",
      reason: "historical duplicate test",
    });
    const replacement = fixture.edgeRepository.addEdge(buildEdgeInput(atlas.id, deploy.id));

    expect(fixture.edgeRepository.listEdges()).toEqual([replacement]);
    expect(fixture.edgeRepository.listEdges({ includeInvalid: true })).toHaveLength(2);
  });

  it("hides closed semantic edges by default", async () => {
    const fixture = await createSemanticFixture();

    cleanup.push(async () => {
      fixture.db.close();
      await fixture.store.close();
      rmSync(fixture.tempDir, { recursive: true, force: true });
    });

    const { atlas, deploy } = await insertEdgeEndpoints(fixture);
    const closed = fixture.edgeRepository.addEdge(buildEdgeInput(atlas.id, deploy.id));

    fixture.edgeRepository.invalidateEdge(closed.id, {
      at: 1_000,
      by_process: "manual",
    });
    const open = fixture.edgeRepository.addEdge(buildEdgeInput(atlas.id, deploy.id));

    expect(fixture.edgeRepository.listEdges()).toEqual([open]);
  });

  it("lists semantic edges valid at an asOf timestamp", async () => {
    const fixture = await createSemanticFixture();

    cleanup.push(async () => {
      fixture.db.close();
      await fixture.store.close();
      rmSync(fixture.tempDir, { recursive: true, force: true });
    });

    const { atlas, deploy } = await insertEdgeEndpoints(fixture);
    const oldEdge = fixture.edgeRepository.addEdge({
      ...buildEdgeInput(atlas.id, deploy.id),
      created_at: 1_000,
      last_verified_at: 1_000,
      valid_from: 1_000,
    });

    fixture.edgeRepository.invalidateEdge(oldEdge.id, {
      at: 1_500,
      by_process: "manual",
    });
    const newEdge = fixture.edgeRepository.addEdge({
      ...buildEdgeInput(atlas.id, deploy.id),
      created_at: 1_600,
      last_verified_at: 1_600,
      valid_from: 1_600,
    });

    expect(fixture.edgeRepository.listEdges({ asOf: 1_250 }).map((edge) => edge.id)).toEqual([
      oldEdge.id,
    ]);
    expect(fixture.edgeRepository.listEdges({ asOf: 1_550 })).toEqual([]);
    expect(fixture.edgeRepository.listEdges({ asOf: 1_700 }).map((edge) => edge.id)).toEqual([
      newEdge.id,
    ]);
  });

  it("can include invalid semantic edges in list results", async () => {
    const fixture = await createSemanticFixture();

    cleanup.push(async () => {
      fixture.db.close();
      await fixture.store.close();
      rmSync(fixture.tempDir, { recursive: true, force: true });
    });

    const { atlas, deploy } = await insertEdgeEndpoints(fixture);
    const closed = fixture.edgeRepository.addEdge(buildEdgeInput(atlas.id, deploy.id));

    fixture.edgeRepository.invalidateEdge(closed.id, {
      at: 1_500,
      by_process: "manual",
    });
    const open = fixture.edgeRepository.addEdge(buildEdgeInput(atlas.id, deploy.id));

    expect(
      fixture.edgeRepository.listEdges({ includeInvalid: true }).map((edge) => edge.id),
    ).toEqual(expect.arrayContaining([closed.id, open.id]));
  });

  it("invalidates semantic edges with provenance without changing confidence or evidence", async () => {
    const fixture = await createSemanticFixture();

    cleanup.push(async () => {
      fixture.db.close();
      await fixture.store.close();
      rmSync(fixture.tempDir, { recursive: true, force: true });
    });

    const { atlas, deploy } = await insertEdgeEndpoints(fixture);
    const edge = fixture.edgeRepository.addEdge({
      ...buildEdgeInput(atlas.id, deploy.id),
      confidence: 0.73,
      evidence_episode_ids: [
        "ep_aaaaaaaaaaaaaaaa" as EpisodeId,
        "ep_bbbbbbbbbbbbbbbb" as EpisodeId,
      ],
    });
    const invalidatingEdgeId = createSemanticEdgeId();
    const invalidated = fixture.edgeRepository.invalidateEdge(edge.id, {
      at: 1_200,
      by_edge_id: invalidatingEdgeId,
      by_review_id: 42,
      by_process: "manual",
      reason: "superseded by newer evidence",
    });

    expect(invalidated).toMatchObject({
      id: edge.id,
      confidence: 0.73,
      evidence_episode_ids: edge.evidence_episode_ids,
      valid_to: 1_200,
      invalidated_at: 1_000,
      invalidated_by_edge_id: invalidatingEdgeId,
      invalidated_by_review_id: 42,
      invalidated_by_process: "manual",
      invalidated_reason: "superseded by newer evidence",
    });

    const afterSecondCall = fixture.edgeRepository.invalidateEdge(edge.id, {
      at: 1_300,
      by_process: "review",
      reason: "different reason",
    });

    expect(afterSecondCall).toEqual(invalidated);
    expect(fixture.edgeRepository.getEdge(edge.id)).toEqual(invalidated);
  });

  it("rejects invalidating an edge with 'at' before its valid_from", async () => {
    const fixture = await createSemanticFixture();

    cleanup.push(async () => {
      fixture.db.close();
      await fixture.store.close();
      rmSync(fixture.tempDir, { recursive: true, force: true });
    });

    const { atlas, deploy } = await insertEdgeEndpoints(fixture);
    const edge = fixture.edgeRepository.addEdge(buildEdgeInput(atlas.id, deploy.id));

    expect(() =>
      fixture.edgeRepository.invalidateEdge(edge.id, {
        at: edge.valid_from - 1,
        by_process: "manual",
      }),
    ).toThrow(/SEMANTIC_EDGE_INVALIDATE_BEFORE_VALID_FROM|precedes valid_from/);

    expect(fixture.edgeRepository.getEdge(edge.id)?.valid_to).toBeNull();
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

  it("logs duplicate-review background failures without an unhandled rejection", async () => {
    const fixture = await createSemanticFixture();
    const writer = new StreamWriter({
      dataDir: fixture.tempDir,
      clock: fixture.clock,
    });
    const logged: Promise<void>[] = [];
    const nodeRepository = new SemanticNodeRepository({
      table: fixture.table,
      db: fixture.db,
      clock: fixture.clock,
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

    vi.spyOn(nodeRepository, "searchByVector").mockRejectedValue(new Error("vector exploded"));

    await nodeRepository.insert({
      ...buildNode(createSemanticNodeId(), "Atlas is unstable"),
      kind: "proposition",
    });
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
