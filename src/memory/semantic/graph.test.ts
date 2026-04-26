import { mkdtempSync, rmSync } from "node:fs";
import { join } from "node:path";
import { tmpdir } from "node:os";

import { afterEach, describe, expect, it } from "vitest";

import { LanceDbStore } from "../../storage/lancedb/index.js";
import { openDatabase } from "../../storage/sqlite/index.js";
import { FixedClock, ManualClock } from "../../util/clock.js";
import { createSemanticNodeId, type EpisodeId } from "../../util/ids.js";
import { SemanticGraph } from "./graph.js";
import { semanticMigrations } from "./migrations.js";
import {
  SemanticEdgeRepository,
  SemanticNodeRepository,
  createSemanticNodesTableSchema,
} from "./repository.js";

describe("semantic graph", () => {
  const cleanup: Array<() => Promise<void>> = [];

  afterEach(async () => {
    while (cleanup.length > 0) {
      await cleanup.pop()?.();
    }
  });

  it("walks with depth and max-node limits without looping on cycles", async () => {
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
    const edgeRepository = new SemanticEdgeRepository({
      db,
      clock,
    });
    const graph = new SemanticGraph({
      nodeRepository,
      edgeRepository,
    });

    cleanup.push(async () => {
      db.close();
      await store.close();
      rmSync(tempDir, { recursive: true, force: true });
    });

    const episodeIds = ["ep_aaaaaaaaaaaaaaaa" as EpisodeId];
    const alpha = await nodeRepository.insert({
      id: createSemanticNodeId(),
      kind: "concept",
      label: "Alpha",
      description: "Alpha concept",
      aliases: [],
      confidence: 0.7,
      source_episode_ids: episodeIds,
      created_at: 1_000,
      updated_at: 1_000,
      last_verified_at: 1_000,
      embedding: Float32Array.from([1, 0, 0, 0]),
      archived: false,
      superseded_by: null,
    });
    const beta = await nodeRepository.insert({
      id: createSemanticNodeId(),
      kind: "concept",
      label: "Beta",
      description: "Beta concept",
      aliases: [],
      confidence: 0.7,
      source_episode_ids: episodeIds,
      created_at: 1_000,
      updated_at: 1_000,
      last_verified_at: 1_000,
      embedding: Float32Array.from([1, 0, 0, 0]),
      archived: false,
      superseded_by: null,
    });
    const gamma = await nodeRepository.insert({
      id: createSemanticNodeId(),
      kind: "concept",
      label: "Gamma",
      description: "Gamma concept",
      aliases: [],
      confidence: 0.7,
      source_episode_ids: episodeIds,
      created_at: 1_000,
      updated_at: 1_000,
      last_verified_at: 1_000,
      embedding: Float32Array.from([1, 0, 0, 0]),
      archived: false,
      superseded_by: null,
    });

    edgeRepository.addEdge({
      from_node_id: alpha.id,
      to_node_id: beta.id,
      relation: "supports",
      confidence: 0.7,
      evidence_episode_ids: episodeIds,
      created_at: 1_000,
      last_verified_at: 1_000,
    });
    edgeRepository.addEdge({
      from_node_id: beta.id,
      to_node_id: gamma.id,
      relation: "supports",
      confidence: 0.7,
      evidence_episode_ids: episodeIds,
      created_at: 1_000,
      last_verified_at: 1_000,
    });
    edgeRepository.addEdge({
      from_node_id: gamma.id,
      to_node_id: alpha.id,
      relation: "supports",
      confidence: 0.7,
      evidence_episode_ids: episodeIds,
      created_at: 1_000,
      last_verified_at: 1_000,
    });

    const depthOne = await graph.walk(alpha.id, {
      relations: ["supports"],
      depth: 1,
      maxNodes: 8,
    });
    const bounded = await graph.walk(alpha.id, {
      relations: ["supports"],
      depth: 3,
      maxNodes: 2,
    });

    expect(depthOne.map((step) => step.node.id)).toEqual([beta.id]);
    expect(bounded).toHaveLength(2);
    expect(await graph.supportsFor(alpha.id)).toEqual(
      expect.arrayContaining([expect.objectContaining({ id: beta.id })]),
    );
  });

  it("treats is_a as outbound by default and excludes archived neighbors", async () => {
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
    const edgeRepository = new SemanticEdgeRepository({
      db,
      clock,
    });
    const graph = new SemanticGraph({
      nodeRepository,
      edgeRepository,
    });

    cleanup.push(async () => {
      db.close();
      await store.close();
      rmSync(tempDir, { recursive: true, force: true });
    });

    const episodeIds = ["ep_aaaaaaaaaaaaaaaa" as EpisodeId];
    const atlas = await nodeRepository.insert({
      ...{
        id: createSemanticNodeId(),
        kind: "entity" as const,
        label: "Atlas",
        description: "Atlas entity",
        aliases: [],
        confidence: 0.8,
        source_episode_ids: episodeIds,
        created_at: 1_000,
        updated_at: 1_000,
        last_verified_at: 1_000,
        embedding: Float32Array.from([1, 0, 0, 0]),
        archived: false,
        superseded_by: null,
      },
    });
    const service = await nodeRepository.insert({
      ...{
        id: createSemanticNodeId(),
        kind: "concept" as const,
        label: "Service",
        description: "Service category",
        aliases: [],
        confidence: 0.8,
        source_episode_ids: episodeIds,
        created_at: 1_000,
        updated_at: 1_000,
        last_verified_at: 1_000,
        embedding: Float32Array.from([1, 0, 0, 0]),
        archived: false,
        superseded_by: null,
      },
    });
    const instance = await nodeRepository.insert({
      ...{
        id: createSemanticNodeId(),
        kind: "entity" as const,
        label: "Atlas Instance",
        description: "Inbound instance node",
        aliases: [],
        confidence: 0.8,
        source_episode_ids: episodeIds,
        created_at: 1_000,
        updated_at: 1_000,
        last_verified_at: 1_000,
        embedding: Float32Array.from([1, 0, 0, 0]),
        archived: false,
        superseded_by: null,
      },
    });
    const archivedCategory = await nodeRepository.insert({
      ...{
        id: createSemanticNodeId(),
        kind: "concept" as const,
        label: "Archived Category",
        description: "Archived category node",
        aliases: [],
        confidence: 0.2,
        source_episode_ids: episodeIds,
        created_at: 1_000,
        updated_at: 1_000,
        last_verified_at: 1_000,
        embedding: Float32Array.from([1, 0, 0, 0]),
        archived: true,
        superseded_by: null,
      },
    });

    edgeRepository.addEdge({
      from_node_id: atlas.id,
      to_node_id: service.id,
      relation: "is_a",
      confidence: 0.8,
      evidence_episode_ids: episodeIds,
      created_at: 1_000,
      last_verified_at: 1_000,
    });
    edgeRepository.addEdge({
      from_node_id: instance.id,
      to_node_id: atlas.id,
      relation: "is_a",
      confidence: 0.8,
      evidence_episode_ids: episodeIds,
      created_at: 1_000,
      last_verified_at: 1_000,
    });
    edgeRepository.addEdge({
      from_node_id: atlas.id,
      to_node_id: archivedCategory.id,
      relation: "is_a",
      confidence: 0.5,
      evidence_episode_ids: episodeIds,
      created_at: 1_000,
      last_verified_at: 1_000,
    });

    const walked = await graph.walk(atlas.id, {
      relations: ["is_a"],
      depth: 1,
      maxNodes: 8,
    });

    expect(walked.map((step) => step.node.id)).toEqual([service.id]);
  });

  it("filters closed edges by default and can walk them as-of or with invalid edges included", async () => {
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
    const clock = new ManualClock(1_000);
    const nodeRepository = new SemanticNodeRepository({
      table,
      db,
      clock,
    });
    const edgeRepository = new SemanticEdgeRepository({
      db,
      clock,
    });
    const graph = new SemanticGraph({
      nodeRepository,
      edgeRepository,
    });

    cleanup.push(async () => {
      db.close();
      await store.close();
      rmSync(tempDir, { recursive: true, force: true });
    });

    const episodeIds = ["ep_aaaaaaaaaaaaaaaa" as EpisodeId];
    const alpha = await nodeRepository.insert({
      id: createSemanticNodeId(),
      kind: "concept",
      label: "Alpha",
      description: "Alpha concept",
      aliases: [],
      confidence: 0.7,
      source_episode_ids: episodeIds,
      created_at: 1_000,
      updated_at: 1_000,
      last_verified_at: 1_000,
      embedding: Float32Array.from([1, 0, 0, 0]),
      archived: false,
      superseded_by: null,
    });
    const beta = await nodeRepository.insert({
      id: createSemanticNodeId(),
      kind: "concept",
      label: "Beta",
      description: "Beta concept",
      aliases: [],
      confidence: 0.7,
      source_episode_ids: episodeIds,
      created_at: 1_000,
      updated_at: 1_000,
      last_verified_at: 1_000,
      embedding: Float32Array.from([1, 0, 0, 0]),
      archived: false,
      superseded_by: null,
    });

    const edge = edgeRepository.addEdge({
      from_node_id: alpha.id,
      to_node_id: beta.id,
      relation: "supports",
      confidence: 0.7,
      evidence_episode_ids: episodeIds,
      created_at: 1_000,
      last_verified_at: 1_000,
    });
    edgeRepository.invalidateEdge(edge.id, {
      at: 1_500,
      by_process: "manual",
    });
    clock.set(2_000);

    const current = await graph.walk(alpha.id, {
      relations: ["supports"],
      depth: 1,
    });
    const historical = await graph.walk(alpha.id, {
      relations: ["supports"],
      depth: 1,
      asOf: 1_250,
    });
    const invalidIncluded = await graph.walk(alpha.id, {
      relations: ["supports"],
      depth: 1,
      includeInvalid: true,
    });

    expect(current).toEqual([]);
    expect(historical.map((step) => step.node.id)).toEqual([beta.id]);
    expect(invalidIncluded.map((step) => step.node.id)).toEqual([beta.id]);
  });
});
