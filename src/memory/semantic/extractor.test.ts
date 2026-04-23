import { mkdtempSync, rmSync } from "node:fs";
import { join } from "node:path";
import { tmpdir } from "node:os";

import { afterEach, describe, expect, it, vi } from "vitest";

import type { EmbeddingClient } from "../../embeddings/index.js";
import { FakeLLMClient } from "../../llm/index.js";
import { LanceDbStore } from "../../storage/lancedb/index.js";
import { openDatabase } from "../../storage/sqlite/index.js";
import { FixedClock } from "../../util/clock.js";
import { createEntityId, createSemanticNodeId, type EpisodeId } from "../../util/ids.js";
import type { Episode } from "../episodic/types.js";
import { SemanticExtractor } from "./extractor.js";
import { semanticMigrations } from "./migrations.js";
import {
  SemanticEdgeRepository,
  SemanticNodeRepository,
  createSemanticNodesTableSchema,
} from "./repository.js";

const SEMANTIC_TOOL_NAME = "EmitSemanticCandidates";

function createSemanticToolResponse(input: { nodes: unknown[]; edges: unknown[] }) {
  return {
    text: "",
    input_tokens: 1,
    output_tokens: 1,
    stop_reason: "tool_use" as const,
    tool_calls: [
      {
        id: "toolu_1",
        name: SEMANTIC_TOOL_NAME,
        input,
      },
    ],
  };
}

class SemanticEmbeddingClient implements EmbeddingClient {
  async embed(text: string): Promise<Float32Array> {
    return this.vector(text);
  }

  async embedBatch(texts: readonly string[]): Promise<Float32Array[]> {
    return texts.map((text) => this.vector(text));
  }

  private vector(text: string): Float32Array {
    if (/atlas/i.test(text)) {
      return Float32Array.from([1, 0, 0, 0]);
    }

    return Float32Array.from([0, 1, 0, 0]);
  }
}

function buildEpisode(id: Episode["id"], title: string, overrides: Partial<Episode> = {}): Episode {
  return {
    id,
    title,
    narrative: overrides.narrative ?? `${title} narrative.`,
    participants: overrides.participants ?? ["team"],
    location: overrides.location ?? null,
    start_time: overrides.start_time ?? 1,
    end_time: overrides.end_time ?? 2,
    source_stream_ids: overrides.source_stream_ids ?? [
      "strm_aaaaaaaaaaaaaaaa" as Episode["source_stream_ids"][number],
    ],
    significance: overrides.significance ?? 0.8,
    tags: overrides.tags ?? ["atlas"],
    confidence: overrides.confidence ?? 0.8,
    lineage: overrides.lineage ?? {
      derived_from: [],
      supersedes: [],
    },
    emotional_arc: overrides.emotional_arc ?? null,
    audience_entity_id: overrides.audience_entity_id,
    shared: overrides.shared,
    embedding: overrides.embedding ?? Float32Array.from([1, 0, 0, 0]),
    created_at: overrides.created_at ?? 1,
    updated_at: overrides.updated_at ?? 1,
  };
}

function createEpisodeLookup(episodes: readonly Episode[]) {
  const episodeById = new Map(episodes.map((episode) => [episode.id, episode]));

  return {
    getMany: async (ids: readonly Episode["id"][]) =>
      ids
        .map((id) => episodeById.get(id))
        .filter((episode): episode is Episode => episode !== undefined),
  };
}

describe("semantic extractor", () => {
  const cleanup: Array<() => Promise<void>> = [];

  afterEach(async () => {
    while (cleanup.length > 0) {
      await cleanup.pop()?.();
    }
  });

  it("extracts nodes and edges, rejects hallucinated refs, and merges duplicates", async () => {
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

    cleanup.push(async () => {
      db.close();
      await store.close();
      rmSync(tempDir, { recursive: true, force: true });
    });

    await nodeRepository.insert({
      id: createSemanticNodeId(),
      kind: "entity",
      label: "Atlas",
      description: "Atlas existing node",
      domain: "tech",
      aliases: ["Project Atlas"],
      confidence: 0.6,
      source_episode_ids: ["ep_aaaaaaaaaaaaaaaa" as EpisodeId],
      created_at: 1,
      updated_at: 1,
      last_verified_at: 1,
      embedding: Float32Array.from([1, 0, 0, 0]),
      archived: false,
      superseded_by: null,
    });

    const llm = new FakeLLMClient({
      responses: [
        createSemanticToolResponse({
          nodes: [
            {
              kind: "entity",
              label: "Atlas",
              description: "Atlas updated node",
              domain: " Technology ",
              aliases: ["Atlas service"],
              confidence: 0.7,
              source_episode_ids: ["ep_aaaaaaaaaaaaaaaa"],
            },
            {
              kind: "concept",
              label: "Rollback",
              description: "Rollback plan",
              aliases: [],
              confidence: 0.6,
              source_episode_ids: ["ep_aaaaaaaaaaaaaaaa"],
            },
          ],
          edges: [
            {
              from_label: "Atlas",
              to_label: "Rollback",
              relation: "supports",
              confidence: 0.6,
              evidence_episode_ids: ["ep_aaaaaaaaaaaaaaaa"],
            },
          ],
        }),
      ],
    });
    const extractor = new SemanticExtractor({
      nodeRepository,
      edgeRepository,
      embeddingClient: new SemanticEmbeddingClient(),
      episodicRepository: createEpisodeLookup([
        buildEpisode("ep_aaaaaaaaaaaaaaaa" as Episode["id"], "Atlas incident"),
      ]),
      llmClient: llm,
      model: "haiku",
      clock,
    });
    const nodeInsertSpy = vi.spyOn(nodeRepository, "insert");
    const edgeAddSpy = vi.spyOn(edgeRepository, "addEdge");

    const result = await extractor.extractFromEpisodes([
      buildEpisode("ep_aaaaaaaaaaaaaaaa" as Episode["id"], "Atlas incident"),
    ]);

    expect(result).toEqual({
      insertedNodes: 1,
      updatedNodes: 1,
      skippedNodes: 0,
      insertedEdges: 1,
      skippedEdges: 0,
    });
    const nodesAfterMerge = await nodeRepository.list();

    expect(nodesAfterMerge.map((node) => node.label)).toEqual(
      expect.arrayContaining(["Atlas", "Rollback"]),
    );
    expect(nodesAfterMerge.find((node) => node.label === "Atlas")?.domain).toBe("tech");
    expect(edgeRepository.listEdges()).toHaveLength(1);
    expect(nodeInsertSpy).toHaveBeenCalled();
    expect(edgeAddSpy).toHaveBeenCalled();
    expect(llm.requests[0]?.tool_choice).toEqual({
      type: "tool",
      name: SEMANTIC_TOOL_NAME,
    });
    expect(Math.max(...nodeInsertSpy.mock.invocationCallOrder)).toBeLessThan(
      edgeAddSpy.mock.invocationCallOrder[0] ?? Number.MAX_SAFE_INTEGER,
    );

    const hallucinatingExtractor = new SemanticExtractor({
      nodeRepository,
      edgeRepository,
      embeddingClient: new SemanticEmbeddingClient(),
      episodicRepository: createEpisodeLookup([
        buildEpisode("ep_aaaaaaaaaaaaaaaa" as Episode["id"], "Atlas incident"),
      ]),
      llmClient: new FakeLLMClient({
        responses: [
          createSemanticToolResponse({
            nodes: [
              {
                kind: "concept",
                label: "Bad node",
                description: "Bad node",
                aliases: [],
                confidence: 0.6,
                source_episode_ids: ["ep_missing"],
              },
            ],
            edges: [],
          }),
        ],
      }),
      model: "haiku",
      clock,
    });

    await expect(
      hallucinatingExtractor.extractFromEpisodes([
        buildEpisode("ep_aaaaaaaaaaaaaaaa" as Episode["id"], "Atlas incident"),
      ]),
    ).rejects.toThrow("unknown source_episode_ids");
  });

  it("creates edges between existing nodes even when the batch omits node candidates", async () => {
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
    const episode = buildEpisode("ep_aaaaaaaaaaaaaaaa" as Episode["id"], "Alice and Bob");

    cleanup.push(async () => {
      db.close();
      await store.close();
      rmSync(tempDir, { recursive: true, force: true });
    });

    const alice = await nodeRepository.insert({
      id: createSemanticNodeId(),
      kind: "entity",
      label: "Alice",
      description: "Alice existing node",
      domain: "people",
      aliases: [],
      confidence: 0.8,
      source_episode_ids: [episode.id],
      created_at: 1,
      updated_at: 1,
      last_verified_at: 1,
      embedding: Float32Array.from([1, 0, 0, 0]),
      archived: false,
      superseded_by: null,
    });
    const bob = await nodeRepository.insert({
      id: createSemanticNodeId(),
      kind: "entity",
      label: "Bob",
      description: "Bob existing node",
      domain: "people",
      aliases: [],
      confidence: 0.8,
      source_episode_ids: [episode.id],
      created_at: 1,
      updated_at: 1,
      last_verified_at: 1,
      embedding: Float32Array.from([0, 1, 0, 0]),
      archived: false,
      superseded_by: null,
    });
    const extractor = new SemanticExtractor({
      nodeRepository,
      edgeRepository,
      embeddingClient: new SemanticEmbeddingClient(),
      episodicRepository: createEpisodeLookup([episode]),
      llmClient: new FakeLLMClient({
        responses: [
          createSemanticToolResponse({
            nodes: [],
            edges: [
              {
                from_label: "Alice",
                to_label: "Bob",
                relation: "related_to",
                confidence: 0.8,
                evidence_episode_ids: [episode.id],
              },
            ],
          }),
        ],
      }),
      model: "haiku",
      clock,
    });

    const result = await extractor.extractFromEpisodes([episode]);
    const [edge] = edgeRepository.listEdges();

    expect(result.insertedEdges).toBe(1);
    expect(edge).toMatchObject({
      from_node_id: alice.id,
      to_node_id: bob.id,
      relation: "related_to",
    });
  });

  it("does not resolve label-only edges to archived nodes", async () => {
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
    const episode = buildEpisode("ep_aaaaaaaaaaaaaaaa" as Episode["id"], "Archived Alice");

    cleanup.push(async () => {
      db.close();
      await store.close();
      rmSync(tempDir, { recursive: true, force: true });
    });

    await nodeRepository.insert({
      id: createSemanticNodeId(),
      kind: "entity",
      label: "Alice",
      description: "Archived Alice node",
      domain: "people",
      aliases: [],
      confidence: 0.8,
      source_episode_ids: [episode.id],
      created_at: 1,
      updated_at: 1,
      last_verified_at: 1,
      embedding: Float32Array.from([1, 0, 0, 0]),
      archived: true,
      superseded_by: null,
    });
    await nodeRepository.insert({
      id: createSemanticNodeId(),
      kind: "entity",
      label: "Bob",
      description: "Bob existing node",
      domain: "people",
      aliases: [],
      confidence: 0.8,
      source_episode_ids: [episode.id],
      created_at: 1,
      updated_at: 1,
      last_verified_at: 1,
      embedding: Float32Array.from([0, 1, 0, 0]),
      archived: false,
      superseded_by: null,
    });
    const extractor = new SemanticExtractor({
      nodeRepository,
      edgeRepository,
      embeddingClient: new SemanticEmbeddingClient(),
      episodicRepository: createEpisodeLookup([episode]),
      llmClient: new FakeLLMClient({
        responses: [
          createSemanticToolResponse({
            nodes: [],
            edges: [
              {
                from_label: "Alice",
                to_label: "Bob",
                relation: "related_to",
                confidence: 0.8,
                evidence_episode_ids: [episode.id],
              },
            ],
          }),
        ],
      }),
      model: "haiku",
      clock,
    });

    await expect(extractor.extractFromEpisodes([episode])).rejects.toMatchObject({
      code: "SEMANTIC_EXTRACTOR_INVALID_REF",
    });
    expect(edgeRepository.listEdges()).toHaveLength(0);
  });

  it("does not resolve label-only edges across audience scopes", async () => {
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
    const bobAudience = createEntityId();
    const publicEpisode = buildEpisode("ep_aaaaaaaaaaaaaaaa" as Episode["id"], "Public Alice");
    const privateEpisode = buildEpisode("ep_bbbbbbbbbbbbbbbb" as Episode["id"], "Private Alice", {
      audience_entity_id: bobAudience,
      shared: false,
    });

    cleanup.push(async () => {
      db.close();
      await store.close();
      rmSync(tempDir, { recursive: true, force: true });
    });

    await nodeRepository.insert({
      id: createSemanticNodeId(),
      kind: "entity",
      label: "Alice",
      description: "Private Alice node",
      domain: "people",
      aliases: [],
      confidence: 0.8,
      source_episode_ids: [privateEpisode.id],
      created_at: 1,
      updated_at: 1,
      last_verified_at: 1,
      embedding: Float32Array.from([1, 0, 0, 0]),
      archived: false,
      superseded_by: null,
    });
    await nodeRepository.insert({
      id: createSemanticNodeId(),
      kind: "entity",
      label: "Bob",
      description: "Public Bob node",
      domain: "people",
      aliases: [],
      confidence: 0.8,
      source_episode_ids: [publicEpisode.id],
      created_at: 1,
      updated_at: 1,
      last_verified_at: 1,
      embedding: Float32Array.from([0, 1, 0, 0]),
      archived: false,
      superseded_by: null,
    });
    const extractor = new SemanticExtractor({
      nodeRepository,
      edgeRepository,
      embeddingClient: new SemanticEmbeddingClient(),
      episodicRepository: createEpisodeLookup([publicEpisode, privateEpisode]),
      llmClient: new FakeLLMClient({
        responses: [
          createSemanticToolResponse({
            nodes: [],
            edges: [
              {
                from_label: "Alice",
                to_label: "Bob",
                relation: "related_to",
                confidence: 0.8,
                evidence_episode_ids: [publicEpisode.id],
              },
            ],
          }),
        ],
      }),
      model: "haiku",
      clock,
    });

    await expect(extractor.extractFromEpisodes([publicEpisode])).rejects.toMatchObject({
      code: "SEMANTIC_EXTRACTOR_INVALID_REF",
    });
    expect(edgeRepository.listEdges()).toHaveLength(0);
  });

  it("re-embeds updated nodes from the final stored text", async () => {
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

    cleanup.push(async () => {
      db.close();
      await store.close();
      rmSync(tempDir, { recursive: true, force: true });
    });

    const existing = await nodeRepository.insert({
      id: createSemanticNodeId(),
      kind: "entity",
      label: "Atlas",
      description: "Atlas existing node",
      domain: "tech",
      aliases: ["Project Atlas"],
      confidence: 0.8,
      source_episode_ids: ["ep_aaaaaaaaaaaaaaaa" as EpisodeId],
      created_at: 1,
      updated_at: 1,
      last_verified_at: 1,
      embedding: Float32Array.from([1, 0, 0, 0]),
      archived: false,
      superseded_by: null,
    });
    const embed = vi.fn(async () => Float32Array.from([1, 0, 0, 0]));
    const embeddingClient: EmbeddingClient = {
      embed,
      embedBatch: async (texts) => Promise.all(texts.map((text) => embed(text))),
    };
    const extractor = new SemanticExtractor({
      nodeRepository,
      edgeRepository,
      embeddingClient,
      episodicRepository: createEpisodeLookup([
        buildEpisode("ep_aaaaaaaaaaaaaaaa" as Episode["id"], "Atlas incident"),
      ]),
      llmClient: new FakeLLMClient({
        responses: [
          createSemanticToolResponse({
            nodes: [
              {
                kind: "entity",
                label: "Atlas",
                description: "Atlas newer description that should not win",
                domain: "tech",
                aliases: ["Atlas service"],
                confidence: 0.4,
                source_episode_ids: ["ep_aaaaaaaaaaaaaaaa"],
              },
            ],
            edges: [],
          }),
        ],
      }),
      model: "haiku",
      clock,
    });

    await extractor.extractFromEpisodes([
      buildEpisode("ep_aaaaaaaaaaaaaaaa" as Episode["id"], "Atlas incident"),
    ]);
    const updated = await nodeRepository.get(existing.id);

    expect(updated?.description).toBe("Atlas existing node");
    expect(updated?.aliases).toEqual(["Project Atlas", "Atlas", "Atlas service"]);
    expect(embed.mock.calls.at(-1)?.[0]).toBe(
      "Atlas\nAtlas existing node\nProject Atlas Atlas Atlas service",
    );
  });

  it("normalizes string-wrapped tool payloads before validation", async () => {
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

    cleanup.push(async () => {
      db.close();
      await store.close();
      rmSync(tempDir, { recursive: true, force: true });
    });

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
              name: SEMANTIC_TOOL_NAME,
              input: {
                nodes:
                  '[{"kind":"entity","label":"Atlas","description":"Atlas node","aliases":[],"confidence":0.7,"source_episode_ids":["ep_aaaaaaaaaaaaaaaa"]},{"kind":"concept","label":"Rollback","description":"Rollback concept","aliases":[],"confidence":0.6,"source_episode_ids":["ep_aaaaaaaaaaaaaaaa"]}]<parameter name="edges">[{"from_label":"Atlas","to_label":"Rollback","relation":"related_to","confidence":0.5,"evidence_episode_ids":["ep_aaaaaaaaaaaaaaaa"]}]',
              },
            },
          ],
        },
      ],
    });
    const extractor = new SemanticExtractor({
      nodeRepository,
      edgeRepository,
      embeddingClient: new SemanticEmbeddingClient(),
      episodicRepository: createEpisodeLookup([
        buildEpisode("ep_aaaaaaaaaaaaaaaa" as Episode["id"], "Atlas incident"),
      ]),
      llmClient: llm,
      model: "haiku",
      clock,
    });

    const result = await extractor.extractFromEpisodes([
      buildEpisode("ep_aaaaaaaaaaaaaaaa" as Episode["id"], "Atlas incident"),
    ]);

    expect(result.insertedNodes).toBe(2);
    expect(result.insertedEdges).toBe(1);
  });

  it("keeps homonyms in separate nodes when domains differ", async () => {
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
    const episodeA = buildEpisode("ep_aaaaaaaaaaaaaaaa" as Episode["id"], "Tomasz conversation", {
      narrative: "Tomasz joined the call and asked about plans.",
      participants: ["Alice", "Tomasz"],
      audience_entity_id: "ent_aaaaaaaaaaaaaaaa" as Episode["audience_entity_id"],
      shared: false,
      tags: ["people"],
    });
    const episodeB = buildEpisode("ep_bbbbbbbbbbbbbbbb" as Episode["id"], "Tomasz travel note", {
      narrative: "We discussed the city of Tomasz and nearby routes.",
      participants: ["team"],
      tags: ["travel", "places"],
      location: "Poland",
    });
    const llm = new FakeLLMClient({
      responses: [
        createSemanticToolResponse({
          nodes: [
            {
              kind: "entity",
              label: "Tomasz",
              description: "A person participating in the conversation.",
              domain: "people",
              aliases: [],
              confidence: 0.7,
              source_episode_ids: ["ep_aaaaaaaaaaaaaaaa"],
            },
          ],
          edges: [],
        }),
        createSemanticToolResponse({
          nodes: [
            {
              kind: "entity",
              label: "Tomasz",
              description: "A city mentioned in the travel discussion.",
              domain: "places",
              aliases: [],
              confidence: 0.7,
              source_episode_ids: ["ep_bbbbbbbbbbbbbbbb"],
            },
          ],
          edges: [],
        }),
      ],
    });
    const extractor = new SemanticExtractor({
      nodeRepository,
      edgeRepository,
      embeddingClient: new SemanticEmbeddingClient(),
      episodicRepository: createEpisodeLookup([episodeA, episodeB]),
      llmClient: llm,
      model: "haiku",
      clock,
    });

    cleanup.push(async () => {
      db.close();
      await store.close();
      rmSync(tempDir, { recursive: true, force: true });
    });

    await extractor.extractFromEpisodes([episodeA]);
    await extractor.extractFromEpisodes([episodeB]);

    const matches = await nodeRepository.findByLabelOrAlias("Tomasz", 5, {
      includeArchived: true,
    });

    expect(matches).toHaveLength(2);
    expect(matches.map((node) => node.domain).sort()).toEqual(["people", "places"]);
    expect(matches.map((node) => node.source_episode_ids[0])).toEqual(
      expect.arrayContaining([episodeA.id, episodeB.id]),
    );
  });

  it("does not merge nodes when either side lacks a domain", async () => {
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
    const episode = buildEpisode("ep_aaaaaaaaaaaaaaaa" as Episode["id"], "Tomasz ambiguous note", {
      narrative: "Tomasz came up in an ambiguous context.",
      participants: ["Alice", "Tomasz"],
      audience_entity_id: "ent_aaaaaaaaaaaaaaaa" as Episode["audience_entity_id"],
      shared: false,
    });
    const llm = new FakeLLMClient({
      responses: [
        createSemanticToolResponse({
          nodes: [
            {
              kind: "entity",
              label: "Tomasz",
              description: "An ambiguous Tomasz reference.",
              aliases: [],
              confidence: 0.7,
              source_episode_ids: ["ep_aaaaaaaaaaaaaaaa"],
            },
          ],
          edges: [],
        }),
        createSemanticToolResponse({
          nodes: [
            {
              kind: "entity",
              label: "Tomasz",
              description: "A person named Tomasz.",
              domain: "people",
              aliases: [],
              confidence: 0.7,
              source_episode_ids: ["ep_aaaaaaaaaaaaaaaa"],
            },
          ],
          edges: [],
        }),
      ],
    });
    const extractor = new SemanticExtractor({
      nodeRepository,
      edgeRepository,
      embeddingClient: new SemanticEmbeddingClient(),
      episodicRepository: createEpisodeLookup([episode]),
      llmClient: llm,
      model: "haiku",
      clock,
    });

    cleanup.push(async () => {
      db.close();
      await store.close();
      rmSync(tempDir, { recursive: true, force: true });
    });

    await nodeRepository.insert({
      id: createSemanticNodeId(),
      kind: "entity",
      label: "Tomasz",
      description: "Existing ambiguous Tomasz node.",
      domain: null,
      aliases: [],
      confidence: 0.6,
      source_episode_ids: [episode.id],
      created_at: 1,
      updated_at: 1,
      last_verified_at: 1,
      embedding: Float32Array.from([0, 1, 0, 0]),
      archived: false,
      superseded_by: null,
    });

    await extractor.extractFromEpisodes([episode]);

    const afterNullCandidate = await nodeRepository.findByLabelOrAlias("Tomasz", 5, {
      includeArchived: true,
    });

    expect(afterNullCandidate).toHaveLength(2);
    expect(afterNullCandidate.map((node) => node.domain)).toEqual([null, null]);

    await extractor.extractFromEpisodes([episode]);

    const afterSpecificCandidate = await nodeRepository.findByLabelOrAlias("Tomasz", 5, {
      includeArchived: true,
    });

    expect(afterSpecificCandidate).toHaveLength(3);
    expect(afterSpecificCandidate.map((node) => node.domain)).toEqual(
      expect.arrayContaining([null, null, "people"]),
    );
  });

  it("stores unknown domains as trimmed lowercase free-form strings", async () => {
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
    const episode = buildEpisode("ep_aaaaaaaaaaaaaaaa" as Episode["id"], "Craft fair note", {
      narrative: "The note discussed artisanal craft details.",
    });
    const extractor = new SemanticExtractor({
      nodeRepository,
      edgeRepository,
      embeddingClient: new SemanticEmbeddingClient(),
      episodicRepository: createEpisodeLookup([episode]),
      llmClient: new FakeLLMClient({
        responses: [
          createSemanticToolResponse({
            nodes: [
              {
                kind: "concept",
                label: "Artisanal craft",
                description: "A handmade craft category.",
                domain: "  Artisanal-Craft  ",
                aliases: [],
                confidence: 0.7,
                source_episode_ids: ["ep_aaaaaaaaaaaaaaaa"],
              },
            ],
            edges: [],
          }),
        ],
      }),
      model: "haiku",
      clock,
    });

    cleanup.push(async () => {
      db.close();
      await store.close();
      rmSync(tempDir, { recursive: true, force: true });
    });

    await extractor.extractFromEpisodes([episode]);

    expect(await nodeRepository.list()).toEqual([
      expect.objectContaining({
        label: "Artisanal craft",
        domain: "artisanal-craft",
      }),
    ]);
  });
});
