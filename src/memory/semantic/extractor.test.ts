import { mkdtempSync, rmSync } from "node:fs";
import { join } from "node:path";
import { tmpdir } from "node:os";

import { afterEach, describe, expect, it, vi } from "vitest";

import type { EmbeddingClient } from "../../embeddings/index.js";
import { FakeLLMClient } from "../../llm/index.js";
import { LanceDbStore } from "../../storage/lancedb/index.js";
import { openDatabase } from "../../storage/sqlite/index.js";
import { FixedClock } from "../../util/clock.js";
import { createSemanticNodeId, type EpisodeId } from "../../util/ids.js";
import type { Episode } from "../episodic/types.js";
import { SemanticExtractor } from "./extractor.js";
import { semanticMigrations } from "./migrations.js";
import {
  SemanticEdgeRepository,
  SemanticNodeRepository,
  createSemanticNodesTableSchema,
} from "./repository.js";

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

function buildEpisode(id: Episode["id"], title: string): Episode {
  return {
    id,
    title,
    narrative: `${title} narrative.`,
    participants: ["team"],
    location: null,
    start_time: 1,
    end_time: 2,
    source_stream_ids: ["strm_aaaaaaaaaaaaaaaa" as Episode["source_stream_ids"][number]],
    significance: 0.8,
    tags: ["atlas"],
    confidence: 0.8,
    lineage: {
      derived_from: [],
      supersedes: [],
    },
    embedding: Float32Array.from([1, 0, 0, 0]),
    created_at: 1,
    updated_at: 1,
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

    const extractor = new SemanticExtractor({
      nodeRepository,
      edgeRepository,
      embeddingClient: new SemanticEmbeddingClient(),
      llmClient: new FakeLLMClient({
        responses: [
          {
            text: JSON.stringify({
              nodes: [
                {
                  kind: "entity",
                  label: "Atlas",
                  description: "Atlas updated node",
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
            input_tokens: 1,
            output_tokens: 1,
            stop_reason: "end_turn",
            tool_calls: [],
          },
        ],
      }),
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
    expect((await nodeRepository.list()).map((node) => node.label)).toEqual(
      expect.arrayContaining(["Atlas", "Rollback"]),
    );
    expect(edgeRepository.listEdges()).toHaveLength(1);
    expect(nodeInsertSpy).toHaveBeenCalled();
    expect(edgeAddSpy).toHaveBeenCalled();
    expect(Math.max(...nodeInsertSpy.mock.invocationCallOrder)).toBeLessThan(
      edgeAddSpy.mock.invocationCallOrder[0] ?? Number.MAX_SAFE_INTEGER,
    );

    const hallucinatingExtractor = new SemanticExtractor({
      nodeRepository,
      edgeRepository,
      embeddingClient: new SemanticEmbeddingClient(),
      llmClient: new FakeLLMClient({
        responses: [
          {
            text: JSON.stringify({
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
            input_tokens: 1,
            output_tokens: 1,
            stop_reason: "end_turn",
            tool_calls: [],
          },
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
});
