import { mkdtempSync, rmSync } from "node:fs";
import { join } from "node:path";
import { tmpdir } from "node:os";

import { afterEach, describe, expect, it, vi } from "vitest";

import { SuppressionSet, computeWeights } from "../cognition/attention/index.js";
import type { EmbeddingClient } from "../embeddings/index.js";
import { StreamReader, StreamWriter } from "../stream/index.js";
import { LanceDbStore } from "../storage/lancedb/index.js";
import { openDatabase } from "../storage/sqlite/index.js";
import { FixedClock } from "../util/clock.js";
import { selfMigrations } from "../memory/self/migrations.js";
import { episodicMigrations } from "../memory/episodic/migrations.js";
import { EpisodicRepository, createEpisodesTableSchema } from "../memory/episodic/repository.js";
import { retrievalMigrations } from "./migrations.js";
import { RetrievalPipeline } from "./pipeline.js";
import type { Episode } from "../memory/episodic/types.js";

class ScriptedEmbeddingClient implements EmbeddingClient {
  async embed(text: string): Promise<Float32Array> {
    return this.embedVector(text);
  }

  async embedBatch(texts: readonly string[]): Promise<Float32Array[]> {
    return texts.map((text) => this.embedVector(text));
  }

  private embedVector(text: string): Float32Array {
    if (text.includes("planning")) {
      return Float32Array.from([1, 0, 0, 0]);
    }

    if (text.includes("retrospective")) {
      return Float32Array.from([0, 1, 0, 0]);
    }

    return Float32Array.from([0, 0, 1, 0]);
  }
}

function createEpisode(id: string, sourceId: string, embedding: number[]): Episode {
  return {
    id: id as Episode["id"],
    title: `${id} title`,
    narrative: `${id} narrative.`,
    participants: ["user"],
    location: null,
    start_time: 1_000,
    end_time: 2_000,
    source_stream_ids: [sourceId as Episode["source_stream_ids"][number]],
    significance: 0.8,
    tags: ["planning"],
    confidence: 0.9,
    lineage: {
      derived_from: [],
      supersedes: [],
    },
    embedding: Float32Array.from(embedding),
    created_at: 1_000,
    updated_at: 1_000,
  };
}

describe("retrieval pipeline", () => {
  const cleanup: Array<() => Promise<void>> = [];

  afterEach(async () => {
    vi.restoreAllMocks();

    while (cleanup.length > 0) {
      await cleanup.pop()?.();
    }
  });

  it("retrieves episodes, resolves citations, and records retrieval stats", async () => {
    const tempDir = mkdtempSync(join(tmpdir(), "borg-"));
    const store = new LanceDbStore({
      uri: join(tempDir, "lancedb"),
    });
    const db = openDatabase(join(tempDir, "borg.db"), {
      migrations: [...episodicMigrations, ...selfMigrations, ...retrievalMigrations],
    });
    const table = await store.openTable({
      name: "episodes",
      schema: createEpisodesTableSchema(4),
    });
    const repo = new EpisodicRepository({
      table,
      db,
      clock: new FixedClock(5_000),
    });
    const writer = new StreamWriter({
      dataDir: tempDir,
      clock: new FixedClock(2_000),
    });

    cleanup.push(async () => {
      writer.close();
      db.close();
      await store.close();
      rmSync(tempDir, { recursive: true, force: true });
    });

    const firstEntry = await writer.append({
      kind: "user_msg",
      content: "planning kickoff",
    });
    const secondEntry = await writer.append({
      kind: "agent_msg",
      content: "retrospective note",
    });

    await repo.insert(createEpisode("ep_aaaaaaaaaaaaaaaa", firstEntry.id, [1, 0, 0, 0]));
    await repo.insert(createEpisode("ep_bbbbbbbbbbbbbbbb", secondEntry.id, [0, 1, 0, 0]));

    const pipeline = new RetrievalPipeline({
      embeddingClient: new ScriptedEmbeddingClient(),
      episodicRepository: repo,
      dataDir: tempDir,
      clock: new FixedClock(10_000),
    });

    const results = await pipeline.search("planning", {
      limit: 1,
    });

    expect(results).toEqual([
      expect.objectContaining({
        episode: expect.objectContaining({
          id: "ep_aaaaaaaaaaaaaaaa",
        }),
        citationChain: [
          expect.objectContaining({
            id: firstEntry.id,
          }),
        ],
      }),
    ]);
    expect(repo.getStats("ep_aaaaaaaaaaaaaaaa" as Episode["id"])?.retrieval_count).toBe(1);
  });

  it("batches citation resolution into a single stream scan per session", async () => {
    const tempDir = mkdtempSync(join(tmpdir(), "borg-"));
    const store = new LanceDbStore({
      uri: join(tempDir, "lancedb"),
    });
    const db = openDatabase(join(tempDir, "borg.db"), {
      migrations: [...episodicMigrations, ...selfMigrations, ...retrievalMigrations],
    });
    const table = await store.openTable({
      name: "episodes",
      schema: createEpisodesTableSchema(4),
    });
    const repo = new EpisodicRepository({
      table,
      db,
      clock: new FixedClock(5_000),
    });
    const writer = new StreamWriter({
      dataDir: tempDir,
      clock: new FixedClock(2_000),
    });

    cleanup.push(async () => {
      writer.close();
      db.close();
      await store.close();
      rmSync(tempDir, { recursive: true, force: true });
    });

    const firstEntry = await writer.append({
      kind: "user_msg",
      content: "planning kickoff",
    });
    const secondEntry = await writer.append({
      kind: "agent_msg",
      content: "planning follow-up",
    });

    await repo.insert(createEpisode("ep_aaaaaaaaaaaaaaaa", firstEntry.id, [1, 0, 0, 0]));
    await repo.insert(createEpisode("ep_bbbbbbbbbbbbbbbb", secondEntry.id, [0.9, 0.1, 0, 0]));

    const iterateSpy = vi.spyOn(StreamReader.prototype, "iterate");
    const pipeline = new RetrievalPipeline({
      embeddingClient: new ScriptedEmbeddingClient(),
      episodicRepository: repo,
      dataDir: tempDir,
      clock: new FixedClock(10_000),
    });

    const results = await pipeline.search("planning", {
      limit: 2,
    });

    expect(results).toHaveLength(2);
    expect(iterateSpy).toHaveBeenCalledTimes(1);
  });

  it("rescales results with attention weights and suppression", async () => {
    const tempDir = mkdtempSync(join(tmpdir(), "borg-"));
    const store = new LanceDbStore({
      uri: join(tempDir, "lancedb"),
    });
    const db = openDatabase(join(tempDir, "borg.db"), {
      migrations: [...episodicMigrations, ...selfMigrations, ...retrievalMigrations],
    });
    const table = await store.openTable({
      name: "episodes",
      schema: createEpisodesTableSchema(4),
    });
    const repo = new EpisodicRepository({
      table,
      db,
      clock: new FixedClock(5_000),
    });
    const writer = new StreamWriter({
      dataDir: tempDir,
      clock: new FixedClock(2_000),
    });

    cleanup.push(async () => {
      writer.close();
      db.close();
      await store.close();
      rmSync(tempDir, { recursive: true, force: true });
    });

    const firstEntry = await writer.append({
      kind: "user_msg",
      content: "release planning",
    });
    const secondEntry = await writer.append({
      kind: "agent_msg",
      content: "release planning followup",
    });

    await repo.insert({
      ...createEpisode("ep_aaaaaaaaaaaaaaaa", firstEntry.id, [1, 0, 0, 0]),
      title: "release goal",
      narrative: "release goal context",
    });
    await repo.insert({
      ...createEpisode("ep_bbbbbbbbbbbbbbbb", secondEntry.id, [1, 0, 0, 0]),
      title: "generic note",
    });

    const suppression = new SuppressionSet();

    suppression.suppress("ep_bbbbbbbbbbbbbbbb", "already seen", 2);

    const pipeline = new RetrievalPipeline({
      embeddingClient: new ScriptedEmbeddingClient(),
      episodicRepository: repo,
      dataDir: tempDir,
      clock: new FixedClock(10_000),
    });

    const results = await pipeline.search("release planning", {
      limit: 2,
      attentionWeights: computeWeights("reflective", {
        currentGoals: [
          {
            id: "goal_aaaaaaaaaaaaaaaa",
            description: "release goal",
            priority: 1,
            parent_goal_id: null,
            status: "active",
            progress_notes: null,
            created_at: 0,
            target_at: null,
          },
        ],
        hasTemporalCue: false,
      }),
      goalDescriptions: ["release goal"],
      suppressionSet: suppression,
    });

    expect(results[0]?.episode.id).toBe("ep_aaaaaaaaaaaaaaaa");
    expect(results[1]?.scoreBreakdown.suppressionPenalty).toBe(1);
  });
});
