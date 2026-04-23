import { mkdtempSync, rmSync } from "node:fs";
import { join } from "node:path";
import { tmpdir } from "node:os";

import { afterEach, describe, expect, it } from "vitest";

import type { EmbeddingClient } from "../../embeddings/index.js";
import { FakeLLMClient } from "../../llm/index.js";
import { StreamWriter } from "../../stream/index.js";
import { LanceDbStore } from "../../storage/lancedb/index.js";
import { openDatabase } from "../../storage/sqlite/index.js";
import { ManualClock } from "../../util/clock.js";
import { EmbeddingError, LLMError } from "../../util/errors.js";
import { retrievalMigrations } from "../../retrieval/migrations.js";
import { EntityRepository } from "../commitments/index.js";
import { selfMigrations } from "../self/migrations.js";
import { episodicMigrations } from "./migrations.js";
import { EpisodicExtractor } from "./extractor.js";
import { EpisodicRepository, createEpisodesTableSchema } from "./repository.js";

const EPISODE_TOOL_NAME = "EmitEpisodeCandidates";

function createEpisodeToolResponse(episodes: unknown[]) {
  return {
    text: "",
    input_tokens: 10,
    output_tokens: 20,
    stop_reason: "tool_use" as const,
    tool_calls: [
      {
        id: "toolu_1",
        name: EPISODE_TOOL_NAME,
        input: {
          episodes: episodes.map((episode) =>
            typeof episode === "object" && episode !== null && !("location" in episode)
              ? { ...episode, location: null }
              : episode,
          ),
        },
      },
    ],
  };
}

class TitleEmbeddingClient implements EmbeddingClient {
  async embed(text: string): Promise<Float32Array> {
    return this.vector(text);
  }

  async embedBatch(texts: readonly string[]): Promise<Float32Array[]> {
    return texts.map((text) => this.vector(text));
  }

  private vector(text: string): Float32Array {
    if (text.includes("Planning sync")) {
      return Float32Array.from([1, 0, 0, 0]);
    }

    return Float32Array.from([0, 1, 0, 0]);
  }
}

class FailingOnceEmbeddingClient implements EmbeddingClient {
  private failed = false;

  async embed(text: string): Promise<Float32Array> {
    if (!this.failed && text.includes("Skip me")) {
      this.failed = true;
      throw new EmbeddingError("embedding failed");
    }

    return Float32Array.from([1, 0, 0, 0]);
  }

  async embedBatch(texts: readonly string[]): Promise<Float32Array[]> {
    return Promise.all(texts.map((text) => this.embed(text)));
  }
}

describe("episodic extractor", () => {
  const cleanup: Array<() => Promise<void>> = [];

  afterEach(async () => {
    while (cleanup.length > 0) {
      await cleanup.pop()?.();
    }
  });

  it("keeps repeated similar episodes on different days as distinct episodes", async () => {
    const tempDir = mkdtempSync(join(tmpdir(), "borg-"));
    const clock = new ManualClock(1_000);
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
      clock,
    });
    const entityRepository = new EntityRepository({
      db,
      clock,
    });
    const writer = new StreamWriter({
      dataDir: tempDir,
      clock,
    });

    cleanup.push(async () => {
      writer.close();
      db.close();
      await store.close();
      rmSync(tempDir, { recursive: true, force: true });
    });

    const first = await writer.append({
      kind: "user_msg",
      content: "We reviewed the borg architecture and memory bands together.",
    });
    const llm = new FakeLLMClient({
      responses: [
        createEpisodeToolResponse([
          {
            title: "Reviewed borg architecture",
            narrative:
              "We reviewed the borg architecture and discussed the memory bands. The conversation focused on how the pieces fit together.",
            source_stream_ids: [first.id],
            participants: ["team"],
            tags: ["architecture", "borg"],
            confidence: 0.8,
            significance: 0.7,
          },
        ]),
      ],
    });
    const extractor = new EpisodicExtractor({
      dataDir: tempDir,
      episodicRepository: repo,
      embeddingClient: new TitleEmbeddingClient(),
      llmClient: llm,
      model: "claude-haiku",
      entityRepository,
      clock,
    });

    const firstRun = await extractor.extractFromStream();
    clock.advance(4 * 24 * 60 * 60 * 1_000);
    const second = await writer.append({
      kind: "agent_msg",
      content: "We reviewed the borg architecture again and compared the retrieval pipeline changes.",
    });
    llm.pushResponse(
      createEpisodeToolResponse([
        {
          title: "Reviewed borg architecture",
          narrative:
            "We revisited the borg architecture and compared the retrieval pipeline changes. This was a later review, not the original conversation.",
          source_stream_ids: [second.id],
          participants: ["team", "pm"],
          tags: ["architecture", "retrieval"],
          confidence: 0.9,
          significance: 0.9,
        },
      ]),
    );
    const secondRun = await extractor.extractFromStream({
      sinceTs: second.timestamp,
    });
    const listed = await repo.listAll();

    expect(firstRun).toEqual({
      inserted: 1,
      updated: 0,
      skipped: 0,
    });
    expect(secondRun).toEqual({
      inserted: 1,
      updated: 0,
      skipped: 0,
    });
    expect(listed).toHaveLength(2);
    expect(listed.map((episode) => episode.source_stream_ids)).toEqual(
      expect.arrayContaining([[first.id], [second.id]]),
    );
    expect(listed.map((episode) => episode.start_time)).toEqual(
      expect.arrayContaining([first.timestamp, second.timestamp]),
    );
    expect(llm.requests[0]?.tool_choice).toEqual({
      type: "tool",
      name: EPISODE_TOOL_NAME,
    });
  });

  it("filters internal scaffolding out of episodic extraction chunks", async () => {
    const tempDir = mkdtempSync(join(tmpdir(), "borg-"));
    const clock = new ManualClock(1_000);
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
      clock,
    });
    const entityRepository = new EntityRepository({
      db,
      clock,
    });
    const writer = new StreamWriter({
      dataDir: tempDir,
      clock,
    });

    cleanup.push(async () => {
      writer.close();
      db.close();
      await store.close();
      rmSync(tempDir, { recursive: true, force: true });
    });

    const user = await writer.append({
      kind: "user_msg",
      content: "Sam asked about the Atlas deployment issue.",
    });
    clock.advance(10);
    await writer.append({
      kind: "thought",
      content: "internal plan: inspect pnpm lockfile before answering",
    });
    clock.advance(10);
    await writer.append({
      kind: "internal_event",
      content: {
        kind: "debug_hook",
        detail: "non-conversational scaffolding",
      },
    });
    clock.advance(10);
    await writer.append({
      kind: "perception",
      content: {
        mode: "problem_solving",
        entities: ["Atlas"],
        temporalCue: null,
        affectiveSignal: {
          valence: -0.3,
          arousal: 0.4,
          dominant_emotion: "frustration",
        },
      },
    });
    clock.advance(10);
    await writer.append({
      kind: "tool_call",
      content: {
        call_id: "call_1",
        tool_name: "tool.test.echo",
        input: {
          value: "Atlas",
        },
        origin: "deliberator",
      },
    });
    clock.advance(10);
    const agent = await writer.append({
      kind: "agent_msg",
      content: "I suggested rerunning pnpm install before the next deploy.",
    });

    const llm = new FakeLLMClient({
      responses: [
        createEpisodeToolResponse([
          {
            title: "Atlas deploy debugging",
            narrative:
              "Sam asked about the Atlas deployment issue. I suggested rerunning pnpm install before the next deploy.",
            source_stream_ids: [user.id, agent.id],
            participants: ["Sam"],
            tags: ["atlas", "deploy"],
            confidence: 0.8,
            significance: 0.7,
          },
        ]),
      ],
    });
    const extractor = new EpisodicExtractor({
      dataDir: tempDir,
      episodicRepository: repo,
      embeddingClient: new TitleEmbeddingClient(),
      llmClient: llm,
      model: "claude-haiku",
      entityRepository,
      clock,
    });

    const result = await extractor.extractFromStream();
    const prompt = String(llm.requests[0]?.messages[0]?.content ?? "");
    const listed = await repo.listAll();

    expect(result).toEqual({
      inserted: 1,
      updated: 0,
      skipped: 0,
    });
    expect(prompt).not.toContain("internal plan");
    expect(prompt).not.toContain('"perception"');
    expect(prompt).not.toContain('"tool_call"');
    expect(prompt).not.toContain("non-conversational scaffolding");
    expect(listed[0]?.source_stream_ids).toEqual([user.id, agent.id]);
  });

  it("treats replayed chunks as idempotent no-ops keyed by source stream ids", async () => {
    const tempDir = mkdtempSync(join(tmpdir(), "borg-"));
    const clock = new ManualClock(1_000);
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
      clock,
    });
    const entityRepository = new EntityRepository({
      db,
      clock,
    });
    const writer = new StreamWriter({
      dataDir: tempDir,
      clock,
    });

    cleanup.push(async () => {
      writer.close();
      db.close();
      await store.close();
      rmSync(tempDir, { recursive: true, force: true });
    });

    const first = await writer.append({
      kind: "user_msg",
      content: "We reviewed the retrieval boundary.",
    });
    const extractor = new EpisodicExtractor({
      dataDir: tempDir,
      episodicRepository: repo,
      embeddingClient: new TitleEmbeddingClient(),
      llmClient: new FakeLLMClient({
        responses: [
          createEpisodeToolResponse([
            {
              title: "Retrieval boundary review",
              narrative: "We reviewed the retrieval boundary and hard audience scoping.",
              source_stream_ids: [first.id],
              participants: ["team"],
              tags: ["retrieval"],
              confidence: 0.8,
              significance: 0.7,
            },
          ]),
          createEpisodeToolResponse([
            {
              title: "Retrieval boundary review",
              narrative: "We reviewed the retrieval boundary and hard audience scoping.",
              source_stream_ids: [first.id],
              participants: ["team"],
              tags: ["retrieval"],
              confidence: 0.8,
              significance: 0.7,
            },
          ]),
        ],
      }),
      model: "claude-haiku",
      entityRepository,
      clock,
    });

    const firstRun = await extractor.extractFromStream();
    const secondRun = await extractor.extractFromStream();

    expect(firstRun).toEqual({
      inserted: 1,
      updated: 0,
      skipped: 0,
    });
    expect(secondRun).toEqual({
      inserted: 0,
      updated: 0,
      skipped: 1,
    });
    expect((await repo.listAll()).map((episode) => episode.source_stream_ids)).toEqual([[first.id]]);
  });

  it("rejects hallucinated source stream ids", async () => {
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
    });
    const entityRepository = new EntityRepository({
      db,
    });
    const writer = new StreamWriter({
      dataDir: tempDir,
    });
    const entry = await writer.append({
      kind: "user_msg",
      content: "hello",
    });
    void entry;

    cleanup.push(async () => {
      writer.close();
      db.close();
      await store.close();
      rmSync(tempDir, { recursive: true, force: true });
    });

    const extractor = new EpisodicExtractor({
      dataDir: tempDir,
      episodicRepository: repo,
      embeddingClient: new TitleEmbeddingClient(),
      llmClient: new FakeLLMClient({
        responses: [
          createEpisodeToolResponse([
            {
              title: "Planning sync",
              narrative: "A grounded narrative.",
              source_stream_ids: ["strm_missingmissing"],
              participants: [],
              tags: [],
              confidence: 0.8,
              significance: 0.8,
            },
          ]),
        ],
      }),
      model: "claude-haiku",
      entityRepository,
    });

    await expect(extractor.extractFromStream()).rejects.toBeInstanceOf(LLMError);
  });

  it("raises a typed error naming the tool when the llm returns bare text", async () => {
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
    });
    const entityRepository = new EntityRepository({
      db,
    });
    const writer = new StreamWriter({
      dataDir: tempDir,
    });

    await writer.append({
      kind: "user_msg",
      content: "hello",
    });

    cleanup.push(async () => {
      writer.close();
      db.close();
      await store.close();
      rmSync(tempDir, { recursive: true, force: true });
    });

    const extractor = new EpisodicExtractor({
      dataDir: tempDir,
      episodicRepository: repo,
      embeddingClient: new TitleEmbeddingClient(),
      llmClient: new FakeLLMClient({
        responses: [
          {
            text: '{"episodes":[]}',
            input_tokens: 1,
            output_tokens: 1,
            stop_reason: "end_turn",
            tool_calls: [],
          },
        ],
      }),
      model: "claude-haiku",
      entityRepository,
    });

    await expect(extractor.extractFromStream()).rejects.toMatchObject({
      code: "EXTRACTOR_OUTPUT_INVALID",
      message: expect.stringContaining(EPISODE_TOOL_NAME),
    });
  });

  it("skips candidates whose embeddings fail and continues the extraction run", async () => {
    const tempDir = mkdtempSync(join(tmpdir(), "borg-"));
    const clock = new ManualClock(1_000);
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
      clock,
    });
    const entityRepository = new EntityRepository({
      db,
      clock,
    });
    const writer = new StreamWriter({
      dataDir: tempDir,
      clock,
    });

    cleanup.push(async () => {
      writer.close();
      db.close();
      await store.close();
      rmSync(tempDir, { recursive: true, force: true });
    });

    const first = await writer.append({
      kind: "user_msg",
      content: "candidate one",
    });
    const second = await writer.append({
      kind: "agent_msg",
      content: "candidate two",
    });
    const extractor = new EpisodicExtractor({
      dataDir: tempDir,
      episodicRepository: repo,
      embeddingClient: new FailingOnceEmbeddingClient(),
      llmClient: new FakeLLMClient({
        responses: [
          createEpisodeToolResponse([
            {
              title: "Skip me",
              narrative: "This candidate will fail embedding.",
              source_stream_ids: [first.id],
              participants: [],
              tags: [],
              confidence: 0.5,
              significance: 0.5,
            },
            {
              title: "Keep me",
              narrative: "This candidate should still be inserted.",
              source_stream_ids: [second.id],
              participants: [],
              tags: ["kept"],
              confidence: 0.9,
              significance: 0.9,
            },
          ]),
        ],
      }),
      model: "claude-haiku",
      entityRepository,
      clock,
    });

    const result = await extractor.extractFromStream();
    const listed = await repo.list();

    expect(result).toEqual({
      inserted: 1,
      updated: 0,
      skipped: 1,
    });
    expect(listed.items).toHaveLength(1);
    expect(listed.items[0]?.title).toBe("Keep me");
  });
});
