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
import { selfMigrations } from "../self/migrations.js";
import { episodicMigrations } from "./migrations.js";
import { EpisodicExtractor } from "./extractor.js";
import { EpisodicRepository, createEpisodesTableSchema } from "./repository.js";

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

  it("extracts episodes from the stream and deduplicates similar candidates", async () => {
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
      content: "We planned the sprint backlog.",
    });
    const llm = new FakeLLMClient({
      responses: [
        {
          text: JSON.stringify({
            episodes: [
              {
                title: "Planning sync",
                narrative:
                  "The team planned the sprint backlog together. They aligned on the first deliverables.",
                source_stream_ids: [first.id],
                participants: ["team"],
                tags: ["planning"],
                confidence: 0.8,
                significance: 0.7,
              },
            ],
          }),
          input_tokens: 10,
          output_tokens: 20,
          stop_reason: "end_turn",
          tool_calls: [],
        },
      ],
    });
    const extractor = new EpisodicExtractor({
      dataDir: tempDir,
      episodicRepository: repo,
      embeddingClient: new TitleEmbeddingClient(),
      llmClient: llm,
      model: "claude-haiku",
      clock,
    });

    const firstRun = await extractor.extractFromStream();
    clock.advance(1_000);
    const second = await writer.append({
      kind: "agent_msg",
      content: "We refined the sprint plan after review.",
    });
    llm.pushResponse({
      text: JSON.stringify({
        episodes: [
          {
            title: "Planning sync",
            narrative:
              "The team refined the sprint plan and captured next actions. The same meeting gained a little more detail.",
            source_stream_ids: [second.id],
            participants: ["team", "pm"],
            tags: ["planning", "review"],
            confidence: 0.9,
            significance: 0.9,
          },
        ],
      }),
      input_tokens: 10,
      output_tokens: 20,
      stop_reason: "end_turn",
      tool_calls: [],
    });
    const secondRun = await extractor.extractFromStream({
      sinceTs: second.timestamp,
    });
    const listed = await repo.list();

    expect(firstRun).toEqual({
      inserted: 1,
      updated: 0,
      skipped: 0,
    });
    expect(secondRun).toEqual({
      inserted: 0,
      updated: 1,
      skipped: 0,
    });
    expect(listed.items).toHaveLength(1);
    expect(listed.items[0]?.source_stream_ids).toEqual(
      expect.arrayContaining([first.id, second.id]),
    );
    expect(listed.items[0]?.tags).toEqual(expect.arrayContaining(["planning", "review"]));
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
          {
            text: JSON.stringify({
              episodes: [
                {
                  title: "Planning sync",
                  narrative: "A grounded narrative.",
                  source_stream_ids: ["strm_missingmissing"],
                  participants: [],
                  tags: [],
                  confidence: 0.8,
                  significance: 0.8,
                },
              ],
            }),
            input_tokens: 10,
            output_tokens: 10,
            stop_reason: "end_turn",
            tool_calls: [],
          },
        ],
      }),
      model: "claude-haiku",
    });

    await expect(extractor.extractFromStream()).rejects.toBeInstanceOf(LLMError);
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
          {
            text: JSON.stringify({
              episodes: [
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
              ],
            }),
            input_tokens: 10,
            output_tokens: 10,
            stop_reason: "end_turn",
            tool_calls: [],
          },
        ],
      }),
      model: "claude-haiku",
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
