import { mkdtempSync, rmSync } from "node:fs";
import { join } from "node:path";
import { tmpdir } from "node:os";

import { afterEach, describe, expect, it, vi } from "vitest";

import type { EmbeddingClient } from "./embeddings/index.js";
import { FakeLLMClient } from "./llm/index.js";
import { episodicMigrations } from "./memory/episodic/index.js";
import { EpisodicRepository, createEpisodesTableSchema } from "./memory/episodic/repository.js";
import { selfMigrations } from "./memory/self/index.js";
import { retrievalMigrations } from "./retrieval/index.js";
import { LanceDbStore } from "./storage/lancedb/index.js";
import { openDatabase, SqliteDatabase } from "./storage/sqlite/index.js";
import { ManualClock } from "./util/clock.js";
import { Borg } from "./borg.js";

class ScriptedEmbeddingClient implements EmbeddingClient {
  async embed(text: string): Promise<Float32Array> {
    return this.vector(text);
  }

  async embedBatch(texts: readonly string[]): Promise<Float32Array[]> {
    return texts.map((text) => this.vector(text));
  }

  private vector(text: string): Float32Array {
    if (/Planning sync|planning|Atlas|atlas|pnpm|deploy|rollback/.test(text)) {
      return Float32Array.from([1, 0, 0, 0]);
    }

    return Float32Array.from([0, 1, 0, 0]);
  }
}

describe("Borg", () => {
  const tempDirs: string[] = [];

  afterEach(() => {
    vi.restoreAllMocks();

    while (tempDirs.length > 0) {
      rmSync(tempDirs.pop() as string, { recursive: true, force: true });
    }
  });

  it("opens the sprint 2 facade and reuses injected clients", async () => {
    const tempDir = mkdtempSync(join(tmpdir(), "borg-"));
    tempDirs.push(tempDir);

    const clock = new ManualClock(1_000);
    const llm = new FakeLLMClient();
    const borg = await Borg.open({
      dataDir: tempDir,
      clock,
      embeddingDimensions: 4,
      embeddingClient: new ScriptedEmbeddingClient(),
      llmClient: llm,
    });

    try {
      const entry = await borg.stream.append({
        kind: "user_msg",
        content: "planning kickoff",
      });
      llm.pushResponse({
        text: JSON.stringify({
          episodes: [
            {
              title: "Planning sync",
              narrative:
                "The team aligned on the sprint plan. They captured the first follow-up actions.",
              source_stream_ids: [entry.id],
              participants: ["team"],
              tags: ["planning"],
              confidence: 0.8,
              significance: 0.8,
            },
          ],
        }),
        input_tokens: 1,
        output_tokens: 1,
        stop_reason: "end_turn",
        tool_calls: [],
      });

      const extracted = await borg.episodic.extract({
        sinceTs: entry.timestamp,
      });
      const results = await borg.episodic.search("planning", {
        limit: 1,
      });
      const value = borg.self.values.add({
        label: "clarity",
        description: "Prefer explicit, auditable state.",
        priority: 5,
      });

      expect(extracted.inserted).toBe(1);
      expect(results[0]?.citationChain[0]?.id).toBe(entry.id);
      expect(borg.stream.tail(1)).toHaveLength(1);
      expect(borg.self.values.list()).toEqual([
        expect.objectContaining({
          id: value.id,
        }),
      ]);
    } finally {
      await borg.close();
    }
  });

  it("closes opened resources if a later Borg.open step fails", async () => {
    const tempDir = mkdtempSync(join(tmpdir(), "borg-"));
    tempDirs.push(tempDir);

    const sqliteCloseSpy = vi.spyOn(SqliteDatabase.prototype, "close");
    const lanceCloseSpy = vi.spyOn(LanceDbStore.prototype, "close");
    const failure = new Error("embedding init failed");
    const openOptions = {
      dataDir: tempDir,
    } as {
      dataDir: string;
      embeddingClient?: ScriptedEmbeddingClient;
    };

    Object.defineProperty(openOptions, "embeddingClient", {
      get() {
        throw failure;
      },
    });

    await expect(Borg.open(openOptions)).rejects.toThrow(failure);
    expect(sqliteCloseSpy).toHaveBeenCalledTimes(1);
    expect(lanceCloseSpy).toHaveBeenCalledTimes(1);
  });

  it("runs the full cognitive turn loop", async () => {
    const tempDir = mkdtempSync(join(tmpdir(), "borg-"));
    tempDirs.push(tempDir);

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

    await repo.insert({
      id: "ep_aaaaaaaaaaaaaaaa" as never,
      title: "Atlas release incident",
      narrative: "Atlas release hit a pnpm failure during deploy.",
      participants: ["team"],
      location: null,
      start_time: 0,
      end_time: 1,
      source_stream_ids: ["strm_aaaaaaaaaaaaaaaa" as never],
      significance: 0.8,
      tags: ["atlas", "release"],
      confidence: 0.8,
      lineage: {
        derived_from: [],
        supersedes: [],
      },
      embedding: Float32Array.from([1, 0, 0, 0]),
      created_at: 0,
      updated_at: 0,
    });
    db.close();
    await store.close();

    const llm = new FakeLLMClient({
      responses: [
        {
          text: "Check Atlas release assumptions before answering.",
          input_tokens: 10,
          output_tokens: 5,
          stop_reason: "end_turn",
          tool_calls: [],
        },
        {
          text: "To stabilize the Atlas release, rerun pnpm install. Next step: rerun the deploy.",
          input_tokens: 20,
          output_tokens: 10,
          stop_reason: "end_turn",
          tool_calls: [],
        },
      ],
    });
    const borg = await Borg.open({
      config: {
        dataDir: tempDir,
        perception: {
          useLlmFallback: false,
        },
        embedding: {
          baseUrl: "http://localhost:1234/v1",
          apiKey: "test",
          model: "fake-embed",
          dims: 4,
        },
        anthropic: {
          apiKey: "test",
          models: {
            cognition: "sonnet",
            background: "haiku",
            extraction: "haiku",
          },
        },
      },
      clock,
      embeddingDimensions: 4,
      embeddingClient: new ScriptedEmbeddingClient(),
      llmClient: llm,
    });

    try {
      const goal = borg.self.goals.add({
        description: "stabilize atlas release",
        priority: 5,
      });
      const result = await borg.turn({
        userMessage: "Project Atlas has a pnpm error and this is high stakes.",
        stakes: "high",
      });

      expect(result.mode).toBe("problem_solving");
      expect(result.path).toBe("system_2");
      expect(result.response).toContain("rerun pnpm install");
      expect(result.retrievedEpisodeIds).toEqual(["ep_aaaaaaaaaaaaaaaa"]);
      expect(result.intents[0]?.next_action).toContain("rerun the deploy");
      expect(borg.workmem.load().turn_counter).toBe(1);
      expect(borg.self.goals.list({ status: "active" })[0]?.id).toBe(goal.id);
      expect(borg.self.goals.list({ status: "active" })[0]?.progress_notes).toContain(
        "Heuristic turn progress",
      );
      expect(borg.self.traits.list()[0]?.label).toBe("engaged");
      expect(borg.stream.tail(3).map((entry) => entry.kind)).toEqual([
        "user_msg",
        "thought",
        "agent_msg",
      ]);
    } finally {
      await borg.close();
    }
  });

  it("persists suppression across turns and Borg reopen", async () => {
    const tempDir = mkdtempSync(join(tmpdir(), "borg-"));
    tempDirs.push(tempDir);

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

    await repo.insert({
      id: "ep_aaaaaaaaaaaaaaaa" as never,
      title: "Atlas deploy fix",
      narrative: "Rerun pnpm install to recover the Atlas deploy.",
      participants: ["team"],
      location: null,
      start_time: 0,
      end_time: 1,
      source_stream_ids: ["strm_aaaaaaaaaaaaaaaa" as never],
      significance: 0.9,
      tags: ["atlas", "deploy"],
      confidence: 0.9,
      lineage: {
        derived_from: [],
        supersedes: [],
      },
      embedding: Float32Array.from([1, 0, 0, 0]),
      created_at: 0,
      updated_at: 0,
    });
    await repo.insert({
      id: "ep_bbbbbbbbbbbbbbbb" as never,
      title: "Fallback checklist",
      narrative: "Use the backup recovery checklist if the first fix fails.",
      participants: ["team"],
      location: null,
      start_time: 0,
      end_time: 1,
      source_stream_ids: ["strm_bbbbbbbbbbbbbbbb" as never],
      significance: 0.85,
      tags: ["fallback"],
      confidence: 0.85,
      lineage: {
        derived_from: [],
        supersedes: [],
      },
      embedding: Float32Array.from([1, 0, 0, 0]),
      created_at: 0,
      updated_at: 0,
    });
    db.close();
    await store.close();

    const firstBorg = await Borg.open({
      config: {
        dataDir: tempDir,
        perception: {
          useLlmFallback: false,
        },
        embedding: {
          baseUrl: "http://localhost:1234/v1",
          apiKey: "test",
          model: "fake-embed",
          dims: 4,
        },
        anthropic: {
          apiKey: "test",
          models: {
            cognition: "sonnet",
            background: "haiku",
            extraction: "haiku",
          },
        },
      },
      clock,
      embeddingDimensions: 4,
      embeddingClient: new ScriptedEmbeddingClient(),
      llmClient: new FakeLLMClient({
        responses: [
          {
            text: "Rerun pnpm install for the Atlas deploy.",
            input_tokens: 10,
            output_tokens: 5,
            stop_reason: "end_turn",
            tool_calls: [],
          },
        ],
      }),
    });

    try {
      const firstResult = await firstBorg.turn({
        userMessage: "Atlas deploy failed with pnpm",
      });

      expect(firstResult.retrievedEpisodeIds[0]).toBe("ep_aaaaaaaaaaaaaaaa");
      expect(firstBorg.workmem.load().suppressed).toEqual([
        expect.objectContaining({
          id: "ep_aaaaaaaaaaaaaaaa",
          reason: "already surfaced",
        }),
      ]);
    } finally {
      await firstBorg.close();
    }

    const reopenedBorg = await Borg.open({
      config: {
        dataDir: tempDir,
        perception: {
          useLlmFallback: false,
        },
        embedding: {
          baseUrl: "http://localhost:1234/v1",
          apiKey: "test",
          model: "fake-embed",
          dims: 4,
        },
        anthropic: {
          apiKey: "test",
          models: {
            cognition: "sonnet",
            background: "haiku",
            extraction: "haiku",
          },
        },
      },
      clock,
      embeddingDimensions: 4,
      embeddingClient: new ScriptedEmbeddingClient(),
      llmClient: new FakeLLMClient({
        responses: [
          {
            text: "Use the rollback fallback.",
            input_tokens: 10,
            output_tokens: 5,
            stop_reason: "end_turn",
            tool_calls: [],
          },
        ],
      }),
    });

    try {
      const secondResult = await reopenedBorg.turn({
        userMessage: "Atlas deploy failed with pnpm",
      });

      expect(reopenedBorg.workmem.load().suppressed).toEqual(
        expect.arrayContaining([
          expect.objectContaining({
            id: "ep_aaaaaaaaaaaaaaaa",
          }),
        ]),
      );
      expect(secondResult.retrievedEpisodeIds[0]).toBe("ep_bbbbbbbbbbbbbbbb");
    } finally {
      await reopenedBorg.close();
    }
  });

  it("saves working memory early and logs an internal event when a turn fails", async () => {
    const tempDir = mkdtempSync(join(tmpdir(), "borg-"));
    tempDirs.push(tempDir);

    const clock = new ManualClock(1_000);
    const borg = await Borg.open({
      config: {
        dataDir: tempDir,
        perception: {
          useLlmFallback: false,
        },
        embedding: {
          baseUrl: "http://localhost:1234/v1",
          apiKey: "test",
          model: "fake-embed",
          dims: 4,
        },
        anthropic: {
          apiKey: "test",
          models: {
            cognition: "sonnet",
            background: "haiku",
            extraction: "haiku",
          },
        },
      },
      clock,
      embeddingDimensions: 4,
      embeddingClient: new ScriptedEmbeddingClient(),
      llmClient: new FakeLLMClient({
        responses: [
          {
            text: "Check the deploy state before answering.",
            input_tokens: 10,
            output_tokens: 5,
            stop_reason: "end_turn",
            tool_calls: [],
          },
        ],
      }),
    });

    try {
      await expect(
        borg.turn({
          userMessage: "Atlas deploy failed with pnpm and this is high stakes.",
          stakes: "high",
        }),
      ).rejects.toThrow("FakeLLMClient has no scripted response available");

      expect(borg.workmem.load()).toMatchObject({
        turn_counter: 1,
        mode: "problem_solving",
      });
      expect(borg.stream.tail(2).map((entry) => entry.kind)).toEqual([
        "user_msg",
        "internal_event",
      ]);
    } finally {
      await borg.close();
    }
  });
});
