import { mkdtempSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";

import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import type { EmbeddedItems } from "./embed-items.js";
import type { ModelEmbeddingRuntime } from "./gateway.js";
import type { EpisodeDocument } from "./types.js";

const mocks = vi.hoisted(() => ({
  createGatewayLlmClient: vi.fn(),
  createModelEmbeddingRuntime: vi.fn(),
  createOpenAIClient: vi.fn(),
  discoverGatewayModels: vi.fn(),
  embedItems: vi.fn(),
  generateGoldQuestions: vi.fn(),
  loadActiveEpisodeBank: vi.fn(),
}));

vi.mock("./bank.js", () => ({
  loadActiveEpisodeBank: mocks.loadActiveEpisodeBank,
}));

vi.mock("./embed-items.js", () => ({
  embedItems: mocks.embedItems,
}));

vi.mock("./gateway.js", async (importOriginal) => {
  const actual = await importOriginal<typeof import("./gateway.js")>();
  return {
    ...actual,
    createGatewayLlmClient: mocks.createGatewayLlmClient,
    createModelEmbeddingRuntime: mocks.createModelEmbeddingRuntime,
    createOpenAIClient: mocks.createOpenAIClient,
    discoverGatewayModels: mocks.discoverGatewayModels,
  };
});

vi.mock("./llm-tasks.js", async (importOriginal) => {
  const actual = await importOriginal<typeof import("./llm-tasks.js")>();
  return {
    ...actual,
    generateGoldQuestions: mocks.generateGoldQuestions,
  };
});

import { runEmbeddingAbEvaluation } from "./evaluate.js";
import { renderEmbeddingAbReport } from "./report.js";

function episode(id: string): EpisodeDocument {
  return {
    id,
    title: id,
    narrative: `${id} narrative`,
    tags: [],
    embedding_text: `${id}\n${id} narrative\n`,
    embedding_text_sha256: `${id}-sha256`,
  };
}

describe("embedding A/B evaluation", () => {
  let outDir: string;

  beforeEach(() => {
    vi.clearAllMocks();
    outDir = mkdtempSync(join(tmpdir(), "borg-embedding-ab-evaluate-"));
  });

  afterEach(() => {
    rmSync(outDir, { recursive: true, force: true });
  });

  it("uses a common corpus and preserves completed results after an initialization failure", async () => {
    const episodes = [episode("source"), episode("failed-distractor"), episode("shared")];
    mocks.loadActiveEpisodeBank.mockResolvedValue({
      dataDir: "/copied-bank",
      episodes,
      allEpisodeCount: episodes.length,
      sourceEmbeddingDimensions: 2,
      activeCorpusSha256: "active-corpus-hash",
    });
    mocks.createOpenAIClient.mockReturnValue({});
    mocks.discoverGatewayModels.mockResolvedValue([
      { id: "generative-apis/qwen3-235b-a22b-instruct-2507", metadata: {} },
      { id: "left", metadata: {} },
      { id: "broken", metadata: {} },
      { id: "right", metadata: {} },
    ]);
    mocks.createGatewayLlmClient.mockReturnValue({});
    mocks.generateGoldQuestions.mockResolvedValue([
      {
        index: 1,
        source_episode_id: "source",
        question: "Where is the source?",
        cache_hit: false,
      },
    ]);
    mocks.createModelEmbeddingRuntime.mockImplementation(
      async (input: { model: string }): Promise<ModelEmbeddingRuntime> => {
        if (input.model === "broken") {
          throw new Error("dimension probe exhausted");
        }
        return {
          model: input.model,
          dimensions: 2,
          client: {} as never,
          transport: { calls: [] } as never,
        };
      },
    );
    mocks.embedItems.mockImplementation(
      async (input: {
        items: readonly { key: string; text: string }[];
        runtime: ModelEmbeddingRuntime;
        purpose: "episode" | "gold_question" | "real_query";
      }): Promise<EmbeddedItems> => {
        const unavailable =
          input.runtime.model === "right" && input.purpose === "episode"
            ? new Set(["failed-distractor"])
            : new Set<string>();
        const vectors = new Map<string, Float32Array>();
        for (const item of input.items) {
          if (unavailable.has(item.key)) {
            continue;
          }
          const vector =
            input.purpose !== "episode"
              ? new Float32Array([1, 0])
              : item.key === "failed-distractor"
                ? new Float32Array([1, 0])
                : item.key === "source"
                  ? new Float32Array([0.8, 0.2])
                  : new Float32Array([0, 1]);
          vectors.set(item.key, vector);
        }
        const failures = input.items
          .filter((item) => unavailable.has(item.key))
          .map((item) => ({
            key: item.key,
            error: { name: "GatewayError", message: "embedding failed" },
            timeout: false,
          }));
        return {
          vectors,
          coverage: {
            requested: input.items.length,
            available: vectors.size,
            cache_hits: 0,
            cache_misses: input.items.length,
            embedded_this_run: vectors.size,
            failed: failures.length,
            failures,
          },
        };
      },
    );

    const results = await runEmbeddingAbEvaluation({
      dataDir: "/copied-bank",
      models: ["left", "broken", "right"],
      outDir,
      queries: ["find the source"],
      queriesSource: "inline-json",
      goldSize: 1,
      judgeRequested: false,
      batchSize: 8,
      baseUrl: "https://gateway.example/v1",
      apiKey: "test-key",
    });

    expect(results.comparison_corpus).toMatchObject({
      participating_models: ["left", "right"],
      active_bank_episode_count: 3,
      common_episode_count: 2,
      excluded_episode_count: 1,
      episode_ids: ["source", "shared"],
      excluded_episode_ids: ["failed-distractor"],
      coverage_complete: false,
      comparative_metrics_comparable_to_full_bank_recall: false,
    });
    expect(results.models.map((model) => [model.model, model.status])).toEqual([
      ["left", "completed"],
      ["broken", "initialization_failed"],
      ["right", "completed"],
    ]);

    const completed = results.models.filter((model) => model.status === "completed");
    expect(completed).toHaveLength(2);
    for (const model of completed) {
      expect(model.gold.per_question[0]?.rank).toBe(1);
      expect(model.real_queries[0]?.top_10.map((candidate) => candidate.episode_id)).toEqual([
        "source",
        "shared",
      ]);
    }
    expect(results.replay_comparisons).toHaveLength(1);

    const failed = results.models.find((model) => model.status === "initialization_failed");
    expect(failed?.initialization_error.message).toContain("dimension probe exhausted");

    const report = renderEmbeddingAbReport(results);
    expect(report).toContain("Common candidate corpus | 2/3 active episodes");
    expect(report).toContain("not comparable to full-bank recall");
    expect(report).toContain("Partial run: 1 model initialization failure");
  });
});
