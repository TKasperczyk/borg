import { afterEach, describe, expect, it, vi } from "vitest";

import { FakeLLMClient } from "../llm/test-support/fake-client.js";
import { createOfflineTestHarness, type OfflineTestHarness } from "../offline/test-support.js";
import type { TurnTracer } from "../tracing/tracer.js";
import { DEFAULT_SESSION_ID } from "../util/ids.js";
import { RetrievalPipeline } from "./pipeline.js";
import { SELF_RECALL_SCOPE } from "./recall-context.js";
import { expandRecall } from "./recall-expansion.js";
import { DEFAULT_RAW_QUERY_MAX_BYTES } from "./raw-query.js";

vi.mock("./recall-expansion.js", async (importOriginal) => {
  const actual = await importOriginal<typeof import("./recall-expansion.js")>();
  return {
    ...actual,
    expandRecall: vi.fn(async () => ({
      resolved_query: "expanded query",
      semantic_variants: [{ strategy: "combined", query: "expanded query" }],
      named_terms: [],
      typed_queries: [],
      temporalCue: null,
    })),
  };
});

const harnesses: OfflineTestHarness[] = [];
afterEach(async () => {
  for (const harness of harnesses.splice(0)) {
    await harness.cleanup();
  }
  vi.clearAllMocks();
});

async function setup(rawQueryMaxBytes?: number) {
  const harness = await createOfflineTestHarness();
  harnesses.push(harness);
  const embed = vi.spyOn(harness.embeddingClient, "embed");
  const emit = vi.fn<TurnTracer["emit"]>();
  const pipeline = new RetrievalPipeline({
    embeddingClient: harness.embeddingClient,
    llmClient: new FakeLLMClient({ responses: [] }),
    episodicRepository: harness.episodicRepository,
    dataDir: harness.tempDir,
    clock: harness.clock,
    tracer: { enabled: true, includePayloads: false, emit },
    ...(rawQueryMaxBytes === undefined ? {} : { rawQueryMaxBytes }),
  });
  const options = {
    recallContext: {
      reader: SELF_RECALL_SCOPE,
      currentSessionId: DEFAULT_SESSION_ID,
      currentAudienceEntityId: null,
      currentParticipantEntityIds: [],
    },
    traceTurnId: "raw-query-test",
    scoringFeatures: { goalVectors: [], valueVectors: [] },
  };
  return { pipeline, options, embed, emit };
}

describe("raw-query recall integration", () => {
  it("embeds short queries unchanged and traces that they were not bounded", async () => {
    const { pipeline, options, embed, emit } = await setup();
    const query = "  Zażółć 中文 🦉\n";
    await pipeline.recallEpisodesForCognition(query, options);
    expect(embed).toHaveBeenCalledWith(query);
    expect(emit.mock.calls.filter(([event]) => event === "retrieval.raw_query")).toEqual([
      [
        "retrieval.raw_query",
        {
          turnId: "raw-query-test",
          session_id: undefined,
          bounded: false,
          mode: "unchanged",
          original_bytes: Buffer.byteLength(query),
          query_bytes: Buffer.byteLength(query),
          max_bytes: DEFAULT_RAW_QUERY_MAX_BYTES,
          selected_message_count: 0,
        },
      ],
    ]);
  });

  it("bounds only the raw intent while expansion receives the full original focus", async () => {
    const { pipeline, options, embed, emit } = await setup(8);
    const query = "old context".repeat(5_000) + "Zażółć";
    await pipeline.recallEpisodesForCognition(query, options);
    expect(embed).toHaveBeenCalledWith("żółć");
    expect(embed).not.toHaveBeenCalledWith(query);
    expect(embed).toHaveBeenCalledWith("expanded query");
    expect(expandRecall).toHaveBeenCalledWith(expect.objectContaining({ focus: query }));
    const records = emit.mock.calls.filter(([event]) => event === "retrieval.raw_query");
    expect(records).toHaveLength(1);
    expect(records[0]?.[1]).toEqual({
      turnId: "raw-query-test",
      session_id: undefined,
      bounded: true,
      mode: "query_tail",
      original_bytes: Buffer.byteLength(query),
      query_bytes: 8,
      max_bytes: 8,
      selected_message_count: 0,
    });
  });

  it("uses the default cap when no override is supplied", async () => {
    const { pipeline, options, embed } = await setup();
    const query = "x".repeat(DEFAULT_RAW_QUERY_MAX_BYTES + 50) + "tail";
    await pipeline.recallEpisodesForCognition(query, options);
    expect(embed).toHaveBeenCalledWith(query.slice(-DEFAULT_RAW_QUERY_MAX_BYTES));
    expect(embed).not.toHaveBeenCalledWith(query);
  });

  it("embeds newest whole batch messages while expansion sees the full batch", async () => {
    const { pipeline, options, embed, emit } = await setup(13);
    const messages = ["oldest message", "older", "newest"];
    const query = messages.join("\n\n");
    await pipeline.recallEpisodesForCognition(query, { ...options, rawQueryMessages: messages });
    expect(embed).toHaveBeenCalledWith("newest\n\nolder");
    expect(expandRecall).toHaveBeenCalledWith(expect.objectContaining({ focus: query }));
    expect(emit).toHaveBeenCalledWith(
      "retrieval.raw_query",
      expect.objectContaining({
        bounded: true,
        mode: "messages",
        query_bytes: 13,
        selected_message_count: 2,
      }),
    );
  });

  it("records bounding once across the disclosure time-range fallback", async () => {
    const { pipeline, embed, emit } = await setup(4);
    const result = await pipeline.searchEpisodesForDisclosureWithTimeRangeFallback("long tail", {
      timeRange: { start: 1_000, end: 2_000 },
      traceTurnId: "raw-query-test",
      scoringFeatures: { goalVectors: [], valueVectors: [] },
    });
    expect(result.timeRangeFallback).toBe(true);
    expect(embed).toHaveBeenCalledWith("tail");
    expect(emit.mock.calls.filter(([event]) => event === "retrieval.raw_query")).toHaveLength(1);
    expect(expandRecall).toHaveBeenCalledTimes(1);
  });
});
