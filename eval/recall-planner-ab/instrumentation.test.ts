import { mkdtempSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";

import { afterEach, describe, expect, it } from "vitest";

import type { LLMCompleteOptions, LLMCompleteResult } from "../../src/llm/index.js";
import { FakeLLMClient } from "../../src/llm/test-support/fake-client.js";
import { classifyLlmStopReason } from "../../src/llm/stop-reason.js";

import { JsonlValueCache } from "../embedding-ab/cache.js";
import {
  ScratchPlannerLlmClient,
  parseCachedPlannerResponse,
  plannerRequestCacheKey,
} from "./instrumentation.js";

const REQUEST: LLMCompleteOptions = {
  model: "fake-planner",
  budget: "recall-expansion",
  system: "Return a plan.",
  messages: [{ role: "user", content: "FOCUS" }],
  tools: [
    {
      name: "EmitRecallQueryPlan",
      description: "Return a plan.",
      inputSchema: { type: "object", properties: {} },
    },
  ],
  tool_choice: { type: "tool", name: "EmitRecallQueryPlan" },
};

function response(resolvedQuery: string): LLMCompleteResult {
  return {
    text: "",
    input_tokens: 1,
    output_tokens: 1,
    stop_reason: "tool_use",
    tool_calls: [
      {
        id: "toolu_plan",
        name: "EmitRecallQueryPlan",
        input: {
          resolved_query: resolvedQuery,
          semantic_variants: [{ strategy: "combined", query: resolvedQuery }],
          named_terms: [],
          typed_queries: [],
        },
      },
    ],
  };
}

describe("planner response instrumentation", () => {
  const directories: string[] = [];

  afterEach(() => {
    for (const directory of directories.splice(0)) {
      rmSync(directory, { recursive: true, force: true });
    }
  });

  it.each([
    undefined,
    classifyLlmStopReason("tool_calls", "openai"),
    classifyLlmStopReason("refusal", "openai"),
  ])("persists only accepted planner responses with classification %j", async (classification) => {
    const outDir = mkdtempSync(join(tmpdir(), "borg-recall-planner-cache-"));
    directories.push(outDir);
    const cachePath = join(outDir, "cache", "planner.jsonl");
    const cache = new JsonlValueCache<LLMCompleteResult>(
      cachePath,
      parseCachedPlannerResponse,
      join(outDir, "cache"),
    );
    const acceptedResponse = {
      ...response("zaakceptowana odpowiedź"),
      ...(classification === undefined ? {} : { stop_reason_classification: classification }),
    };
    const inner = new FakeLLMClient({
      responses: [response("odrzucona odpowiedź"), acceptedResponse],
    });
    const client = new ScratchPlannerLlmClient(inner, cache);
    const key = plannerRequestCacheKey(REQUEST);

    await client.complete(REQUEST);
    await client.settleCallsSince(0, false);
    expect(cache.get(key)).toBeUndefined();

    await client.complete(REQUEST);
    await client.settleCallsSince(1, true);
    expect(cache.get(key)).toEqual(acceptedResponse);

    const reloadedCache = new JsonlValueCache<LLMCompleteResult>(
      cachePath,
      parseCachedPlannerResponse,
      join(outDir, "cache"),
    );
    const unusedInner = new FakeLLMClient();
    const cachedClient = new ScratchPlannerLlmClient(unusedInner, reloadedCache);
    await expect(cachedClient.complete(REQUEST)).resolves.toEqual(acceptedResponse);
    expect(unusedInner.requests).toHaveLength(0);
  });

  it.each([
    { kind: "unsupported", raw: "tool_calls" },
    { kind: "unknown", raw: 42 },
    { kind: "tool_use", raw: "tool_calls", extra: true },
  ])("rejects malformed cached classification metadata: %j", (classification) => {
    expect(() =>
      parseCachedPlannerResponse({
        ...response("plan"),
        stop_reason_classification: classification,
      }),
    ).toThrow();
  });
});
