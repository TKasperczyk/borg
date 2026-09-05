import { createHash } from "node:crypto";
import { performance } from "node:perf_hooks";

import { z } from "zod";

import type { EmbeddingClient } from "../../src/embeddings/index.js";
import type {
  LLMClient,
  LLMCompleteOptions,
  LLMCompleteResult,
  LLMConverseOptions,
  LLMConverseResult,
} from "../../src/llm/index.js";

import { JsonlValueCache, VectorCache } from "../embedding-ab/cache.js";
import { summarizeError, type ModelEmbeddingRuntime } from "../embedding-ab/gateway.js";
import type { EmbeddingCallRecord } from "../embedding-ab/types.js";
import type { EmbeddingLogicalCallRecord, PlannerInvocationRecord } from "./types.js";

const llmToolCallSchema = z
  .object({
    id: z.string(),
    name: z.string(),
    input: z.unknown(),
  })
  .strict();

const plannerResponseSchema = z
  .object({
    text: z.string(),
    input_tokens: z.number().nonnegative(),
    output_tokens: z.number().nonnegative(),
    cache_creation_input_tokens: z.number().nonnegative().optional(),
    cache_read_input_tokens: z.number().nonnegative().optional(),
    stop_reason: z.string().nullable(),
    tool_calls: z.array(llmToolCallSchema),
    structured_output: z.unknown().optional(),
  })
  .strict();

export function parseCachedPlannerResponse(value: unknown): LLMCompleteResult {
  return plannerResponseSchema.parse(value) as LLMCompleteResult;
}

function sha256(value: string): string {
  return createHash("sha256").update(value).digest("hex");
}

function plannerRequestIdentity(options: LLMCompleteOptions): unknown {
  return {
    model: options.model,
    system: options.system,
    messages: options.messages,
    tools: options.tools,
    tool_choice: options.tool_choice,
    output_config: options.output_config,
    max_tokens: options.max_tokens,
    temperature: options.temperature,
    thinking: options.thinking,
    effort: options.effort,
    budget: options.budget,
  };
}

export function plannerRequestCacheKey(options: LLMCompleteOptions): string {
  return sha256(JSON.stringify(plannerRequestIdentity(options)));
}

function clonePlannerResponse(response: LLMCompleteResult): LLMCompleteResult {
  return {
    ...response,
    tool_calls: response.tool_calls.map((call) => ({ ...call })),
  };
}

export class ScratchPlannerLlmClient implements LLMClient {
  readonly calls: PlannerInvocationRecord[] = [];
  private readonly uncommittedResponses = new Map<
    number,
    { key: string; response: LLMCompleteResult }
  >();
  private readonly discardWhenSettled = new Set<number>();

  constructor(
    private readonly inner: LLMClient,
    private readonly cache: JsonlValueCache<LLMCompleteResult>,
  ) {}

  async complete(options: LLMCompleteOptions): Promise<LLMCompleteResult> {
    const key = plannerRequestCacheKey(options);
    const started = performance.now();
    const cached = this.cache.get(key);
    const record: PlannerInvocationRecord = {
      sequence: this.calls.length + 1,
      cache_key: key,
      cache_hit: cached !== undefined,
      started_at: new Date().toISOString(),
      latency_ms: null,
      outcome: "pending",
    };
    this.calls.push(record);

    if (cached !== undefined) {
      record.latency_ms = performance.now() - started;
      record.outcome = "success";
      return clonePlannerResponse(cached);
    }

    try {
      const response = await this.inner.complete(options);
      record.latency_ms = performance.now() - started;
      record.outcome = "success";
      if (!this.discardWhenSettled.delete(record.sequence)) {
        this.uncommittedResponses.set(record.sequence, {
          key,
          response: clonePlannerResponse(response),
        });
      }
      return response;
    } catch (error) {
      this.discardWhenSettled.delete(record.sequence);
      record.latency_ms = performance.now() - started;
      record.outcome = "error";
      record.error = summarizeError(error);
      throw error;
    }
  }

  async converse(options: LLMConverseOptions): Promise<LLMConverseResult> {
    return this.inner.converse(options);
  }

  /**
   * Commit only responses that the real recall-expansion parser accepted. A transport-successful
   * response with a missing/invalid tool payload must not poison every later scratch-cache run.
   */
  async settleCallsSince(callIndex: number, cacheAcceptedResponse: boolean): Promise<void> {
    for (const call of this.calls.slice(callIndex)) {
      const uncommitted = this.uncommittedResponses.get(call.sequence);
      if (uncommitted !== undefined) {
        try {
          if (cacheAcceptedResponse) {
            await this.cache.put(uncommitted.key, uncommitted.response);
          }
        } finally {
          this.uncommittedResponses.delete(call.sequence);
        }
      } else if (call.outcome === "pending") {
        // The pipeline's deadline can settle before a transport observes its AbortSignal. If that
        // late request eventually succeeds, discard it instead of retaining or caching it.
        this.discardWhenSettled.add(call.sequence);
      }
    }
  }
}

export class ScratchCachingEmbeddingClient implements EmbeddingClient {
  readonly calls: EmbeddingLogicalCallRecord[] = [];
  private readonly pending = new Map<string, Promise<Float32Array>>();
  private cacheWriteTail: Promise<void> = Promise.resolve();

  constructor(
    private readonly runtime: ModelEmbeddingRuntime,
    private readonly cache: VectorCache,
  ) {}

  get gatewayCalls(): readonly EmbeddingCallRecord[] {
    return this.runtime.transport.calls;
  }

  async embed(text: string): Promise<Float32Array> {
    const started = performance.now();
    const textHash = this.cache.textHash(text);
    const cached = this.cache.get(text);
    const pending = cached === undefined ? this.pending.get(textHash) : undefined;
    const record: EmbeddingLogicalCallRecord = {
      sequence: this.calls.length + 1,
      text_sha256: textHash,
      source:
        cached !== undefined ? "disk_cache" : pending !== undefined ? "pending_cache" : "gateway",
      started_at: new Date().toISOString(),
      latency_ms: null,
      outcome: "pending",
    };
    this.calls.push(record);

    try {
      const vector =
        cached ??
        (pending === undefined ? await this.embedAndCache(textHash, text) : await pending);
      record.latency_ms = performance.now() - started;
      record.outcome = "success";
      return new Float32Array(vector);
    } catch (error) {
      record.latency_ms = performance.now() - started;
      record.outcome = "error";
      record.error = summarizeError(error);
      throw error;
    }
  }

  async embedBatch(texts: readonly string[]): Promise<Float32Array[]> {
    return Promise.all(texts.map((text) => this.embed(text)));
  }

  private async embedAndCache(textHash: string, text: string): Promise<Float32Array> {
    const request = this.runtime.transport
      .withPurpose("real_query", () => this.runtime.client.embed(text))
      .then(async (vector) => {
        await this.enqueueCacheWrite(text, vector);
        return new Float32Array(vector);
      });
    this.pending.set(textHash, request);

    try {
      return await request;
    } finally {
      if (this.pending.get(textHash) === request) {
        this.pending.delete(textHash);
      }
    }
  }

  private async enqueueCacheWrite(text: string, vector: Float32Array): Promise<void> {
    const write = this.cacheWriteTail.then(() => this.cache.putMany([{ text, vector }]));
    this.cacheWriteTail = write.catch(() => undefined);
    await write;
  }
}
