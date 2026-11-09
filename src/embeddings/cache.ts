import { createHash } from "node:crypto";

import { ConfigError } from "../util/errors.js";
import type {
  EmbeddingClient,
  EmbeddingClientStats,
  EmbeddingCallOptions,
  EmbeddingBatchOptions,
} from "./index.js";
import { raceWithSignal } from "../util/deadline.js";
import { assertEmbeddingProfilesMatch } from "./bank-profile.js";

const DEFAULT_MAX_ENTRIES = 5_000;

export type EmbeddingCacheStats = EmbeddingClientStats & {
  cache_hits: number;
  cache_misses: number;
  cache_evictions: number;
  pending_overflow: number;
};

type CachedEmbedding = Promise<Float32Array>;
type CacheEntry = {
  value: CachedEmbedding;
  settled: boolean;
  batch: boolean;
};

function normalizeMaxEntries(maxEntries: number | undefined): number {
  if (maxEntries === undefined) {
    return DEFAULT_MAX_ENTRIES;
  }

  if (!Number.isInteger(maxEntries) || maxEntries <= 0) {
    throw new ConfigError("Embedding cache maxEntries must be a positive integer");
  }

  return maxEntries;
}

// Hash the original input: transport truncation must not merge distinct source texts.
function cacheKey(input: { model: string; dims: number; text: string }): string {
  return createHash("sha256")
    .update(input.model)
    .update("\0")
    .update(String(input.dims))
    .update("\0")
    .update(input.text)
    .digest("hex");
}

class CachingEmbeddingClient implements EmbeddingClient {
  private readonly records = new Map<string, CacheEntry>();
  private readonly maxEntries: number;
  private cacheHits = 0;
  private cacheMisses = 0;
  private cacheEvictions = 0;
  private pendingOverflow = 0;

  constructor(
    private readonly inner: EmbeddingClient,
    private readonly options: { model: string; dims: number; maxEntries?: number },
  ) {
    if (inner.profile)
      assertEmbeddingProfilesMatch(inner.profile, {
        model: options.model,
        dimensions: options.dims,
      });
    this.maxEntries = normalizeMaxEntries(options.maxEntries);
  }

  get profile() {
    return this.inner.profile;
  }

  get maxInputBytes() {
    return this.inner.maxInputBytes;
  }

  requestState() {
    return (
      this.inner.requestState?.() ?? { retryAt: null, inFlight: 0, oldestRequestStartedAt: null }
    );
  }

  async embed(text: string, options: EmbeddingCallOptions = {}): Promise<Float32Array> {
    options.signal?.throwIfAborted();
    const promise = this.getOrCreate(text, () => this.inner.embed(text));
    // A subscriber's cancellation must not cancel work shared by other callers.
    const embedding = await raceWithSignal(() => promise, options.signal);

    return new Float32Array(embedding);
  }

  async embedBatch(
    texts: readonly string[],
    options: EmbeddingBatchOptions = {},
  ): Promise<Float32Array[]> {
    options.signal?.throwIfAborted();
    if (texts.length === 0) {
      return [];
    }

    const uniqueMissingTexts: string[] = [];
    const missingByKey = new Map<string, { text: string; index: number }>();
    const promisesByKey = new Map<string, CachedEmbedding>();

    for (const text of texts) {
      const key = this.keyForText(text);

      if (promisesByKey.has(key) || missingByKey.has(key)) {
        continue;
      }

      const cached = this.getCached(key);

      if (cached !== undefined) {
        promisesByKey.set(key, cached);
        continue;
      }

      if (!missingByKey.has(key)) {
        missingByKey.set(key, { text, index: uniqueMissingTexts.length });
        uniqueMissingTexts.push(text);
      }
    }

    if (uniqueMissingTexts.length > 0) {
      const pending: Array<{
        resolve: (vector: Float32Array) => void;
        reject: (error: unknown) => void;
      }> = [];
      for (const [key, missing] of missingByKey) {
        const promise = new Promise<Float32Array>((resolve, reject) => {
          pending[missing.index] = { resolve, reject };
        });
        promisesByKey.set(key, this.setPending(key, promise, true));
      }
      // Every entry is installed before the adapter can synchronously publish a
      // chunk. These resolvers belong to this batch, never a replacement entry.
      const settleChunk = (start: number, vectors: readonly Float32Array[]) => {
        vectors.forEach((vector, offset) =>
          pending[start + offset]?.resolve(new Float32Array(vector)),
        );
      };
      void (async () => {
        // Publish entries and attach every waiter before invoking an adapter.
        await Promise.resolve();
        try {
          const vectors = await this.inner.embedBatch(uniqueMissingTexts, { onChunk: settleChunk });
          // Also support injected adapters that only implement the original API.
          settleChunk(0, vectors);
          for (let index = vectors.length; index < pending.length; index += 1) {
            pending[index]?.reject(new Error("Embedding batch response was missing an entry"));
          }
        } catch (error) {
          // Settled successes ignore reject; only failed/unattempted inputs leave
          // the cache, including when a later request of this batch fails.
          for (const entry of pending) entry.reject(error);
        }
      })();
    }

    return raceWithSignal(
      () =>
        Promise.all(
          texts.map(async (text, index) => {
            const promise = promisesByKey.get(this.keyForText(text));

            if (promise === undefined) {
              throw new Error("Embedding cache failed to resolve a batch entry");
            }

            const embedding = await promise;
            if (!options.signal?.aborted) options.onChunk?.(index, [new Float32Array(embedding)]);
            return new Float32Array(embedding);
          }),
        ),
      options.signal,
    );
  }

  stats(): EmbeddingCacheStats {
    return {
      cache_hits: this.cacheHits,
      cache_misses: this.cacheMisses,
      cache_evictions: this.cacheEvictions,
      pending_overflow: this.pendingOverflow,
      input_truncations: this.inner.stats?.().input_truncations ?? 0,
    };
  }

  private keyForText(text: string): string {
    return cacheKey({
      model: this.options.model,
      dims: this.options.dims,
      text,
    });
  }

  private getOrCreate(text: string, create: () => CachedEmbedding): CachedEmbedding {
    const key = this.keyForText(text);
    const cached = this.getCached(key, false);

    if (cached !== undefined) {
      return cached;
    }

    let resolve!: (embedding: Float32Array) => void;
    let reject!: (error: unknown) => void;
    const promise = new Promise<Float32Array>((done, fail) => {
      resolve = done;
      reject = fail;
    });
    const pending = this.setPending(key, promise);
    void (async () => {
      try {
        resolve(new Float32Array(await create()));
      } catch (error) {
        reject(error);
      }
    })();
    return pending;
  }

  private getCached(key: string, allowPendingBatch = true): CachedEmbedding | undefined {
    const entry = this.records.get(key);

    // A background batch has a much longer stall budget than a query. A query
    // may supersede its pending cache entry; settled batch vectors are reusable.
    if (entry === undefined || (!allowPendingBatch && entry.batch && !entry.settled)) {
      this.cacheMisses += 1;
      return undefined;
    }

    this.records.delete(key);
    this.records.set(key, entry);
    this.cacheHits += 1;
    return entry.value;
  }

  private setPending(key: string, promise: CachedEmbedding, batch = false): CachedEmbedding {
    let entry: CacheEntry;
    const value = promise.then(
      (embedding) => {
        entry.settled = true;
        this.evictOldest();
        return embedding;
      },
      (error) => {
        entry.settled = true;
        if (this.records.get(key) === entry) {
          this.records.delete(key);
        }

        this.evictOldest();
        throw error;
      },
    );

    entry = {
      settled: false,
      batch,
      value,
    };

    this.records.delete(key);
    this.records.set(key, entry);
    this.evictOldest();
    return entry.value;
  }

  private evictOldest(): void {
    while (this.records.size > this.maxEntries) {
      let evicted = false;

      for (const [key, entry] of this.records) {
        if (!entry.settled) {
          continue;
        }

        this.records.delete(key);
        this.cacheEvictions += 1;
        evicted = true;
        break;
      }

      if (!evicted) {
        this.pendingOverflow += 1;
        break;
      }
    }
  }
}

export function createCachingEmbeddingClient(
  inner: EmbeddingClient,
  options: { model: string; dims: number; maxEntries?: number },
): EmbeddingClient {
  return new CachingEmbeddingClient(inner, options);
}
