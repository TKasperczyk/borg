import { createHash } from "node:crypto";

import { ConfigError } from "../util/errors.js";
import type { EmbeddingClient } from "./index.js";

const DEFAULT_MAX_ENTRIES = 5_000;

export type EmbeddingCacheStats = {
  cache_hits: number;
  cache_misses: number;
  cache_evictions: number;
  pending_overflow: number;
};

type CachedEmbedding = Promise<Float32Array>;
type CacheEntry = {
  value: CachedEmbedding;
  settled: boolean;
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
    this.maxEntries = normalizeMaxEntries(options.maxEntries);
  }

  async embed(text: string): Promise<Float32Array> {
    const promise = this.getOrCreate(text, () => this.inner.embed(text));
    const embedding = await promise;

    return new Float32Array(embedding);
  }

  async embedBatch(texts: readonly string[]): Promise<Float32Array[]> {
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
      const batchPromise = this.inner.embedBatch(uniqueMissingTexts);

      for (const [key, missing] of missingByKey) {
        const promise = batchPromise.then((embeddings) => {
          const embedding = embeddings[missing.index];

          if (embedding === undefined) {
            throw new Error("Embedding batch response was missing an entry");
          }

          return new Float32Array(embedding);
        });

        promisesByKey.set(key, this.setPending(key, promise));
      }
    }

    return Promise.all(
      texts.map(async (text) => {
        const promise = promisesByKey.get(this.keyForText(text));

        if (promise === undefined) {
          throw new Error("Embedding cache failed to resolve a batch entry");
        }

        const embedding = await promise;
        return new Float32Array(embedding);
      }),
    );
  }

  stats(): EmbeddingCacheStats {
    return {
      cache_hits: this.cacheHits,
      cache_misses: this.cacheMisses,
      cache_evictions: this.cacheEvictions,
      pending_overflow: this.pendingOverflow,
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
    const cached = this.getCached(key);

    if (cached !== undefined) {
      return cached;
    }

    const promise = create().then((embedding) => new Float32Array(embedding));
    return this.setPending(key, promise);
  }

  private getCached(key: string): CachedEmbedding | undefined {
    const entry = this.records.get(key);

    if (entry === undefined) {
      this.cacheMisses += 1;
      return undefined;
    }

    this.records.delete(key);
    this.records.set(key, entry);
    this.cacheHits += 1;
    return entry.value;
  }

  private setPending(key: string, promise: CachedEmbedding): CachedEmbedding {
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
