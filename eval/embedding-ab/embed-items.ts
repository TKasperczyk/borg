import type { VectorCache } from "./cache.js";
import { isTimeoutError, summarizeError, type ModelEmbeddingRuntime } from "./gateway.js";
import type { EmbeddingCoverage, EmbeddingPurpose } from "./types.js";

export type EmbeddingItem = {
  key: string;
  text: string;
};

export type EmbeddedItems = {
  vectors: Map<string, Float32Array>;
  coverage: EmbeddingCoverage;
};

type MissingText = {
  text: string;
  keys: string[];
};

export async function embedItems(input: {
  items: readonly EmbeddingItem[];
  runtime: ModelEmbeddingRuntime;
  cache: VectorCache;
  purpose: Exclude<EmbeddingPurpose, "dimension_probe">;
  batchSize: number;
  onBatch?: (progress: { completed: number; total: number }) => void;
}): Promise<EmbeddedItems> {
  const vectors = new Map<string, Float32Array>();
  const missingByTextHash = new Map<string, MissingText>();
  let cacheHits = 0;

  for (const item of input.items) {
    const cached = input.cache.get(item.text);
    if (cached !== undefined) {
      vectors.set(item.key, cached);
      cacheHits += 1;
      continue;
    }

    const textHash = input.cache.textHash(item.text);
    const missing = missingByTextHash.get(textHash);
    if (missing === undefined) {
      missingByTextHash.set(textHash, { text: item.text, keys: [item.key] });
    } else {
      missing.keys.push(item.key);
    }
  }

  const uniqueMissing = [...missingByTextHash.values()];
  const failures: EmbeddingCoverage["failures"] = [];
  let embeddedThisRun = 0;

  for (let start = 0; start < uniqueMissing.length; start += input.batchSize) {
    const batch = uniqueMissing.slice(start, start + input.batchSize);
    try {
      const embedded = await input.runtime.transport.withPurpose(input.purpose, () =>
        input.runtime.client.embedBatch(batch.map((entry) => entry.text)),
      );
      if (embedded.length !== batch.length) {
        throw new Error(
          `Embedding batch size mismatch: expected ${batch.length}, received ${embedded.length}`,
        );
      }

      await input.cache.putMany(
        batch.map((entry, index) => {
          const vector = embedded[index];
          if (vector === undefined) {
            throw new Error(`Embedding batch response omitted vector ${index}`);
          }
          return { text: entry.text, vector };
        }),
      );

      for (let index = 0; index < batch.length; index += 1) {
        const entry = batch[index];
        const vector = embedded[index];
        if (entry === undefined || vector === undefined) {
          continue;
        }
        for (const key of entry.keys) {
          vectors.set(key, new Float32Array(vector));
          embeddedThisRun += 1;
        }
      }
    } catch (error) {
      const summary = summarizeError(error);
      const timeout = isTimeoutError(error);
      for (const entry of batch) {
        for (const key of entry.keys) {
          failures.push({ key, error: summary, timeout });
        }
      }
    }

    input.onBatch?.({
      completed: Math.min(start + input.batchSize, uniqueMissing.length),
      total: uniqueMissing.length,
    });
  }

  return {
    vectors,
    coverage: {
      requested: input.items.length,
      available: vectors.size,
      cache_hits: cacheHits,
      cache_misses: input.items.length - cacheHits,
      embedded_this_run: embeddedThisRun,
      failed: failures.length,
      failures,
    },
  };
}
