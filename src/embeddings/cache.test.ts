import { beforeEach, describe, expect, it, vi } from "vitest";

import { createEmbeddingClient } from "../borg/clients.js";
import type { Config } from "../config/index.js";
import type { EmbeddingClient } from "./index.js";
import { createCachingEmbeddingClient, type EmbeddingCacheStats } from "./cache.js";

const openAiMocks = vi.hoisted(() => {
  const createEmbeddings = vi.fn();
  const OpenAI = vi.fn(function OpenAIMock() {
    return {
      embeddings: {
        create: createEmbeddings,
      },
    };
  });

  return { createEmbeddings, OpenAI };
});

vi.mock("openai", () => ({
  default: openAiMocks.OpenAI,
}));

type StatsClient = EmbeddingClient & {
  stats(): EmbeddingCacheStats;
};

function vectorFor(text: string): Float32Array {
  return Float32Array.from([text.charCodeAt(0) ?? 0]);
}

function createInnerClient() {
  const embedBatch = vi.fn(async (texts: readonly string[]) =>
    texts.map((text) => vectorFor(text)),
  );
  const embed = vi.fn(async (text: string) => {
    const [embedding] = await embedBatch([text]);

    if (embedding === undefined) {
      throw new Error("missing embedding");
    }

    return embedding;
  });

  return {
    client: { embed, embedBatch } satisfies EmbeddingClient,
    embed,
    embedBatch,
  };
}

function createDeferred<T>() {
  let resolve: (value: T) => void = () => {};
  let reject: (reason?: unknown) => void = () => {};
  const promise = new Promise<T>((promiseResolve, promiseReject) => {
    resolve = promiseResolve;
    reject = promiseReject;
  });

  return { promise, resolve, reject };
}

describe("createCachingEmbeddingClient", () => {
  beforeEach(() => {
    openAiMocks.createEmbeddings.mockReset();
    openAiMocks.createEmbeddings.mockImplementation(
      async (params: { input: string | string[]; model: string; encoding_format?: "float" }) => {
        const texts = Array.isArray(params.input) ? params.input : [params.input];

        return {
          data: texts.map((text, index) => ({
            index,
            embedding: Array.from(vectorFor(text)),
          })),
        };
      },
    );
  });

  it("serves repeated embed calls from cache", async () => {
    const inner = createInnerClient();
    const client = createCachingEmbeddingClient(inner.client, {
      model: "model-a",
      dims: 1,
    });

    const first = await client.embed("foo");
    const second = await client.embed("foo");

    expect(Array.from(first)).toEqual(Array.from(second));
    expect(inner.embed).toHaveBeenCalledTimes(1);
  });

  it("misses for different texts", async () => {
    const inner = createInnerClient();
    const client = createCachingEmbeddingClient(inner.client, {
      model: "model-a",
      dims: 1,
    });

    await client.embed("foo");
    await client.embed("bar");

    expect(inner.embed).toHaveBeenCalledTimes(2);
  });

  it("shares an in-flight promise for concurrent embed calls", async () => {
    const deferred = createDeferred<Float32Array>();
    const embed = vi.fn(() => deferred.promise);
    const inner = {
      embed,
      embedBatch: vi.fn(async (texts: readonly string[]) => texts.map((text) => vectorFor(text))),
    } satisfies EmbeddingClient;
    const client = createCachingEmbeddingClient(inner, {
      model: "model-a",
      dims: 1,
    });

    const first = client.embed("foo");
    const second = client.embed("foo");
    deferred.resolve(vectorFor("foo"));

    await expect(first).resolves.toEqual(vectorFor("foo"));
    await expect(second).resolves.toEqual(vectorFor("foo"));
    expect(embed).toHaveBeenCalledTimes(1);
  });

  it("evicts rejected promises so retry can call the provider again", async () => {
    const embed = vi
      .fn()
      .mockRejectedValueOnce(new Error("provider unavailable"))
      .mockResolvedValueOnce(vectorFor("foo"));
    const inner = {
      embed,
      embedBatch: vi.fn(async (texts: readonly string[]) => texts.map((text) => vectorFor(text))),
    } satisfies EmbeddingClient;
    const client = createCachingEmbeddingClient(inner, {
      model: "model-a",
      dims: 1,
    });

    await expect(client.embed("foo")).rejects.toThrow("provider unavailable");
    await expect(client.embed("foo")).resolves.toEqual(vectorFor("foo"));
    expect(embed).toHaveBeenCalledTimes(2);
  });

  it("deduplicates batch inputs and preserves output positions", async () => {
    const inner = createInnerClient();
    const client = createCachingEmbeddingClient(inner.client, {
      model: "model-a",
      dims: 1,
    });

    const embeddings = await client.embedBatch(["a", "b", "a", "c", "b"]);

    expect(inner.embedBatch).toHaveBeenCalledTimes(1);
    expect(inner.embedBatch).toHaveBeenCalledWith(["a", "b", "c"]);
    expect(embeddings.map((embedding) => Array.from(embedding))).toEqual([
      [97],
      [98],
      [97],
      [99],
      [98],
    ]);
  });

  it("evicts the least-recently-used entry when the cap is exceeded", async () => {
    const inner = createInnerClient();
    const client = createCachingEmbeddingClient(inner.client, {
      model: "model-a",
      dims: 1,
      maxEntries: 2,
    }) as StatsClient;

    await client.embed("a");
    await client.embed("b");
    await client.embed("c");
    await client.embed("a");

    expect(inner.embed).toHaveBeenCalledTimes(4);
    expect(client.stats().cache_evictions).toBe(2);
  });

  it("keeps pending entries during eviction and evicts the oldest resolved entry later", async () => {
    const deferredByText = new Map<string, ReturnType<typeof createDeferred<Float32Array>>>();
    const deferredFor = (text: string) => {
      const existing = deferredByText.get(text);

      if (existing !== undefined) {
        return existing;
      }

      const deferred = createDeferred<Float32Array>();
      deferredByText.set(text, deferred);
      return deferred;
    };
    const embed = vi.fn((text: string) => deferredFor(text).promise);
    const inner = {
      embed,
      embedBatch: vi.fn(async (texts: readonly string[]) => texts.map((text) => vectorFor(text))),
    } satisfies EmbeddingClient;
    const client = createCachingEmbeddingClient(inner, {
      model: "model-a",
      dims: 1,
      maxEntries: 1,
    }) as StatsClient;

    const firstA = client.embed("a");
    const firstB = client.embed("b");
    const secondA = client.embed("a");
    const secondB = client.embed("b");

    expect(embed).toHaveBeenCalledTimes(2);
    expect(client.stats().pending_overflow).toBe(1);

    deferredFor("a").resolve(vectorFor("a"));
    await expect(firstA).resolves.toEqual(vectorFor("a"));
    await expect(secondA).resolves.toEqual(vectorFor("a"));

    const firstC = client.embed("c");
    const secondC = client.embed("c");

    expect(embed).toHaveBeenCalledTimes(3);
    expect(client.stats().cache_evictions).toBe(1);

    const thirdA = client.embed("a");

    expect(embed).toHaveBeenCalledTimes(4);

    deferredFor("b").resolve(vectorFor("b"));
    deferredFor("c").resolve(vectorFor("c"));
    await expect(firstB).resolves.toEqual(vectorFor("b"));
    await expect(secondB).resolves.toEqual(vectorFor("b"));
    await expect(firstC).resolves.toEqual(vectorFor("c"));
    await expect(secondC).resolves.toEqual(vectorFor("c"));
    await expect(thirdA).resolves.toEqual(vectorFor("a"));
  });

  it("allows temporary overflow when every entry is pending", async () => {
    const deferredByText = new Map<string, ReturnType<typeof createDeferred<Float32Array>>>();
    const deferredFor = (text: string) => {
      const existing = deferredByText.get(text);

      if (existing !== undefined) {
        return existing;
      }

      const deferred = createDeferred<Float32Array>();
      deferredByText.set(text, deferred);
      return deferred;
    };
    const embed = vi.fn((text: string) => deferredFor(text).promise);
    const inner = {
      embed,
      embedBatch: vi.fn(async (texts: readonly string[]) => texts.map((text) => vectorFor(text))),
    } satisfies EmbeddingClient;
    const client = createCachingEmbeddingClient(inner, {
      model: "model-a",
      dims: 1,
      maxEntries: 2,
    }) as StatsClient;

    const firstA = client.embed("a");
    const firstB = client.embed("b");
    const firstC = client.embed("c");
    const secondC = client.embed("c");

    expect(embed).toHaveBeenCalledTimes(3);
    expect(client.stats().pending_overflow).toBe(1);

    deferredFor("a").resolve(vectorFor("a"));
    deferredFor("b").resolve(vectorFor("b"));
    deferredFor("c").resolve(vectorFor("c"));
    await expect(firstA).resolves.toEqual(vectorFor("a"));
    await expect(firstB).resolves.toEqual(vectorFor("b"));
    await expect(firstC).resolves.toEqual(vectorFor("c"));
    await expect(secondC).resolves.toEqual(vectorFor("c"));

    const firstD = client.embed("d");
    deferredFor("d").resolve(vectorFor("d"));
    await expect(firstD).resolves.toEqual(vectorFor("d"));

    expect(embed).toHaveBeenCalledTimes(4);
    expect(client.stats().cache_evictions).toBe(2);
  });

  it("drains pending overflow after a pending entry settles", async () => {
    const deferredByText = new Map<string, ReturnType<typeof createDeferred<Float32Array>>>();
    const deferredFor = (text: string) => {
      const existing = deferredByText.get(text);

      if (existing !== undefined) {
        return existing;
      }

      const deferred = createDeferred<Float32Array>();
      deferredByText.set(text, deferred);
      return deferred;
    };
    const embed = vi.fn((text: string) => deferredFor(text).promise);
    const inner = {
      embed,
      embedBatch: vi.fn(async (texts: readonly string[]) => texts.map((text) => vectorFor(text))),
    } satisfies EmbeddingClient;
    const client = createCachingEmbeddingClient(inner, {
      model: "model-a",
      dims: 1,
      maxEntries: 2,
    }) as StatsClient;

    const firstA = client.embed("a");
    const firstB = client.embed("b");
    const firstC = client.embed("c");

    expect(embed).toHaveBeenCalledTimes(3);
    expect(client.stats().pending_overflow).toBe(1);

    deferredFor("a").resolve(vectorFor("a"));
    await expect(firstA).resolves.toEqual(vectorFor("a"));

    expect(client.stats().cache_evictions).toBe(1);

    const secondA = client.embed("a");

    expect(embed).toHaveBeenCalledTimes(4);

    deferredFor("b").resolve(vectorFor("b"));
    deferredFor("c").resolve(vectorFor("c"));
    await expect(firstB).resolves.toEqual(vectorFor("b"));
    await expect(firstC).resolves.toEqual(vectorFor("c"));
    await expect(secondA).resolves.toEqual(vectorFor("a"));

    expect(client.stats().cache_evictions).toBe(2);
  });

  it("drains pending overflow after rejection without poisoning retries", async () => {
    const firstA = createDeferred<Float32Array>();
    const firstB = createDeferred<Float32Array>();
    const embed = vi
      .fn()
      .mockImplementationOnce(() => firstA.promise)
      .mockImplementationOnce(() => firstB.promise)
      .mockResolvedValueOnce(vectorFor("a"));
    const inner = {
      embed,
      embedBatch: vi.fn(async (texts: readonly string[]) => texts.map((text) => vectorFor(text))),
    } satisfies EmbeddingClient;
    const client = createCachingEmbeddingClient(inner, {
      model: "model-a",
      dims: 1,
      maxEntries: 1,
    }) as StatsClient;

    const pendingA = client.embed("a");
    const pendingB = client.embed("b");

    expect(embed).toHaveBeenCalledTimes(2);
    expect(client.stats().pending_overflow).toBe(1);

    firstA.reject(new Error("provider unavailable"));
    await expect(pendingA).rejects.toThrow("provider unavailable");

    const retryA = client.embed("a");

    expect(embed).toHaveBeenCalledTimes(3);
    await expect(retryA).resolves.toEqual(vectorFor("a"));

    firstB.resolve(vectorFor("b"));
    await expect(pendingB).resolves.toEqual(vectorFor("b"));
  });

  it("keeps different model ids in separate cache keys", async () => {
    const inner = createInnerClient();
    const modelA = createCachingEmbeddingClient(inner.client, {
      model: "model-a",
      dims: 1,
    });
    const modelB = createCachingEmbeddingClient(inner.client, {
      model: "model-b",
      dims: 1,
    });

    await modelA.embed("foo");
    await modelB.embed("foo");

    expect(inner.embed).toHaveBeenCalledTimes(2);
  });

  it("keeps cache state per wrapper instance for the same model id", async () => {
    const firstInner = createInnerClient();
    const secondInner = createInnerClient();
    const first = createCachingEmbeddingClient(firstInner.client, {
      model: "model-a",
      dims: 1,
    });
    const second = createCachingEmbeddingClient(secondInner.client, {
      model: "model-a",
      dims: 1,
    });

    await first.embed("foo");
    await second.embed("foo");
    await first.embed("foo");
    await second.embed("foo");

    expect(firstInner.embed).toHaveBeenCalledTimes(1);
    expect(secondInner.embed).toHaveBeenCalledTimes(1);
  });

  it("keeps createEmbeddingClient cache state per factory-created instance", async () => {
    const config = {
      embedding: {
        baseUrl: "http://localhost:1234/v1",
        apiKey: "test-key",
        model: "model-a",
        dims: 1,
      },
    } as Config;
    const first = createEmbeddingClient(config);
    const second = createEmbeddingClient(config);

    await first.embed("foo");
    await second.embed("foo");
    await first.embed("foo");
    await second.embed("foo");

    expect(openAiMocks.createEmbeddings).toHaveBeenCalledTimes(2);
  });
});
