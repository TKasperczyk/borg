import { beforeEach, describe, expect, it, vi } from "vitest";

import { createEmbeddingClient } from "../borg/clients.js";
import { configSchema, type Config } from "../config/index.js";
import {
  OpenAICompatibleEmbeddingClient,
  type EmbeddingClient,
  type EmbeddingBatchOptions,
} from "./index.js";
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

type StatsClient = Omit<EmbeddingClient, "stats"> & {
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

  it("keeps validated chunks and re-sends only failed and unattempted texts", async () => {
    const texts = Array.from({ length: 70 }, (_, index) => `text-${index}`);
    const create = vi.fn(async (params: { input: string | string[] }) => {
      if (create.mock.calls.length === 2) throw { status: 429, headers: { "retry-after": "0" } };
      return {
        data: (Array.isArray(params.input) ? params.input : [params.input]).map((_, index) => ({
          index,
          embedding: [index],
        })),
      };
    });
    const client = createCachingEmbeddingClient(
      new OpenAICompatibleEmbeddingClient({
        model: "test",
        dims: 1,
        maxBatchSize: 32,
        rateLimitMaxRetries: 0,
        client: { embeddings: { create } },
      }),
      { model: "test", dims: 1 },
    );
    await expect(client.embedBatch(texts)).rejects.toMatchObject({ code: "EMBEDDING_BACKOFF" });
    expect(create).toHaveBeenCalledTimes(2);
    await expect(client.embedBatch([...texts, texts[0]!])).resolves.toHaveLength(71);
    expect(create.mock.calls.map(([params]) => params.input)).toEqual([
      texts.slice(0, 32),
      texts.slice(32, 64),
      texts.slice(32, 64),
      texts.slice(64),
    ]);
  });

  it("installs pending entries before synchronous chunk callbacks and ignores late superseded chunks", async () => {
    const batch = createDeferred<Float32Array[]>();
    let publish!: NonNullable<EmbeddingBatchOptions["onChunk"]>;
    let joined!: Promise<Float32Array[]>;
    const inner: EmbeddingClient = {
      embed: vi.fn(async () => Float32Array.of(42)),
      embedBatch: vi.fn((_texts, options) => {
        publish = options!.onChunk!;
        joined = client.embedBatch(["a"]);
        return batch.promise;
      }),
    };
    const client = createCachingEmbeddingClient(inner, { model: "test", dims: 1 });
    const original = client.embedBatch(["a", "b"]);
    await Promise.resolve();
    await expect(client.embed("a")).resolves.toEqual(Float32Array.of(42));
    publish(0, [Float32Array.of(1)]);
    await expect(joined).resolves.toEqual([Float32Array.of(1)]);
    batch.reject(new Error("second request failed"));
    await expect(original).rejects.toThrow("second request failed");
    await expect(client.embed("a")).resolves.toEqual(Float32Array.of(42));
    expect(inner.embedBatch).toHaveBeenCalledTimes(1);
    expect(inner.embed).toHaveBeenCalledTimes(1);
  });

  it("settles uneven byte-limited chunks and retries only failed or unattempted inputs", async () => {
    const texts = ["a", "bb", "cccc", "dd", "ee", "f", "gggg", "h"];
    const create = vi.fn(async (params: { input: string | string[] }) => {
      if (create.mock.calls.length === 3) throw { status: 429, headers: { "retry-after": "0" } };
      const inputs = Array.isArray(params.input) ? params.input : [params.input];
      return {
        data: inputs.map((text, index) => ({ index, embedding: Array.from(vectorFor(text)) })),
      };
    });
    const client = createCachingEmbeddingClient(
      new OpenAICompatibleEmbeddingClient({
        model: "test",
        dims: 1,
        maxBatchBytes: 5,
        rateLimitMaxRetries: 0,
        client: { embeddings: { create } },
      }),
      { model: "test", dims: 1 },
    );
    const onChunk = vi.fn();

    await expect(client.embedBatch(texts, { onChunk })).rejects.toMatchObject({
      code: "EMBEDDING_BACKOFF",
    });
    expect(onChunk.mock.calls).toEqual([
      [0, [vectorFor("a")]],
      [1, [vectorFor("bb")]],
      [2, [vectorFor("cccc")]],
    ]);
    await expect(client.embedBatch(texts.slice(0, 3))).resolves.toEqual(
      texts.slice(0, 3).map(vectorFor),
    );
    expect(create).toHaveBeenCalledTimes(3);

    const retriedTexts = [...texts, texts[0]!];
    await expect(client.embedBatch(retriedTexts)).resolves.toEqual(retriedTexts.map(vectorFor));
    expect(create.mock.calls.map(([params]) => params.input)).toEqual([
      ["a", "bb"],
      "cccc",
      ["dd", "ee", "f"],
      ["dd", "ee", "f"],
      ["gggg", "h"],
    ]);
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

  it("keeps original-text keys when different inputs share a truncated transport prefix", async () => {
    const create = vi.fn().mockResolvedValue({
      data: [
        { index: 1, embedding: [22] },
        { index: 0, embedding: [11] },
      ],
    });
    const inner = new OpenAICompatibleEmbeddingClient({
      model: "model-a",
      dims: 1,
      maxInputBytes: 6,
      client: { embeddings: { create } },
    });
    const client = createCachingEmbeddingClient(inner, {
      model: "model-a",
      dims: 1,
    }) as StatsClient;
    const texts = ["prefix-one", "prefix-two", "prefix-one"];
    expect(client.maxInputBytes).toBe(6);

    const first = await client.embedBatch(texts);
    first[0]![0] = 99;
    const second = await client.embedBatch(["prefix-two", "prefix-one"]);
    const single = await client.embed("prefix-one");

    expect(create).toHaveBeenCalledExactlyOnceWith(
      {
        input: ["prefix", "prefix"],
        model: "model-a",
        encoding_format: "float",
      },
      { signal: expect.any(AbortSignal), maxRetries: 0, timeout: 120_000 },
    );
    expect(first.map((vector) => Array.from(vector))).toEqual([[99], [22], [11]]);
    expect(second.map((vector) => Array.from(vector))).toEqual([[22], [11]]);
    expect(Array.from(single)).toEqual([11]);
    expect(texts).toEqual(["prefix-one", "prefix-two", "prefix-one"]);
    expect(client.stats()).toMatchObject({ cache_misses: 2, input_truncations: 2 });
  });

  it("reports zero truncations for clients without transport statistics", async () => {
    const inner = createInnerClient();
    const client = createCachingEmbeddingClient(inner.client, {
      model: "model-a",
      dims: 1,
    }) as StatsClient;
    await client.embed("foo");
    expect(client.stats().input_truncations).toBe(0);
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

  it.each(["resolve", "reject"] as const)(
    "lets a query bypass a stalled batch, including after its late %s",
    async (settlement) => {
      const batch = createDeferred<Float32Array[]>();
      const fresh = Float32Array.from([42]);
      const inner = {
        embed: vi.fn(async () => fresh),
        embedBatch: vi.fn(() => batch.promise),
      } satisfies EmbeddingClient;
      const client = createCachingEmbeddingClient(inner, { model: "model-a", dims: 1 });
      const abandoned = client.embedBatch(["focus", "activity"]);
      const settled = abandoned.catch(() => undefined);

      // Resolve neither batch nor its cache entries before this query finishes.
      await expect(client.embed("focus")).resolves.toEqual(fresh);
      expect(inner.embed).toHaveBeenCalledTimes(1);
      if (settlement === "resolve") batch.resolve([vectorFor("old"), vectorFor("activity")]);
      else batch.reject(new Error("late batch failure"));
      await settled;
      await expect(client.embed("focus")).resolves.toEqual(fresh);
      expect(inner.embed).toHaveBeenCalledTimes(1);
    },
  );

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
    expect(inner.embedBatch).toHaveBeenCalledWith(["a", "b", "c"], {
      onChunk: expect.any(Function),
    });
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

  it("carries configured provider routing and input byte cap through the production client factory", async () => {
    const provider = { data_collection: "deny", ignore: ["SiliconFlow"] };
    const config = configSchema.parse({
      embedding: {
        baseUrl: "https://embeddings.invalid/v1",
        apiKey: "test-key",
        model: "model-a",
        dims: 1,
        provider,
        maxInputBytes: 2,
      },
    });

    await createEmbeddingClient(config).embed("foo");

    expect(openAiMocks.createEmbeddings).toHaveBeenCalledExactlyOnceWith(
      {
        input: "fo",
        model: "model-a",
        encoding_format: "float",
        provider,
      },
      { signal: expect.any(AbortSignal), maxRetries: 0, timeout: 120_000 },
    );
  });

  it("carries the configured batch byte budget through the production client factory", async () => {
    const config = configSchema.parse({
      embedding: { model: "model-a", dims: 1, maxBatchBytes: 5 },
    });
    const texts = ["aa", "bbb", "cccc"];

    await expect(createEmbeddingClient(config).embedBatch(texts)).resolves.toEqual(
      texts.map(vectorFor),
    );

    expect(openAiMocks.createEmbeddings.mock.calls.map(([params]) => params.input)).toEqual([
      ["aa", "bbb"],
      "cccc",
    ]);
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
