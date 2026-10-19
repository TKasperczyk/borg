import OpenAI from "openai";
import { describe, expect, it, vi } from "vitest";

import type { EmbeddingProvider } from "../config/embedding-provider.js";
import { ConfigError, EmbeddingError } from "../util/errors.js";
import { FakeEmbeddingClient, OpenAICompatibleEmbeddingClient } from "./index.js";

describe("embeddings", () => {
  it("rejects duplicate or missing response indices instead of assigning vectors to wrong texts", async () => {
    const client = new OpenAICompatibleEmbeddingClient({
      model: "embed-model",
      dims: 2,
      client: {
        embeddings: {
          create: async () => ({
            data: [
              { index: 0, embedding: [1, 0] },
              { index: 0, embedding: [0, 1] },
            ],
          }),
        },
      },
    });
    await expect(client.embedBatch(["one", "two"])).rejects.toThrow(
      "indices do not uniquely cover",
    );
  });
  it("wraps an OpenAI-compatible embeddings client", async () => {
    const create = vi.fn().mockResolvedValue({
      data: [
        { index: 1, embedding: [4, 5, 6] },
        { index: 0, embedding: [1, 2, 3] },
      ],
    });

    const client = new OpenAICompatibleEmbeddingClient({
      model: "embed-model",
      dims: 3,
      client: {
        embeddings: { create },
      },
    });

    const embeddings = await client.embedBatch(["one", "two"]);

    expect(create).toHaveBeenCalledWith({
      input: ["one", "two"],
      model: "embed-model",
      encoding_format: "float",
    });
    expect(Array.from(embeddings[0] ?? [])).toEqual([1, 2, 3]);
    expect(Array.from(embeddings[1] ?? [])).toEqual([4, 5, 6]);
  });

  it.each<EmbeddingProvider | undefined>([
    undefined,
    {},
    {
      data_collection: "deny",
      ignore: ["SiliconFlow"],
      future_option: { values: [null, true, 2.5, "étiquette"] },
    },
  ])("serializes provider routing through the OpenAI SDK: %j", async (provider) => {
    const bodies: string[] = [];
    const fetch = vi.fn(async (_url: unknown, init?: RequestInit) => {
      bodies.push(String(init?.body));
      return new Response(JSON.stringify({ data: [{ index: 0, embedding: [1, 2] }] }), {
        headers: { "content-type": "application/json" },
      });
    });
    const client = new OpenAICompatibleEmbeddingClient({
      model: "embed-model",
      dims: 2,
      provider,
      client: new OpenAI({
        apiKey: "test-key",
        baseURL: "https://embeddings.invalid/v1",
        maxRetries: 0,
        fetch,
      }),
    });

    await client.embed("hello");

    const originalBody = { input: "hello", model: "embed-model", encoding_format: "float" };
    expect(bodies).toEqual([
      JSON.stringify(provider === undefined ? originalBody : { ...originalBody, provider }),
    ]);
    if (provider === undefined) {
      expect(JSON.parse(bodies[0]!)).not.toHaveProperty("provider");
    } else {
      expect(JSON.parse(bodies[0]!).provider).toEqual(provider);
    }
  });

  it("carries the same provider object through batch chunks and model reload retries", async () => {
    const provider = { data_collection: "deny", ignore: ["SiliconFlow"] };
    const create = vi
      .fn(async (params: { input: string | string[]; provider?: unknown }) => {
        const inputs = Array.isArray(params.input) ? params.input : [params.input];
        return { data: inputs.map((text, index) => ({ index, embedding: [Number(text)] })) };
      })
      .mockRejectedValueOnce(
        Object.assign(new Error("model not loaded"), { status: 404, code: "model_not_found" }),
      );
    const client = new OpenAICompatibleEmbeddingClient({
      model: "embed-model",
      dims: 1,
      maxBatchSize: 2,
      modelReloadRetryDelaysMs: [0],
      provider,
      client: { embeddings: { create } },
    });

    const result = await client.embedBatch(["1", "2", "3"]);

    expect(result.map((vector) => Array.from(vector))).toEqual([[1], [2], [3]]);
    expect(create).toHaveBeenCalledTimes(3);
    for (const [params] of create.mock.calls) expect(params.provider).toBe(provider);
  });

  it("splits an oversized batch into sequential requests, preserving order", async () => {
    // A local inference server returns `400 "Model has unloaded or crashed."`
    // on a large batch, taking the model down for every consumer on the host,
    // so an unbounded caller array must never reach it in one request.
    const seen: string[][] = [];
    // A single-text chunk is sent as a bare string, not a one-element array.
    const create = vi.fn(async (params: { input: string | string[] }) => {
      const inputs = Array.isArray(params.input) ? params.input : [params.input];
      seen.push([...inputs]);
      return {
        data: inputs.map((text, index) => ({ index, embedding: [Number(text)] })),
      };
    });

    const client = new OpenAICompatibleEmbeddingClient({
      model: "embed-model",
      dims: 1,
      maxBatchSize: 2,
      client: { embeddings: { create } } as never,
    });

    const embeddings = await client.embedBatch(["1", "2", "3", "4", "5"]);

    expect(seen).toEqual([["1", "2"], ["3", "4"], ["5"]]);
    expect(embeddings.map((e) => Array.from(e)[0])).toEqual([1, 2, 3, 4, 5]);
  });

  it("sends a batch at or under the cap as a single request", async () => {
    const create = vi.fn(async (params: { input: string | string[] }) => {
      const inputs = Array.isArray(params.input) ? params.input : [params.input];
      return { data: inputs.map((_, index) => ({ index, embedding: [index] })) };
    });

    const client = new OpenAICompatibleEmbeddingClient({
      model: "embed-model",
      dims: 1,
      maxBatchSize: 4,
      client: { embeddings: { create } } as never,
    });

    await client.embedBatch(["a", "b", "c", "d"]);

    expect(create).toHaveBeenCalledTimes(1);
  });

  it("retries when the server reports an evicted model, in any observed error shape", async () => {
    // 404/model_not_found = never loaded. 400 "Model has unloaded or crashed."
    // = evicted after sitting idle. 400 "Failed to load model ... Model does
    // not exist." = a request racing into the eviction-to-JIT-reload window
    // while another consumer's load evicted ours. Each shape was a hard turn
    // failure until recognised here.
    for (const failure of [
      Object.assign(new Error("nope"), { status: 404, code: "model_not_found" }),
      Object.assign(new Error("Model has unloaded or crashed."), { status: 400 }),
      Object.assign(new Error("boom"), {
        status: 400,
        error: { message: "Model has unloaded or crashed." },
      }),
      Object.assign(
        new Error('400 "Failed to load model \\"embed-model\\". Error: Model does not exist."'),
        { status: 400 },
      ),
    ]) {
      const create = vi
        .fn()
        .mockRejectedValueOnce(failure)
        .mockResolvedValue({ data: [{ index: 0, embedding: [1] }] });

      const client = new OpenAICompatibleEmbeddingClient({
        model: "embed-model",
        dims: 1,
        modelReloadRetryDelaysMs: [0],
        client: { embeddings: { create } } as never,
      });

      await expect(client.embed("text")).resolves.toBeInstanceOf(Float32Array);
      expect(create).toHaveBeenCalledTimes(2);
    }
  });

  it("does not retry an ordinary 400 that merely mentions a model", async () => {
    const create = vi
      .fn()
      .mockRejectedValue(
        Object.assign(new Error("Unknown parameter for model embed-model"), { status: 400 }),
      );

    const client = new OpenAICompatibleEmbeddingClient({
      model: "embed-model",
      dims: 1,
      modelReloadRetryDelaysMs: [0],
      client: { embeddings: { create } } as never,
    });

    await expect(client.embed("text")).rejects.toThrow(EmbeddingError);
    expect(create).toHaveBeenCalledTimes(1);
  });

  it("surfaces the underlying cause in the error message", async () => {
    // Every embedding failure previously read as an identical, information-free
    // "Failed to generate embeddings", which is what made a live incident take
    // three attempts to diagnose. The distinguishing fields must reach the log.
    const create = vi.fn().mockRejectedValue(
      Object.assign(new Error("Model has unloaded or crashed."), {
        name: "APIError",
        status: 400,
        code: "model_unloaded",
      }),
    );

    const client = new OpenAICompatibleEmbeddingClient({
      model: "embed-model",
      dims: 1,
      modelReloadRetryDelaysMs: [0],
      client: { embeddings: { create } } as never,
    });

    await expect(client.embed("text")).rejects.toThrow(/status=400/);
    await expect(client.embed("text")).rejects.toThrow(/code=model_unloaded/);
    await expect(client.embed("text")).rejects.toThrow(/unloaded or crashed/);
  });

  it("rejects a non-positive max batch size", () => {
    expect(
      () =>
        new OpenAICompatibleEmbeddingClient({
          model: "embed-model",
          dims: 3,
          maxBatchSize: 0,
          client: { embeddings: { create: vi.fn() } } as never,
        }),
    ).toThrow(ConfigError);
  });

  it("validates configuration and dimensions", async () => {
    expect(
      () =>
        new OpenAICompatibleEmbeddingClient({
          model: "embed-model",
          dims: 0,
          client: {
            embeddings: {
              create: vi.fn(),
            },
          },
        }),
    ).toThrow(ConfigError);

    const client = new OpenAICompatibleEmbeddingClient({
      model: "embed-model",
      dims: 4,
      client: {
        embeddings: {
          create: vi.fn().mockResolvedValue({
            data: [{ index: 0, embedding: [1, 2, 3] }],
          }),
        },
      },
    });

    await expect(client.embed("hello")).rejects.toBeInstanceOf(EmbeddingError);
  });

  it("retries on JIT model_not_found 404 then succeeds", async () => {
    const notLoaded = Object.assign(new Error("404 model_not_found"), {
      status: 404,
      code: "model_not_found",
    });
    const expectedEmbedding = [0.5, 0.25, 0.125];
    const create = vi
      .fn()
      .mockRejectedValueOnce(notLoaded)
      .mockRejectedValueOnce(notLoaded)
      .mockResolvedValueOnce({
        data: [{ index: 0, embedding: expectedEmbedding }],
      });

    const client = new OpenAICompatibleEmbeddingClient({
      model: "embed-model",
      dims: expectedEmbedding.length,
      modelReloadRetryDelaysMs: [0, 0, 0],
      client: { embeddings: { create } },
    });

    const embedding = await client.embed("hello");

    expect(create).toHaveBeenCalledTimes(3);
    expect(Array.from(embedding)).toEqual(expectedEmbedding);
  });

  it("gives up after exhausting retries on persistent model_not_found", async () => {
    const notLoaded = Object.assign(new Error("404 model_not_found"), {
      status: 404,
      code: "model_not_found",
    });
    const create = vi.fn().mockRejectedValue(notLoaded);

    const client = new OpenAICompatibleEmbeddingClient({
      model: "embed-model",
      dims: 3,
      modelReloadRetryDelaysMs: [0, 0],
      client: { embeddings: { create } },
    });

    await expect(client.embed("hello")).rejects.toBeInstanceOf(EmbeddingError);
    expect(create).toHaveBeenCalledTimes(3);
  });

  it("does not retry on non-model_not_found errors", async () => {
    const otherError = Object.assign(new Error("500 internal"), { status: 500 });
    const create = vi.fn().mockRejectedValue(otherError);

    const client = new OpenAICompatibleEmbeddingClient({
      model: "embed-model",
      dims: 3,
      modelReloadRetryDelaysMs: [0, 0, 0],
      client: { embeddings: { create } },
    });

    await expect(client.embed("hello")).rejects.toBeInstanceOf(EmbeddingError);
    expect(create).toHaveBeenCalledTimes(1);
  });

  it("produces deterministic fake embeddings", async () => {
    const client = new FakeEmbeddingClient(4);

    const first = await client.embed("hello");
    const second = await client.embed("hello");
    const different = await client.embed("world");

    expect(Array.from(first)).toEqual(Array.from(second));
    expect(Array.from(first)).not.toEqual(Array.from(different));
  });

  it("sends well-formed strings even when a caller passes a lone surrogate", async () => {
    const create = vi.fn().mockResolvedValue({
      data: [
        { index: 0, embedding: [1, 2, 3] },
        { index: 1, embedding: [4, 5, 6] },
      ],
    });
    const client = new OpenAICompatibleEmbeddingClient({
      model: "embed-model",
      dims: 3,
      client: { embeddings: { create } } as never,
    });

    // The gateway rejects the WHOLE batch for one lone surrogate (prod, 2026-09-09).
    await client.embedBatch(["ok", "### \ud83c"]);

    const sent = create.mock.calls[0]?.[0]?.input as string[];
    expect(sent).toEqual(["ok", "### \ufffd"]);
    expect(sent.every((text) => !/[\uD800-\uDFFF]/u.test(text))).toBe(true);
  });
});
