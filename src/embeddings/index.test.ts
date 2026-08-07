import { describe, expect, it, vi } from "vitest";

import { ConfigError, EmbeddingError } from "../util/errors.js";
import { FakeEmbeddingClient, OpenAICompatibleEmbeddingClient } from "./index.js";

describe("embeddings", () => {
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

  it("retries when the server reports an evicted model, in either error shape", async () => {
    // 404/model_not_found = never loaded. 400 "Model has unloaded or crashed."
    // = evicted after sitting idle, which is what a multi-minute deliberation
    // turn actually produces. Only handling the 404 made the second a hard
    // turn failure.
    for (const failure of [
      Object.assign(new Error("nope"), { status: 404, code: "model_not_found" }),
      Object.assign(new Error("Model has unloaded or crashed."), { status: 400 }),
      Object.assign(new Error("boom"), {
        status: 400,
        error: { message: "Model has unloaded or crashed." },
      }),
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
});
