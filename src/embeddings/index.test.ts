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
