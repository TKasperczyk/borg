import { afterEach, describe, expect, it, vi } from "vitest";

import { createModelEmbeddingRuntime, InstrumentedEmbeddingTransport } from "./gateway.js";

describe("embedding A/B gateway retry policy", () => {
  afterEach(() => {
    vi.useRealTimers();
  });

  it("disables the embedding client's outer model-reload retry ladder", async () => {
    const providerError = Object.assign(new Error("model not found"), {
      status: 404,
      code: "model_not_found",
    });
    const create = vi.fn().mockRejectedValue(providerError);
    const runtime = await createModelEmbeddingRuntime({
      model: "generative-apis/qwen3-embedding-8b",
      models: [],
      openai: { embeddings: { create } } as never,
      batchSize: 8,
    });

    await expect(runtime.client.embedBatch(["query"])).rejects.toThrow(
      /Failed to generate embeddings/,
    );
    expect(create).toHaveBeenCalledTimes(1);
    expect(runtime.transport.calls.map((call) => call.attempt)).toEqual([1]);
  });

  it("records the instrumented transport's two retries as monotonic attempts", async () => {
    vi.useFakeTimers();
    const providerError = Object.assign(new Error("temporarily unavailable"), { status: 503 });
    const create = vi
      .fn()
      .mockRejectedValueOnce(providerError)
      .mockRejectedValueOnce(providerError)
      .mockResolvedValueOnce({ data: [{ index: 0, embedding: [1, 0] }] });
    const transport = new InstrumentedEmbeddingTransport({ embeddings: { create } } as never);

    const responsePromise = transport.adapter.embeddings.create({
      input: "query",
      model: "embedding-model",
      encoding_format: "float",
    });
    await vi.runAllTimersAsync();

    await expect(responsePromise).resolves.toEqual({
      data: [{ index: 0, embedding: [1, 0] }],
    });
    expect(create).toHaveBeenCalledTimes(3);
    expect(transport.calls.map((call) => call.attempt)).toEqual([1, 2, 3]);
  });

  it("preserves exhausted dimension-probe attempts on the initialization error", async () => {
    vi.useFakeTimers();
    const providerError = Object.assign(new Error("temporarily unavailable"), { status: 503 });
    const create = vi.fn().mockRejectedValue(providerError);

    const runtimePromise = createModelEmbeddingRuntime({
      model: "unknown-embedding-model",
      models: [],
      openai: { embeddings: { create } } as never,
      batchSize: 8,
    });
    const rejection = expect(runtimePromise).rejects.toMatchObject({
      name: "ModelEmbeddingInitializationError",
      model: "unknown-embedding-model",
      calls: [
        expect.objectContaining({ purpose: "dimension_probe", attempt: 1 }),
        expect.objectContaining({ purpose: "dimension_probe", attempt: 2 }),
        expect.objectContaining({ purpose: "dimension_probe", attempt: 3 }),
      ],
    });
    await vi.runAllTimersAsync();

    await rejection;
    expect(create).toHaveBeenCalledTimes(3);
  });
});
