import OpenAI from "openai";
import { afterEach, describe, expect, it, vi } from "vitest";
import { EmbeddingBackoffError, embeddingRetryAt } from "./request-retry.js";
import { OpenAICompatibleEmbeddingClient } from "./index.js";

describe("embedding request retry policy", () => {
  afterEach(() => vi.useRealTimers());

  it.each([
    [{ "Retry-After": "1.25" }, 2250],
    [{ "retry-after-ms": "250", "retry-after": "5" }, 1250],
    [new Headers({ "retry-after": "Thu, 01 Jan 1970 00:00:05 GMT" }), 5000],
    [{ "retry-after": "599" }, 600_000],
    [{ "retry-after-ms": "599000" }, 600_000],
    [{ "retry-after": "Thu, 01 Jan 1970 00:10:00 GMT" }, 600_000],
    [{ "retry-after": "Thu, 01 Jan 1970 00:00:00 GMT" }, 1000],
    [{ "retry-after": "-1" }, null],
    [{ "retry-after": "invalid" }, null],
    [{ "retry-after": "Infinity" }, null],
    [{ "retry-after-ms": "bad", "retry-after": "2" }, 3000],
  ])("parses provider headers %j without shortening the wait", (headers, expected) => {
    expect(embeddingRetryAt(headers, 1000)).toBe(expected);
  });

  it.each([
    { "retry-after": "86400" },
    { "retry-after-ms": "86400000" },
    { "retry-after": "Fri, 02 Jan 1970 00:00:01 GMT" },
  ])("caps a one-day provider backoff at ten minutes: %j", (headers) => {
    expect(embeddingRetryAt(headers, 1000)).toBe(601_000);
  });

  it("owns SDK retries and waits for retry-after on the failing request only", async () => {
    vi.useFakeTimers();
    vi.setSystemTime(1000);
    const fetch = vi
      .fn()
      .mockResolvedValueOnce(
        new Response(JSON.stringify({ error: { message: "tokens quota" } }), {
          status: 429,
          headers: { "retry-after": "2.5" },
        }),
      )
      .mockResolvedValueOnce(
        new Response(JSON.stringify({ data: [{ index: 0, embedding: [1] }] }), {
          headers: { "content-type": "application/json" },
        }),
      );
    const client = new OpenAICompatibleEmbeddingClient({
      model: "test",
      dims: 1,
      client: new OpenAI({
        apiKey: "test",
        baseURL: "https://embeddings.invalid/v1",
        maxRetries: 9,
        fetch,
      }),
    });
    const result = client.embed("a");
    // Drain SDK response parsing before advancing its advertised wait.
    await vi.advanceTimersByTimeAsync(0);
    expect(client.requestState()).toMatchObject({
      retryAt: 3500,
      inFlight: 1,
      oldestRequestStartedAt: 1000,
    });
    await vi.advanceTimersByTimeAsync(2499);
    expect(fetch).toHaveBeenCalledTimes(1);
    await vi.advanceTimersByTimeAsync(1);
    await expect(result).resolves.toEqual(Float32Array.of(1));
    expect(fetch).toHaveBeenCalledTimes(2);
    expect(client.requestState()).toEqual({
      retryAt: null,
      inFlight: 0,
      oldestRequestStartedAt: null,
    });
    expect(vi.getTimerCount()).toBe(0);
  });

  it("bounds 429 retries, keeps retry-at across calls, and refuses waits beyond the request deadline", async () => {
    vi.useFakeTimers();
    vi.setSystemTime(1000);
    const create = vi.fn().mockRejectedValue({ status: 429 });
    const client = new OpenAICompatibleEmbeddingClient({
      model: "test",
      dims: 1,
      client: { embeddings: { create } },
      rateLimitBaseDelayMs: 100,
    });
    const result = client.embed("a");
    const rejected = expect(result).rejects.toMatchObject({
      retryAt: 1700,
      code: "EMBEDDING_BACKOFF",
    });
    await vi.advanceTimersByTimeAsync(300);
    await rejected;
    expect(create).toHaveBeenCalledTimes(3);
    expect(create.mock.calls.every((call) => call[1].maxRetries === 0)).toBe(true);
    const next = client.embed("a");
    const nextRejected = expect(next).rejects.toBeInstanceOf(EmbeddingBackoffError);
    await vi.advanceTimersByTimeAsync(399);
    expect(create).toHaveBeenCalledTimes(3);
    await vi.runAllTimersAsync();
    await nextRejected;

    const longWait = new OpenAICompatibleEmbeddingClient({
      model: "test",
      dims: 1,
      requestTimeoutMs: 50,
      client: {
        embeddings: {
          create: vi.fn().mockRejectedValue({ status: 429, headers: { "retry-after": "120" } }),
        },
      },
    });
    await expect(longWait.embed("b")).rejects.toBeInstanceOf(EmbeddingBackoffError);
    expect(longWait.requestState().retryAt).toBeGreaterThan(Date.now() + 50);
  });

  it("retains the final 503 Retry-After after retries are exhausted and delays the next call", async () => {
    vi.useFakeTimers();
    vi.setSystemTime(1000);
    const create = vi
      .fn()
      .mockRejectedValueOnce({ status: 503 })
      .mockRejectedValueOnce({ status: 503 })
      .mockRejectedValueOnce({ status: 503, headers: { "retry-after": "2" } })
      .mockResolvedValue({ data: [{ index: 0, embedding: [1] }] });
    const client = new OpenAICompatibleEmbeddingClient({
      model: "test",
      dims: 1,
      rateLimitBaseDelayMs: 100,
      client: { embeddings: { create } },
    });
    const rejected = expect(client.embed("a")).rejects.toMatchObject({
      code: "EMBEDDING_BACKOFF",
      retryAt: 3300,
    });
    await vi.advanceTimersByTimeAsync(300);
    await rejected;
    expect(create).toHaveBeenCalledTimes(3);
    expect(client.requestState().retryAt).toBe(3300);
    const next = client.embed("b");
    await vi.advanceTimersByTimeAsync(1999);
    expect(create).toHaveBeenCalledTimes(3);
    await vi.advanceTimersByTimeAsync(1);
    await expect(next).resolves.toEqual(Float32Array.of(1));
    expect(create).toHaveBeenCalledTimes(4);
  });

  it("honors model-reload Retry-After and retains the last header when that ladder is exhausted", async () => {
    vi.useFakeTimers();
    vi.setSystemTime(1000);
    const create = vi
      .fn()
      .mockRejectedValueOnce({
        status: 400,
        message: "Model has unloaded or crashed.",
        headers: { "retry-after": "2" },
      })
      .mockRejectedValueOnce({
        status: 400,
        message: "Model has unloaded or crashed.",
        headers: { "retry-after": "3" },
      })
      .mockResolvedValue({ data: [{ index: 0, embedding: [1] }] });
    const client = new OpenAICompatibleEmbeddingClient({
      model: "test",
      dims: 1,
      modelReloadRetryDelaysMs: [10],
      client: { embeddings: { create } },
    });
    const rejected = expect(client.embed("a")).rejects.toMatchObject({
      code: "EMBEDDING_BACKOFF",
      retryAt: 6000,
    });
    await vi.advanceTimersByTimeAsync(1999);
    expect(create).toHaveBeenCalledTimes(1);
    await vi.advanceTimersByTimeAsync(1);
    await rejected;
    expect(client.requestState().retryAt).toBe(6000);
    const next = client.embed("b");
    await vi.advanceTimersByTimeAsync(2999);
    expect(create).toHaveBeenCalledTimes(2);
    await vi.advanceTimersByTimeAsync(1);
    await expect(next).resolves.toEqual(Float32Array.of(1));
    expect(create).toHaveBeenCalledTimes(3);
  });

  it("recovers SDK connection failures without enabling SDK retries", async () => {
    vi.useFakeTimers();
    const fetch = vi
      .fn()
      .mockRejectedValueOnce(new TypeError("connection reset"))
      .mockResolvedValueOnce(
        new Response(JSON.stringify({ data: [{ index: 0, embedding: [1] }] }), {
          headers: { "content-type": "application/json" },
        }),
      );
    const client = new OpenAICompatibleEmbeddingClient({
      model: "test",
      dims: 1,
      rateLimitBaseDelayMs: 100,
      client: new OpenAI({
        apiKey: "test",
        baseURL: "https://embeddings.invalid/v1",
        maxRetries: 9,
        fetch,
      }),
    });
    const result = client.embed("a");
    await vi.advanceTimersByTimeAsync(99);
    expect(fetch).toHaveBeenCalledTimes(1);
    await vi.advanceTimersByTimeAsync(1);
    await expect(result).resolves.toEqual(Float32Array.of(1));
    expect(fetch).toHaveBeenCalledTimes(2);
    expect(vi.getTimerCount()).toBe(0);
  });

  it.each([0, 2])(
    "bounds typed connection recovery to %s configured retries",
    async (rateLimitMaxRetries) => {
      vi.useFakeTimers();
      const failure = new OpenAI.APIConnectionError({ cause: new Error("reset") });
      const create = vi.fn().mockRejectedValue(failure);
      const client = new OpenAICompatibleEmbeddingClient({
        model: "test",
        dims: 1,
        rateLimitBaseDelayMs: 100,
        rateLimitMaxRetries,
        client: { embeddings: { create } },
      });
      const rejected = expect(client.embed("a")).rejects.toMatchObject({ cause: failure });
      await vi.advanceTimersByTimeAsync((2 ** rateLimitMaxRetries - 1) * 100);
      await rejected;
      expect(create).toHaveBeenCalledTimes(1 + rateLimitMaxRetries);
      expect(client.requestState().inFlight).toBe(0);
      expect(vi.getTimerCount()).toBe(0);
    },
  );

  it.each(["cancellation", "deadline"] as const)(
    "stops connection recovery at the request %s",
    async (stop) => {
      vi.useFakeTimers();
      const create = vi
        .fn()
        .mockRejectedValue(new OpenAI.APIConnectionError({ cause: new Error("reset") }));
      const client = new OpenAICompatibleEmbeddingClient({
        model: "test",
        dims: 1,
        rateLimitBaseDelayMs: 100,
        requestTimeoutMs: 50,
        client: { embeddings: { create } },
      });
      const controller = new AbortController();
      const cancelled = new Error("caller cancelled");
      const rejected = expect(
        client.embed("a", { signal: controller.signal }),
      ).rejects.toMatchObject(
        stop === "deadline" ? { code: "EMBEDDING_REQUEST_TIMEOUT" } : { cause: cancelled },
      );
      await vi.advanceTimersByTimeAsync(stop === "deadline" ? 50 : 20);
      if (stop === "cancellation") controller.abort(cancelled);
      await rejected;
      expect(create).toHaveBeenCalledTimes(1);
      expect(client.requestState().inFlight).toBe(0);
      expect(vi.getTimerCount()).toBe(0);
    },
  );

  it.each([new TypeError("adapter bug"), new OpenAI.APIUserAbortError()])(
    "does not retry untyped failures or provider cancellation: %s",
    async (failure) => {
      const create = vi.fn().mockRejectedValue(failure);
      const client = new OpenAICompatibleEmbeddingClient({
        model: "test",
        dims: 1,
        client: { embeddings: { create } },
      });
      await expect(client.embed("a")).rejects.toMatchObject({ cause: failure });
      expect(create).toHaveBeenCalledTimes(1);
    },
  );

  it("caps a transport which ignores abort and publishes no invalid chunks", async () => {
    vi.useFakeTimers();
    const stalled = new OpenAICompatibleEmbeddingClient({
      model: "test",
      dims: 1,
      requestTimeoutMs: 20,
      client: { embeddings: { create: async () => await new Promise<never>(() => {}) } },
    });
    const result = stalled.embed("a");
    const rejected = expect(result).rejects.toMatchObject({ code: "EMBEDDING_REQUEST_TIMEOUT" });
    await vi.advanceTimersByTimeAsync(20);
    await rejected;
    expect(stalled.requestState().inFlight).toBe(0);
    const onChunk = vi.fn();
    const invalid = new OpenAICompatibleEmbeddingClient({
      model: "test",
      dims: 1,
      client: {
        embeddings: {
          create: async () => ({
            data: [
              { index: 0, embedding: [1] },
              { index: 0, embedding: [2] },
            ],
          }),
        },
      },
    });
    await expect(invalid.embedBatch(["a", "b"], { onChunk })).rejects.toThrow("indices");
    expect(onChunk).not.toHaveBeenCalled();
  });
});
