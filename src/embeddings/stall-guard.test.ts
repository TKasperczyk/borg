import { describe, expect, it } from "vitest";

import { ConfigError, EmbeddingError } from "../util/errors.js";

import type { EmbeddingClient } from "./index.js";
import { StallGuardEmbeddingClient } from "./stall-guard.js";

const VECTOR = Float32Array.from([1, 0, 0, 0]);

function never(): Promise<never> {
  return new Promise<never>(() => {});
}

function stallThenSucceed(stalledAttempts: number): EmbeddingClient & { calls: number } {
  const client = {
    calls: 0,
    async embed(): Promise<Float32Array> {
      client.calls += 1;
      return client.calls <= stalledAttempts ? never() : VECTOR;
    },
    async embedBatch(texts: readonly string[]): Promise<Float32Array[]> {
      client.calls += 1;
      return client.calls <= stalledAttempts ? never() : texts.map(() => VECTOR);
    },
  };

  return client;
}

describe("StallGuardEmbeddingClient", () => {
  it("rescues a stalled embed call on the retry attempt", async () => {
    const inner = stallThenSucceed(1);
    const guarded = new StallGuardEmbeddingClient(inner, { timeoutMs: 20 });

    await expect(guarded.embed("query")).resolves.toEqual(VECTOR);
    expect(inner.calls).toBe(2);
  });

  it("rescues a stalled batch call and applies the batch timeout", async () => {
    const inner = stallThenSucceed(1);
    const guarded = new StallGuardEmbeddingClient(inner, {
      timeoutMs: 1,
      batchTimeoutMs: 50,
    });

    // With the single-input timeout (1ms) both attempts would stall out; a
    // multi-input batch must get batchTimeoutMs per attempt instead.
    const stalledFirst = guarded.embedBatch(["a", "b"]);

    await expect(stalledFirst).resolves.toHaveLength(2);
    expect(inner.calls).toBe(2);
  });

  it("proves multi-input batches use the batch deadline, not the single-input one", async () => {
    // Resolves after 40ms: past the 1ms single-input timeout, well inside the
    // batch deadline. With maxRetries 0 the call succeeds only if the batch
    // deadline was actually applied.
    const inner: EmbeddingClient = {
      async embed(): Promise<Float32Array> {
        return VECTOR;
      },
      embedBatch(texts: readonly string[]): Promise<Float32Array[]> {
        return new Promise((resolve) =>
          setTimeout(() => resolve(texts.map(() => VECTOR)), 40),
        );
      },
    };
    const guarded = new StallGuardEmbeddingClient(inner, {
      timeoutMs: 1,
      batchTimeoutMs: 500,
      maxRetries: 0,
    });

    await expect(guarded.embedBatch(["a", "b"])).resolves.toHaveLength(2);
  });

  it("scales the batch deadline with input count", async () => {
    // Base 100ms + 100ms per input: 30 inputs -> 3100ms deadline. An inner
    // call resolving at 200ms would exceed a flat 100ms base but must succeed
    // under the scaled deadline.
    const inner: EmbeddingClient = {
      async embed(): Promise<Float32Array> {
        return VECTOR;
      },
      embedBatch(texts: readonly string[]): Promise<Float32Array[]> {
        return new Promise((resolve) =>
          setTimeout(() => resolve(texts.map(() => VECTOR)), 200),
        );
      },
    };
    const guarded = new StallGuardEmbeddingClient(inner, {
      timeoutMs: 1,
      batchTimeoutMs: 100,
      maxRetries: 0,
    });
    const texts = Array.from({ length: 30 }, (_, index) => `text ${index}`);

    await expect(guarded.embedBatch(texts)).resolves.toHaveLength(30);
  });

  it("fails with EmbeddingError when every attempt stalls", async () => {
    const inner = stallThenSucceed(Number.POSITIVE_INFINITY);
    const guarded = new StallGuardEmbeddingClient(inner, { timeoutMs: 10, maxRetries: 2 });

    await expect(guarded.embed("query")).rejects.toThrow(
      /Embedding call stalled: 3 attempt\(s\) exceeded 10ms/,
    );
    expect(inner.calls).toBe(3);
  });

  it("does not retry non-timeout errors", async () => {
    let calls = 0;
    const inner: EmbeddingClient = {
      async embed(): Promise<Float32Array> {
        calls += 1;
        throw new EmbeddingError("upstream rejected the request");
      },
      async embedBatch(): Promise<Float32Array[]> {
        calls += 1;
        throw new EmbeddingError("upstream rejected the request");
      },
    };
    const guarded = new StallGuardEmbeddingClient(inner, { timeoutMs: 50 });

    await expect(guarded.embed("query")).rejects.toThrow("upstream rejected the request");
    expect(calls).toBe(1);
  });

  it("passes empty batches through without racing a timer", async () => {
    const inner = stallThenSucceed(0);
    const guarded = new StallGuardEmbeddingClient(inner, { timeoutMs: 1 });

    await expect(guarded.embedBatch([])).resolves.toEqual([]);
    expect(inner.calls).toBe(1);
  });

  it("treats a single-element batch with the single-input timeout", async () => {
    const inner = stallThenSucceed(Number.POSITIVE_INFINITY);
    const guarded = new StallGuardEmbeddingClient(inner, {
      timeoutMs: 10,
      batchTimeoutMs: 60_000,
      maxRetries: 0,
    });

    await expect(guarded.embedBatch(["only"])).rejects.toThrow(/exceeded 10ms/);
  });

  it("rejects invalid configuration", () => {
    const inner = stallThenSucceed(0);

    expect(() => new StallGuardEmbeddingClient(inner, { timeoutMs: 0 })).toThrow(ConfigError);
    expect(() => new StallGuardEmbeddingClient(inner, { batchTimeoutMs: -5 })).toThrow(ConfigError);
    expect(() => new StallGuardEmbeddingClient(inner, { maxRetries: 1.5 })).toThrow(ConfigError);
  });
});
