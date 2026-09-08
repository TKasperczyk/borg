import { afterEach, describe, expect, it, vi } from "vitest";
import type { LanceDbMemoryStats, LanceDbTableStats } from "borg";

import { MemorySampler, readMemorySamplerConfig } from "./memory.js";

const usage = { rss: 1000, heapUsed: 100, heapTotal: 200, external: 50, arrayBuffers: 40 };
const activity = { activeTurns: 2, ledgerCacheEntries: 3, bufferedSessions: 1, clients: 0 };
const storage: LanceDbMemoryStats = { cache: null, tables: [], lastOptimization: null };
const table: LanceDbTableStats = {
  table: "episodes",
  fragments: 12,
  smallFragments: 11,
  version: 123,
  versions: 25,
  durationMs: 2,
};

function harness(input: { historyLimit?: number; tableIntervalMs?: number } = {}) {
  const provider = {
    storageMemoryStats: vi.fn(() => storage),
    storageTableStats: vi.fn(async (): Promise<LanceDbTableStats[]> => [table]),
  };
  const log = vi.fn();
  const logger = { error: vi.fn() };
  const tracer = { enabled: true, includePayloads: false, emit: vi.fn() };
  const sampler = new MemorySampler({
    config: {
      intervalMs: 1000,
      tableIntervalMs: input.tableIntervalMs ?? 5000,
      historyLimit: input.historyLimit ?? 3,
    },
    getStorage: () => provider,
    getActivity: () => activity,
    memoryUsage: () => usage,
    uptime: () => 123,
    log,
    logger,
    tracer,
  });
  return { sampler, provider, log, tracer, logger };
}

afterEach(() => {
  vi.useRealTimers();
});

describe("demo memory sampler", () => {
  it("validates configuration, including explicit disabling and bounded history", () => {
    expect(readMemorySamplerConfig({})).toEqual({
      intervalMs: 300000,
      tableIntervalMs: 1800000,
      historyLimit: 288,
    });
    expect(
      readMemorySamplerConfig({
        DEMO_MEMORY_SAMPLE_INTERVAL_MS: "0",
        DEMO_MEMORY_TABLE_INTERVAL_MS: "0",
      }),
    ).toMatchObject({ intervalMs: 0, tableIntervalMs: 0 });
    for (const value of ["-1", "NaN", "1.5", "2147483648"]) {
      expect(() => readMemorySamplerConfig({ DEMO_MEMORY_SAMPLE_INTERVAL_MS: value })).toThrow();
    }
    expect(() => readMemorySamplerConfig({ DEMO_MEMORY_HISTORY_LIMIT: "100000" })).toThrow();
  });

  it("logs and traces exact byte values, samples periodically, bounds history and stops", async () => {
    vi.useFakeTimers();
    vi.setSystemTime(10000);
    const { sampler, log, tracer } = harness({ tableIntervalMs: 0 });
    sampler.start();
    sampler.start();
    await vi.advanceTimersByTimeAsync(4000);
    expect(log).toHaveBeenCalledTimes(5);
    const snapshot = sampler.snapshot();
    expect(snapshot.history.map((sample) => sample.ts)).toEqual([12000, 13000, 14000]);
    expect(snapshot.latest).toMatchObject({
      ...usage,
      ...activity,
      event: "process.memory.sample",
      uptime: 123,
      lanceTables: null,
    });
    expect(JSON.parse(log.mock.calls.at(-1)![0])).toEqual(snapshot.latest);
    expect(tracer.emit).toHaveBeenCalledTimes(5);
    expect(tracer.emit).toHaveBeenLastCalledWith(
      "process.memory.sample",
      expect.objectContaining({ ...usage, ts: 14000 }),
    );
    const unchangedCalls = log.mock.calls.length;
    await sampler.stop();
    await vi.advanceTimersByTimeAsync(5000);
    expect(log).toHaveBeenCalledTimes(unchangedCalls);
    expect(vi.getTimerCount()).toBe(0);
  });

  it("keeps sampling while metadata stalls and never queues another metadata read", async () => {
    vi.useFakeTimers();
    const { sampler, provider, log } = harness();
    let resolve!: (value: LanceDbTableStats[]) => void;
    provider.storageTableStats.mockImplementationOnce(
      () =>
        new Promise((done) => {
          resolve = done;
        }),
    );
    sampler.start();
    await vi.advanceTimersByTimeAsync(10000);
    expect(log).toHaveBeenCalledTimes(11);
    expect(provider.storageTableStats).toHaveBeenCalledOnce();
    expect(sampler.snapshot().latest).toMatchObject({ lanceTables: null, tableStatsPending: true });
    resolve([table]);
    await vi.advanceTimersByTimeAsync(1000);
    expect(sampler.snapshot().latest?.lanceTables?.tables).toEqual([table]);
    await sampler.stop();
  });

  it("uses the slower cadence, sanitizes metadata errors, and GET-style snapshots do no work", async () => {
    vi.useFakeTimers();
    const { sampler, provider, logger, log, tracer } = harness();
    const cause = new Error("native metadata unavailable at /private/lance/_versions/123.manifest");
    provider.storageTableStats.mockRejectedValueOnce(cause);
    sampler.start();
    await vi.advanceTimersByTimeAsync(4000);
    expect(provider.storageTableStats).toHaveBeenCalledOnce();
    expect(sampler.snapshot().latest?.lanceTables).toMatchObject({
      errorCode: "LANCEDB_TABLE_STATS_FAILED",
      tables: [],
    });
    expect(logger.error).toHaveBeenCalledExactlyOnceWith("LanceDB table metadata refresh failed", {
      code: "LANCEDB_TABLE_STATS_FAILED",
      cause,
    });
    expect(JSON.stringify(sampler.snapshot())).not.toContain(cause.message);
    expect(JSON.stringify(log.mock.calls)).not.toContain("/private/lance");
    expect(JSON.stringify(tracer.emit.mock.calls)).not.toContain("/private/lance");
    const calls = provider.storageMemoryStats.mock.calls.length;
    for (let i = 0; i < 100; i++) sampler.snapshot();
    expect(provider.storageMemoryStats).toHaveBeenCalledTimes(calls);
    await vi.advanceTimersByTimeAsync(2000);
    expect(provider.storageTableStats).toHaveBeenCalledTimes(2);
    expect(sampler.snapshot().latest?.lanceTables?.tables).toEqual([table]);
    expect(sampler.snapshot().latest?.lanceTables).not.toHaveProperty("errorCode");
    await sampler.stop();
  });

  it("discards metadata from a Borg handle replaced during reset", async () => {
    vi.useFakeTimers();
    let resolve!: (value: LanceDbTableStats[]) => void;
    const old = {
      storageMemoryStats: () => storage,
      storageTableStats: () =>
        new Promise<LanceDbTableStats[]>((done) => {
          resolve = done;
        }),
    };
    const next = {
      storageMemoryStats: () => storage,
      storageTableStats: vi.fn(async () => [{ ...table, table: "new" }]),
    };
    let provider: typeof old | null = old;
    const sampler = new MemorySampler({
      config: { intervalMs: 1000, tableIntervalMs: 5000, historyLimit: 3 },
      getStorage: () => provider,
      getActivity: () => activity,
      log: vi.fn(),
    });
    sampler.start();
    provider = null;
    await vi.advanceTimersByTimeAsync(1000);
    expect(sampler.snapshot().latest?.lance).toBeNull();
    provider = next;
    resolve([table]);
    await vi.advanceTimersByTimeAsync(2000);
    expect(sampler.snapshot().latest?.lanceTables?.tables[0]?.table).toBe("new");
    await sampler.stop();
  });

  it("keeps process samples when storage stats fail and waits for metadata on stop", async () => {
    vi.useFakeTimers();
    const { sampler, provider, logger, log, tracer } = harness();
    const cause = new Error("native store closing at /private/lance/table");
    provider.storageMemoryStats.mockImplementation(() => {
      throw cause;
    });
    let resolve!: (value: LanceDbTableStats[]) => void;
    provider.storageTableStats.mockImplementation(
      () =>
        new Promise((done) => {
          resolve = done;
        }),
    );
    sampler.start();
    expect(sampler.snapshot().latest).toMatchObject({
      ...usage,
      lance: null,
      storageErrorCode: "LANCEDB_MEMORY_STATS_FAILED",
    });
    expect(logger.error).toHaveBeenCalledExactlyOnceWith("LanceDB memory diagnostics failed", {
      code: "LANCEDB_MEMORY_STATS_FAILED",
      cause,
    });
    expect(JSON.stringify(sampler.snapshot())).not.toContain(cause.message);
    expect(JSON.stringify(log.mock.calls)).not.toContain("/private/lance");
    expect(JSON.stringify(tracer.emit.mock.calls)).not.toContain("/private/lance");
    let stopped = false;
    const stop = (async () => {
      await sampler.stop();
      stopped = true;
    })();
    await vi.advanceTimersByTimeAsync(2000);
    expect(stopped).toBe(false);
    resolve([]);
    await stop;
    expect(stopped).toBe(true);
  });

  it("does no work when disabled", async () => {
    vi.useFakeTimers();
    const getStorage = vi.fn(() => null);
    const sampler = new MemorySampler({
      config: { intervalMs: 0, tableIntervalMs: 0, historyLimit: 3 },
      getStorage,
      getActivity: () => activity,
    });
    sampler.start();
    await vi.advanceTimersByTimeAsync(10000);
    expect(sampler.snapshot().latest).toBeNull();
    expect(getStorage).not.toHaveBeenCalled();
    await sampler.stop();
  });
});
