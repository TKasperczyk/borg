import { hostname } from "node:os";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import { ManualClock } from "../../util/clock.js";
import { drainMemorySidecar } from "../memory-sidecar-shutdown.js";
import type { ObservatoryConfig } from "./config.js";
import {
  ObservatoryEmitter, emitObservatoryEvent, startObservatoryStage, endObservatoryStage,
  type ObservatoryEvent, type ObservatoryObserver,
} from "./emitter.js";

const config: Extract<ObservatoryConfig, { enabled: true }> = {
  enabled: true, baseUrl: "http://observatory:8080/prefix/", token: "secret",
  flushMs: 250, maxBatch: 100, queueCap: 5000, timeoutMs: 2000,
};
const event: ObservatoryEvent = { tenant: "acme", kind: "inbox.enqueued", phase: "point" };
type WireEvent = ObservatoryEvent & { v: number; id: string; ts: number; source: string; instance: string; data: Record<string, unknown> };

function sentEvents(fetchFn: ReturnType<typeof vi.fn<typeof fetch>>, call = 0): WireEvent[] {
  const body = fetchFn.mock.calls[call]?.[1]?.body;
  if (typeof body !== "string") throw new Error("Expected serialized observatory batch");
  return (JSON.parse(body) as { events: WireEvent[] }).events;
}

describe("ObservatoryEmitter", () => {
  beforeEach(() => vi.useFakeTimers());
  afterEach(() => { vi.useRealTimers(); vi.restoreAllMocks(); });

  it("queues synchronously, uses an unref'd timer, and posts the contract envelope", async () => {
    const fetchFn = vi.fn<typeof fetch>().mockResolvedValue(new Response(null, { status: 202 }));
    const handles: Array<ReturnType<typeof setTimeout>> = [];
    const emitter = new ObservatoryEmitter({
      config, fetchFn, clock: new ManualClock(1234),
      setTimeoutFn: (callback, delayMs) => {
        const handle = setTimeout(callback, delayMs);
        handles.push(handle);
        return handle;
      },
    });
    emitter.emit(event);
    expect(fetchFn).not.toHaveBeenCalled();
    expect(handles[0]?.hasRef()).toBe(false);
    await vi.advanceTimersByTimeAsync(250);
    expect(fetchFn).toHaveBeenCalledWith("http://observatory:8080/prefix/ingest/events", expect.objectContaining({
      method: "POST", redirect: "manual",
      headers: { authorization: "Bearer secret", "content-type": "application/json" },
    }));
    expect(sentEvents(fetchFn)[0]).toMatchObject({
      ...event, v: 1, ts: 1234, id: expect.stringMatching(/^evt_/), source: "borg", instance: hostname(), data: {},
    });
    await emitter.shutdown();
  });

  it("flushes full batches on the next timer and keeps only one request in flight", async () => {
    let resolve!: (response: Response) => void;
    const pending = new Promise<Response>((done) => { resolve = done; });
    const fetchFn = vi.fn<typeof fetch>().mockReturnValueOnce(pending)
      .mockResolvedValue(new Response(null, { status: 202 }));
    const emitter = new ObservatoryEmitter({ config: { ...config, maxBatch: 2 }, fetchFn });
    for (let i = 0; i < 5; i += 1) emitter.emit(event);
    expect(fetchFn).not.toHaveBeenCalled();
    await vi.advanceTimersByTimeAsync(1);
    expect(sentEvents(fetchFn)).toHaveLength(2);
    await vi.advanceTimersByTimeAsync(500);
    expect(fetchFn).toHaveBeenCalledTimes(1);
    resolve(new Response(null, { status: 202 }));
    await vi.advanceTimersByTimeAsync(1);
    expect(fetchFn).toHaveBeenCalledTimes(2);
    await emitter.shutdown();
    expect(sentEvents(fetchFn, 2)).toHaveLength(1);
  });

  it.each([400, 401, 413, 429, 302])("does not retry HTTP %s", async (status) => {
    const fetchFn = vi.fn<typeof fetch>().mockResolvedValue(new Response(null, { status }));
    vi.spyOn(console, "error").mockImplementation(() => {});
    const emitter = new ObservatoryEmitter({ config, fetchFn });
    emitter.emit(event);
    await vi.advanceTimersByTimeAsync(250);
    expect(fetchFn).toHaveBeenCalledTimes(1);
    await emitter.shutdown();
  });

  it.each(["network", "5xx", "timeout"])("retries %s at most once with identical event ids", async (failure) => {
    const fetchFn = vi.fn<typeof fetch>();
    if (failure === "network") fetchFn.mockRejectedValue(new Error("offline"));
    else if (failure === "5xx") fetchFn.mockResolvedValue(new Response(null, { status: 503 }));
    else fetchFn.mockImplementation(async () => new Promise<Response>(() => {}));
    const log = vi.spyOn(console, "error").mockImplementation(() => {});
    const emitter = new ObservatoryEmitter({ config, fetchFn });
    emitter.emit(event);
    await vi.advanceTimersByTimeAsync(4_251);
    expect(fetchFn).toHaveBeenCalledTimes(2);
    expect(sentEvents(fetchFn, 1)[0]?.id).toBe(sentEvents(fetchFn)[0]?.id);
    expect(log).toHaveBeenCalledWith(expect.any(String), { dropped_total: 1 });
    await emitter.shutdown();
  });

  it("bounds the queue and logs cumulative drops at most once a minute without content", async () => {
    const clock = new ManualClock(0);
    const log = vi.spyOn(console, "error").mockImplementation(() => {});
    const fetchFn = vi.fn<typeof fetch>().mockResolvedValue(new Response(null, { status: 202 }));
    const emitter = new ObservatoryEmitter({ config: { ...config, queueCap: 1 }, clock, fetchFn });
    emitter.emit(event);
    for (let i = 0; i < 10; i += 1) emitter.emit({ ...event, data: { text_preview: "private content" } });
    expect(log).toHaveBeenCalledTimes(1);
    clock.advance(60_000);
    emitter.emit(event);
    expect(log).toHaveBeenLastCalledWith(expect.any(String), { dropped_total: 11 });
    expect(JSON.stringify(log.mock.calls)).not.toContain("private content");
    await emitter.shutdown();
    expect(sentEvents(fetchFn)).toHaveLength(1);
  });

  it("clamps nested fields and lists, summarizes oversized data, and snapshots at emit", async () => {
    const fetchFn = vi.fn<typeof fetch>().mockResolvedValue(new Response(null, { status: 202 }));
    const emitter = new ObservatoryEmitter({ config, fetchFn });
    const data = { text_preview: "p".repeat(300), nested: { text: "a".repeat(4100) }, episodes: Array.from({ length: 51 }, (_, i) => i) };
    emitter.emit({ ...event, refs: { entry_ids: Array.from({ length: 51 }, (_, i) => `entry-${i}`) }, data });
    data.text_preview = "mutated";
    emitter.emit({ ...event, data: { chunks: Array.from({ length: 20 }, () => "ż".repeat(4096)) } });
    await emitter.shutdown();
    const [first, second] = sentEvents(fetchFn);
    expect(first?.data.text_preview).toBe("p".repeat(280));
    expect(first?.data.nested).toEqual({ text: `${"a".repeat(4096)}…[+4 chars]` });
    expect(first?.data.episodes).toHaveLength(50);
    expect(first?.refs?.entry_ids).toHaveLength(50);
    expect(first?.data.truncated_lists).toEqual(["refs.entry_ids", "episodes"]);
    expect(second?.data).toMatchObject({ truncated: true, keys: ["chunks"], bytes: expect.any(Number) });
  });

  it("splits batches by UTF-8 body bytes as well as event count", async () => {
    const fetchFn = vi.fn<typeof fetch>().mockResolvedValue(new Response(null, { status: 202 }));
    const emitter = new ObservatoryEmitter({ config, fetchFn });
    for (let i = 0; i < 100; i += 1) emitter.emit({ ...event, data: { chunks: Array.from({ length: 5 }, () => "x".repeat(4000)) } });
    await vi.advanceTimersByTimeAsync(1000);
    expect(fetchFn.mock.calls.length).toBeGreaterThan(1);
    expect(fetchFn.mock.calls.flatMap((_, i) => sentEvents(fetchFn, i))).toHaveLength(100);
    for (const [, init] of fetchFn.mock.calls) expect(Buffer.byteLength(String(init?.body))).toBeLessThanOrEqual(1024 * 1024);
    await emitter.shutdown();
  });

  it("pairs stage ids, measures duration, defaults status and clamps errors", async () => {
    const clock = new ManualClock(10);
    const fetchFn = vi.fn<typeof fetch>().mockResolvedValue(new Response(null, { status: 202 }));
    const emitter = new ObservatoryEmitter({ config, clock, fetchFn });
    const stage = emitter.stage({ tenant: "acme", kind: "recall.request", data: { route: "/memory/context" } });
    clock.advance(30);
    stage.end({ error: new TypeError("failure"), turn_id: "turn_attempt" });
    stage.end();
    await emitter.shutdown();
    const events = sentEvents(fetchFn);
    expect(events).toHaveLength(2);
    expect(events[0]?.span_id).toBe(events[1]?.span_id);
    expect(events[1]).toMatchObject({ phase: "end", duration_ms: 30, status: "error", turn_id: "turn_attempt", data: { route: "/memory/context", error_type: "TypeError", error: "TypeError: failure" } });
  });

  it("flushes once on shutdown, does not retry, and ignores later emissions", async () => {
    vi.spyOn(console, "error").mockImplementation(() => {});
    const fetchFn = vi.fn<typeof fetch>().mockResolvedValue(new Response(null, { status: 503 }));
    const emitter = new ObservatoryEmitter({ config: { ...config, maxBatch: 2 }, fetchFn });
    for (let i = 0; i < 5; i += 1) emitter.emit(event);
    await emitter.shutdown();
    emitter.emit(event);
    await emitter.shutdown();
    await vi.advanceTimersByTimeAsync(1000);
    expect(fetchFn).toHaveBeenCalledTimes(1);
    expect(sentEvents(fetchFn)).toHaveLength(2);
  });

  it.each([60, 100])("aborts a stalled batch within the shared drain deadline after %s ms of HTTP drain", async (httpMs) => {
    vi.setSystemTime(0);
    vi.spyOn(console, "error").mockImplementation(() => {});
    // Deliberately ignore the signal: cancellation must also settle the transport race.
    const fetchFn = vi.fn<typeof fetch>().mockReturnValue(new Promise<Response>(() => {}));
    const emitter = new ObservatoryEmitter({ config, fetchFn });
    emitter.emit(event);
    await vi.advanceTimersByTimeAsync(250);
    const signal = fetchFn.mock.calls[0]?.[1]?.signal;
    emitter.emit(event);
    let settled = false;
    const draining = (async () => {
      const result = await drainMemorySidecar({
        timeoutMs: 100,
        beginShutdown: () => [],
        closeIdleConnections: () => {},
        closeHttp: () => new Promise<void>((resolve) => {
          if (httpMs < 100) setTimeout(resolve, httpMs);
        }),
        shutdownPool: async () => {},
        shutdownObservatory: (options) => emitter.shutdown(options),
      });
      settled = true;
      return result;
    })();

    await vi.advanceTimersByTimeAsync(99);
    expect(settled).toBe(false);
    expect(signal?.aborted).toBe(false);
    await vi.advanceTimersByTimeAsync(1);
    expect(settled).toBe(true);
    expect(signal?.aborted).toBe(true);
    expect((await draining).http.status).toBe(httpMs < 100 ? "completed" : "timed_out");
    expect(Date.now()).toBe(350);
    await emitter.shutdown();
    await vi.advanceTimersByTimeAsync(6000);
    expect(fetchFn).toHaveBeenCalledTimes(1);
    expect(vi.getTimerCount()).toBe(0);
  });

  it("bounds standalone shutdown by its absolute deadline even when fetch ignores abort", async () => {
    vi.setSystemTime(0);
    vi.spyOn(console, "error").mockImplementation(() => {});
    const fetchFn = vi.fn<typeof fetch>().mockReturnValue(new Promise<Response>(() => {}));
    const emitter = new ObservatoryEmitter({ config, fetchFn });
    emitter.emit(event);
    await vi.advanceTimersByTimeAsync(250);
    emitter.emit(event);
    let settled = false;
    const shutdown = (async () => {
      await emitter.shutdown({ deadlineAt: 350 });
      settled = true;
    })();
    await vi.advanceTimersByTimeAsync(99);
    expect(settled).toBe(false);
    await vi.advanceTimersByTimeAsync(1);
    expect(settled).toBe(true);
    expect(fetchFn.mock.calls[0]?.[1]?.signal?.aborted).toBe(true);
    await shutdown;
    expect(Date.now()).toBe(350);
    expect(fetchFn).toHaveBeenCalledTimes(1);
    expect(vi.getTimerCount()).toBe(0);
  });

  it("contains malformed telemetry and throwing injected hooks", async () => {
    vi.spyOn(console, "error").mockImplementation(() => {});
    const fetchFn = vi.fn<typeof fetch>().mockResolvedValue(new Response(null, { status: 202 }));
    const emitter = new ObservatoryEmitter({ config, fetchFn });
    const circular: Record<string, unknown> = {};
    circular.self = circular;
    expect(() => emitter.emit({ ...event, data: circular })).not.toThrow();
    emitter.emit({ ...event, tenant: "" });
    const broken: ObservatoryObserver = { emit() { throw new Error("broken"); }, stage() { throw new Error("broken"); } };
    expect(() => emitObservatoryEvent(broken, event)).not.toThrow();
    expect(startObservatoryStage(broken, { tenant: "acme", kind: "recall.request" })).toBeUndefined();
    expect(() => endObservatoryStage({ spanId: "sp_x", end() { throw new Error("broken"); } })).not.toThrow();
    await emitter.shutdown();
    expect(fetchFn).not.toHaveBeenCalled();
  });
});
