import { afterEach, describe, expect, it, vi } from "vitest";

import { TRACE_EVENT_TAXONOMY } from "../../tracing/taxonomy.js";
import { CallbackTracer, compositeTracer, type CallbackTraceEntry, type TurnTraceData } from "../../tracing/tracer.js";
import { ManualClock } from "../../util/clock.js";
import type { JsonValue } from "../../util/json-value.js";
import { ObservatoryEmitter, type ObservatoryEvent } from "./emitter.js";
import { ObservatoryTracerSink } from "./tracer.js";

const emitters: ObservatoryEmitter[] = [];
afterEach(async () => {
  while (emitters.length > 0) await emitters.pop()?.shutdown();
  vi.restoreAllMocks();
});

function fixture(includePayloads = false, maxMappings = 10_000, queueCap = 5000) {
  const events: ObservatoryEvent[] = [];
  const clock = new ManualClock(100);
  const emitter = new ObservatoryEmitter({
    config: { enabled: true, baseUrl: "http://observatory", token: "secret", flushMs: 250,
      maxBatch: 500, queueCap, timeoutMs: 2000 },
    clock,
    fetchFn: async (_url, init) => {
      events.push(...(JSON.parse(String(init?.body)) as { events: ObservatoryEvent[] }).events);
      return new Response(null, { status: 202 });
    },
  });
  emitters.push(emitter);
  const sink = new ObservatoryTracerSink({ emitter, clock, includePayloads, maxMappings, mappingTtlMs: 1000 });
  return { events, emitter, clock, sink };
}

describe("ObservatoryTracerSink", () => {
  it("maps every catalogued trace, keeping lane completions in recall.lane", async () => {
    const { sink, events, emitter } = fixture();
    const tracer = sink.tracerFor("acme");
    const catalogue = Object.values(TRACE_EVENT_TAXONOMY).flat();
    for (const event of catalogue) {
      tracer.emit(event, { turnId: "borg-trace", candidate_count: 2, duration_ms: 12, outcome: "fulfilled" });
    }
    await emitter.shutdown();
    expect(events).toHaveLength(catalogue.length);
    catalogue.forEach((event, index) => {
      expect(events[index]).toMatchObject({
        tenant: "acme", kind: event === "retrieval.lane_settled" ? "recall.lane" : `trace.${event}`,
        phase: event === "retrieval.lane_settled" ? "end" : "point",
        data: { trace_turn_id: "borg-trace", candidate_count: 2 },
      });
    });
    expect(events.find((event) => event.kind === "recall.lane")).toMatchObject({ duration_ms: 12, status: "ok" });
  });

  it("resolves tenant-scoped correlations with capacity eviction and TTL pruning", async () => {
    const { sink, events, clock, emitter } = fixture(false, 2);
    sink.register("a", "same-trace", "turn_a");
    sink.register("b", "same-trace", "turn_b");
    sink.tracerFor("a").emit("retrieval.started", { turnId: "same-trace" });
    sink.tracerFor("b").emit("retrieval.started", { turnId: "same-trace" });
    sink.register("c", "new-trace", "turn_c");
    sink.tracerFor("a").emit("retrieval.started", { turnId: "same-trace" });
    clock.advance(1001);
    sink.tracerFor("b").emit("retrieval.started", { turnId: "same-trace" });
    await emitter.shutdown();
    expect(events.map((event) => event.turn_id)).toEqual(["turn_a", "turn_b", undefined, undefined]);
  });

  it("omits gated fields even alongside a payload-enabled underlying tracer", async () => {
    const { sink, events, emitter } = fixture();
    const tracer = compositeTracer([
      new CallbackTracer({ includePayloads: true, sink: () => {} }), sink.tracerFor("a"),
    ]);
    tracer.emit("recall_expansion.completed", { turnId: "t", query: "private", focus: "private", context_turns: [], candidate_count: 3 });
    await emitter.shutdown();
    expect(events[0]?.data).toEqual({ candidate_count: 3, trace_turn_id: "t" });
    const enabled = fixture(true);
    enabled.sink.tracerFor("a").emit("retrieval.started", { turnId: "t", query: "visible" });
    await enabled.emitter.shutdown();
    expect(enabled.events[0]?.data?.query).toBe("visible");
  });

  it("removes nested retrieval payloads in mixed composition without changing sibling data", async () => {
    const { sink, events, emitter } = fixture();
    const sibling = vi.fn<(entry: CallbackTraceEntry) => void>();
    const tracer = compositeTracer([
      new CallbackTracer({ includePayloads: true, sink: sibling }), sink.tracerFor("a"),
    ]);
    const data = {
      turnId: "t",
      entityTerms: ["private top-level entity"],
      query: "private query",
      options: {
        entityTerms: ["private entity"],
        entityTermCount: 1,
        collectors: [{ nested: { prompt: "private prompt", candidate_count: 2 } }],
      },
    };
    tracer.emit("retrieval.started", data);
    await emitter.shutdown();

    expect(events[0]?.data).toEqual({
      trace_turn_id: "t",
      options: { entityTermCount: 1, collectors: [{ nested: { candidate_count: 2 } }] },
    });
    expect(sibling).toHaveBeenCalledWith(expect.objectContaining(data));
    expect(data.options.entityTerms).toEqual(["private entity"]);
    expect(data.options.collectors[0]?.nested.prompt).toBe("private prompt");
    const enabled = fixture(true);
    enabled.sink.tracerFor("a").emit("retrieval.started", data);
    await enabled.emitter.shutdown();
    expect(enabled.events[0]?.data?.options).toEqual(data.options);
  });

  it("drops a trace at capacity before reading or enumerating its data", async () => {
    const { sink, events, emitter } = fixture(false, 10_000, 1);
    const log = vi.spyOn(console, "error").mockImplementation(() => {});
    const tracer = sink.tracerFor("a");
    tracer.emit("retrieval.started", { turnId: "accepted" });
    const read = vi.fn<() => never>(() => { throw new Error("must remain unread"); });
    const data: TurnTraceData = new Proxy({ turnId: "dropped" }, { get: read, ownKeys: read });
    expect(() => tracer.emit("retrieval.started", data)).not.toThrow();
    expect(read).not.toHaveBeenCalled();
    expect(log).toHaveBeenCalledWith(expect.any(String), { dropped_total: 1 });
    await emitter.shutdown();
    expect(events).toHaveLength(1);
    expect(events[0]?.data?.trace_turn_id).toBe("accepted");
  });

  it("redacts and clamps oversized, deep and wide payloads in one bounded pass", async () => {
    const { sink, events, emitter } = fixture();
    const unread = vi.fn<() => never>(() => { throw new Error("outside traversal budget"); });
    const options = { entityTermCount: 1 };
    Object.defineProperty(options, "entityTerms", { enumerable: true, get: unread });
    const counts: JsonValue[] = Array.from({ length: 50 }, (_, index) => index);
    counts.length = 1_000_000;
    Object.defineProperty(counts, "50", { get: unread });
    let deep: JsonValue = {};
    Object.defineProperty(deep, "unread", { enumerable: true, get: unread });
    for (let level = 0; level < 8; level += 1) deep = { child: deep };
    let reads = 0;
    const wide: Record<string, JsonValue> = {};
    for (let index = 0; index < 500; index += 1) {
      Object.defineProperty(wide, `field_${index}`, {
        enumerable: true, get: () => { reads += 1; return index; },
      });
    }
    const data: TurnTraceData = {
      turnId: "bounded", options, counts, deep, note: "x".repeat(100_000), wide,
    };
    const sibling = vi.fn<(entry: CallbackTraceEntry) => void>();
    compositeTracer([
      new CallbackTracer({ includePayloads: true, sink: sibling }), sink.tracerFor("a"),
    ]).emit("retrieval.started", data);
    await emitter.shutdown();

    const emitted = events[0]?.data;
    expect(emitted).toMatchObject({
      trace_turn_id: "bounded", options: { entityTermCount: 1 },
      note: `${"x".repeat(4096)}…[+95904 chars]`, truncated: true, truncated_lists: ["counts"],
      deep: { child: { child: { child: { child: { child: "[truncated]" } } } } },
    });
    expect(emitted?.counts).toHaveLength(50);
    expect(unread).not.toHaveBeenCalled();
    expect(reads).toBeGreaterThan(0);
    expect(reads).toBeLessThanOrEqual(200);
    const siblingEntry = sibling.mock.calls[0]?.[0];
    expect(siblingEntry?.options).toBe(options);
    expect(siblingEntry?.counts).toBe(counts);
    expect(siblingEntry?.deep).toBe(deep);
    expect(siblingEntry?.wide).toBe(wide);
    expect(data.note).toHaveLength(100_000);
    expect(counts).toHaveLength(1_000_000);
  });

  it("bounds nested array fanout and keeps list metadata in oversized summaries", async () => {
    const { sink, events, emitter } = fixture(true);
    const row: JsonValue[] = Array.from({ length: 50 }, () => "x".repeat(4096));
    const rows: JsonValue[] = Array.from({ length: 50 }, () => row);
    sink.tracerFor("a").emit("retrieval.started", { turnId: "fanout", rows });
    await emitter.shutdown();
    expect(events[0]?.data).toMatchObject({
      truncated: true, keys: expect.any(Array), bytes: expect.any(Number), truncated_lists: ["rows"],
    });
    expect(Buffer.byteLength(JSON.stringify(events[0]?.data ?? {}))).toBeLessThanOrEqual(24 * 1024);
    expect(rows).toHaveLength(50);
    expect(row).toHaveLength(50);
  });

  it.each(["fulfilled", "rejected"] as const)("preserves lane detail fields on the wire for %s", async (outcome) => {
    const { sink, events, emitter } = fixture();
    sink.tracerFor("a").emit("retrieval.lane_settled", {
      turnId: "lane", intent_id: "intent_1", intent_kind: "open_question", intent_source: "planner",
      candidate_count: 3, duration_ms: 17, outcome,
      ...(outcome === "rejected" ? { error_name: "TypeError" } : {}),
    });
    await emitter.shutdown();
    expect(events[0]).toMatchObject({
      kind: "recall.lane", phase: "end", duration_ms: 17,
      status: outcome === "rejected" ? "error" : "ok",
      data: { intent_kind: "open_question", intent_source: "planner", candidate_count: 3,
        duration_ms: 17, outcome, trace_turn_id: "lane" },
    });
  });

  it("isolates a failing sink", () => {
    const { sink, emitter } = fixture();
    const log = vi.spyOn(console, "error").mockImplementation(() => {});
    emitter.emitLazy = () => { throw new Error("sensitive detail"); };
    expect(() => sink.tracerFor("a").emit("retrieval.started", { turnId: "t" })).not.toThrow();
    expect(JSON.stringify(log.mock.calls)).not.toContain("sensitive detail");
    log.mockRestore();
  });
});
