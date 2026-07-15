import { describe, expect, it } from "vitest";

import {
  DEFAULT_MEMORY_TRACE_CAPACITY,
  DEFAULT_MEMORY_TRACE_MAX_TENANTS,
  MemoryTraceRegistry,
  memoryTraceCapacityFromEnv,
  memoryTraceEnabledFromEnv,
  memoryTraceMaxTenantsFromEnv,
} from "./memory-trace.js";

describe("MemoryTraceRegistry", () => {
  it("keeps bounded isolated per-tenant buffers and filters by since", () => {
    let now = 1_000;
    const registry = new MemoryTraceRegistry({
      capacity: 2,
      now: () => now,
    });
    const alpha = registry.tracerFor("alpha");
    const beta = registry.tracerFor("beta");

    alpha.emit("retrieval.started", {
      turnId: "turn_alpha_1",
      query: "first",
    });
    alpha.emit("turn.token", {
      turnId: "turn_alpha_ignored",
      phase: "final",
      chunk_text: "ignored",
      sequence: 1,
    });
    now = 1_001;
    beta.emit("retrieval.started", {
      turnId: "turn_beta_1",
      query: "beta",
    });
    alpha.emit("retrieval.completed", {
      turnId: "turn_alpha_2",
      episodeCount: 1,
      semanticHits: 0,
    });
    alpha.emit("recall_expansion.completed", {
      turnId: "turn_alpha_3",
      clipped: false,
      facet_count: 0,
      named_term_count: 0,
    });

    const alphaAll = registry.query("alpha", 0);
    expect(alphaAll.events.map((event) => event.turnId)).toEqual(["turn_alpha_2", "turn_alpha_3"]);
    expect(alphaAll.truncated).toBe(false);
    expect(alphaAll.nextSince).toBe(alphaAll.events.at(-1)?.ts);

    const alphaSince = registry.query("alpha", alphaAll.events[0]!.ts);
    expect(alphaSince.events.map((event) => event.turnId)).toEqual(["turn_alpha_3"]);

    const alphaTruncated = registry.query("alpha", 1);
    expect(alphaTruncated.truncated).toBe(true);

    const betaAll = registry.query("beta", 0);
    expect(betaAll.events.map((event) => event.turnId)).toEqual(["turn_beta_1"]);
  });

  it("parses memory trace env flags and capacity", () => {
    expect(memoryTraceEnabledFromEnv({})).toBe(false);
    expect(memoryTraceEnabledFromEnv({ BORG_MEMORY_TRACE_ENABLED: "true" })).toBe(true);
    expect(memoryTraceEnabledFromEnv({ BORG_MEMORY_TRACE_ENABLED: "1" })).toBe(true);
    expect(memoryTraceCapacityFromEnv({})).toBe(DEFAULT_MEMORY_TRACE_CAPACITY);
    expect(memoryTraceCapacityFromEnv({ BORG_MEMORY_TRACE_CAP: "3" })).toBe(3);
    expect(memoryTraceCapacityFromEnv({ BORG_MEMORY_TRACE_CAP: "not-a-number" })).toBe(
      DEFAULT_MEMORY_TRACE_CAPACITY,
    );
    expect(memoryTraceMaxTenantsFromEnv({})).toBe(DEFAULT_MEMORY_TRACE_MAX_TENANTS);
    expect(memoryTraceMaxTenantsFromEnv({ BORG_MEMORY_TRACE_MAX_TENANTS: "2" })).toBe(2);
    expect(memoryTraceMaxTenantsFromEnv({ BORG_MEMORY_TRACE_MAX_TENANTS: "bad" })).toBe(
      DEFAULT_MEMORY_TRACE_MAX_TENANTS,
    );
  });

  it("keeps corrective-preference classifier and ingestion outcomes", () => {
    const registry = new MemoryTraceRegistry({
      capacity: 4,
      now: () => 2_000,
    });
    const tracer = registry.tracerFor("alpha");

    tracer.emit("llm_call.completed", {
      turnId: "turn_commitment_llm",
      label: "corrective_preference_extractor",
      input_tokens: 5,
      output_tokens: 2,
    });
    tracer.emit("corrective_preference.ingestion.completed", {
      turnId: "turn_commitment_result",
      outcome: "none",
      tokens_used: 7,
    });

    expect(registry.query("alpha", 0).events).toEqual([
      expect.objectContaining({
        event: "llm_call.completed",
        turnId: "turn_commitment_llm",
      }),
      expect.objectContaining({
        event: "corrective_preference.ingestion.completed",
        turnId: "turn_commitment_result",
      }),
    ]);
  });

  it("bounds tenant buffers and evicts the least-recently-written tenant", () => {
    const registry = new MemoryTraceRegistry({
      capacity: 2,
      maxTenants: 2,
      now: () => 10_000,
    });

    registry.tracerFor("oldest").emit("retrieval.started", {
      turnId: "turn_oldest",
      query: "oldest",
    });
    registry.tracerFor("kept").emit("retrieval.started", {
      turnId: "turn_kept_1",
      query: "kept",
    });
    registry.tracerFor("oldest").emit("retrieval.completed", {
      turnId: "turn_oldest_recent",
      episodeCount: 0,
      semanticHits: 0,
    });
    registry.tracerFor("newest").emit("retrieval.started", {
      turnId: "turn_newest",
      query: "newest",
    });

    expect(registry.tenantBufferCount()).toBe(2);
    expect(registry.query("kept", 0).events).toEqual([]);
    expect(registry.query("oldest", 0).events.map((event) => event.turnId)).toEqual([
      "turn_oldest",
      "turn_oldest_recent",
    ]);
    expect(registry.query("newest", 0).events.map((event) => event.turnId)).toEqual([
      "turn_newest",
    ]);
  });
});
