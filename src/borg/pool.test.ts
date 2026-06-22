import { existsSync, mkdtempSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";

import { afterEach, describe, expect, it, vi } from "vitest";

import { Borg } from "../borg.js";
import { BorgPool, type BorgPoolOptions } from "./pool.js";
import { FakeEmbeddingClient } from "../embeddings/index.js";
import { FakeLLMClient } from "../llm/test-support/fake-client.js";
import type { TurnTraceData, TurnTraceEventName, TurnTracer } from "../tracing/tracer.js";
import { ConfigError } from "../util/errors.js";
import { createSessionId } from "../util/ids.js";

const cleanups: Array<() => Promise<void> | void> = [];

afterEach(async () => {
  while (cleanups.length > 0) {
    await cleanups.pop()?.();
  }
});

function baseOpenOptions(): BorgPoolOptions["openOptions"] {
  return {
    embeddingDimensions: 4,
    embeddingClient: new FakeEmbeddingClient(4),
    llmClient: new FakeLLMClient(),
  };
}

function makePool(overrides: Partial<BorgPoolOptions> = {}): { pool: BorgPool; root: string } {
  const root = mkdtempSync(join(tmpdir(), "borg-pool-"));
  const pool = new BorgPool({ root, openOptions: baseOpenOptions(), ...overrides });
  cleanups.push(async () => {
    await pool.closeAll().catch(() => {});
    rmSync(root, { recursive: true, force: true });
  });
  return { pool, root };
}

function append(content: string) {
  return (borg: Borg) => borg.stream.append({ kind: "user_msg", content });
}

function tailContents(borg: Borg): unknown[] {
  return borg.stream.tail(20).map((entry) => entry.content);
}

function recordingTracer(events: Array<{ event: TurnTraceEventName; data: TurnTraceData }>) {
  return {
    enabled: true,
    includePayloads: false,
    emit: (event, data) => {
      events.push({ event, data });
    },
  } satisfies TurnTracer;
}

describe("BorgPool", () => {
  it("isolates tenants in separate dataDirs (no cross-tenant recall)", async () => {
    const { pool, root } = makePool();

    await pool.withTenant("alpha", append("alpha-secret"));
    await pool.withTenant("beta", append("beta-secret"));

    const alpha = await pool.withTenant("alpha", tailContents);
    const beta = await pool.withTenant("beta", tailContents);

    expect(alpha).toContain("alpha-secret");
    expect(alpha).not.toContain("beta-secret");
    expect(beta).toContain("beta-secret");
    expect(beta).not.toContain("alpha-secret");

    expect(existsSync(join(root, "alpha", "borg.db"))).toBe(true);
    expect(existsSync(join(root, "beta", "borg.db"))).toBe(true);
  });

  it("reuses one being per tenant and dedupes concurrent opens", async () => {
    const { pool } = makePool();

    const [a1, a2] = await Promise.all([
      pool.withTenant("alpha", (b) => b),
      pool.withTenant("alpha", (b) => b),
    ]);

    expect(a1).toBe(a2);
    expect(pool.size()).toBe(1);
  });

  it("composes shared openOptions tracer with per-tenant tracerFor", async () => {
    const sharedEvents: Array<{ event: TurnTraceEventName; data: TurnTraceData }> = [];
    const tenantEvents = new Map<string, Array<{ event: TurnTraceEventName; data: TurnTraceData }>>();
    const { pool } = makePool({
      openOptions: {
        ...baseOpenOptions(),
        tracer: recordingTracer(sharedEvents),
      },
      tracerFor: (tenantId) => {
        const events: Array<{ event: TurnTraceEventName; data: TurnTraceData }> = [];
        tenantEvents.set(tenantId, events);
        return recordingTracer(events);
      },
    });
    const alphaSession = createSessionId();
    const betaSession = createSessionId();

    await pool.withTenant("alpha", (borg) => borg.endSession(alphaSession));
    await pool.withTenant("beta", (borg) => borg.endSession(betaSession));

    expect(sharedEvents.filter((entry) => entry.event === "session.completed")).toHaveLength(2);
    expect(tenantEvents.get("alpha")?.filter((entry) => entry.event === "session.completed").map((entry) => entry.data.turnId)).toEqual([
      `session_end:${alphaSession}`,
    ]);
    expect(tenantEvents.get("beta")?.filter((entry) => entry.event === "session.completed").map((entry) => entry.data.turnId)).toEqual([
      `session_end:${betaSession}`,
    ]);
  });

  it("evict closes the being; reopening preserves persisted data", async () => {
    const { pool } = makePool();

    await pool.withTenant("alpha", append("persist-me"));
    await pool.evict("alpha");
    expect(pool.has("alpha")).toBe(false);

    const reopened = await pool.withTenant("alpha", tailContents);
    expect(reopened).toContain("persist-me");
  });

  it("enforces maxOpen by closing the least-recently-used being", async () => {
    const { pool } = makePool({ maxOpen: 1 });

    await pool.withTenant("alpha", append("a"));
    expect(pool.openTenantIds()).toEqual(["alpha"]);

    await pool.withTenant("beta", append("b"));
    expect(pool.has("alpha")).toBe(false); // evicted as LRU
    expect(pool.has("beta")).toBe(true);
    expect(pool.size()).toBe(1);

    const alpha = await pool.withTenant("alpha", tailContents);
    expect(alpha).toContain("a"); // evicted tenant's data survived + reopened
    expect(pool.has("beta")).toBe(false); // beta evicted when alpha reopened
  });

  it("never evicts a being with an in-flight operation (LRU path)", async () => {
    const { pool } = makePool({ maxOpen: 1 });

    let release: () => void = () => {};
    const held = new Promise<void>((resolve) => {
      release = resolve;
    });
    const alphaOp = pool.withTenant("alpha", async (borg) => {
      await borg.stream.append({ kind: "user_msg", content: "held" });
      await held; // keep alpha inUse
      return "alpha-done";
    });

    await pool.withTenant("beta", append("b")); // requested while alpha in use
    expect(pool.has("alpha")).toBe(true); // alpha NOT evicted despite maxOpen=1

    release();
    expect(await alphaOp).toBe("alpha-done");
  });

  it("evict drains an in-flight op before closing (no use-after-close)", async () => {
    const { pool } = makePool();

    let release: () => void = () => {};
    const held = new Promise<void>((resolve) => {
      release = resolve;
    });
    const op = pool.withTenant("alpha", async (borg) => {
      await borg.stream.append({ kind: "user_msg", content: "before" });
      await held;
      // Would throw if storage were torn down under us.
      await borg.stream.append({ kind: "user_msg", content: "after" });
      return "done";
    });

    const ev = pool.evict("alpha"); // inUse > 0 -> must defer until the op finishes
    expect(pool.has("alpha")).toBe(true);

    release();
    expect(await op).toBe("done");
    await ev;
    expect(pool.has("alpha")).toBe(false);
  });

  it("closeAll drains in-flight ops before closing", async () => {
    const { pool } = makePool();

    let release: () => void = () => {};
    const held = new Promise<void>((resolve) => {
      release = resolve;
    });
    const op = pool.withTenant("alpha", async (borg) => {
      await borg.stream.append({ kind: "user_msg", content: "before" });
      await held;
      await borg.stream.append({ kind: "user_msg", content: "after" });
      return "done";
    });

    const all = pool.closeAll();
    release();
    expect(await op).toBe("done");
    await all;
    expect(pool.size()).toBe(0);
  });

  it("an evicted being's close failure does not fail the unrelated tenant", async () => {
    const { pool } = makePool({ maxOpen: 1 });
    const errorSpy = vi.spyOn(console, "error").mockImplementation(() => {});

    try {
      let alphaBorg!: Borg;
      await pool.withTenant("alpha", (borg) => {
        alphaBorg = borg;
      });
      alphaBorg.close = () => Promise.reject(new Error("boom"));

      // Opening beta evicts alpha (LRU); alpha.close rejects, beta must still succeed.
      await expect(
        pool.withTenant("beta", append("b")),
      ).resolves.toBeDefined();
      expect(pool.has("beta")).toBe(true);
      // alpha's close failed -> its entry is RETAINED (fail-closed: never reopen the
      // same dataDir), not silently dropped. beta succeeding is what matters here.
      expect(pool.has("alpha")).toBe(true);
      expect(errorSpy).toHaveBeenCalled();
    } finally {
      errorSpy.mockRestore();
    }
  });

  it("surfaces a close failure from a deliberate evict deferred behind an in-flight op", async () => {
    const { pool } = makePool();

    let alphaBorg!: Borg;
    let signalReady: () => void = () => {};
    const ready = new Promise<void>((resolve) => {
      signalReady = resolve;
    });
    let release: () => void = () => {};
    const held = new Promise<void>((resolve) => {
      release = resolve;
    });

    const op = pool.withTenant("alpha", async (borg) => {
      alphaBorg = borg;
      signalReady();
      await held;
      return "done";
    });

    await ready; // alpha captured, op parked (inUse > 0)
    alphaBorg.close = () => Promise.reject(new Error("boom"));

    const ev = pool.evict("alpha"); // deferred behind the in-flight op
    release();

    await expect(op).resolves.toBe("done"); // the op itself completes fine
    await expect(ev).rejects.toThrow("boom"); // evict surfaces the deferred close failure
    // Close failed -> entry retained (fail-closed), not deleted, so a later acquire
    // can't open a second Borg over the same dataDir.
    expect(pool.has("alpha")).toBe(true);
  });

  it("rejects tenant ids that could escape the root", async () => {
    const { pool } = makePool();

    for (const bad of ["../evil", "a/b", "..", "", "UPPER", "has space", "a\\b"]) {
      await expect(pool.withTenant(bad, (b) => b)).rejects.toBeInstanceOf(ConfigError);
    }
    expect(pool.size()).toBe(0);
  });

  it("has() and evict() are non-throwing for invalid tenant ids", async () => {
    const { pool } = makePool();

    expect(pool.has("../evil")).toBe(false);
    expect(pool.has("")).toBe(false);
    await expect(pool.evict("../evil")).resolves.toBeUndefined();
  });

  it("fails closed for a pathological root (rejects all tenants)", async () => {
    const pool = new BorgPool({ root: "/", openOptions: baseOpenOptions() });

    await expect(pool.withTenant("alpha", (b) => b)).rejects.toBeInstanceOf(ConfigError);
    expect(pool.size()).toBe(0);
  });

  it("closeAll closes every open being", async () => {
    const { pool } = makePool();

    await pool.withTenant("alpha", (b) => b);
    await pool.withTenant("beta", (b) => b);
    expect(pool.size()).toBe(2);

    await pool.closeAll();
    expect(pool.size()).toBe(0);
  });

  // A withTenant callback that records peak concurrency: it bumps a shared counter
  // on entry, yields twice (an interleaving impl would let another callback in
  // during the yields), then decrements. Serialized callbacks keep the peak at 1.
  function makeConcurrencyProbe() {
    let active = 0;
    const state = { peak: 0, ran: [] as number[] };
    const op = (pool: BorgPool, tenant: string, id: number, exclusive: boolean) =>
      pool.withTenant(
        tenant,
        async () => {
          active += 1;
          state.peak = Math.max(state.peak, active);
          state.ran.push(id);
          await Promise.resolve();
          await Promise.resolve();
          active -= 1;
          return id;
        },
        exclusive ? { exclusive: true } : undefined,
      );
    return { state, op };
  }

  it("serializes exclusive ops for the same tenant (peak concurrency 1)", async () => {
    const { pool } = makePool();
    const { state, op } = makeConcurrencyProbe();

    const results = await Promise.all([
      op(pool, "alpha", 1, true),
      op(pool, "alpha", 2, true),
      op(pool, "alpha", 3, true),
    ]);

    expect(state.peak).toBe(1); // never two exclusive ops on one being at once
    expect(results).toEqual([1, 2, 3]); // each op returns its own result
    expect(new Set(state.ran)).toEqual(new Set([1, 2, 3])); // all three ran
  });

  it("does not serialize non-exclusive ops on the same tenant (reads stay concurrent)", async () => {
    const { pool } = makePool();
    const { state, op } = makeConcurrencyProbe();

    await Promise.all([
      op(pool, "alpha", 1, false),
      op(pool, "alpha", 2, false),
      op(pool, "alpha", 3, false),
    ]);

    expect(state.peak).toBeGreaterThan(1); // overlapped -> not serialized
  });

  it("exclusive ops on different tenants run concurrently (independent chains)", async () => {
    const { pool } = makePool();

    let alphaStarted: () => void = () => {};
    const alphaReady = new Promise<void>((resolve) => {
      alphaStarted = resolve;
    });
    let releaseAlpha: () => void = () => {};
    const gateAlpha = new Promise<void>((resolve) => {
      releaseAlpha = resolve;
    });

    const opAlpha = pool.withTenant(
      "alpha",
      async () => {
        alphaStarted();
        await gateAlpha; // park alpha's exclusive chain indefinitely
        return "alpha";
      },
      { exclusive: true },
    );

    await alphaReady; // alpha is parked inside its exclusive section
    // beta is a different tenant -> different chain -> must NOT be blocked by the
    // parked alpha. If chains weren't per-tenant this would deadlock.
    const beta = await pool.withTenant("beta", () => "beta", { exclusive: true });
    expect(beta).toBe("beta");

    releaseAlpha();
    expect(await opAlpha).toBe("alpha");
  });

  it("a failed exclusive op does not block the next exclusive op", async () => {
    const { pool } = makePool();

    const op1 = pool.withTenant("alpha", () => Promise.reject(new Error("boom")), {
      exclusive: true,
    });
    const op2 = pool.withTenant("alpha", () => "ok", { exclusive: true });

    await expect(op1).rejects.toThrow("boom");
    await expect(op2).resolves.toBe("ok");
  });

  it("shutdown() closes open beings and rejects all subsequent work", async () => {
    const { pool } = makePool();

    await pool.withTenant("alpha", append("x"));
    expect(pool.size()).toBe(1);

    await pool.shutdown();
    expect(pool.size()).toBe(0);

    await expect(pool.withTenant("alpha", (b) => b)).rejects.toThrow(/shutting down/i);
    await expect(pool.withTenant("beta", (b) => b)).rejects.toThrow(/shutting down/i);
  });

  it("shutdown() drains an in-flight op while rejecting newly-arriving work (barrier)", async () => {
    const { pool } = makePool();

    let started: () => void = () => {};
    const ready = new Promise<void>((resolve) => {
      started = resolve;
    });
    let release: () => void = () => {};
    const held = new Promise<void>((resolve) => {
      release = resolve;
    });

    const inflight = pool.withTenant("alpha", async (borg) => {
      await borg.stream.append({ kind: "user_msg", content: "before" });
      started();
      await held;
      // Would throw if storage were torn down under us mid-shutdown.
      await borg.stream.append({ kind: "user_msg", content: "after" });
      return "done";
    });

    await ready; // alpha op is in-flight (inUse > 0)
    const shut = pool.shutdown(); // sets closing, then drains the in-flight op

    // A request arriving DURING shutdown must be rejected, not opened behind the barrier.
    await expect(pool.withTenant("beta", (b) => b)).rejects.toThrow(/shutting down/i);

    release();
    expect(await inflight).toBe("done"); // the in-flight op completed cleanly
    await shut;
    expect(pool.size()).toBe(0);
  });

  it("a failed close keeps the entry and refuses to reopen the same dataDir", async () => {
    const { pool } = makePool();
    const errorSpy = vi.spyOn(console, "error").mockImplementation(() => {});

    try {
      let alphaBorg!: Borg;
      await pool.withTenant("alpha", (borg) => {
        alphaBorg = borg;
      });
      alphaBorg.close = () => Promise.reject(new Error("close boom"));

      await expect(pool.evict("alpha")).rejects.toThrow("close boom");
      expect(pool.has("alpha")).toBe(true); // entry kept, NOT deleted

      // No second Borg over the same dataDir: acquire fails closed.
      await expect(pool.withTenant("alpha", (b) => b)).rejects.toThrow(
        /failed to close|restart required/i,
      );
    } finally {
      errorSpy.mockRestore();
    }
  });
});
