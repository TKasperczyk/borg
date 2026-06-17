import { existsSync, mkdtempSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";

import { afterEach, describe, expect, it, vi } from "vitest";

import { Borg } from "../borg.js";
import { BorgPool, type BorgPoolOptions } from "./pool.js";
import { FakeEmbeddingClient } from "../embeddings/index.js";
import { FakeLLMClient } from "../llm/test-support/fake-client.js";
import { ConfigError } from "../util/errors.js";

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
      expect(pool.has("alpha")).toBe(false);
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
    expect(pool.has("alpha")).toBe(false);
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
});
