import { createServer, type Server } from "node:http";
import { AddressInfo } from "node:net";

import { afterEach, describe, expect, it } from "vitest";

import { createMemoryHandler, type MemoryPool } from "./memory-handler.js";
import type { Borg } from "../borg.js";

const TOKEN = "secret-token";

const servers: Server[] = [];

afterEach(() => {
  while (servers.length > 0) {
    servers.pop()?.close();
  }
});

type Recorder = { tenants: string[]; lastRecallLimit?: number };

function stubBorg(rec: Recorder): Borg {
  return {
    stream: {
      append: async (input: { content: string }) => ({ timestamp: 1000, content: input.content }),
    },
    episodic: {
      // Real facade returns numeric counts.
      extract: async () => ({ inserted: 1, updated: 0, skipped: 0 }),
      search: async (_query: string, opts: { limit?: number }) => {
        rec.lastRecallLimit = opts.limit;
        return [{ episode: { id: "ep_1", title: "Title", narrative: "Narrative" }, score: 0.91 }];
      },
    },
  } as unknown as Borg;
}

function recordingPool(): { pool: MemoryPool; rec: Recorder } {
  const rec: Recorder = { tenants: [] };
  const pool: MemoryPool = {
    async withTenant(tenantId, fn) {
      rec.tenants.push(tenantId);
      return fn(stubBorg(rec));
    },
  };
  return { pool, rec };
}

function start(pool: MemoryPool, token = TOKEN): Promise<string> {
  const server = createServer(createMemoryHandler({ pool, token }));
  servers.push(server);
  return new Promise((resolve) => {
    server.listen(0, "127.0.0.1", () => {
      const { port } = server.address() as AddressInfo;
      resolve(`http://127.0.0.1:${port}`);
    });
  });
}

async function post(base: string, path: string, body: unknown, token?: string, raw?: string) {
  const headers: Record<string, string> = { "content-type": "application/json" };
  if (token !== undefined) {
    headers["x-borg-token"] = token;
  }
  return fetch(`${base}${path}`, { method: "POST", headers, body: raw ?? JSON.stringify(body) });
}

describe("memory sidecar handler", () => {
  it("serves /healthz without auth", async () => {
    const { pool } = recordingPool();
    const base = await start(pool);
    const res = await fetch(`${base}/healthz`);
    expect(res.status).toBe(200);
    expect(await res.json()).toEqual({ ok: true });
  });

  it("rejects missing/wrong token with 401 and does not touch the pool", async () => {
    const { pool, rec } = recordingPool();
    const base = await start(pool);
    expect((await post(base, "/memory/recall", { tenant: "acme", query: "q" })).status).toBe(401);
    expect((await post(base, "/memory/recall", { tenant: "acme", query: "q" }, "nope")).status).toBe(401);
    expect(rec.tenants).toEqual([]);
  });

  it("fails closed when the configured token is empty", async () => {
    const { pool } = recordingPool();
    const base = await start(pool, "");
    expect((await post(base, "/memory/recall", { tenant: "acme", query: "q" }, "")).status).toBe(401);
    expect((await post(base, "/memory/recall", { tenant: "acme", query: "q" }, "anything")).status).toBe(401);
  });

  it("404s unknown routes and non-POST methods (after auth)", async () => {
    const { pool } = recordingPool();
    const base = await start(pool);
    expect((await post(base, "/memory/nope", { tenant: "acme" }, TOKEN)).status).toBe(404);
    const getRes = await fetch(`${base}/memory/recall`, { headers: { "x-borg-token": TOKEN } });
    expect(getRes.status).toBe(404);
  });

  it("recalls and maps episodes, routing by tenant, clamping the limit", async () => {
    const { pool, rec } = recordingPool();
    const base = await start(pool);
    const res = await post(base, "/memory/recall", { tenant: "acme", query: "who leads", limit: 999 }, TOKEN);
    expect(res.status).toBe(200);
    expect(await res.json()).toEqual({
      ok: true,
      episodes: [{ id: "ep_1", title: "Title", narrative: "Narrative", score: 0.91 }],
    });
    expect(rec.tenants).toEqual(["acme"]);
    expect(rec.lastRecallLimit).toBe(50); // clamped to maxRecallLimit
  });

  it("remembers (append + extract), routing by tenant", async () => {
    const { pool, rec } = recordingPool();
    const base = await start(pool);
    const res = await post(base, "/memory/remember", { tenant: "acme", content: "fact", author: "Bob" }, TOKEN);
    expect(res.status).toBe(200);
    expect(await res.json()).toEqual({ ok: true, extracted: { inserted: 1, updated: 0, skipped: 0 } });
    expect(rec.tenants).toEqual(["acme"]);
  });

  it("validates required fields", async () => {
    const { pool } = recordingPool();
    const base = await start(pool);
    expect((await post(base, "/memory/recall", { query: "q" }, TOKEN)).status).toBe(400); // no tenant
    expect((await post(base, "/memory/recall", { tenant: "acme" }, TOKEN)).status).toBe(400); // no query
    expect((await post(base, "/memory/remember", { tenant: "acme" }, TOKEN)).status).toBe(400); // no content
  });

  it("400s on invalid or non-object JSON", async () => {
    const { pool } = recordingPool();
    const base = await start(pool);
    expect((await post(base, "/memory/recall", undefined, TOKEN, "{not json")).status).toBe(400);
    expect((await post(base, "/memory/recall", undefined, TOKEN, "null")).status).toBe(400);
    expect((await post(base, "/memory/recall", undefined, TOKEN, "42")).status).toBe(400);
  });

  it("413s an oversized body", async () => {
    const { pool } = recordingPool();
    const base = await start(pool);
    const big = "x".repeat(70 * 1024); // > 64KB default
    const res = await post(base, "/memory/remember", undefined, TOKEN, JSON.stringify({ tenant: "acme", content: big }));
    expect(res.status).toBe(413);
  });

  it("rejects a malformed tenant id with 400 before touching the pool", async () => {
    const calls: string[] = [];
    const pool: MemoryPool = {
      async withTenant(tenantId, fn) {
        calls.push(tenantId);
        return fn({} as Borg);
      },
    };
    const base = await start(pool);
    expect((await post(base, "/memory/recall", { tenant: "../evil", query: "q" }, TOKEN)).status).toBe(400);
    expect((await post(base, "/memory/recall", { tenant: "UPPER", query: "q" }, TOKEN)).status).toBe(400);
    expect(calls).toEqual([]); // pool never reached for an invalid tenant
  });

  it("does not leak internals on an unexpected error (generic 500)", async () => {
    const boomPool: MemoryPool = {
      async withTenant() {
        throw new Error("sqlite path /secret/db.sqlite is locked");
      },
    };
    const base = await start(boomPool);
    const res = await post(base, "/memory/recall", { tenant: "acme", query: "q" }, TOKEN);
    expect(res.status).toBe(500);
    expect(await res.json()).toEqual({ error: "internal error" }); // no internal detail leaked
  });
});
