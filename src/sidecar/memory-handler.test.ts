import { createServer, type Server } from "node:http";
import { AddressInfo } from "node:net";

import { createHash } from "node:crypto";

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

type Recorder = {
  tenants: string[];
  exclusives: Array<boolean | undefined>;
  lastRecallLimit?: number;
  appendMany?: {
    inputs: unknown[];
    session?: string;
  };
  ingestSessions: string[];
};

function stubBorg(rec: Recorder): Borg {
  return {
    stream: {
      append: async (input: { content: string }) => ({ timestamp: 1000, content: input.content }),
      appendMany: async (inputs: unknown[], options?: { session?: string }) => {
        rec.appendMany = { inputs, session: options?.session };
        return inputs.map((input, index) => ({
          id: `strm_${String(index).padStart(16, "a")}`,
          kind: (input as { kind?: string }).kind,
        }));
      },
    },
    episodic: {
      // Real facade returns numeric counts.
      extract: async () => ({ inserted: 1, updated: 0, skipped: 0 }),
      ingest: async (options?: { session?: string }) => {
        rec.ingestSessions.push(options?.session ?? "");
        return { ran: true, processedEntries: 2 };
      },
      search: async (_query: string, opts: { limit?: number }) => {
        rec.lastRecallLimit = opts.limit;
        return [{ episode: { id: "ep_1", title: "Title", narrative: "Narrative" }, score: 0.91 }];
      },
    },
  } as unknown as Borg;
}

function recordingPool(): { pool: MemoryPool; rec: Recorder } {
  const rec: Recorder = { tenants: [], exclusives: [], ingestSessions: [] };
  const pool: MemoryPool = {
    async withTenant(tenantId, fn, opts) {
      rec.tenants.push(tenantId);
      rec.exclusives.push(opts?.exclusive);
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
    expect(
      (await post(base, "/memory/recall", { tenant: "acme", query: "q" }, "nope")).status,
    ).toBe(401);
    expect(rec.tenants).toEqual([]);
  });

  it("fails closed when the configured token is empty", async () => {
    const { pool } = recordingPool();
    const base = await start(pool, "");
    expect((await post(base, "/memory/recall", { tenant: "acme", query: "q" }, "")).status).toBe(
      401,
    );
    expect(
      (await post(base, "/memory/recall", { tenant: "acme", query: "q" }, "anything")).status,
    ).toBe(401);
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
    const res = await post(
      base,
      "/memory/recall",
      { tenant: "acme", query: "who leads", limit: 999 },
      TOKEN,
    );
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
    const res = await post(
      base,
      "/memory/remember",
      { tenant: "acme", content: "fact", author: "Bob" },
      TOKEN,
    );
    expect(res.status).toBe(200);
    expect(await res.json()).toEqual({
      ok: true,
      extracted: { inserted: 1, updated: 0, skipped: 0 },
    });
    expect(rec.tenants).toEqual(["acme"]);
  });

  it("appends a raw turn and schedules background ingestion", async () => {
    const { pool, rec } = recordingPool();
    const base = await start(pool);
    const rawSession = "tenant::user::conversation";
    const expectedSession = `sess_${createHash("sha256").update(rawSession).digest("hex").slice(0, 16)}`;
    const res = await post(
      base,
      "/memory/append-turn",
      {
        tenant: "acme",
        session: rawSession,
        user: "hello",
        assistant: "hi there",
      },
      TOKEN,
    );

    expect(res.status).toBe(200);
    expect(await res.json()).toEqual({
      ok: true,
      session: expectedSession,
      entries: [
        { id: "strm_aaaaaaaaaaaaaaa0", kind: "user_msg" },
        { id: "strm_aaaaaaaaaaaaaaa1", kind: "agent_msg" },
      ],
    });
    expect(rec.appendMany).toEqual({
      session: expectedSession,
      inputs: [
        { kind: "user_msg", content: "hello" },
        { kind: "agent_msg", content: "hi there" },
      ],
    });
    expect(rec.tenants).toEqual(["acme", "acme"]);
    expect(rec.exclusives).toEqual([true, undefined]);
    expect(rec.ingestSessions).toEqual([expectedSession]);
  });

  it("does not serialize later append-turn requests behind pending ingestion", async () => {
    const appendSessions: string[] = [];
    let ingestionStarted = false;
    let releaseIngestion!: () => void;
    const ingestion = new Promise<{ ran: boolean; processedEntries: number }>((resolve) => {
      releaseIngestion = () => resolve({ ran: true, processedEntries: 2 });
    });
    const borg = {
      stream: {
        appendMany: async (_inputs: unknown[], options?: { session?: string }) => {
          appendSessions.push(options?.session ?? "");
          return [
            { id: "strm_aaaaaaaaaaaaaaaa", kind: "user_msg" },
            { id: "strm_bbbbbbbbbbbbbbbb", kind: "agent_msg" },
          ];
        },
      },
      episodic: {
        ingest: async () => {
          ingestionStarted = true;
          return ingestion;
        },
      },
    } as unknown as Borg;
    let exclusiveTail: Promise<unknown> = Promise.resolve();
    const pool: MemoryPool = {
      withTenant<T>(
        _tenantId: string,
        fn: (borg: Borg) => T | Promise<T>,
        opts?: { exclusive?: boolean },
      ) {
        if (opts?.exclusive === true) {
          const run = exclusiveTail.then(() => fn(borg));
          exclusiveTail = run.then(
            () => undefined,
            () => undefined,
          );
          return run;
        }
        return Promise.resolve(fn(borg));
      },
    };
    const base = await start(pool);

    const first = await post(
      base,
      "/memory/append-turn",
      { tenant: "acme", session: "first", user: "u1", assistant: "a1" },
      TOKEN,
    );
    expect(first.status).toBe(200);
    await first.json();
    expect(ingestionStarted).toBe(true);

    const secondStatus = await Promise.race([
      post(
        base,
        "/memory/append-turn",
        { tenant: "acme", session: "second", user: "u2", assistant: "a2" },
        TOKEN,
      ).then(async (res) => {
        await res.json();
        return res.status;
      }),
      new Promise<number>((resolve) => {
        setTimeout(() => resolve(599), 50);
      }),
    ]);
    releaseIngestion();

    expect(secondStatus).toBe(200);
    expect(appendSessions).toHaveLength(2);
  });

  it("accepts an already-valid borg session id for append-turn", async () => {
    const { pool, rec } = recordingPool();
    const base = await start(pool);
    const session = "sess_aaaaaaaaaaaaaaaa";
    const res = await post(
      base,
      "/memory/append-turn",
      { tenant: "acme", session, user: "u", assistant: "a" },
      TOKEN,
    );

    expect(res.status).toBe(200);
    expect(rec.appendMany?.session).toBe(session);
  });

  it("validates required fields", async () => {
    const { pool } = recordingPool();
    const base = await start(pool);
    expect((await post(base, "/memory/recall", { query: "q" }, TOKEN)).status).toBe(400); // no tenant
    expect((await post(base, "/memory/recall", { tenant: "acme" }, TOKEN)).status).toBe(400); // no query
    expect((await post(base, "/memory/remember", { tenant: "acme" }, TOKEN)).status).toBe(400); // no content
    expect(
      (
        await post(
          base,
          "/memory/append-turn",
          { tenant: "acme", user: "u", assistant: "a" },
          TOKEN,
        )
      ).status,
    ).toBe(400);
    expect(
      (
        await post(
          base,
          "/memory/append-turn",
          { tenant: "acme", session: "s", assistant: "a" },
          TOKEN,
        )
      ).status,
    ).toBe(400);
    expect(
      (await post(base, "/memory/append-turn", { tenant: "acme", session: "s", user: "u" }, TOKEN))
        .status,
    ).toBe(400);
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
    const res = await post(
      base,
      "/memory/remember",
      undefined,
      TOKEN,
      JSON.stringify({ tenant: "acme", content: big }),
    );
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
    expect(
      (await post(base, "/memory/recall", { tenant: "../evil", query: "q" }, TOKEN)).status,
    ).toBe(400);
    expect(
      (await post(base, "/memory/recall", { tenant: "UPPER", query: "q" }, TOKEN)).status,
    ).toBe(400);
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
