import { createServer, request as httpRequest, type Server } from "node:http";
import { AddressInfo } from "node:net";

import { createHash } from "node:crypto";

import { afterEach, describe, expect, it } from "vitest";

import { createMemoryHandler, type MemoryHandlerOptions, type MemoryPool } from "./memory-handler.js";
import { MemoryTraceRegistry } from "./memory-trace.js";
import type { Borg } from "../borg.js";
import type { Episode } from "../memory/episodic/index.js";

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
  lastRecallTraceTurnId?: string;
  lastListOptions?: {
    limit?: number;
    cursor?: string;
  };
  inspectIds: string[];
  appendMany?: {
    inputs: unknown[];
    session?: string;
  };
  ingestSessions: string[];
};

function testEpisode(id: Episode["id"] = "ep_aaaaaaaaaaaaaaaa" as Episode["id"]): Episode {
  return {
    id,
    title: "Title",
    narrative: "Narrative",
    participants: ["Ada"],
    location: null,
    start_time: 10,
    end_time: 20,
    source_stream_ids: ["strm_aaaaaaaaaaaaaaaa" as Episode["source_stream_ids"][number]],
    significance: 0.72,
    tags: ["planning", "admin"],
    confidence: 0.9,
    lineage: {
      derived_from: [],
      supersedes: [],
    },
    emotional_arc: null,
    audience_entity_id: null,
    origin_audience_entity_ids: [],
    shared: false,
    episode_kind: "raw",
    consolidation_family_id: null,
    consolidation_coverage_hash: null,
    embedding: Float32Array.from([1, 0, 0, 0]),
    created_at: 1,
    updated_at: 2,
  };
}

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
      search: async (_query: string, opts: { limit?: number; traceTurnId?: string }) => {
        rec.lastRecallLimit = opts.limit;
        rec.lastRecallTraceTurnId = opts.traceTurnId;
        return [{ episode: { id: "ep_1", title: "Title", narrative: "Narrative" }, score: 0.91 }];
      },
      list: async (options?: { limit?: number; cursor?: string }) => {
        rec.lastListOptions = options;
        return {
          items: [testEpisode()],
          nextCursor: "next-cursor",
        };
      },
      inspect: async (id: Episode["id"]) => {
        rec.inspectIds.push(id);
        return id === ("ep_missingmissing00" as Episode["id"]) ? null : testEpisode(id);
      },
    },
  } as unknown as Borg;
}

function recordingPool(): { pool: MemoryPool; rec: Recorder } {
  const rec: Recorder = { tenants: [], exclusives: [], inspectIds: [], ingestSessions: [] };
  const pool: MemoryPool = {
    async withTenant(tenantId, fn, opts) {
      rec.tenants.push(tenantId);
      rec.exclusives.push(opts?.exclusive);
      return fn(stubBorg(rec));
    },
  };
  return { pool, rec };
}

function start(
  pool: MemoryPool,
  token = TOKEN,
  handlerOptions: Omit<MemoryHandlerOptions, "pool" | "token"> = {},
): Promise<string> {
  const server = createServer(createMemoryHandler({ pool, token, ...handlerOptions }));
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

async function get(base: string, path: string, token?: string) {
  const headers: Record<string, string> = {};
  if (token !== undefined) {
    headers["x-borg-token"] = token;
  }
  return fetch(`${base}${path}`, { headers });
}

async function requestRaw(
  base: string,
  path: string,
  options: {
    method?: string;
    token?: string;
    body?: unknown;
  } = {},
): Promise<{ status: number; body: unknown; text: string }> {
  const baseUrl = new URL(base);
  const rawBody = options.body === undefined ? undefined : JSON.stringify(options.body);
  const headers: Record<string, string> = {};
  if (options.token !== undefined) {
    headers["x-borg-token"] = options.token;
  }
  if (rawBody !== undefined) {
    headers["content-type"] = "application/json";
    headers["content-length"] = String(Buffer.byteLength(rawBody));
  }

  return new Promise((resolve, reject) => {
    const req = httpRequest(
      {
        hostname: baseUrl.hostname,
        port: Number(baseUrl.port),
        method: options.method ?? "GET",
        path,
        headers,
      },
      (res) => {
        const chunks: Buffer[] = [];
        res.on("data", (chunk: Buffer) => chunks.push(chunk));
        res.on("end", () => {
          const text = Buffer.concat(chunks).toString("utf8");
          let body: unknown = text;
          try {
            body = JSON.parse(text) as unknown;
          } catch {
            // Keep non-JSON bodies as text for diagnostics.
          }
          resolve({ status: res.statusCode ?? 0, body, text });
        });
      },
    );
    req.on("error", reject);
    req.end(rawBody);
  });
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

  it("uses the raw request path for auth and routing without dot-segment normalization", async () => {
    const { pool, rec } = recordingPool();
    const base = await start(pool);

    const authProbe = await requestRaw(base, "/foo/../healthz");
    expect(authProbe.status).toBe(401);
    expect(authProbe.body).toEqual({ error: "unauthorized" });

    const routeProbe = await requestRaw(base, "/memory/nope/../recall", {
      method: "POST",
      token: TOKEN,
      body: { tenant: "acme", query: "q" },
    });
    expect(routeProbe.status).toBe(404);
    expect(routeProbe.body).toEqual({ error: "not found" });
    expect(rec.tenants).toEqual([]);
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
    expect(rec.lastRecallTraceTurnId).toBeUndefined();
  });

  it("keeps recall response unchanged while passing a trace turn id when tracing is enabled", async () => {
    const { pool: offPool, rec: offRec } = recordingPool();
    const offBase = await start(offPool);
    const offRes = await post(
      offBase,
      "/memory/recall",
      { tenant: "acme", query: "who leads", limit: 3 },
      TOKEN,
    );
    const offBody = await offRes.json();

    const { pool: onPool, rec: onRec } = recordingPool();
    const onBase = await start(onPool, TOKEN, { traceRegistry: new MemoryTraceRegistry() });
    const onRes = await post(
      onBase,
      "/memory/recall",
      { tenant: "acme", query: "who leads", limit: 3 },
      TOKEN,
    );
    const onBody = await onRes.json();

    expect(onRes.status).toBe(200);
    expect(onBody).toEqual(offBody);
    expect(offRec.lastRecallTraceTurnId).toBeUndefined();
    expect(onRec.lastRecallTraceTurnId).toMatch(/^sidecar_recall:acme:/);
  });

  it("lists episodes from the query tenant without a body, clamping limit and passing cursor", async () => {
    const { pool, rec } = recordingPool();
    const base = await start(pool);
    const res = await get(
      base,
      "/memory/episodes?tenant=acme&limit=999&cursor=opaque-cursor",
      TOKEN,
    );

    expect(res.status).toBe(200);
    expect(await res.json()).toEqual({
      ok: true,
      episodes: [
        {
          id: "ep_aaaaaaaaaaaaaaaa",
          title: "Title",
          narrative: "Narrative",
          significance: 0.72,
          tags: ["planning", "admin"],
          source_stream_ids: ["strm_aaaaaaaaaaaaaaaa"],
        },
      ],
      nextCursor: "next-cursor",
    });
    expect(rec.tenants).toEqual(["acme"]);
    expect(rec.exclusives).toEqual([undefined]);
    expect(rec.lastListOptions).toEqual({ limit: 100, cursor: "opaque-cursor" });
  });

  it("returns a disabled empty trace response without opening a tenant", async () => {
    const { pool, rec } = recordingPool();
    const base = await start(pool);
    const res = await get(base, "/memory/trace?tenant=acme", TOKEN);

    expect(res.status).toBe(200);
    expect(await res.json()).toEqual({
      ok: true,
      tenant: "acme",
      events: [],
      disabled: true,
    });
    expect(rec.tenants).toEqual([]);
  });

  it("returns enabled trace events filtered by since without opening a tenant", async () => {
    let now = 5_000;
    const traceRegistry = new MemoryTraceRegistry({
      capacity: 5,
      now: () => now,
    });
    const tracer = traceRegistry.tracerFor("acme");
    tracer.emit("retrieval.started", {
      turnId: "turn_trace_1",
      query: "first",
    });
    now = 5_100;
    tracer.emit("retrieval.completed", {
      turnId: "turn_trace_2",
      episodeCount: 1,
      semanticHits: 0,
    });
    const since = traceRegistry.query("acme", 0).events[0]!.ts;
    const { pool, rec } = recordingPool();
    const base = await start(pool, TOKEN, { traceRegistry });
    const res = await get(base, `/memory/trace?tenant=acme&since=${since}`, TOKEN);

    expect(res.status).toBe(200);
    expect(await res.json()).toMatchObject({
      ok: true,
      tenant: "acme",
      events: [
        expect.objectContaining({
          turnId: "turn_trace_2",
          event: "retrieval.completed",
        }),
      ],
      nextSince: expect.any(Number),
      truncated: false,
    });
    expect(rec.tenants).toEqual([]);
  });

  it("rejects invalid trace tenant or since before touching the pool", async () => {
    const { pool, rec } = recordingPool();
    const base = await start(pool, TOKEN, { traceRegistry: new MemoryTraceRegistry() });

    expect((await get(base, "/memory/trace", TOKEN)).status).toBe(400);
    expect((await get(base, "/memory/trace?tenant=UPPER", TOKEN)).status).toBe(400);
    expect((await get(base, "/memory/trace?tenant=acme&since=nope", TOKEN)).status).toBe(400);
    expect(rec.tenants).toEqual([]);
  });

  it("treats empty or whitespace list cursors as absent", async () => {
    const { pool, rec } = recordingPool();
    const base = await start(pool);

    expect((await get(base, "/memory/episodes?tenant=acme&cursor=", TOKEN)).status).toBe(200);
    expect(rec.lastListOptions).toEqual({ limit: 20 });

    expect((await get(base, "/memory/episodes?tenant=acme&cursor=%20%20", TOKEN)).status).toBe(200);
    expect(rec.lastListOptions).toEqual({ limit: 20 });
  });

  it("maps malformed list cursors to 400 client errors", async () => {
    const calls: string[] = [];
    const pool: MemoryPool = {
      async withTenant(tenantId, fn) {
        calls.push(tenantId);
        const invalidCursor = Object.assign(new Error("Invalid episode cursor"), {
          code: "EPISODE_CURSOR_INVALID",
        });
        return fn({
          episodic: {
            list: async () => {
              throw invalidCursor;
            },
          },
        } as unknown as Borg);
      },
    };
    const base = await start(pool);
    const res = await get(base, "/memory/episodes?tenant=acme&cursor=not-a-cursor", TOKEN);

    expect(res.status).toBe(400);
    expect(await res.json()).toEqual({ error: "invalid 'cursor'" });
    expect(calls).toEqual(["acme"]);
  });

  it("rejects missing or invalid query tenant for episode GET routes before touching the pool", async () => {
    const calls: string[] = [];
    const pool: MemoryPool = {
      async withTenant(tenantId, fn) {
        calls.push(tenantId);
        return fn({} as Borg);
      },
    };
    const base = await start(pool);

    expect((await get(base, "/memory/episodes", TOKEN)).status).toBe(400);
    expect((await get(base, "/memory/episodes?tenant=UPPER", TOKEN)).status).toBe(400);
    expect(
      (await get(base, "/memory/episodes/ep_aaaaaaaaaaaaaaaa?tenant=../evil", TOKEN)).status,
    ).toBe(400);
    expect(calls).toEqual([]);
  });

  it("inspects one episode by query tenant and excludes the embedding vector", async () => {
    const { pool, rec } = recordingPool();
    const base = await start(pool);
    const res = await get(base, "/memory/episodes/ep_aaaaaaaaaaaaaaaa?tenant=acme", TOKEN);

    expect(res.status).toBe(200);
    const body = (await res.json()) as { episode: Record<string, unknown> };
    expect(body).toMatchObject({
      ok: true,
      episode: {
        id: "ep_aaaaaaaaaaaaaaaa",
        title: "Title",
        narrative: "Narrative",
        participants: ["Ada"],
        source_stream_ids: ["strm_aaaaaaaaaaaaaaaa"],
        significance: 0.72,
        tags: ["planning", "admin"],
      },
    });
    expect(body.episode).not.toHaveProperty("embedding");
    expect(rec.tenants).toEqual(["acme"]);
    expect(rec.exclusives).toEqual([undefined]);
    expect(rec.inspectIds).toEqual(["ep_aaaaaaaaaaaaaaaa"]);
  });

  it("404s an unknown episode id and 400s an invalid episode id without touching the pool", async () => {
    const { pool, rec } = recordingPool();
    const base = await start(pool);
    const missing = await get(base, "/memory/episodes/ep_missingmissing00?tenant=acme", TOKEN);

    expect(missing.status).toBe(404);
    expect(await missing.json()).toEqual({ ok: false });
    expect(rec.tenants).toEqual(["acme"]);
    expect(rec.inspectIds).toEqual(["ep_missingmissing00"]);

    const invalidCalls: string[] = [];
    const invalidPool: MemoryPool = {
      async withTenant(tenantId, fn) {
        invalidCalls.push(tenantId);
        return fn({} as Borg);
      },
    };
    const invalidBase = await start(invalidPool);
    expect(
      (await get(invalidBase, "/memory/episodes/not-an-episode?tenant=acme", TOKEN)).status,
    ).toBe(400);
    expect(invalidCalls).toEqual([]);
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
