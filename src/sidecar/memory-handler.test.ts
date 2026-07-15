import { createServer, request as httpRequest, type Server } from "node:http";
import { AddressInfo } from "node:net";

import { createHash } from "node:crypto";

import { afterEach, describe, expect, it } from "vitest";

import {
  createMemoryHandler,
  type MemoryHandlerOptions,
  type MemoryPool,
} from "./memory-handler.js";
import { MemoryTraceRegistry } from "./memory-trace.js";
import type { Borg } from "../borg.js";
import { commitmentSchema, type CommitmentRecord } from "../memory/commitments/index.js";
import type { Episode } from "../memory/episodic/index.js";
import {
  createCommitmentId,
  createEntityId,
  createMaintenanceRunId,
  type EntityId,
} from "../util/ids.js";

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
  appendManyCalls: Array<{
    inputs: unknown[];
    session?: string;
  }>;
  resolvedExternalSenders: unknown[];
  lookedUpExternalSenders: Array<{ source: string; externalId: string }>;
  externalSenderIds: Map<string, EntityId>;
  ingestSessions: string[];
  extractOptions: unknown[];
  commitments: CommitmentRecord[];
  commitmentAdds: unknown[];
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

function testCommitment(overrides: Partial<CommitmentRecord> = {}): CommitmentRecord {
  return commitmentSchema.parse({
    id: overrides.id ?? createCommitmentId(),
    record_version: overrides.record_version ?? 1,
    type: overrides.type ?? "preference",
    kind: overrides.kind ?? "participant_preference",
    enforcement_class: overrides.enforcement_class ?? "advisory",
    critical_domain: overrides.critical_domain ?? null,
    directive_family: overrides.directive_family ?? "concise_replies",
    closure_pressure_relevance: overrides.closure_pressure_relevance ?? "neutral",
    directive: overrides.directive ?? "Keep replies concise.",
    priority: overrides.priority ?? 5,
    made_to_entity: overrides.made_to_entity ?? null,
    restricted_audience: overrides.restricted_audience ?? null,
    about_entity: overrides.about_entity ?? null,
    committed_by_entity_id: overrides.committed_by_entity_id ?? null,
    provenance: overrides.provenance ?? { kind: "manual" },
    source_stream_entry_ids: overrides.source_stream_entry_ids,
    created_at: overrides.created_at ?? 100,
    expires_at: overrides.expires_at ?? null,
    expired_at: overrides.expired_at ?? null,
    revoked_at: overrides.revoked_at ?? null,
    revoked_reason: overrides.revoked_reason ?? null,
    revoke_provenance: overrides.revoke_provenance ?? null,
    superseded_by: overrides.superseded_by ?? null,
    canonicalized_by_artifact_entry_id: overrides.canonicalized_by_artifact_entry_id ?? null,
    last_reinforced_at: overrides.last_reinforced_at ?? 100,
  });
}

function stubBorg(rec: Recorder): Borg {
  return {
    stream: {
      append: async (input: { content: string }) => ({ timestamp: 1000, content: input.content }),
      appendMany: async (inputs: unknown[], options?: { session?: string }) => {
        rec.appendMany = { inputs, session: options?.session };
        rec.appendManyCalls.push(rec.appendMany);
        return inputs.map((input, index) => ({
          id: `strm_${String(index).padStart(16, "a")}`,
          kind: (input as { kind?: string }).kind,
        }));
      },
    },
    entities: {
      resolveExternal: (input: { source: string; externalId: string }) => {
        rec.resolvedExternalSenders.push(input);
        const existing = rec.externalSenderIds.get(input.externalId);

        if (existing !== undefined) {
          return existing;
        }

        const entityId = createEntityId();
        rec.externalSenderIds.set(input.externalId, entityId);
        return entityId;
      },
      findByExternalId: (source: string, externalId: string) => {
        rec.lookedUpExternalSenders.push({ source, externalId });
        return rec.externalSenderIds.get(externalId) ?? null;
      },
      get: (id: EntityId) =>
        [...rec.externalSenderIds.values()].some((entityId) => entityId === id)
          ? {
              id,
              canonical_name: "Known sender",
              aliases: [],
              kind: "person",
              borg_role: null,
              created_at: 1,
            }
          : null,
    },
    identity: {
      addCommitment: (input: {
        type: CommitmentRecord["type"];
        kind: CommitmentRecord["kind"];
        enforcementClass: CommitmentRecord["enforcement_class"];
        criticalDomain: CommitmentRecord["critical_domain"];
        directiveFamily: string;
        directive: string;
        priority: number;
        restrictedAudience: EntityId | null;
      }) => {
        rec.commitmentAdds.push(input);
        const commitment = testCommitment({
          type: input.type,
          kind: input.kind,
          enforcement_class: input.enforcementClass,
          critical_domain: input.criticalDomain,
          directive_family: input.directiveFamily,
          directive: input.directive,
          priority: input.priority,
          restricted_audience: input.restrictedAudience,
          created_at: 100 + rec.commitments.length,
          last_reinforced_at: 100 + rec.commitments.length,
        });
        rec.commitments.push(commitment);
        return commitment;
      },
    },
    commitments: {
      get: (id: CommitmentRecord["id"]) =>
        rec.commitments.find((commitment) => commitment.id === id) ?? null,
      list: (options?: { activeOnly?: boolean; audienceEntityId?: EntityId | null }) =>
        rec.commitments.filter((commitment) => {
          if (
            options?.activeOnly === true &&
            (commitment.revoked_at !== null ||
              commitment.expired_at !== null ||
              commitment.superseded_by !== null)
          ) {
            return false;
          }

          if (options?.audienceEntityId === undefined) {
            return true;
          }

          const scope = commitment.restricted_audience ?? commitment.made_to_entity;
          return scope === null || scope === options.audienceEntityId;
        }),
      revoke: (id: CommitmentRecord["id"], reason: string) => {
        const index = rec.commitments.findIndex((commitment) => commitment.id === id);
        const current = rec.commitments[index];
        if (current === undefined) {
          return null;
        }
        const revoked = testCommitment({
          ...current,
          record_version: (current.record_version ?? 1) + 1,
          revoked_at: 500,
          revoked_reason: reason,
          revoke_provenance: { kind: "manual" },
        });
        rec.commitments[index] = revoked;
        return revoked;
      },
    },
    episodic: {
      // Real facade returns numeric counts.
      extract: async (options: unknown) => {
        rec.extractOptions.push(options);
        return { inserted: 1, updated: 0, skipped: 0 };
      },
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
  const rec: Recorder = {
    tenants: [],
    exclusives: [],
    inspectIds: [],
    appendManyCalls: [],
    resolvedExternalSenders: [],
    lookedUpExternalSenders: [],
    externalSenderIds: new Map(),
    ingestSessions: [],
    extractOptions: [],
    commitments: [],
    commitmentAdds: [],
  };
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

async function del(base: string, path: string, token?: string) {
  const headers: Record<string, string> = {};
  if (token !== undefined) {
    headers["x-borg-token"] = token;
  }
  return fetch(`${base}${path}`, { method: "DELETE", headers });
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

    const maintenanceProbe = await requestRaw(
      base,
      "/memory/nope/../maintenance?tenant=acme&mode=light&dryRun=0",
      { method: "POST", token: TOKEN },
    );
    expect(maintenanceProbe.status).toBe(404);
    expect(maintenanceProbe.body).toEqual({ error: "not found" });
    expect(rec.tenants).toEqual([]);
  });

  it("authenticates and accepts a detached maintenance run from query parameters", async () => {
    const { pool } = recordingPool();
    const runId = createMaintenanceRunId();
    const starts: unknown[] = [];
    const maintenanceCoordinator = {
      tryReserve: (input: unknown) => {
        starts.push(input);
        return {
          status: "accepted",
          runId,
          completion: new Promise(() => {}),
        };
      },
      startReserved: (tenant: string, startedRunId: string) => {
        starts.push({ scheduled: [tenant, startedRunId] });
        return true;
      },
      hasReservation: () => true,
      cancelReservation: () => true,
      getStatus: () => ({ current: null, last: null }),
    } as unknown as NonNullable<MemoryHandlerOptions["maintenanceCoordinator"]>;
    const base = await start(pool, TOKEN, { maintenanceCoordinator });

    const unauthorized = await post(
      base,
      "/memory/maintenance?tenant=acme&mode=heavy&dryRun=1",
      {},
    );
    expect(unauthorized.status).toBe(401);
    expect(starts).toEqual([]);

    const accepted = await post(
      base,
      "/memory/maintenance?tenant=acme&mode=heavy&dryRun=1",
      {},
      TOKEN,
    );
    expect(accepted.status).toBe(202);
    expect(await accepted.json()).toEqual({ run_id: runId });
    expect(starts).toEqual([
      { tenant: "acme", mode: "heavy", dryRun: true },
      { scheduled: ["acme", runId] },
    ]);
  });

  it("holds a synchronous reservation during readiness and rejects a racing POST", async () => {
    const runId = createMaintenanceRunId();
    let active = false;
    let scheduled = false;
    let readinessStarted!: () => void;
    const startedReadiness = new Promise<void>((resolve) => {
      readinessStarted = resolve;
    });
    let releaseReadiness!: () => void;
    const readinessGate = new Promise<void>((resolve) => {
      releaseReadiness = resolve;
    });
    const pool: MemoryPool = {
      async withTenant(_tenant, fn) {
        readinessStarted();
        await readinessGate;
        return fn({} as Borg);
      },
    };
    const maintenanceCoordinator = {
      tryReserve: () => {
        if (active) {
          return { status: "conflict" as const, runId };
        }
        active = true;
        return { status: "accepted" as const, runId, completion: new Promise(() => {}) };
      },
      startReserved: () => {
        scheduled = true;
        return true;
      },
      hasReservation: () => active,
      cancelReservation: () => {
        active = false;
        return true;
      },
      getStatus: () => ({ current: null, last: null }),
    } as unknown as NonNullable<MemoryHandlerOptions["maintenanceCoordinator"]>;
    const base = await start(pool, TOKEN, { maintenanceCoordinator });
    const path = "/memory/maintenance?tenant=acme&mode=light&dryRun=0";

    const firstResponse = post(base, path, {}, TOKEN);
    await startedReadiness;
    const racing = await post(base, path, {}, TOKEN);

    expect(racing.status).toBe(409);
    expect(await racing.json()).toEqual({
      error: "maintenance already running",
      run_id: runId,
    });
    expect(scheduled).toBe(false);

    releaseReadiness();
    const accepted = await firstResponse;
    expect(accepted.status).toBe(202);
    expect(await accepted.json()).toEqual({ run_id: runId });
    expect(scheduled).toBe(true);
  });

  it("clears a reservation and returns 503 when tenant readiness fails", async () => {
    const runId = createMaintenanceRunId();
    const cancellations: unknown[] = [];
    const pool: MemoryPool = {
      async withTenant() {
        throw new Error("tenant config is invalid");
      },
    };
    const maintenanceCoordinator = {
      tryReserve: () => ({
        status: "accepted" as const,
        runId,
        completion: new Promise(() => {}),
      }),
      startReserved: () => true,
      hasReservation: () => true,
      cancelReservation: (tenant: string, cancelledRunId: string) => {
        cancellations.push([tenant, cancelledRunId]);
        return true;
      },
      getStatus: () => ({ current: null, last: null }),
    } as unknown as NonNullable<MemoryHandlerOptions["maintenanceCoordinator"]>;
    const base = await start(pool, TOKEN, { maintenanceCoordinator });

    const response = await post(
      base,
      "/memory/maintenance?tenant=acme&mode=heavy&dryRun=1",
      {},
      TOKEN,
    );

    expect(response.status).toBe(503);
    expect(await response.json()).toEqual({ error: "maintenance tenant unavailable" });
    expect(cancellations).toEqual([["acme", runId]]);
  });

  it("maps maintenance conflicts and disabled configuration to explicit statuses", async () => {
    const { pool } = recordingPool();
    const runId = createMaintenanceRunId();
    let outcome: "conflict" | "disabled" = "conflict";
    const maintenanceCoordinator = {
      tryReserve: () =>
        outcome === "conflict" ? { status: "conflict", runId } : { status: "disabled" },
      getStatus: () => ({ current: null, last: null }),
    } as unknown as NonNullable<MemoryHandlerOptions["maintenanceCoordinator"]>;
    const base = await start(pool, TOKEN, { maintenanceCoordinator });
    const path = "/memory/maintenance?tenant=acme&mode=light&dryRun=0";

    const conflict = await post(base, path, {}, TOKEN);
    expect(conflict.status).toBe(409);
    expect(await conflict.json()).toEqual({
      error: "maintenance already running",
      run_id: runId,
    });

    outcome = "disabled";
    const disabled = await post(base, path, {}, TOKEN);
    expect(disabled.status).toBe(503);
    expect(await disabled.json()).toEqual({ error: "maintenance disabled" });
  });

  it("validates the maintenance query before starting a run", async () => {
    const { pool } = recordingPool();
    let starts = 0;
    const maintenanceCoordinator = {
      tryReserve: () => {
        starts += 1;
        return { status: "disabled" };
      },
      getStatus: () => ({ current: null, last: null }),
    } as unknown as NonNullable<MemoryHandlerOptions["maintenanceCoordinator"]>;
    const base = await start(pool, TOKEN, { maintenanceCoordinator });

    expect((await post(base, "/memory/maintenance", {}, TOKEN)).status).toBe(400);
    expect(
      (await post(base, "/memory/maintenance?tenant=acme&mode=nope&dryRun=0", {}, TOKEN)).status,
    ).toBe(400);
    expect(
      (await post(base, "/memory/maintenance?tenant=acme&mode=light&dryRun=yes", {}, TOKEN)).status,
    ).toBe(400);
    expect(
      (
        await post(
          base,
          "/memory/maintenance?tenant=acme&tenant=other&mode=light&dryRun=0",
          {},
          TOKEN,
        )
      ).status,
    ).toBe(400);
    expect(starts).toBe(0);
  });

  it("returns coordinator status without opening a pooled being", async () => {
    const { pool, rec } = recordingPool();
    const runId = createMaintenanceRunId();
    const maintenanceCoordinator = {
      tryReserve: () => ({ status: "disabled" }),
      getStatus: (tenant: string) => ({
        current: { tenant, run_id: runId, state: "running" },
        last: null,
      }),
    } as unknown as NonNullable<MemoryHandlerOptions["maintenanceCoordinator"]>;
    const base = await start(pool, TOKEN, { maintenanceCoordinator });

    const response = await get(base, "/memory/maintenance/status?tenant=acme", TOKEN);
    expect(response.status).toBe(200);
    expect(await response.json()).toEqual({
      ok: true,
      tenant: "acme",
      current: { tenant: "acme", run_id: runId, state: "running" },
      last: null,
    });
    expect(rec.tenants).toEqual([]);
  });

  it("wraps maintenance audit listing and exclusive revert", async () => {
    const runId = createMaintenanceRunId();
    const calls: Array<{ tenant: string; exclusive: boolean | undefined }> = [];
    const auditCalls: unknown[] = [];
    const borg = {
      audit: {
        list: (options: unknown) => {
          auditCalls.push(["list", options]);
          return [{ id: 12, run_id: runId }];
        },
        revert: (auditId: number, revertedBy: string) => {
          auditCalls.push(["revert", auditId, revertedBy]);
          return Promise.resolve({ id: auditId, run_id: runId, reverted_by: revertedBy });
        },
      },
    } as unknown as Borg;
    const pool: MemoryPool = {
      async withTenant(tenant, fn, options) {
        calls.push({ tenant, exclusive: options?.exclusive });
        return fn(borg);
      },
    };
    const base = await start(pool);

    const listed = await get(base, `/memory/maintenance/audit?tenant=acme&run_id=${runId}`, TOKEN);
    expect(listed.status).toBe(200);
    expect(await listed.json()).toEqual({
      ok: true,
      tenant: "acme",
      run_id: runId,
      audit: [{ id: 12, run_id: runId }],
    });

    const reverted = await post(
      base,
      "/memory/maintenance/revert?tenant=acme&audit_id=12",
      {},
      TOKEN,
    );
    expect(reverted.status).toBe(200);
    expect(await reverted.json()).toEqual({
      ok: true,
      tenant: "acme",
      audit: { id: 12, run_id: runId, reverted_by: "memory-sidecar" },
    });
    expect(auditCalls).toEqual([
      ["list", { runId }],
      ["revert", 12, "memory-sidecar"],
    ]);
    expect(calls).toEqual([
      { tenant: "acme", exclusive: undefined },
      { tenant: "acme", exclusive: true },
    ]);
  });

  it("returns 404 for a missing maintenance audit and validates audit queries", async () => {
    const calls: Array<{ tenant: string; exclusive: boolean | undefined }> = [];
    const borg = {
      audit: {
        list: () => [],
        revert: () => Promise.resolve(null),
      },
    } as unknown as Borg;
    const pool: MemoryPool = {
      async withTenant(tenant, fn, options) {
        calls.push({ tenant, exclusive: options?.exclusive });
        return fn(borg);
      },
    };
    const base = await start(pool);

    expect(
      (await get(base, "/memory/maintenance/audit?tenant=acme&run_id=bad", TOKEN)).status,
    ).toBe(400);
    expect(
      (await post(base, "/memory/maintenance/revert?tenant=acme&audit_id=bad", {}, TOKEN)).status,
    ).toBe(400);
    const missing = await post(
      base,
      "/memory/maintenance/revert?tenant=acme&audit_id=99",
      {},
      TOKEN,
    );
    expect(missing.status).toBe(404);
    expect(await missing.json()).toEqual({ error: "audit record not found" });
    expect(calls).toEqual([{ tenant: "acme", exclusive: true }]);
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

  it("lists applicable active commitments with critical-first ordering and a bounded response", async () => {
    const { pool, rec } = recordingPool();
    const audienceId = createEntityId();
    const otherAudienceId = createEntityId();
    rec.externalSenderIds.set("known-audience", audienceId);
    rec.commitments.push(
      testCommitment({
        directive_family: "global_advisory",
        directive: "Global advisory",
        priority: 100,
        enforcement_class: "advisory",
      }),
      testCommitment({
        type: "boundary",
        kind: "boundary",
        directive_family: "global_critical",
        directive: "Global critical",
        priority: 1,
        enforcement_class: "critical",
        critical_domain: "privacy",
      }),
      testCommitment({
        type: "boundary",
        kind: "audience_rule",
        directive_family: "audience_critical",
        directive: "Audience critical",
        priority: 9,
        enforcement_class: "critical",
        critical_domain: "audience_scope",
        restricted_audience: audienceId,
      }),
      testCommitment({
        type: "boundary",
        kind: "audience_rule",
        directive_family: "other_audience",
        directive: "Other audience only",
        priority: 99,
        enforcement_class: "critical",
        critical_domain: "audience_scope",
        restricted_audience: otherAudienceId,
      }),
      testCommitment({
        directive_family: "retired",
        directive: "Retired",
        revoked_at: 200,
        revoked_reason: "done",
        revoke_provenance: { kind: "manual" },
      }),
    );
    const base = await start(pool);
    const response = await get(
      base,
      `/memory/commitments?tenant=acme&audience=${audienceId}`,
      TOKEN,
    );

    expect(response.status).toBe(200);
    const body = (await response.json()) as {
      audience_entity_id: string;
      commitments: Array<Record<string, unknown>>;
      truncated: boolean;
    };
    expect(body.audience_entity_id).toBe(audienceId);
    expect(body.commitments.map((commitment) => commitment.family)).toEqual([
      "audience_critical",
      "global_critical",
      "global_advisory",
    ]);
    expect(body.commitments[0]).toMatchObject({
      type: "boundary",
      kind: "audience_rule",
      enforcement_class: "critical",
      critical_domain: "audience_scope",
      directive: "Audience critical",
      family: "audience_critical",
      priority: 9,
      audience_entity_id: audienceId,
    });
    expect(body.truncated).toBe(false);

    rec.commitments = Array.from({ length: 101 }, (_, index) =>
      testCommitment({
        directive_family: `bounded_${index}`,
        directive: `Bounded ${index}`,
        priority: index,
        created_at: index,
        last_reinforced_at: index,
      }),
    );
    const bounded = await get(base, "/memory/commitments?tenant=acme", TOKEN);
    const boundedBody = (await bounded.json()) as {
      commitments: unknown[];
      truncated: boolean;
    };
    expect(boundedBody.commitments).toHaveLength(100);
    expect(boundedBody.truncated).toBe(true);
  });

  it("resolves commitment audience scope from the append-turn sender external id", async () => {
    const { pool, rec } = recordingPool();
    const audienceId = createEntityId();
    const otherAudienceId = createEntityId();
    const externalId = "platform/user 42";
    rec.externalSenderIds.set(externalId, audienceId);
    rec.commitments.push(
      testCommitment({
        directive_family: "tenant_wide",
        directive: "Tenant-wide rule",
        priority: 10,
      }),
      testCommitment({
        directive_family: "sender_scoped",
        directive: "Sender-scoped rule",
        restricted_audience: audienceId,
      }),
      testCommitment({
        directive_family: "other_sender",
        directive: "Other sender rule",
        restricted_audience: otherAudienceId,
      }),
    );
    const base = await start(pool);
    const response = await get(
      base,
      `/memory/commitments?tenant=acme&audience_external_id=${encodeURIComponent(externalId)}`,
      TOKEN,
    );

    expect(response.status).toBe(200);
    expect(await response.json()).toMatchObject({
      ok: true,
      tenant: "acme",
      audience_entity_id: audienceId,
      audience_external_id: externalId,
      audience_resolved: true,
      commitments: [
        expect.objectContaining({ family: "tenant_wide" }),
        expect.objectContaining({ family: "sender_scoped" }),
      ],
      truncated: false,
    });
    expect(rec.lookedUpExternalSenders).toEqual([
      { source: "team-agent.sender", externalId },
    ]);
    expect(rec.resolvedExternalSenders).toEqual([]);
  });

  it("returns only tenant-wide commitments for an unknown external audience id", async () => {
    const { pool, rec } = recordingPool();
    rec.commitments.push(
      testCommitment({
        directive_family: "tenant_wide",
        directive: "Tenant-wide rule",
      }),
      testCommitment({
        directive_family: "known_sender_only",
        directive: "Known sender rule",
        restricted_audience: createEntityId(),
      }),
    );
    const base = await start(pool);
    const response = await get(
      base,
      "/memory/commitments?tenant=acme&audience_external_id=not-seen-yet",
      TOKEN,
    );

    expect(response.status).toBe(200);
    expect(await response.json()).toMatchObject({
      ok: true,
      tenant: "acme",
      audience_entity_id: null,
      audience_external_id: "not-seen-yet",
      audience_resolved: false,
      commitments: [expect.objectContaining({ family: "tenant_wide" })],
      truncated: false,
    });
    expect(rec.lookedUpExternalSenders).toEqual([
      { source: "team-agent.sender", externalId: "not-seen-yet" },
    ]);
  });

  it("rejects internal and external commitment audiences supplied together", async () => {
    const { pool, rec } = recordingPool();
    const base = await start(pool);
    const response = await get(
      base,
      `/memory/commitments?tenant=acme&audience=${createEntityId()}&audience_external_id=sender-1`,
      TOKEN,
    );

    expect(response.status).toBe(400);
    expect(await response.json()).toEqual({
      error: "'audience' and 'audience_external_id' are mutually exclusive",
    });
    expect(rec.tenants).toEqual([]);
    expect(rec.lookedUpExternalSenders).toEqual([]);
  });

  it("strictly validates and creates an operator-set audience commitment", async () => {
    const { pool, rec } = recordingPool();
    const audienceId = createEntityId();
    rec.externalSenderIds.set("known-audience", audienceId);
    const base = await start(pool);
    const payload = {
      tenant: "acme",
      type: "boundary",
      kind: "audience_rule",
      enforcement_class: "critical",
      critical_domain: "privacy",
      directive: "Never disclose private launch notes.",
      family: "Private Launch Notes",
      priority: 12,
      audience_entity_id: audienceId,
    };
    const response = await post(base, "/memory/commitments", payload, TOKEN);

    expect(response.status).toBe(201);
    expect(await response.json()).toMatchObject({
      ok: true,
      commitment: {
        type: "boundary",
        kind: "audience_rule",
        enforcement_class: "critical",
        critical_domain: "privacy",
        directive: "Never disclose private launch notes.",
        family: "private_launch_notes",
        priority: 12,
        audience_entity_id: audienceId,
      },
    });
    expect(rec.commitmentAdds).toHaveLength(1);
    expect(rec.exclusives).toEqual([true]);

    const unknownField = await post(
      base,
      "/memory/commitments",
      { ...payload, unexpected: true },
      TOKEN,
    );
    expect(unknownField.status).toBe(400);
    expect(await unknownField.json()).toEqual({ error: "invalid commitment body" });

    const invalidClassification = await post(
      base,
      "/memory/commitments",
      {
        ...payload,
        type: "rule",
        kind: "process_norm",
        critical_domain: "safety",
      },
      TOKEN,
    );
    expect(invalidClassification.status).toBe(400);
  });

  it("retires an active commitment and rejects invalid or inactive ids", async () => {
    const { pool, rec } = recordingPool();
    const commitment = testCommitment();
    rec.commitments.push(commitment);
    const base = await start(pool);

    expect(
      (await del(base, "/memory/commitments?tenant=acme&id=not-a-commitment", TOKEN)).status,
    ).toBe(400);

    const retired = await del(base, `/memory/commitments?tenant=acme&id=${commitment.id}`, TOKEN);
    expect(retired.status).toBe(200);
    expect(await retired.json()).toMatchObject({
      ok: true,
      commitment: {
        id: commitment.id,
      },
    });
    expect(rec.commitments[0]?.revoked_reason).toBe("retired_by_operator");
    expect(rec.exclusives.at(-1)).toBe(true);

    expect(
      (await del(base, `/memory/commitments?tenant=acme&id=${commitment.id}`, TOKEN)).status,
    ).toBe(409);
  });

  it("validates commitment audience ids before opening a tenant", async () => {
    const { pool, rec } = recordingPool();
    const base = await start(pool);

    expect((await get(base, "/memory/commitments?tenant=acme&audience=bad", TOKEN)).status).toBe(
      400,
    );
    expect(rec.tenants).toEqual([]);
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
    expect(rec.extractOptions).toEqual([
      {
        sinceTs: 1000,
        bypassSalienceGate: true,
      },
    ]);
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

  it("resolves optional sender identities and stamps only their user stream entries", async () => {
    const { pool, rec } = recordingPool();
    const base = await start(pool);

    for (const sender of [
      { external_id: "platform-alice", display_name: "Alice Nowak" },
      { external_id: "platform-bob", display_name: "Bob Chen" },
    ]) {
      const response = await post(
        base,
        "/memory/append-turn",
        {
          tenant: "acme",
          session: "shared-room",
          user: `message from ${sender.display_name}`,
          assistant: "acknowledged",
          sender,
        },
        TOKEN,
      );

      expect(response.status).toBe(200);
      await response.json();
    }

    const aliceId = rec.externalSenderIds.get("platform-alice");
    const bobId = rec.externalSenderIds.get("platform-bob");

    expect(aliceId).toBeDefined();
    expect(bobId).toBeDefined();
    expect(aliceId).not.toBe(bobId);
    expect(rec.resolvedExternalSenders).toEqual([
      {
        source: "team-agent.sender",
        externalId: "platform-alice",
        canonicalName: "Alice Nowak",
        kind: "person",
        provenance: "transport_sender",
      },
      {
        source: "team-agent.sender",
        externalId: "platform-bob",
        canonicalName: "Bob Chen",
        kind: "person",
        provenance: "transport_sender",
      },
    ]);
    expect(rec.appendManyCalls).toHaveLength(2);
    expect(rec.appendManyCalls[0]?.inputs).toEqual([
      { kind: "user_msg", content: "message from Alice Nowak", sender_entity_id: aliceId },
      { kind: "agent_msg", content: "acknowledged" },
    ]);
    expect(rec.appendManyCalls[1]?.inputs).toEqual([
      { kind: "user_msg", content: "message from Bob Chen", sender_entity_id: bobId },
      { kind: "agent_msg", content: "acknowledged" },
    ]);
  });

  it("rejects malformed optional sender objects before touching the tenant", async () => {
    const { pool, rec } = recordingPool();
    const base = await start(pool);
    const response = await post(
      base,
      "/memory/append-turn",
      {
        tenant: "acme",
        session: "room",
        user: "hello",
        assistant: "hi",
        sender: { external_id: "platform-alice", display_name: "" },
      },
      TOKEN,
    );

    expect(response.status).toBe(400);
    expect(await response.json()).toEqual({
      error: "invalid 'sender'; expected non-empty 'external_id' and 'display_name'",
    });
    expect(rec.tenants).toEqual([]);
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
