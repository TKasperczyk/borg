// HTTP request handler for the borg memory sidecar: a thin, tenant-routed wrapper
// over BorgPool that exposes long-term memory to an external (e.g. Python) service.
//
//   POST /memory/remember    { tenant, content, author? }          -> append + extract episode(s)
//   POST /memory/append-turn { tenant, session, user, assistant, sender? } -> append + async extract
//   POST /memory/recall      { tenant, query, limit? }             -> semantic episodic search
//   GET  /memory/episodes?tenant=<id>&limit=<n>&cursor=<c> -> list raw episodic bank
//   GET  /memory/episodes/{id}?tenant=<id>                  -> inspect one raw episode
//   GET  /memory/trace?tenant=<id>&since=<ts>                -> inspect recall trace buffer
//   POST /memory/maintenance?tenant=<id>&mode=<light|heavy>&dryRun=<0|1>
//   GET  /memory/maintenance/status?tenant=<id>
//   GET  /memory/maintenance/audit?tenant=<id>&run_id=<id>
//   POST /memory/maintenance/revert?tenant=<id>&audit_id=<id>
//   GET  /healthz                                           -> liveness (no auth)
//
// Recall is tenant-wide by design: the pool routes to one being per tenant and,
// within a being, recall is global (borg's "recall is global to the being", with
// being == tenant). All authenticated routes require the shared x-borg-token.

import { createHash, timingSafeEqual } from "node:crypto";
import type { IncomingMessage, ServerResponse } from "node:http";

import type { Borg } from "../borg.js";
import type { Episode } from "../memory/episodic/index.js";
import type { StreamEntryInput } from "../stream/index.js";
import {
  parseAuditId,
  parseEpisodeId,
  parseMaintenanceRunId,
  parseSessionId,
  type EpisodeId,
  type SessionId,
} from "../util/ids.js";
import type { MemoryMaintenanceCoordinator } from "./memory-maintenance.js";
import type { MemoryTraceRegistry } from "./memory-trace.js";

// Mirror of BorgPool's DEFAULT_TENANT_ID_PATTERN so the handler returns a clean
// 400 for a malformed tenant id at the boundary, rather than relying on (and
// risking a message leak from) the pool's ConfigError deeper in.
const TENANT_ID_RE = /^[a-z0-9][a-z0-9_-]{0,63}$/;

// The handler only needs withTenant from the pool; typing it structurally keeps
// the handler unit-testable with a stub.
export type MemoryPool = {
  withTenant<T>(
    tenantId: string,
    fn: (borg: Borg) => T | Promise<T>,
    opts?: { exclusive?: boolean },
  ): Promise<T>;
};

export type MemoryHandlerOptions = {
  pool: MemoryPool;
  // Shared bearer presented as the x-borg-token header. Required; an empty token
  // rejects every authenticated request (fail closed).
  token: string;
  maxBodyBytes?: number;
  maxRecallLimit?: number;
  traceRegistry?: MemoryTraceRegistry;
  maintenanceCoordinator?: Pick<
    MemoryMaintenanceCoordinator,
    "cancelReservation" | "getStatus" | "hasReservation" | "startReserved" | "tryReserve"
  >;
};

type RequestHandler = (req: IncomingMessage, res: ServerResponse) => void;

const DEFAULT_MAX_BODY_BYTES = 64 * 1024;
const DEFAULT_MAX_RECALL_LIMIT = 50;
const DEFAULT_EPISODE_LIST_LIMIT = 20;
const MAX_EPISODE_LIST_LIMIT = 100;
const APPEND_TURN_SENDER_EXTERNAL_ID_SOURCE = "team-agent.sender";

class PayloadTooLargeError extends Error {}

function send(res: ServerResponse, status: number, body: unknown): void {
  res.writeHead(status, { "content-type": "application/json" });
  res.end(JSON.stringify(body));
}

function readBody(req: IncomingMessage, maxBytes: number): Promise<string> {
  return new Promise((resolve, reject) => {
    let size = 0;
    let aborted = false;
    const chunks: Buffer[] = [];
    req.on("data", (chunk: Buffer) => {
      if (aborted) {
        return; // keep draining to let the response flush; stop buffering
      }
      size += chunk.length;
      if (size > maxBytes) {
        aborted = true;
        reject(new PayloadTooLargeError("request body too large"));
        return;
      }
      chunks.push(chunk);
    });
    req.on("end", () => {
      if (!aborted) {
        resolve(Buffer.concat(chunks).toString("utf8"));
      }
    });
    req.on("error", reject);
  });
}

// Constant-time bearer check. Fail closed on an empty configured token, a missing
// header, or a folded/duplicated header (array) — duplicate-header semantics are
// proxy-dependent and not worth trusting.
function tokenMatches(provided: string | string[] | undefined, expected: string): boolean {
  if (expected === "" || typeof provided !== "string") {
    return false;
  }
  const a = Buffer.from(provided);
  const b = Buffer.from(expected);
  return a.length === b.length && timingSafeEqual(a, b);
}

function asString(value: unknown): string {
  return typeof value === "string" ? value.trim() : "";
}

function asContentString(value: unknown): string {
  return typeof value === "string" ? value : "";
}

type AppendTurnSender = {
  externalId: string;
  displayName: string;
};

function parseAppendTurnSender(
  value: unknown,
): { valid: true; sender: AppendTurnSender | null } | { valid: false } {
  if (value === undefined) {
    return { valid: true, sender: null };
  }

  if (typeof value !== "object" || value === null || Array.isArray(value)) {
    return { valid: false };
  }

  const sender = value as Record<string, unknown>;
  const externalId = asString(sender.external_id);
  const displayName = asString(sender.display_name);

  if (externalId.length === 0 || displayName.length === 0) {
    return { valid: false };
  }

  return {
    valid: true,
    sender: {
      externalId,
      displayName,
    },
  };
}

function parseRawRequestTarget(rawUrl: string): {
  rawPath: string;
  searchParams: URLSearchParams;
} {
  const queryStart = rawUrl.indexOf("?");
  if (queryStart === -1) {
    return { rawPath: rawUrl, searchParams: new URLSearchParams() };
  }

  return {
    rawPath: rawUrl.slice(0, queryStart),
    searchParams: new URLSearchParams(rawUrl.slice(queryStart + 1)),
  };
}

function validateTenantForResponse(res: ServerResponse, tenant: string): boolean {
  if (tenant === "") {
    send(res, 400, { error: "missing 'tenant'" });
    return false;
  }
  if (!TENANT_ID_RE.test(tenant)) {
    send(res, 400, { error: "invalid 'tenant'" });
    return false;
  }

  return true;
}

function requiredSingleQueryValue(
  res: ServerResponse,
  searchParams: URLSearchParams,
  name: string,
): string | null {
  const values = searchParams.getAll(name);
  if (values.length !== 1 || values[0]?.trim() === "") {
    send(res, 400, { error: `missing or duplicate '${name}'` });
    return null;
  }
  return values[0]!.trim();
}

function episodeListLimitFromQuery(searchParams: URLSearchParams): number {
  const raw = searchParams.get("limit");
  const rawLimit = raw === null || raw.trim() === "" ? DEFAULT_EPISODE_LIST_LIMIT : Number(raw);
  const finiteLimit = Number.isFinite(rawLimit) ? rawLimit : DEFAULT_EPISODE_LIST_LIMIT;
  return Math.max(1, Math.min(MAX_EPISODE_LIST_LIMIT, Math.floor(finiteLimit)));
}

function episodeListCursorFromQuery(searchParams: URLSearchParams): string | undefined {
  const raw = searchParams.get("cursor");
  return raw === null || raw.trim() === "" ? undefined : raw;
}

function traceSinceFromQuery(searchParams: URLSearchParams): number | null {
  const raw = searchParams.get("since");
  if (raw === null || raw.trim() === "") {
    return 0;
  }

  const since = Number(raw);
  return Number.isFinite(since) ? since : null;
}

function parseEpisodeIdFromPath(pathname: string): EpisodeId | null | undefined {
  const segments = pathname.split("/");
  if (segments.length !== 4 || segments[1] !== "memory" || segments[2] !== "episodes") {
    return undefined;
  }

  try {
    return parseEpisodeId(segments[3] ?? "");
  } catch {
    return null;
  }
}

function projectEpisodeForList(episode: Episode): {
  id: Episode["id"];
  title: string;
  narrative: string;
  significance: number;
  tags: string[];
  source_stream_ids: Episode["source_stream_ids"];
} {
  return {
    id: episode.id,
    title: episode.title,
    narrative: episode.narrative,
    significance: episode.significance,
    tags: episode.tags,
    source_stream_ids: episode.source_stream_ids,
  };
}

function episodeWithoutEmbedding(episode: Episode): Omit<Episode, "embedding"> {
  const { embedding: _embedding, ...rest } = episode;
  return rest;
}

function errorCode(error: unknown): unknown {
  return error !== null && typeof error === "object" && "code" in error
    ? (error as { code?: unknown }).code
    : undefined;
}

function isInvalidEpisodeCursorError(error: unknown): boolean {
  return errorCode(error) === "EPISODE_CURSOR_INVALID";
}

function sessionFromCaller(value: string): SessionId {
  try {
    return parseSessionId(value);
  } catch {
    const hash = createHash("sha256").update(value).digest("hex").slice(0, 16);
    return parseSessionId(`sess_${hash}`);
  }
}

function scheduleIngestion(pool: MemoryPool, tenant: string, session: SessionId): void {
  void pool
    .withTenant(tenant, (borg) => borg.episodic.ingest({ session }))
    .catch((error: unknown) => {
      console.error(`memory-sidecar: background ingestion failed for tenant "${tenant}"`, error);
    });
}

export function createMemoryHandler(options: MemoryHandlerOptions): RequestHandler {
  const { pool, token, traceRegistry, maintenanceCoordinator } = options;
  const maxBodyBytes = options.maxBodyBytes ?? DEFAULT_MAX_BODY_BYTES;
  const maxRecallLimit = options.maxRecallLimit ?? DEFAULT_MAX_RECALL_LIMIT;
  let recallTraceSequence = 0;

  const nextRecallTraceTurnId = (tenant: string): string => {
    recallTraceSequence += 1;
    return `sidecar_recall:${tenant}:${Date.now()}:${recallTraceSequence}`;
  };

  async function handle(req: IncomingMessage, res: ServerResponse): Promise<void> {
    const method = req.method ?? "GET";
    const { rawPath, searchParams } = parseRawRequestTarget(req.url ?? "/");

    if (method === "GET" && rawPath === "/healthz") {
      send(res, 200, { ok: true });
      return;
    }

    if (!tokenMatches(req.headers["x-borg-token"], token)) {
      send(res, 401, { error: "unauthorized" });
      return;
    }

    if (method === "POST" && rawPath === "/memory/maintenance") {
      const tenant = requiredSingleQueryValue(res, searchParams, "tenant");
      if (tenant === null) {
        return;
      }
      const mode = requiredSingleQueryValue(res, searchParams, "mode");
      if (mode === null) {
        return;
      }
      const dryRun = requiredSingleQueryValue(res, searchParams, "dryRun");
      if (dryRun === null) {
        return;
      }
      if (!validateTenantForResponse(res, tenant)) {
        return;
      }
      if (mode !== "light" && mode !== "heavy") {
        send(res, 400, { error: "invalid 'mode'" });
        return;
      }
      if (dryRun !== "0" && dryRun !== "1") {
        send(res, 400, { error: "invalid 'dryRun'" });
        return;
      }
      if (maintenanceCoordinator === undefined) {
        send(res, 503, { error: "maintenance unavailable" });
        return;
      }

      const started = maintenanceCoordinator.tryReserve({
        tenant,
        mode,
        dryRun: dryRun === "1",
      });
      switch (started.status) {
        case "accepted": {
          try {
            // Admission is already reserved, so racing POSTs see 409 while the
            // tenant is opened and its pool initializer establishes readiness.
            await pool.withTenant(tenant, () => undefined);
          } catch {
            maintenanceCoordinator.cancelReservation(tenant, started.runId);
            send(res, 503, { error: "maintenance tenant unavailable" });
            return;
          }
          if (!maintenanceCoordinator.hasReservation(tenant, started.runId)) {
            send(res, 503, { error: "maintenance shutting down" });
            return;
          }
          try {
            send(res, 202, { run_id: started.runId });
          } finally {
            // Scheduling happens only after the acceptance response is handed
            // to the server, and remains detached from the client connection.
            maintenanceCoordinator.startReserved(tenant, started.runId);
          }
          return;
        }
        case "conflict":
          send(res, 409, {
            error: "maintenance already running",
            run_id: started.runId,
          });
          return;
        case "disabled":
          send(res, 503, { error: "maintenance disabled" });
          return;
        case "shutting_down":
          send(res, 503, { error: "maintenance shutting down" });
          return;
      }
    }

    if (method === "POST" && rawPath === "/memory/maintenance/revert") {
      const tenant = requiredSingleQueryValue(res, searchParams, "tenant");
      if (tenant === null) {
        return;
      }
      const auditIdRaw = requiredSingleQueryValue(res, searchParams, "audit_id");
      if (auditIdRaw === null) {
        return;
      }
      if (!validateTenantForResponse(res, tenant)) {
        return;
      }

      let auditId;
      try {
        auditId = parseAuditId(auditIdRaw);
      } catch {
        send(res, 400, { error: "invalid 'audit_id'" });
        return;
      }

      try {
        const audit = await pool.withTenant(
          tenant,
          (borg) => borg.audit.revert(auditId, "memory-sidecar"),
          { exclusive: true },
        );
        if (audit === null) {
          send(res, 404, { error: "audit record not found" });
          return;
        }
        send(res, 200, { ok: true, tenant, audit });
      } catch (error) {
        console.error(`memory-sidecar: ${rawPath} failed for tenant "${tenant}"`, error);
        send(res, 500, { error: "internal error" });
      }
      return;
    }

    if (method === "GET") {
      const isTracePath = rawPath === "/memory/trace";
      const isEpisodeListPath = rawPath === "/memory/episodes";
      const isMaintenanceStatusPath = rawPath === "/memory/maintenance/status";
      const isMaintenanceAuditPath = rawPath === "/memory/maintenance/audit";
      const episodeId =
        isTracePath || isEpisodeListPath || isMaintenanceStatusPath || isMaintenanceAuditPath
          ? undefined
          : parseEpisodeIdFromPath(rawPath);
      if (
        !isTracePath &&
        !isEpisodeListPath &&
        !isMaintenanceStatusPath &&
        !isMaintenanceAuditPath &&
        episodeId === undefined
      ) {
        send(res, 404, { error: "not found" });
        return;
      }

      const strictMaintenanceQuery = isMaintenanceStatusPath || isMaintenanceAuditPath;
      const tenant = strictMaintenanceQuery
        ? requiredSingleQueryValue(res, searchParams, "tenant")
        : asString(searchParams.get("tenant"));
      if (tenant === null) {
        return;
      }
      if (!validateTenantForResponse(res, tenant)) {
        return;
      }

      try {
        if (isMaintenanceStatusPath) {
          if (maintenanceCoordinator === undefined) {
            send(res, 503, { error: "maintenance unavailable" });
            return;
          }
          const status = maintenanceCoordinator.getStatus(tenant);
          send(res, 200, { ok: true, tenant, ...status });
          return;
        }

        if (isMaintenanceAuditPath) {
          const runIdRaw = requiredSingleQueryValue(res, searchParams, "run_id");
          if (runIdRaw === null) {
            return;
          }
          let runId;
          try {
            runId = parseMaintenanceRunId(runIdRaw);
          } catch {
            send(res, 400, { error: "invalid 'run_id'" });
            return;
          }
          const audit = await pool.withTenant(tenant, (borg) => borg.audit.list({ runId }));
          send(res, 200, { ok: true, tenant, run_id: runId, audit });
          return;
        }

        if (isTracePath) {
          const since = traceSinceFromQuery(searchParams);
          if (since === null) {
            send(res, 400, { error: "invalid 'since'" });
            return;
          }

          if (traceRegistry === undefined) {
            send(res, 200, { ok: true, tenant, events: [], disabled: true });
            return;
          }

          const result = traceRegistry.query(tenant, since);
          send(res, 200, {
            ok: true,
            tenant,
            events: result.events,
            nextSince: result.nextSince,
            truncated: result.truncated,
          });
          return;
        }

        if (rawPath === "/memory/episodes") {
          const limit = episodeListLimitFromQuery(searchParams);
          const cursor = episodeListCursorFromQuery(searchParams);
          const result = await pool.withTenant(tenant, (borg) =>
            borg.episodic.list({
              limit,
              ...(cursor === undefined ? {} : { cursor }),
            }),
          );

          send(res, 200, {
            ok: true,
            episodes: result.items.map((episode) => projectEpisodeForList(episode)),
            ...(result.nextCursor === undefined ? {} : { nextCursor: result.nextCursor }),
          });
          return;
        }

        if (episodeId === null) {
          send(res, 400, { error: "invalid episode id" });
          return;
        }
        if (episodeId !== undefined) {
          const episode = await pool.withTenant(tenant, (borg) => borg.episodic.inspect(episodeId));
          if (episode === null) {
            send(res, 404, { ok: false });
            return;
          }

          send(res, 200, { ok: true, episode: episodeWithoutEmbedding(episode) });
          return;
        }

        send(res, 404, { error: "not found" });
      } catch (error) {
        if (isInvalidEpisodeCursorError(error)) {
          send(res, 400, { error: "invalid 'cursor'" });
          return;
        }

        console.error(`memory-sidecar: ${rawPath} failed for tenant "${tenant}"`, error);
        send(res, 500, { error: "internal error" });
      }
      return;
    }

    if (
      method !== "POST" ||
      (rawPath !== "/memory/remember" &&
        rawPath !== "/memory/recall" &&
        rawPath !== "/memory/append-turn")
    ) {
      send(res, 404, { error: "not found" });
      return;
    }

    let body: Record<string, unknown>;
    try {
      const raw = await readBody(req, maxBodyBytes);
      const parsed: unknown = raw.trim() === "" ? {} : JSON.parse(raw);
      if (typeof parsed !== "object" || parsed === null || Array.isArray(parsed)) {
        send(res, 400, { error: "request body must be a JSON object" });
        return;
      }
      body = parsed as Record<string, unknown>;
    } catch (error) {
      if (error instanceof PayloadTooLargeError) {
        send(res, 413, { error: "request body too large" });
        return;
      }
      send(res, 400, { error: "invalid JSON body" });
      return;
    }

    const tenant = asString(body.tenant);
    if (!validateTenantForResponse(res, tenant)) {
      return;
    }

    try {
      if (rawPath === "/memory/remember") {
        const content = asString(body.content);
        if (content === "") {
          send(res, 400, { error: "missing 'content'" });
          return;
        }
        const author = asString(body.author);
        const text = author === "" ? content : `[${author}] ${content}`;
        // Exclusive: append + extract must run serialized per tenant, else two
        // concurrent remembers for one tenant interleave and each extract (with an
        // open-ended sinceTs) sweeps the other's just-appended entry -> duplicates.
        const extracted = await pool.withTenant(
          tenant,
          async (borg) => {
            const entry = await borg.stream.append({ kind: "user_msg", content: text });
            return borg.episodic.extract({ sinceTs: entry.timestamp });
          },
          { exclusive: true },
        );
        send(res, 200, { ok: true, extracted });
        return;
      }

      if (rawPath === "/memory/append-turn") {
        const sessionRaw = asString(body.session);
        if (sessionRaw === "") {
          send(res, 400, { error: "missing 'session'" });
          return;
        }
        const user = asContentString(body.user);
        if (user.trim() === "") {
          send(res, 400, { error: "missing 'user'" });
          return;
        }
        const assistant = asContentString(body.assistant);
        if (assistant.trim() === "") {
          send(res, 400, { error: "missing 'assistant'" });
          return;
        }
        const parsedSender = parseAppendTurnSender(body.sender);
        if (!parsedSender.valid) {
          send(res, 400, {
            error: "invalid 'sender'; expected non-empty 'external_id' and 'display_name'",
          });
          return;
        }

        const session = sessionFromCaller(sessionRaw);
        const entries = await pool.withTenant(
          tenant,
          (borg) => {
            const senderEntityId =
              parsedSender.sender === null
                ? undefined
                : borg.entities.resolveExternal({
                    source: APPEND_TURN_SENDER_EXTERNAL_ID_SOURCE,
                    externalId: parsedSender.sender.externalId,
                    canonicalName: parsedSender.sender.displayName,
                    kind: "person",
                    provenance: "transport_sender",
                  });
            const inputs: StreamEntryInput[] = [
              {
                kind: "user_msg",
                content: user,
                ...(senderEntityId === undefined ? {} : { sender_entity_id: senderEntityId }),
              },
              { kind: "agent_msg", content: assistant },
            ];
            return borg.stream.appendMany(inputs, { session });
          },
          { exclusive: true },
        );
        send(res, 200, {
          ok: true,
          session,
          entries: entries.map((entry) => ({
            id: entry.id,
            kind: entry.kind,
          })),
        });
        scheduleIngestion(pool, tenant, session);
        return;
      }

      // /memory/recall
      const query = asString(body.query);
      if (query === "") {
        send(res, 400, { error: "missing 'query'" });
        return;
      }
      const rawLimit =
        typeof body.limit === "number" && Number.isFinite(body.limit) ? body.limit : 10;
      const limit = Math.max(1, Math.min(maxRecallLimit, Math.floor(rawLimit)));
      const traceTurnId = traceRegistry === undefined ? undefined : nextRecallTraceTurnId(tenant);
      const hits = await pool.withTenant(tenant, (borg) =>
        borg.episodic.search(query, {
          limit,
          ...(traceTurnId === undefined ? {} : { traceTurnId }),
        }),
      );
      send(res, 200, {
        ok: true,
        episodes: hits.map((hit) => ({
          id: hit.episode.id,
          title: hit.episode.title,
          narrative: hit.episode.narrative,
          score: hit.score,
        })),
      });
    } catch (error) {
      // Tenant id is validated above, so anything thrown here is an internal
      // failure (open / storage / provider) that may carry sensitive detail —
      // log server-side, return a generic error.
      console.error(`memory-sidecar: ${rawPath} failed for tenant "${tenant}"`, error);
      send(res, 500, { error: "internal error" });
    }
  }

  return (req, res) => {
    void handle(req, res).catch((error: unknown) => {
      console.error("memory-sidecar: unhandled request error", error);
      send(res, 500, { error: "internal error" });
    });
  };
}
