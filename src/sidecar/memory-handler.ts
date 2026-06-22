// HTTP request handler for the borg memory sidecar: a thin, tenant-routed wrapper
// over BorgPool that exposes long-term memory to an external (e.g. Python) service.
//
//   POST /memory/remember    { tenant, content, author? }          -> append + extract episode(s)
//   POST /memory/append-turn { tenant, session, user, assistant }  -> append + async extract
//   POST /memory/recall      { tenant, query, limit? }             -> semantic episodic search
//   GET  /memory/episodes?tenant=<id>&limit=<n>&cursor=<c> -> list raw episodic bank
//   GET  /memory/episodes/{id}?tenant=<id>                  -> inspect one raw episode
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
import { parseEpisodeId, parseSessionId, type EpisodeId, type SessionId } from "../util/ids.js";

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
};

type RequestHandler = (req: IncomingMessage, res: ServerResponse) => void;

const DEFAULT_MAX_BODY_BYTES = 64 * 1024;
const DEFAULT_MAX_RECALL_LIMIT = 50;
const DEFAULT_EPISODE_LIST_LIMIT = 20;
const MAX_EPISODE_LIST_LIMIT = 100;

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
  const { pool, token } = options;
  const maxBodyBytes = options.maxBodyBytes ?? DEFAULT_MAX_BODY_BYTES;
  const maxRecallLimit = options.maxRecallLimit ?? DEFAULT_MAX_RECALL_LIMIT;

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

    if (method === "GET") {
      const isEpisodeListPath = rawPath === "/memory/episodes";
      const episodeId = isEpisodeListPath ? undefined : parseEpisodeIdFromPath(rawPath);
      if (!isEpisodeListPath && episodeId === undefined) {
        send(res, 404, { error: "not found" });
        return;
      }

      const tenant = asString(searchParams.get("tenant"));
      if (!validateTenantForResponse(res, tenant)) {
        return;
      }

      try {
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

        const session = sessionFromCaller(sessionRaw);
        const entries = await pool.withTenant(
          tenant,
          (borg) => {
            const inputs: StreamEntryInput[] = [
              { kind: "user_msg", content: user },
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
      const hits = await pool.withTenant(tenant, (borg) => borg.episodic.search(query, { limit }));
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
