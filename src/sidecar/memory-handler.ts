// HTTP request handler for the borg memory sidecar: a thin, tenant-routed wrapper
// over BorgPool that exposes long-term memory to an external (e.g. Python) service.
//
//   POST /memory/remember  { tenant, content, author? }  -> append + extract episode(s)
//   POST /memory/recall    { tenant, query, limit? }     -> semantic episodic search
//   GET  /healthz                                         -> liveness (no auth)
//
// Recall is tenant-wide by design: the pool routes to one being per tenant and,
// within a being, recall is global (borg's "recall is global to the being", with
// being == tenant). All authenticated routes require the shared x-borg-token.

import { timingSafeEqual } from "node:crypto";
import type { IncomingMessage, ServerResponse } from "node:http";

import type { Borg } from "../borg.js";

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

export function createMemoryHandler(options: MemoryHandlerOptions): RequestHandler {
  const { pool, token } = options;
  const maxBodyBytes = options.maxBodyBytes ?? DEFAULT_MAX_BODY_BYTES;
  const maxRecallLimit = options.maxRecallLimit ?? DEFAULT_MAX_RECALL_LIMIT;

  async function handle(req: IncomingMessage, res: ServerResponse): Promise<void> {
    const method = req.method ?? "GET";
    const url = (req.url ?? "").split("?")[0];

    if (method === "GET" && url === "/healthz") {
      send(res, 200, { ok: true });
      return;
    }

    if (!tokenMatches(req.headers["x-borg-token"], token)) {
      send(res, 401, { error: "unauthorized" });
      return;
    }

    if (method !== "POST" || (url !== "/memory/remember" && url !== "/memory/recall")) {
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
    if (tenant === "") {
      send(res, 400, { error: "missing 'tenant'" });
      return;
    }
    if (!TENANT_ID_RE.test(tenant)) {
      send(res, 400, { error: "invalid 'tenant'" });
      return;
    }

    try {
      if (url === "/memory/remember") {
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

      // /memory/recall
      const query = asString(body.query);
      if (query === "") {
        send(res, 400, { error: "missing 'query'" });
        return;
      }
      const rawLimit = typeof body.limit === "number" && Number.isFinite(body.limit) ? body.limit : 10;
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
      console.error(`memory-sidecar: ${url} failed for tenant "${tenant}"`, error);
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
