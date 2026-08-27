// HTTP request handler for the borg memory sidecar: a thin, tenant-routed wrapper
// over BorgPool that exposes long-term memory to an external (e.g. Python) service.
//
//   POST /memory/remember    { tenant, content, author? }          -> append + extract episode(s)
//   POST /memory/append-turn { tenant, session, user, assistant, sender? } -> append + async extract
//   POST /memory/recall      { tenant, query, limit? }             -> semantic episodic search
//   GET  /memory/commitments?tenant=<id>&audience=<entity_id>      -> active commitments
//        Alternative audience_external_id resolves team-agent sender identity.
//   POST /memory/commitments { tenant, ...commitment }             -> operator-set commitment
//   DELETE /memory/commitments?tenant=<id>&id=<commitment_id>      -> retire commitment
//   GET  /memory/episodes?tenant=<id>&limit=<n>&cursor=<c> -> list raw episodic bank
//   GET  /memory/self?tenant=<id>&limit=<n>      -> growth markers, periods, open questions
//   GET  /memory/semantic?tenant=<id>&limit=<n>  -> semantic nodes (no embeddings)
//   GET  /memory/review?tenant=<id>&openOnly=<0|1>&kind=<k>&limit=<n> -> review queue
//   GET  /memory/episodes/{id}?tenant=<id>                  -> inspect one raw episode
//   GET  /memory/trace?tenant=<id>&since=<ts>                -> inspect recall trace buffer
//   POST /memory/maintenance?tenant=<id|*>&mode=<light|heavy>&dryRun=<0|1>
//        tenant is optional; absent or "*" fans out across every tenant with a
//        bank on disk and answers {runs:[{tenant,run_id}],skipped:[...]}.
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

import { z } from "zod";

import type { Borg } from "../borg.js";
import { normalizeCommitmentClassification } from "../cognition/commitments/classification-normalizer.js";
import {
  commitmentCriticalDomainSchema,
  commitmentEnforcementClassSchema,
  commitmentKindSchema,
  commitmentTypeSchema,
  directiveFamilySchema,
  entityIdSchema,
  type CommitmentRecord,
} from "../memory/commitments/index.js";
import type { Episode } from "../memory/episodic/index.js";
import type { RetrievalDegradation } from "../retrieval/pipeline.js";
import type { StreamEntryInput } from "../stream/index.js";
import { EmbeddingError } from "../util/errors.js";
import {
  parseAuditId,
  parseCommitmentId,
  parseEpisodeId,
  parseMaintenanceRunId,
  parseSessionId,
  type EpisodeId,
  type MaintenanceRunId,
  type SessionId,
} from "../util/ids.js";
import type { MemoryMaintenanceCoordinator } from "./memory-maintenance.js";
import type { MemoryTraceRegistry } from "./memory-trace.js";

// Mirror of BorgPool's DEFAULT_TENANT_ID_PATTERN so the handler returns a clean
// 400 for a malformed tenant id at the boundary, rather than relying on (and
// risking a message leak from) the pool's ConfigError deeper in.
const TENANT_ID_RE = /^[a-z0-9][a-z0-9_-]{0,63}$/;

// The handler needs withTenant plus tenant discovery (maintenance fan-out);
// typing it structurally keeps the handler unit-testable with a stub.
export type MemoryPool = {
  withTenant<T>(
    tenantId: string,
    fn: (borg: Borg) => T | Promise<T>,
    opts?: { exclusive?: boolean },
  ): Promise<T>;
  listTenantIds(): Promise<string[]>;
};

export type MemoryHandlerOptions = {
  pool: MemoryPool;
  // Shared bearer presented as the x-borg-token header. Required; an empty token
  // rejects every authenticated request (fail closed).
  token: string;
  maxBodyBytes?: number;
  maxRecallLimit?: number;
  // Mechanism only; no usable threshold currently exists. Query-independent
  // heat + salience can give negative controls raw scores above known-positive
  // recalls, so the planned production mechanism is similarity-gated. Keep 0
  // (the default) until that exists.
  recallAbstainThreshold?: number;
  // Hard ceiling on a single /memory/recall (ms), so the client never gives up
  // first and turns a structured degradation into an opaque transport timeout.
  // This is a BACKSTOP for a wedged call, not the operative limit: it must sit
  // ABOVE borg's own worst-case graceful path (measured healthy prod recall
  // 2.0-2.6s + a double-stalled query embedding 2x1000 = ~4.6s) so the partial
  // results and degraded reason still get reported, and BELOW the caller's
  // recall timeout. Set it too low and it pre-empts the very degradation it
  // exists to deliver. 0 disables the ceiling.
  recallDeadlineMs?: number;
  traceRegistry?: MemoryTraceRegistry;
  maintenanceCoordinator?: Pick<
    MemoryMaintenanceCoordinator,
    "cancelReservation" | "getStatus" | "hasReservation" | "startReserved" | "tryReserve"
  >;
};

type RequestHandler = (req: IncomingMessage, res: ServerResponse) => void;

const DEFAULT_MAX_BODY_BYTES = 64 * 1024;
const DEFAULT_MAX_RECALL_LIMIT = 50;
const DEFAULT_RECALL_DEADLINE_MS = 5000;
const DEFAULT_EPISODE_LIST_LIMIT = 20;
const MAX_EPISODE_LIST_LIMIT = 100;
const MAX_COMMITMENT_RESPONSE_ITEMS = 100;

class RecallDeadlineExceeded extends Error {
  constructor(deadlineMs: number) {
    super(`recall exceeded ${deadlineMs}ms deadline`);
    this.name = "RecallDeadlineExceeded";
  }
}

// Mirrors the pipeline's expansion guard: the abandoned search keeps running
// and is left to settle on its own (its rejection swallowed) while the caller
// gets an answer within the deadline.
async function raceRecallDeadline<T>(search: Promise<T>, deadlineMs: number): Promise<T> {
  if (deadlineMs <= 0) {
    return search;
  }

  let timer: ReturnType<typeof setTimeout> | undefined;
  search.catch(() => undefined);

  try {
    return await Promise.race([
      search,
      new Promise<never>((_, reject) => {
        timer = setTimeout(() => reject(new RecallDeadlineExceeded(deadlineMs)), deadlineMs);
        timer.unref?.();
      }),
    ]);
  } finally {
    if (timer !== undefined) {
      clearTimeout(timer);
    }
  }
}

const APPEND_TURN_SENDER_EXTERNAL_ID_SOURCE = "team-agent.sender";

const operatorCommitmentBodySchema = z
  .object({
    tenant: z.string().trim().regex(TENANT_ID_RE),
    type: commitmentTypeSchema.exclude(["promise"]),
    kind: commitmentKindSchema.exclude(["assistant_commitment"]),
    enforcement_class: commitmentEnforcementClassSchema,
    critical_domain: commitmentCriticalDomainSchema.nullable(),
    directive: z.string().trim().min(1),
    family: directiveFamilySchema,
    priority: z.number().int(),
    audience_entity_id: entityIdSchema.nullable().optional().default(null),
  })
  .strict()
  .superRefine((value, ctx) => {
    const normalized = normalizeCommitmentClassification({
      kind: value.kind,
      type: value.type,
      enforcement_class: value.enforcement_class,
      critical_domain: value.critical_domain,
    });

    if (
      normalized.enforcement_class !== value.enforcement_class ||
      normalized.critical_domain !== value.critical_domain
    ) {
      ctx.addIssue({
        code: "custom",
        message: "invalid enforcement_class/critical_domain for commitment kind and type",
        path: ["enforcement_class"],
      });
    }
  });

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

function optionalSingleQueryValue(
  res: ServerResponse,
  searchParams: URLSearchParams,
  name: string,
): string | null | undefined {
  const values = searchParams.getAll(name);

  if (values.length === 0) {
    return undefined;
  }

  if (values.length !== 1 || values[0]?.trim() === "") {
    send(res, 400, { error: `invalid or duplicate '${name}'` });
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

// Nodes carry a float32 embedding; the read surface exists to show WHAT was
// written, so the vector is dropped rather than serialized.
function projectSemanticNodeForList(node: {
  id: string;
  kind: string;
  label: string;
  description: string;
  confidence: number;
  status: string;
  archived: boolean;
  source_episode_ids: readonly string[];
  created_at: number;
}): Record<string, unknown> {
  return {
    id: node.id,
    kind: node.kind,
    label: node.label,
    description: node.description,
    confidence: node.confidence,
    status: node.status,
    archived: node.archived,
    source_episode_ids: [...node.source_episode_ids],
    created_at: node.created_at,
  };
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

function projectCommitment(commitment: CommitmentRecord): {
  id: CommitmentRecord["id"];
  type: CommitmentRecord["type"];
  kind: CommitmentRecord["kind"];
  enforcement_class: CommitmentRecord["enforcement_class"];
  critical_domain: CommitmentRecord["critical_domain"];
  directive: string;
  family: string;
  priority: number;
  audience_entity_id: CommitmentRecord["restricted_audience"];
  created_at: number;
} {
  return {
    id: commitment.id,
    type: commitment.type,
    kind: commitment.kind,
    enforcement_class: commitment.enforcement_class,
    critical_domain: commitment.critical_domain,
    directive: commitment.directive,
    family: commitment.directive_family,
    priority: commitment.priority,
    audience_entity_id: commitment.restricted_audience ?? commitment.made_to_entity,
    created_at: commitment.created_at,
  };
}

function compareCommitmentsForResponse(left: CommitmentRecord, right: CommitmentRecord): number {
  const enforcementOrder =
    (left.enforcement_class === "critical" ? 0 : 1) -
    (right.enforcement_class === "critical" ? 0 : 1);

  return (
    enforcementOrder ||
    right.priority - left.priority ||
    left.created_at - right.created_at ||
    left.id.localeCompare(right.id)
  );
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
  const recallAbstainThreshold = options.recallAbstainThreshold ?? 0;
  const recallDeadlineMs = options.recallDeadlineMs ?? DEFAULT_RECALL_DEADLINE_MS;
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

    if (method === "DELETE" && rawPath === "/memory/commitments") {
      const tenant = requiredSingleQueryValue(res, searchParams, "tenant");
      if (tenant === null) {
        return;
      }
      const commitmentIdRaw = requiredSingleQueryValue(res, searchParams, "id");
      if (commitmentIdRaw === null) {
        return;
      }
      if (!validateTenantForResponse(res, tenant)) {
        return;
      }

      let commitmentId;
      try {
        commitmentId = parseCommitmentId(commitmentIdRaw);
      } catch {
        send(res, 400, { error: "invalid 'id'" });
        return;
      }

      try {
        const result = await pool.withTenant(
          tenant,
          (borg) => {
            const commitment = borg.commitments.get(commitmentId);

            if (commitment === null) {
              return { status: "missing" as const };
            }

            const active = borg.commitments
              .list({ activeOnly: true })
              .some((candidate) => candidate.id === commitmentId);

            if (!active) {
              return { status: "inactive" as const };
            }

            return {
              status: "retired" as const,
              commitment: borg.commitments.revoke(commitmentId, "retired_by_operator", {
                kind: "manual",
              }),
            };
          },
          { exclusive: true },
        );

        if (result.status === "missing") {
          send(res, 404, { error: "commitment not found" });
          return;
        }
        if (result.status === "inactive") {
          send(res, 409, { error: "commitment is not active" });
          return;
        }
        if (result.commitment === null) {
          send(res, 409, { error: "commitment could not be retired" });
          return;
        }

        send(res, 200, {
          ok: true,
          commitment: projectCommitment(result.commitment),
        });
      } catch (error) {
        console.error(`memory-sidecar: ${rawPath} failed for tenant "${tenant}"`, error);
        send(res, 500, { error: "internal error" });
      }
      return;
    }

    if (method === "POST" && rawPath === "/memory/maintenance") {
      // tenant is OPTIONAL. Absent (or "*") means "every tenant that has a bank
      // on disk", so onboarding a tenant cannot silently skip maintenance the
      // way a hard-coded caller does. "*" mirrors the same convention on
      // team-agent's role-run endpoint.
      const tenantRaw = optionalSingleQueryValue(res, searchParams, "tenant");
      if (tenantRaw === null) {
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
      const fanOut = tenantRaw === undefined || tenantRaw === "*";
      if (!fanOut && !validateTenantForResponse(res, tenantRaw)) {
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
      const coordinator = maintenanceCoordinator;

      let tenants: string[];
      if (fanOut) {
        try {
          tenants = await pool.listTenantIds();
        } catch {
          send(res, 503, { error: "tenant discovery unavailable" });
          return;
        }
        if (tenants.length === 0) {
          // No bank on the volume yet. Reported rather than treated as success
          // so a misconfigured root does not read as a clean no-op run.
          send(res, 503, { error: "no tenants discovered" });
          return;
        }
      } else {
        tenants = [tenantRaw];
      }

      type Reserved = { readonly tenant: string; readonly runId: MaintenanceRunId };
      type Skipped = {
        readonly tenant: string;
        readonly reason: string;
        readonly runId?: MaintenanceRunId;
      };
      const reserved: Reserved[] = [];
      const skipped: Skipped[] = [];

      for (const tenant of tenants) {
        const started = coordinator.tryReserve({ tenant, mode, dryRun: dryRun === "1" });
        if (started.status !== "accepted") {
          skipped.push(
            started.status === "conflict"
              ? { tenant, reason: "already running", runId: started.runId }
              : { tenant, reason: started.status === "disabled" ? "disabled" : "shutting down" },
          );
          continue;
        }
        try {
          // Admission is already reserved, so racing POSTs see 409 while the
          // tenant is opened and its pool initializer establishes readiness.
          await pool.withTenant(tenant, () => undefined);
        } catch {
          coordinator.cancelReservation(tenant, started.runId);
          skipped.push({ tenant, reason: "tenant unavailable" });
          continue;
        }
        if (!coordinator.hasReservation(tenant, started.runId)) {
          skipped.push({ tenant, reason: "shutting down" });
          continue;
        }
        reserved.push({ tenant, runId: started.runId });
      }

      // A single named tenant keeps its original response contract exactly --
      // 202 {run_id} on success, and the specific 409/503 the caller had before
      // -- so existing callers see no change.
      if (!fanOut) {
        const only = reserved[0];
        if (only === undefined) {
          const reason = skipped[0]?.reason;
          if (reason === "already running") {
            send(res, 409, { error: "maintenance already running", run_id: skipped[0]?.runId });
          } else if (reason === "disabled") {
            send(res, 503, { error: "maintenance disabled" });
          } else if (reason === "tenant unavailable") {
            send(res, 503, { error: "maintenance tenant unavailable" });
          } else {
            send(res, 503, { error: "maintenance shutting down" });
          }
          return;
        }
        try {
          send(res, 202, { run_id: only.runId });
        } finally {
          // Scheduling happens only after the acceptance response is handed
          // to the server, and remains detached from the client connection.
          coordinator.startReserved(only.tenant, only.runId);
        }
        return;
      }

      // Fan-out. Runs proceed CONCURRENTLY across tenants (each holds only its
      // own exclusive per-tenant reservation); nothing is serialized here.
      if (reserved.length === 0) {
        // Nothing started: answer non-2xx so a `curl --fail` cron surfaces it as
        // a failed job instead of a silent no-op.
        const allConflicts = skipped.every((entry) => entry.reason === "already running");
        send(res, allConflicts ? 409 : 503, {
          error: allConflicts ? "maintenance already running" : "no maintenance run started",
          skipped,
        });
        return;
      }
      try {
        send(res, 202, {
          runs: reserved.map((entry) => ({ tenant: entry.tenant, run_id: entry.runId })),
          skipped,
        });
      } finally {
        for (const entry of reserved) {
          coordinator.startReserved(entry.tenant, entry.runId);
        }
      }
      return;
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
      const isCommitmentListPath = rawPath === "/memory/commitments";
      const isMaintenanceStatusPath = rawPath === "/memory/maintenance/status";
      const isMaintenanceAuditPath = rawPath === "/memory/maintenance/audit";
      // Read surface for what the offline processes WRITE. Without these, a
      // maintenance run can only be inspected as counts plus the audit log's
      // record ids -- self-narrator growth markers, associator open questions and
      // reflector insight nodes had no read path at all, so "0 errors" could not be
      // distinguished from "wrote nine mis-voiced records".
      const isSelfPath = rawPath === "/memory/self";
      const isSemanticPath = rawPath === "/memory/semantic";
      // The review queue is where the reflector's insights and the overseer's flags
      // actually live: both are PROPOSALS, so nothing appears in the semantic graph
      // until a resolution accepts them. Without this the only visible trace of a
      // heavy run's flags was a change count.
      const isReviewPath = rawPath === "/memory/review";
      const nonEpisodePath =
        isTracePath ||
        isEpisodeListPath ||
        isCommitmentListPath ||
        isMaintenanceStatusPath ||
        isMaintenanceAuditPath ||
        isSelfPath ||
        isSemanticPath ||
        isReviewPath;
      const episodeId = nonEpisodePath ? undefined : parseEpisodeIdFromPath(rawPath);
      if (!nonEpisodePath && episodeId === undefined) {
        send(res, 404, { error: "not found" });
        return;
      }

      const strictQuery =
        isCommitmentListPath ||
        isMaintenanceStatusPath ||
        isMaintenanceAuditPath ||
        isSelfPath ||
        isSemanticPath ||
        isReviewPath;
      const tenant = strictQuery
        ? requiredSingleQueryValue(res, searchParams, "tenant")
        : asString(searchParams.get("tenant"));
      if (tenant === null) {
        return;
      }
      if (!validateTenantForResponse(res, tenant)) {
        return;
      }
      const audienceRaw = isCommitmentListPath
        ? optionalSingleQueryValue(res, searchParams, "audience")
        : undefined;
      if (audienceRaw === null) {
        return;
      }
      const audienceExternalIdRaw = isCommitmentListPath
        ? optionalSingleQueryValue(res, searchParams, "audience_external_id")
        : undefined;
      if (audienceExternalIdRaw === null) {
        return;
      }
      if (audienceRaw !== undefined && audienceExternalIdRaw !== undefined) {
        send(res, 400, {
          error: "'audience' and 'audience_external_id' are mutually exclusive",
        });
        return;
      }
      const parsedAudience =
        audienceRaw === undefined
          ? { success: true as const, data: null }
          : entityIdSchema.safeParse(audienceRaw);
      if (!parsedAudience.success) {
        send(res, 400, { error: "invalid 'audience'" });
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

        if (isCommitmentListPath) {
          const result = await pool.withTenant(tenant, (borg) => {
            const audienceEntityId =
              audienceExternalIdRaw === undefined
                ? parsedAudience.data
                : borg.entities.findByExternalId(
                    APPEND_TURN_SENDER_EXTERNAL_ID_SOURCE,
                    audienceExternalIdRaw,
                  );

            return {
              audienceEntityId,
              commitments: borg.commitments.list({
                activeOnly: true,
                audienceEntityId,
              }),
            };
          });
          const ordered = [...result.commitments].sort(compareCommitmentsForResponse);
          const bounded = ordered.slice(0, MAX_COMMITMENT_RESPONSE_ITEMS);

          send(res, 200, {
            ok: true,
            tenant,
            audience_entity_id: result.audienceEntityId,
            ...(audienceExternalIdRaw === undefined
              ? {}
              : {
                  audience_external_id: audienceExternalIdRaw,
                  audience_resolved: result.audienceEntityId !== null,
                }),
            commitments: bounded.map((commitment) => projectCommitment(commitment)),
            truncated: ordered.length > bounded.length,
          });
          return;
        }

        if (isSelfPath) {
          const limit = episodeListLimitFromQuery(searchParams);
          const self = await pool.withTenant(tenant, (borg) => ({
            growth_markers: borg.self.growthMarkers.list({ limit }),
            periods: borg.self.autobiographical.listPeriods(),
            open_questions: borg.self.openQuestions.list({ limit }),
          }));

          send(res, 200, { ok: true, tenant, ...self });
          return;
        }

        if (isReviewPath) {
          const openOnly = searchParams.get("openOnly") !== "0";
          const kindRaw = searchParams.get("kind");
          const items = await pool.withTenant(tenant, (borg) =>
            borg.review.list({
              openOnly,
              ...(kindRaw === null || kindRaw.trim() === "" ? {} : { kind: kindRaw as never }),
            }),
          );
          const limit = episodeListLimitFromQuery(searchParams);

          send(res, 200, {
            ok: true,
            tenant,
            open_only: openOnly,
            total: items.length,
            items: items.slice(0, limit),
            truncated: items.length > limit,
          });
          return;
        }

        if (isSemanticPath) {
          const limit = episodeListLimitFromQuery(searchParams);
          const nodes = await pool.withTenant(tenant, (borg) =>
            borg.semantic.nodes.list({ limit }),
          );

          send(res, 200, {
            ok: true,
            tenant,
            nodes: nodes.map((node) => projectSemanticNodeForList(node)),
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
        rawPath !== "/memory/append-turn" &&
        rawPath !== "/memory/commitments")
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
      if (rawPath === "/memory/commitments") {
        const parsed = operatorCommitmentBodySchema.safeParse(body);

        if (!parsed.success) {
          send(res, 400, { error: "invalid commitment body" });
          return;
        }

        const result = await pool.withTenant(
          tenant,
          (borg) => {
            if (
              parsed.data.audience_entity_id !== null &&
              borg.entities.get(parsed.data.audience_entity_id) === null
            ) {
              return { status: "unknown_audience" as const };
            }

            return {
              status: "created" as const,
              commitment: borg.identity.addCommitment({
                type: parsed.data.type,
                kind: parsed.data.kind,
                enforcementClass: parsed.data.enforcement_class,
                criticalDomain: parsed.data.critical_domain,
                directiveFamily: parsed.data.family,
                directive: parsed.data.directive,
                priority: parsed.data.priority,
                madeToEntity: null,
                restrictedAudience: parsed.data.audience_entity_id,
                aboutEntity: null,
                committedByEntityId: null,
                provenance: { kind: "manual" },
              }),
            };
          },
          { exclusive: true },
        );

        if (result.status === "unknown_audience") {
          send(res, 400, { error: "unknown 'audience_entity_id'" });
          return;
        }

        send(res, 201, {
          ok: true,
          commitment: projectCommitment(result.commitment),
        });
        return;
      }

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
            return borg.episodic.extract({
              sinceTs: entry.timestamp,
              bypassSalienceGate: true,
            });
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
      const degradations: RetrievalDegradation[] = [];
      let hits;
      try {
        hits = await raceRecallDeadline(
          pool.withTenant(tenant, (borg) =>
            borg.episodic.search(query, {
              limit,
              onDegraded: (degradation) => degradations.push(degradation),
              ...(traceTurnId === undefined ? {} : { traceTurnId }),
            }),
          ),
          recallDeadlineMs,
        );
      } catch (error) {
        if (!(error instanceof RecallDeadlineExceeded)) {
          throw error;
        }
        console.error(`memory-sidecar: /memory/recall hit its deadline for tenant "${tenant}"`);
        send(res, 200, {
          ok: true,
          episodes: [],
          degraded: true,
          degraded_reason: `deadline: ${error.message}`,
        });
        return;
      }
      const topRawScore = hits.length === 0 ? null : Math.max(...hits.map((hit) => hit.rawScore));
      // A partial recall must say so: the client cannot otherwise tell "nothing
      // is stored" from "the search broke", and only the latter justifies
      // telling the user their memory is unavailable.
      const degraded =
        degradations.length === 0
          ? {}
          : {
              degraded: true,
              degraded_reason: degradations
                .map((entry) => `${entry.subsystem}: ${entry.reason}`)
                .join("; "),
            };

      if (
        recallAbstainThreshold > 0 &&
        (topRawScore === null || topRawScore < recallAbstainThreshold)
      ) {
        send(res, 200, {
          ok: true,
          episodes: [],
          abstained: true,
          abstain_reason: "low_relevance",
          top_raw_score: topRawScore,
          ...degraded,
        });
        return;
      }

      send(res, 200, {
        ok: true,
        top_raw_score: topRawScore,
        ...degraded,
        episodes: hits.map((hit) => ({
          id: hit.episode.id,
          title: hit.episode.title,
          narrative: hit.episode.narrative,
          score: hit.score,
          raw_score: hit.rawScore,
        })),
      });
    } catch (error) {
      // Tenant id is validated above, so anything thrown here is an internal
      // failure (open / storage / provider) that may carry sensitive detail —
      // log server-side, return a generic error.
      console.error(`memory-sidecar: ${rawPath} failed for tenant "${tenant}"`, error);
      // An embedding stall on the recall path is a known-transient gateway
      // fault, not a broken request. Answering it as an explicit degradation
      // lets the caller distinguish it from an empty memory and keeps it off
      // the 5xx path, where a retry-happy client would only add load.
      if (rawPath === "/memory/recall" && error instanceof EmbeddingError) {
        send(res, 200, {
          ok: true,
          episodes: [],
          degraded: true,
          degraded_reason: `embeddings: ${error.message}`,
        });
        return;
      }
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
