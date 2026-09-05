// HTTP request handler for the borg memory sidecar: a thin, tenant-routed wrapper
// over BorgPool that exposes long-term memory to an external (e.g. Python) service.
//
//   POST /memory/remember    { tenant, content, author? }          -> append + extract episode(s)
//   POST /memory/enqueue     { tenant, session, conversation, sender, text, ... } -> durable inbox
//   POST /memory/await-response { tenant, sidecar_session_id, entry_id, timeout_ms? } -> long poll
//   POST /memory/inbox-progress { tenant, sidecar_session_id, entry_ids, phase } -> interim status
//   POST /memory/append-turn { tenant, session, user?, assistant?, observed_at?, sender?, conversation? }
//        sender.operator? and conversation.external_id? enrich sessions/audience/activity;
//        absent assistant records an observation; absent user records a reply-only turn;
//        incomplete identity keeps legacy append behavior
//   POST /memory/context { tenant, session, sender, conversation, query?, focus?, context_turns?,
//                          limit?, sections?,
//                          participants?, entity_terms?, time_range?, exclude?, venue_since?,
//                          venue_limit? }
//                                                            -> audience-scoped turn context
//   POST /memory/recall      { tenant, query, limit?, time_range?, exclude? }
//                                                            -> semantic episodic search
//   GET  /memory/commitments?tenant=<id>&audience=<entity_id>      -> active commitments
//        Alternative audience_external_id resolves team-agent sender identity.
//   POST /memory/commitments { tenant, ...commitment }             -> operator-set commitment
//   DELETE /memory/commitments?tenant=<id>&id=<commitment_id>      -> retire commitment
//   POST /memory/directives { tenant, kind, text, content_scope, ... } -> queue operator directive
//   GET  /memory/directives?tenant=<id>                            -> list active directives
//   DELETE /memory/directives/{id}?tenant=<id> { reason }          -> revoke directive
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
// Cognition recall remains global within each tenant being. These HTTP routes are
// disclosure/export surfaces: /memory/context applies audience visibility before
// returning episodes or activity. All authenticated routes require x-borg-token.

import { timingSafeEqual } from "node:crypto";
import type { IncomingMessage, ServerResponse } from "node:http";

import { z } from "zod";

import type { Borg } from "../borg.js";
import { normalizeCommitmentClassification } from "../cognition/commitments/classification-normalizer.js";
import type { ActivityVisibleSessionEvent } from "../memory/activity/index.js";
import {
  commitmentCriticalDomainSchema,
  commitmentEnforcementClassSchema,
  commitmentKindSchema,
  commitmentTypeSchema,
  directiveFamilySchema,
  entityIdSchema,
  type CommitmentRecord,
  type EntityRecord,
} from "../memory/commitments/index.js";
import {
  creatorDirectiveContentScopeSchema,
  creatorDirectiveKindSchema,
  creatorDirectiveMentionPolicySchema,
  creatorDirectiveQueueInputSchema,
  creatorDirectiveTopicTagSchema,
  type CreatorDirective,
  type CreatorDirectiveApplicable,
  type CreatorDirectiveQueueInput,
} from "../memory/creator-directives/index.js";
import type {
  AutobiographicalRecallResult,
  AutobiographicalRecallSourceKind,
} from "../cognition/autobiographical-recall.js";
import type { TemporalCue } from "../contracts/cognitive-contracts.js";
import {
  isMemoryDisclosureLabelVisibleToAnyAudience,
  memoryDisclosureLabelFromEpisodeAccess,
} from "../memory/common/index.js";
import {
  isEpisodeAccessVisibleToAnyAudience,
  parseEpisodeParticipantEntityIdTerm,
  type Episode,
} from "../memory/episodic/index.js";
import {
  clipRecalledEvidenceText,
  MAX_RECALLED_SOURCE_MESSAGES_PER_EPISODE,
} from "../retrieval/evidence-bounds.js";
import type { EpisodeRecencyPrior, RetrievedEpisode } from "../retrieval/index.js";
import type { RecallPlanOutcome, RetrievalDegradation } from "../retrieval/pipeline.js";
import {
  MAX_RECALL_EXPANSION_SEMANTIC_VARIANTS,
  MAX_RECALL_QUERY_ACTIVITY_ROWS,
  MAX_RECALL_QUERY_CONTEXT_TURN_CHARS,
  MAX_RECALL_QUERY_ENTITY_TERMS,
  MAX_RECALL_QUERY_HANDLE_CHARS,
  MIN_RECALL_EXPANSION_SEMANTIC_VARIANTS,
} from "../retrieval/recall-expansion.js";
import {
  isNarrativeStreamEntry,
  type StreamEntry,
  type StreamEntryInput,
} from "../stream/index.js";
import { sessionIdSchema, streamEntryIdSchema } from "../util/id-schemas.js";
import { dedupePreservingOrder } from "../util/collections.js";
import { ConfigError, EmbeddingError } from "../util/errors.js";
import {
  createStreamEntryId,
  parseAuditId,
  parseCommitmentId,
  parseCreatorDirectiveId,
  parseEpisodeId,
  parseMaintenanceRunId,
  type EpisodeId,
  type EntityId,
  type MaintenanceRunId,
  type SessionId,
  type StreamEntryId,
} from "../util/ids.js";
import { formatRelativeAge } from "../util/relative-time.js";
import type { MemoryMaintenanceCoordinator } from "./memory-maintenance.js";
import type { MemoryTraceRegistry } from "./memory-trace.js";
import {
  awaitResponseForTerminal,
  type ResponseWaiterRegistry,
} from "./response-waiter-registry.js";
import {
  resolveTeamAgentIdentity,
  sessionFromCaller,
  sidecarConversationSchema,
  TEAM_AGENT_CONVERSATION_EXTERNAL_ID_SOURCE,
  TEAM_AGENT_SENDER_EXTERNAL_ID_SOURCE,
  type SidecarConversation,
  type TeamAgentIdentity,
} from "./team-agent-identity.js";
import { MAX_INBOX_REPLY_ACTIVITY_RECONCILE_LIMIT } from "../cognition/ingestion/index.js";

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
  recentActivityWindowMs?: number;
  recentActivityLimit?: number;
  activityExcerptHydrationBudgetMs?: number;
  recencyPrior?: EpisodeRecencyPrior;
  recallSemanticVariantCount?: number;
  traceRegistry?: MemoryTraceRegistry;
  maintenanceCoordinator?: Pick<
    MemoryMaintenanceCoordinator,
    "cancelReservation" | "getStatus" | "hasReservation" | "startReserved" | "tryReserve"
  >;
  inboxWaiters?: ResponseWaiterRegistry;
};

type RequestHandler = (req: IncomingMessage, res: ServerResponse) => void;

const DEFAULT_MAX_BODY_BYTES = 64 * 1024;
const DEFAULT_MAX_RECALL_LIMIT = 50;
const DEFAULT_RECALL_DEADLINE_MS = 5000;
export const DEFAULT_RECENT_ACTIVITY_WINDOW_MS = 24 * 60 * 60_000;
export const DEFAULT_RECENT_ACTIVITY_LIMIT = 12;
// The planner also sees the owner's closed-day summaries (lived-experience spine) for the last
// week, so references to earlier days resolve even when the 24 h activity window has moved on.
export const PLANNER_LIVED_EXPERIENCE_WINDOW_MS = 7 * 24 * 60 * 60_000;
export const PLANNER_LIVED_EXPERIENCE_LIMIT = 7;
export const DEFAULT_ACTIVITY_EXCERPT_HYDRATION_BUDGET_MS = 50;
export const MEMORY_RECALL_SEMANTIC_VARIANT_COUNT_ENV = "BORG_MEMORY_RECALL_SEMANTIC_VARIANT_COUNT";
const RECENT_ACTIVITY_EXCERPT_HYDRATION_FAILURE_REASON = "recent_activity_excerpt_hydration_failed";
const DEFAULT_VENUE_RECENT_LIMIT = 12;
const MAX_VENUE_RECENT_LIMIT = 50;
// Rows of the owner's own record returned for a cued period. The service ranks up to 48; a prompt
// block only needs the top of that.
const MAX_AUTOBIOGRAPHICAL_ROWS = 12;
// The service scans other sessions' streams for the owner's reflections and reaches; the sidecar
// bounds that scan below Sol's defaults because it runs inside an interactive request.
const AUTOBIOGRAPHICAL_SESSION_CAP = 8;
const AUTOBIOGRAPHICAL_TOTAL_CAP = 24;
// The second pass only gets what the episodes pass left of the request deadline, minus headroom for
// the rest of the handler, and never more than a short cap: the interactive client's own timeout
// sits at the recall deadline, so a second pass that ran the deadline out would lose the episodes
// too. Below the floor it is skipped rather than started. Measured 2026-09-05 in production: the
// episodes pass alone takes 3.5-5.3 s of a 5 s deadline, so this pass often has little or no room.
const MIN_AUTOBIOGRAPHICAL_BUDGET_MS = 500;
const MAX_AUTOBIOGRAPHICAL_BUDGET_MS = 1500;
const AUTOBIOGRAPHICAL_HEADROOM_MS = 700;
// Kinds whose disclosure label comes from exactly one source record, so "one visible audience is
// among the entities it is private to" is the whole story. Open questions, goals, actions and
// autobiographical periods carry labels combined across several sources (one visible source would
// admit text derived from another private one), and episodes are already served by the `episodes`
// section with the request's exclusions applied; both are omitted here.
const SIDECAR_AUTOBIOGRAPHICAL_KINDS: ReadonlySet<AutobiographicalRecallSourceKind> = new Set([
  "activity",
  "observed_social_event",
  "stream_reflection",
  "silence_decision",
  "outbound_attempt",
  "observed_presence",
]);
const MAX_CONTEXT_PARTICIPANTS = 32;
const MAX_CONTEXT_ENTITY_TERMS = MAX_RECALL_QUERY_ENTITY_TERMS;
const MAX_CONTEXT_ENTITY_TERM_CHARS = MAX_RECALL_QUERY_HANDLE_CHARS;
const MAX_CONTEXT_TURNS = 3;
const EPISODE_OVERFETCH_MULTIPLIER = 3;
const OBSERVATION_MAX_PAST_AGE_MS = 5 * 60_000;
const OBSERVATION_MAX_FUTURE_SKEW_MS = 60_000;
const DEFAULT_EPISODE_LIST_LIMIT = 20;
const MAX_EPISODE_LIST_LIMIT = 100;
const MAX_COMMITMENT_RESPONSE_ITEMS = 100;

export function memoryRecallSemanticVariantCountFromEnv(
  env: NodeJS.ProcessEnv = process.env,
): number {
  const raw = env[MEMORY_RECALL_SEMANTIC_VARIANT_COUNT_ENV]?.trim();
  if (raw === undefined || raw === "") {
    return 1;
  }

  const count = Number(raw);
  if (
    !Number.isInteger(count) ||
    count < MIN_RECALL_EXPANSION_SEMANTIC_VARIANTS ||
    count > MAX_RECALL_EXPANSION_SEMANTIC_VARIANTS
  ) {
    throw new ConfigError(
      `${MEMORY_RECALL_SEMANTIC_VARIANT_COUNT_ENV} must be an integer between ${MIN_RECALL_EXPANSION_SEMANTIC_VARIANTS} and ${MAX_RECALL_EXPANSION_SEMANTIC_VARIANTS}`,
    );
  }

  return count;
}

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

const SIDECAR_ADMIN_EXTERNAL_ID_SOURCE = "memory-sidecar.admin";
const SIDECAR_ADMIN_EXTERNAL_ID = "operator-api";
const SIDECAR_ADMIN_SESSION_EXTERNAL_ID = "memory-sidecar::admin-api";

const contextConversationSchema = sidecarConversationSchema.strict();

const contextSenderSchema = z
  .object({
    external_id: z.string().trim().min(1),
    display_name: z.string().trim().min(1),
    operator: z.boolean().optional().default(false),
  })
  .strict();

const contextTurnSchema = z
  .object({
    role: z.enum(["user", "assistant"]),
    text: z.string().trim().min(1).max(MAX_RECALL_QUERY_CONTEXT_TURN_CHARS),
  })
  .strict();

const epochMillisecondsSchema = z.number().int().nonnegative();
const episodeTimeRangeSchema = z
  .object({
    start: epochMillisecondsSchema,
    end: epochMillisecondsSchema,
  })
  .strict()
  .refine((range) => range.start <= range.end, {
    message: "time range start must be less than or equal to end",
    path: ["end"],
  });
const episodeExclusionsSchema = z
  .object({
    title_prefixes: z.array(z.string().min(1)).max(8).optional().default([]),
    narrative_markers: z.array(z.string().min(1)).max(8).optional().default([]),
  })
  .strict();

const memoryContextSectionSchema = z.enum([
  "audience",
  "episodes",
  "recent_activity",
  "commitments",
  "directives",
  "venue_recent",
  "autobiographical",
]);
const DEFAULT_MEMORY_CONTEXT_SECTIONS = [
  "audience",
  "episodes",
  "recent_activity",
  "commitments",
  "directives",
] as const satisfies readonly z.infer<typeof memoryContextSectionSchema>[];

const memoryContextBodySchema = z
  .object({
    tenant: z.string().trim().regex(TENANT_ID_RE),
    session: z.string().trim().min(1),
    sender: contextSenderSchema,
    conversation: contextConversationSchema,
    participants: z.array(contextSenderSchema).max(MAX_CONTEXT_PARTICIPANTS).optional(),
    entity_terms: z
      .array(z.string().trim().min(1).max(MAX_CONTEXT_ENTITY_TERM_CHARS))
      .max(MAX_CONTEXT_ENTITY_TERMS)
      .optional(),
    query: z.string().trim().min(1).optional(),
    focus: z.string().trim().min(1).optional(),
    context_turns: z.array(contextTurnSchema).max(MAX_CONTEXT_TURNS).optional(),
    limit: z.number().finite().optional(),
    sections: z.array(memoryContextSectionSchema).min(1).optional(),
    time_range: episodeTimeRangeSchema.optional(),
    exclude: episodeExclusionsSchema.optional(),
    venue_since: epochMillisecondsSchema.optional(),
    venue_limit: z.number().int().min(1).max(MAX_VENUE_RECENT_LIMIT).optional(),
  })
  .strict()
  .superRefine((value, ctx) => {
    const episodesRequested = value.sections === undefined || value.sections.includes("episodes");
    const venueRecentRequested = value.sections?.includes("venue_recent") === true;
    const autobiographicalRequested = value.sections?.includes("autobiographical") === true;

    if (autobiographicalRequested && !episodesRequested) {
      ctx.addIssue({
        code: "custom",
        path: ["sections"],
        message: "autobiographical requires episodes (its period comes from the recall plan)",
      });
    }

    if (value.conversation.type !== "personal" && value.conversation.external_id === undefined) {
      ctx.addIssue({
        code: "custom",
        path: ["conversation", "external_id"],
        message: "groupChat and channel context requires conversation.external_id",
      });
    }

    if (value.context_turns !== undefined && value.focus === undefined) {
      ctx.addIssue({
        code: "custom",
        path: ["context_turns"],
        message: "context_turns requires focus",
      });
    }

    if (episodesRequested && !value.focus && !value.query) {
      ctx.addIssue({
        code: "custom",
        path: ["focus"],
        message: "focus or query is required when episodes are requested",
      });
    }

    if (venueRecentRequested && value.venue_since === undefined) {
      ctx.addIssue({
        code: "custom",
        path: ["venue_since"],
        message: "venue_since is required when venue_recent is requested",
      });
    }
  });

const memoryRecallBodySchema = z
  .object({
    tenant: z.string().trim().regex(TENANT_ID_RE),
    query: z.string().trim().min(1),
    limit: z.number().finite().optional(),
    time_range: episodeTimeRangeSchema.optional(),
    exclude: episodeExclusionsSchema.optional(),
  })
  .strict();

const memoryEnqueueBodySchema = z
  .object({
    tenant: z.string().trim().regex(TENANT_ID_RE),
    session: z.string().trim().min(1),
    conversation: sidecarConversationSchema
      .extend({ external_id: z.string().trim().min(1) })
      .strict(),
    sender: z
      .object({
        external_id: z.string().trim().min(1),
        display_name: z.string().trim().min(1),
        bot: z.boolean(),
        operator: z.boolean(),
      })
      .strict(),
    text: z.string().min(1),
    external_message_id: z.string().trim().min(1),
    observed_at: z.iso.datetime({ offset: true }),
    flags: z.object({ mentioned: z.boolean(), quotes_bot: z.boolean() }).strict(),
  })
  .strict();

const memoryAwaitResponseBodySchema = z
  .object({
    tenant: z.string().trim().regex(TENANT_ID_RE),
    sidecar_session_id: sessionIdSchema,
    entry_id: streamEntryIdSchema,
    timeout_ms: z.number().int().min(0).max(120_000).optional().default(90_000),
    seen_generating: z.boolean().optional().default(false),
  })
  .strict();

const memoryInboxProgressBodySchema = z
  .object({
    tenant: z.string().trim().regex(TENANT_ID_RE),
    sidecar_session_id: sessionIdSchema,
    entry_ids: z.array(streamEntryIdSchema).min(1),
    phase: z.literal("generating"),
  })
  .strict();

const inboxReplyActivityReconcileBodySchema = z
  .object({
    tenant: z.string().trim().regex(TENANT_ID_RE),
    dry_run: z.boolean().default(true),
    since: z.iso.datetime({ offset: true }).optional(),
    until: z.iso.datetime({ offset: true }).optional(),
    limit: z.number().int().positive().max(MAX_INBOX_REPLY_ACTIVITY_RECONCILE_LIMIT).optional(),
  })
  .strict()
  .refine(
    (body) =>
      body.since === undefined ||
      body.until === undefined ||
      Date.parse(body.since) <= Date.parse(body.until),
    { message: "since must not be later than until" },
  );

const directiveAdminBodySchema = z
  .object({
    tenant: z.string().trim().regex(TENANT_ID_RE),
    kind: creatorDirectiveKindSchema,
    text: z.string().trim().min(1),
    content_scope: creatorDirectiveContentScopeSchema,
    allowed_external_ids: z.array(z.string().trim().min(1)).optional().default([]),
    excluded_external_ids: z.array(z.string().trim().min(1)).optional().default([]),
    allowed_group_external_ids: z.array(z.string().trim().min(1)).optional().default([]),
    excluded_group_external_ids: z.array(z.string().trim().min(1)).optional().default([]),
    subject_external_id: z.string().trim().min(1).optional(),
    mention_policy: creatorDirectiveMentionPolicySchema.optional().default("answer_if_asked"),
    priority: z.number().int().optional().default(0),
    topic_tags: z.array(creatorDirectiveTopicTagSchema).max(32).optional().default([]),
  })
  .strict()
  .superRefine((value, ctx) => {
    if (value.kind === "subject_fact" && value.subject_external_id === undefined) {
      ctx.addIssue({
        code: "custom",
        path: ["subject_external_id"],
        message: "subject_fact requires subject_external_id",
      });
    }

    if (value.kind !== "subject_fact" && value.subject_external_id !== undefined) {
      ctx.addIssue({
        code: "custom",
        path: ["subject_external_id"],
        message: "subject_external_id is only valid for subject_fact",
      });
    }
  });

const directiveRevokeBodySchema = z
  .object({
    reason: z.string().trim().min(1),
  })
  .strict();

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

async function readJsonObjectBody(
  req: IncomingMessage,
  res: ServerResponse,
  maxBytes: number,
): Promise<Record<string, unknown> | null> {
  try {
    const raw = await readBody(req, maxBytes);
    const parsed: unknown = raw.trim() === "" ? {} : JSON.parse(raw);

    if (typeof parsed !== "object" || parsed === null || Array.isArray(parsed)) {
      send(res, 400, { error: "request body must be a JSON object" });
      return null;
    }

    return parsed as Record<string, unknown>;
  } catch (error) {
    if (error instanceof PayloadTooLargeError) {
      send(res, 413, { error: "request body too large" });
      return null;
    }

    send(res, 400, { error: "invalid JSON body" });
    return null;
  }
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
  operator: unknown;
};

type EnhancedAppendTurnSender = {
  externalId: string;
  displayName: string;
  operator: boolean;
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
  const operator = sender.operator;

  if (externalId.length === 0 || displayName.length === 0) {
    return { valid: false };
  }

  return {
    valid: true,
    sender: {
      externalId,
      displayName,
      operator,
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

function parseCreatorDirectiveIdFromPath(
  pathname: string,
): ReturnType<typeof parseCreatorDirectiveId> | null | undefined {
  const prefix = "/memory/directives/";
  if (!pathname.startsWith(prefix)) {
    return undefined;
  }

  const rawId = pathname.slice(prefix.length);
  if (rawId === "" || rawId.includes("/")) {
    return null;
  }

  try {
    return parseCreatorDirectiveId(rawId);
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

type PublicEpisodeMetadata = {
  occurred_at: number;
  participant_names: string[];
};

function parsePublicEpisodeParticipantEntityId(value: string): EntityRecord["id"] | null {
  const prefixedEntityId = parseEpisodeParticipantEntityIdTerm(value);
  if (prefixedEntityId !== null) {
    return prefixedEntityId;
  }

  const bareEntityId = entityIdSchema.safeParse(value);
  return bareEntityId.success ? bareEntityId.data : null;
}

function createPublicEpisodeMetadataProjector(
  episodes: readonly Pick<Episode, "participants">[],
  entities: Pick<Borg["entities"], "get" | "getSelf">,
): (episode: Pick<Episode, "start_time" | "participants">) => PublicEpisodeMetadata {
  const referencedEntityIds = dedupePreservingOrder(
    episodes.flatMap((episode) =>
      episode.participants.flatMap((participant) => {
        const entityId = parsePublicEpisodeParticipantEntityId(participant.trim());

        return entityId === null ? [] : [entityId];
      }),
    ),
  );
  const selfEntity = entities.getSelf() ?? undefined;
  const entitiesById = new Map<EntityRecord["id"], EntityRecord>();
  if (selfEntity !== undefined) {
    entitiesById.set(selfEntity.id, selfEntity);
  }

  for (const entityId of referencedEntityIds) {
    if (entitiesById.has(entityId)) {
      continue;
    }

    const entity = entities.get(entityId);
    if (entity !== null) {
      entitiesById.set(entityId, entity);
    }
  }
  const selfNames = new Set(
    selfEntity === undefined
      ? []
      : [selfEntity.canonical_name, ...selfEntity.aliases].map((name) => name.trim()),
  );

  return (episode) => ({
    occurred_at: episode.start_time,
    participant_names: dedupePreservingOrder(
      episode.participants.flatMap((participant) => {
        const displayName = participant.trim();
        const entityId = parsePublicEpisodeParticipantEntityId(displayName);

        if (entityId !== null) {
          const entity = entitiesById.get(entityId);
          return entity === undefined ? [] : [entity.canonical_name];
        }

        if (displayName === "") {
          return [];
        }

        return selfEntity !== undefined && selfNames.has(displayName)
          ? [selfEntity.canonical_name]
          : [displayName];
      }),
    ),
  });
}

type EpisodeExclusions = z.infer<typeof episodeExclusionsSchema>;
type SidecarEpisodeHit = Pick<RetrievedEpisode, "episode" | "score" | "rawScore"> &
  Partial<Pick<RetrievedEpisode, "citationChain">>;

// The period a planner cue names, in the same shape as an explicit time_range, so the response can
// prefer in-period episodes and flag them whichever way the period arrived. Open ends fall back to
// the beginning of time and to now.
function temporalCueRange(
  cue: TemporalCue | null,
  nowMs: number,
): { start: number; end: number } | undefined {
  if (cue === null || (cue.sinceTs === undefined && cue.untilTs === undefined)) {
    return undefined;
  }
  const start = cue.sinceTs ?? 0;
  const end = cue.untilTs ?? nowMs;
  return start <= end ? { start, end } : undefined;
}

function projectAutobiographicalRecallForResponse(
  recall: AutobiographicalRecallResult,
  visibleAudienceEntityIds: readonly EntityId[],
): Record<string, unknown> {
  const eligible = recall.evidence.filter((item) => SIDECAR_AUTOBIOGRAPHICAL_KINDS.has(item.kind));
  const visible = eligible.filter((item) =>
    isMemoryDisclosureLabelVisibleToAnyAudience(item.disclosureLabel, visibleAudienceEntityIds),
  );
  const included = visible.slice(0, MAX_AUTOBIOGRAPHICAL_ROWS);
  return {
    window: {
      since: recall.window.startMs,
      until: recall.window.endMs,
      label: recall.window.label,
      // The service names its cue source after Sol's perception; here the cue is the planner's.
      source:
        recall.window.source === "perception_temporal_cue"
          ? "planner_temporal_cue"
          : recall.window.source,
    },
    evidence: included.map((item) => ({
      id: item.id,
      kind: item.kind,
      group: item.groupLabel,
      occurred_at: item.occurredAt,
      relative_age: item.relativeAge,
      text: item.text,
      source_episode_ids: [...item.sourceEpisodeIds],
      disclosure: {
        class: item.disclosureLabel.disclosureClass,
        origin_audience_entity_ids: [...item.disclosureLabel.originAudienceEntityIds],
        private_to_entity_ids: [...item.disclosureLabel.privateToEntityIds],
        public_to_entity_ids: [...item.disclosureLabel.publicToEntityIds],
      },
    })),
    hidden_count: eligible.length - visible.length,
    truncated_count: visible.length - included.length,
  };
}

// These patterns are explicit protocol handles supplied by the caller. Matching them mechanically
// does not interpret user-authored language or infer episode meaning.
function episodeMatchesExclusions(episode: Episode, exclusions?: EpisodeExclusions): boolean {
  if (exclusions === undefined) {
    return false;
  }

  return (
    exclusions.title_prefixes.some((prefix) => episode.title.startsWith(prefix)) ||
    exclusions.narrative_markers.some((marker) => episode.narrative.includes(marker))
  );
}

function projectEpisodeHitsForResponse(
  hits: readonly SidecarEpisodeHit[],
  entities: Pick<Borg["entities"], "get" | "getSelf">,
  includeDisclosure: boolean,
  options: {
    includeSourceMessages?: boolean;
    timeRange?: { start: number; end: number };
  } = {},
): Array<Record<string, unknown>> {
  if (hits.length === 0) {
    return [];
  }

  const projectMetadata = createPublicEpisodeMetadataProjector(
    hits.map((hit) => hit.episode),
    entities,
  );
  const originAudienceEntityIds = includeDisclosure
    ? dedupePreservingOrder(
        hits.flatMap(
          (hit) => memoryDisclosureLabelFromEpisodeAccess(hit.episode).originAudienceEntityIds,
        ),
      )
    : [];
  const originAudienceNames = new Map<EntityId, string>();

  for (const entityId of originAudienceEntityIds) {
    const entity = entities.get(entityId);
    if (entity !== null) {
      originAudienceNames.set(entityId, entity.canonical_name);
    }
  }

  return hits.map((hit) => {
    const sourceMessages = options.includeSourceMessages
      ? (hit.citationChain ?? [])
          .filter(isNarrativeStreamEntry)
          .flatMap((entry) =>
            typeof entry.content === "string" ? [{ entry, content: entry.content }] : [],
          )
          .slice(0, MAX_RECALLED_SOURCE_MESSAGES_PER_EPISODE)
          .map(({ entry, content }) => {
            const speaker =
              entry.sender_entity_id === null ? null : entities.get(entry.sender_entity_id);

            return {
              id: entry.id,
              kind: entry.kind,
              occurred_at: entry.observed_at ?? entry.timestamp,
              ...(speaker === null ? {} : { speaker_name: speaker.canonical_name }),
              text: clipRecalledEvidenceText(content),
            };
          })
      : undefined;
    const base = {
      id: hit.episode.id,
      title: hit.episode.title,
      narrative: hit.episode.narrative,
      score: hit.score,
      raw_score: hit.rawScore,
      location: hit.episode.location,
      ...projectMetadata(hit.episode),
      ...(options.timeRange === undefined
        ? {}
        : {
            in_time_range:
              hit.episode.start_time >= options.timeRange.start &&
              hit.episode.start_time <= options.timeRange.end,
          }),
      ...(sourceMessages === undefined ? {} : { source_messages: sourceMessages }),
    };

    if (!includeDisclosure) {
      return base;
    }

    const disclosure = memoryDisclosureLabelFromEpisodeAccess(hit.episode);
    return {
      ...base,
      disclosure: {
        class: disclosure.disclosureClass,
        origin_audience_names: disclosure.originAudienceEntityIds.flatMap((entityId) => {
          const name = originAudienceNames.get(entityId);
          return name === undefined ? [] : [name];
        }),
      },
    };
  });
}

function projectEpisodeForList(
  episode: Episode,
  metadata: PublicEpisodeMetadata,
): {
  id: Episode["id"];
  title: string;
  narrative: string;
  significance: number;
  tags: string[];
  source_stream_ids: Episode["source_stream_ids"];
  location: string | null;
  occurred_at: number;
  participant_names: string[];
} {
  return {
    id: episode.id,
    title: episode.title,
    narrative: episode.narrative,
    significance: episode.significance,
    tags: episode.tags,
    source_stream_ids: episode.source_stream_ids,
    location: episode.location,
    ...metadata,
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

function directiveContentText(directive: CreatorDirective): string {
  return directive.operational_directive ?? directive.canonical_fact ?? "";
}

function projectCreatorDirectiveForAdmin(directive: CreatorDirective) {
  return {
    id: directive.id,
    kind: directive.kind,
    status: directive.status,
    text: directiveContentText(directive),
    content_scope: directive.disclosure_policy.content_scope,
    priority: directive.priority,
    topic_tags: [...directive.disclosure_policy.topic_tags],
    created_at: directive.created_at,
  };
}

function projectApplicableCreatorDirective(applicable: CreatorDirectiveApplicable) {
  const text =
    applicable.render_mode === "boundary"
      ? (applicable.directive.disclosure_policy.boundary_prompt ?? "")
      : directiveContentText(applicable.directive);

  return {
    id: applicable.directive.id,
    kind: applicable.directive.kind,
    render_mode: applicable.render_mode,
    text,
    content_scope: applicable.directive.disclosure_policy.content_scope,
    priority: applicable.directive.priority,
    topic_tags: [...applicable.directive.disclosure_policy.topic_tags],
  };
}

function errorCode(error: unknown): unknown {
  return error !== null && typeof error === "object" && "code" in error
    ? (error as { code?: unknown }).code
    : undefined;
}

function isInvalidEpisodeCursorError(error: unknown): boolean {
  return errorCode(error) === "EPISODE_CURSOR_INVALID";
}

function projectRecentActivity(
  event: ActivityVisibleSessionEvent,
  nowMs: number,
  sourceEntries: ReadonlyMap<StreamEntryId, StreamEntry>,
) {
  const relativeAge = formatRelativeAge(event.occurredAt, nowMs);
  const conversation =
    event.conversationKind === "dm"
      ? { type: "personal" as const, name: event.conversationName }
      : event.conversationKind === "thread"
        ? { type: "groupChat" as const, name: event.conversationName }
        : { type: "channel" as const, name: event.conversationName };
  const location =
    conversation.type === "personal"
      ? `personal chat "${conversation.name}"`
      : conversation.type === "groupChat"
        ? `group chat "${conversation.name}"`
        : `channel "${conversation.name}"`;
  const text =
    event.kind === "user_contact"
      ? `${event.participantLabel} contacted the agent ${relativeAge} in ${location}.`
      : `The agent replied to ${event.participantLabel} ${relativeAge} in ${location}.`;
  const expectedSourceKind = event.kind === "user_contact" ? "user_msg" : "agent_msg";
  const sourceEntry = event.sourceStreamEntryIds
    .map((entryId) => sourceEntries.get(entryId))
    .find(
      (entry) =>
        entry?.session_id === event.sessionId &&
        entry.kind === expectedSourceKind &&
        typeof entry.content === "string",
    );
  const excerpt =
    sourceEntry !== undefined && typeof sourceEntry.content === "string"
      ? clipRecalledEvidenceText(sourceEntry.content)
      : null;

  return {
    kind: event.kind,
    occurred_at: event.occurredAt,
    occurred_at_iso: new Date(event.occurredAt).toISOString(),
    relative_age: relativeAge,
    session: event.sessionId,
    conversation,
    participant_name: event.participantLabel,
    text,
    ...(excerpt === null ? {} : { excerpt }),
  };
}

function resolveKnownExternalEntityIds(input: {
  borg: Borg;
  source: string;
  externalIds: readonly string[];
  kind: "person" | "group";
}): EntityId[] | null {
  const resolved: EntityId[] = [];

  for (const externalId of dedupePreservingOrder(input.externalIds)) {
    const entityId = input.borg.entities.findByExternalId(input.source, externalId);
    const entity = entityId === null ? null : input.borg.entities.get(entityId);

    if (entity === null || entity.kind !== input.kind) {
      return null;
    }

    resolved.push(entity.id);
  }

  return resolved;
}

function buildCreatorDirectiveQueueInput(input: {
  body: z.infer<typeof directiveAdminBodySchema>;
  adminEntityId: EntityId;
  adminSessionId: SessionId;
  sourceStreamEntryId: ReturnType<typeof createStreamEntryId>;
  allowedEntityIds: readonly EntityId[];
  excludedEntityIds: readonly EntityId[];
  subjectEntityId: EntityId | null;
}): CreatorDirectiveQueueInput {
  const operational =
    input.body.kind === "response_policy" || input.body.kind === "routing_instruction";
  const subjectKind =
    input.body.kind === "self_identity"
      ? "borg_self"
      : input.body.kind === "subject_fact"
        ? "entity"
        : "system";

  return {
    kind: input.body.kind,
    createdByEntityId: input.adminEntityId,
    sourceSessionId: input.adminSessionId,
    authorizationStreamEntryIds: [input.sourceStreamEntryId],
    contentSourceStreamEntryIds: [input.sourceStreamEntryId],
    subjectKind,
    subjectEntityId: subjectKind === "entity" ? input.subjectEntityId : null,
    canonicalFact: operational ? null : input.body.text,
    operationalDirective: operational ? input.body.text : null,
    disclosurePolicy: {
      content_scope: input.body.content_scope,
      allowed_entity_ids: [...input.allowedEntityIds],
      excluded_entity_ids: [...input.excludedEntityIds],
      subject_may_know: input.body.content_scope === "subject_only" ? true : null,
      mention_policy: input.body.mention_policy,
      denied_audience_behavior: "omit",
      boundary_prompt: input.body.text,
      topic_tags: [...input.body.topic_tags],
    },
    activationPolicy: {
      scope: "same_as_disclosure",
      allowed_entity_ids: [],
      excluded_entity_ids: [],
    },
    priority: input.body.priority,
  };
}

type DirectiveAdminIdentity = {
  entityId: EntityId;
  sessionId: SessionId;
};

function ensureDirectiveAdminIdentity(borg: Borg): DirectiveAdminIdentity {
  const entityId = borg.entities.resolveExternal({
    source: SIDECAR_ADMIN_EXTERNAL_ID_SOURCE,
    externalId: SIDECAR_ADMIN_EXTERNAL_ID,
    canonicalName: "Memory sidecar admin API",
    kind: "abstract",
    provenance: "creator_directive",
  });
  const sessionId = sessionFromCaller(SIDECAR_ADMIN_SESSION_EXTERNAL_ID);
  borg.sessions.ensure({
    session_id: sessionId,
    source_type: "memory_sidecar",
    source_external_id: SIDECAR_ADMIN_SESSION_EXTERNAL_ID,
    label: "Memory sidecar admin API",
    audience_label: "Memory sidecar admin API",
    audience_entity_id: entityId,
    conversation_kind: "dm",
    audience_role: "operator",
    status: "active",
  });

  return { entityId, sessionId };
}

async function appendDirectiveAdminEvent(input: {
  borg: Borg;
  admin: DirectiveAdminIdentity;
  content: Record<string, unknown>;
}) {
  const entry = await input.borg.stream.append(
    {
      kind: "internal_event",
      content: input.content,
      audience: input.admin.entityId,
    },
    { session: input.admin.sessionId },
  );
  input.borg.sessions.touch(input.admin.sessionId, {
    at: entry.timestamp,
    messageCountDelta: 1,
  });
  return entry;
}

function scheduleIngestion(pool: MemoryPool, tenant: string, session: SessionId): void {
  void pool
    .withTenant(tenant, (borg) => borg.episodic.ingest({ session }))
    .catch((error: unknown) => {
      console.error(`memory-sidecar: background ingestion failed for tenant "${tenant}"`, error);
    });
}

export function createMemoryHandler(options: MemoryHandlerOptions): RequestHandler {
  const { pool, token, traceRegistry, maintenanceCoordinator, inboxWaiters } = options;
  const maxBodyBytes = options.maxBodyBytes ?? DEFAULT_MAX_BODY_BYTES;
  const maxRecallLimit = options.maxRecallLimit ?? DEFAULT_MAX_RECALL_LIMIT;
  const recallAbstainThreshold = options.recallAbstainThreshold ?? 0;
  const recallDeadlineMs = options.recallDeadlineMs ?? DEFAULT_RECALL_DEADLINE_MS;
  const recentActivityWindowMs = Math.max(
    0,
    Math.floor(options.recentActivityWindowMs ?? DEFAULT_RECENT_ACTIVITY_WINDOW_MS),
  );
  const recentActivityLimit = Math.max(
    1,
    Math.min(
      MAX_RECALL_QUERY_ACTIVITY_ROWS,
      Math.floor(options.recentActivityLimit ?? DEFAULT_RECENT_ACTIVITY_LIMIT),
    ),
  );
  const activityExcerptHydrationBudgetMs = Math.max(
    0,
    Math.floor(
      options.activityExcerptHydrationBudgetMs ?? DEFAULT_ACTIVITY_EXCERPT_HYDRATION_BUDGET_MS,
    ),
  );
  const recencyPrior = options.recencyPrior;
  const recallSemanticVariantCount = Math.max(
    MIN_RECALL_EXPANSION_SEMANTIC_VARIANTS,
    Math.min(
      MAX_RECALL_EXPANSION_SEMANTIC_VARIANTS,
      Math.floor(options.recallSemanticVariantCount ?? 1),
    ),
  );
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

    if (method === "POST" && rawPath === "/memory/maintenance/inbox-reply-activity") {
      // Repairs inbox sessions whose reply terminals never got a borg_replied activity event
      // (inbox path before 2026-09-05, or a crash between terminal commit and projection).
      // dry_run defaults to true; only an explicit false writes.
      const body = await readJsonObjectBody(req, res, maxBodyBytes);
      if (body === null) {
        return;
      }
      const parsed = inboxReplyActivityReconcileBodySchema.safeParse(body);
      if (!parsed.success) {
        send(res, 400, { error: "invalid inbox reply activity reconcile body" });
        return;
      }
      const input = parsed.data;
      try {
        const result = await pool.withTenant(
          input.tenant,
          (borg) =>
            borg.inbox.reconcileReplyActivity({
              dryRun: input.dry_run,
              ...(input.limit === undefined ? {} : { limit: input.limit }),
              ...(input.since === undefined ? {} : { sinceMs: Date.parse(input.since) }),
              ...(input.until === undefined ? {} : { untilMs: Date.parse(input.until) }),
            }),
          { exclusive: true },
        );
        send(res, 200, { ok: true, tenant: input.tenant, ...result });
      } catch (error) {
        console.error(`memory-sidecar: ${rawPath} failed for tenant "${input.tenant}"`, error);
        send(res, 503, { error: "tenant unavailable" });
      }
      return;
    }

    if (
      method === "POST" &&
      (rawPath === "/memory/enqueue" ||
        rawPath === "/memory/await-response" ||
        rawPath === "/memory/inbox-progress")
    ) {
      if (inboxWaiters === undefined) {
        send(res, 503, { error: "teams inbox unavailable" });
        return;
      }
      const body = await readJsonObjectBody(req, res, maxBodyBytes);
      if (body === null) {
        return;
      }

      if (rawPath === "/memory/enqueue") {
        const parsed = memoryEnqueueBodySchema.safeParse(body);
        if (!parsed.success) {
          send(res, 400, { error: "invalid memory enqueue body" });
          return;
        }
        const input = parsed.data;
        const session = sessionFromCaller(input.session);
        try {
          const result = await pool.withTenant(
            input.tenant,
            async (borg) => {
              const claimsInbox = borg.sessions.get(session)?.source_type !== "teams_inbox";
              const identity = resolveTeamAgentIdentity({
                borg,
                session,
                rawSession: input.session,
                sender: {
                  externalId: input.sender.external_id,
                  displayName: input.sender.display_name,
                  operator: input.sender.operator,
                },
                conversation: input.conversation,
                claimInbox: true,
              });
              if (identity.senderEntityId === null) {
                throw new Error("memory enqueue identity requires a sender entity");
              }
              if (claimsInbox) {
                await borg.inbox.sealPendingBacklog({
                  sessionId: session,
                  reason: "Legacy append-turn backlog sealed when the session joined Teams inbox",
                });
              }
              return borg.enqueueMessage({
                session: {
                  ...identity.sessionEnsureInput,
                  source_external_id: input.conversation.external_id,
                },
                userMessage: input.text,
                senderEntityId: identity.senderEntityId,
                sourceMessageKey: {
                  source_type: "teams_inbox",
                  source_external_id: input.conversation.external_id,
                  external_message_id: input.external_message_id,
                },
                observedAt: Date.parse(input.observed_at),
                audience: identity.audienceEntity.canonical_name,
                audienceEntityId: identity.audienceEntity.id,
                conversation: identity.conversation,
                metadata: {
                  teams_inbox: {
                    thread_id: input.session,
                    sender: {
                      external_id: input.sender.external_id,
                      display_name: input.sender.display_name,
                      bot: input.sender.bot,
                    },
                    mentioned: input.flags.mentioned,
                    quotes_bot: input.flags.quotes_bot,
                  },
                },
              });
            },
            { exclusive: true },
          );
          send(res, 200, {
            status: result.status,
            sidecar_session_id: result.sessionId,
            entry_id: result.streamEntryId,
          });
        } catch (error) {
          console.error(`memory-sidecar: ${rawPath} failed for tenant "${input.tenant}"`, error);
          send(res, 503, { error: "tenant unavailable" });
        }
        return;
      }

      if (rawPath === "/memory/inbox-progress") {
        const parsed = memoryInboxProgressBodySchema.safeParse(body);
        if (!parsed.success) {
          send(res, 400, { error: "invalid memory inbox-progress body" });
          return;
        }
        const input = parsed.data;
        try {
          const sessionExists = await pool.withTenant(
            input.tenant,
            (borg) => borg.sessions.get(input.sidecar_session_id) !== null,
          );
          if (!sessionExists) {
            send(res, 404, { error: "session not found" });
            return;
          }
          inboxWaiters.markGenerating({
            tenant: input.tenant,
            sessionId: input.sidecar_session_id,
            entryIds: input.entry_ids,
          });
          send(res, 200, { ok: true });
        } catch (error) {
          console.error(`memory-sidecar: ${rawPath} failed for tenant "${input.tenant}"`, error);
          send(res, 503, { error: "tenant unavailable" });
        }
        return;
      }

      const parsed = memoryAwaitResponseBodySchema.safeParse(body);
      if (!parsed.success) {
        send(res, 400, { error: "invalid memory await-response body" });
        return;
      }
      const input = parsed.data;
      const scan = async () =>
        pool.withTenant(input.tenant, (borg) =>
          borg.inbox.findTerminalCoveringEntry({
            sessionId: input.sidecar_session_id,
            entryId: input.entry_id,
          }),
        );
      let cancelWaiter: (() => void) | undefined;
      let disconnected = false;
      const connectionDestroyed = () =>
        req.aborted || res.destroyed || (req.destroyed && !req.complete);
      const markDisconnected = () => {
        disconnected = true;
        cancelWaiter?.();
      };
      const onRequestClose = () => {
        if (req.aborted || !req.complete) {
          markDisconnected();
        }
      };
      const onResponseClose = () => {
        if (!res.writableEnded) {
          markDisconnected();
        }
      };
      const removeDisconnectListeners = () => {
        req.off("aborted", markDisconnected);
        req.off("close", onRequestClose);
        res.off("close", onResponseClose);
      };
      req.once("aborted", markDisconnected);
      req.once("close", onRequestClose);
      res.once("close", onResponseClose);
      if (connectionDestroyed()) {
        markDisconnected();
      }
      try {
        const first = await scan();
        if (disconnected || connectionDestroyed()) {
          markDisconnected();
          return;
        }
        if (first.status === "unknown_entry" || first.status === "session_mismatch") {
          send(res, 404, { error: "entry not found in session" });
          return;
        }
        if (first.status === "found") {
          inboxWaiters.resolveTerminal(input.tenant, first.terminalEntry);
          send(res, 200, awaitResponseForTerminal({ terminalEntry: first.terminalEntry }));
          return;
        }

        const waiter = inboxWaiters.register({
          tenant: input.tenant,
          sessionId: input.sidecar_session_id,
          entryId: input.entry_id,
          timeoutMs: input.timeout_ms,
          seenGenerating: input.seen_generating,
        });
        cancelWaiter = waiter.cancel;
        if (disconnected || connectionDestroyed()) {
          markDisconnected();
          return;
        }
        const second = await scan();
        if (second.status === "unknown_entry" || second.status === "session_mismatch") {
          waiter.cancel();
          if (!disconnected) {
            send(res, 404, { error: "entry not found in session" });
          }
          return;
        }
        if (second.status === "found") {
          inboxWaiters.resolveTerminal(input.tenant, second.terminalEntry);
          if (!disconnected) {
            send(res, 200, awaitResponseForTerminal({ terminalEntry: second.terminalEntry }));
          }
          return;
        }

        const response = await waiter.promise;
        if (!disconnected) {
          send(res, 200, response);
        }
      } catch (error) {
        cancelWaiter?.();
        console.error(`memory-sidecar: ${rawPath} failed for tenant "${input.tenant}"`, error);
        if (!disconnected) {
          send(res, 503, { error: "tenant unavailable" });
        }
      } finally {
        removeDisconnectListeners();
      }
      return;
    }

    const creatorDirectiveId = parseCreatorDirectiveIdFromPath(rawPath);

    if (method === "DELETE" && creatorDirectiveId !== undefined) {
      if (creatorDirectiveId === null) {
        send(res, 400, { error: "invalid directive id" });
        return;
      }
      const tenant = requiredSingleQueryValue(res, searchParams, "tenant");
      if (tenant === null || !validateTenantForResponse(res, tenant)) {
        return;
      }
      const body = await readJsonObjectBody(req, res, maxBodyBytes);
      if (body === null) {
        return;
      }
      const parsedBody = directiveRevokeBodySchema.safeParse(body);
      if (!parsedBody.success) {
        send(res, 400, { error: "invalid directive revoke body" });
        return;
      }

      try {
        const result = await pool.withTenant(
          tenant,
          async (borg) => {
            const current = borg.creatorDirectives.get(creatorDirectiveId);

            if (current === null) {
              return { status: "missing" as const };
            }
            if (current.status !== "active") {
              return { status: "inactive" as const };
            }

            const admin = ensureDirectiveAdminIdentity(borg);
            const provenanceEntry = await appendDirectiveAdminEvent({
              borg,
              admin,
              content: {
                event: "memory_sidecar.operator_directive_revoke_requested",
                directive_id: creatorDirectiveId,
                reason: parsedBody.data.reason,
              },
            });

            let directive: CreatorDirective | null;
            try {
              directive = borg.creatorDirectives.revoke(creatorDirectiveId, parsedBody.data.reason);
            } catch (error) {
              try {
                const code = errorCode(error);
                await appendDirectiveAdminEvent({
                  borg,
                  admin,
                  content: {
                    event: "memory_sidecar.operator_directive_revoke_failed",
                    directive_id: creatorDirectiveId,
                    reason: parsedBody.data.reason,
                    provenance_stream_entry_id: provenanceEntry.id,
                    failure_code: typeof code === "string" ? code : "UNKNOWN",
                  },
                });
              } catch (auditError) {
                console.error(
                  `memory-sidecar: failed to record directive revoke failure for tenant "${tenant}"`,
                  auditError,
                );
              }
              throw error;
            }

            if (directive !== null) {
              return { status: "revoked" as const, directive };
            }

            await appendDirectiveAdminEvent({
              borg,
              admin,
              content: {
                event: "memory_sidecar.operator_directive_revoke_failed",
                directive_id: creatorDirectiveId,
                reason: parsedBody.data.reason,
                provenance_stream_entry_id: provenanceEntry.id,
                failure_code: "DIRECTIVE_NOT_ACTIVE",
              },
            });
            return { status: "inactive" as const };
          },
          { exclusive: true },
        );

        if (result.status === "missing") {
          send(res, 404, { error: "directive not found" });
          return;
        }
        if (result.status === "inactive") {
          send(res, 409, { error: "directive is not active" });
          return;
        }

        send(res, 200, {
          ok: true,
          directive: projectCreatorDirectiveForAdmin(result.directive),
        });
      } catch (error) {
        console.error(`memory-sidecar: ${rawPath} failed for tenant "${tenant}"`, error);
        send(res, 500, { error: "internal error" });
      }
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
      const isDirectiveListPath = rawPath === "/memory/directives";
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
        isDirectiveListPath ||
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
        isDirectiveListPath ||
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
                    TEAM_AGENT_SENDER_EXTERNAL_ID_SOURCE,
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

        if (isDirectiveListPath) {
          const directives = await pool.withTenant(tenant, (borg) =>
            borg.creatorDirectives.list({ status: "active" }).map(projectCreatorDirectiveForAdmin),
          );
          send(res, 200, { ok: true, directives });
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
          const result = await pool.withTenant(tenant, async (borg) => {
            const listed = await borg.episodic.list({
              limit,
              ...(cursor === undefined ? {} : { cursor }),
            });
            if (listed.items.length === 0) {
              return listed;
            }

            const projectMetadata = createPublicEpisodeMetadataProjector(
              listed.items,
              borg.entities,
            );

            return {
              ...listed,
              items: listed.items.map((episode) =>
                projectEpisodeForList(episode, projectMetadata(episode)),
              ),
            };
          });

          send(res, 200, {
            ok: true,
            episodes: result.items,
            ...(result.nextCursor === undefined ? {} : { nextCursor: result.nextCursor }),
          });
          return;
        }

        if (episodeId === null) {
          send(res, 400, { error: "invalid episode id" });
          return;
        }
        if (episodeId !== undefined) {
          const episode = await pool.withTenant(tenant, async (borg) => {
            const inspected = await borg.episodic.inspect(episodeId);

            if (inspected === null) {
              return null;
            }

            const projectMetadata = createPublicEpisodeMetadataProjector(
              [inspected],
              borg.entities,
            );
            return {
              ...episodeWithoutEmbedding(inspected),
              ...projectMetadata(inspected),
            };
          });
          if (episode === null) {
            send(res, 404, { ok: false });
            return;
          }

          send(res, 200, { ok: true, episode });
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
        rawPath !== "/memory/commitments" &&
        rawPath !== "/memory/context" &&
        rawPath !== "/memory/directives")
    ) {
      send(res, 404, { error: "not found" });
      return;
    }

    const body = await readJsonObjectBody(req, res, maxBodyBytes);
    if (body === null) {
      return;
    }

    const tenant = asString(body.tenant);
    if (!validateTenantForResponse(res, tenant)) {
      return;
    }

    try {
      if (rawPath === "/memory/context") {
        const parsed = memoryContextBodySchema.safeParse(body);

        if (!parsed.success) {
          send(res, 400, { error: "invalid memory context body" });
          return;
        }

        const requestedSections = new Set(parsed.data.sections ?? DEFAULT_MEMORY_CONTEXT_SECTIONS);
        const recallFocus = parsed.data.focus ?? parsed.data.query ?? "";
        const episodeLimit = Math.max(
          1,
          Math.min(
            maxRecallLimit,
            Math.floor(parsed.data.limit === undefined ? 8 : parsed.data.limit),
          ),
        );
        // Overfetch and account after selection whenever the response may be reordered or filtered
        // after retrieval: exclusions, an explicit range, or a planner-driven recall (focus) whose
        // cue can promote in-period episodes that a plain limit would already have cut.
        const deferEpisodeRetrievalAccounting =
          parsed.data.exclude !== undefined ||
          parsed.data.time_range !== undefined ||
          parsed.data.focus !== undefined;
        const episodeSearchLimit = deferEpisodeRetrievalAccounting
          ? Math.min(
              maxRecallLimit * EPISODE_OVERFETCH_MULTIPLIER,
              episodeLimit * EPISODE_OVERFETCH_MULTIPLIER,
            )
          : episodeLimit;
        const venueLimit = parsed.data.venue_limit ?? DEFAULT_VENUE_RECENT_LIMIT;
        const venueSearchLimit =
          parsed.data.exclude === undefined
            ? venueLimit
            : Math.min(
                MAX_VENUE_RECENT_LIMIT * EPISODE_OVERFETCH_MULTIPLIER,
                venueLimit * EPISODE_OVERFETCH_MULTIPLIER,
              );
        const session = sessionFromCaller(parsed.data.session);
        const identity = await pool.withTenant(
          tenant,
          (borg) => {
            const resolved = resolveTeamAgentIdentity({
              borg,
              session,
              rawSession: parsed.data.session,
              sender: {
                externalId: parsed.data.sender.external_id,
                displayName: parsed.data.sender.display_name,
                operator: parsed.data.sender.operator,
              },
              conversation: parsed.data.conversation,
            });
            if (resolved.senderEntityId === null) {
              throw new Error("memory context identity requires a sender entity");
            }
            const senderEntityId = resolved.senderEntityId;
            const seenParticipantExternalIds = new Set([parsed.data.sender.external_id]);
            const participantEntityIds: EntityId[] = [];

            for (const participant of parsed.data.participants ?? []) {
              if (seenParticipantExternalIds.has(participant.external_id)) {
                continue;
              }
              seenParticipantExternalIds.add(participant.external_id);
              participantEntityIds.push(
                borg.entities.resolveExternal({
                  source: TEAM_AGENT_SENDER_EXTERNAL_ID_SOURCE,
                  externalId: participant.external_id,
                  canonicalName: participant.display_name,
                  kind: "person",
                  provenance: "transport_sender",
                }),
              );
            }

            borg.sessions.ensure(resolved.sessionEnsureInput);
            return { ...resolved, senderEntityId, participantEntityIds };
          },
          { exclusive: true },
        );
        const nowMs = Date.now();
        const context = await pool.withTenant(tenant, async (borg) => {
          const observedGroupAudienceEntityIds =
            parsed.data.conversation.type === "personal" &&
            (requestedSections.has("episodes") || requestedSections.has("recent_activity"))
              ? borg.activity.listObservedGroupAudienceEntityIdsForSpeaker(identity.senderEntityId)
              : [];
          const visibleAudienceEntityIds = dedupePreservingOrder([
            identity.audienceEntity.id,
            ...observedGroupAudienceEntityIds,
          ]);
          const recentActivityEvents = requestedSections.has("recent_activity")
            ? borg.activity.listRecentVisibleOtherSessionEvents({
                currentSessionId: session,
                audienceEntityIds: visibleAudienceEntityIds,
                sinceMs: nowMs - recentActivityWindowMs,
                limit: recentActivityLimit,
              })
            : [];
          // Planner context reads the memory owner's own recent replies elsewhere through an
          // owner-only pass of the same visibility-gated query. Deriving them from the shared
          // response list starved the planner on busy group days: the 12 newest visible rows
          // were all user_contact messages, so zero owner rows reached the planner.
          const plannerOwnerActivityEvents = requestedSections.has("episodes")
            ? borg.activity.listRecentVisibleOtherSessionEvents({
                currentSessionId: session,
                audienceEntityIds: visibleAudienceEntityIds,
                sinceMs: nowMs - recentActivityWindowMs,
                limit: recentActivityLimit,
                kinds: ["borg_replied"],
              })
            : [];
          // One hydration pass for both lists. Owner rows go first: indexed hydration keeps input
          // order and stops at its budget, so under pressure the response excerpts degrade before
          // the planner loses its owner rows again.
          const recentActivitySourceIds = dedupePreservingOrder(
            [...plannerOwnerActivityEvents, ...recentActivityEvents].flatMap(
              (event) => event.sourceStreamEntryIds,
            ),
          );
          let recentActivitySourceEntries = new Map<StreamEntryId, StreamEntry>();
          if (recentActivitySourceIds.length > 0) {
            try {
              recentActivitySourceEntries = await borg.stream.hydrateIndexed(
                recentActivitySourceIds,
                { budgetMs: activityExcerptHydrationBudgetMs, activeOnly: true },
              );
            } catch (error) {
              // Excerpts are optional disclosure context; the event itself remains useful.
              console.warn("memory-sidecar: recent-activity excerpt hydration failed", {
                reason: RECENT_ACTIVITY_EXCERPT_HYDRATION_FAILURE_REASON,
                tenant,
                source_id_count: recentActivitySourceIds.length,
                error_name: error instanceof Error ? error.name : typeof error,
              });
            }
          }
          const recentActivity = recentActivityEvents.map((event) =>
            projectRecentActivity(event, nowMs, recentActivitySourceEntries),
          );
          const plannerOwnerActivity = plannerOwnerActivityEvents.flatMap((event) => {
            const projected = projectRecentActivity(event, nowMs, recentActivitySourceEntries);
            if (event.kind !== "borg_replied" || projected.excerpt === undefined) {
              return [];
            }

            return [
              {
                excerpt: projected.excerpt,
                occurredAt: event.occurredAt,
                venue: projected.conversation,
                counterpartyName: event.participantLabel,
              },
            ];
          });
          const commitments = requestedSections.has("commitments")
            ? borg.commitments
                .list({
                  activeOnly: true,
                  audienceEntityId: identity.audienceEntity.id,
                })
                .sort(compareCommitmentsForResponse)
                .slice(0, MAX_COMMITMENT_RESPONSE_ITEMS)
                .map(projectCommitment)
            : [];
          const participantEntityIds = dedupePreservingOrder([
            identity.senderEntityId,
            ...identity.participantEntityIds,
            identity.audienceEntity.id,
          ]);
          const directives = requestedSections.has("directives")
            ? borg.creatorDirectives
                .listApplicable({
                  currentAudienceEntityId: identity.audienceEntity.id,
                  participantEntityIds,
                  allowListAudienceEntityIds: participantEntityIds,
                  sessionRole: identity.audienceRole,
                  trustedTenantOperator: parsed.data.sender.operator,
                })
                .filter(
                  (applicable) => applicable.activation.active && applicable.render_mode !== "omit",
                )
                .map(projectApplicableCreatorDirective)
            : [];
          const venueCandidates = requestedSections.has("venue_recent")
            ? await borg.episodic.listRecentForSession({
                sessionId: session,
                sinceMs: parsed.data.venue_since!,
                audienceEntityId: identity.audienceEntity.id,
                limit: venueSearchLimit,
              })
            : [];
          const visibleVenueCandidates = venueCandidates.filter((candidate) =>
            isEpisodeAccessVisibleToAnyAudience(candidate.episode, [identity.audienceEntity.id]),
          );
          const venueRecent = projectEpisodeHitsForResponse(
            visibleVenueCandidates
              .filter(
                (candidate) => !episodeMatchesExclusions(candidate.episode, parsed.data.exclude),
              )
              .slice(0, venueLimit)
              .map((candidate) => ({
                episode: candidate.episode,
                score: 0,
                rawScore: 0,
              })),
            borg.entities,
            true,
          );

          const plannerOwnerLivedExperience = requestedSections.has("episodes")
            ? borg.self.livedExperience
                .listDaySummaries({
                  fromMs: nowMs - PLANNER_LIVED_EXPERIENCE_WINDOW_MS,
                  toMs: nowMs,
                  limit: PLANNER_LIVED_EXPERIENCE_LIMIT,
                })
                .map((summary) => ({
                  day: summary.utc_day,
                  gist: summary.gist,
                  salience: summary.salience,
                  disclosure: {
                    class: summary.disclosure_label.disclosureClass,
                    origin_audience_entity_ids: [
                      ...summary.disclosure_label.originAudienceEntityIds,
                    ],
                    private_to_entity_ids: [...summary.disclosure_label.privateToEntityIds],
                    public_to_entity_ids: [...summary.disclosure_label.publicToEntityIds],
                  },
                }))
            : [];

          return {
            visibleAudienceEntityIds,
            recentActivity,
            plannerOwnerActivity,
            plannerOwnerLivedExperience,
            memoryOwnerName: borg.entities.getSelf()?.canonical_name,
            commitments,
            directives,
            venueRecent,
          };
        });
        const degradations: RetrievalDegradation[] = [];
        let episodes: Array<Record<string, unknown>> = [];
        let hiddenEpisodeCount = 0;
        let degraded = false;
        let degradedReason = "";
        let abstained = false;
        let plannerTemporalCue: TemporalCue | null = null;

        if (requestedSections.has("episodes")) {
          const traceTurnId =
            traceRegistry === undefined ? undefined : nextRecallTraceTurnId(tenant);

          try {
            const recallResult = await raceRecallDeadline(
              pool.withTenant(tenant, async (borg) => {
                let recallPlan: RecallPlanOutcome | null = null;
                const recallOptions = {
                  limit: episodeSearchLimit,
                  audienceEntityId: identity.audienceEntity.id,
                  visibleAudienceEntityIds: context.visibleAudienceEntityIds,
                  onDegraded: (degradation: RetrievalDegradation) => degradations.push(degradation),
                  onRecallPlan: (plan: RecallPlanOutcome) => {
                    recallPlan = plan;
                  },
                  ...(parsed.data.entity_terms === undefined
                    ? {}
                    : { entityTerms: parsed.data.entity_terms }),
                  semanticVariantCount: recallSemanticVariantCount,
                  recallQueryPlannerContext: {
                    contextTurns: (parsed.data.context_turns ?? []).map((turn) => ({
                      role: turn.role,
                      content: turn.text,
                    })),
                    identity: {
                      ...(context.memoryOwnerName === undefined
                        ? {}
                        : { memoryOwnerName: context.memoryOwnerName }),
                      currentSenderName: parsed.data.sender.display_name,
                      currentAudienceName: identity.audienceEntity.canonical_name,
                      currentVenue: identity.conversation,
                      ...(parsed.data.entity_terms === undefined
                        ? {}
                        : { entityTerms: parsed.data.entity_terms }),
                    },
                    ownerRecentActivity: context.plannerOwnerActivity,
                    ownerLivedExperience: context.plannerOwnerLivedExperience,
                  },
                  ...(recencyPrior === undefined ? {} : { recencyPrior }),
                  ...(deferEpisodeRetrievalAccounting ? { recordRetrieval: false } : {}),
                  ...(traceTurnId === undefined ? {} : { traceTurnId }),
                };
                const recalled = await borg.episodic.search(recallFocus, {
                  ...recallOptions,
                  ...(parsed.data.time_range === undefined
                    ? {}
                    : {
                        timeRange: parsed.data.time_range,
                        strictTimeRange: false,
                      }),
                });
                const visible = recalled.filter((hit) =>
                  isEpisodeAccessVisibleToAnyAudience(
                    hit.episode,
                    context.visibleAudienceEntityIds,
                  ),
                );
                const eligible = visible.filter(
                  (hit) => !episodeMatchesExclusions(hit.episode, parsed.data.exclude),
                );
                // The period to prefer: an explicit time_range, else the cue the planner resolved
                // from FOCUS, which is the range a caller with its own parser used to send.
                const actedCue: TemporalCue | null =
                  (recallPlan as RecallPlanOutcome | null)?.temporalCue ?? null;
                const preferredRange = parsed.data.time_range ?? temporalCueRange(actedCue, nowMs);
                const ordered =
                  preferredRange === undefined
                    ? eligible
                    : [
                        ...eligible.filter(
                          (hit) =>
                            hit.episode.start_time >= preferredRange.start &&
                            hit.episode.start_time <= preferredRange.end,
                        ),
                        ...eligible.filter(
                          (hit) =>
                            hit.episode.start_time < preferredRange.start ||
                            hit.episode.start_time > preferredRange.end,
                        ),
                      ];
                const included = ordered.slice(0, episodeLimit);
                const topRawScore =
                  included.length === 0 ? null : Math.max(...included.map((hit) => hit.rawScore));
                const shouldAbstain =
                  recallAbstainThreshold > 0 &&
                  (topRawScore === null || topRawScore < recallAbstainThreshold);

                if (deferEpisodeRetrievalAccounting && !shouldAbstain) {
                  for (const hit of included) {
                    borg.episodic.recordRetrieval(hit.episode.id, hit.score);
                  }
                }

                return {
                  hiddenEpisodeCount: recalled.length - visible.length,
                  topRawScore,
                  plannerTemporalCue: parsed.data.time_range === undefined ? actedCue : null,
                  episodes: projectEpisodeHitsForResponse(included, borg.entities, true, {
                    includeSourceMessages: true,
                    ...(preferredRange === undefined ? {} : { timeRange: preferredRange }),
                  }),
                };
              }),
              recallDeadlineMs,
            );

            hiddenEpisodeCount = recallResult.hiddenEpisodeCount;
            plannerTemporalCue = recallResult.plannerTemporalCue;
            if (
              recallAbstainThreshold > 0 &&
              (recallResult.topRawScore === null ||
                recallResult.topRawScore < recallAbstainThreshold)
            ) {
              abstained = true;
              episodes = [];
            } else {
              episodes = recallResult.episodes;
            }
          } catch (error) {
            if (error instanceof RecallDeadlineExceeded) {
              degraded = true;
              degradedReason = `deadline: ${error.message}`;
            } else if (error instanceof EmbeddingError) {
              degraded = true;
              degradedReason = `embeddings: ${error.message}`;
            } else {
              throw error;
            }
          }

          if (degradations.length > 0) {
            degraded = true;
            const pipelineReason = degradations
              .map((entry) => `${entry.subsystem}: ${entry.reason}`)
              .join("; ");
            degradedReason =
              degradedReason.length === 0 ? pipelineReason : `${degradedReason}; ${pipelineReason}`;
          }
        }

        // The owner's own record for the cued period. A second, deadline-bounded pass so a slow
        // scan of other sessions' streams can only cost this section, never the episodes.
        let autobiographical: Record<string, unknown> | null = null;
        if (requestedSections.has("autobiographical") && plannerTemporalCue !== null) {
          const cue = plannerTemporalCue;
          const remainingMs = Math.min(
            MAX_AUTOBIOGRAPHICAL_BUDGET_MS,
            recallDeadlineMs - (Date.now() - nowMs) - AUTOBIOGRAPHICAL_HEADROOM_MS,
          );
          const noteDegradation = (reason: string): void => {
            degraded = true;
            degradedReason = degradedReason.length === 0 ? reason : `${degradedReason}; ${reason}`;
          };
          if (remainingMs < MIN_AUTOBIOGRAPHICAL_BUDGET_MS) {
            noteDegradation("autobiographical_recall: no deadline budget left after episodes");
          } else {
            try {
              autobiographical = await raceRecallDeadline(
                pool.withTenant(tenant, async (borg) => {
                  const recall = await borg.self.autobiographical.recall(
                    {
                      sessionId: session,
                      temporalCue: cue,
                      // Teams audiences are never the owner and the gate here is the cue alone: an
                      // operator role would open it on every turn, which is not what a period ask
                      // is.
                      isSelfAudience: false,
                      sessionAudienceRole: "participant",
                      perceptionMode: "problem_solving",
                    },
                    {
                      sessionCap: AUTOBIOGRAPHICAL_SESSION_CAP,
                      totalCap: AUTOBIOGRAPHICAL_TOTAL_CAP,
                    },
                  );
                  return recall === null
                    ? null
                    : projectAutobiographicalRecallForResponse(
                        recall,
                        context.visibleAudienceEntityIds,
                      );
                }),
                remainingMs,
              );
            } catch (error) {
              noteDegradation(
                `autobiographical_recall: ${error instanceof Error ? error.message : String(error)}`,
              );
            }
          }
        }

        const response: Record<string, unknown> = {
          ok: true,
          degraded,
          degraded_reason: degradedReason,
        };

        if (requestedSections.has("audience")) {
          response.audience = {
            entity_id: identity.audienceEntity.id,
            kind: identity.audienceEntity.kind,
            name: identity.audienceEntity.canonical_name,
            role: identity.audienceRole,
          };
        }
        if (requestedSections.has("episodes")) {
          response.episodes = episodes;
          response.hidden_episode_count = hiddenEpisodeCount;
          if (abstained) {
            response.abstained = true;
            response.abstain_reason = "low_relevance";
          }
        }
        if (requestedSections.has("recent_activity")) {
          response.recent_activity = context.recentActivity;
        }
        if (requestedSections.has("commitments")) {
          response.commitments = context.commitments;
        }
        if (requestedSections.has("directives")) {
          response.directives = context.directives;
        }
        if (requestedSections.has("venue_recent")) {
          response.venue_recent = context.venueRecent;
        }
        if (requestedSections.has("autobiographical")) {
          response.autobiographical = autobiographical;
        }

        send(res, 200, response);
        return;
      }

      if (rawPath === "/memory/directives") {
        const parsed = directiveAdminBodySchema.safeParse(body);

        if (!parsed.success) {
          send(res, 400, { error: "invalid directive body" });
          return;
        }

        const result = await pool.withTenant(
          tenant,
          async (borg) => {
            const allowedPeople = resolveKnownExternalEntityIds({
              borg,
              source: TEAM_AGENT_SENDER_EXTERNAL_ID_SOURCE,
              externalIds: parsed.data.allowed_external_ids,
              kind: "person",
            });
            const excludedPeople = resolveKnownExternalEntityIds({
              borg,
              source: TEAM_AGENT_SENDER_EXTERNAL_ID_SOURCE,
              externalIds: parsed.data.excluded_external_ids,
              kind: "person",
            });
            const allowedGroups = resolveKnownExternalEntityIds({
              borg,
              source: TEAM_AGENT_CONVERSATION_EXTERNAL_ID_SOURCE,
              externalIds: parsed.data.allowed_group_external_ids,
              kind: "group",
            });
            const excludedGroups = resolveKnownExternalEntityIds({
              borg,
              source: TEAM_AGENT_CONVERSATION_EXTERNAL_ID_SOURCE,
              externalIds: parsed.data.excluded_group_external_ids,
              kind: "group",
            });

            if (
              allowedPeople === null ||
              excludedPeople === null ||
              allowedGroups === null ||
              excludedGroups === null
            ) {
              return { status: "unknown_external_id" as const };
            }

            const subjectEntityIds =
              parsed.data.subject_external_id === undefined
                ? []
                : resolveKnownExternalEntityIds({
                    borg,
                    source: TEAM_AGENT_SENDER_EXTERNAL_ID_SOURCE,
                    externalIds: [parsed.data.subject_external_id],
                    kind: "person",
                  });

            if (subjectEntityIds === null) {
              return { status: "unknown_external_id" as const };
            }

            const allowedEntityIds = dedupePreservingOrder([...allowedPeople, ...allowedGroups]);
            const excludedEntityIds = dedupePreservingOrder([...excludedPeople, ...excludedGroups]);
            const excludedEntityIdSet = new Set(excludedEntityIds);

            if (allowedEntityIds.some((entityId) => excludedEntityIdSet.has(entityId))) {
              return { status: "ambiguous_external_id" as const };
            }

            const admin = ensureDirectiveAdminIdentity(borg);
            const queueInput = buildCreatorDirectiveQueueInput({
              body: parsed.data,
              adminEntityId: admin.entityId,
              adminSessionId: admin.sessionId,
              sourceStreamEntryId: createStreamEntryId(),
              allowedEntityIds,
              excludedEntityIds,
              subjectEntityId: subjectEntityIds[0] ?? null,
            });

            if (!creatorDirectiveQueueInputSchema.safeParse(queueInput).success) {
              return { status: "invalid_policy" as const };
            }

            const sourceEntry = await appendDirectiveAdminEvent({
              borg,
              admin,
              content: {
                event: "memory_sidecar.operator_directive_queue_requested",
                kind: parsed.data.kind,
                text: parsed.data.text,
                content_scope: parsed.data.content_scope,
              },
            });
            const persistedQueueInput = buildCreatorDirectiveQueueInput({
              body: parsed.data,
              adminEntityId: admin.entityId,
              adminSessionId: admin.sessionId,
              sourceStreamEntryId: sourceEntry.id,
              allowedEntityIds,
              excludedEntityIds,
              subjectEntityId: subjectEntityIds[0] ?? null,
            });
            let directive: CreatorDirective;

            try {
              directive = borg.creatorDirectives.queue(persistedQueueInput);
            } catch (error) {
              try {
                const code = errorCode(error);
                await appendDirectiveAdminEvent({
                  borg,
                  admin,
                  content: {
                    event: "memory_sidecar.operator_directive_queue_failed",
                    kind: parsed.data.kind,
                    content_scope: parsed.data.content_scope,
                    provenance_stream_entry_id: sourceEntry.id,
                    failure_code: typeof code === "string" ? code : "UNKNOWN",
                  },
                });
              } catch (auditError) {
                console.error(
                  `memory-sidecar: failed to record directive queue failure for tenant "${tenant}"`,
                  auditError,
                );
              }
              throw error;
            }

            return { status: "created" as const, directive };
          },
          { exclusive: true },
        );

        if (result.status === "unknown_external_id") {
          send(res, 400, { error: "unknown directive external id" });
          return;
        }
        if (result.status === "ambiguous_external_id") {
          send(res, 400, { error: "ambiguous directive external ids" });
          return;
        }
        if (result.status === "invalid_policy") {
          send(res, 400, { error: "invalid directive policy" });
          return;
        }

        send(res, 201, {
          ok: true,
          directive: projectCreatorDirectiveForAdmin(result.directive),
        });
        return;
      }

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
        const userProvided = body.user !== undefined;
        const assistantProvided = body.assistant !== undefined;
        if (!userProvided && !assistantProvided) {
          send(res, 400, { error: "missing 'user' or 'assistant'" });
          return;
        }
        const user = asContentString(body.user);
        if (userProvided && user.trim() === "") {
          send(res, 400, { error: "missing 'user'" });
          return;
        }
        const assistant = asContentString(body.assistant);
        if (assistantProvided && assistant.trim() === "") {
          send(res, 400, { error: "missing 'assistant'" });
          return;
        }
        const observation = userProvided && !assistantProvided;
        const replyOnly = !userProvided && assistantProvided;
        const parsedObservedAt =
          !observation || body.observed_at === undefined
            ? { success: true as const, data: undefined }
            : epochMillisecondsSchema.safeParse(body.observed_at);
        if (!parsedObservedAt.success) {
          send(res, 400, { error: "invalid 'observed_at'; expected epoch milliseconds" });
          return;
        }
        if (parsedObservedAt.data !== undefined) {
          const serverNow = Date.now();

          if (
            parsedObservedAt.data < serverNow - OBSERVATION_MAX_PAST_AGE_MS ||
            parsedObservedAt.data > serverNow + OBSERVATION_MAX_FUTURE_SKEW_MS
          ) {
            send(res, 400, {
              error: "invalid 'observed_at'; outside the accepted server-time window",
            });
            return;
          }
        }
        const parsedSender = parseAppendTurnSender(body.sender);
        if (!parsedSender.valid) {
          send(res, 400, {
            error: "invalid 'sender'; expected non-empty 'external_id' and 'display_name'",
          });
          return;
        }
        let conversation: SidecarConversation | undefined;
        if (body.conversation !== undefined) {
          const parsedConversation = sidecarConversationSchema.safeParse(body.conversation);

          if (!parsedConversation.success) {
            send(res, 400, {
              error:
                "invalid 'conversation'; expected type 'personal', 'groupChat', or 'channel' and string 'name'",
            });
            return;
          }

          conversation = parsedConversation.data;
        }

        const session = sessionFromCaller(sessionRaw);
        const conversationIdentityAvailable =
          conversation !== undefined &&
          (conversation.type === "personal" || conversation.external_id !== undefined);
        const enhancedIdentityAvailable =
          conversationIdentityAvailable &&
          (parsedSender.sender !== null ||
            (replyOnly && conversation !== undefined && conversation.type !== "personal"));
        let enhancedSender: EnhancedAppendTurnSender | null = null;

        if (enhancedIdentityAvailable && parsedSender.sender !== null) {
          if (
            parsedSender.sender.operator !== undefined &&
            typeof parsedSender.sender.operator !== "boolean"
          ) {
            send(res, 400, { error: "invalid 'sender.operator'; expected boolean" });
            return;
          }

          enhancedSender = {
            externalId: parsedSender.sender.externalId,
            displayName: parsedSender.sender.displayName,
            operator: parsedSender.sender.operator === true,
          };
        }
        const entries = await pool.withTenant(
          tenant,
          async (borg) => {
            if (enhancedIdentityAvailable && conversation !== undefined) {
              const identity: TeamAgentIdentity = resolveTeamAgentIdentity({
                borg,
                session,
                rawSession: sessionRaw,
                sender: enhancedSender,
                conversation,
              });
              const userEntryInput =
                userProvided && identity.senderEntityId !== null
                  ? {
                      kind: "user_msg" as const,
                      content: user,
                      audience: identity.audienceEntity.id,
                      sender_entity_id: identity.senderEntityId,
                      conversation: identity.conversation,
                      ...(observation && parsedObservedAt.data !== undefined
                        ? { observed_at: parsedObservedAt.data }
                        : {}),
                    }
                  : undefined;
              const assistantEntryInput = assistantProvided
                ? {
                    kind: "agent_msg" as const,
                    content: assistant,
                    audience: identity.audienceEntity.id,
                    conversation: identity.conversation,
                  }
                : undefined;
              let enrichedEntries: StreamEntry[];

              if (observation) {
                if (userEntryInput === undefined) {
                  throw new Error("enhanced observation did not produce a user entry input");
                }
                enrichedEntries = [await borg.stream.append(userEntryInput, { session })];
              } else if (replyOnly) {
                if (assistantEntryInput === undefined) {
                  throw new Error(
                    "enhanced reply-only append did not produce an agent entry input",
                  );
                }
                enrichedEntries = [await borg.stream.append(assistantEntryInput, { session })];
              } else {
                if (userEntryInput === undefined || assistantEntryInput === undefined) {
                  throw new Error("enhanced completed turn did not produce both entry inputs");
                }
                enrichedEntries = await borg.stream.appendMany(
                  [userEntryInput, assistantEntryInput],
                  { session },
                );
              }

              const userEntry = replyOnly ? undefined : enrichedEntries[0];
              const assistantEntry = observation ? undefined : enrichedEntries[replyOnly ? 0 : 1];

              if (
                (userProvided && userEntry === undefined) ||
                (assistantProvided && assistantEntry === undefined)
              ) {
                throw new Error("enhanced append did not produce the requested entries");
              }

              try {
                const firstEntry = enrichedEntries[0];

                if (firstEntry === undefined) {
                  throw new Error("enhanced append did not produce a projection source entry");
                }

                const sessionProjection = {
                  ...identity.sessionEnsureInput,
                  created_at: firstEntry.timestamp,
                  last_activity_at: firstEntry.timestamp,
                };

                if (observation) {
                  if (userEntry === undefined || identity.senderEntityId === null) {
                    throw new Error("enhanced observation requires a sender entry");
                  }

                  borg.activity.projectObservedTurn({
                    session: sessionProjection,
                    userContact: {
                      kind: "user_contact",
                      occurredAt: userEntry.timestamp,
                      sessionId: userEntry.session_id,
                      speakerEntityId: identity.senderEntityId,
                      actorEntityId: identity.senderEntityId,
                      audienceEntityId: identity.audienceEntity.id,
                      participantEntityIds: dedupePreservingOrder([
                        identity.senderEntityId,
                        identity.audienceEntity.id,
                      ]),
                      sourceStreamEntryIds: [userEntry.id],
                    },
                    touch: {
                      at: userEntry.timestamp,
                      messageCountDelta: 1,
                    },
                  });
                } else {
                  if (assistantEntry === undefined) {
                    throw new Error("enhanced append did not produce an assistant entry");
                  }

                  const selfEntity = borg.entities.getSelf();

                  if (selfEntity === null) {
                    throw new Error("enhanced append awareness projection requires a self entity");
                  }

                  const borgReplied = {
                    kind: "borg_replied" as const,
                    occurredAt: assistantEntry.timestamp,
                    sessionId: assistantEntry.session_id,
                    speakerEntityId: selfEntity.id,
                    actorEntityId: selfEntity.id,
                    audienceEntityId: identity.audienceEntity.id,
                    participantEntityIds: dedupePreservingOrder([
                      selfEntity.id,
                      ...(identity.senderEntityId === null ? [] : [identity.senderEntityId]),
                      identity.audienceEntity.id,
                    ]),
                    sourceStreamEntryIds: [assistantEntry.id],
                  };
                  const touch = {
                    at: assistantEntry.timestamp,
                    messageCountDelta: 1,
                  };

                  if (replyOnly) {
                    borg.activity.projectRepliedTurn({
                      session: sessionProjection,
                      borgReplied,
                      touch,
                    });
                  } else {
                    if (userEntry === undefined || identity.senderEntityId === null) {
                      throw new Error("enhanced completed turn requires a sender entry");
                    }

                    borg.activity.projectCompletedTurn({
                      session: sessionProjection,
                      userContact: {
                        kind: "user_contact",
                        occurredAt: userEntry.timestamp,
                        sessionId: userEntry.session_id,
                        speakerEntityId: identity.senderEntityId,
                        actorEntityId: identity.senderEntityId,
                        audienceEntityId: identity.audienceEntity.id,
                        participantEntityIds: dedupePreservingOrder([
                          identity.senderEntityId,
                          identity.audienceEntity.id,
                        ]),
                        sourceStreamEntryIds: [userEntry.id],
                      },
                      borgReplied,
                      touch,
                    });
                  }
                }
              } catch (error) {
                const projectionEntries = enrichedEntries.map((entry) => entry.id);
                const lastProjectionEntryId = projectionEntries.at(-1);

                if (lastProjectionEntryId === undefined) {
                  throw error;
                }

                console.error(
                  `memory-sidecar: append-turn awareness projection failed for tenant "${tenant}"`,
                  error,
                );
                const projectionErrorCode = errorCode(error);
                traceRegistry?.tracerFor(tenant).emit("sidecar.append_projection.degraded", {
                  turnId: `sidecar_append:${lastProjectionEntryId}`,
                  session_id: session,
                  reason: "awareness_projection_failed",
                  error_code:
                    typeof projectionErrorCode === "string" ? projectionErrorCode : undefined,
                  source_stream_entry_ids: projectionEntries,
                });
              }

              return enrichedEntries;
            }

            const senderEntityId =
              parsedSender.sender === null
                ? undefined
                : borg.entities.resolveExternal({
                    source: TEAM_AGENT_SENDER_EXTERNAL_ID_SOURCE,
                    externalId: parsedSender.sender.externalId,
                    canonicalName: parsedSender.sender.displayName,
                    kind: "person",
                    provenance: "transport_sender",
                  });
            const persistedConversation =
              conversation === undefined
                ? undefined
                : { type: conversation.type, name: conversation.name };
            const inputs: StreamEntryInput[] = [];

            if (userProvided) {
              inputs.push({
                kind: "user_msg",
                content: user,
                ...(observation && parsedObservedAt.data !== undefined
                  ? { observed_at: parsedObservedAt.data }
                  : {}),
                ...(senderEntityId === undefined ? {} : { sender_entity_id: senderEntityId }),
                ...(persistedConversation === undefined
                  ? {}
                  : { conversation: persistedConversation }),
              });
            }

            if (assistantProvided) {
              inputs.push({
                kind: "agent_msg",
                content: assistant,
                ...(persistedConversation === undefined
                  ? {}
                  : { conversation: persistedConversation }),
              });
            }

            if (inputs.length === 1) {
              const singleEntry = inputs[0];

              if (singleEntry === undefined) {
                throw new Error("single-entry append did not produce an input");
              }

              return [await borg.stream.append(singleEntry, { session })];
            }

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
      const parsedRecall = memoryRecallBodySchema.safeParse(body);
      if (!parsedRecall.success) {
        send(res, 400, { error: "invalid memory recall body" });
        return;
      }
      const query = parsedRecall.data.query;
      const rawLimit = parsedRecall.data.limit ?? 10;
      const limit = Math.max(1, Math.min(maxRecallLimit, Math.floor(rawLimit)));
      const deferRetrievalAccounting = parsedRecall.data.exclude !== undefined;
      const searchLimit = deferRetrievalAccounting
        ? Math.min(
            maxRecallLimit * EPISODE_OVERFETCH_MULTIPLIER,
            limit * EPISODE_OVERFETCH_MULTIPLIER,
          )
        : limit;
      const traceTurnId = traceRegistry === undefined ? undefined : nextRecallTraceTurnId(tenant);
      const degradations: RetrievalDegradation[] = [];
      let hits;
      let episodesTimeRangeFallback = false;
      try {
        hits = await raceRecallDeadline(
          pool.withTenant(tenant, async (borg) => {
            const memoryOwner = borg.entities.getSelf();
            const recallOptions = {
              limit: searchLimit,
              onDegraded: (degradation: RetrievalDegradation) => degradations.push(degradation),
              semanticVariantCount: recallSemanticVariantCount,
              recallQueryPlannerContext: {
                identity: {
                  ...(memoryOwner === null ? {} : { memoryOwnerName: memoryOwner.canonical_name }),
                },
              },
              ...(recencyPrior === undefined ? {} : { recencyPrior }),
              ...(deferRetrievalAccounting ? { recordRetrieval: false } : {}),
              ...(traceTurnId === undefined ? {} : { traceTurnId }),
            };
            const recalledResult =
              parsedRecall.data.time_range === undefined
                ? {
                    episodes: await borg.episodic.search(query, recallOptions),
                    timeRangeFallback: false,
                  }
                : await borg.episodic.searchWithTimeRangeFallback(query, {
                    ...recallOptions,
                    timeRange: parsedRecall.data.time_range,
                  });
            const included = recalledResult.episodes
              .filter((hit) => !episodeMatchesExclusions(hit.episode, parsedRecall.data.exclude))
              .slice(0, limit);
            const topRawScore =
              included.length === 0 ? null : Math.max(...included.map((hit) => hit.rawScore));
            const shouldAbstain =
              recallAbstainThreshold > 0 &&
              (topRawScore === null || topRawScore < recallAbstainThreshold);

            if (deferRetrievalAccounting && !shouldAbstain) {
              for (const hit of included) {
                borg.episodic.recordRetrieval(hit.episode.id, hit.score);
              }
            }

            return {
              episodes: included,
              projected: projectEpisodeHitsForResponse(included, borg.entities, false),
              topRawScore,
              timeRangeFallback: recalledResult.timeRangeFallback,
            };
          }),
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
      episodesTimeRangeFallback = hits.timeRangeFallback;
      const topRawScore = hits.topRawScore;
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
          ...(episodesTimeRangeFallback ? { episodes_time_range_fallback: true } : {}),
          ...degraded,
        });
        return;
      }

      send(res, 200, {
        ok: true,
        top_raw_score: topRawScore,
        ...(episodesTimeRangeFallback ? { episodes_time_range_fallback: true } : {}),
        ...degraded,
        episodes: hits.projected,
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
