import { similarityThresholds } from "../config/similarity.js";
import { rememberTenantFactInputSchema } from "../memory/episodic/types.js";
import {
  tenantFactAuthorizationFromEntry,
  tenantFactSharingGuidance,
} from "../memory/episodic/remember.js";
import { EmbeddingBankError } from "../embeddings/bank-profile.js";
// HTTP request handler for the borg memory sidecar: a thin, tenant-routed wrapper
// over BorgPool that exposes long-term memory to an external (e.g. Python) service.
//
//   POST /memory/remember    { tenant, content, author? }          -> append + extract episode(s)
//   POST /memory/forget      { tenant, id }                       -> archive episode or semantic node
//   POST /memory/enqueue     { tenant, session, conversation, sender, text, ... } -> durable inbox
//   POST /memory/await-response { tenant, sidecar_session_id, entry_id, timeout_ms? } -> long poll
//   POST /memory/inbox-progress { tenant, sidecar_session_id, entry_ids, phase } -> interim status
//   POST /memory/append-turn { tenant, session, sender, conversation, user?, assistant?, observed_at? }
//        structured identity includes sender.operator and conversation.external_id;
//        absent assistant records an observation; absent user records a reply-only turn
//   POST /memory/context { tenant, session, sender, conversation, focus, context_turns,
//                          limit?, sections?,
//                          participants?, entity_terms?, time_range?, exclude?, venue_since?,
//                          venue_limit? }
//                                                            -> labeled episodic turn context
//   POST /memory/guard-reply { tenant, session, sender, conversation, context_id, response,
//                              current_turn_user_texts? }     -> identifier-only reply check
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
//   GET  /memory/episodes/{id}/why?tenant=<id>              -> correction provenance and citations
//   GET  /memory/trace?tenant=<id>&since=<ts>                -> inspect recall trace buffer
//   POST /memory/maintenance?tenant=<id|*>&mode=<light|heavy>&dryRun=<0|1>
//        tenant is optional; absent or "*" fans out across every tenant with a
//        bank on disk and answers {runs:[{tenant,run_id}],skipped:[...]}.
//   GET  /memory/maintenance/status?tenant=<id>
//   GET  /memory/maintenance/audit?tenant=<id>&run_id=<id>
//   POST /memory/maintenance/revert?tenant=<id>&audit_id=<id>
//   GET  /healthz                                           -> liveness (no auth)
//
// Cognition recall remains global within each tenant being.
// /memory/context supplies global, labeled episodes and recent activity for external
// cognition; venue recency stays scoped to the current venue. All authenticated routes
// require x-borg-token.

import { timingSafeEqual } from "node:crypto";
import type { IncomingMessage, ServerResponse } from "node:http";

import { z } from "zod";

import type { Borg } from "../borg.js";
import { normalizeCommitmentClassification } from "../cognition/commitments/classification-normalizer.js";
import {
  recentLivedExperienceDisclosureLabel,
  type ActivityProjectionSourceEvent,
} from "../memory/activity/index.js";
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
  memoryDisclosureLabelMetadata,
  type MemoryDisclosureLabel,
} from "../memory/common/index.js";
import {
  episodeIdSchema,
  isEpisodeAccessVisibleToAnyAudience,
  parseEpisodeParticipantEntityIdTerm,
  type Episode,
} from "../memory/episodic/index.js";
import { semanticNodeIdSchema } from "../memory/semantic/index.js";
import {
  clipRecalledEvidenceText,
  MAX_RECALLED_SOURCE_MESSAGES_PER_EPISODE,
} from "../retrieval/evidence-bounds.js";
import { MEMORY_DISCLOSURE_GUIDANCE_FOR_MODEL, SELF_RECALL_SCOPE } from "../retrieval/index.js";
import { ServedMemoryContextRegistry } from "./served-memory-context.js";
import type {
  DisclosureContext,
  EpisodeRecencyPrior,
  RetrievedEpisode,
} from "../retrieval/index.js";
import type { RecallPlanOutcome, RetrievalDegradation } from "../retrieval/pipeline.js";
import {
  MAX_RECALL_EXPANSION_SEMANTIC_VARIANTS,
  MAX_RECALL_QUERY_ACTIVITY_ROWS,
  MAX_RECALL_QUERY_CONTEXT_TURN_CHARS,
  MAX_RECALL_QUERY_ENTITY_TERMS,
  MAX_RECALL_QUERY_HANDLE_CHARS,
  MIN_RECALL_EXPANSION_SEMANTIC_VARIANTS,
} from "../retrieval/recall-expansion.js";
import { isNarrativeStreamEntry, type StreamEntry } from "../stream/index.js";
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
} from "./team-agent-identity.js";
import { MAX_INBOX_REPLY_ACTIVITY_RECONCILE_LIMIT } from "../cognition/ingestion/index.js";
import { taskEventSchema } from "../stream/types.js";
import { agentDeliveryAckSchema } from "../cognition/ingestion/agent-deliveries.js";
import type { DeliveryWaiterRegistry } from "./delivery-waiter-registry.js";

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
  evict?(tenantId: string): Promise<void>;
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
  // One /memory/context or /memory/recall budget, including identity, activity
  // ranking and recall. Response headroom is reserved inside it. Keep this below
  // the caller's timeout and allow for the query provider's own stall guards;
  // exhausting it returns explicit degradation. 0 disables the ceiling.
  recallDeadlineMs?: number;
  recentActivityWindowMs?: number;
  recentActivityLimit?: number;
  recentActivityCandidateLimit?: number;
  recentActivityRankingBudgetMs?: number;
  activityExcerptHydrationBudgetMs?: number;
  recencyPrior?: EpisodeRecencyPrior;
  recallSemanticVariantCount?: number;
  traceRegistry?: MemoryTraceRegistry;
  servedContexts?: ServedMemoryContextRegistry;
  maintenanceCoordinator?: Pick<
    MemoryMaintenanceCoordinator,
    "cancelReservation" | "getStatus" | "hasReservation" | "startReserved" | "tryReserve"
  >;
  inboxWaiters?: ResponseWaiterRegistry;
  deliveryWaiters?: DeliveryWaiterRegistry;
};

type RequestHandler = (req: IncomingMessage, res: ServerResponse) => void;

const DEFAULT_MAX_BODY_BYTES = 64 * 1024;
const DEFAULT_MAX_RECALL_LIMIT = 50;
const DEFAULT_RECALL_DEADLINE_MS = 5000;
export const DEFAULT_RECENT_ACTIVITY_WINDOW_MS = 7 * 24 * 60 * 60_000;
export const DEFAULT_RECENT_ACTIVITY_LIMIT = 12;
export const DEFAULT_RECENT_ACTIVITY_CANDIDATE_LIMIT = 96;
export const DEFAULT_RECENT_ACTIVITY_RANKING_BUDGET_MS = 1_000;
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
// The second pass gets only the remaining request budget, with response headroom
// already reserved. Below the floor it is skipped rather than started.
const MIN_AUTOBIOGRAPHICAL_BUDGET_MS = 500;
const MAX_AUTOBIOGRAPHICAL_BUDGET_MS = 1500;
const MEMORY_RESPONSE_HEADROOM_MS = 700;
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
async function raceRecallDeadline<T>(search: () => Promise<T>, deadlineAt: number): Promise<T> {
  const deadlineMs = deadlineAt - Date.now();
  if (deadlineMs <= 0) throw new RecallDeadlineExceeded(0);
  if (!Number.isFinite(deadlineAt)) return search();
  const pending = search();

  let timer: ReturnType<typeof setTimeout> | undefined;
  pending.catch(() => undefined);

  try {
    return await Promise.race([
      pending,
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

const contextConversationSchema = sidecarConversationSchema
  .extend({ external_id: z.string().trim().min(1) })
  .strict();

const contextSenderSchema = z
  .object({
    external_id: z.string().trim().min(1),
    display_name: z.string().trim().min(1),
    operator: z.boolean(),
  })
  .strict();

const memoryTransportIdentitySchema = z.object({
  session: z.string().trim().min(1),
  sender: contextSenderSchema.strip(),
  conversation: contextConversationSchema.strip(),
});

const memoryTenantRememberBodySchema = memoryTransportIdentitySchema
  .extend({
    tenant: z.string().trim().regex(TENANT_ID_RE),
    scope: z.literal("tenant"),
    request_id: rememberTenantFactInputSchema.shape.requestId,
    content: rememberTenantFactInputSchema.shape.content,
    source_episode_ids: rememberTenantFactInputSchema.shape.sourceEpisodeIds,
    source_message_ids: rememberTenantFactInputSchema.shape.sourceMessageIds,
    authorization_message_ids: rememberTenantFactInputSchema.shape.authorizationMessageIds,
    sender: contextSenderSchema,
    conversation: contextConversationSchema,
  })
  .strict()
  .refine(
    (value) => value.authorization_message_ids.every((id) => value.source_message_ids.includes(id)),
    {
      path: ["authorization_message_ids"],
      message: "Authorization messages must occur in source_message_ids",
    },
  );

const guardReplyBodySchema = memoryTransportIdentitySchema
  .extend({
    tenant: z.string().trim().regex(TENANT_ID_RE),
    context_id: z.string().max(128),
    response: z.string().min(1),
    current_turn_user_texts: z.array(z.string()).max(32).optional(),
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
    participants: z
      .array(contextSenderSchema.extend({ operator: z.boolean().optional().default(false) }))
      .max(MAX_CONTEXT_PARTICIPANTS)
      .optional(),
    entity_terms: z
      .array(z.string().trim().min(1).max(MAX_CONTEXT_ENTITY_TERM_CHARS))
      .max(MAX_CONTEXT_ENTITY_TERMS)
      .optional(),
    focus: z.string({ error: "focus must be a string" }).trim().min(1).optional(),
    context_turns: z
      .array(contextTurnSchema, {
        error: "context_turns must be an array of structured turns",
      })
      .max(MAX_CONTEXT_TURNS)
      .optional(),
    limit: z.number().finite().optional(),
    sections: z.array(memoryContextSectionSchema).min(1).optional(),
    time_range: episodeTimeRangeSchema.optional(),
    exclude: episodeExclusionsSchema.optional(),
    venue_since: epochMillisecondsSchema.optional(),
    venue_limit: z.number().int().min(1).max(MAX_VENUE_RECENT_LIMIT).optional(),
  })
  .strict()
  .superRefine((value, ctx) => {
    const requestedSections = new Set(value.sections ?? DEFAULT_MEMORY_CONTEXT_SECTIONS);
    const episodesRequested = requestedSections.has("episodes");
    const venueRecentRequested = requestedSections.has("venue_recent");
    const autobiographicalRequested = requestedSections.has("autobiographical");

    if (episodesRequested || autobiographicalRequested) {
      if (value.focus === undefined) {
        ctx.addIssue({
          code: "custom",
          path: ["focus"],
          message: "focus is required when episodes or autobiographical are requested",
        });
      }
      if (value.context_turns === undefined) {
        ctx.addIssue({
          code: "custom",
          path: ["context_turns"],
          message: "context_turns is required when episodes or autobiographical are requested",
        });
      }
    }

    if (autobiographicalRequested && !episodesRequested) {
      ctx.addIssue({
        code: "custom",
        path: ["sections"],
        message: "autobiographical requires episodes (its period comes from the recall plan)",
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

const memoryAgentEventBodySchema = taskEventSchema
  .omit({ schema_version: true })
  .extend({
    tenant: z.string().trim().regex(TENANT_ID_RE),
    sidecar_session_id: sessionIdSchema,
  })
  .strict();

const memoryDeliveryClaimBodySchema = z
  .object({
    tenant: z.string().trim().regex(TENANT_ID_RE),
    sidecar_session_ids: z.array(sessionIdSchema).max(200),
    wait_ms: z.number().int().min(0).max(60_000).default(0),
    lease_ms: z.number().int().positive().default(120_000),
  })
  .strict();

const memoryDeliveryAckBodySchema = agentDeliveryAckSchema
  .extend({
    tenant: z.string().trim().regex(TENANT_ID_RE),
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

const forgetBodySchema = z
  .object({
    tenant: z.string().trim().regex(TENANT_ID_RE),
    id: z.union([episodeIdSchema, semanticNodeIdSchema]),
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
  Partial<Pick<RetrievedEpisode, "citationChain" | "disclosureLabel">>;

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

function createPublicDisclosureProjector(
  entities: Pick<Borg["entities"], "get">,
  context: Pick<DisclosureContext, "senderEntityId" | "currentAudienceEntityId">,
) {
  const names = new Map<EntityId, string | null>();
  return (label: MemoryDisclosureLabel) => {
    const disclosure = memoryDisclosureLabelMetadata(label);
    for (const entityId of [
      ...disclosure.origin_audience_entity_ids,
      ...disclosure.private_to_entity_ids,
    ]) {
      if (!names.has(entityId)) {
        names.set(entityId, entities.get(entityId)?.canonical_name ?? null);
      }
    }
    return {
      class: disclosure.disclosure_class,
      origin_audience_names: disclosure.origin_audience_entity_ids.flatMap(
        (entityId) => names.get(entityId) ?? [],
      ),
      private_to_names: disclosure.private_to_entity_ids.flatMap(
        (entityId) => names.get(entityId) ?? [],
      ),
      private_to_current_sender: disclosure.private_to_entity_ids.some(
        (entityId) => entityId === context.senderEntityId,
      ),
      private_to_current_audience: disclosure.private_to_entity_ids.some(
        (entityId) => entityId === context.currentAudienceEntityId,
      ),
    };
  };
}

function projectEpisodeHitsForResponse(
  hits: readonly SidecarEpisodeHit[],
  entities: Pick<Borg["entities"], "get" | "getSelf">,
  disclosureContext: Pick<DisclosureContext, "senderEntityId" | "currentAudienceEntityId"> | null,
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
  const projectDisclosure =
    disclosureContext === null
      ? null
      : createPublicDisclosureProjector(entities, disclosureContext);

  return hits.map((hit) => {
    const sharingAuthorizations = (hit.citationChain ?? [])
      .map(tenantFactAuthorizationFromEntry)
      .filter((authorization) => authorization !== null);
    const sharingFacts = [
      ...new Map(
        sharingAuthorizations.map((authorization) => [
          authorization.episode_id,
          {
            scope: authorization.scope,
            speaker_name: authorization.speaker_name,
            fact: authorization.fact,
            guidance: tenantFactSharingGuidance(authorization.speaker_name),
            disclosure: { class: "public", scope: "tenant" },
          },
        ]),
      ).values(),
    ];
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
      ...(sharingFacts.length === 0
        ? {}
        : {
            sharing_authorizations: sharingFacts,
            ...(sharingFacts.length === 1 ? { sharing_authorization: sharingFacts[0] } : {}),
          }),
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

    if (projectDisclosure === null) {
      return base;
    }

    return {
      ...base,
      disclosure: projectDisclosure(
        hit.disclosureLabel ?? memoryDisclosureLabelFromEpisodeAccess(hit.episode),
      ),
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
  event: ActivityProjectionSourceEvent,
  nowMs: number,
  sourceEntries: ReadonlyMap<StreamEntryId, StreamEntry>,
  projectDisclosure: ReturnType<typeof createPublicDisclosureProjector>,
) {
  const disclosure = projectDisclosure(
    recentLivedExperienceDisclosureLabel({
      originAudienceEntityIds: event.audienceEntityId === null ? [] : [event.audienceEntityId],
    }),
  );
  const participantName =
    event.kind === "borg_replied"
      ? (disclosure.origin_audience_names[0] ?? event.conversationName)
      : event.participantLabel;
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
      ? `${participantName} contacted the agent ${relativeAge} in ${location}.`
      : event.kind === "borg_replied"
        ? `The agent replied to ${participantName} ${relativeAge} in ${location}.`
        : `The agent completed a turn with ${participantName} ${relativeAge} in ${location}.`;
  const expectedSourceKind =
    event.kind === "user_contact" ? "user_msg" : event.kind === "borg_replied" ? "agent_msg" : null;
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
    participant_name: participantName,
    text,
    disclosure,
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

function sendEmbeddingBankUnavailable(res: ServerResponse, error: unknown): boolean {
  if (!(error instanceof EmbeddingBankError)) return false;
  send(res, 503, {
    error: "tenant unavailable",
    code: error.code,
    degraded: true,
    degraded_reason: error.message,
  });
  return true;
}

export function createMemoryHandler(options: MemoryHandlerOptions): RequestHandler {
  const { pool, token, traceRegistry, maintenanceCoordinator, inboxWaiters, deliveryWaiters } =
    options;
  const maxBodyBytes = options.maxBodyBytes ?? DEFAULT_MAX_BODY_BYTES;
  const maxRecallLimit = options.maxRecallLimit ?? DEFAULT_MAX_RECALL_LIMIT;
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
  const recentActivityCandidateLimit = Math.max(
    recentActivityLimit,
    Math.min(
      256,
      Math.floor(
        Number.isFinite(options.recentActivityCandidateLimit)
          ? options.recentActivityCandidateLimit!
          : DEFAULT_RECENT_ACTIVITY_CANDIDATE_LIMIT,
      ),
    ),
  );
  const recentActivityRankingBudgetMs = Math.max(
    1,
    Math.floor(
      Number.isFinite(options.recentActivityRankingBudgetMs)
        ? options.recentActivityRankingBudgetMs!
        : DEFAULT_RECENT_ACTIVITY_RANKING_BUDGET_MS,
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
  const servedContexts = options.servedContexts ?? new ServedMemoryContextRegistry();
  let recallTraceSequence = 0;

  const nextRecallTraceTurnId = (tenant: string): string => {
    recallTraceSequence += 1;
    return `sidecar_recall:${tenant}:${Date.now()}:${recallTraceSequence}`;
  };

  async function handle(req: IncomingMessage, res: ServerResponse): Promise<void> {
    // One budget starts before parsing, identity resolution, and activity ranking.
    // Leave response headroom inside it rather than adding time after recall.
    const requestDeadlineAt =
      recallDeadlineMs <= 0
        ? Infinity
        : Date.now() +
          recallDeadlineMs -
          Math.min(MEMORY_RESPONSE_HEADROOM_MS, recallDeadlineMs / 10);
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

    if (method === "POST" && rawPath === "/memory/admin/evict") {
      const tenant = requiredSingleQueryValue(res, searchParams, "tenant");
      if (tenant === null || !validateTenantForResponse(res, tenant)) return;
      if (pool.evict === undefined) {
        send(res, 503, { error: "tenant eviction unavailable" });
        return;
      }
      await pool.evict(tenant);
      send(res, 200, { ok: true, tenant, status: "closed" });
      return;
    }

    if (
      method === "POST" &&
      (rawPath === "/memory/agent-events" ||
        rawPath === "/memory/agent-deliveries/claim" ||
        rawPath === "/memory/agent-deliveries/ack")
    ) {
      if (inboxWaiters === undefined || deliveryWaiters === undefined) {
        send(res, 503, { error: "teams inbox unavailable" });
        return;
      }
      const body = await readJsonObjectBody(req, res, maxBodyBytes);
      if (body === null) return;
      if (rawPath === "/memory/agent-events") {
        const parsed = memoryAgentEventBodySchema.safeParse(body);
        if (!parsed.success) {
          send(res, 400, { error: "invalid agent event body" });
          return;
        }
        const { tenant, sidecar_session_id, ...event } = parsed.data;
        try {
          const result = await pool.withTenant(
            tenant,
            (borg) =>
              borg.inbox.enqueueTaskEvent({
                sessionId: sidecar_session_id,
                event: { schema_version: 1, ...event },
              }),
            { exclusive: true },
          );
          if (result === null) send(res, 404, { error: "Teams inbox session not found" });
          else send(res, 200, result);
        } catch (error) {
          if (sendEmbeddingBankUnavailable(res, error)) return;
          console.error(`memory-sidecar: ${rawPath} failed for tenant "${tenant}"`, error);
          send(res, 503, { error: "tenant unavailable" });
        }
        return;
      }
      if (rawPath === "/memory/agent-deliveries/ack") {
        const parsed = memoryDeliveryAckBodySchema.safeParse(body);
        if (!parsed.success) {
          send(res, 400, { error: "invalid agent delivery ack body" });
          return;
        }
        const { tenant, ...ack } = parsed.data;
        try {
          const status = await pool.withTenant(tenant, (borg) => borg.inbox.deliveries.ack(ack), {
            exclusive: true,
          });
          if (status === null) send(res, 404, { error: "delivery not found" });
          else send(res, 200, { status });
        } catch (error) {
          if (sendEmbeddingBankUnavailable(res, error)) return;
          console.error(`memory-sidecar: ${rawPath} failed for tenant "${tenant}"`, error);
          send(res, 503, { error: "tenant unavailable" });
        }
        return;
      }
      const parsed = memoryDeliveryClaimBodySchema.safeParse(body);
      if (!parsed.success) {
        send(res, 400, { error: "invalid agent delivery claim body" });
        return;
      }
      const input = parsed.data;
      const deadline = Date.now() + input.wait_ms;
      let cancelWaiter: (() => void) | undefined;
      let disconnected = req.aborted || res.destroyed;
      const disconnect = () => {
        disconnected = true;
        cancelWaiter?.();
      };
      req.once("aborted", disconnect);
      res.once("close", disconnect);
      const scan = () =>
        pool.withTenant(
          input.tenant,
          (borg) =>
            disconnected
              ? { deliveries: [], nextLeaseUntil: null }
              : borg.inbox.deliveries.claim({
                  sessionIds: input.sidecar_session_ids,
                  leaseMs: input.lease_ms,
                }),
          { exclusive: true },
        );
      try {
        while (!disconnected) {
          const first = await scan();
          if (disconnected) return;
          if (
            first.deliveries.length > 0 ||
            Date.now() >= deadline ||
            input.sidecar_session_ids.length === 0
          ) {
            send(res, 200, { deliveries: first.deliveries });
            return;
          }
          // Scan/register/scan closes the append race. Expired leases wake a poll
          // even if no new delivery is created during its wait.
          const waiter = deliveryWaiters.register({
            tenant: input.tenant,
            sessionIds: input.sidecar_session_ids,
            timeoutMs: Math.max(
              0,
              Math.min(deadline, first.nextLeaseUntil ?? deadline) - Date.now(),
            ),
          });
          cancelWaiter = waiter.cancel;
          if (disconnected) {
            waiter.cancel();
            return;
          }
          const second = await scan();
          if (disconnected) return;
          if (second.deliveries.length > 0) {
            send(res, 200, { deliveries: second.deliveries });
            return;
          }
          if (
            second.nextLeaseUntil !== null &&
            second.nextLeaseUntil < (first.nextLeaseUntil ?? deadline)
          ) {
            waiter.cancel();
            cancelWaiter = undefined;
            continue;
          }
          const wake = await waiter.promise;
          cancelWaiter = undefined;
          if (wake === "closed") {
            if (!disconnected) send(res, 200, { deliveries: [] });
            return;
          }
        }
      } catch (error) {
        if (sendEmbeddingBankUnavailable(res, error)) return;
        console.error(`memory-sidecar: ${rawPath} failed for tenant "${input.tenant}"`, error);
        if (!disconnected) send(res, 503, { error: "tenant unavailable" });
      } finally {
        cancelWaiter?.();
        req.off("aborted", disconnect);
        res.off("close", disconnect);
      }
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
        if (sendEmbeddingBankUnavailable(res, error)) return;
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
          if (sendEmbeddingBankUnavailable(res, error)) return;
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
          if (sendEmbeddingBankUnavailable(res, error)) return;
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
        if (sendEmbeddingBankUnavailable(res, error)) return;
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
        if (sendEmbeddingBankUnavailable(res, error)) return;
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
        if (sendEmbeddingBankUnavailable(res, error)) return;
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
        if (sendEmbeddingBankUnavailable(res, error)) return;
        console.error(`memory-sidecar: ${rawPath} failed for tenant "${tenant}"`, error);
        send(res, 500, { error: "internal error" });
      }
      return;
    }

    if (method === "GET") {
      const whyEpisodeId = rawPath.endsWith("/why")
        ? parseEpisodeIdFromPath(rawPath.slice(0, -"/why".length))
        : undefined;
      const isEpisodeWhyPath = whyEpisodeId !== undefined;
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
      const episodeId = nonEpisodePath
        ? undefined
        : isEpisodeWhyPath
          ? whyEpisodeId
          : parseEpisodeIdFromPath(rawPath);
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
            const active = listed.items.filter(
              (episode) => borg.episodic.getStats(episode.id)?.archived !== true,
            );
            if (active.length === 0) {
              return { ...listed, items: active };
            }

            const projectMetadata = createPublicEpisodeMetadataProjector(active, borg.entities);

            return {
              ...listed,
              items: active.map((episode) =>
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
          if (isEpisodeWhyPath) {
            const why = await pool.withTenant(tenant, (borg) => borg.correction.why(episodeId));
            send(res, 200, { ok: true, ...why });
            return;
          }
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
        if (sendEmbeddingBankUnavailable(res, error)) return;
        if (isEpisodeWhyPath && errorCode(error) === "EPISODE_NOT_FOUND") {
          send(res, 404, { error: "episode not found" });
          return;
        }
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
        rawPath !== "/memory/forget" &&
        rawPath !== "/memory/recall" &&
        rawPath !== "/memory/append-turn" &&
        rawPath !== "/memory/commitments" &&
        rawPath !== "/memory/context" &&
        rawPath !== "/memory/guard-reply" &&
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
      if (rawPath === "/memory/forget") {
        const parsed = forgetBodySchema.safeParse(body);
        if (!parsed.success) {
          send(res, 400, { error: "invalid forget body" });
          return;
        }
        const result = await pool.withTenant(
          tenant,
          (borg) => borg.correction.forget(parsed.data.id),
          { exclusive: true },
        );
        send(res, 200, {
          ok: true,
          id: result.id,
          target_type: result.target_type,
          archived: result.archived,
        });
        return;
      }
      if (rawPath === "/memory/guard-reply") {
        const parsed = guardReplyBodySchema.safeParse(body);
        if (!parsed.success) {
          send(res, 400, { error: "invalid guard reply body" });
          return;
        }
        const session = sessionFromCaller(parsed.data.session);
        const identity = await pool.withTenant(
          tenant,
          (borg) =>
            resolveTeamAgentIdentity({
              borg,
              session,
              rawSession: parsed.data.session,
              sender: {
                externalId: parsed.data.sender.external_id,
                displayName: parsed.data.sender.display_name,
                operator: parsed.data.sender.operator,
              },
              conversation: parsed.data.conversation,
            }),
          { exclusive: true },
        );
        const result = await pool.withTenant(tenant, async (borg) => {
          const candidate = servedContexts.get(tenant, session, parsed.data.context_id);
          const snapshot =
            candidate?.audienceEntityId === identity.audienceEntity.id &&
            candidate?.senderEntityId === identity.senderEntityId
              ? candidate
              : undefined;
          const reasons: string[] = snapshot === undefined ? ["context_snapshot_miss"] : [];
          const turnId = parsed.data.context_id || nextRecallTraceTurnId(tenant);
          const emission = await borg.guardReply({
            turnId,
            sessionId: session,
            sessionSourceType: identity.sessionEnsureInput.source_type,
            sessionAudienceRole: identity.audienceRole,
            audienceEntityId: identity.audienceEntity.id,
            response: parsed.data.response,
            currentTurnUserTexts: parsed.data.current_turn_user_texts,
            retrievedEpisodes: snapshot?.episodes ?? [],
            activeCommitments:
              snapshot?.commitments ??
              borg.commitments.list({
                activeOnly: true,
                audienceEntityId: identity.audienceEntity.id,
              }),
            knownInternalIdentifiers: [parsed.data.context_id, identity.senderEntityId ?? ""],
          });
          const verdict = emission.kind === "suppressed" ? "blocked" : "pass";
          if (emission.kind === "suppressed") reasons.push(emission.reason);
          traceRegistry?.tracerFor(tenant).emit("sidecar.guard_reply.completed", {
            turnId,
            session_id: session,
            verdict,
            reasons,
          });
          return { ok: true, verdict, reasons };
        });
        send(res, 200, result);
        return;
      }

      if (rawPath === "/memory/context") {
        const parsed = memoryContextBodySchema.safeParse(body);

        if (!parsed.success) {
          send(res, 400, {
            error: `invalid memory context body: ${parsed.error.issues
              .flatMap((issue) =>
                issue.code === "unrecognized_keys"
                  ? issue.keys.map((key) => `${[...issue.path, key].join(".")}: unrecognized field`)
                  : [`${issue.path.join(".") || "body"}: ${issue.message}`],
              )
              .join("; ")}`,
          });
          return;
        }

        const requestedSections = new Set(parsed.data.sections ?? DEFAULT_MEMORY_CONTEXT_SECTIONS);
        const episodeLimit = Math.max(
          1,
          Math.min(
            maxRecallLimit,
            Math.floor(parsed.data.limit === undefined ? 8 : parsed.data.limit),
          ),
        );
        // Planner cues can promote candidates below the response limit. Account
        // only for the episodes returned after cue ordering and exclusions.
        const episodeSearchLimit = Math.min(
          maxRecallLimit * EPISODE_OVERFETCH_MULTIPLIER,
          episodeLimit * EPISODE_OVERFETCH_MULTIPLIER,
        );
        const venueLimit = parsed.data.venue_limit ?? DEFAULT_VENUE_RECENT_LIMIT;
        const venueSearchLimit =
          parsed.data.exclude === undefined
            ? venueLimit
            : Math.min(
                MAX_VENUE_RECENT_LIMIT * EPISODE_OVERFETCH_MULTIPLIER,
                venueLimit * EPISODE_OVERFETCH_MULTIPLIER,
              );
        const session = sessionFromCaller(parsed.data.session);
        const identity = await raceRecallDeadline(
          () =>
            pool.withTenant(
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
            ),
          requestDeadlineAt,
        );
        const nowMs = Date.now();
        const context = await raceRecallDeadline(
          () =>
            pool.withTenant(tenant, async (borg) => {
              const observedGroupAudienceEntityIds =
                parsed.data.conversation.type === "personal" &&
                requestedSections.has("autobiographical")
                  ? borg.activity.listObservedGroupAudienceEntityIdsForSpeaker(
                      identity.senderEntityId,
                    )
                  : [];
              const visibleAudienceEntityIds = dedupePreservingOrder([
                identity.audienceEntity.id,
                ...observedGroupAudienceEntityIds,
              ]);
              const activityCandidateLimit =
                parsed.data.focus === undefined
                  ? recentActivityLimit
                  : recentActivityCandidateLimit;
              let recentActivityEvents = requestedSections.has("recent_activity")
                ? borg.activity.listRecentOtherActiveSessionEvents({
                    currentSessionId: session,
                    sinceMs: nowMs - recentActivityWindowMs,
                    limit: activityCandidateLimit,
                  })
                : [];
              // Planner context reads the memory owner's own recent replies elsewhere through an
              // owner-only pass of the same cognition query. Deriving them from the shared
              // response list starved the planner on busy group days: the 12 selected rows
              // were all user_contact messages, so zero owner rows reached the planner.
              let plannerOwnerActivityEvents = requestedSections.has("episodes")
                ? borg.activity.listRecentOtherActiveSessionEvents({
                    currentSessionId: session,
                    sinceMs: nowMs - recentActivityWindowMs,
                    limit: activityCandidateLimit,
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
              const projectDisclosure = createPublicDisclosureProjector(borg.entities, {
                senderEntityId: identity.senderEntityId,
                currentAudienceEntityId: identity.audienceEntity.id,
              });
              let activitySelection =
                parsed.data.focus === undefined ? "recency_without_focus" : "relevance";
              let activityDegradation = "";
              if (
                parsed.data.focus !== undefined &&
                recentActivityEvents.length + plannerOwnerActivityEvents.length > 0
              ) {
                const activityFocus = parsed.data.focus;
                const events = [...recentActivityEvents, ...plannerOwnerActivityEvents];
                try {
                  const keys = await raceRecallDeadline(
                    () =>
                      borg.activity.rankByRelevance({
                        focus: activityFocus,
                        nowMs,
                        candidates: events.map((event, index) => {
                          const row = projectRecentActivity(
                            event,
                            nowMs,
                            recentActivitySourceEntries,
                            projectDisclosure,
                          );
                          return {
                            key: String(index),
                            occurredAt: event.occurredAt,
                            // Stable text makes embedding-cache reuse possible across turns.
                            text: `${row.participant_name} | ${row.conversation.name}\n${row.excerpt ?? ""}`,
                          };
                        }),
                      }),
                    Math.min(requestDeadlineAt, Date.now() + recentActivityRankingBudgetMs),
                  );
                  const ranks = new Map(keys.map((key, index) => [Number(key), index]));
                  const recentCount = recentActivityEvents.length;
                  const ranked = events
                    .map((event, index) => ({ event, index }))
                    .sort(
                      (left, right) =>
                        (ranks.get(left.index) ?? Infinity) - (ranks.get(right.index) ?? Infinity),
                    );
                  recentActivityEvents = ranked
                    .filter(({ index }) => index < recentCount)
                    .slice(0, recentActivityLimit)
                    .map(({ event }) => event);
                  plannerOwnerActivityEvents = ranked
                    .filter(({ index }) => index >= recentCount)
                    .slice(0, recentActivityLimit)
                    .map(({ event }) => event);
                } catch (error) {
                  activitySelection = "recency_fallback";
                  activityDegradation = `recent_activity_relevance: ${error instanceof Error ? error.message : String(error)}`;
                  console.warn("memory-sidecar: activity relevance degraded", {
                    tenant,
                    reason: activityDegradation,
                  });
                }
              }
              recentActivityEvents = recentActivityEvents.slice(0, recentActivityLimit);
              plannerOwnerActivityEvents = plannerOwnerActivityEvents.slice(0, recentActivityLimit);
              const recentActivity = recentActivityEvents.map((event) =>
                projectRecentActivity(event, nowMs, recentActivitySourceEntries, projectDisclosure),
              );
              const plannerOwnerActivity = plannerOwnerActivityEvents.flatMap((event) => {
                const projected = projectRecentActivity(
                  event,
                  nowMs,
                  recentActivitySourceEntries,
                  projectDisclosure,
                );
                if (event.kind !== "borg_replied" || projected.excerpt === undefined) {
                  return [];
                }

                return [
                  {
                    excerpt: projected.excerpt,
                    occurredAt: event.occurredAt,
                    venue: projected.conversation,
                    counterpartyName: projected.participant_name,
                  },
                ];
              });
              const applicableCommitments =
                requestedSections.has("commitments") || requestedSections.has("episodes")
                  ? borg.commitments
                      .list({
                        activeOnly: true,
                        audienceEntityId: identity.audienceEntity.id,
                      })
                      .sort(compareCommitmentsForResponse)
                      .slice(0, MAX_COMMITMENT_RESPONSE_ITEMS)
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
                      (applicable) =>
                        applicable.activation.active && applicable.render_mode !== "omit",
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
                isEpisodeAccessVisibleToAnyAudience(candidate.episode, [
                  identity.audienceEntity.id,
                ]),
              );
              const venueRecent = projectEpisodeHitsForResponse(
                visibleVenueCandidates
                  .filter(
                    (candidate) =>
                      !episodeMatchesExclusions(candidate.episode, parsed.data.exclude),
                  )
                  .slice(0, venueLimit)
                  .map((candidate) => ({
                    episode: candidate.episode,
                    score: 0,
                    rawScore: 0,
                  })),
                borg.entities,
                {
                  senderEntityId: identity.senderEntityId,
                  currentAudienceEntityId: identity.audienceEntity.id,
                },
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
                activitySelection,
                activityDegradation,
                plannerOwnerActivity,
                plannerOwnerLivedExperience,
                memoryOwnerName: borg.entities.getSelf()?.canonical_name,
                commitments: applicableCommitments.map(projectCommitment),
                applicableCommitments,
                directives,
                venueRecent,
              };
            }),
          requestDeadlineAt,
        );
        const degradations: RetrievalDegradation[] = [];
        let episodes: Array<Record<string, unknown>> = [];
        let servedEpisodes: readonly RetrievedEpisode[] = [];
        let degraded = false;
        let degradedReason = "";
        let abstained = false;
        let plannerTemporalCue: TemporalCue | null = null;

        if (requestedSections.has("episodes")) {
          // The schema requires both fields for sections that use recall planning.
          const recallFocus = parsed.data.focus!;
          const contextTurns = parsed.data.context_turns!;
          const traceTurnId =
            traceRegistry === undefined ? undefined : nextRecallTraceTurnId(tenant);

          try {
            const recallResult = await raceRecallDeadline(
              () =>
                pool.withTenant(tenant, async (borg) => {
                  const recallAbstainThreshold =
                    options.recallAbstainThreshold ??
                    borg.similarityThresholds?.recallAbstain ??
                    similarityThresholds().recallAbstain;
                  let recallPlan: RecallPlanOutcome | null = null;
                  const recallOptions = {
                    limit: episodeSearchLimit,
                    recallContext: {
                      reader: SELF_RECALL_SCOPE,
                      currentSessionId: session,
                      currentAudienceEntityId: identity.audienceEntity.id,
                      currentParticipantEntityIds: dedupePreservingOrder([
                        identity.senderEntityId,
                        ...identity.participantEntityIds,
                        identity.audienceEntity.id,
                      ]),
                    },
                    disclosureContext: {
                      currentSessionId: session,
                      currentAudienceEntityId: identity.audienceEntity.id,
                      audienceRole: identity.audienceRole,
                      senderEntityId: identity.senderEntityId,
                      senderRole: null,
                      participantEntityIds: dedupePreservingOrder([
                        identity.senderEntityId,
                        ...identity.participantEntityIds,
                        identity.audienceEntity.id,
                      ]),
                      isPrivateSelfCognition: false,
                    },
                    onDegraded: (degradation: RetrievalDegradation) =>
                      degradations.push(degradation),
                    onRecallPlan: (plan: RecallPlanOutcome) => {
                      recallPlan = plan;
                    },
                    // Caller hints may include transport names or lexical guesses. The shared
                    // planner selects their focus-bearing subset; only its output seeds exact lanes.
                    // Opt out of facade-derived audience aliases using the existing API; Sol's
                    // explicit audienceTerms still seed its independent cold-memory rescue lane.
                    audienceTerms: [],
                    semanticVariantCount: recallSemanticVariantCount,
                    recallQueryPlannerContext: {
                      contextTurns: contextTurns.map((turn) => ({
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
                    recordRetrieval: false,
                    ...(traceTurnId === undefined ? {} : { traceTurnId }),
                  };
                  const recalled = await borg.episodic.recallForCognition(recallFocus, {
                    ...recallOptions,
                    ...(parsed.data.time_range === undefined
                      ? {}
                      : {
                          timeRange: parsed.data.time_range,
                          strictTimeRange: false,
                        }),
                  });
                  const eligible = recalled.filter(
                    (hit) => !episodeMatchesExclusions(hit.episode, parsed.data.exclude),
                  );
                  // The period to prefer: an explicit time_range, else the cue the planner resolved
                  // from FOCUS, which is the range a caller with its own parser used to send.
                  const actedCue: TemporalCue | null =
                    (recallPlan as RecallPlanOutcome | null)?.temporalCue ?? null;
                  const preferredRange =
                    parsed.data.time_range ?? temporalCueRange(actedCue, nowMs);
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

                  if (!shouldAbstain) {
                    for (const hit of included) {
                      borg.episodic.recordRetrieval(hit.episode.id, hit.score);
                    }
                  }

                  return {
                    shouldAbstain,
                    servedEpisodes: shouldAbstain ? [] : included,
                    topRawScore,
                    plannerTemporalCue: parsed.data.time_range === undefined ? actedCue : null,
                    episodes: projectEpisodeHitsForResponse(
                      included,
                      borg.entities,
                      recallOptions.disclosureContext,
                      {
                        includeSourceMessages: true,
                        ...(preferredRange === undefined ? {} : { timeRange: preferredRange }),
                      },
                    ),
                  };
                }),
              requestDeadlineAt,
            );

            servedEpisodes = recallResult.servedEpisodes;
            plannerTemporalCue = recallResult.plannerTemporalCue;
            if (recallResult.shouldAbstain) {
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
            requestDeadlineAt - Date.now(),
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
                () =>
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
                Math.min(requestDeadlineAt, Date.now() + remainingMs),
              );
            } catch (error) {
              noteDegradation(
                `autobiographical_recall: ${error instanceof Error ? error.message : String(error)}`,
              );
            }
          }
        }

        if (context.activityDegradation.length > 0) {
          degraded = true;
          degradedReason = [degradedReason, context.activityDegradation].filter(Boolean).join("; ");
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
          response.hidden_episode_count = 0;
          response.disclosure_guidance = MEMORY_DISCLOSURE_GUIDANCE_FOR_MODEL;
          // Store only after the deadline has resolved, so abandoned recall cannot publish later.
          response.context_id = servedContexts.put(tenant, {
            sessionId: session,
            audienceEntityId: identity.audienceEntity.id,
            senderEntityId: identity.senderEntityId,
            episodes: servedEpisodes,
            commitments: context.applicableCommitments,
          });
          if (abstained) {
            response.abstained = true;
            response.abstain_reason = "low_relevance";
          }
        }
        if (requestedSections.has("recent_activity")) {
          response.recent_activity = context.recentActivity;
          response.recent_activity_selection = context.activitySelection;
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
        // Presence of the explicit scope discriminant selects the consent contract.
        // Invalid scoped requests cannot fall through to the legacy outcome writer.
        if ("scope" in body) {
          const parsed = memoryTenantRememberBodySchema.safeParse(body);
          if (!parsed.success) {
            send(res, 400, { error: "invalid memory remember body" });
            return;
          }
          const request = parsed.data;
          const session = sessionFromCaller(request.session);
          const result = await pool.withTenant(
            tenant,
            async (borg) => {
              const identity = resolveTeamAgentIdentity({
                borg,
                session,
                rawSession: request.session,
                sender: {
                  externalId: request.sender.external_id,
                  displayName: request.sender.display_name,
                  operator: request.sender.operator,
                },
                conversation: request.conversation,
              });
              if (identity.senderEntityId === null) throw new Error("Remember requires a sender");
              return await borg.episodic.rememberForTenant({
                sessionId: session,
                speakerEntityId: identity.senderEntityId,
                requestId: request.request_id,
                content: request.content,
                sourceEpisodeIds: request.source_episode_ids,
                sourceMessageIds: request.source_message_ids,
                authorizationMessageIds: request.authorization_message_ids,
              });
            },
            { exclusive: true },
          );
          send(res, 200, {
            ok: true,
            episode_id: result.episodeId,
            authorization_entry_id: result.authorizationEntryId,
            authorization: result.authorization,
            disclosure: { class: "public", scope: "tenant" },
            guidance: tenantFactSharingGuidance(result.authorization.speaker_name),
            duplicate: result.duplicate,
          });
          return;
        }
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
        const parsedIdentity = memoryTransportIdentitySchema.safeParse(body);
        if (!parsedIdentity.success) {
          send(res, 400, {
            error: `invalid transport identity: ${parsedIdentity.error.issues
              .map((issue) => `${issue.path.join(".")}: ${issue.message}`)
              .join("; ")}`,
          });
          return;
        }
        const sessionRaw = parsedIdentity.data.session;
        const session = sessionFromCaller(sessionRaw);
        const sender = {
          externalId: parsedIdentity.data.sender.external_id,
          displayName: parsedIdentity.data.sender.display_name,
          operator: parsedIdentity.data.sender.operator,
        };
        const conversation = parsedIdentity.data.conversation;
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
        const entries = await pool.withTenant(
          tenant,
          async (borg) => {
            const identity = resolveTeamAgentIdentity({
              borg,
              session,
              rawSession: sessionRaw,
              sender,
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
                throw new Error("observation did not produce a user entry input");
              }
              enrichedEntries = [await borg.stream.append(userEntryInput, { session })];
            } else if (replyOnly) {
              if (assistantEntryInput === undefined) {
                throw new Error("reply-only append did not produce an agent entry input");
              }
              enrichedEntries = [await borg.stream.append(assistantEntryInput, { session })];
            } else {
              if (userEntryInput === undefined || assistantEntryInput === undefined) {
                throw new Error("completed turn did not produce both entry inputs");
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
              throw new Error("append did not produce the requested entries");
            }

            try {
              const firstEntry = enrichedEntries[0];

              if (firstEntry === undefined) {
                throw new Error("append did not produce a projection source entry");
              }

              const sessionProjection = {
                ...identity.sessionEnsureInput,
                created_at: firstEntry.timestamp,
                last_activity_at: firstEntry.timestamp,
              };

              if (observation) {
                if (userEntry === undefined || identity.senderEntityId === null) {
                  throw new Error("observation requires a sender entry");
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
                  throw new Error("append did not produce an assistant entry");
                }

                const selfEntity = borg.entities.getSelf();

                if (selfEntity === null) {
                  throw new Error("append awareness projection requires a self entity");
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
                    throw new Error("completed turn requires a sender entry");
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
          () =>
            pool.withTenant(tenant, async (borg) => {
              const recallAbstainThreshold =
                options.recallAbstainThreshold ??
                borg.similarityThresholds?.recallAbstain ??
                similarityThresholds().recallAbstain;
              const memoryOwner = borg.entities.getSelf();
              const recallOptions = {
                limit: searchLimit,
                onDegraded: (degradation: RetrievalDegradation) => degradations.push(degradation),
                semanticVariantCount: recallSemanticVariantCount,
                recallQueryPlannerContext: {
                  identity: {
                    ...(memoryOwner === null
                      ? {}
                      : { memoryOwnerName: memoryOwner.canonical_name }),
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
                shouldAbstain,
                episodes: included,
                projected: projectEpisodeHitsForResponse(included, borg.entities, null),
                topRawScore,
                timeRangeFallback: recalledResult.timeRangeFallback,
              };
            }),
          requestDeadlineAt,
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

      if (hits.shouldAbstain) {
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
      if (sendEmbeddingBankUnavailable(res, error)) return;
      if (error instanceof RecallDeadlineExceeded) {
        send(res, 503, {
          error: "memory context deadline exceeded",
          degraded: true,
          degraded_reason: `deadline: ${error.message}`,
        });
        return;
      }
      if (rawPath === "/memory/remember") {
        const code = errorCode(error);
        if (error instanceof z.ZodError) {
          send(res, 400, { error: "invalid consent provenance" });
          return;
        }
        const status =
          code === "MEMORY_REMEMBER_NOT_AUTHORIZED"
            ? 422
            : code === "MEMORY_REMEMBER_SOURCE_INVALID"
              ? 400
              : code === "MEMORY_REMEMBER_CONFLICT"
                ? 409
                : null;
        if (status !== null) {
          send(res, status, {
            error:
              status === 422
                ? "source-backed tenant authorization required"
                : status === 409
                  ? "request_id already used with different input"
                  : "invalid consent provenance",
            code,
          });
          return;
        }
      }
      if (
        rawPath === "/memory/forget" &&
        (errorCode(error) === "EPISODE_NOT_FOUND" || errorCode(error) === "SEMANTIC_NODE_NOT_FOUND")
      ) {
        send(res, 404, { error: "memory not found" });
        return;
      }
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
      if (sendEmbeddingBankUnavailable(res, error)) return;
      console.error("memory-sidecar: unhandled request error", error);
      send(res, 500, { error: "internal error" });
    });
  };
}
