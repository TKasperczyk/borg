import { Buffer } from "node:buffer";

import { createNodeWebSocket } from "@hono/node-ws";
import { Hono, type Context } from "hono";
import { cors } from "hono/cors";
import { HTTPException } from "hono/http-exception";
import {
  DEFAULT_SESSION_ID,
  OFFLINE_PROCESS_NAMES,
  STREAM_ENTRY_KINDS,
  VERSION,
  type AttachmentId,
  type Borg,
  type CommitmentEnforcementClass,
  type CommitmentRecord,
  type EntityId,
  type ImagePerceptionRecord,
  type MaintenanceAuditRecord,
  type OfflineProcessName,
  type RelationalSlotState,
  type ReviewQueueItem,
  type StreamCursor,
  type StreamEntry,
  type StreamEntryKind,
  type StoredAttachmentRecord,
} from "borg";
import { z } from "zod";

import type { LiveBridge } from "./live.js";

type CursorPayload = {
  ts: number;
  entryId: string;
};

const cursorPayloadSchema = z.object({
  ts: z.number().finite(),
  entryId: z.string().min(1),
});

const csvKindsSchema = z
  .string()
  .optional()
  .transform((value, ctx) => {
    if (value === undefined || value.trim().length === 0) {
      return undefined;
    }

    const kinds = value
      .split(",")
      .map((item) => item.trim())
      .filter((item) => item.length > 0);
    const parsed = z.array(z.enum(STREAM_ENTRY_KINDS)).safeParse(kinds);

    if (!parsed.success) {
      ctx.addIssue({
        code: "custom",
        message: `kind must be one or more of ${STREAM_ENTRY_KINDS.join(",")}`,
      });
      return z.NEVER;
    }

    return parsed.data;
  });

const limitSchema = z.coerce.number().int().min(1).max(500);

const streamQuerySchema = z.object({
  kind: csvKindsSchema,
  audience: z.string().min(1).optional(),
  limit: limitSchema.default(50),
  before: z
    .string()
    .optional()
    .transform((value, ctx) => {
      if (value === undefined || value.length === 0) {
        return undefined;
      }

      const parsed = decodeCursor(value);

      if (parsed === null) {
        ctx.addIssue({ code: "custom", message: "before is not a valid stream cursor" });
        return z.NEVER;
      }

      return parsed;
    }),
});

const audienceQuerySchema = z.object({
  audience: z.string().min(1).optional(),
});

const auditQuerySchema = z.object({
  limit: limitSchema.default(50),
});

const commitmentQuerySchema = z.object({
  audience: z.string().min(1).optional(),
  state: z.enum(["active", "all", "revoked", "expired"]).default("active"),
  enforcement: z.enum(["critical", "advisory"]).optional(),
});

const attachmentQuerySchema = z.object({
  audience: z.string().min(1).nullable().optional(),
});
const attachmentIdParamSchema = z
  .string()
  .regex(/^att_[a-z0-9]{16}$/, "Invalid attachment id")
  .transform((value) => value as AttachmentId);
const attachmentParamSchema = z.object({
  id: attachmentIdParamSchema,
});
const attachmentBatchQuerySchema = z.object({
  ids: z.string().min(1).transform((value, ctx) => {
    const ids = value
      .split(",")
      .map((item) => item.trim())
      .filter((item) => item.length > 0);

    if (ids.length === 0) {
      ctx.addIssue({ code: "custom", message: "ids must include at least one attachment id" });
      return z.NEVER;
    }

    if (ids.length > 200) {
      ctx.addIssue({ code: "custom", message: "ids must include at most 200 attachment ids" });
      return z.NEVER;
    }

    const parsed = z.array(attachmentIdParamSchema).safeParse(ids);
    if (!parsed.success) {
      ctx.addIssue({ code: "custom", message: "ids contains an invalid attachment id" });
      return z.NEVER;
    }

    return [...new Set(parsed.data)];
  }),
});

const memoryBandIdSchema = z.enum([
  "episodic",
  "semantic",
  "procedural",
  "affective",
  "self",
  "commitments",
  "social",
  "relational",
]);

const relationalStateQuerySchema = z.object({
  state: z
    .enum(["established", "contested", "quarantined", "revoked"])
    .optional(),
  limit: limitSchema.default(100),
});

const turnBodySchema = z.object({
  message: z.string().min(1),
  audience: z.string().min(1),
  stakes: z.enum(["low", "medium", "high"]).optional(),
});

function parseRequest<T>(schema: z.ZodType<T>, value: unknown): T {
  const parsed = schema.safeParse(value);

  if (!parsed.success) {
    throw new HTTPException(400, { message: parsed.error.message });
  }

  return parsed.data;
}

async function parseJsonBody(c: Context): Promise<unknown> {
  try {
    return await c.req.json();
  } catch {
    throw new HTTPException(400, { message: "Malformed JSON body" });
  }
}

function jsonError(status: number, message: string): Response {
  return new Response(
    JSON.stringify({
      error: {
        status,
        message,
      },
    }),
    {
      status,
      headers: {
        "Content-Type": "application/json",
      },
    },
  );
}

function encodeCursor(entry: Pick<StreamEntry, "timestamp" | "id">): string {
  const payload: CursorPayload = {
    ts: entry.timestamp,
    entryId: entry.id,
  };

  return Buffer.from(JSON.stringify(payload), "utf8").toString("base64url");
}

function decodeCursor(cursor: string): StreamCursor | null {
  try {
    const raw = Buffer.from(cursor, "base64url").toString("utf8");
    const parsed = cursorPayloadSchema.parse(JSON.parse(raw));
    return {
      ts: parsed.ts,
      entryId: parsed.entryId as StreamCursor["entryId"],
    };
  } catch {
    return null;
  }
}

async function readStream(input: {
  borg: Borg;
  kinds?: readonly StreamEntryKind[];
  audience?: string;
  limit: number;
  before?: StreamCursor;
}): Promise<{ entries: StreamEntry[]; next_cursor: string | null }> {
  const collected: StreamEntry[] = [];
  const reader = input.borg.stream.reader();

  for await (const entry of reader.iterate({
    kinds: input.kinds,
    untilCursor: input.before,
  })) {
    if (input.audience !== undefined && entry.audience !== input.audience) {
      continue;
    }

    collected.push(entry);
  }

  const cursorIndex =
    input.before === undefined
      ? -1
      : collected.findIndex(
          (entry) => entry.timestamp === input.before?.ts && entry.id === input.before.entryId,
        );
  const beforeCursor = cursorIndex === -1 ? collected : collected.slice(0, cursorIndex);
  const page = beforeCursor.slice(-(input.limit + 1));
  const entries = page.length > input.limit ? page.slice(1) : page;
  const next_cursor =
    page.length > input.limit && entries[0] !== undefined ? encodeCursor(entries[0]) : null;

  return { entries, next_cursor };
}

async function countTurns(borg: Borg): Promise<number> {
  let count = 0;

  for await (const entry of borg.stream.reader().iterate({ kinds: ["user_msg"] })) {
    if (entry.turn_status !== "aborted") {
      count += 1;
    }
  }

  return count;
}

function listAudiences(borg: Borg): string[] {
  return [
    ...new Set(
      borg.stream
        .tail(500)
        .flatMap((entry) => (entry.audience === undefined ? [] : [entry.audience])),
    ),
  ].sort();
}

function sumRecord(record: Record<string, number>): number {
  return Object.values(record).reduce((sum, value) => sum + value, 0);
}

function sparkFrom(count: number): number[] {
  const base = Math.max(1, Math.min(12, count));
  return Array.from({ length: 15 }, (_, index) => Math.max(1, Math.round(base * (0.45 + index / 20))));
}

function entityLabel(borg: Borg, id: EntityId | string | null | undefined): string | null {
  if (id === null || id === undefined) {
    return null;
  }

  return borg.entities.get(id as EntityId)?.canonical_name ?? String(id);
}

function commitmentState(record: CommitmentRecord): "active" | "revoked" | "expired" {
  if (record.expired_at !== null) {
    return "expired";
  }

  if (record.revoked_at !== null || record.superseded_by !== null) {
    return "revoked";
  }

  return "active";
}

function mapCommitment(borg: Borg, record: CommitmentRecord) {
  return {
    id: record.id,
    text: record.directive,
    type: record.type,
    kind: record.kind,
    enforcement_class: record.enforcement_class,
    critical_domain: record.critical_domain,
    state: commitmentState(record),
    priority: record.priority,
    directive_family: record.directive_family,
    audience: entityLabel(borg, record.restricted_audience),
    made_to: entityLabel(borg, record.made_to_entity),
    about: entityLabel(borg, record.about_entity),
    committed_by: entityLabel(borg, record.committed_by_entity_id ?? null),
    source: record.provenance.kind,
    source_stream_entry_ids: record.source_stream_entry_ids ?? [],
    created_at: record.created_at,
    expires_at: record.expires_at,
    expired_at: record.expired_at,
    revoked_at: record.revoked_at,
    revoked_reason: record.revoked_reason,
    superseded_by_id: record.superseded_by,
    canonicalized_by_artifact_entry_id: record.canonicalized_by_artifact_entry_id ?? null,
    last_reinforced_at: record.last_reinforced_at,
  };
}

function mapEpisode(borg: Borg, item: Awaited<ReturnType<Borg["episodic"]["list"]>>["items"][number]) {
  return {
    id: item.id,
    title: item.title,
    narrative: item.narrative,
    participants: item.participants,
    location: item.location,
    start_time: item.start_time,
    end_time: item.end_time,
    audience: entityLabel(borg, item.audience_entity_id ?? null),
    significance: item.significance,
    confidence: item.confidence,
    tags: item.tags,
    source_stream_ids: item.source_stream_ids,
    source_count: item.source_stream_ids.length,
    lineage: item.lineage,
    emotional_arc: item.emotional_arc,
    vector_dims: item.embedding.length,
    created_at: item.created_at,
    updated_at: item.updated_at,
  };
}

function mapAttachmentMetadata(input: {
  attachment: StoredAttachmentRecord;
  perception: ImagePerceptionRecord | null;
  status: {
    active: boolean;
    quarantined: boolean;
    stream_active?: boolean;
    parent_active?: boolean;
  };
}) {
  return {
    attachment: input.attachment,
    perception: input.perception,
    status: input.status,
  };
}

function mapAttachmentStatus(input: {
  attachment: StoredAttachmentRecord;
  status: {
    active: boolean;
    quarantined: boolean;
    stream_active?: boolean;
    parent_active?: boolean;
  };
}) {
  return {
    id: input.attachment.attachment_id,
    status: input.status,
  };
}

function mapReviewRow(row: ReviewQueueItem) {
  return {
    id: row.id,
    kind: row.kind,
    refs: row.refs,
    reason: row.reason,
    created_at: row.created_at,
    resolved_at: row.resolved_at,
    resolution: row.resolution,
  };
}

function processDescription(name: OfflineProcessName): string {
  const descriptions: Record<OfflineProcessName, string> = {
    consolidator: "merge redundant episodes",
    reflector: "episodes to semantic insights",
    "semantic-extractor": "extract graph facts",
    curator: "salience, heat, archive, decay",
    overseer: "flag substrate issues",
    "review-resolver": "process review queue items",
    ruminator: "open-question rumination",
    "self-narrator": "autobiography and growth markers",
    "procedural-synthesizer": "skill abstractions",
    "belief-reviser": "invalidate, weaken, contradict",
  };

  return descriptions[name];
}

function streamDreamProcesses(entry: StreamEntry): OfflineProcessName[] {
  if (entry.kind !== "dream_report" || entry.content === null || typeof entry.content !== "object") {
    return [];
  }

  const processes = (entry.content as { processes?: unknown }).processes;

  if (!Array.isArray(processes)) {
    return [];
  }

  return processes.filter((value): value is OfflineProcessName =>
    OFFLINE_PROCESS_NAMES.includes(value as OfflineProcessName),
  );
}

function streamDreamHasProcessError(entry: StreamEntry, process: OfflineProcessName): boolean {
  if (entry.content === null || typeof entry.content !== "object") {
    return false;
  }

  const errors = (entry.content as { errors?: unknown }).errors;

  if (!Array.isArray(errors)) {
    return false;
  }

  return errors.some((error) => {
    if (error === null || typeof error !== "object") {
      return false;
    }

    return (error as { process?: unknown }).process === process;
  });
}

function dreamScheduleFromAudit(rows: ReadonlyArray<Pick<MaintenanceAuditRecord, "id" | "applied_at"> & { process: string }>) {
  return rows.flatMap((row) => {
    if (!OFFLINE_PROCESS_NAMES.includes(row.process as OfflineProcessName)) {
      return [];
    }

    return [
      {
        process: row.process as OfflineProcessName,
        scheduled_at: row.applied_at,
        source: "audit" as const,
        audit_id: row.id,
      },
    ];
  });
}

function latestDreamRunForProcess(
  process: OfflineProcessName,
  dreamReports: readonly StreamEntry[],
  auditRows: ReadonlyArray<Pick<MaintenanceAuditRecord, "id" | "applied_at"> & { process: string }>,
) {
  const streamMatches = dreamReports.filter((entry) => streamDreamProcesses(entry).includes(process));
  const auditMatches = auditRows.filter((row) => row.process === process);
  const lastRunAt = Math.max(
    ...streamMatches.map((entry) => entry.timestamp),
    ...auditMatches.map((row) => row.applied_at),
    Number.NEGATIVE_INFINITY,
  );

  if (lastRunAt === Number.NEGATIVE_INFINITY) {
    return {
      last_run_at: null,
      last_status: null,
      last_audit_id: null,
    };
  }

  const latestStream = streamMatches.find((entry) => entry.timestamp === lastRunAt);
  const latestAudit = auditMatches
    .filter((row) => row.applied_at === lastRunAt)
    .sort((left, right) => right.id - left.id)[0];

  return {
    last_run_at: lastRunAt,
    last_status:
      latestStream === undefined ? "ok" : streamDreamHasProcessError(latestStream, process) ? "error" : "ok",
    last_audit_id: latestAudit?.id ?? null,
  };
}

async function memoryBands(borg: Borg) {
  const episodes = await borg.episodic.list({ limit: 500 });
  const semanticCounts = borg.semantic.nodes.countByStatus();
  const procedural = borg.skills.list(500);
  const moodHistory = borg.mood.history(DEFAULT_SESSION_ID, { limit: 500 });
  const values = borg.self.values.list();
  const goals = borg.self.goals.list();
  const traits = borg.self.traits.list();
  const openQuestions = borg.self.openQuestions.list({ status: "open" });
  const growthMarkers = borg.self.growthMarkers.list({ limit: 500 });
  const periods = borg.self.autobiographical.listPeriods({ limit: 500 });
  const relationalCounts = borg.relationalSlots.countByState();
  const audiences = listAudiences(borg);

  return [
    {
      id: "episodic",
      n: "01",
      name: "episodic",
      desc: "what happened",
      count: episodes.items.length,
      growth: sparkFrom(episodes.items.length),
      stats: [{ k: "items", v: episodes.items.length }],
    },
    {
      id: "semantic",
      n: "02",
      name: "semantic",
      desc: "what Borg believes",
      count: sumRecord(semanticCounts),
      growth: sparkFrom(sumRecord(semanticCounts)),
      stats: Object.entries(semanticCounts).map(([k, v]) => ({ k, v })),
    },
    {
      id: "procedural",
      n: "03",
      name: "procedural",
      desc: "how Borg solves things",
      count: procedural.length,
      growth: sparkFrom(procedural.length),
      stats: [{ k: "skills", v: procedural.length }],
    },
    {
      id: "affective",
      n: "04",
      name: "affective",
      desc: "mood and trajectory",
      count: moodHistory.length,
      growth: sparkFrom(moodHistory.length),
      stats: [{ k: "points", v: moodHistory.length }],
    },
    {
      id: "self",
      n: "05",
      name: "self",
      desc: "values, goals, traits, narrative",
      count:
        values.length +
        goals.length +
        traits.length +
        openQuestions.length +
        growthMarkers.length +
        periods.length,
      growth: sparkFrom(values.length + goals.length + traits.length + openQuestions.length),
      stats: [
        { k: "values", v: values.length },
        { k: "goals", v: goals.length },
        { k: "traits", v: traits.length },
        { k: "open_questions", v: openQuestions.length },
        { k: "growth_markers", v: growthMarkers.length },
        { k: "periods", v: periods.length },
      ],
    },
    {
      id: "commitments",
      n: "06",
      name: "commitments",
      desc: "scoped promises and boundaries",
      count: borg.commitments.countActive(),
      growth: sparkFrom(borg.commitments.countActive()),
      stats: [
        { k: "active", v: borg.commitments.countActive() },
        { k: "revoked", v: borg.commitments.countRevoked() },
      ],
    },
    {
      id: "social",
      n: "07",
      name: "social",
      desc: "per-entity trust and history",
      count: borg.social.list(500).length,
      growth: sparkFrom(audiences.length),
      stats: [{ k: "profiles", v: borg.social.list(500).length }],
    },
    {
      id: "relational",
      n: "08",
      name: "relational",
      desc: "evidence-backed relationship facts",
      count: sumRecord(relationalCounts),
      growth: sparkFrom(sumRecord(relationalCounts)),
      stats: Object.entries(relationalCounts).map(([k, v]) => ({ k, v })),
    },
  ];
}

function selfSnapshot(borg: Borg) {
  return {
    values: borg.self.values.list(),
    goals: borg.self.goals.list(),
    traits: borg.self.traits.list(),
    open_questions: borg.self.openQuestions.list({ limit: 250 }),
    growth_markers: borg.self.growthMarkers.list({ limit: 100 }),
    periods: borg.self.autobiographical.listPeriods({ limit: 100 }),
    open_question_events: borg.identity.listEvents({ recordType: "open_question", limit: 250 }),
  };
}

function dreamState(borg: Borg) {
  const auditRows = borg.audit.list().slice(0, 50);
  const dreamReports = borg.stream.tail(500).filter((entry) => entry.kind === "dream_report");
  const config = borg.maintenance.config();

  const processes = OFFLINE_PROCESS_NAMES.map((name) => {
    const lastRun = latestDreamRunForProcess(name, dreamReports, auditRows);

    return {
      name,
      description: processDescription(name),
      last_run_at: lastRun.last_run_at,
      last_status: lastRun.last_status,
      last_audit_id: lastRun.last_audit_id,
      budget: config.processBudgets[name] ?? null,
      enabled:
        config.lightProcesses.includes(name) ||
        config.heavyProcesses.includes(name) ||
        config.processBudgets[name] !== undefined,
    };
  });

  const streamSchedule = dreamReports.flatMap((entry) =>
    streamDreamProcesses(entry).map((process) => ({
      process,
      scheduled_at: entry.timestamp,
      source: "stream" as const,
      stream_entry_id: entry.id,
    })),
  );

  return {
    processes,
    schedule: [...streamSchedule, ...dreamScheduleFromAudit(auditRows)]
      .sort((left, right) => right.scheduled_at - left.scheduled_at)
      .slice(0, 80),
    audit_rows: auditRows,
    belief_revision_rows: borg.review
      .list({ kind: "belief_revision", openOnly: true })
      .map((row) => mapReviewRow(row)),
    scheduler: {
      enabled: borg.maintenance.scheduler.isEnabled(),
      light_interval_ms: config.lightIntervalMs,
      heavy_interval_ms: config.heavyIntervalMs,
      light_processes: config.lightProcesses,
      heavy_processes: config.heavyProcesses,
      process_budgets: config.processBudgets,
    },
  };
}

export function createDemoServerApp(input: {
  borg: Borg;
  live: LiveBridge;
  corsOrigins?: readonly string[];
}) {
  const app = new Hono();
  const { injectWebSocket, upgradeWebSocket } = createNodeWebSocket({ app });
  const allowedOrigins = input.corsOrigins ?? ["http://localhost:5173"];

  app.onError((error) => {
    if (error instanceof HTTPException) {
      return jsonError(error.status, error.message);
    }

    console.error(error instanceof Error ? error.message : String(error));
    return jsonError(500, "Internal Server Error");
  });

  app.use(
    "/api/*",
    cors({
      origin: (origin) => (allowedOrigins.includes(origin) ? origin : (allowedOrigins[0] ?? "")),
    }),
  );

  app.get(
    "/api/live",
    upgradeWebSocket(() => ({
      onOpen: (_event, ws) => input.live.broadcaster.add(ws),
      onClose: (_event, ws) => input.live.broadcaster.remove(ws),
      onError: (_event, ws) => input.live.broadcaster.remove(ws),
    })),
  );

  app.get("/api/state", async (c) => {
    const auditRows = input.borg.audit.list();

    return c.json({
      active_session: DEFAULT_SESSION_ID,
      audiences: listAudiences(input.borg),
      counts: {
        turns: await countTurns(input.borg),
        commitments: input.borg.commitments.countActive(),
        open_qs: input.borg.self.openQuestions.list({ status: "open" }).length,
        dream_audit_rows: auditRows.length,
      },
      current_mood: input.borg.mood.current(DEFAULT_SESSION_ID),
      version: VERSION,
    });
  });

  app.get("/api/stream", async (c) => {
    const query = parseRequest(streamQuerySchema, c.req.query());
    return c.json(
      await readStream({
        borg: input.borg,
        kinds: query.kind,
        audience: query.audience,
        limit: query.limit,
        before: query.before,
      }),
    );
  });

  app.get("/api/turns/:id/ledger", (c) => {
    const turnId = c.req.param("id");
    const ledger = input.live.ledgerCache.get(turnId);

    if (ledger === undefined) {
      throw new HTTPException(404, { message: "ledger not found" });
    }

    return c.json({ turn_id: turnId, ledger });
  });

  app.get("/api/memory/bands", async (c) => c.json({ bands: await memoryBands(input.borg) }));

  app.get("/api/memory/bands/:id", async (c) => {
    const band = parseRequest(memoryBandIdSchema, c.req.param("id"));

    if (band === "episodic") {
      const result = await input.borg.episodic.list({ limit: 50 });
      return c.json({
        band,
        items: result.items.map((item) => mapEpisode(input.borg, item)),
        nextCursor: result.nextCursor ?? null,
      });
    }

    if (band === "semantic") {
      const nodes = await input.borg.semantic.nodes.list({ limit: 50 });
      const edges = input.borg.semantic.edges.list().slice(0, 50);

      return c.json({
        band,
        nodes: nodes.map((node) => ({
          id: node.id,
          kind: node.kind,
          label: node.label,
          description: node.description,
          domain: node.domain,
          aliases: node.aliases,
          confidence: node.confidence,
          status: node.status,
          source_episode_ids: node.source_episode_ids,
          source_count: node.source_episode_ids.length,
          created_at: node.created_at,
          updated_at: node.updated_at,
        })),
        edges: edges.map((edge) => ({
          id: edge.id,
          from_node_id: edge.from_node_id,
          to_node_id: edge.to_node_id,
          relation: edge.relation,
          confidence: edge.confidence,
          evidence_episode_ids: edge.evidence_episode_ids,
          source_count: edge.evidence_episode_ids.length,
          valid_from: edge.valid_from,
          valid_to: edge.valid_to,
          invalidated_at: edge.invalidated_at,
          invalidated_by_edge_id: edge.invalidated_by_edge_id,
          invalidated_by_review_id: edge.invalidated_by_review_id,
          invalidated_by_process: edge.invalidated_by_process,
          invalidated_reason: edge.invalidated_reason,
        })),
      });
    }

    if (band === "procedural") {
      return c.json({
        band,
        items: input.borg.skills.list(100).map((skill) => ({
          id: skill.id,
          applies_when: skill.applies_when,
          approach: skill.approach,
          status: skill.status,
          alpha: skill.alpha,
          beta: skill.beta,
          attempts: skill.attempts,
          successes: skill.successes,
          failures: skill.failures,
          sample_count: skill.source_episode_ids.length,
          source_episode_ids: skill.source_episode_ids,
          last_used: skill.last_used,
          last_successful: skill.last_successful,
          requires_manual_review: skill.requires_manual_review,
          created_at: skill.created_at,
          updated_at: skill.updated_at,
        })),
      });
    }

    if (band === "affective") {
      return c.json({
        band,
        current: input.borg.mood.current(DEFAULT_SESSION_ID),
        history: input.borg.mood.history(DEFAULT_SESSION_ID, { limit: 100 }),
      });
    }

    if (band === "commitments") {
      return c.json({
        band,
        items: input.borg.commitments.list({ activeOnly: false }).map((record) => mapCommitment(input.borg, record)),
      });
    }

    if (band === "self") {
      return c.json({ band, ...selfSnapshot(input.borg) });
    }

    if (band === "social") {
      return c.json({
        band,
        items: input.borg.social.list(100).map((profile) => ({
          entity_id: profile.entity_id,
          name: entityLabel(input.borg, profile.entity_id),
          trust: profile.trust,
          attachment: profile.attachment,
          interaction_count: profile.interaction_count,
          history_count: profile.interaction_count,
          commitment_count: profile.commitment_count,
          last_interaction_at: profile.last_interaction_at,
          updated_at: profile.updated_at,
        })),
      });
    }

    const relationalQuery = parseRequest(relationalStateQuerySchema, c.req.query());
    return c.json({
      band,
      counts: input.borg.relationalSlots.countByState(),
      items: input.borg.relationalSlots
        .list({
          limit: relationalQuery.limit,
          states:
            relationalQuery.state === undefined
              ? undefined
              : ([relationalQuery.state] as RelationalSlotState[]),
        })
        .map((slot) => ({
          id: slot.id,
          slot: `${entityLabel(input.borg, slot.subject_entity_id) ?? slot.subject_entity_id}.${slot.slot_key}`,
          subject_entity_id: slot.subject_entity_id,
          subject: entityLabel(input.borg, slot.subject_entity_id),
          slot_key: slot.slot_key,
          value: slot.value,
          state: slot.state,
          sources_count: slot.evidence_stream_entry_ids.length,
          contradicted_count: slot.contradicted_by_stream_entry_ids.length,
          alternate_count: slot.alternate_values.length,
          name_provenance: slot.name_provenance ?? "unknown",
          created_at: slot.created_at,
          updated_at: slot.updated_at,
        })),
    });
  });

  app.get("/api/commitments", (c) => {
    const query = parseRequest(commitmentQuerySchema, c.req.query());
    const activeOnly = query.state === "active";
    const filterByState = query.state === "all" ? undefined : query.state;

    if (query.audience !== undefined) {
      const entity = input.borg.entities.find(query.audience);

      if (entity === null) {
        return c.json({ commitments: [] });
      }

      const commitments = input.borg.commitments
        .list({
          activeOnly,
          audience: entity.canonical_name,
        })
        .map((record) => mapCommitment(input.borg, record))
        .filter((record) => filterByState === undefined || record.state === filterByState)
        .filter(
          (record) =>
            query.enforcement === undefined || record.enforcement_class === query.enforcement,
        );

      return c.json({
        commitments,
      });
    }

    const commitments = input.borg.commitments
      .list({ activeOnly })
      .map((record) => mapCommitment(input.borg, record))
      .filter((record) => filterByState === undefined || record.state === filterByState)
      .filter(
        (record) => query.enforcement === undefined || record.enforcement_class === query.enforcement,
      );

    return c.json({ commitments });
  });

  app.get("/api/shared-state", (c) => {
    const query = parseRequest(audienceQuerySchema, c.req.query());
    const audience = query.audience ?? "self";
    return c.json({ audience, entries: input.borg.sharedState.listEntriesForAudience(audience) });
  });

  app.get("/api/identity", (c) => c.json(selfSnapshot(input.borg)));

  app.get("/api/dream/audit", (c) => {
    const query = parseRequest(auditQuerySchema, c.req.query());
    return c.json({ rows: input.borg.audit.list().slice(0, query.limit) });
  });

  app.get("/api/dream/state", (c) => c.json(dreamState(input.borg)));

  app.get("/api/attachments", (c) => {
    const query = parseRequest(attachmentBatchQuerySchema, c.req.query());
    // Batch status lookup keeps visible stream attachment rows badged without
    // exposing bytes or adding per-row request fanout on the client.
    return c.json(
      query.ids.flatMap((id) => {
        const result = input.borg.attachments.get(id);
        return result === null ? [] : [mapAttachmentStatus(result)];
      }),
    );
  });

  app.get("/api/attachments/:id", (c) => {
    const params = parseRequest(attachmentParamSchema, c.req.param());
    const result = input.borg.attachments.get(params.id);

    if (result === null) {
      throw new HTTPException(404, { message: "attachment not found" });
    }

    return c.json(mapAttachmentMetadata(result));
  });

  app.get("/api/attachments/:id/bytes", (c) => {
    const params = parseRequest(attachmentParamSchema, c.req.param());
    const query = parseRequest(attachmentQuerySchema, c.req.query());
    const result = input.borg.attachments.getBytes(params.id, {
      audience: query.audience,
    });

    if (result === null) {
      throw new HTTPException(404, { message: "attachment not found" });
    }

    return new Response(result.bytes, {
      status: 200,
      headers: {
        "Content-Type": result.mediaType,
        "Content-Length": String(result.bytes.byteLength),
      },
    });
  });

  app.post("/api/turn", async (c) => {
    const body = parseRequest(turnBodySchema, await parseJsonBody(c));
    const result = await input.borg.turn({
      userMessage: body.message,
      audience: body.audience,
      stakes: body.stakes,
    });

    return c.json({ turn_id: result.turn_id, ok: true });
  });

  return { app, injectWebSocket };
}
