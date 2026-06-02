import { Buffer } from "node:buffer";
import { performance } from "node:perf_hooks";

import { createNodeWebSocket } from "@hono/node-ws";
import { Hono, type Context } from "hono";
import { cors } from "hono/cors";
import { HTTPException } from "hono/http-exception";
import {
  BorgError,
  COMMITMENT_KINDS,
  DEFAULT_SESSION_ID,
  OFFLINE_PROCESS_NAMES,
  REVIEW_KINDS,
  REVIEW_RESOLUTIONS,
  SESSION_PARTICIPATION_POLICIES,
  STREAM_ENTRY_KINDS,
  VERSION,
  type AttachmentId,
  type Borg,
  type CommitmentEnforcementClass,
  type CommitmentRecord,
  type CreatorDirective,
  type EntityId,
  type Episode,
  type ImageMediaType,
  type ImagePerceptionRecord,
  type MaintenanceCadence,
  type MaintenanceAuditRecord,
  type MaintenanceTickResult,
  type MaintenancePlan,
  type OfflineProcessName,
  type OrchestratorResult,
  type RelationalSlotState,
  type ReviewQueueItem,
  type SemanticEdge,
  type SemanticNode,
  type SemanticNodeStatus,
  type SessionId,
  type StreamCursor,
  type StreamEntry,
  type StreamEntryKind,
  type StoredAttachmentRecord,
  type TurnInputAttachment,
  createSessionId,
  creatorDirectiveIdSchema,
  creatorDirectiveReconciliationReviewRefsSchema,
  parseCommitmentId,
  parseEntityId,
  parseGoalId,
  parseOpenQuestionId,
  parseSessionId,
  PROMPT_KEYS,
  semanticNodeIdSchema,
  type AuditId,
  type BorgReviewResolutionInput,
  type PromptKey,
  type ReviewKind,
} from "borg";
import { z } from "zod";

import type { LiveBridge, MaintenanceTickFrame, MaintenanceTickFrameStatus } from "./live.js";
import type { BorgHandle } from "./reset.js";

type CursorPayload = {
  ts: number;
  entryId: string;
};

type SemanticExtractionEpisodicFacade = Borg["episodic"] & {
  listAll: () => Promise<Episode[]>;
  getStats: (id: Episode["id"]) => { archived?: boolean } | null;
};

type DemoMaintenanceAuditRow = ReturnType<Borg["audit"]["list"]>[number];

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
  session: z
    .string()
    .min(1)
    .optional()
    .transform((value, ctx) => parseOptionalSessionQuery(value, ctx)),
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

const sessionQuerySchema = z.object({
  session: z
    .string()
    .min(1)
    .optional()
    .transform((value, ctx) => parseOptionalSessionQuery(value, ctx)),
});

const sessionParamSchema = z.object({
  id: z.string().transform((value, ctx) => {
    try {
      return parseSessionId(value);
    } catch {
      ctx.addIssue({ code: "custom", message: "Invalid session id" });
      return z.NEVER;
    }
  }),
});

const sessionParticipationBodySchema = z
  .object({
    policy: z.enum(SESSION_PARTICIPATION_POLICIES),
    reason: z
      .string()
      .trim()
      .max(500)
      .optional()
      .transform((value) => (value === undefined || value.length === 0 ? undefined : value)),
  })
  .strict();

const entityParamSchema = z.object({
  id: z.string().transform((value, ctx) => {
    try {
      return parseEntityId(value);
    } catch {
      ctx.addIssue({ code: "custom", message: "Invalid entity id" });
      return z.NEVER;
    }
  }),
});

const entityBorgRoleBodySchema = z
  .object({
    role: z.literal("creator").nullable(),
  })
  .strict();

const creatorNameBodySchema = z
  .object({
    name: z.string().trim().min(1).max(200),
  })
  .strict();

const auditQuerySchema = z.object({
  limit: limitSchema.default(50),
});

const auditParamSchema = z.object({
  id: z.coerce
    .number()
    .int()
    .positive()
    .transform((value, ctx) => {
      if (!Number.isSafeInteger(value)) {
        ctx.addIssue({ code: "custom", message: "Invalid audit id" });
        return z.NEVER;
      }

      return value as AuditId;
    }),
});

const SEMANTIC_GRAPH_DEFAULT_LIMIT = 300;
const SEMANTIC_GRAPH_MAX_LIMIT = 500;
const semanticGraphQuerySchema = z.object({
  limit: z.coerce
    .number()
    .int()
    .min(1)
    .optional()
    .transform((value) =>
      Math.min(value ?? SEMANTIC_GRAPH_DEFAULT_LIMIT, SEMANTIC_GRAPH_MAX_LIMIT),
    ),
});

const commitmentQuerySchema = z.object({
  audience: z.string().min(1).optional(),
  state: z.enum(["active", "all", "revoked", "expired"]).default("active"),
  enforcement: z.enum(["critical", "advisory"]).optional(),
});

const attachmentQuerySchema = z.object({
  audience: z.string().min(1),
});
const attachmentIdParamSchema = z
  .string()
  .regex(/^att_[a-z0-9]{16}$/, "Invalid attachment id")
  .transform((value) => value as AttachmentId);
const attachmentParamSchema = z.object({
  id: attachmentIdParamSchema,
});
const attachmentBatchQuerySchema = z.object({
  ids: z
    .string()
    .min(1)
    .transform((value, ctx) => {
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
  state: z.enum(["established", "contested", "quarantined", "revoked"]).optional(),
  limit: limitSchema.default(100),
});

const DEMO_TURN_ATTACHMENT_MAX_BYTES = 8 * 1024 * 1024;
const DEMO_TURN_ATTACHMENT_MEDIA_TYPES = [
  "image/png",
  "image/jpeg",
  "image/gif",
  "image/webp",
] as const satisfies readonly ImageMediaType[];

const turnAttachmentMediaTypeSchema = z.enum(DEMO_TURN_ATTACHMENT_MEDIA_TYPES);
const turnBodySchema = z.object({
  message: z.string().trim().min(1),
  external_message_id: z.string().trim().min(1),
  audience: z.string().trim().min(1).optional(),
  audience_entity_id: z.string().trim().min(1).optional(),
  sender_entity_id: z.string().trim().min(1).optional(),
  session: z.string().trim().min(1).optional(),
});

const entityCreateBodySchema = z.object({
  name: z.string().trim().min(1),
  kind: z.enum(["person", "group", "self", "abstract"]).optional(),
});

const offlineProcessNameSchema = z.enum(OFFLINE_PROCESS_NAMES);

const dreamPlanBodySchema = z
  .object({
    processes: z.array(offlineProcessNameSchema).min(1).optional(),
    budget: z.number().int().positive().optional(),
  })
  .strict();

const dreamApplyBodySchema = dreamPlanBodySchema
  .extend({
    plan_id: z.string().min(1).optional(),
  })
  .strict();

const textFieldSchema = z.string().trim().min(1);
const optionalTextFieldSchema = z.string().trim().min(1).optional();

const identityValueBodySchema = z
  .object({
    name: textFieldSchema,
    description: optionalTextFieldSchema,
  })
  .strict();

const identityGoalBodySchema = z
  .object({
    description: textFieldSchema,
    priority: z.number().finite().optional(),
  })
  .strict();

const COMMITMENT_TEXT_MAX_LENGTH = 2_000;
const COMMITMENT_DIRECTIVE_FAMILY_MAX_LENGTH = 64;
const demoCommitmentTypeSchema = z.enum(["rule", "preference", "boundary"]);
const commitmentOptionalLabelSchema = z
  .string()
  .trim()
  .min(1)
  .max(COMMITMENT_TEXT_MAX_LENGTH)
  .optional();
const commitmentCreateBodySchema = z
  .object({
    type: demoCommitmentTypeSchema,
    kind: z.enum(COMMITMENT_KINDS),
    directive: z.string().trim().min(1).max(COMMITMENT_TEXT_MAX_LENGTH),
    priority: z.number().int().min(1).max(10),
    audience: commitmentOptionalLabelSchema,
    made_to: commitmentOptionalLabelSchema,
    about: commitmentOptionalLabelSchema,
    directive_family: z
      .string()
      .trim()
      .min(1)
      .max(COMMITMENT_DIRECTIVE_FAMILY_MAX_LENGTH)
      .optional(),
    expires_at: z.number().int().nonnegative().optional(),
  })
  .strict();

const commitmentRevokeBodySchema = z
  .object({
    reason: z.string().trim().max(COMMITMENT_TEXT_MAX_LENGTH).optional(),
  })
  .strict();

const commitmentParamSchema = z.object({
  id: z.string().transform((value, ctx) => {
    try {
      return parseCommitmentId(value);
    } catch {
      ctx.addIssue({ code: "custom", message: "Invalid commitment id" });
      return z.NEVER;
    }
  }),
});

const goalPatchBodySchema = z.discriminatedUnion("action", [
  z
    .object({
      action: z.literal("complete"),
      note: optionalTextFieldSchema,
    })
    .strict(),
  z
    .object({
      action: z.literal("block"),
      note: optionalTextFieldSchema,
    })
    .strict(),
  z
    .object({
      action: z.literal("progress"),
      note: optionalTextFieldSchema,
      progress: z.number().min(0).max(100).optional(),
    })
    .strict()
    .refine((value) => value.note !== undefined || value.progress !== undefined, {
      message: "progress requires note or progress",
    }),
]);

const identityGrowthMarkerBodySchema = z
  .object({
    description: textFieldSchema,
    source: optionalTextFieldSchema,
  })
  .strict();

const openQuestionPatchBodySchema = z.discriminatedUnion("action", [
  z
    .object({
      action: z.literal("resolve"),
      resolution: textFieldSchema,
    })
    .strict(),
  z
    .object({
      action: z.literal("abandon"),
      reason: textFieldSchema,
    })
    .strict(),
  z
    .object({
      action: z.literal("bump"),
      delta: z.number().min(-1).max(1).optional(),
    })
    .strict(),
]);

const reviewPatchBodySchema = z
  .object({
    action: z.literal("dismiss"),
    note: optionalTextFieldSchema,
  })
  .strict();

const reviewQuerySchema = z
  .object({
    open_only: z
      .enum(["true", "false"])
      .optional()
      .transform((value) => value !== "false"),
    kind: z.enum(REVIEW_KINDS).optional(),
  })
  .strict();

const reviewGenericPatchBodySchema = z
  .object({
    action: z.enum(REVIEW_RESOLUTIONS),
    note: optionalTextFieldSchema,
    winner_node_id: semanticNodeIdSchema.optional(),
  })
  .strict();

const correctionCorrectBodySchema = z
  .object({
    patch: z.record(z.string(), z.unknown()),
    reason: optionalTextFieldSchema,
  })
  .strict();

const correctionSemanticEdgeInvalidateBodySchema = z
  .object({
    at: z.number().finite().optional(),
    reason: optionalTextFieldSchema,
  })
  .strict();

const correctionReviewPatchBodySchema = z
  .object({
    action: z.enum(["accept", "reject"]),
    note: optionalTextFieldSchema,
  })
  .strict();

const creatorDirectiveParamSchema = z.object({
  id: creatorDirectiveIdSchema,
});

const creatorDirectiveRevokeBodySchema = z
  .object({
    reason: textFieldSchema,
  })
  .strict();

const creatorDirectiveSupersedeBodySchema = z
  .object({
    replacement_id: creatorDirectiveIdSchema,
  })
  .strict();

const creatorDirectiveReconciliationActionBodySchema = z.discriminatedUnion("action", [
  z
    .object({
      action: z.literal("supersede"),
      survivor_id: creatorDirectiveIdSchema,
      reason: optionalTextFieldSchema,
    })
    .strict(),
  z
    .object({
      action: z.literal("keep"),
      reason: optionalTextFieldSchema,
    })
    .strict(),
]);

const goalParamSchema = z.object({
  id: z.string().transform((value, ctx) => {
    try {
      return parseGoalId(value);
    } catch {
      ctx.addIssue({ code: "custom", message: "Invalid goal id" });
      return z.NEVER;
    }
  }),
});

const openQuestionParamSchema = z.object({
  id: z.string().transform((value, ctx) => {
    try {
      return parseOpenQuestionId(value);
    } catch {
      ctx.addIssue({ code: "custom", message: "Invalid open question id" });
      return z.NEVER;
    }
  }),
});

const reviewParamSchema = z.object({
  id: z.coerce.number().int().positive(),
});

const DEFAULT_OPEN_QUESTION_BUMP_DELTA = 0.1;
const DEFAULT_GROWTH_MARKER_CATEGORY = "understanding";
const RESET_CONFIRM_TOKEN = "RESET";
const BORG_UNAVAILABLE_MESSAGE = "Borg is unavailable after a failed reset; retry /api/admin/reset";
export const DEMO_DEFAULT_AUDIENCE_LABEL = "alice";
export const DEMO_DEFAULT_CREATOR_ENTITY_NAME = "Tom";
const DEMO_SOURCE_TYPE = "demo";
const DEMO_CONVERSATION_KIND = "demo";
const DEMO_DEFAULT_SESSION_LABEL = "demo (default)";
const DEMO_OPERATOR_SESSION_EXTERNAL_ID = "operator";
const DEMO_OPERATOR_SESSION_LABEL = "operator chat";

const promptKeyParamSchema = z.enum(PROMPT_KEYS);
const promptPutBodySchema = z
  .object({
    text: z.string().trim().min(1).max(50_000),
  })
  .strict();
const resetBodySchema = z
  .object({
    confirm: z.literal(RESET_CONFIRM_TOKEN),
  })
  .strict();

function parseRequest<T>(schema: z.ZodType<T>, value: unknown): T {
  const parsed = schema.safeParse(value);

  if (!parsed.success) {
    throw new HTTPException(400, { message: parsed.error.message });
  }

  return parsed.data;
}

function parseOptionalSessionQuery(value: string | undefined, ctx: z.RefinementCtx): SessionId {
  if (value === undefined || value.length === 0) {
    return DEFAULT_SESSION_ID;
  }

  try {
    return parseSessionId(value);
  } catch {
    ctx.addIssue({ code: "custom", message: "Invalid session id" });
    return z.NEVER;
  }
}

function demoSessionLabel(sessionId: SessionId): string {
  return sessionId === DEFAULT_SESSION_ID ? DEMO_DEFAULT_SESSION_LABEL : `demo (${sessionId})`;
}

export function ensureDemoSession(
  borg: Borg,
  input: {
    sessionId: SessionId;
    audienceLabel?: string;
    audienceEntityId?: EntityId | null;
    audienceRole?: "participant" | "operator";
    label?: string;
    sourceExternalId?: string | null;
  },
) {
  return borg.sessions.ensure({
    session_id: input.sessionId,
    source_type: DEMO_SOURCE_TYPE,
    source_external_id: input.sourceExternalId ?? null,
    source_url: null,
    label: input.label ?? demoSessionLabel(input.sessionId),
    audience_label: input.audienceLabel ?? DEMO_DEFAULT_AUDIENCE_LABEL,
    audience_entity_id: input.audienceEntityId ?? null,
    conversation_kind: DEMO_CONVERSATION_KIND,
    ...(input.audienceRole === undefined ? {} : { audience_role: input.audienceRole }),
  });
}

export function ensureDemoCreator(borg: Borg, name = DEMO_DEFAULT_CREATOR_ENTITY_NAME) {
  const existing = borg.entities.getCreator();

  if (existing !== null) {
    return existing;
  }

  const entityId = borg.entities.resolve(name, {
    kind: "person",
    provenance: "config_default_user",
  });
  const creator = borg.entities.setBorgRole(entityId, "creator");

  if (creator === null) {
    throw new HTTPException(500, { message: "Failed to initialize demo creator" });
  }

  return creator;
}

export function ensureDemoDefaultSession(
  borg: Borg,
  options: { demoCreatorEntityName?: string } = {},
) {
  ensureDemoCreator(borg, options.demoCreatorEntityName ?? DEMO_DEFAULT_CREATOR_ENTITY_NAME);

  return ensureDemoSession(borg, {
    sessionId: DEFAULT_SESSION_ID,
    audienceLabel: DEMO_DEFAULT_AUDIENCE_LABEL,
  });
}

export function ensureDemoOperatorSession(borg: Borg) {
  const creator = borg.entities.getCreator();

  if (creator === null) {
    throw new HTTPException(409, { message: "Mark a creator first" });
  }

  const existing = borg.sessions
    .list({ limit: 1000 })
    .find((session) => session.audience_role === "operator");

  if (existing !== undefined) {
    return existing;
  }

  return ensureDemoSession(borg, {
    sessionId: createSessionId(),
    audienceLabel: creator.canonical_name,
    audienceEntityId: creator.id,
    audienceRole: "operator",
    label: DEMO_OPERATOR_SESSION_LABEL,
    sourceExternalId: DEMO_OPERATOR_SESSION_EXTERNAL_ID,
  });
}

function parseKnownEntityId(borg: Borg, value: string, label: string): EntityId {
  const entityId = parseRequest(entityParamSchema, { id: value }).id;

  if (borg.entities.get(entityId) === null) {
    throw new HTTPException(400, { message: `${label} entity not found` });
  }

  return entityId;
}

function resolveTurnAudienceEntityId(
  borg: Borg,
  input: {
    audienceLabel: string;
    explicitAudienceEntityId?: string;
    existingAudienceEntityId?: EntityId | null;
  },
): EntityId | null {
  if (input.explicitAudienceEntityId !== undefined) {
    return parseKnownEntityId(borg, input.explicitAudienceEntityId, "audience");
  }

  if (input.existingAudienceEntityId !== undefined && input.existingAudienceEntityId !== null) {
    if (borg.entities.get(input.existingAudienceEntityId) === null) {
      throw new HTTPException(400, { message: "audience entity not found" });
    }

    return input.existingAudienceEntityId;
  }

  return borg.entities.resolve(input.audienceLabel, {
    kind: "person",
    provenance: "transport_audience_label",
  });
}

function resolveTurnSenderEntityId(
  borg: Borg,
  input: {
    explicitSenderEntityId?: string;
    audienceEntityId: EntityId | null;
    demoCreatorEntityName: string;
  },
): EntityId {
  if (input.explicitSenderEntityId !== undefined) {
    return parseKnownEntityId(borg, input.explicitSenderEntityId, "sender");
  }

  const audienceEntity =
    input.audienceEntityId === null ? null : borg.entities.get(input.audienceEntityId);

  if (audienceEntity !== null && audienceEntity.kind !== "group") {
    return audienceEntity.id;
  }

  return ensureDemoCreator(borg, input.demoCreatorEntityName).id;
}

async function parseJsonBody(c: Context): Promise<unknown> {
  try {
    return await c.req.json();
  } catch {
    throw new HTTPException(400, { message: "Malformed JSON body" });
  }
}

type ParsedTurnBody = z.infer<typeof turnBodySchema> & {
  attachments: TurnInputAttachment[];
};

function isMultipartRequest(c: Context): boolean {
  const contentType = c.req.header("content-type") ?? "";
  return contentType.split(";")[0]?.trim().toLowerCase() === "multipart/form-data";
}

function optionalFormValue(value: ReturnType<FormData["get"]>) {
  return value === null || value === "" ? undefined : value;
}

async function parseMultipartAttachments(formData: FormData): Promise<TurnInputAttachment[]> {
  const files = [...formData.getAll("attachments[]"), ...formData.getAll("attachments")];
  const attachments: TurnInputAttachment[] = [];

  for (const value of files) {
    if (typeof value === "string") {
      throw new HTTPException(400, { message: "attachments must be image files" });
    }

    const mediaType = parseRequest(turnAttachmentMediaTypeSchema, value.type);
    if (value.size > DEMO_TURN_ATTACHMENT_MAX_BYTES) {
      throw new HTTPException(400, {
        message: `image attachment exceeds ${DEMO_TURN_ATTACHMENT_MAX_BYTES} bytes`,
      });
    }

    attachments.push({
      mediaType,
      bytes: new Uint8Array(await value.arrayBuffer()),
    });
  }

  return attachments;
}

async function parseTurnBody(c: Context): Promise<ParsedTurnBody> {
  if (!isMultipartRequest(c)) {
    return {
      ...parseRequest(turnBodySchema, await parseJsonBody(c)),
      attachments: [],
    };
  }

  let formData: FormData;
  try {
    formData = await c.req.formData();
  } catch {
    throw new HTTPException(400, { message: "Malformed multipart body" });
  }

  const body = parseRequest(turnBodySchema, {
    message: formData.get("message"),
    external_message_id: formData.get("external_message_id"),
    audience: optionalFormValue(formData.get("audience")),
    audience_entity_id: optionalFormValue(formData.get("audience_entity_id")),
    sender_entity_id: optionalFormValue(formData.get("sender_entity_id")),
    session: optionalFormValue(formData.get("session")),
  });

  return {
    ...body,
    attachments: await parseMultipartAttachments(formData),
  };
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

function mapBorgErrorToHttp(error: unknown): never {
  if (error instanceof BorgError) {
    const status =
      error.code.endsWith("_NOT_FOUND")
        ? 404
        : error.code === "MAINTENANCE_REVERSER_MISSING" || error.code.endsWith("_STALE")
          ? 409
          : 400;
    throw new HTTPException(status, { message: error.message });
  }

  throw error;
}

function requireIdentityApplied<T>(
  result:
    | {
        status: "applied";
        record: T;
      }
    | {
        status: "requires_review";
        current: T;
      },
  action: string,
): T {
  if (result.status === "applied") {
    return result.record;
  }

  throw new HTTPException(400, { message: `${action} requires identity review` });
}

function dreamPlanProcessSummary(result: OrchestratorResult["results"][number]): string {
  if (result.changes.length === 0 && result.errors.length === 0) {
    return "no changes";
  }

  const changeCount = `${result.changes.length} ${
    result.changes.length === 1 ? "change" : "changes"
  }`;

  if (result.errors.length === 0) {
    return changeCount;
  }

  return `${changeCount}, ${result.errors.length} ${
    result.errors.length === 1 ? "error" : "errors"
  }`;
}

function mapDreamPreview(planId: string, preview: OrchestratorResult) {
  return {
    plan_id: planId,
    processes: preview.results.map((result) => ({
      name: result.process,
      would_change: result.changes.length > 0,
      summary: dreamPlanProcessSummary(result),
      budget_used: result.tokens_used,
      changes: result.changes,
      errors: result.errors,
      budget_exhausted: result.budget_exhausted,
    })),
    total_budget_used: preview.tokens_used,
    changes: preview.changes.length,
  };
}

function mapDreamApply(
  result: OrchestratorResult,
  beforeAuditIds: ReadonlySet<number>,
  afterAuditRows: ReadonlyArray<{ id: number; process: string }>,
  durationMs: number,
) {
  const auditIdsByProcess = new Map<OfflineProcessName, number[]>();

  for (const row of afterAuditRows) {
    if (
      !beforeAuditIds.has(row.id) &&
      OFFLINE_PROCESS_NAMES.includes(row.process as OfflineProcessName)
    ) {
      const process = row.process as OfflineProcessName;
      auditIdsByProcess.set(process, [...(auditIdsByProcess.get(process) ?? []), row.id]);
    }
  }

  return {
    run_id: result.run_id,
    applied: result.results
      .filter((processResult) => processResult.errors.length === 0)
      .map((processResult) => {
        const auditIds = auditIdsByProcess.get(processResult.process) ?? [];
        return {
          name: processResult.process,
          audit_id: auditIds[0] ?? null,
          audit_ids: auditIds,
          changes: processResult.changes.length,
        };
      }),
    failed: result.errors.map((error) => ({
      name: error.process,
      message: error.message,
      ...(error.code === undefined ? {} : { code: error.code }),
    })),
    duration_ms: Math.round(durationMs),
    total_budget_used: result.tokens_used,
  };
}

function progressNote(input: { note?: string; progress?: number }): string {
  if (input.progress === undefined) {
    return input.note ?? "progress updated";
  }

  const progress = `progress ${input.progress}%`;
  return input.note === undefined ? progress : `${progress}: ${input.note}`;
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

function isDemoTerminalEntry(entry: StreamEntry): entry is StreamEntry & { turn_id: string } {
  return (
    (entry.kind === "agent_msg" ||
      entry.kind === "agent_suppressed" ||
      entry.kind === "agent_observed") &&
    typeof entry.turn_id === "string" &&
    entry.turn_id.length > 0
  );
}

function updateDemoSessionLastTurnIds(borg: Borg, entries: readonly StreamEntry[]): void {
  for (const entry of entries) {
    if (!isDemoTerminalEntry(entry)) {
      continue;
    }

    borg.sessions.touch(entry.session_id, {
      lastTurnId: entry.turn_id,
      messageCountDelta: 0,
    });
  }
}

async function readStream(input: {
  borg: Borg;
  sessionId: SessionId;
  kinds?: readonly StreamEntryKind[];
  audience?: string;
  limit: number;
  before?: StreamCursor;
}): Promise<{ entries: StreamEntry[]; next_cursor: string | null }> {
  const collected: StreamEntry[] = [];
  const reader = input.borg.stream.reader({ session: input.sessionId });

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

async function countTurns(borg: Borg, sessionId: SessionId): Promise<number> {
  let count = 0;

  for await (const entry of borg.stream
    .reader({ session: sessionId })
    .iterate({ kinds: ["user_msg"] })) {
    if (entry.turn_status !== "aborted") {
      count += 1;
    }
  }

  return count;
}

function listAudiences(borg: Borg, sessionId: SessionId): string[] {
  return [
    ...new Set(
      borg.stream
        .tail(500, { session: sessionId })
        .flatMap((entry) => (entry.audience === undefined ? [] : [entry.audience])),
    ),
  ].sort();
}

function sumRecord(record: Record<string, number>): number {
  return Object.values(record).reduce((sum, value) => sum + value, 0);
}

function sparkFrom(count: number): number[] {
  if (count <= 0) {
    return Array.from({ length: 15 }, () => 0);
  }
  const base = Math.max(1, Math.min(12, count));
  return Array.from({ length: 15 }, (_, index) =>
    Math.max(1, Math.round(base * (0.45 + index / 20))),
  );
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

function creatorDirectiveText(record: CreatorDirective): string | null {
  return record.operational_directive ?? record.canonical_fact;
}

function mapCreatorDirective(borg: Borg, record: CreatorDirective) {
  return {
    id: record.id,
    kind: record.kind,
    text: creatorDirectiveText(record),
    canonical_fact: record.canonical_fact,
    operational_directive: record.operational_directive,
    activation_scope: record.activation_policy.scope,
    activation_allowed_entity_ids: record.activation_policy.allowed_entity_ids,
    activation_excluded_entity_ids: record.activation_policy.excluded_entity_ids,
    content_scope: record.disclosure_policy.content_scope,
    mention_policy: record.disclosure_policy.mention_policy,
    status: record.status,
    subject_kind: record.subject_kind,
    subject_entity_id: record.subject_entity_id,
    subject_entity_name: entityLabel(borg, record.subject_entity_id),
    priority: record.priority,
    created_at: record.created_at,
  };
}

function mapEpisode(
  borg: Borg,
  item: Awaited<ReturnType<Borg["episodic"]["list"]>>["items"][number],
) {
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

function mapMaintenanceAuditRow(row: DemoMaintenanceAuditRow) {
  return {
    id: row.id,
    run_id: row.run_id,
    process: row.process,
    action: row.action,
    targets: row.targets,
    reversal: row.reversal,
    applied_at: row.applied_at,
    reverted_at: row.reverted_at,
    reverted_by: row.reverted_by,
  };
}

function openReviewListOptions(kind?: ReviewKind): { kind?: ReviewKind; openOnly: true } {
  return kind === undefined ? { openOnly: true } : { kind, openOnly: true };
}

function findOpenReviewItem(borg: Borg, id: number, kind?: ReviewKind): ReviewQueueItem | null {
  return borg.review.list(openReviewListOptions(kind)).find((row) => row.id === id) ?? null;
}

function parseCreatorDirectiveReconciliationRefs(row: ReviewQueueItem) {
  const parsed = creatorDirectiveReconciliationReviewRefsSchema.safeParse(row.refs);

  if (!parsed.success) {
    throw new HTTPException(400, {
      message: `creator directive reconciliation review refs are invalid: ${parsed.error.message}`,
    });
  }

  return parsed.data;
}

function assertCreatorDirectiveReviewMembers(
  refs: ReturnType<typeof parseCreatorDirectiveReconciliationRefs>,
  ids: readonly string[],
  label: string,
): void {
  const memberIds = new Set<string>(refs.directive_ids);
  const seen = new Set<string>();

  for (const id of ids) {
    if (!memberIds.has(id)) {
      throw new HTTPException(400, {
        message: `${label} must reference creator directives in the review item`,
      });
    }

    if (seen.has(id)) {
      throw new HTTPException(400, {
        message: `${label} must not contain duplicate directive ids`,
      });
    }

    seen.add(id);
  }
}

function loadCreatorDirectiveReviewMembers(
  borg: Borg,
  refs: ReturnType<typeof parseCreatorDirectiveReconciliationRefs>,
): Map<string, CreatorDirective> {
  const members = new Map<string, CreatorDirective>();

  for (const id of refs.directive_ids) {
    const directive = borg.creatorDirectives.get(id);

    if (directive === null) {
      throw new HTTPException(404, { message: `creator directive ${id} not found` });
    }

    members.set(id, directive);
  }

  return members;
}

function requireCreatorDirectiveReviewMember(
  members: ReadonlyMap<string, CreatorDirective>,
  id: string,
): CreatorDirective {
  const directive = members.get(id);

  if (directive === undefined) {
    throw new HTTPException(400, {
      message: "directive id must reference creator directives in the review item",
    });
  }

  return directive;
}

function assertCreatorDirectiveActive(directive: CreatorDirective): void {
  if (directive.status !== "active") {
    throw new HTTPException(409, {
      message: `creator directive ${directive.id} changed or inactive`,
    });
  }
}

type SemanticGraphNodeStatus = "active" | "contested" | "contradicted" | "quarantined";

function mapSemanticGraphStatus(status: SemanticNodeStatus): SemanticGraphNodeStatus {
  if (status === "superseded") {
    return "contested";
  }

  return status;
}

async function semanticGraphSnapshot(borg: Borg, limit: number) {
  const statusCounts = borg.semantic.nodes.countByStatus();
  const totalNodes = sumRecord(statusCounts);
  const nodes = totalNodes === 0 ? [] : await borg.semantic.nodes.list({ limit: totalNodes });
  const edges = borg.semantic.edges.list();
  const edgeCounts = new Map<string, number>();

  for (const edge of edges) {
    edgeCounts.set(edge.from_node_id, (edgeCounts.get(edge.from_node_id) ?? 0) + 1);
    edgeCounts.set(edge.to_node_id, (edgeCounts.get(edge.to_node_id) ?? 0) + 1);
  }

  const selectedNodes = nodes
    .map((node) => ({
      node,
      edgeCount: edgeCounts.get(node.id) ?? 0,
    }))
    .sort(
      (left, right) =>
        right.edgeCount - left.edgeCount || right.node.updated_at - left.node.updated_at,
    )
    .slice(0, limit);
  const selectedIds = new Set(selectedNodes.map((entry) => entry.node.id));
  const selectedEdges = edges.filter(
    (edge) => selectedIds.has(edge.from_node_id) && selectedIds.has(edge.to_node_id),
  );

  return {
    nodes: selectedNodes.map(({ node, edgeCount }) => mapSemanticGraphNode(node, edgeCount)),
    edges: selectedEdges.map((edge) => mapSemanticGraphEdge(edge)),
    total_nodes: totalNodes,
    total_edges: edges.length,
    rendered: {
      nodes: selectedNodes.length,
      edges: selectedEdges.length,
    },
  };
}

function mapSemanticGraphNode(node: SemanticNode, edgeCount: number) {
  return {
    id: node.id,
    label: node.label,
    status: mapSemanticGraphStatus(node.status),
    kind: node.kind,
    edge_count: edgeCount,
  };
}

function mapSemanticGraphEdge(edge: SemanticEdge) {
  return {
    id: edge.id,
    source: edge.from_node_id,
    target: edge.to_node_id,
    type: edge.relation,
    weight: edge.confidence,
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
    "creator-directive-reconciler": "reconcile redundant or conflicting creator directives",
    ruminator: "open-question rumination",
    "self-narrator": "autobiography and growth markers",
    "procedural-synthesizer": "skill abstractions",
    "belief-reviser": "invalidate, weaken, contradict",
  };

  return descriptions[name];
}

function streamDreamProcesses(entry: StreamEntry): OfflineProcessName[] {
  if (
    entry.kind !== "dream_report" ||
    entry.content === null ||
    typeof entry.content !== "object"
  ) {
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

function dreamScheduleFromAudit(
  rows: ReadonlyArray<Pick<MaintenanceAuditRecord, "id" | "applied_at"> & { process: string }>,
) {
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
  const streamMatches = dreamReports.filter((entry) =>
    streamDreamProcesses(entry).includes(process),
  );
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
      latestStream === undefined
        ? "ok"
        : streamDreamHasProcessError(latestStream, process)
          ? "error"
          : "ok",
    last_audit_id: latestAudit?.id ?? null,
  };
}

async function memoryBands(borg: Borg, sessionId: SessionId) {
  const episodes = await borg.episodic.list({ limit: 500 });
  const semanticCounts = borg.semantic.nodes.countByStatus();
  const procedural = borg.skills.list(500);
  const moodHistory = borg.mood.history(sessionId, { limit: 500 });
  const values = borg.self.values.list();
  const goals = borg.self.goals.list();
  const traits = borg.self.traits.list();
  const openQuestions = borg.self.openQuestions.list({ status: "open" });
  const growthMarkers = borg.self.growthMarkers.list({ limit: 500 });
  const periods = borg.self.autobiographical.listPeriods({ limit: 500 });
  const relationalCounts = borg.relationalSlots.countByState();

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
      growth: sparkFrom(borg.social.list(500).length),
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

async function pendingSemanticExtractionEpisodes(borg: Borg): Promise<number> {
  const episodic = borg.episodic as SemanticExtractionEpisodicFacade;
  const processed = new Set<Episode["id"]>();
  const [episodes, semanticNodes] = await Promise.all([
    episodic.listAll(),
    borg.semantic.nodes.list({ includeArchived: true, limit: 100_000 }),
  ]);

  for (const node of semanticNodes) {
    for (const episodeId of node.source_episode_ids) {
      processed.add(episodeId);
    }
  }

  for (const edge of borg.semantic.edges.list({ includeInvalid: true })) {
    for (const episodeId of edge.evidence_episode_ids) {
      processed.add(episodeId);
    }
  }

  for (const audit of borg.audit.list({ process: "semantic-extractor", reverted: false })) {
    const episodeIds = audit.targets.episode_ids;

    if (!Array.isArray(episodeIds)) {
      continue;
    }

    for (const episodeId of episodeIds) {
      if (typeof episodeId === "string") {
        processed.add(episodeId as Episode["id"]);
      }
    }
  }

  let pending = 0;

  for (const episode of episodes) {
    if (episodic.getStats(episode.id)?.archived === true) {
      continue;
    }

    if (!processed.has(episode.id)) {
      pending += 1;
    }
  }

  return pending;
}

async function dreamState(borg: Borg) {
  const auditRows = borg.audit.list().slice(0, 50);
  const dreamReports = borg.stream.tail(500).filter((entry) => entry.kind === "dream_report");
  const config = borg.maintenance.config();
  const pendingExtractionEpisodes = await pendingSemanticExtractionEpisodes(borg);

  const processes = OFFLINE_PROCESS_NAMES.map((name) => {
    const lastRun = latestDreamRunForProcess(name, dreamReports, auditRows);

    return {
      name,
      description: processDescription(name),
      last_run_at: lastRun.last_run_at,
      last_status: lastRun.last_status,
      last_audit_id: lastRun.last_audit_id,
      budget: config.processBudgets[name] ?? null,
      enabled: config.lightProcesses.includes(name) || config.heavyProcesses.includes(name),
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
    pending_extraction_episodes: pendingExtractionEpisodes,
    schedule: [...streamSchedule, ...dreamScheduleFromAudit(auditRows)]
      .sort((left, right) => right.scheduled_at - left.scheduled_at)
      .slice(0, 80),
    audit_rows: auditRows.map((row) => mapMaintenanceAuditRow(row)),
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

function maintenanceResultProcessNames(result: OrchestratorResult | null): OfflineProcessName[] {
  return result?.results.map((processResult) => processResult.process) ?? [];
}

function errorReason(error: unknown): string {
  return error instanceof Error ? error.message : String(error);
}

export async function broadcastMaintenanceTick(input: {
  borg: Borg;
  live: LiveBridge;
  cadence: MaintenanceCadence | "manual";
  status: MaintenanceTickFrameStatus;
  result: OrchestratorResult | null;
  ts?: number;
  durationMs?: number;
  errors?: number;
  reason?: string;
}): Promise<void> {
  const pendingExtractionEpisodes = await pendingSemanticExtractionEpisodes(input.borg);
  const changes = input.result?.changes.length ?? 0;
  const errors = input.errors ?? input.result?.errors.length ?? 0;
  const frame: MaintenanceTickFrame = {
    type: "maintenance:tick",
    ts: input.ts ?? Date.now(),
    cadence: input.cadence,
    status: input.status,
    processes: maintenanceResultProcessNames(input.result),
    changed: changes > 0,
    changes,
    errors,
    pending_extraction_episodes: pendingExtractionEpisodes,
    ...(input.result === null ? {} : { run_id: input.result.run_id }),
    ...(input.durationMs === undefined ? {} : { duration_ms: Math.round(input.durationMs) }),
    ...(input.reason === undefined ? {} : { reason: input.reason }),
  };

  input.live.broadcaster.broadcast(frame);
}

export function wireMaintenanceSchedulerLiveObserver(borg: Borg, live: LiveBridge): void {
  borg.maintenance.scheduler.setObserver({
    onTick: (result: MaintenanceTickResult) =>
      broadcastMaintenanceTick({
        borg,
        live,
        cadence: result.cadence,
        status: result.status,
        result: result.result,
        ts: result.ts,
        ...(result.reason === undefined ? {} : { reason: result.reason }),
      }),
    onError: (error: unknown, cadence: MaintenanceCadence) =>
      broadcastMaintenanceTick({
        borg,
        live,
        cadence,
        status: "error",
        result: null,
        errors: 1,
        reason: errorReason(error),
      }),
  });
}

export type DemoServerAppInput = {
  borgHandle: BorgHandle;
  live: LiveBridge;
  corsOrigins?: readonly string[];
  resetBorg?: () => Promise<void>;
  requestGate?: BorgRequestGate;
  demoCreatorEntityName?: string;
};

type BorgRequestGateLease = {
  release(): void;
};

export class BorgRequestGate {
  private inflight = 0;
  private resetting = false;

  acquire(): BorgRequestGateLease {
    if (this.resetting) {
      throw new HTTPException(503, { message: "Borg reset in progress" });
    }

    this.inflight += 1;
    let released = false;

    return {
      release: () => {
        if (released) {
          return;
        }

        released = true;
        this.inflight = Math.max(0, this.inflight - 1);
      },
    };
  }

  beginReset(): BorgRequestGateLease {
    if (this.resetting) {
      throw new HTTPException(409, { message: "Borg reset already in progress" });
    }

    if (this.inflight > 0) {
      throw new HTTPException(409, { message: "Borg is busy" });
    }

    this.resetting = true;
    let released = false;

    return {
      release: () => {
        if (released) {
          return;
        }

        released = true;
        this.resetting = false;
      },
    };
  }
}

export function createDemoServerApp(args: DemoServerAppInput) {
  const input = {
    get borg(): Borg {
      if (args.borgHandle.state === "dead" || args.borgHandle.state === "closing") {
        throw new HTTPException(503, { message: BORG_UNAVAILABLE_MESSAGE });
      }

      return args.borgHandle.current;
    },
    live: args.live,
    corsOrigins: args.corsOrigins,
    resetBorg: args.resetBorg,
    requestGate: args.requestGate ?? new BorgRequestGate(),
    demoCreatorEntityName: args.demoCreatorEntityName ?? DEMO_DEFAULT_CREATOR_ENTITY_NAME,
  };
  const app = new Hono();
  const { injectWebSocket, upgradeWebSocket } = createNodeWebSocket({ app });
  ensureDemoDefaultSession(input.borg, {
    demoCreatorEntityName: input.demoCreatorEntityName,
  });
  input.live.observeStreamAppend((entries) => {
    updateDemoSessionLastTurnIds(input.borg, entries);
  });
  const allowedOrigins = input.corsOrigins ?? ["http://localhost:5173"];
  const dreamPlans = new Map<
    string,
    { plan: MaintenancePlan; applied?: ReturnType<typeof mapDreamApply> }
  >();
  let dreamPlanCounter = 0;
  let commitmentFamilyCounter = 0;

  function clearAppCaches(): void {
    dreamPlans.clear();
    dreamPlanCounter = 0;
    commitmentFamilyCounter = 0;
  }

  function nextDreamPlanId(): string {
    dreamPlanCounter += 1;
    return `demo_plan_${Date.now()}_${dreamPlanCounter}`;
  }

  function nextOperatorDirectiveFamily(): string {
    commitmentFamilyCounter += 1;
    return `demo_operator_manual_${Date.now()}_${commitmentFamilyCounter}`;
  }

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

  app.use("/api/*", async (c, next) => {
    const pathname = new URL(c.req.url).pathname;
    if (pathname === "/api/live" || pathname === "/api/admin/reset") {
      return next();
    }

    const lease = input.requestGate.acquire();
    try {
      await next();
    } finally {
      lease.release();
    }
  });

  app.get(
    "/api/live",
    upgradeWebSocket(() => ({
      onOpen: (_event, ws) => input.live.broadcaster.add(ws),
      onMessage: (event, ws) => input.live.broadcaster.handleSubscriptionMessage(ws, event.data),
      onClose: (_event, ws) => input.live.broadcaster.remove(ws),
      onError: (_event, ws) => input.live.broadcaster.remove(ws),
    })),
  );

  app.get("/api/state", async (c) => {
    const query = parseRequest(sessionQuerySchema, c.req.query());
    const auditRows = input.borg.audit.list();

    return c.json({
      active_session: query.session,
      audiences: listAudiences(input.borg, query.session),
      counts: {
        turns: await countTurns(input.borg, query.session),
        commitments: input.borg.commitments.countActive(),
        open_qs: input.borg.self.openQuestions.list({ status: "open" }).length,
        open_reviews: input.borg.review.list({ openOnly: true }).length,
        dream_audit_rows: auditRows.length,
      },
      current_mood: input.borg.mood.current(query.session),
      version: VERSION,
    });
  });

  app.get("/api/sessions", (c) => c.json({ sessions: input.borg.sessions.list({ limit: 1000 }) }));

  app.get("/api/entities/creator", (c) => c.json(input.borg.entities.getCreator()));

  app.post("/api/entities/creator", async (c) => {
    const body = parseRequest(creatorNameBodySchema, await parseJsonBody(c));
    const entityId = input.borg.entities.resolve(body.name, {
      kind: "person",
      provenance: "user_declared",
    });
    const entity = input.borg.entities.setBorgRole(entityId, "creator");

    if (entity === null) {
      throw new HTTPException(404, { message: "Entity not found" });
    }

    return c.json(entity);
  });

  app.post("/api/entities/:id/borg-role", async (c) => {
    const params = parseRequest(entityParamSchema, c.req.param());
    const body = parseRequest(entityBorgRoleBodySchema, await parseJsonBody(c));
    const entity = input.borg.entities.setBorgRole(params.id, body.role);

    if (entity === null) {
      throw new HTTPException(404, { message: "Entity not found" });
    }

    return c.json(entity);
  });

  app.post("/api/entities", async (c) => {
    const body = parseRequest(entityCreateBodySchema, await parseJsonBody(c));
    const entityId = input.borg.entities.resolve(body.name, body.kind ? { kind: body.kind } : {});
    return c.json(input.borg.entities.get(entityId));
  });

  app.post("/api/sessions/operator", (c) => c.json(ensureDemoOperatorSession(input.borg)));

  app.post("/api/sessions/:id/participation", async (c) => {
    const params = parseRequest(sessionParamSchema, c.req.param());
    const body = parseRequest(sessionParticipationBodySchema, await parseJsonBody(c));

    try {
      const session = await input.borg.sessions.setParticipationPolicy(params.id, body.policy, {
        reason: body.reason,
      });

      return c.json(session);
    } catch (error) {
      mapBorgErrorToHttp(error);
    }
  });

  app.get("/api/stream", async (c) => {
    const query = parseRequest(streamQuerySchema, c.req.query());
    return c.json(
      await readStream({
        borg: input.borg,
        sessionId: query.session,
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

  app.get("/api/memory/bands", async (c) => {
    const query = parseRequest(sessionQuerySchema, c.req.query());
    return c.json({ bands: await memoryBands(input.borg, query.session) });
  });

  app.get("/api/semantic/graph", async (c) => {
    const query = parseRequest(semanticGraphQuerySchema, c.req.query());
    return c.json(await semanticGraphSnapshot(input.borg, query.limit));
  });

  app.get("/api/memory/bands/:id", async (c) => {
    const band = parseRequest(memoryBandIdSchema, c.req.param("id"));
    const query = parseRequest(sessionQuerySchema, c.req.query());

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
        current: input.borg.mood.current(query.session),
        history: input.borg.mood.history(query.session, { limit: 100 }),
      });
    }

    if (band === "commitments") {
      return c.json({
        band,
        items: input.borg.commitments
          .list({ activeOnly: false })
          .map((record) => mapCommitment(input.borg, record)),
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
        (record) =>
          query.enforcement === undefined || record.enforcement_class === query.enforcement,
      );

    return c.json({ commitments });
  });

  app.get("/api/creator-directives", (c) => {
    const directives = input.borg.creatorDirectives
      .list({ status: "active" })
      .map((record) => mapCreatorDirective(input.borg, record));

    return c.json({ directives });
  });

  app.post("/api/creator-directives/:id/revoke", async (c) => {
    const params = parseRequest(creatorDirectiveParamSchema, c.req.param());
    const body = parseRequest(creatorDirectiveRevokeBodySchema, await parseJsonBody(c));

    try {
      const directive = input.borg.creatorDirectives.revoke(params.id, body.reason);

      if (directive === null) {
        throw new HTTPException(404, { message: "creator directive not found" });
      }

      return c.json(mapCreatorDirective(input.borg, directive));
    } catch (error) {
      mapBorgErrorToHttp(error);
    }
  });

  app.post("/api/creator-directives/:id/supersede", async (c) => {
    const params = parseRequest(creatorDirectiveParamSchema, c.req.param());
    const body = parseRequest(creatorDirectiveSupersedeBodySchema, await parseJsonBody(c));

    try {
      if (params.id === body.replacement_id) {
        throw new HTTPException(400, {
          message: "replacement creator directive must be different from the superseded directive",
        });
      }

      const replacement = input.borg.creatorDirectives.get(body.replacement_id);

      if (replacement === null) {
        throw new HTTPException(404, { message: "replacement creator directive not found" });
      }

      if (replacement.status !== "active") {
        throw new HTTPException(400, {
          message: "replacement creator directive must be active",
        });
      }

      const directive = input.borg.creatorDirectives.supersede(params.id, body.replacement_id);

      if (directive === null) {
        throw new HTTPException(404, { message: "creator directive not found" });
      }

      return c.json(mapCreatorDirective(input.borg, directive));
    } catch (error) {
      mapBorgErrorToHttp(error);
    }
  });

  app.get("/api/reviews", (c) => {
    const query = parseRequest(reviewQuerySchema, c.req.query());
    const options: { kind?: ReviewKind; openOnly?: boolean } =
      query.kind === undefined ? {} : { kind: query.kind };
    if (query.open_only) {
      options.openOnly = true;
    }

    return c.json({
      rows: input.borg.review.list(options).map((row) => mapReviewRow(row)),
    });
  });

  app.patch("/api/reviews/:id", async (c) => {
    const params = parseRequest(reviewParamSchema, c.req.param());
    const body = parseRequest(reviewGenericPatchBodySchema, await parseJsonBody(c));

    try {
      const reviewItem = findOpenReviewItem(input.borg, params.id);

      if (reviewItem === null) {
        throw new HTTPException(404, { message: "review item not found" });
      }

      if (reviewItem.kind === "creator_directive_reconciliation") {
        throw new HTTPException(409, {
          message:
            "creator directive reconciliation reviews must be resolved through POST /api/reviews/:id/creator-directive-reconciliation",
        });
      }

      const decision: BorgReviewResolutionInput = {
        decision: body.action,
        ...(body.winner_node_id === undefined ? {} : { winner_node_id: body.winner_node_id }),
        ...(body.note === undefined ? {} : { reason: body.note }),
      };
      const resolved = await input.borg.review.resolve(params.id, decision, {
        source: "manual",
      });

      if (resolved === null) {
        throw new HTTPException(404, { message: "review item not found" });
      }

      return c.json(mapReviewRow(resolved));
    } catch (error) {
      mapBorgErrorToHttp(error);
    }
  });

  app.post("/api/reviews/:id/creator-directive-reconciliation", async (c) => {
    const params = parseRequest(reviewParamSchema, c.req.param());
    const body = parseRequest(
      creatorDirectiveReconciliationActionBodySchema,
      await parseJsonBody(c),
    );

    try {
      const reviewItem = findOpenReviewItem(
        input.borg,
        params.id,
        "creator_directive_reconciliation",
      );

      if (reviewItem === null) {
        throw new HTTPException(404, { message: "review item not found" });
      }

      const refs = parseCreatorDirectiveReconciliationRefs(reviewItem);
      const defaultReason = `creator directive reconciliation ${body.action}`;

      if (body.action === "supersede") {
        assertCreatorDirectiveReviewMembers(refs, [body.survivor_id], "survivor_id");
        const members = loadCreatorDirectiveReviewMembers(input.borg, refs);
        const survivor = requireCreatorDirectiveReviewMember(members, body.survivor_id);
        assertCreatorDirectiveActive(survivor);
        const losers = refs.directive_ids
          .filter((id) => id !== body.survivor_id)
          .map((id) => requireCreatorDirectiveReviewMember(members, id));

        for (const loser of losers) {
          assertCreatorDirectiveActive(loser);
        }

        const superseded = input.borg.creatorDirectives.supersedeFamilyAtomic({
          survivorId: survivor.id,
          expectedSurvivorVersion: survivor.record_version,
          losers: losers.map((loser) => ({
            id: loser.id,
            expectedVersion: loser.record_version,
          })),
        });

        if (superseded === null) {
          throw new HTTPException(409, {
            message: "creator directive reconciliation changed before supersede could apply",
          });
        }

        const resolved = await input.borg.review.resolve(
          params.id,
          {
            decision: "accept",
            reason: body.reason ?? defaultReason,
          },
          { source: "manual" },
        );

        if (resolved === null) {
          throw new HTTPException(404, { message: "review item not found" });
        }

        return c.json(mapReviewRow(resolved));
      }

      const resolved = await input.borg.review.resolve(
        params.id,
        {
          decision: "keep",
          reason: body.reason ?? defaultReason,
        },
        { source: "manual" },
      );

      if (resolved === null) {
        throw new HTTPException(404, { message: "review item not found" });
      }

      return c.json(mapReviewRow(resolved));
    } catch (error) {
      mapBorgErrorToHttp(error);
    }
  });

  app.post("/api/commitments", async (c) => {
    const body = parseRequest(commitmentCreateBodySchema, await parseJsonBody(c));

    try {
      const commitment = input.borg.commitments.add({
        type: body.type,
        kind: body.kind,
        // Operator standing instructions are trusted advice, not hard guard constraints.
        enforcementClass: "advisory",
        directiveFamily: body.directive_family ?? nextOperatorDirectiveFamily(),
        directive: body.directive,
        priority: body.priority,
        audience: body.audience,
        madeTo: body.made_to,
        about: body.about,
        provenance: {
          kind: "manual",
        },
        expiresAt: body.expires_at ?? null,
      });

      return c.json(mapCommitment(input.borg, commitment));
    } catch (error) {
      mapBorgErrorToHttp(error);
    }
  });

  app.post("/api/commitments/:id/revoke", async (c) => {
    const params = parseRequest(commitmentParamSchema, c.req.param());
    const body = parseRequest(commitmentRevokeBodySchema, await parseJsonBody(c));

    try {
      const commitment = input.borg.commitments.revoke(params.id, body.reason ?? "", {
        kind: "manual",
      });

      if (commitment === null) {
        throw new HTTPException(404, { message: "commitment not found" });
      }

      return c.json(mapCommitment(input.borg, commitment));
    } catch (error) {
      mapBorgErrorToHttp(error);
    }
  });

  app.get("/api/shared-state", (c) => {
    const query = parseRequest(audienceQuerySchema, c.req.query());
    const audience = query.audience ?? "self";
    return c.json({ audience, entries: input.borg.sharedState.listEntriesForAudience(audience) });
  });

  app.get("/api/identity", (c) => c.json(selfSnapshot(input.borg)));

  app.get("/api/dream/audit", (c) => {
    const query = parseRequest(auditQuerySchema, c.req.query());
    return c.json({
      rows: input.borg.audit
        .list()
        .slice(0, query.limit)
        .map((row) => mapMaintenanceAuditRow(row)),
    });
  });

  app.post("/api/dream/audit/:id/revert", async (c) => {
    const params = parseRequest(auditParamSchema, c.req.param());

    try {
      const current = input.borg.audit.list().find((row) => row.id === params.id);

      if (current === undefined) {
        throw new HTTPException(404, { message: "audit row not found" });
      }

      if (current.reverted_at !== null) {
        throw new HTTPException(409, { message: "audit row is already reverted" });
      }

      const reverted = await input.borg.audit.revert(params.id, "demo_operator");

      if (reverted === null) {
        throw new HTTPException(404, { message: "audit row not found" });
      }

      return c.json(mapMaintenanceAuditRow(reverted));
    } catch (error) {
      mapBorgErrorToHttp(error);
    }
  });

  app.get("/api/dream/state", async (c) => c.json(await dreamState(input.borg)));

  app.get("/api/correction/reviews", (c) =>
    c.json({
      rows: input.borg.review
        .list({ kind: "correction", openOnly: true })
        .map((row) => mapReviewRow(row)),
    }),
  );

  app.get("/api/correction/:id/why", async (c) => {
    try {
      return c.json(await input.borg.correction.why(c.req.param("id")));
    } catch (error) {
      mapBorgErrorToHttp(error);
    }
  });

  app.post("/api/correction/:id/forget", async (c) => {
    try {
      return c.json(await input.borg.correction.forget(c.req.param("id")));
    } catch (error) {
      mapBorgErrorToHttp(error);
    }
  });

  app.post("/api/correction/:id/correct", async (c) => {
    const body = parseRequest(correctionCorrectBodySchema, await parseJsonBody(c));

    try {
      const queued = await input.borg.correction.correct(
        c.req.param("id"),
        body.patch,
        {
          kind: "manual",
        },
        {
          reason: body.reason,
        },
      );

      return c.json(mapReviewRow(queued));
    } catch (error) {
      mapBorgErrorToHttp(error);
    }
  });

  app.post("/api/correction/semantic-edges/:id/invalidate", async (c) => {
    const body = parseRequest(correctionSemanticEdgeInvalidateBodySchema, await parseJsonBody(c));

    try {
      return c.json(
        input.borg.correction.invalidateSemanticEdge(c.req.param("id"), {
          at: body.at,
          reason: body.reason,
        }),
      );
    } catch (error) {
      mapBorgErrorToHttp(error);
    }
  });

  app.patch("/api/correction/reviews/:id", async (c) => {
    const params = parseRequest(reviewParamSchema, c.req.param());
    const body = parseRequest(correctionReviewPatchBodySchema, await parseJsonBody(c));

    try {
      const correctionReview = input.borg.review
        .list({ kind: "correction", openOnly: true })
        .find((row) => row.id === params.id);

      if (correctionReview === undefined) {
        throw new HTTPException(404, { message: "correction review item not found" });
      }

      const resolved = await input.borg.review.resolve(
        params.id,
        {
          decision: body.action,
          reason: body.note ?? `${body.action}ed from demo correction queue`,
        },
        {
          source: "manual",
        },
      );

      if (resolved === null) {
        throw new HTTPException(404, { message: "review item not found" });
      }

      return c.json(mapReviewRow(resolved));
    } catch (error) {
      mapBorgErrorToHttp(error);
    }
  });

  app.post("/api/dream/plan", async (c) => {
    const body = parseRequest(dreamPlanBodySchema, await parseJsonBody(c));

    try {
      // borg.dream.plan(...) + borg.dream.preview(...); demo v1 writes no audience-scoped state.
      const plan = await input.borg.dream.plan({
        processes: body.processes,
        budget: body.budget,
      });
      const planId = nextDreamPlanId();
      dreamPlans.set(planId, { plan });

      return c.json(mapDreamPreview(planId, input.borg.dream.preview(plan)));
    } catch (error) {
      mapBorgErrorToHttp(error);
    }
  });

  app.post("/api/dream/apply", async (c) => {
    const body = parseRequest(dreamApplyBodySchema, await parseJsonBody(c));
    const cachedPlan = body.plan_id === undefined ? undefined : dreamPlans.get(body.plan_id);

    if (body.plan_id !== undefined && cachedPlan === undefined) {
      throw new HTTPException(404, { message: "dream plan not found" });
    }

    if (cachedPlan?.applied !== undefined) {
      return c.json(cachedPlan.applied);
    }

    try {
      const plan =
        cachedPlan?.plan ??
        (await input.borg.dream.plan({
          processes: body.processes,
          budget: body.budget,
        }));
      const beforeAuditIds = new Set(input.borg.audit.list().map((row) => row.id));
      const startedAt = performance.now();
      // borg.dream.apply(...); demo v1 uses the default/global maintenance substrate.
      const result = await input.borg.dream.apply(plan);
      const durationMs = performance.now() - startedAt;
      const response = mapDreamApply(result, beforeAuditIds, input.borg.audit.list(), durationMs);

      if (body.plan_id !== undefined) {
        dreamPlans.set(body.plan_id, { plan, applied: response });
      }

      await broadcastMaintenanceTick({
        borg: input.borg,
        live: input.live,
        cadence: "manual",
        status: "ok",
        result,
        durationMs,
      });

      return c.json(response);
    } catch (error) {
      mapBorgErrorToHttp(error);
    }
  });

  app.post("/api/identity/values", async (c) => {
    const body = parseRequest(identityValueBodySchema, await parseJsonBody(c));

    try {
      // borg.self.values.add(...); demo v1 writes default/global identity scope.
      return c.json(
        input.borg.self.values.add({
          label: body.name,
          description: body.description ?? body.name,
          priority: 0,
          provenance: {
            kind: "manual",
          },
        }),
      );
    } catch (error) {
      mapBorgErrorToHttp(error);
    }
  });

  app.post("/api/identity/goals", async (c) => {
    const body = parseRequest(identityGoalBodySchema, await parseJsonBody(c));

    try {
      // borg.self.goals.add(...); demo v1 writes default/global identity scope.
      return c.json(
        input.borg.self.goals.add({
          description: body.description,
          priority: body.priority ?? 0,
          parentId: null,
          provenance: {
            kind: "manual",
          },
        }),
      );
    } catch (error) {
      mapBorgErrorToHttp(error);
    }
  });

  app.patch("/api/identity/goals/:id", async (c) => {
    const params = parseRequest(goalParamSchema, c.req.param());
    const body = parseRequest(goalPatchBodySchema, await parseJsonBody(c));

    if (input.borg.self.goals.get(params.id) === null) {
      throw new HTTPException(404, { message: "goal not found" });
    }

    try {
      // borg.self.goals.updateStatus/updateProgress(...); demo operator actions apply through review.
      if (body.action === "complete") {
        return c.json(
          requireIdentityApplied(
            input.borg.self.goals.updateStatus(
              params.id,
              "done",
              { kind: "manual" },
              { throughReview: true, reason: body.note ?? null },
            ),
            "Completing goal",
          ),
        );
      }

      if (body.action === "block") {
        return c.json(
          requireIdentityApplied(
            input.borg.self.goals.updateStatus(
              params.id,
              "blocked",
              { kind: "manual" },
              { throughReview: true, reason: body.note ?? null },
            ),
            "Blocking goal",
          ),
        );
      }

      return c.json(
        requireIdentityApplied(
          input.borg.self.goals.updateProgress(
            params.id,
            progressNote(body),
            { kind: "manual" },
            { throughReview: true, reason: body.note ?? null },
          ),
          "Updating goal progress",
        ),
      );
    } catch (error) {
      mapBorgErrorToHttp(error);
    }
  });

  app.post("/api/identity/growth-markers", async (c) => {
    const body = parseRequest(identityGrowthMarkerBodySchema, await parseJsonBody(c));

    try {
      const evidence = await input.borg.stream.append({
        kind: "internal_event",
        content: {
          event: "demo_operator.growth_marker.add",
          description: body.description,
          source: body.source ?? "manual",
        },
      });

      // borg.self.growthMarkers.add(...); demo v1 writes default/global identity scope.
      return c.json(
        input.borg.self.growthMarkers.add({
          ts: Date.now(),
          category: DEFAULT_GROWTH_MARKER_CATEGORY,
          what_changed: body.description,
          evidence_episode_ids: [evidence.id],
          confidence: 0.6,
          source_process: body.source ?? "manual",
          provenance: {
            kind: "manual",
          },
        }),
      );
    } catch (error) {
      mapBorgErrorToHttp(error);
    }
  });

  app.patch("/api/identity/open-questions/:id", async (c) => {
    const params = parseRequest(openQuestionParamSchema, c.req.param());
    const body = parseRequest(openQuestionPatchBodySchema, await parseJsonBody(c));
    const current =
      input.borg.self.openQuestions
        .list({ limit: 500 })
        .find((question) => question.id === params.id) ?? null;

    if (current === null) {
      throw new HTTPException(404, { message: "open question not found" });
    }

    if (current.status !== "open") {
      throw new HTTPException(400, { message: `open question is already ${current.status}` });
    }

    try {
      // borg.self.openQuestions.resolve/abandon/bumpUrgency(...); demo operator actions apply through review.
      if (body.action === "resolve") {
        const evidence = await input.borg.stream.append({
          kind: "internal_event",
          content: {
            event: "demo_operator.open_question.resolve",
            open_question_id: params.id,
            resolution: body.resolution,
          },
        });

        return c.json(
          requireIdentityApplied(
            input.borg.self.openQuestions.resolve(
              params.id,
              {
                resolution_evidence_stream_entry_ids: [evidence.id],
                resolution_note: body.resolution,
              },
              { kind: "manual" },
              { throughReview: true, reason: "demo operator resolution" },
            ),
            "Resolving open question",
          ),
        );
      }

      if (body.action === "abandon") {
        return c.json(
          requireIdentityApplied(
            input.borg.self.openQuestions.abandon(
              params.id,
              body.reason,
              { kind: "manual" },
              {
                throughReview: true,
                reason: body.reason,
              },
            ),
            "Abandoning open question",
          ),
        );
      }

      return c.json(
        requireIdentityApplied(
          input.borg.self.openQuestions.bumpUrgency(
            params.id,
            body.delta ?? DEFAULT_OPEN_QUESTION_BUMP_DELTA,
            { kind: "manual" },
            { throughReview: true, reason: "demo operator urgency bump" },
          ),
          "Bumping open question urgency",
        ),
      );
    } catch (error) {
      mapBorgErrorToHttp(error);
    }
  });

  app.patch("/api/dream/review/:id", async (c) => {
    const params = parseRequest(reviewParamSchema, c.req.param());
    const body = parseRequest(reviewPatchBodySchema, await parseJsonBody(c));

    try {
      // borg.review.resolve(...): belief_revision rows currently only allow the
      // "dismiss" resolution (see BELIEF_REVISION_REVIEW_RESOLUTIONS); applying a
      // revision happens through the belief-reviser apply step, not the review
      // queue. The demo's UI exposes a single dismiss action.
      const resolved = await input.borg.review.resolve(params.id, {
        decision: "dismiss",
        reason: body.note ?? "Dismissed from demo operator",
      });

      if (resolved === null) {
        throw new HTTPException(404, { message: "review item not found" });
      }

      return c.json(mapReviewRow(resolved));
    } catch (error) {
      mapBorgErrorToHttp(error);
    }
  });

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
    const body = await parseTurnBody(c);
    let sessionId: SessionId;
    try {
      sessionId = parseSessionId(body.session ?? DEFAULT_SESSION_ID);
    } catch {
      throw new HTTPException(400, { message: "Invalid session id" });
    }

    try {
      const existingSession = input.borg.sessions.get(sessionId);
      const audienceLabel =
        body.audience ?? existingSession?.audience_label ?? DEMO_DEFAULT_AUDIENCE_LABEL;
      const audienceEntityId = resolveTurnAudienceEntityId(input.borg, {
        audienceLabel,
        explicitAudienceEntityId: body.audience_entity_id,
        existingAudienceEntityId: existingSession?.audience_entity_id,
      });
      const senderEntityId = resolveTurnSenderEntityId(input.borg, {
        explicitSenderEntityId: body.sender_entity_id,
        audienceEntityId,
        demoCreatorEntityName: input.demoCreatorEntityName,
      });
      const sourceExternalId = existingSession?.source_external_id ?? sessionId;
      const sourceMessageKey = {
        source_type: DEMO_SOURCE_TYPE,
        source_external_id: sourceExternalId,
        external_message_id: body.external_message_id,
      };
      // Demo uploads accept png/jpeg/gif/webp images up to 8 MiB; Borg revalidates before persistence.
      const session = ensureDemoSession(input.borg, {
        sessionId,
        audienceLabel,
        audienceEntityId,
        sourceExternalId,
      });
      const result = await input.borg.enqueueMessage({
        session: {
          session_id: session.session_id,
          source_type: session.source_type,
          source_external_id: sourceExternalId,
          source_url: session.source_url,
          label: session.label,
          audience_label: session.audience_label,
          audience_entity_id: session.audience_entity_id,
          conversation_kind: session.conversation_kind,
          audience_role: session.audience_role,
        },
        userMessage: body.message,
        senderEntityId,
        sourceMessageKey,
        arrivedAt: Date.now(),
        audience: session.audience_label,
        audienceEntityId: session.audience_entity_id,
        attachments: body.attachments,
      });

      return c.json({
        ok: true,
        status: result.status,
        stream_entry_id: result.streamEntryId,
      });
    } catch (error) {
      mapBorgErrorToHttp(error);
    }
  });

  app.get("/api/prompts", (c) => c.json({ blocks: input.borg.prompts.list() }));

  app.put("/api/prompts/:key", async (c) => {
    const parsed = promptKeyParamSchema.safeParse(c.req.param("key"));
    if (!parsed.success) {
      throw new HTTPException(404, { message: "Unknown prompt key" });
    }
    const body = parseRequest(promptPutBodySchema, await parseJsonBody(c));

    try {
      const block = input.borg.prompts.set(parsed.data as PromptKey, body.text);
      return c.json(block);
    } catch (error) {
      mapBorgErrorToHttp(error);
    }
  });

  app.delete("/api/prompts/:key", (c) => {
    const parsed = promptKeyParamSchema.safeParse(c.req.param("key"));
    if (!parsed.success) {
      throw new HTTPException(404, { message: "Unknown prompt key" });
    }

    try {
      const block = input.borg.prompts.clear(parsed.data as PromptKey);
      return c.json(block);
    } catch (error) {
      mapBorgErrorToHttp(error);
    }
  });

  app.post("/api/admin/reset", async (c) => {
    parseRequest(resetBodySchema, await parseJsonBody(c));

    if (input.resetBorg === undefined) {
      throw new HTTPException(501, { message: "Reset not wired up in this server" });
    }

    const resetLease = input.requestGate.beginReset();
    try {
      clearAppCaches();
      await input.resetBorg();
      ensureDemoDefaultSession(input.borg, {
        demoCreatorEntityName: input.demoCreatorEntityName,
      });
      return c.json({ ok: true });
    } catch (error) {
      if (error instanceof HTTPException) {
        throw error;
      }
      if (error instanceof Error && !(error instanceof BorgError)) {
        throw new HTTPException(500, { message: error.message });
      }
      mapBorgErrorToHttp(error);
    } finally {
      resetLease.release();
    }
  });

  return { app, injectWebSocket };
}
