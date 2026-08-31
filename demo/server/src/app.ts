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
  classifySuppressionReason,
  memoryDisclosureLabelFromEpisodeAccess,
  type AttachmentId,
  type AutonomyWakeRecord,
  type Borg,
  type CommitmentEnforcementClass,
  type CommitmentRecord,
  type Config,
  type CreatorDirective,
  type EntityId,
  type Episode,
  type ImageMediaType,
  type ImagePerceptionRecord,
  type MaintenanceCadence,
  type MaintenanceAuditRecord,
  type MaintenanceTickResult,
  type MaintenancePlan,
  type MemoryDisclosureLabel,
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
  parseEpisodeId,
  parseGoalId,
  parseOpenQuestionId,
  parseSemanticEdgeId,
  parseSemanticNodeId,
  parseSessionId,
  parseTraitId,
  parseValueId,
  PROMPT_KEYS,
  semanticEdgeIdSchema,
  semanticNodeIdSchema,
  type AuditId,
  type BorgReviewResolutionInput,
  type PromptKey,
  type ReviewKind,
  type SuppressionOutcomeClass,
} from "borg";
import { z } from "zod";

import type { LiveBridge, MaintenanceTickFrame, MaintenanceTickFrameStatus } from "./live.js";
import type { BorgHandle } from "./reset.js";

type CursorPayload = {
  ts: number;
  entryId: string;
};

type DemoMaintenanceAuditRow = ReturnType<Borg["audit"]["list"]>[number];
type SemanticExtractionMaintenanceFacade = Borg["maintenance"] & {
  countPendingSemanticExtractionEpisodes: () => Promise<number>;
};

const PENDING_SEMANTIC_EXTRACTION_TTL_MS = 45_000;
const pendingSemanticExtractionMemo = new WeakMap<
  Borg,
  { expiresAt: number; promise: Promise<number> }
>();

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

const inflightQuerySchema = z.object({
  session: z
    .string()
    .min(1)
    .optional()
    .transform((value, ctx) => parseOptionalSessionQuery(value, ctx)),
});

const turnHistoryQuerySchema = z.object({
  session: z
    .string()
    .min(1)
    .optional()
    .transform((value, ctx) => parseOptionalSessionQuery(value, ctx)),
  limit: limitSchema.default(50),
  cursor: z
    .string()
    .optional()
    .transform((value, ctx) => {
      if (value === undefined || value.length === 0) {
        return undefined;
      }

      const parsed = decodeCursor(value);

      if (parsed === null) {
        ctx.addIssue({ code: "custom", message: "cursor is not a valid stream cursor" });
        return z.NEVER;
      }

      return parsed;
    }),
});

const dayQuerySchema = z
  .string()
  .regex(/^\d{4}-\d{2}-\d{2}$/, { message: "day must be YYYY-MM-DD" });

const activityQuerySchema = z
  .object({
    day: dayQuerySchema.optional(),
  })
  .strict();

const journalQuerySchema = z
  .object({
    limit: limitSchema.default(10),
    day: dayQuerySchema.optional(),
  })
  .strict();

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

const optionalNonEmptyQueryString = z
  .string()
  .optional()
  .transform((value) => {
    if (value === undefined) {
      return undefined;
    }

    const trimmed = value.trim();
    return trimmed.length === 0 ? undefined : trimmed;
  });

const memoryBandDetailQuerySchema = z
  .object({
    session: z
      .string()
      .min(1)
      .optional()
      .transform((value, ctx) => parseOptionalSessionQuery(value, ctx)),
    limit: limitSchema.default(50),
    cursor: optionalNonEmptyQueryString,
    query: optionalNonEmptyQueryString,
  })
  .strict();

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

const creatorDirectiveStatusQuerySchema = z
  .object({
    status: z.enum(["active", "revoked", "superseded", "all"]).default("active"),
  })
  .strict();

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
const episodeIdParamSchema = z.string().transform((value, ctx): Episode["id"] => {
  try {
    return parseEpisodeId(value) as Episode["id"];
  } catch {
    ctx.addIssue({ code: "custom", message: "Invalid episode id" });
    return z.NEVER;
  }
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

const CORRECTION_TARGET_ROUTE_PARSERS: readonly {
  prefix: string;
  parse: (value: string) => string;
}[] = [
  { prefix: "ep_", parse: parseEpisodeId },
  { prefix: "semn_", parse: parseSemanticNodeId },
  { prefix: "seme_", parse: parseSemanticEdgeId },
  { prefix: "val_", parse: parseValueId },
  { prefix: "goal_", parse: parseGoalId },
  { prefix: "trt_", parse: parseTraitId },
  { prefix: "cmt_", parse: parseCommitmentId },
  { prefix: "oq_", parse: parseOpenQuestionId },
];

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

function validateCorrectionWhyRouteId(id: string): void {
  const parser = CORRECTION_TARGET_ROUTE_PARSERS.find((candidate) =>
    id.startsWith(candidate.prefix),
  );

  if (parser === undefined) {
    return;
  }

  try {
    parser.parse(id);
  } catch (error) {
    if (error instanceof TypeError) {
      throw new HTTPException(400, { message: error.message });
    }
    throw error;
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
  // sessions.ensure() upserts source_type unconditionally, so typing every demo-route
  // write as DEMO_SOURCE_TYPE would re-label a session a connector owns. Session
  // source_type is a trust key -- peer-channel frame tolerance and the internal-identifier
  // guard's substrate-privileged exemption both read it -- so a demo HTTP call into a
  // connector-owned session must not silently downgrade it. Adopt the stored type,
  // and its transport identity with it, whenever the session already exists.
  const existing = borg.sessions.get(input.sessionId);
  const sourceExternalId = input.sourceExternalId ?? existing?.source_external_id ?? null;

  return borg.sessions.ensure({
    session_id: input.sessionId,
    source_type: existing?.source_type ?? DEMO_SOURCE_TYPE,
    source_external_id: sourceExternalId,
    source_url: existing?.source_url ?? null,
    label: input.label ?? existing?.label ?? demoSessionLabel(input.sessionId),
    // sessions.ensure() overwrites both audience columns unconditionally, so a caller
    // that does not name an audience must adopt the stored one rather than re-stamp the
    // demo default over it -- the same adoption the source/label fields above rely on.
    audience_label: input.audienceLabel ?? existing?.audience_label ?? DEMO_DEFAULT_AUDIENCE_LABEL,
    audience_entity_id: input.audienceEntityId ?? existing?.audience_entity_id ?? null,
    conversation_kind: existing?.conversation_kind ?? DEMO_CONVERSATION_KIND,
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

  // Seed the demo persona only when this session does not exist yet. A deployment whose
  // default session is the entity's own autonomous space (wakes, reflection, offline work
  // -- no other participant in the room) sets its audience once; re-passing the demo
  // default here would re-stamp that on every boot, and every record written in the
  // session inherits its audience as origin provenance.
  return ensureDemoSession(borg, { sessionId: DEFAULT_SESSION_ID });
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
    const status = error.code.endsWith("_NOT_FOUND")
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

type TurnHistoryOutcomeClass = "emitted" | "failed" | SuppressionOutcomeClass;

type TurnHistoryRow = {
  turn_id: string;
  started_at: number;
  audience: string | null;
  outcome: TurnHistoryOutcomeClass;
  suppression_reason: string | null;
};

type TurnHistoryRowWithCursor = TurnHistoryRow & {
  cursorEntry: StreamEntry;
};

type TurnHistoryAnchor = {
  entry: StreamEntry;
};

type TurnHistorySourceLookup = {
  byId: Map<string, StreamEntry>;
  byTurnId: Map<string, StreamEntry[]>;
};

const ABORTED_TURN_EVENT = "aborted_turn";
// Phase 1 serves a bounded recent persisted history window. Durable deep
// replay/list indexing is intentionally left for the later persistence sprint.
const TURN_HISTORY_REVERSE_SCAN_MAX_ENTRIES = 4096;
const TURN_HISTORY_REVERSE_SCAN_MAX_BYTES = 8 * 1024 * 1024;

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}

function contentString(value: unknown, key: string): string | null {
  if (!isRecord(value)) {
    return null;
  }

  const field = value[key];
  return typeof field === "string" && field.length > 0 ? field : null;
}

function turnIdFromAbortedMarker(entry: StreamEntry): string | null {
  if (entry.kind !== "internal_event" || !isRecord(entry.content)) {
    return null;
  }

  if (entry.content.event !== ABORTED_TURN_EVENT && entry.turn_status !== "aborted") {
    return null;
  }

  const contentTurnId = contentString(entry.content, "turn_id");
  if (contentTurnId !== null) {
    return contentTurnId;
  }

  return typeof entry.turn_id === "string" && entry.turn_id.length > 0 ? entry.turn_id : null;
}

function turnIdFromHistoryAnchor(entry: StreamEntry): string | null {
  if (isDemoTerminalEntry(entry)) {
    return entry.turn_id;
  }

  return turnIdFromAbortedMarker(entry);
}

function abortedAnchorFailureReason(entry: StreamEntry): string | null {
  if (turnIdFromAbortedMarker(entry) === null || !isRecord(entry.content)) {
    return null;
  }

  return contentString(entry.content, "reason");
}

function sourceEntryIds(entry: StreamEntry): string[] {
  const sourceEntryIds = entry.response_to?.source_entry_ids;
  if (!Array.isArray(sourceEntryIds)) {
    return [];
  }

  return sourceEntryIds.filter((entryId) => typeof entryId === "string");
}

function terminalOutcome(entry: StreamEntry): {
  outcome: TurnHistoryOutcomeClass;
  suppressionReason: string | null;
} | null {
  if (entry.kind === "agent_msg") {
    return { outcome: "emitted", suppressionReason: null };
  }

  if (entry.kind === "agent_observed") {
    return { outcome: "observed", suppressionReason: null };
  }

  if (entry.kind === "agent_suppressed") {
    const reason = contentString(entry.content, "reason");
    return {
      outcome: classifySuppressionReason(reason),
      suppressionReason: reason,
    };
  }

  return null;
}

function isTurnHistoryAnchorEntry(entry: StreamEntry): boolean {
  return terminalOutcome(entry) !== null || turnIdFromAbortedMarker(entry) !== null;
}

function cursorMatchesEntry(cursor: StreamCursor, entry: StreamEntry): boolean {
  return entry.timestamp === cursor.ts && entry.id === cursor.entryId;
}

function turnHistoryScanStop(input: { cursor?: StreamCursor; limit: number }) {
  let cursorSeen = input.cursor === undefined;
  let acceptedAfterCursor = 0;

  return (entries: StreamEntry[]): boolean => {
    const entry = entries.at(-1);
    if (entry === undefined) {
      return false;
    }

    if (!cursorSeen) {
      if (input.cursor !== undefined && cursorMatchesEntry(input.cursor, entry)) {
        cursorSeen = true;
      }
      return false;
    }

    acceptedAfterCursor += 1;
    return acceptedAfterCursor >= input.limit + 1;
  };
}

function lookupTurnHistorySources(
  reader: ReturnType<Borg["stream"]["reader"]>,
  anchors: readonly TurnHistoryAnchor[],
): TurnHistorySourceLookup {
  const wantedSourceIds = new Set<string>();
  const wantedTurnIds = new Set<string>();

  for (const anchor of anchors) {
    const turnId = turnIdFromHistoryAnchor(anchor.entry);
    if (turnId !== null) {
      wantedTurnIds.add(turnId);
    }
    for (const sourceEntryId of sourceEntryIds(anchor.entry)) {
      wantedSourceIds.add(sourceEntryId);
    }
  }

  if (wantedSourceIds.size === 0 && wantedTurnIds.size === 0) {
    return { byId: new Map(), byTurnId: new Map() };
  }

  const scan = reader.scanReverse({
    maxEntries: TURN_HISTORY_REVERSE_SCAN_MAX_ENTRIES,
    maxBytes: TURN_HISTORY_REVERSE_SCAN_MAX_BYTES,
    filter: (entry) =>
      wantedSourceIds.has(entry.id) ||
      (entry.turn_id !== undefined && wantedTurnIds.has(entry.turn_id)),
  });
  const byId = new Map<string, StreamEntry>();
  const byTurnId = new Map<string, StreamEntry[]>();

  for (const entry of scan.entries) {
    byId.set(entry.id, entry);

    if (entry.turn_id !== undefined && wantedTurnIds.has(entry.turn_id)) {
      byTurnId.set(entry.turn_id, [...(byTurnId.get(entry.turn_id) ?? []), entry]);
    }
  }

  return { byId, byTurnId };
}

function supportEntriesForAnchor(
  entry: StreamEntry,
  turnId: string,
  sources: TurnHistorySourceLookup,
): StreamEntry[] {
  const byId = sourceEntryIds(entry)
    .map((sourceEntryId) => sources.byId.get(sourceEntryId))
    .filter((source): source is StreamEntry => source !== undefined);
  const byTurn = sources.byTurnId.get(turnId) ?? [];
  const unique = new Map<string, StreamEntry>();

  for (const source of [...byId, ...byTurn]) {
    if (source.id !== entry.id) {
      unique.set(source.id, source);
    }
  }

  return [...unique.values()].sort((left, right) => left.timestamp - right.timestamp);
}

function turnHistoryRowFromAnchor(
  anchor: TurnHistoryAnchor,
  sources: TurnHistorySourceLookup,
): TurnHistoryRowWithCursor | null {
  const turnId = turnIdFromHistoryAnchor(anchor.entry);
  if (turnId === null) {
    return null;
  }

  const terminal = terminalOutcome(anchor.entry);
  const outcome =
    terminal === null ? { outcome: "failed" as const, suppressionReason: null } : terminal;
  const supportEntries = supportEntriesForAnchor(anchor.entry, turnId, sources);
  const startedAt = supportEntries.reduce(
    (earliest, entry) => Math.min(earliest, entry.timestamp),
    anchor.entry.timestamp,
  );
  const audience =
    supportEntries.find((entry) => entry.audience !== undefined)?.audience ??
    anchor.entry.audience ??
    null;

  return {
    turn_id: turnId,
    started_at: startedAt,
    audience,
    outcome: outcome.outcome,
    suppression_reason: outcome.suppressionReason,
    cursorEntry: anchor.entry,
  };
}

function readTurns(input: {
  borg: Borg;
  sessionId: SessionId;
  limit: number;
  cursor?: StreamCursor;
}): { rows: TurnHistoryRow[]; next_cursor: string | null } {
  const reader = input.borg.stream.reader({ session: input.sessionId });
  const scan = reader.scanReverse({
    maxEntries: TURN_HISTORY_REVERSE_SCAN_MAX_ENTRIES,
    maxBytes: TURN_HISTORY_REVERSE_SCAN_MAX_BYTES,
    filter: isTurnHistoryAnchorEntry,
    stop: turnHistoryScanStop({ cursor: input.cursor, limit: input.limit }),
  });
  const anchorsNewest = scan.entries.map<TurnHistoryAnchor>((entry) => ({ entry })).reverse();
  const cursorIndex =
    input.cursor === undefined
      ? -1
      : anchorsNewest.findIndex((anchor) =>
          input.cursor === undefined ? false : cursorMatchesEntry(input.cursor, anchor.entry),
        );
  const afterCursor =
    input.cursor === undefined
      ? anchorsNewest
      : cursorIndex === -1
        ? []
        : anchorsNewest.slice(cursorIndex + 1);
  const pageAnchors = afterCursor.slice(0, input.limit + 1);
  const visibleAnchors = pageAnchors.slice(0, input.limit);
  const sources = lookupTurnHistorySources(reader, visibleAnchors);
  const rows = visibleAnchors
    .map((anchor) => turnHistoryRowFromAnchor(anchor, sources))
    .filter((row): row is TurnHistoryRowWithCursor => row !== null);
  const lastRow = rows.at(-1);
  const next_cursor =
    pageAnchors.length > input.limit && lastRow !== undefined
      ? encodeCursor(lastRow.cursorEntry)
      : null;

  return {
    rows: rows.map((row) => ({
      turn_id: row.turn_id,
      started_at: row.started_at,
      audience: row.audience,
      outcome: row.outcome,
      suppression_reason: row.suppression_reason,
    })),
    next_cursor,
  };
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

function entityLabel(borg: Borg, id: EntityId | string | null | undefined): string | null {
  if (id === null || id === undefined) {
    return null;
  }

  return borg.entities.get(id as EntityId)?.canonical_name ?? String(id);
}

const ENTITY_ID_SHAPE = /^ent_[a-z0-9]{16}$/;

type LabelRef = {
  value: string;
  id: string | null;
  label: string | null;
};

function createLabelResolver(borg: Borg) {
  const entityLabels = new Map<string, string | null>();
  const sessions = new Map<SessionId, ReturnType<Borg["sessions"]["get"]>>();

  function entityName(id: EntityId | string | null | undefined): string | null {
    if (id === null || id === undefined) {
      return null;
    }

    const raw = String(id);
    if (entityLabels.has(raw)) {
      return entityLabels.get(raw) ?? null;
    }

    const label = borg.entities.get(raw as EntityId)?.canonical_name ?? null;
    entityLabels.set(raw, label);
    return label;
  }

  function sessionRecord(sessionId: SessionId) {
    const cached = sessions.get(sessionId);
    if (cached !== undefined || sessions.has(sessionId)) {
      return cached ?? null;
    }

    const session = borg.sessions.get(sessionId);
    sessions.set(sessionId, session);
    return session;
  }

  function participantRef(value: string): LabelRef {
    if (ENTITY_ID_SHAPE.test(value)) {
      return { value, id: value, label: entityName(value) };
    }

    return { value, id: null, label: value };
  }

  return {
    entityName,
    sessionRecord,
    participantRef,
  };
}

type LabelResolver = ReturnType<typeof createLabelResolver>;

type SemanticDisclosurePayload = {
  disclosureLabel?: MemoryDisclosureLabel | null;
};

type SerializedDisclosureLabel = {
  disclosure_class: MemoryDisclosureLabel["disclosureClass"];
  origin_audience_entity_ids: string[];
  private_to_entity_ids: string[];
  public_to_entity_ids: string[];
};

function serializeDisclosureLabel(
  label: MemoryDisclosureLabel,
  labels: LabelResolver,
): {
  origin_audience_entity_ids: string[];
  origin_audience_refs: LabelRef[];
  disclosure_class: SerializedDisclosureLabel["disclosure_class"];
  disclosure_label: SerializedDisclosureLabel;
} {
  const metadata: SerializedDisclosureLabel = {
    disclosure_class: label.disclosureClass,
    origin_audience_entity_ids: [...label.originAudienceEntityIds],
    private_to_entity_ids: [...label.privateToEntityIds],
    public_to_entity_ids: [...label.publicToEntityIds],
  };
  return {
    origin_audience_entity_ids: metadata.origin_audience_entity_ids,
    origin_audience_refs: metadata.origin_audience_entity_ids.map((id) =>
      labels.participantRef(id),
    ),
    disclosure_class: metadata.disclosure_class,
    disclosure_label: metadata,
  };
}

export function serializeStreamEntries(
  borg: Borg,
  entries: readonly StreamEntry[],
): Array<
  StreamEntry & {
    display_content?: unknown;
    sender_label: string | null;
    session_label: string | null;
    audience_label: string | null;
  }
> {
  const labels = createLabelResolver(borg);

  return entries.map((entry) => {
    const session = labels.sessionRecord(entry.session_id);
    const senderLabel = labels.entityName(entry.sender_entity_id);
    const audienceEntityLabel = labels.entityName(session?.audience_entity_id ?? null);
    const displayContent = streamDisplayContent(entry);

    return {
      ...entry,
      ...(displayContent === undefined ? {} : { display_content: displayContent }),
      sender_label: senderLabel,
      session_label: session?.label ?? null,
      audience_label: audienceEntityLabel,
    };
  });
}

function streamDisplayContent(entry: StreamEntry): unknown | undefined {
  if (
    entry.kind !== "user_msg" ||
    entry.source_message_key?.source_type !== "botarena" ||
    typeof entry.content !== "string"
  ) {
    return undefined;
  }

  const firstLineBreak = entry.content.indexOf("\n");
  if (firstLineBreak <= 0 || entry.content[0] !== "[") {
    return undefined;
  }

  const closingBracket = entry.content.indexOf("]");
  if (closingBracket <= 0 || closingBracket > firstLineBreak) {
    return undefined;
  }

  let bodyStart = firstLineBreak + 1;
  while (entry.content[bodyStart] === "\n" || entry.content[bodyStart] === "\r") {
    bodyStart += 1;
  }

  const markerEnd = bodyStart + "[message]".length;
  if (
    entry.content.startsWith("[message]", bodyStart) &&
    (markerEnd === entry.content.length ||
      entry.content[markerEnd] === "\n" ||
      entry.content[markerEnd] === "\r")
  ) {
    bodyStart = markerEnd;
    while (entry.content[bodyStart] === "\n" || entry.content[bodyStart] === "\r") {
      bodyStart += 1;
    }
  }

  const body = entry.content.slice(bodyStart).trim();
  return body.length === 0 ? undefined : body;
}

type SerializedStreamEntry = ReturnType<typeof serializeStreamEntries>[number];
type JournalEntryForActivity = Awaited<ReturnType<Borg["self"]["journal"]["list"]>>[number];
type AutonomousActionEntry = SerializedStreamEntry & {
  content: {
    kind: "autonomous_action";
    trigger?: unknown;
    outcome_summary?: unknown;
    turn_result_id?: unknown;
    ts?: unknown;
  };
};
type AutonomousWakeEntry = SerializedStreamEntry & {
  content: {
    kind: "autonomous_wake";
    source_name?: unknown;
    trigger_type?: unknown;
    source_category?: unknown;
    payload?: unknown;
    ts?: unknown;
  };
};
type ActivityOrigin = "user" | "autonomous" | "dream";
type ActivityTurnRow = {
  id: string;
  kind: "turn";
  started_at: number;
  session_id: string;
  session_label: string | null;
  origin: Exclude<ActivityOrigin, "dream">;
  trigger: string | null;
  outcome: TurnHistoryOutcomeClass;
  suppression_reason: string | null;
  duration_ms: number | null;
  excerpt: string | null;
  turn_id: string;
};
type ActivityDreamRow = {
  id: string;
  kind: "dream";
  started_at: number;
  session_id: string;
  session_label: string | null;
  origin: "dream";
  trigger: string | null;
  outcome: "dream";
  suppression_reason: null;
  duration_ms: number | null;
  excerpt: string | null;
  turn_id: null;
  dream: {
    run_id: string;
    process_count: number;
    changes: number;
    errors: number;
  };
};
type ActivityRow = ActivityTurnRow | ActivityDreamRow;

const ACTIVITY_SESSION_LIMIT = 200;
const ACTIVITY_SESSION_TAIL_LIMIT = 800;
const ACTIVITY_DAY_ROW_LIMIT = 200;
const ACTIVITY_AVAILABLE_DAY_LIMIT = 7;

function localDayString(ts = Date.now()): string {
  const date = new Date(ts);
  const year = date.getFullYear();
  const month = String(date.getMonth() + 1).padStart(2, "0");
  const day = String(date.getDate()).padStart(2, "0");
  return `${year}-${month}-${day}`;
}

function dayBounds(day: string): { start: number; end: number } {
  const [yearRaw, monthRaw, dayRaw] = day.split("-");
  const year = Number(yearRaw);
  const month = Number(monthRaw);
  const date = Number(dayRaw);
  const start = new Date(year, month - 1, date).getTime();
  return {
    start,
    end: new Date(year, month - 1, date + 1).getTime(),
  };
}

function conciseText(value: string | null | undefined, maxLength = 200): string | null {
  const trimmed = value?.trim();
  if (trimmed === undefined || trimmed.length === 0) {
    return null;
  }

  return trimmed.length <= maxLength ? trimmed : trimmed.slice(0, maxLength);
}

function stringContent(value: unknown): string | null {
  return typeof value === "string" && value.trim().length > 0 ? value : null;
}

function persistedEntryExcerpt(
  entry: SerializedStreamEntry | undefined,
  fallbackReason: string | null,
): string | null {
  if (entry === undefined) {
    return conciseText(fallbackReason);
  }

  if (entry.kind === "agent_msg") {
    return conciseText(stringContent(entry.content));
  }

  if (entry.kind === "agent_suppressed" || entry.kind === "agent_observed") {
    return conciseText(contentString(entry.content, "reason") ?? fallbackReason);
  }

  return conciseText(stringContent(entry.display_content) ?? stringContent(entry.content));
}

function isAutonomousActionEntry(
  entry: SerializedStreamEntry,
): entry is AutonomousActionEntry {
  return (
    entry.kind === "internal_event" &&
    isRecord(entry.content) &&
    entry.content.kind === "autonomous_action"
  );
}

function isAutonomousWakeEntry(entry: SerializedStreamEntry): entry is AutonomousWakeEntry {
  return (
    entry.kind === "internal_event" &&
    isRecord(entry.content) &&
    entry.content.kind === "autonomous_wake"
  );
}

function autonomousActionTrigger(entry: AutonomousActionEntry): string | null {
  return typeof entry.content.trigger === "string" && entry.content.trigger.length > 0
    ? entry.content.trigger
    : null;
}

function autonomousWakeSourceName(entry: AutonomousWakeEntry): string | null {
  return typeof entry.content.source_name === "string" && entry.content.source_name.length > 0
    ? entry.content.source_name
    : null;
}

function entryOrder(left: SerializedStreamEntry, right: SerializedStreamEntry): number {
  if (left.timestamp !== right.timestamp) {
    return left.timestamp - right.timestamp;
  }

  return (left.entry_index ?? 0) - (right.entry_index ?? 0);
}

function activityRecentEntries(borg: Borg): SerializedStreamEntry[] {
  const sessions = borg.sessions.list({ limit: ACTIVITY_SESSION_LIMIT });
  const entries: StreamEntry[] = [];
  const seen = new Set<string>();

  for (const session of sessions) {
    for (const entry of borg.stream.tail(ACTIVITY_SESSION_TAIL_LIMIT, { session: session.session_id })) {
      if (seen.has(entry.id)) {
        continue;
      }

      seen.add(entry.id);
      entries.push(entry);
    }
  }

  return serializeStreamEntries(borg, entries);
}

function autonomousTriggersByTerminalEntryId(
  entries: readonly SerializedStreamEntry[],
): Map<string, string> {
  const byTerminalEntryId = new Map<string, string>();
  const bySession = new Map<string, SerializedStreamEntry[]>();

  for (const entry of entries) {
    const sessionEntries = bySession.get(entry.session_id) ?? [];
    sessionEntries.push(entry);
    bySession.set(entry.session_id, sessionEntries);
  }

  for (const sessionEntries of bySession.values()) {
    let pendingWake: AutonomousWakeEntry | null = null;
    let pendingTerminal: SerializedStreamEntry | null = null;

    for (const entry of [...sessionEntries].sort(entryOrder)) {
      if (isAutonomousWakeEntry(entry)) {
        pendingWake = entry;
        pendingTerminal = null;
        continue;
      }

      if (isTurnHistoryAnchorEntry(entry)) {
        pendingTerminal = pendingWake === null ? null : entry;
        continue;
      }

      if (!isAutonomousActionEntry(entry)) {
        continue;
      }

      const trigger = autonomousActionTrigger(entry) ?? (pendingWake === null ? null : autonomousWakeSourceName(pendingWake));
      if (typeof entry.content.turn_result_id === "string" && trigger !== null) {
        byTerminalEntryId.set(entry.content.turn_result_id, trigger);
      } else if (pendingWake !== null && pendingTerminal !== null && trigger !== null) {
        byTerminalEntryId.set(pendingTerminal.id, trigger);
      }

      pendingWake = null;
      pendingTerminal = null;
    }
  }

  return byTerminalEntryId;
}

function activityTurnKey(sessionId: string, turnId: string): string {
  return `${sessionId}\u0000${turnId}`;
}

function activitySupportEntriesForAnchor(
  entry: SerializedStreamEntry,
  turnId: string,
  sources: {
    byId: Map<string, SerializedStreamEntry>;
    bySessionTurnId: Map<string, SerializedStreamEntry[]>;
  },
): SerializedStreamEntry[] {
  const byId = sourceEntryIds(entry)
    .map((sourceEntryId) => sources.byId.get(sourceEntryId))
    .filter((source): source is SerializedStreamEntry => source !== undefined);
  const byTurn = sources.bySessionTurnId.get(activityTurnKey(entry.session_id, turnId)) ?? [];
  const unique = new Map<string, SerializedStreamEntry>();

  for (const source of [...byId, ...byTurn]) {
    if (source.id !== entry.id) {
      unique.set(source.id, source);
    }
  }

  return [...unique.values()].sort((left, right) => left.timestamp - right.timestamp);
}

function activityTurnRows(entries: readonly SerializedStreamEntry[]): ActivityTurnRow[] {
  const bySessionTurnId = new Map<string, SerializedStreamEntry[]>();
  const byId = new Map<string, SerializedStreamEntry>();
  const autonomousTriggerByTerminalEntryId = autonomousTriggersByTerminalEntryId(entries);

  for (const entry of entries) {
    byId.set(entry.id, entry);

    if (typeof entry.turn_id === "string" && entry.turn_id.length > 0) {
      const key = activityTurnKey(entry.session_id, entry.turn_id);
      const bucket = bySessionTurnId.get(key) ?? [];
      bucket.push(entry);
      bySessionTurnId.set(key, bucket);
    }

  }

  const rows: ActivityTurnRow[] = [];
  const seenAnchors = new Set<string>();
  const sources = { byId, bySessionTurnId };

  for (const entry of entries) {
    if (!isTurnHistoryAnchorEntry(entry) || seenAnchors.has(entry.id)) {
      continue;
    }

    seenAnchors.add(entry.id);
    const turnId = turnIdFromHistoryAnchor(entry);
    if (turnId === null) {
      continue;
    }

    const terminal = terminalOutcome(entry);
    const outcome =
      terminal === null ? { outcome: "failed" as const, suppressionReason: null } : terminal;
    const supportEntries = activitySupportEntriesForAnchor(entry, turnId, sources);
    const startedAt = supportEntries.reduce(
      (earliest, source) => Math.min(earliest, source.timestamp),
      entry.timestamp,
    );
    const terminalEntry = isDemoTerminalEntry(entry) ? entry : undefined;
    const trigger = autonomousTriggerByTerminalEntryId.get(entry.id) ?? null;

    rows.push({
      id: `turn:${entry.session_id}:${turnId}`,
      kind: "turn",
      started_at: startedAt,
      session_id: entry.session_id,
      session_label: entry.session_label,
      origin: trigger === null ? "user" : "autonomous",
      trigger,
      outcome: outcome.outcome,
      suppression_reason: outcome.suppressionReason,
      duration_ms: null,
      excerpt: persistedEntryExcerpt(
        terminalEntry,
        outcome.suppressionReason ?? abortedAnchorFailureReason(entry),
      ),
      turn_id: turnId,
    });
  }

  return rows;
}

function activityDreamRows(entries: readonly SerializedStreamEntry[]): ActivityDreamRow[] {
  return entries.flatMap((entry) => {
    const report = mapDreamReport(entry);
    if (report === null) {
      return [];
    }

    return [
      {
        id: `dream:${entry.id}`,
        kind: "dream" as const,
        started_at: entry.timestamp,
        session_id: entry.session_id,
        session_label: entry.session_label,
        origin: "dream" as const,
        trigger: null,
        outcome: "dream" as const,
        suppression_reason: null,
        duration_ms: null,
        excerpt: conciseText(report.notes[0]),
        turn_id: null,
        dream: {
          run_id: report.run_id,
          process_count: report.processes.length,
          changes: report.changes,
          errors: report.errors.length,
        },
      },
    ];
  });
}

function journalRowsForDay(
  journalEntries: readonly JournalEntryForActivity[],
  day: string,
): JournalEntryForActivity[] {
  return journalEntries.filter((entry) => localDayString(entry.created_at) === day);
}

function activityDigest(rows: readonly ActivityRow[], journalEntries: readonly JournalEntryForActivity[]) {
  return {
    turns: rows.filter((row) => row.kind === "turn").length,
    autonomous_wakes: rows.filter((row) => row.origin === "autonomous").length,
    emissions: rows.filter((row) => row.kind === "turn" && row.outcome === "emitted").length,
    silences: rows.filter((row) => row.kind === "turn" && row.outcome === "deliberate-silence").length,
    observations: rows.filter((row) => row.kind === "turn" && row.outcome === "observed").length,
    suppressions: rows.filter(
      (row) =>
        row.kind === "turn" &&
        (row.outcome === "guard-blocked" ||
          row.outcome === "emission-failed" ||
          row.outcome === "unknown"),
    ).length,
    dream_changes: rows.reduce(
      (sum, row) => sum + (row.kind === "dream" ? row.dream.changes : 0),
      0,
    ),
    journal_notes: journalEntries.length,
  };
}

function activityFeed(borg: Borg, requestedDay?: string) {
  const selectedDay = requestedDay ?? localDayString();
  const bounds = dayBounds(selectedDay);
  const entries = activityRecentEntries(borg);
  const allRows = [...activityTurnRows(entries), ...activityDreamRows(entries)].sort(
    (left, right) => right.started_at - left.started_at,
  );
  const days = [...new Set(allRows.map((row) => localDayString(row.started_at)))]
    .sort((left, right) => right.localeCompare(left))
    .slice(0, ACTIVITY_AVAILABLE_DAY_LIMIT);
  const dayRows = allRows.filter(
    (row) => row.started_at >= bounds.start && row.started_at < bounds.end,
  );
  const rows = dayRows.slice(0, ACTIVITY_DAY_ROW_LIMIT);
  const journalEntries = journalRowsForDay(borg.self.journal.list({ limit: 500 }), selectedDay);

  return {
    day: selectedDay,
    days,
    rows,
    truncated: dayRows.length > rows.length,
    digest: activityDigest(rows, journalEntries),
  };
}

function mapJournalEntry(borg: Borg, entry: JournalEntryForActivity) {
  return {
    ...entry,
    self_label: entityLabel(borg, entry.self_entity_id),
  };
}

function mapAutonomyWake(borg: Borg, wake: AutonomyWakeRecord) {
  return {
    id: wake.id,
    ts: wake.ts,
    trigger_name: wake.trigger_name,
    condition_name: wake.condition_name,
    session_id: wake.session_id,
    session_label: wake.session_id === null ? null : borg.sessions.get(wake.session_id)?.label ?? null,
    wake_source_type: wake.wake_source_type,
    source_category: wake.source_category,
  };
}

async function autonomyState(borg: Borg) {
  const schedulerDescription = await borg.autonomy.scheduler.describe();
  const since = Date.now() - 7 * 24 * 60 * 60 * 1_000;
  const recentWakes = borg.autonomy.wakes.listSince(since, 100);
  const latestByName = new Map<string, AutonomyWakeRecord>();
  const countsByName = new Map<string, number>();

  for (const wake of recentWakes) {
    countsByName.set(wake.trigger_name, (countsByName.get(wake.trigger_name) ?? 0) + 1);
    const current = latestByName.get(wake.trigger_name);
    if (current === undefined || wake.ts > current.ts) {
      latestByName.set(wake.trigger_name, wake);
    }
  }

  return {
    scheduler: {
      enabled: schedulerDescription.enabled,
      interval_ms: schedulerDescription.interval_ms,
      next_tick_at: schedulerDescription.next_tick_at,
    },
    wake_sources: schedulerDescription.sources.map((source) => {
      const latest = latestByName.get(source.name);

      return {
        name: source.name,
        enabled: source.enabled,
        wake_source_type: source.type,
        source_category: source.category,
        ...(source.type === "trigger" ? { next_due_at: source.next_due_at } : {}),
        last_fired: latest?.ts ?? null,
        wake_count: countsByName.get(source.name) ?? 0,
      };
    }),
    wake_budget: {
      used: schedulerDescription.budget.used_in_current_window,
      limit: schedulerDescription.budget.max_wakes_per_window,
      window_ms: schedulerDescription.budget.window_ms,
    },
    self_scheduled_wakes: [],
    can_cancel_wakes: false,
    recent_wakes: recentWakes.map((wake) => mapAutonomyWake(borg, wake)),
  };
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
    source_session_id: record.source_session_id,
    authorization_stream_entry_ids: record.authorization_stream_entry_ids,
    content_source_stream_entry_ids: record.content_source_stream_entry_ids,
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
    superseded_by_id: record.superseded_by,
    revoked_reason: record.revoked_reason,
    created_at: record.created_at,
    updated_at: record.updated_at,
  };
}

function mapEpisode(
  borg: Borg,
  item: Awaited<ReturnType<Borg["episodic"]["list"]>>["items"][number],
  labels: LabelResolver = createLabelResolver(borg),
) {
  const disclosure = serializeDisclosureLabel(memoryDisclosureLabelFromEpisodeAccess(item), labels);

  return {
    id: item.id,
    title: item.title,
    narrative: item.narrative,
    participants: item.participants,
    participant_refs: item.participants.map((participant) => labels.participantRef(participant)),
    location: item.location,
    start_time: item.start_time,
    end_time: item.end_time,
    audience: entityLabel(borg, item.audience_entity_id ?? null),
    origin_audience_entity_ids: disclosure.origin_audience_entity_ids,
    origin_audience_refs: disclosure.origin_audience_refs,
    shared: disclosure.disclosure_class === "public",
    disclosure_class: disclosure.disclosure_class,
    disclosure_label: disclosure.disclosure_label,
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

function mapSemanticMemoryNode(
  node: SemanticNode & SemanticDisclosurePayload,
  searchScore?: number,
  labels?: LabelResolver,
) {
  const displayLabel = ENTITY_ID_SHAPE.test(node.label)
    ? (labels?.entityName(node.label) ?? null)
    : node.label;
  const disclosure =
    node.disclosureLabel === undefined || node.disclosureLabel === null || labels === undefined
      ? {}
      : serializeDisclosureLabel(node.disclosureLabel, labels);

  return {
    id: node.id,
    kind: node.kind,
    label: node.label,
    display_label: displayLabel,
    description: node.description,
    domain: node.domain,
    aliases: node.aliases,
    confidence: node.confidence,
    status: node.status,
    source_episode_ids: node.source_episode_ids,
    source_count: node.source_episode_ids.length,
    ...disclosure,
    created_at: node.created_at,
    updated_at: node.updated_at,
    ...(searchScore === undefined ? {} : { search_score: searchScore }),
  };
}

function mapSemanticMemoryEdge(
  edge: SemanticEdge & SemanticDisclosurePayload,
  labels?: LabelResolver,
) {
  const disclosure =
    edge.disclosureLabel === undefined || edge.disclosureLabel === null || labels === undefined
      ? {}
      : serializeDisclosureLabel(edge.disclosureLabel, labels);

  return {
    id: edge.id,
    from_node_id: edge.from_node_id,
    to_node_id: edge.to_node_id,
    relation: edge.relation,
    confidence: edge.confidence,
    evidence_episode_ids: edge.evidence_episode_ids,
    source_count: edge.evidence_episode_ids.length,
    ...disclosure,
    valid_from: edge.valid_from,
    valid_to: edge.valid_to,
    invalidated_at: edge.invalidated_at,
    invalidated_by_edge_id: edge.invalidated_by_edge_id,
    invalidated_by_review_id: edge.invalidated_by_review_id,
    invalidated_by_process: edge.invalidated_by_process,
    invalidated_reason: edge.invalidated_reason,
  };
}

function mapSkillMemoryItem(
  skill: ReturnType<Borg["skills"]["list"]>[number],
  searchScore?: number,
) {
  return {
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
    ...(searchScore === undefined ? {} : { search_score: searchScore }),
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
  const labels = createLabelResolver(borg);
  const statusCounts = borg.semantic.nodes.countByStatus();
  const totalNodes = sumRecord(statusCounts);
  const nodes = totalNodes === 0 ? [] : await borg.semantic.nodes.list({ limit: totalNodes });
  const edges = await borg.semantic.edges.list();
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
    nodes: selectedNodes.map(({ node, edgeCount }) =>
      mapSemanticGraphNode(node, edgeCount, labels),
    ),
    edges: selectedEdges.map((edge) => mapSemanticGraphEdge(edge)),
    total_nodes: totalNodes,
    total_edges: edges.length,
    rendered: {
      nodes: selectedNodes.length,
      edges: selectedEdges.length,
    },
  };
}

function mapSemanticGraphNode(node: SemanticNode, edgeCount: number, labels: LabelResolver) {
  const displayLabel = ENTITY_ID_SHAPE.test(node.label)
    ? labels.entityName(node.label)
    : node.label;

  return {
    id: node.id,
    label: node.label,
    display_label: displayLabel,
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
    associator: "link related memory records",
    "review-resolver": "process review queue items",
    "creator-directive-reconciler": "reconcile redundant or conflicting creator directives",
    ruminator: "open-question rumination",
    "self-narrator": "autobiography and growth markers",
    "procedural-synthesizer": "skill abstractions",
    "belief-reviser": "invalidate, weaken, contradict",
    "commitment-reconciler": "reconcile redundant or conflicting commitments",
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

function streamDreamStringArray(entry: StreamEntry, key: string): string[] {
  if (!isRecord(entry.content)) {
    return [];
  }

  const value = entry.content[key];
  if (!Array.isArray(value)) {
    return [];
  }

  return value.filter((item): item is string => typeof item === "string");
}

function streamDreamRecordArray(entry: StreamEntry, key: string): Array<Record<string, unknown>> {
  if (!isRecord(entry.content)) {
    return [];
  }

  const value = entry.content[key];
  if (!Array.isArray(value)) {
    return [];
  }

  return value.filter((item): item is Record<string, unknown> => isRecord(item));
}

function streamDreamErrorArray(entry: StreamEntry): Array<Record<string, string>> {
  return streamDreamRecordArray(entry, "errors").map((error) => {
    const safe: Record<string, string> = {};

    for (const key of ["process", "message", "code", "target_type", "target_id"]) {
      const value = error[key];
      if (typeof value === "string") {
        safe[key] = value;
      }
    }

    return safe;
  });
}

function streamDreamNumber(entry: StreamEntry, key: string, fallback: number): number {
  if (!isRecord(entry.content)) {
    return fallback;
  }

  const value = entry.content[key];
  return typeof value === "number" && Number.isFinite(value) ? value : fallback;
}

function streamDreamOptionalNumber(entry: StreamEntry, key: string): number | null {
  if (!isRecord(entry.content)) {
    return null;
  }

  const value = entry.content[key];
  return typeof value === "number" && Number.isFinite(value) ? value : null;
}

function mapDreamReport(entry: StreamEntry) {
  if (entry.kind !== "dream_report" || !isRecord(entry.content)) {
    return null;
  }

  const runId = entry.content.run_id;
  if (typeof runId !== "string" || runId.length === 0) {
    return null;
  }

  return {
    run_id: runId,
    processes: streamDreamProcesses(entry),
    dry_run: entry.content.dry_run === true,
    planned_at: streamDreamOptionalNumber(entry, "planned_at"),
    changes: streamDreamNumber(entry, "changes", 0),
    tokens_used: streamDreamNumber(entry, "tokens_used", 0),
    errors: streamDreamErrorArray(entry),
    budget_exhausted_processes: streamDreamStringArray(entry, "budget_exhausted_processes").filter(
      (process): process is OfflineProcessName =>
        OFFLINE_PROCESS_NAMES.includes(process as OfflineProcessName),
    ),
    notes: streamDreamStringArray(entry, "notes"),
  };
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
  const episodicCountIsLowerBound = episodes.nextCursor !== undefined;
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
      count_is_lower_bound: episodicCountIsLowerBound,
      stats: [
        {
          k: "items",
          v: episodicCountIsLowerBound ? `${episodes.items.length}+` : episodes.items.length,
        },
      ],
    },
    {
      id: "semantic",
      n: "02",
      name: "semantic",
      desc: "what Borg believes",
      count: sumRecord(semanticCounts),
      stats: Object.entries(semanticCounts).map(([k, v]) => ({ k, v })),
    },
    {
      id: "procedural",
      n: "03",
      name: "procedural",
      desc: "how Borg solves things",
      count: procedural.length,
      stats: [{ k: "skills", v: procedural.length }],
    },
    {
      id: "affective",
      n: "04",
      name: "affective",
      desc: "mood and trajectory",
      count: moodHistory.length,
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
      stats: [{ k: "profiles", v: borg.social.list(500).length }],
    },
    {
      id: "relational",
      n: "08",
      name: "relational",
      desc: "evidence-backed relationship facts",
      count: sumRecord(relationalCounts),
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
  const now = Date.now();
  const cached = pendingSemanticExtractionMemo.get(borg);

  if (cached !== undefined && cached.expiresAt > now) {
    return cached.promise;
  }

  const promise = Promise.resolve().then(() =>
    (
      borg.maintenance as SemanticExtractionMaintenanceFacade
    ).countPendingSemanticExtractionEpisodes(),
  );
  const entry = {
    expiresAt: now + PENDING_SEMANTIC_EXTRACTION_TTL_MS,
    promise,
  };
  pendingSemanticExtractionMemo.set(borg, entry);

  try {
    return await promise;
  } catch (error) {
    if (pendingSemanticExtractionMemo.get(borg) === entry) {
      pendingSemanticExtractionMemo.delete(borg);
    }
    throw error;
  }
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
    dream_reports: dreamReports
      .map((entry) => mapDreamReport(entry))
      .filter((report): report is NonNullable<typeof report> => report !== null),
    audit_rows: auditRows.map((row) => mapMaintenanceAuditRow(row)),
    belief_revision_rows: borg.review
      .list({ kind: "belief_revision", openOnly: true })
      .map((row) => mapReviewRow(row)),
    scheduler: {
      enabled: borg.maintenance.scheduler.isEnabled(),
      light_interval_ms: config.lightIntervalMs,
      heavy_interval_ms: config.heavyIntervalMs,
      optimize_storage: config.optimizeStorage,
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
  runtimeConfig?: DemoRuntimeConfig;
};

export type DemoRuntimeConfig = {
  model: string;
  embedding: {
    model: string;
    dims: number;
  };
};

export function runtimeConfigFromConfig(config: Config): DemoRuntimeConfig {
  return {
    model: config.anthropic.models.cognition,
    embedding: {
      model: config.embedding.model,
      dims: config.embedding.dims,
    },
  };
}

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
    runtimeConfig: args.runtimeConfig,
  };
  const app = new Hono();
  const { injectWebSocket, upgradeWebSocket } = createNodeWebSocket({ app });
  ensureDemoDefaultSession(input.borg, {
    demoCreatorEntityName: input.demoCreatorEntityName,
  });
  input.live.setStreamEntrySerializer((entries) => serializeStreamEntries(input.borg, entries));
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
      ...(input.runtimeConfig === undefined ? {} : { runtime: input.runtimeConfig }),
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
    const result = await readStream({
      borg: input.borg,
      sessionId: query.session,
      kinds: query.kind,
      audience: query.audience,
      limit: query.limit,
      before: query.before,
    });
    return c.json({
      ...result,
      entries: serializeStreamEntries(input.borg, result.entries),
    });
  });

  app.get("/api/turns", async (c) => {
    const query = parseRequest(turnHistoryQuerySchema, c.req.query());
    return c.json(
      await readTurns({
        borg: input.borg,
        sessionId: query.session,
        limit: query.limit,
        cursor: query.cursor,
      }),
    );
  });

  app.get("/api/inflight", (c) => {
    const query = parseRequest(inflightQuerySchema, c.req.query());
    return c.json({ inflight: input.live.broadcaster.inflightSnapshot(query.session) });
  });

  app.get("/api/activity", (c) => {
    const query = parseRequest(activityQuerySchema, c.req.query());
    return c.json(activityFeed(input.borg, query.day));
  });

  app.get("/api/autonomy", async (c) => c.json(await autonomyState(input.borg)));

  app.get("/api/journal", (c) => {
    const query = parseRequest(journalQuerySchema, c.req.query());
    const entries =
      query.day === undefined
        ? input.borg.self.journal.list({ limit: query.limit })
        : journalRowsForDay(input.borg.self.journal.list({ limit: 500 }), query.day).slice(
            0,
            query.limit,
          );
    return c.json({
      entries: entries.map((entry) => mapJournalEntry(input.borg, entry)),
    });
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

  app.get("/api/semantic/nodes/:id", async (c) => {
    const id = parseRequest(semanticNodeIdSchema, c.req.param("id"));
    const node = await input.borg.semantic.nodes.get(id);

    if (node === null) {
      throw new HTTPException(404, { message: `semantic node ${id} not found` });
    }

    return c.json({
      node: mapSemanticMemoryNode(node, undefined, createLabelResolver(input.borg)),
    });
  });

  app.get("/api/semantic/edges/:id", async (c) => {
    try {
      const id = parseRequest(semanticEdgeIdSchema, c.req.param("id"));
      const edge = await input.borg.semantic.edges.get(id);

      if (edge === null) {
        throw new HTTPException(404, { message: `semantic edge ${id} not found` });
      }

      return c.json({ edge: mapSemanticMemoryEdge(edge, createLabelResolver(input.borg)) });
    } catch (error) {
      mapBorgErrorToHttp(error);
    }
  });

  app.get("/api/episodes/:id", async (c) => {
    const id = parseRequest(episodeIdParamSchema, c.req.param("id"));
    const result = await input.borg.episodic.get(id, { crossAudience: true });

    if (result === null) {
      throw new HTTPException(404, { message: `episode ${id} not found` });
    }

    return c.json({
      episode: mapEpisode(input.borg, result.episode, createLabelResolver(input.borg)),
    });
  });

  app.get("/api/memory/bands/:id", async (c) => {
    try {
      const band = parseRequest(memoryBandIdSchema, c.req.param("id"));
      const query = parseRequest(memoryBandDetailQuerySchema, c.req.query());

      if (band === "episodic") {
        const labels = createLabelResolver(input.borg);
        if (query.query !== undefined) {
          const results = await input.borg.episodic.search(query.query, {
            limit: query.limit,
            recordRetrieval: false,
          });
          return c.json({
            band,
            mode: "search",
            query: query.query,
            items: results.map((result) => ({
              ...mapEpisode(input.borg, result.episode, labels),
              search_score: result.score,
            })),
            next_cursor: null,
          });
        }

        const result = await input.borg.episodic.list({
          limit: query.limit,
          cursor: query.cursor,
        });
        return c.json({
          band,
          mode: "browse",
          items: result.items.map((item) => mapEpisode(input.borg, item, labels)),
          next_cursor: result.nextCursor ?? null,
        });
      }

      if (band === "semantic") {
        const labels = createLabelResolver(input.borg);
        if (query.query !== undefined) {
          const nodes = await input.borg.semantic.nodes.search(query.query, { limit: query.limit });
          return c.json({
            band,
            mode: "search",
            query: query.query,
            nodes: nodes.map((candidate) =>
              mapSemanticMemoryNode(candidate.node, candidate.similarity, labels),
            ),
            edges: [],
            next_cursor: null,
          });
        }

        const nodes = await input.borg.semantic.nodes.listPage({
          limit: query.limit,
          cursor: query.cursor,
        });
        // Rich topology is served by GET /api/semantic/graph; this band preview keeps edges bounded.
        const edges = (await input.borg.semantic.edges.list()).slice(0, 50);

        return c.json({
          band,
          mode: "browse",
          nodes: nodes.items.map((node) => mapSemanticMemoryNode(node, undefined, labels)),
          edges: edges.map((edge) => mapSemanticMemoryEdge(edge, labels)),
          next_cursor: nodes.nextCursor ?? null,
        });
      }

      if (band === "procedural") {
        if (query.query !== undefined) {
          const results = await input.borg.skills.searchByContext(query.query, query.limit);
          return c.json({
            band,
            mode: "search",
            query: query.query,
            items: results.map((result) => mapSkillMemoryItem(result.skill, result.similarity)),
            next_cursor: null,
          });
        }

        return c.json({
          band,
          mode: "browse",
          items: input.borg.skills.list(100).map((skill) => mapSkillMemoryItem(skill)),
        });
      }

      if (band === "affective") {
        return c.json({
          band,
          mode: "browse",
          current: input.borg.mood.current(query.session),
          history: input.borg.mood.history(query.session, { limit: 100 }),
        });
      }

      if (band === "commitments") {
        return c.json({
          band,
          mode: "browse",
          items: input.borg.commitments
            .list({ activeOnly: false })
            .map((record) => mapCommitment(input.borg, record)),
        });
      }

      if (band === "self") {
        return c.json({ band, mode: "browse", ...selfSnapshot(input.borg) });
      }

      if (band === "social") {
        return c.json({
          band,
          mode: "browse",
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
        mode: "browse",
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
    } catch (error) {
      mapBorgErrorToHttp(error);
    }
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
    const query = parseRequest(creatorDirectiveStatusQuerySchema, c.req.query());
    const directives = input.borg.creatorDirectives
      .list(query.status === "all" ? {} : { status: query.status })
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
      const id = c.req.param("id");
      validateCorrectionWhyRouteId(id);
      return c.json(await input.borg.correction.why(id));
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
      const audienceLabel = body.audience ?? existingSession?.audience_label;
      if (audienceLabel === undefined) {
        throw new HTTPException(400, { message: "audience is required for unknown sessions" });
      }
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
        // enqueueMessage requires the message key to match the session's transport, and
        // ensureDemoSession preserves a connector-owned session's type, so posting into
        // such a session must adopt that transport rather than assert the demo one.
        source_type: existingSession?.source_type ?? DEMO_SOURCE_TYPE,
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

  app.get("/api/prompts/assembled", (c) => c.json(input.borg.prompts.previewAssembledFraming()));

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
