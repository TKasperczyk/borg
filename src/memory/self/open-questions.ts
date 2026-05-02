import { createHash } from "node:crypto";

import { z } from "zod";

import type { EmbeddingClient } from "../../embeddings/index.js";
import {
  LanceDbTable,
  float64Field,
  schema,
  utf8Field,
  vectorField,
} from "../../storage/lancedb/index.js";
import {
  quoteSqlString,
  toFloat32Array,
  type Float32ArrayCodecOptions,
} from "../../storage/codecs.js";
import { SqliteDatabase } from "../../storage/sqlite/index.js";
import { SystemClock, type Clock } from "../../util/clock.js";
import { ProvenanceError, StorageError } from "../../util/errors.js";
import {
  createOpenQuestionId,
  entityIdHelpers,
  openQuestionIdHelpers,
  parseOpenQuestionId,
  streamEntryIdHelpers,
  type EntityId,
  type OpenQuestionId,
  type StreamEntryId,
} from "../../util/ids.js";
import { serializeJsonValue } from "../../util/json-value.js";
import { episodeIdSchema } from "../episodic/types.js";
import { semanticNodeIdSchema } from "../semantic/types.js";
import {
  parseStoredProvenance,
  provenanceSchema,
  toStoredProvenance,
} from "../common/provenance.js";

export const OPEN_QUESTION_STATUSES = ["open", "resolved", "abandoned"] as const;
export const OPEN_QUESTION_SOURCES = [
  "user",
  "reflection",
  "contradiction",
  "ruminator",
  "overseer",
  "autonomy",
  "deliberator",
] as const;

export const openQuestionIdSchema = z
  .string()
  .refine((value) => openQuestionIdHelpers.is(value), {
    message: "Invalid open question id",
  })
  .transform((value) => value as OpenQuestionId);
export const openQuestionAudienceEntityIdSchema = z
  .string()
  .refine((value) => entityIdHelpers.is(value), {
    message: "Invalid open question audience entity id",
  })
  .transform((value) => value as EntityId);

export const openQuestionStatusSchema = z.enum(OPEN_QUESTION_STATUSES);
export const openQuestionSourceSchema = z.enum(OPEN_QUESTION_SOURCES);

export const openQuestionResolutionStreamEntryIdSchema = z
  .string()
  .refine((value) => streamEntryIdHelpers.is(value), {
    message: "Invalid open question resolution stream entry id",
  })
  .transform((value) => value as StreamEntryId);

export const openQuestionSchema = z
  .object({
    id: openQuestionIdSchema,
    question: z.string().min(1),
    urgency: z.number().min(0).max(1),
    status: openQuestionStatusSchema,
    audience_entity_id: openQuestionAudienceEntityIdSchema.nullable().default(null),
    related_episode_ids: z.array(episodeIdSchema),
    related_semantic_node_ids: z.array(semanticNodeIdSchema),
    provenance: provenanceSchema.nullable(),
    source: openQuestionSourceSchema,
    created_at: z.number().finite(),
    last_touched: z.number().finite(),
    resolution_evidence_episode_ids: z.array(episodeIdSchema),
    resolution_evidence_stream_entry_ids: z.array(openQuestionResolutionStreamEntryIdSchema),
    resolution_note: z.string().nullable(),
    resolved_at: z.number().finite().nullable(),
    abandoned_reason: z.string().nullable(),
    abandoned_at: z.number().finite().nullable(),
  })
  .refine(
    (value) =>
      value.related_episode_ids.length > 0 ||
      value.related_semantic_node_ids.length > 0 ||
      value.provenance !== null,
    {
      message:
        "Open question requires related_episode_ids, related_semantic_node_ids, or explicit provenance",
      path: ["provenance"],
    },
  )
  .refine(
    (value) =>
      value.status !== "resolved" ||
      value.resolution_evidence_episode_ids.length > 0 ||
      value.resolution_evidence_stream_entry_ids.length > 0,
    {
      message: "Resolved open question requires episode or stream evidence",
      path: ["resolution_evidence_episode_ids"],
    },
  );

export const openQuestionPatchSchema = z.object({
  question: z.string().min(1).optional(),
  urgency: z.number().min(0).max(1).optional(),
  status: openQuestionStatusSchema.optional(),
  audience_entity_id: openQuestionAudienceEntityIdSchema.nullable().optional(),
  related_episode_ids: z.array(episodeIdSchema).optional(),
  related_semantic_node_ids: z.array(semanticNodeIdSchema).optional(),
  provenance: provenanceSchema.nullable().optional(),
  source: openQuestionSourceSchema.optional(),
  last_touched: z.number().finite().optional(),
  resolution_evidence_episode_ids: z.array(episodeIdSchema).optional(),
  resolution_evidence_stream_entry_ids: z
    .array(openQuestionResolutionStreamEntryIdSchema)
    .optional(),
  resolution_note: z.string().nullable().optional(),
  resolved_at: z.number().finite().nullable().optional(),
  abandoned_reason: z.string().nullable().optional(),
  abandoned_at: z.number().finite().nullable().optional(),
});

export type OpenQuestion = z.infer<typeof openQuestionSchema>;
export type OpenQuestionPatch = z.infer<typeof openQuestionPatchSchema>;
export type OpenQuestionStatus = z.infer<typeof openQuestionStatusSchema>;
export type OpenQuestionSource = z.infer<typeof openQuestionSourceSchema>;

export type OpenQuestionSearchCandidate = {
  question: OpenQuestion;
  similarity: number;
};

export type OpenQuestionEmbeddingBackfillReport = {
  scanned: number;
  embedded: number;
  skipped: number;
  failed: number;
};

export type OpenQuestionEmbeddingFailureDetails = {
  operation: "insert" | "update" | "metadata_sync" | "backfill";
  questionId: OpenQuestionId;
  question: string;
};

type OpenQuestionVectorRow = {
  id: string;
  question: string;
  status: string;
  audience_entity_id: string | null;
  related_semantic_node_ids: string;
  urgency: number;
  created_at: number;
  last_touched: number;
  embedding: number[];
  _distance?: number;
};

export type OpenQuestionsRepositoryOptions = {
  db: SqliteDatabase;
  table?: LanceDbTable;
  embeddingClient?: EmbeddingClient;
  clock?: Clock;
  onEmbeddingFailure?: (
    error: unknown,
    details: OpenQuestionEmbeddingFailureDetails,
  ) => void | Promise<void>;
};

const OPEN_QUESTION_VECTOR_CODEC = {
  arrayLikeErrorMessage: "Open question embedding must be array-like",
  nonFiniteErrorMessage: "Open question embedding contains a non-finite value",
  errorCode: "OPEN_QUESTION_INVALID",
} satisfies Float32ArrayCodecOptions;

export function createOpenQuestionsTableSchema(dimensions: number) {
  return schema([
    utf8Field("id"),
    utf8Field("question"),
    utf8Field("status"),
    utf8Field("audience_entity_id", true),
    utf8Field("related_semantic_node_ids"),
    float64Field("urgency"),
    float64Field("created_at"),
    float64Field("last_touched"),
    vectorField("embedding", dimensions),
  ]);
}

function clamp(value: number, min: number, max: number): number {
  return Math.min(max, Math.max(min, value));
}

function parseIdArray<T>(value: string, schema: z.ZodType<T>, label: string): T[] {
  let parsed: unknown;

  try {
    parsed = JSON.parse(value) as unknown;
  } catch (error) {
    throw new StorageError(`Failed to parse ${label}`, {
      cause: error,
      code: "OPEN_QUESTION_INVALID",
    });
  }

  const result = z.array(schema).safeParse(parsed);

  if (!result.success) {
    throw new StorageError(`Invalid ${label}`, {
      cause: result.error,
      code: "OPEN_QUESTION_INVALID",
    });
  }

  return result.data;
}

function normalizeQuestionForDedupe(text: string): string {
  return text.normalize("NFKC").trim().toLowerCase().replace(/\s+/gu, " ");
}

function toSimilarity(distance: number | undefined): number {
  if (distance === undefined) {
    return 0;
  }

  return Math.max(0, Math.min(1, 1 - distance));
}

function getDistance(row: Record<string, unknown>): number | undefined {
  const value = row._distance;
  return typeof value === "number" && Number.isFinite(value) ? value : undefined;
}

function isVisibleToAudience(
  question: OpenQuestion,
  audienceEntityId: EntityId | null | undefined,
): boolean {
  if (audienceEntityId === null || audienceEntityId === undefined) {
    return question.audience_entity_id === null;
  }

  return question.audience_entity_id === null || question.audience_entity_id === audienceEntityId;
}

function buildVectorVisibilityWhereClause(
  audienceEntityId: EntityId | null | undefined,
): string | undefined {
  if (audienceEntityId === undefined) {
    return undefined;
  }

  if (audienceEntityId === null) {
    return "audience_entity_id IS NULL";
  }

  return `(audience_entity_id IS NULL OR audience_entity_id = ${quoteSqlString(audienceEntityId)})`;
}

function vectorRowFromQuestion(
  question: OpenQuestion,
  embedding: Float32Array,
): OpenQuestionVectorRow {
  return {
    id: question.id,
    question: question.question,
    status: question.status,
    audience_entity_id: question.audience_entity_id,
    related_semantic_node_ids: serializeJsonValue(question.related_semantic_node_ids),
    urgency: question.urgency,
    created_at: question.created_at,
    last_touched: question.last_touched,
    embedding: Array.from(embedding),
  };
}

export function buildOpenQuestionDedupeKey(input: {
  question: string;
  relatedEpisodeIds: readonly z.infer<typeof episodeIdSchema>[];
  relatedSemanticNodeIds: readonly z.infer<typeof semanticNodeIdSchema>[];
  audienceEntityId?: EntityId | null;
}): string {
  const relatedIdsPayload = JSON.stringify({
    relatedEpisodeIds: [...input.relatedEpisodeIds].sort(),
    relatedSemanticNodeIds: [...input.relatedSemanticNodeIds].sort(),
    ...(input.audienceEntityId === null || input.audienceEntityId === undefined
      ? {}
      : {
          audienceEntityId: input.audienceEntityId,
        }),
  });

  const questionHash = createHash("sha1")
    .update(normalizeQuestionForDedupe(input.question))
    .digest("hex");
  const relatedHash = createHash("sha1").update(relatedIdsPayload).digest("hex");

  return `v2:${questionHash}|${relatedHash}`;
}

function mapOpenQuestionRow(row: Record<string, unknown>): OpenQuestion {
  const parsed = openQuestionSchema.safeParse({
    id: row.id,
    question: row.question,
    urgency: Number(row.urgency),
    status: row.status,
    audience_entity_id:
      row.audience_entity_id === null || row.audience_entity_id === undefined
        ? null
        : row.audience_entity_id,
    related_episode_ids: parseIdArray(
      String(row.related_episode_ids ?? "[]"),
      episodeIdSchema,
      "open question related_episode_ids",
    ),
    related_semantic_node_ids: parseIdArray(
      String(row.related_semantic_node_ids ?? "[]"),
      semanticNodeIdSchema,
      "open question related_semantic_node_ids",
    ),
    provenance:
      row.provenance_kind === null || row.provenance_kind === undefined
        ? null
        : parseStoredProvenance({
            provenance_kind: row.provenance_kind,
            provenance_episode_ids: row.provenance_episode_ids,
            provenance_process: row.provenance_process,
          }),
    source: row.source,
    created_at: Number(row.created_at),
    last_touched: Number(row.last_touched),
    resolution_evidence_episode_ids: parseIdArray(
      String(row.resolution_evidence_episode_ids ?? "[]"),
      episodeIdSchema,
      "open question resolution_evidence_episode_ids",
    ),
    resolution_evidence_stream_entry_ids: parseIdArray(
      String(row.resolution_evidence_stream_entry_ids ?? "[]"),
      openQuestionResolutionStreamEntryIdSchema,
      "open question resolution_evidence_stream_entry_ids",
    ),
    resolution_note:
      row.resolution_note === null || row.resolution_note === undefined
        ? null
        : String(row.resolution_note),
    resolved_at:
      row.resolved_at === null || row.resolved_at === undefined ? null : Number(row.resolved_at),
    abandoned_reason:
      row.abandoned_reason === null || row.abandoned_reason === undefined
        ? null
        : String(row.abandoned_reason),
    abandoned_at:
      row.abandoned_at === null || row.abandoned_at === undefined ? null : Number(row.abandoned_at),
  });

  if (!parsed.success) {
    throw new StorageError("Open question row failed validation", {
      cause: parsed.error,
      code: "OPEN_QUESTION_INVALID",
    });
  }

  return parsed.data;
}

export class OpenQuestionsRepository {
  private readonly clock: Clock;
  private readonly pendingEmbeddingTasks = new Set<Promise<void>>();

  constructor(private readonly options: OpenQuestionsRepositoryOptions) {
    this.clock = options.clock ?? new SystemClock();
  }

  private get db(): SqliteDatabase {
    return this.options.db;
  }

  private get table(): LanceDbTable | undefined {
    return this.options.table;
  }

  private get embeddingClient(): EmbeddingClient | undefined {
    return this.options.embeddingClient;
  }

  private hasVectorStorage(): boolean {
    return this.table !== undefined && this.embeddingClient !== undefined;
  }

  private enqueueEmbeddingTask(task: Promise<void>): void {
    this.pendingEmbeddingTasks.add(task);
    void task.finally(() => {
      this.pendingEmbeddingTasks.delete(task);
    });
  }

  private reportEmbeddingFailure(
    error: unknown,
    details: OpenQuestionEmbeddingFailureDetails,
  ): void {
    try {
      void Promise.resolve(this.options.onEmbeddingFailure?.(error, details)).catch(() => {
        // Best-effort failure reporting only.
      });
    } catch {
      // Best-effort failure reporting only.
    }
  }

  private async readStoredEmbedding(id: OpenQuestionId): Promise<Float32Array | null> {
    const table = this.table;

    if (table === undefined) {
      return null;
    }

    const [row] = await table.list({
      where: `id = ${quoteSqlString(id)}`,
      limit: 1,
      columns: ["embedding"],
    });

    if (row === undefined) {
      return null;
    }

    return toFloat32Array(row.embedding, OPEN_QUESTION_VECTOR_CODEC);
  }

  private async upsertQuestionVector(
    question: OpenQuestion,
    operation: OpenQuestionEmbeddingFailureDetails["operation"],
    options: { forceEmbed?: boolean; skipIfMissing?: boolean } = {},
  ): Promise<void> {
    const table = this.table;
    const embeddingClient = this.embeddingClient;

    if (table === undefined || embeddingClient === undefined) {
      return;
    }

    try {
      const storedEmbedding =
        options.forceEmbed === true ? null : await this.readStoredEmbedding(question.id);

      if (options.skipIfMissing === true && storedEmbedding === null) {
        return;
      }

      const embedding = storedEmbedding ?? (await embeddingClient.embed(question.question));

      await table.upsert([vectorRowFromQuestion(question, embedding)], {
        on: "id",
      });
    } catch (error) {
      this.reportEmbeddingFailure(error, {
        operation,
        questionId: question.id,
        question: question.question,
      });
    }
  }

  private scheduleQuestionVectorUpsert(
    question: OpenQuestion,
    operation: OpenQuestionEmbeddingFailureDetails["operation"],
    options: { forceEmbed?: boolean; skipIfMissing?: boolean } = {},
  ): void {
    if (!this.hasVectorStorage()) {
      return;
    }

    this.enqueueEmbeddingTask(this.upsertQuestionVector(question, operation, options));
  }

  async waitForPendingEmbeddings(): Promise<void> {
    await Promise.allSettled([...this.pendingEmbeddingTasks]);
  }

  async getEmbeddedQuestionIds(ids: readonly OpenQuestionId[]): Promise<Set<OpenQuestionId>> {
    const table = this.table;
    const uniqueIds = [...new Set(ids)];

    if (table === undefined || uniqueIds.length === 0) {
      return new Set();
    }

    const rows = await table.list({
      where: `id IN (${uniqueIds.map((id) => quoteSqlString(id)).join(", ")})`,
      columns: ["id"],
    });

    return new Set(
      rows
        .map((row) => String(row.id))
        .filter((id) => openQuestionIdHelpers.is(id))
        .map((id) => parseOpenQuestionId(id)),
    );
  }

  async searchByVector(
    vector: Float32Array,
    options: {
      status?: OpenQuestionStatus;
      visibleToAudienceEntityId?: EntityId | null;
      limit?: number;
      minSimilarity?: number;
    } = {},
  ): Promise<OpenQuestionSearchCandidate[]> {
    const table = this.table;

    if (table === undefined) {
      return [];
    }

    const limit = Math.max(1, options.limit ?? 10);
    const clauses: string[] = [];

    if (options.status !== undefined) {
      clauses.push(`status = ${quoteSqlString(openQuestionStatusSchema.parse(options.status))}`);
    }

    const visibilityWhere = buildVectorVisibilityWhereClause(options.visibleToAudienceEntityId);

    if (visibilityWhere !== undefined) {
      clauses.push(visibilityWhere);
    }

    const rows = await table.search(Array.from(vector), {
      limit: Math.max(limit * 5, 20),
      vectorColumn: "embedding",
      distanceType: "cosine",
      where: clauses.length === 0 ? undefined : clauses.join(" AND "),
    });
    const results: OpenQuestionSearchCandidate[] = [];

    for (const row of rows) {
      const id = String(row.id);

      if (!openQuestionIdHelpers.is(id)) {
        continue;
      }

      const question = this.get(parseOpenQuestionId(id));

      if (question === null) {
        continue;
      }

      if (options.status !== undefined && question.status !== options.status) {
        continue;
      }

      if (
        options.visibleToAudienceEntityId !== undefined &&
        !isVisibleToAudience(question, options.visibleToAudienceEntityId)
      ) {
        continue;
      }

      const similarity = toSimilarity(getDistance(row));

      if (options.minSimilarity !== undefined && similarity < options.minSimilarity) {
        continue;
      }

      results.push({
        question,
        similarity,
      });

      if (results.length >= limit) {
        break;
      }
    }

    return results;
  }

  async backfillMissingEmbeddings(
    options: { limit?: number } = {},
  ): Promise<OpenQuestionEmbeddingBackfillReport> {
    const table = this.table;
    const embeddingClient = this.embeddingClient;
    const report: OpenQuestionEmbeddingBackfillReport = {
      scanned: 0,
      embedded: 0,
      skipped: 0,
      failed: 0,
    };

    if (table === undefined || embeddingClient === undefined) {
      return report;
    }

    const limitClause = options.limit === undefined ? "" : "LIMIT ?";
    const rows = this.db
      .prepare(
        `
          SELECT *
          FROM open_questions
          ORDER BY created_at ASC, id ASC
          ${limitClause}
        `,
      )
      .all(...(options.limit === undefined ? [] : [Math.max(1, options.limit)])) as Record<
      string,
      unknown
    >[];
    const questions = rows.map((row) => mapOpenQuestionRow(row));
    const existingIds = await this.getEmbeddedQuestionIds(questions.map((question) => question.id));

    for (const question of questions) {
      report.scanned += 1;

      if (existingIds.has(question.id)) {
        report.skipped += 1;
        continue;
      }

      try {
        const embedding = await embeddingClient.embed(question.question);
        await table.upsert([vectorRowFromQuestion(question, embedding)], {
          on: "id",
        });
        report.embedded += 1;
      } catch (error) {
        report.failed += 1;
        this.reportEmbeddingFailure(error, {
          operation: "backfill",
          questionId: question.id,
          question: question.question,
        });
      }
    }

    return report;
  }

  getByDedupeKey(key: string): OpenQuestion | null {
    const row = this.db
      .prepare("SELECT * FROM open_questions WHERE dedupe_key = ? LIMIT 1")
      .get(key) as Record<string, unknown> | undefined;

    return row === undefined ? null : mapOpenQuestionRow(row);
  }

  add(input: {
    id?: OpenQuestionId;
    question: string;
    urgency: number;
    related_episode_ids?: readonly z.infer<typeof episodeIdSchema>[];
    related_semantic_node_ids?: readonly z.infer<typeof semanticNodeIdSchema>[];
    audience_entity_id?: EntityId | null;
    provenance?: z.infer<typeof provenanceSchema> | null;
    source: OpenQuestionSource;
    created_at?: number;
    last_touched?: number;
  }): OpenQuestion {
    const relatedEpisodeIds = input.related_episode_ids ?? [];
    const relatedSemanticNodeIds = input.related_semantic_node_ids ?? [];
    if (
      relatedEpisodeIds.length === 0 &&
      relatedSemanticNodeIds.length === 0 &&
      input.provenance === undefined
    ) {
      throw new ProvenanceError("Open question requires related ids or explicit provenance", {
        code: "PROVENANCE_REQUIRED",
      });
    }
    const nowMs = this.clock.now();
    const question = openQuestionSchema.parse({
      id: input.id ?? createOpenQuestionId(),
      question: input.question,
      urgency: input.urgency,
      status: "open",
      audience_entity_id: input.audience_entity_id ?? null,
      related_episode_ids: relatedEpisodeIds,
      related_semantic_node_ids: relatedSemanticNodeIds,
      provenance: input.provenance ?? null,
      source: input.source,
      created_at: input.created_at ?? nowMs,
      last_touched: input.last_touched ?? nowMs,
      resolution_evidence_episode_ids: [],
      resolution_evidence_stream_entry_ids: [],
      resolution_note: null,
      resolved_at: null,
      abandoned_reason: null,
      abandoned_at: null,
    });
    const key = buildOpenQuestionDedupeKey({
      question: question.question,
      relatedEpisodeIds: question.related_episode_ids,
      relatedSemanticNodeIds: question.related_semantic_node_ids,
      audienceEntityId: question.audience_entity_id,
    });
    const existing = this.getByDedupeKey(key);

    if (existing !== null) {
      return existing;
    }
    const storedProvenance =
      question.provenance === null ? null : toStoredProvenance(question.provenance);

    this.db
      .prepare(
        `
          INSERT INTO open_questions (
            id, question, dedupe_key, urgency, status, audience_entity_id, related_episode_ids,
            related_semantic_node_ids, provenance_kind, provenance_episode_ids,
            provenance_process, source, created_at, last_touched, resolution_evidence_episode_ids,
            resolution_evidence_stream_entry_ids, resolution_note, resolved_at, abandoned_reason,
            abandoned_at
          ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        `,
      )
      .run(
        question.id,
        question.question,
        key,
        question.urgency,
        question.status,
        question.audience_entity_id,
        serializeJsonValue(question.related_episode_ids),
        serializeJsonValue(question.related_semantic_node_ids),
        storedProvenance?.provenance_kind ?? null,
        storedProvenance?.provenance_episode_ids ?? null,
        storedProvenance?.provenance_process ?? null,
        question.source,
        question.created_at,
        question.last_touched,
        serializeJsonValue(question.resolution_evidence_episode_ids),
        serializeJsonValue(question.resolution_evidence_stream_entry_ids),
        question.resolution_note,
        question.resolved_at,
        question.abandoned_reason,
        question.abandoned_at,
      );

    this.scheduleQuestionVectorUpsert(question, "insert", {
      forceEmbed: true,
    });

    return question;
  }

  list(
    options: {
      status?: OpenQuestionStatus;
      minUrgency?: number;
      visibleToAudienceEntityId?: EntityId | null;
      limit?: number;
    } = {},
  ): OpenQuestion[] {
    const filters: string[] = [];
    const values: unknown[] = [];

    if (options.status !== undefined) {
      openQuestionStatusSchema.parse(options.status);
      filters.push("status = ?");
      values.push(options.status);
    }

    if (options.minUrgency !== undefined) {
      filters.push("urgency >= ?");
      values.push(options.minUrgency);
    }

    if (options.visibleToAudienceEntityId !== undefined) {
      if (options.visibleToAudienceEntityId === null) {
        filters.push("audience_entity_id IS NULL");
      } else {
        filters.push("(audience_entity_id IS NULL OR audience_entity_id = ?)");
        values.push(openQuestionAudienceEntityIdSchema.parse(options.visibleToAudienceEntityId));
      }
    }

    const whereClause = filters.length === 0 ? "" : `WHERE ${filters.join(" AND ")}`;
    const limit = options.limit ?? 50;
    const rows = this.db
      .prepare(
        `
          SELECT *
          FROM open_questions
          ${whereClause}
          ORDER BY urgency DESC, last_touched DESC, created_at DESC
          LIMIT ?
        `,
      )
      .all(...values, limit) as Record<string, unknown>[];

    return rows.map((row) => mapOpenQuestionRow(row));
  }

  get(id: OpenQuestionId): OpenQuestion | null {
    const row = this.db.prepare("SELECT * FROM open_questions WHERE id = ?").get(id) as
      | Record<string, unknown>
      | undefined;

    return row === undefined ? null : mapOpenQuestionRow(row);
  }

  update(id: OpenQuestionId, patch: OpenQuestionPatch): OpenQuestion {
    const existing = this.get(id);

    if (existing === null) {
      throw new StorageError(`Unknown open question id: ${id}`, {
        code: "OPEN_QUESTION_NOT_FOUND",
      });
    }

    const parsedPatch = openQuestionPatchSchema.parse(patch);
    const next = openQuestionSchema.parse({
      ...existing,
      ...parsedPatch,
      last_touched: this.clock.now(),
    });
    const dedupeKey = buildOpenQuestionDedupeKey({
      question: next.question,
      relatedEpisodeIds: next.related_episode_ids,
      relatedSemanticNodeIds: next.related_semantic_node_ids,
      audienceEntityId: next.audience_entity_id,
    });
    const storedProvenance = next.provenance === null ? null : toStoredProvenance(next.provenance);

    this.db
      .prepare(
        `
          UPDATE open_questions
          SET question = ?, urgency = ?, status = ?, audience_entity_id = ?,
              related_episode_ids = ?, related_semantic_node_ids = ?, provenance_kind = ?,
              provenance_episode_ids = ?, provenance_process = ?, source = ?, last_touched = ?,
              resolution_evidence_episode_ids = ?, resolution_evidence_stream_entry_ids = ?,
              resolution_note = ?, resolved_at = ?, abandoned_reason = ?, abandoned_at = ?,
              dedupe_key = ?
          WHERE id = ?
        `,
      )
      .run(
        next.question,
        next.urgency,
        next.status,
        next.audience_entity_id,
        serializeJsonValue(next.related_episode_ids),
        serializeJsonValue(next.related_semantic_node_ids),
        storedProvenance?.provenance_kind ?? null,
        storedProvenance?.provenance_episode_ids ?? null,
        storedProvenance?.provenance_process ?? null,
        next.source,
        next.last_touched,
        serializeJsonValue(next.resolution_evidence_episode_ids),
        serializeJsonValue(next.resolution_evidence_stream_entry_ids),
        next.resolution_note,
        next.resolved_at,
        next.abandoned_reason,
        next.abandoned_at,
        dedupeKey,
        id,
      );

    this.scheduleQuestionVectorUpsert(next, next.status === "open" ? "update" : "metadata_sync", {
      forceEmbed: parsedPatch.question !== undefined,
      skipIfMissing: next.status !== "open",
    });

    return next;
  }

  touch(id: OpenQuestionId, now = this.clock.now()): OpenQuestion {
    const existing = this.get(id);

    if (existing === null) {
      throw new StorageError(`Unknown open question id: ${id}`, {
        code: "OPEN_QUESTION_NOT_FOUND",
      });
    }

    const nextUrgency =
      existing.status === "open" ? clamp(existing.urgency + 0.02, 0, 1) : existing.urgency;
    this.db
      .prepare("UPDATE open_questions SET urgency = ?, last_touched = ? WHERE id = ?")
      .run(nextUrgency, now, id);

    const touched = {
      ...existing,
      urgency: nextUrgency,
      last_touched: now,
    };

    this.scheduleQuestionVectorUpsert(touched, "metadata_sync", {
      skipIfMissing: true,
    });

    return touched;
  }

  resolve(
    id: OpenQuestionId,
    input: {
      resolution_evidence_episode_ids?: readonly z.infer<typeof episodeIdSchema>[];
      resolution_evidence_stream_entry_ids?: readonly z.infer<
        typeof openQuestionResolutionStreamEntryIdSchema
      >[];
      resolution_note: string;
    },
  ): OpenQuestion {
    const existing = this.get(id);

    if (existing === null) {
      throw new StorageError(`Unknown open question id: ${id}`, {
        code: "OPEN_QUESTION_NOT_FOUND",
      });
    }

    if (existing.status !== "open") {
      throw new StorageError(`Cannot resolve open question in status ${existing.status}`, {
        code: "OPEN_QUESTION_INVALID_TRANSITION",
      });
    }

    const resolutionEvidenceEpisodeIds = [...new Set(input.resolution_evidence_episode_ids ?? [])];
    const resolutionEvidenceStreamEntryIds = [
      ...new Set(input.resolution_evidence_stream_entry_ids ?? []),
    ];

    if (
      resolutionEvidenceEpisodeIds.length === 0 &&
      resolutionEvidenceStreamEntryIds.length === 0
    ) {
      throw new StorageError("Open question resolution requires episode or stream evidence", {
        code: "OPEN_QUESTION_RESOLUTION_EVIDENCE_REQUIRED",
      });
    }

    const resolvedAt = this.clock.now();
    this.db
      .prepare(
        `
          UPDATE open_questions
          SET status = 'resolved', resolution_evidence_episode_ids = ?,
              resolution_evidence_stream_entry_ids = ?, resolution_note = ?, resolved_at = ?,
              abandoned_reason = NULL, abandoned_at = NULL, last_touched = ?
          WHERE id = ?
        `,
      )
      .run(
        serializeJsonValue(resolutionEvidenceEpisodeIds),
        serializeJsonValue(resolutionEvidenceStreamEntryIds),
        input.resolution_note,
        resolvedAt,
        resolvedAt,
        id,
      );

    const resolved: OpenQuestion = {
      ...existing,
      status: "resolved",
      resolution_evidence_episode_ids: resolutionEvidenceEpisodeIds,
      resolution_evidence_stream_entry_ids: resolutionEvidenceStreamEntryIds,
      resolution_note: input.resolution_note,
      resolved_at: resolvedAt,
      abandoned_reason: null,
      abandoned_at: null,
      last_touched: resolvedAt,
    };

    this.scheduleQuestionVectorUpsert(resolved, "metadata_sync", {
      skipIfMissing: true,
    });

    return resolved;
  }

  abandon(id: OpenQuestionId, reason: string): OpenQuestion {
    const existing = this.get(id);

    if (existing === null) {
      throw new StorageError(`Unknown open question id: ${id}`, {
        code: "OPEN_QUESTION_NOT_FOUND",
      });
    }

    if (existing.status !== "open") {
      throw new StorageError(`Cannot abandon open question in status ${existing.status}`, {
        code: "OPEN_QUESTION_INVALID_TRANSITION",
      });
    }

    const abandonedAt = this.clock.now();
    this.db
      .prepare(
        `
          UPDATE open_questions
          SET status = 'abandoned', abandoned_reason = ?, abandoned_at = ?,
              resolution_evidence_episode_ids = '[]', resolution_evidence_stream_entry_ids = '[]',
              resolution_note = NULL, resolved_at = NULL, last_touched = ?
          WHERE id = ?
        `,
      )
      .run(reason, abandonedAt, abandonedAt, id);

    const abandoned: OpenQuestion = {
      ...existing,
      status: "abandoned",
      resolution_evidence_episode_ids: [],
      resolution_evidence_stream_entry_ids: [],
      resolution_note: null,
      resolved_at: null,
      abandoned_reason: reason,
      abandoned_at: abandonedAt,
      last_touched: abandonedAt,
    };

    this.scheduleQuestionVectorUpsert(abandoned, "metadata_sync", {
      skipIfMissing: true,
    });

    return abandoned;
  }

  bumpUrgency(id: OpenQuestionId, delta: number): OpenQuestion {
    const existing = this.get(id);

    if (existing === null) {
      throw new StorageError(`Unknown open question id: ${id}`, {
        code: "OPEN_QUESTION_NOT_FOUND",
      });
    }

    const nextUrgency = clamp(existing.urgency + delta, 0, 1);
    const nowMs = this.clock.now();
    this.db
      .prepare("UPDATE open_questions SET urgency = ?, last_touched = ? WHERE id = ?")
      .run(nextUrgency, nowMs, id);

    const bumped = {
      ...existing,
      urgency: nextUrgency,
      last_touched: nowMs,
    };

    this.scheduleQuestionVectorUpsert(bumped, "metadata_sync", {
      skipIfMissing: true,
    });

    return bumped;
  }

  setUrgency(id: OpenQuestionId, urgency: number): OpenQuestion {
    const existing = this.get(id);

    if (existing === null) {
      throw new StorageError(`Unknown open question id: ${id}`, {
        code: "OPEN_QUESTION_NOT_FOUND",
      });
    }

    const nextUrgency = clamp(urgency, 0, 1);
    const nowMs = this.clock.now();
    this.db
      .prepare("UPDATE open_questions SET urgency = ?, last_touched = ? WHERE id = ?")
      .run(nextUrgency, nowMs, id);

    const updated = {
      ...existing,
      urgency: nextUrgency,
      last_touched: nowMs,
    };

    this.scheduleQuestionVectorUpsert(updated, "metadata_sync", {
      skipIfMissing: true,
    });

    return updated;
  }

  reopenForReversal(id: OpenQuestionId, urgency?: number): OpenQuestion {
    const existing = this.get(id);

    if (existing === null) {
      throw new StorageError(`Unknown open question id: ${id}`, {
        code: "OPEN_QUESTION_NOT_FOUND",
      });
    }

    const nowMs = this.clock.now();
    const nextUrgency = clamp(urgency ?? existing.urgency, 0, 1);
    this.db
      .prepare(
        `
          UPDATE open_questions
          SET status = 'open', urgency = ?, last_touched = ?,
              resolution_evidence_episode_ids = '[]',
              resolution_evidence_stream_entry_ids = '[]', resolution_note = NULL,
              resolved_at = NULL, abandoned_reason = NULL, abandoned_at = NULL
          WHERE id = ?
        `,
      )
      .run(nextUrgency, nowMs, id);

    const reopened: OpenQuestion = {
      ...existing,
      status: "open",
      urgency: nextUrgency,
      last_touched: nowMs,
      resolution_evidence_episode_ids: [],
      resolution_evidence_stream_entry_ids: [],
      resolution_note: null,
      resolved_at: null,
      abandoned_reason: null,
      abandoned_at: null,
    };

    this.scheduleQuestionVectorUpsert(reopened, "update");

    return reopened;
  }
}

export function createOpenQuestionReopener(repository: OpenQuestionsRepository) {
  return (id: OpenQuestionId, urgency?: number) => repository.reopenForReversal(id, urgency);
}
