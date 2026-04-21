import { createHash } from "node:crypto";

import { z } from "zod";

import { SqliteDatabase } from "../../storage/sqlite/index.js";
import { SystemClock, type Clock } from "../../util/clock.js";
import { StorageError } from "../../util/errors.js";
import {
  createOpenQuestionId,
  openQuestionIdHelpers,
  type OpenQuestionId,
} from "../../util/ids.js";
import { serializeJsonValue } from "../../util/json-value.js";
import { tokenizeText } from "../../util/text/tokenize.js";
import { episodeIdSchema } from "../episodic/types.js";
import { semanticNodeIdSchema } from "../semantic/types.js";

export const OPEN_QUESTION_STATUSES = ["open", "resolved", "abandoned"] as const;
export const OPEN_QUESTION_SOURCES = [
  "user",
  "reflection",
  "contradiction",
  "ruminator",
  "overseer",
] as const;

export const openQuestionIdSchema = z
  .string()
  .refine((value) => openQuestionIdHelpers.is(value), {
    message: "Invalid open question id",
  })
  .transform((value) => value as OpenQuestionId);

export const openQuestionStatusSchema = z.enum(OPEN_QUESTION_STATUSES);
export const openQuestionSourceSchema = z.enum(OPEN_QUESTION_SOURCES);

export const openQuestionSchema = z.object({
  id: openQuestionIdSchema,
  question: z.string().min(1),
  urgency: z.number().min(0).max(1),
  status: openQuestionStatusSchema,
  related_episode_ids: z.array(episodeIdSchema),
  related_semantic_node_ids: z.array(semanticNodeIdSchema),
  source: openQuestionSourceSchema,
  created_at: z.number().finite(),
  last_touched: z.number().finite(),
  resolution_episode_id: episodeIdSchema.nullable(),
  resolution_note: z.string().nullable(),
  resolved_at: z.number().finite().nullable(),
  abandoned_reason: z.string().nullable(),
  abandoned_at: z.number().finite().nullable(),
});

export type OpenQuestion = z.infer<typeof openQuestionSchema>;
export type OpenQuestionStatus = z.infer<typeof openQuestionStatusSchema>;
export type OpenQuestionSource = z.infer<typeof openQuestionSourceSchema>;

export type OpenQuestionsRepositoryOptions = {
  db: SqliteDatabase;
  clock?: Clock;
};

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

function normalizeQuestion(text: string): string {
  const tokens = [...tokenizeText(text)];
  return tokens.length === 0 ? text.trim().toLowerCase().replace(/\s+/g, " ") : tokens.join(" ");
}

export function buildOpenQuestionDedupeKey(input: {
  question: string;
  relatedEpisodeIds: readonly z.infer<typeof episodeIdSchema>[];
  relatedSemanticNodeIds: readonly z.infer<typeof semanticNodeIdSchema>[];
}): string {
  const relatedIdsPayload = JSON.stringify({
    relatedEpisodeIds: [...input.relatedEpisodeIds].sort(),
    relatedSemanticNodeIds: [...input.relatedSemanticNodeIds].sort(),
  });

  return `${normalizeQuestion(input.question)}|${createHash("sha1").update(relatedIdsPayload).digest("hex")}`;
}

function mapOpenQuestionRow(row: Record<string, unknown>): OpenQuestion {
  const parsed = openQuestionSchema.safeParse({
    id: row.id,
    question: row.question,
    urgency: Number(row.urgency),
    status: row.status,
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
    source: row.source,
    created_at: Number(row.created_at),
    last_touched: Number(row.last_touched),
    resolution_episode_id:
      row.resolution_episode_id === null || row.resolution_episode_id === undefined
        ? null
        : row.resolution_episode_id,
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

  constructor(private readonly options: OpenQuestionsRepositoryOptions) {
    this.clock = options.clock ?? new SystemClock();
  }

  private get db(): SqliteDatabase {
    return this.options.db;
  }

  private getByDedupeKey(key: string): OpenQuestion | null {
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
    source: OpenQuestionSource;
    created_at?: number;
    last_touched?: number;
  }): OpenQuestion {
    const relatedEpisodeIds = input.related_episode_ids ?? [];
    const relatedSemanticNodeIds = input.related_semantic_node_ids ?? [];
    const key = buildOpenQuestionDedupeKey({
      question: input.question,
      relatedEpisodeIds,
      relatedSemanticNodeIds,
    });
    const existing = this.getByDedupeKey(key);

    if (existing !== null) {
      return existing;
    }

    const nowMs = this.clock.now();
    const question = openQuestionSchema.parse({
      id: input.id ?? createOpenQuestionId(),
      question: input.question,
      urgency: input.urgency,
      status: "open",
      related_episode_ids: relatedEpisodeIds,
      related_semantic_node_ids: relatedSemanticNodeIds,
      source: input.source,
      created_at: input.created_at ?? nowMs,
      last_touched: input.last_touched ?? nowMs,
      resolution_episode_id: null,
      resolution_note: null,
      resolved_at: null,
      abandoned_reason: null,
      abandoned_at: null,
    });

    this.db
      .prepare(
        `
          INSERT INTO open_questions (
            id, question, dedupe_key, urgency, status, related_episode_ids, related_semantic_node_ids,
            source, created_at, last_touched, resolution_episode_id, resolution_note, resolved_at,
            abandoned_reason, abandoned_at
          ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, NULL, NULL, NULL, NULL, NULL)
        `,
      )
      .run(
        question.id,
        question.question,
        key,
        question.urgency,
        question.status,
        serializeJsonValue(question.related_episode_ids),
        serializeJsonValue(question.related_semantic_node_ids),
        question.source,
        question.created_at,
        question.last_touched,
      );

    return question;
  }

  list(
    options: {
      status?: OpenQuestionStatus;
      minUrgency?: number;
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

    return {
      ...existing,
      urgency: nextUrgency,
      last_touched: now,
    };
  }

  resolve(
    id: OpenQuestionId,
    input: {
      resolution_episode_id: z.infer<typeof episodeIdSchema>;
      resolution_note?: string | null;
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

    const resolvedAt = this.clock.now();
    this.db
      .prepare(
        `
          UPDATE open_questions
          SET status = 'resolved', resolution_episode_id = ?, resolution_note = ?, resolved_at = ?,
              abandoned_reason = NULL, abandoned_at = NULL, last_touched = ?
          WHERE id = ?
        `,
      )
      .run(input.resolution_episode_id, input.resolution_note ?? null, resolvedAt, resolvedAt, id);

    return {
      ...existing,
      status: "resolved",
      resolution_episode_id: input.resolution_episode_id,
      resolution_note: input.resolution_note ?? null,
      resolved_at: resolvedAt,
      abandoned_reason: null,
      abandoned_at: null,
      last_touched: resolvedAt,
    };
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
              resolution_episode_id = NULL, resolution_note = NULL, resolved_at = NULL,
              last_touched = ?
          WHERE id = ?
        `,
      )
      .run(reason, abandonedAt, abandonedAt, id);

    return {
      ...existing,
      status: "abandoned",
      resolution_episode_id: null,
      resolution_note: null,
      resolved_at: null,
      abandoned_reason: reason,
      abandoned_at: abandonedAt,
      last_touched: abandonedAt,
    };
  }

  bumpUrgency(id: OpenQuestionId, delta: number): OpenQuestion {
    const existing = this.get(id);

    if (existing === null) {
      throw new StorageError(`Unknown open question id: ${id}`, {
        code: "OPEN_QUESTION_NOT_FOUND",
      });
    }

    const nextUrgency = clamp(existing.urgency + delta, 0, 1);
    this.db
      .prepare("UPDATE open_questions SET urgency = ?, last_touched = ? WHERE id = ?")
      .run(nextUrgency, this.clock.now(), id);

    return {
      ...existing,
      urgency: nextUrgency,
      last_touched: this.clock.now(),
    };
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

    return {
      ...existing,
      urgency: nextUrgency,
      last_touched: nowMs,
    };
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
          SET status = 'open', urgency = ?, last_touched = ?, resolution_episode_id = NULL,
              resolution_note = NULL, resolved_at = NULL, abandoned_reason = NULL, abandoned_at = NULL
          WHERE id = ?
        `,
      )
      .run(nextUrgency, nowMs, id);

    return {
      ...existing,
      status: "open",
      urgency: nextUrgency,
      last_touched: nowMs,
      resolution_episode_id: null,
      resolution_note: null,
      resolved_at: null,
      abandoned_reason: null,
      abandoned_at: null,
    };
  }
}

export function createOpenQuestionReopener(repository: OpenQuestionsRepository) {
  return (id: OpenQuestionId, urgency?: number) => repository.reopenForReversal(id, urgency);
}
