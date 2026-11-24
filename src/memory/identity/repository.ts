import { z } from "zod";
import type { TurnTracer } from "../../tracing/tracer.js";

import { SqliteDatabase } from "../../storage/sqlite/index.js";
import { SystemClock, type Clock } from "../../util/clock.js";
import { StorageError } from "../../util/errors.js";
import { serializeJsonValue, type JsonValue } from "../../util/json-value.js";
import {
  parseStoredProvenance,
  toStoredProvenance,
  type Provenance,
} from "../common/provenance.js";

import {
  identityEventSchema,
  identityRecordTypeSchema,
  type IdentityEvent,
  type IdentityRecordType,
} from "./types.js";
import {
  annotationTargetKey,
  auditedAnnotationPayloadSchema,
  type AnnotationTarget,
  type TargetAnnotations,
} from "./annotations.js";

const ANNOTATION_TARGET_QUERY_BATCH_SIZE = 400;

function parseJsonValue(value: unknown, label: string): JsonValue | null {
  if (value === null || value === undefined) {
    return null;
  }

  if (typeof value !== "string") {
    throw new StorageError(`Invalid identity event ${label}`, {
      code: "IDENTITY_EVENT_INVALID",
    });
  }

  try {
    return JSON.parse(value) as JsonValue;
  } catch (error) {
    throw new StorageError(`Failed to parse identity event ${label}`, {
      cause: error,
      code: "IDENTITY_EVENT_INVALID",
    });
  }
}

export function mapRow(row: Record<string, unknown>): IdentityEvent {
  const parsed = identityEventSchema.safeParse({
    id: Number(row.id),
    record_type: row.record_type,
    record_id: row.record_id,
    action: row.action,
    old_value: parseJsonValue(row.old_value_json, "old_value_json"),
    new_value: parseJsonValue(row.new_value_json, "new_value_json"),
    reason: row.reason === null || row.reason === undefined ? null : String(row.reason),
    provenance: parseStoredProvenance({
      provenance_kind: row.provenance_kind,
      provenance_episode_ids: row.provenance_episode_ids,
      provenance_stream_entry_ids: row.provenance_stream_entry_ids,
      provenance_process: row.provenance_process,
    }),
    review_item_id:
      row.review_item_id === null || row.review_item_id === undefined
        ? null
        : Number(row.review_item_id),
    overwrite_without_review:
      row.overwrite_without_review === true || Number(row.overwrite_without_review) === 1,
    ts: Number(row.ts),
  });

  if (!parsed.success) {
    throw new StorageError("Identity event row failed validation", {
      cause: parsed.error,
      code: "IDENTITY_EVENT_INVALID",
    });
  }

  return parsed.data;
}

export type IdentityEventRepositoryOptions = {
  db: SqliteDatabase;
  clock?: Clock;
  tracer?: TurnTracer;
};

export class IdentityEventRepository {
  private readonly clock: Clock;

  constructor(private readonly options: IdentityEventRepositoryOptions) {
    this.clock = options.clock ?? new SystemClock();
  }

  private get db(): SqliteDatabase {
    return this.options.db;
  }

  runInTransaction<T>(callback: () => T): T {
    return this.db.raw.transaction(callback)();
  }

  record(input: {
    record_type: IdentityRecordType;
    record_id: string;
    action: string;
    old_value?: JsonValue | null;
    new_value?: JsonValue | null;
    reason?: string | null;
    provenance: Provenance;
    review_item_id?: number | null;
    overwrite_without_review?: boolean;
    ts?: number;
  }): IdentityEvent {
    const recordType = identityRecordTypeSchema.parse(input.record_type);
    const storedProvenance = toStoredProvenance(input.provenance);
    const result = this.db
      .prepare(
        `
          INSERT INTO identity_events (
            record_type, record_id, action, old_value_json, new_value_json, reason,
            provenance_kind, provenance_episode_ids, provenance_stream_entry_ids,
            provenance_process, review_item_id, overwrite_without_review, ts
          ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        `,
      )
      .run(
        recordType,
        input.record_id,
        input.action,
        input.old_value === undefined ? null : serializeJsonValue(input.old_value),
        input.new_value === undefined ? null : serializeJsonValue(input.new_value),
        input.reason ?? null,
        storedProvenance.provenance_kind,
        storedProvenance.provenance_episode_ids,
        storedProvenance.provenance_stream_entry_ids ?? "[]",
        storedProvenance.provenance_process,
        input.review_item_id ?? null,
        input.overwrite_without_review === true ? 1 : 0,
        input.ts ?? this.clock.now(),
      );

    const row = this.db
      .prepare("SELECT * FROM identity_events WHERE id = ?")
      .get(result.lastInsertRowid) as Record<string, unknown> | undefined;

    if (row === undefined) {
      throw new StorageError("Failed to read back identity event", {
        code: "IDENTITY_EVENT_INSERT_FAILED",
      });
    }

    return mapRow(row);
  }

  list(
    options: {
      recordType?: IdentityRecordType;
      recordId?: string;
      limit?: number;
    } = {},
  ): IdentityEvent[] {
    const filters: string[] = [];
    const values: unknown[] = [];
    const limit = z
      .number()
      .int()
      .positive()
      .parse(options.limit ?? 50);

    if (options.recordType !== undefined) {
      filters.push("record_type = ?");
      values.push(identityRecordTypeSchema.parse(options.recordType));
    }

    if (options.recordId !== undefined) {
      filters.push("record_id = ?");
      values.push(options.recordId);
    }

    const whereClause = filters.length === 0 ? "" : `WHERE ${filters.join(" AND ")}`;
    const rows = this.db
      .prepare(
        `
          SELECT *
          FROM identity_events
          ${whereClause}
          ORDER BY ts DESC, id DESC
          LIMIT ?
        `,
      )
      .all(...values, limit) as Record<string, unknown>[];

    return rows.map((row) => mapRow(row));
  }

  /** All audited annotations for exact typed targets, with named retractions applied. */
  listAnnotationsForTargets(
    targets: readonly AnnotationTarget[],
    options: { includeRetracted?: boolean } = {},
  ): TargetAnnotations[] {
    const uniqueTargets = [
      ...new Map(targets.map((target) => [annotationTargetKey(target), target])).values(),
    ];
    const results = new Map(
      uniqueTargets.map((target) => [
        annotationTargetKey(target),
        { ...target, annotations: [] } as TargetAnnotations,
      ]),
    );

    const skipped: Record<string, number> = {};
    const skip = (reason: string) => {
      skipped[reason] = (skipped[reason] ?? 0) + 1;
    };
    for (
      let offset = 0;
      offset < uniqueTargets.length;
      offset += ANNOTATION_TARGET_QUERY_BATCH_SIZE
    ) {
      const batch = uniqueTargets.slice(offset, offset + ANNOTATION_TARGET_QUERY_BATCH_SIZE);
      // The typed target join uses idx_identity_events_record_ts. No global history window
      // or action-name filter can discard an older annotation or its later retraction.
      const rows = this.db
        .prepare(
          `
        WITH targets(record_type, record_id) AS (VALUES ${batch.map(() => "(?, ?)").join(", ")})
        SELECT e.record_type, e.record_id, e.new_value_json,
          CASE WHEN typeof(e.ts) IN ('integer', 'real') THEN CAST(e.ts AS REAL) END AS ts
        FROM targets t JOIN identity_events e
          ON e.record_type = t.record_type AND e.record_id = t.record_id
        WHERE CASE
          WHEN e.new_value_json IS NULL THEN 0
          WHEN json_valid(e.new_value_json) = 0 THEN 1
          ELSE json_type(e.new_value_json, '$.annotation') IS NOT NULL
        END
        ORDER BY e.ts DESC, e.id DESC
      `,
        )
        .all(...batch.flatMap((target) => [target.record_type, target.record_id]));
      const retracted = new Map<string, Set<string>>();
      const seen = new Map<string, Set<string>>();
      for (const row of rows) {
        const recordType = identityRecordTypeSchema.safeParse(row.record_type);
        if (!recordType.success || typeof row.record_id !== "string") {
          skip("invalid_target");
          continue;
        }
        const target = { record_type: recordType.data, record_id: row.record_id };
        const key = annotationTargetKey(target);
        let payload: unknown;
        try {
          payload = JSON.parse(String(row.new_value_json));
        } catch {
          skip("invalid_json");
          continue;
        }
        const parsed = auditedAnnotationPayloadSchema.safeParse(payload);
        if (!parsed.success) {
          skip("invalid_payload");
          continue;
        }
        const { stream_entry_id, annotation } = parsed.data;
        const namesTarget =
          annotation.target_ids?.includes(target.record_id) ??
          (annotation.event === "episode_annotation" &&
            target.record_type === "episode" &&
            annotation.episode_id === target.record_id);
        if (!namesTarget) {
          skip("target_mismatch");
          continue;
        }
        if (annotation.event === "episode_annotation_retracted") {
          const ids = retracted.get(key) ?? new Set<string>();
          ids.add(annotation.reverses_stream_entry_id);
          retracted.set(key, ids);
          continue;
        }
        const text = annotation.correction ?? annotation.note;
        if (text === undefined) {
          skip("missing_text");
          continue;
        }
        const ids = seen.get(key) ?? new Set<string>();
        if (ids.has(stream_entry_id)) continue;
        ids.add(stream_entry_id);
        seen.set(key, ids);
        const timestamp = typeof row.ts === "number" ? row.ts : NaN;
        const writtenAt = Number.isFinite(new Date(timestamp).getTime()) ? timestamp : undefined;
        if (writtenAt === undefined) skip("invalid_timestamp");
        results.get(key)!.annotations.push({
          stream_entry_id,
          text,
          ...(writtenAt === undefined ? {} : { written_at: writtenAt }),
          authority: annotation.authority ?? null,
          ...(annotation.self_correction === undefined
            ? {}
            : { self_correction: { ...annotation.self_correction, original_record: target } }),
        });
      }
      for (const target of batch) {
        const key = annotationTargetKey(target);
        const result = results.get(key)!;
        for (const annotation of result.annotations) {
          const correction = annotation.self_correction;
          if (
            correction?.corrects_span !== undefined &&
            correction.targets_correction_seq !== null
          ) {
            // Retraction hides an annotation from display, but leaves its quoted text intact.
            const quoted = result.annotations.find(
              (row) => row.self_correction?.seq === correction.targets_correction_seq,
            );
            if (quoted !== undefined) annotation.quoted_correction_text = quoted.text;
          }
        }
        const retractedIds = retracted.get(key);
        if (retractedIds === undefined) continue;
        result.annotations = result.annotations.flatMap((annotation) =>
          !retractedIds.has(annotation.stream_entry_id)
            ? [annotation]
            : options.includeRetracted === true
              ? [{ ...annotation, retracted: true as const }]
              : [],
        );
      }
    }
    if (Object.keys(skipped).length > 0 && this.options.tracer?.enabled === true) {
      this.options.tracer.emit("memory.annotations.skipped", {
        turnId: "memory-annotation-read",
        counts_by_reason: skipped,
      });
    }
    return [...results.values()];
  }

  /** Historical goal transitions, independent of the goal's current state or audience. */
  listGoalDispositionsInRange(input: {
    sinceMs: number;
    untilMs: number;
    limit: number;
    cursor?: { ts: number; id: number };
  }): IdentityEvent[] {
    const sinceMs = z.number().int().finite().parse(input.sinceMs);
    const untilMs = z.number().int().finite().parse(input.untilMs);
    const limit = z.number().int().positive().parse(input.limit);
    const cursorTs = input.cursor === undefined ? null : z.number().int().parse(input.cursor.ts);
    const cursorId =
      input.cursor === undefined ? null : z.number().int().positive().parse(input.cursor.id);
    // Read the recorded status transition, not the writer's free-text reason or
    // process name. This also reaches automatic unblocks and historical retirements.
    const rows = this.db
      .prepare(
        `
          SELECT * FROM identity_events
          WHERE record_type = 'goal' AND ts >= ? AND ts <= ?
            AND json_extract(old_value_json, '$.status') != json_extract(new_value_json, '$.status')
            AND (
              json_extract(new_value_json, '$.status') IN ('done', 'superseded', 'abandoned', 'blocked')
              OR (json_extract(old_value_json, '$.status') = 'blocked'
                AND json_extract(new_value_json, '$.status') = 'active')
            )
            AND (? IS NULL OR ts < ? OR (ts = ? AND id < ?))
          ORDER BY ts DESC, id DESC
          LIMIT ?
        `,
      )
      .all(sinceMs, untilMs, cursorTs, cursorTs, cursorTs, cursorId, limit) as Record<
      string,
      unknown
    >[];

    return rows.map(mapRow);
  }

  findByReviewKey(input: {
    reviewItemId: number;
    recordType: IdentityRecordType;
    recordId: string;
    action: string;
  }): IdentityEvent | null {
    const row = this.db
      .prepare(
        `
          SELECT *
          FROM identity_events
          WHERE review_item_id = ?
            AND record_type = ?
            AND record_id = ?
            AND action = ?
          ORDER BY ts DESC, id DESC
          LIMIT 1
        `,
      )
      .get(
        z.number().int().positive().parse(input.reviewItemId),
        identityRecordTypeSchema.parse(input.recordType),
        input.recordId,
        input.action,
      ) as Record<string, unknown> | undefined;

    return row === undefined ? null : mapRow(row);
  }
}
