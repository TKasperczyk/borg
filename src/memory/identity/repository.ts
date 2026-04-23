import { z } from "zod";

import { SqliteDatabase } from "../../storage/sqlite/index.js";
import { SystemClock, type Clock } from "../../util/clock.js";
import { StorageError } from "../../util/errors.js";
import { serializeJsonValue, type JsonValue } from "../../util/json-value.js";
import { parseStoredProvenance, toStoredProvenance, type Provenance } from "../common/provenance.js";

import {
  identityEventSchema,
  identityRecordTypeSchema,
  type IdentityEvent,
  type IdentityRecordType,
} from "./types.js";

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

function mapRow(row: Record<string, unknown>): IdentityEvent {
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
};

export class IdentityEventRepository {
  private readonly clock: Clock;

  constructor(private readonly options: IdentityEventRepositoryOptions) {
    this.clock = options.clock ?? new SystemClock();
  }

  private get db(): SqliteDatabase {
    return this.options.db;
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
            provenance_kind, provenance_episode_ids, provenance_process, review_item_id,
            overwrite_without_review, ts
          ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
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

  list(options: {
    recordType?: IdentityRecordType;
    recordId?: string;
    limit?: number;
  } = {}): IdentityEvent[] {
    const filters: string[] = [];
    const values: unknown[] = [];
    const limit = z.number().int().positive().parse(options.limit ?? 50);

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
