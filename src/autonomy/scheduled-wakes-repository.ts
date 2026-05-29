import { z } from "zod";

import type { SqliteDatabase } from "../storage/sqlite/index.js";
import { SystemClock, type Clock } from "../util/clock.js";
import { StorageError } from "../util/errors.js";
import {
  createScheduledWakeId,
  parseScheduledWakeId,
  scheduledWakeIdHelpers,
  type ScheduledWakeId,
} from "../util/ids.js";

export const SCHEDULED_WAKE_STATUSES = ["pending", "fired", "cancelled"] as const;
export type ScheduledWakeStatus = (typeof SCHEDULED_WAKE_STATUSES)[number];

const SCHEDULED_WAKE_COLUMNS =
  "id, fire_at, note, origin_session_id, status, created_at, updated_at, fired_at, cancelled_at";

export const scheduledWakeSchema = z.object({
  id: z
    .string()
    .refine((value) => scheduledWakeIdHelpers.is(value), {
      message: "Invalid scheduled wake id",
    })
    .transform((value) => parseScheduledWakeId(value)),
  fire_at: z.number().int().finite(),
  note: z.string().min(1),
  origin_session_id: z.string().min(1).nullable(),
  status: z.enum(SCHEDULED_WAKE_STATUSES),
  created_at: z.number().int().finite(),
  updated_at: z.number().int().finite(),
  fired_at: z.number().int().finite().nullable(),
  cancelled_at: z.number().int().finite().nullable(),
});

export type ScheduledWake = z.infer<typeof scheduledWakeSchema>;

export type ScheduledWakeScheduleInput = {
  delaySeconds: number;
  note: string;
  originSessionId?: string | null;
};

export type ScheduledWakeListInput = {
  status?: ScheduledWakeStatus;
  limit: number;
};

export type ScheduledWakesRepositoryOptions = {
  db: SqliteDatabase;
  clock?: Clock;
};

function mapScheduledWakeRow(row: Record<string, unknown>): ScheduledWake {
  const parsed = scheduledWakeSchema.safeParse({
    id: row.id,
    fire_at: Number(row.fire_at),
    note: row.note,
    origin_session_id:
      row.origin_session_id === null || row.origin_session_id === undefined
        ? null
        : String(row.origin_session_id),
    status: row.status,
    created_at: Number(row.created_at),
    updated_at: Number(row.updated_at),
    fired_at: row.fired_at === null || row.fired_at === undefined ? null : Number(row.fired_at),
    cancelled_at:
      row.cancelled_at === null || row.cancelled_at === undefined ? null : Number(row.cancelled_at),
  });

  if (!parsed.success) {
    throw new StorageError("Scheduled wake row failed validation", {
      cause: parsed.error,
      code: "SCHEDULED_WAKE_ROW_INVALID",
    });
  }

  return parsed.data;
}

function boundLimit(limit: number): number {
  return Number.isFinite(limit) ? Math.max(0, Math.floor(limit)) : 0;
}

/**
 * Stores one-time self-scheduled wakes. A wake is written `pending`; the
 * scheduled-wake autonomy trigger fires it once when `fire_at` has passed,
 * after which it is reconciled to `fired`. Borg can `cancel` a pending wake
 * before it fires. This is the entity's deliberate lever over when it next
 * wakes itself -- distinct from the data-derived triggers (goals, commitments).
 */
export class ScheduledWakesRepository {
  private readonly clock: Clock;

  constructor(private readonly options: ScheduledWakesRepositoryOptions) {
    this.clock = options.clock ?? new SystemClock();
  }

  private get db(): SqliteDatabase {
    return this.options.db;
  }

  schedule(input: ScheduledWakeScheduleInput): ScheduledWake {
    if (!Number.isFinite(input.delaySeconds) || input.delaySeconds <= 0) {
      throw new StorageError("Scheduled wake requires a positive delay", {
        code: "SCHEDULED_WAKE_DELAY_INVALID",
      });
    }

    const note = input.note.trim();

    if (note.length === 0) {
      throw new StorageError("Scheduled wake requires a non-empty note", {
        code: "SCHEDULED_WAKE_NOTE_REQUIRED",
      });
    }

    const now = this.clock.now();
    const fireAt = now + Math.round(input.delaySeconds * 1_000);
    const id = createScheduledWakeId();

    this.db
      .prepare(
        `
          INSERT INTO scheduled_wakes (
            ${SCHEDULED_WAKE_COLUMNS}
          ) VALUES (?, ?, ?, ?, 'pending', ?, ?, NULL, NULL)
        `,
      )
      .run(id, fireAt, note, input.originSessionId ?? null, now, now);

    const stored = this.get(id);

    if (stored === null) {
      throw new StorageError(`Scheduled wake ${id} was not stored`, {
        code: "SCHEDULED_WAKE_STORE_FAILED",
      });
    }

    return stored;
  }

  get(id: ScheduledWakeId): ScheduledWake | null {
    const row = this.db
      .prepare(`SELECT ${SCHEDULED_WAKE_COLUMNS} FROM scheduled_wakes WHERE id = ?`)
      .get(parseScheduledWakeId(id)) as Record<string, unknown> | undefined;

    return row === undefined ? null : mapScheduledWakeRow(row);
  }

  listDuePending(nowMs: number): ScheduledWake[] {
    const rows = this.db
      .prepare(
        `
          SELECT ${SCHEDULED_WAKE_COLUMNS}
          FROM scheduled_wakes
          WHERE status = 'pending' AND fire_at <= ?
          ORDER BY fire_at ASC, id ASC
        `,
      )
      .all(nowMs) as Record<string, unknown>[];

    return rows.map(mapScheduledWakeRow);
  }

  list(input: ScheduledWakeListInput): ScheduledWake[] {
    const limit = boundLimit(input.limit);
    const rows =
      input.status === undefined
        ? (this.db
            .prepare(
              `
                SELECT ${SCHEDULED_WAKE_COLUMNS}
                FROM scheduled_wakes
                ORDER BY fire_at ASC, id ASC
                LIMIT ?
              `,
            )
            .all(limit) as Record<string, unknown>[])
        : (this.db
            .prepare(
              `
                SELECT ${SCHEDULED_WAKE_COLUMNS}
                FROM scheduled_wakes
                WHERE status = ?
                ORDER BY fire_at ASC, id ASC
                LIMIT ?
              `,
            )
            .all(input.status, limit) as Record<string, unknown>[]);

    return rows.map(mapScheduledWakeRow);
  }

  markFired(ids: readonly ScheduledWakeId[], firedAt: number): void {
    if (ids.length === 0) {
      return;
    }

    const placeholders = ids.map(() => "?").join(", ");
    this.db
      .prepare(
        `
          UPDATE scheduled_wakes
          SET status = 'fired', fired_at = ?, updated_at = ?
          WHERE status = 'pending' AND id IN (${placeholders})
        `,
      )
      .run(firedAt, firedAt, ...ids);
  }

  cancel(id: ScheduledWakeId, now?: number): ScheduledWake | null {
    const ts = now ?? this.clock.now();
    const result = this.db
      .prepare(
        `
          UPDATE scheduled_wakes
          SET status = 'cancelled', cancelled_at = ?, updated_at = ?
          WHERE id = ? AND status = 'pending'
        `,
      )
      .run(ts, ts, parseScheduledWakeId(id));

    return result.changes > 0 ? this.get(id) : null;
  }

  prune(olderThan: number): number {
    const result = this.db
      .prepare(
        `
          DELETE FROM scheduled_wakes
          WHERE status IN ('fired', 'cancelled') AND updated_at < ?
        `,
      )
      .run(olderThan);

    return result.changes;
  }
}
