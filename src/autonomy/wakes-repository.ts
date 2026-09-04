import { z } from "zod";

import type { SqliteDatabase } from "../storage/sqlite/index.js";
import { SystemClock, type Clock } from "../util/clock.js";
import { StorageError } from "../util/errors.js";
import {
  autonomyWakeIdHelpers,
  createAutonomyWakeId,
  isSessionId,
  parseAutonomyWakeId,
  parseSessionId,
  type AutonomyWakeId,
  type SessionId,
} from "../util/ids.js";

import {
  AUTONOMY_CONDITION_NAMES,
  AUTONOMY_WAKE_OUTCOMES,
  AUTONOMY_WAKE_SOURCE_NAMES,
  type AutonomyConditionName,
  type AutonomyWakeOutcome,
  type AutonomyWakeOutcomeDetailTally,
  type AutonomyWakeSourceCategory,
  type AutonomyWakeSourceName,
  type AutonomyWakeSourceType,
} from "./types.js";

const autonomyWakeSourceTypeSchema = z.enum(["trigger", "condition"]);
const autonomyWakeSourceCategorySchema = z.enum(["contemplative", "operational"]);
const autonomyWakeSourceNameSchema = z.enum(AUTONOMY_WAKE_SOURCE_NAMES);
const autonomyConditionNameSchema = z.enum(AUTONOMY_CONDITION_NAMES);
const autonomyWakeOutcomeSchema = z.enum(AUTONOMY_WAKE_OUTCOMES);

const autonomyWakeInputSchema = z.object({
  trigger_name: autonomyWakeSourceNameSchema,
  condition_name: autonomyConditionNameSchema.nullable().optional(),
  session_id: z
    .string()
    .refine((value) => isSessionId(value), {
      message: "Invalid session id",
    })
    .transform((value) => parseSessionId(value))
    .nullable()
    .optional(),
  wake_source_type: autonomyWakeSourceTypeSchema,
  source_category: autonomyWakeSourceCategorySchema.optional().default("operational"),
});

const autonomyWakeRowSchema = z.object({
  id: z
    .string()
    .refine((value) => autonomyWakeIdHelpers.is(value), {
      message: "Invalid autonomy wake id",
    })
    .transform((value) => parseAutonomyWakeId(value)),
  ts: z.number().int().finite(),
  trigger_name: autonomyWakeSourceNameSchema,
  condition_name: autonomyConditionNameSchema.nullable(),
  session_id: z
    .string()
    .refine((value) => isSessionId(value), {
      message: "Invalid session id",
    })
    .transform((value) => parseSessionId(value))
    .nullable(),
  wake_source_type: autonomyWakeSourceTypeSchema,
  source_category: autonomyWakeSourceCategorySchema,
  outcome: autonomyWakeOutcomeSchema.nullable(),
  outcome_detail: z.string().nullable(),
});

/**
 * Upper bound on a stored `outcome_detail`. The detail is a formatted error from
 * an arbitrary layer below the scheduler, so its length is not ours to predict;
 * the cap keeps one pathological failure string from dominating a page that
 * renders these. Truncation is marked so a clipped detail is never read as the
 * whole message.
 */
export const AUTONOMY_WAKE_OUTCOME_DETAIL_MAX_LENGTH = 300;
export const AUTONOMY_WAKE_STARTUP_INTERRUPTED_DETAIL =
  "Wake was still in flight when Borg started; recorded as interrupted.";

function clampOutcomeDetail(detail: string | null | undefined): string | null {
  if (detail === null || detail === undefined) {
    return null;
  }

  const collapsed = detail.replace(/\s+/gu, " ").trim();

  if (collapsed.length === 0) {
    return null;
  }

  return collapsed.length <= AUTONOMY_WAKE_OUTCOME_DETAIL_MAX_LENGTH
    ? collapsed
    : `${collapsed.slice(0, AUTONOMY_WAKE_OUTCOME_DETAIL_MAX_LENGTH)}...`;
}

export type AutonomyWakeRecord = {
  id: AutonomyWakeId;
  ts: number;
  trigger_name: AutonomyWakeSourceName;
  condition_name: AutonomyConditionName | null;
  session_id: SessionId | null;
  wake_source_type: AutonomyWakeSourceType;
  source_category: AutonomyWakeSourceCategory;
  outcome: AutonomyWakeOutcome | null;
  /**
   * Why this wake ended the way it did, when the outcome had a reason to carry.
   * Null means one of two different things and does not distinguish them: the
   * outcome had no detail (every `headway`/`silent`), or the row predates the
   * column. Callers that count details must state the undetailed remainder
   * rather than treat it as zero.
   */
  outcome_detail: string | null;
};

export type AutonomyWakeRecordInput = {
  trigger_name: AutonomyWakeSourceName;
  condition_name?: AutonomyConditionName | null;
  session_id?: SessionId | null;
  wake_source_type: AutonomyWakeSourceType;
  source_category?: AutonomyWakeSourceCategory;
};

export type AutonomyWakesRepositoryOptions = {
  db: SqliteDatabase;
  clock?: Clock;
};

function mapWakeRow(row: Record<string, unknown>): AutonomyWakeRecord {
  const parsed = autonomyWakeRowSchema.safeParse({
    id: row.id,
    ts: Number(row.ts),
    trigger_name: row.trigger_name,
    condition_name:
      row.condition_name === null || row.condition_name === undefined ? null : row.condition_name,
    session_id: row.session_id === null || row.session_id === undefined ? null : row.session_id,
    wake_source_type: row.wake_source_type,
    source_category: row.source_category ?? "operational",
    outcome: row.outcome ?? null,
    outcome_detail: row.outcome_detail === undefined ? null : (row.outcome_detail ?? null),
  });

  if (!parsed.success) {
    throw new StorageError("Autonomy wake row failed validation", {
      cause: parsed.error,
      code: "AUTONOMY_WAKE_ROW_INVALID",
    });
  }

  return parsed.data;
}

export class AutonomyWakesRepository {
  private readonly clock: Clock;

  constructor(private readonly options: AutonomyWakesRepositoryOptions) {
    this.clock = options.clock ?? new SystemClock();
  }

  private get db(): SqliteDatabase {
    return this.options.db;
  }

  record(input: AutonomyWakeRecordInput): AutonomyWakeRecord {
    const parsed = autonomyWakeInputSchema.parse(input);
    const record: AutonomyWakeRecord = {
      id: createAutonomyWakeId(),
      ts: this.clock.now(),
      trigger_name: parsed.trigger_name,
      condition_name: parsed.condition_name ?? null,
      session_id: parsed.session_id ?? null,
      wake_source_type: parsed.wake_source_type,
      source_category: parsed.source_category,
      outcome: null,
      outcome_detail: null,
    };

    this.db
      .prepare(
        `
          INSERT INTO autonomy_wakes (
            id, ts, trigger_name, condition_name, session_id, wake_source_type, source_category
          ) VALUES (?, ?, ?, ?, ?, ?, ?)
        `,
      )
      .run(
        record.id,
        record.ts,
        record.trigger_name,
        record.condition_name,
        record.session_id,
        record.wake_source_type,
        record.source_category,
      );

    return record;
  }

  countSince(
    ts: number,
    options: {
      sourceCategory?: AutonomyWakeSourceCategory;
      outcome?: AutonomyWakeOutcome;
    } = {},
  ): number {
    const categoryFilter = options.sourceCategory;
    const outcomeFilter = options.outcome;
    const conditions = ["ts >= ?"];
    const values: unknown[] = [ts];

    if (categoryFilter !== undefined) {
      conditions.push("source_category = ?");
      values.push(categoryFilter);
    }

    if (outcomeFilter !== undefined) {
      conditions.push("outcome = ?");
      values.push(outcomeFilter);
    }

    const row = this.db
      .prepare(`SELECT COUNT(*) AS count FROM autonomy_wakes WHERE ${conditions.join(" AND ")}`)
      .get(...values) as { count: number } | undefined;

    return Number(row?.count ?? 0);
  }

  listSince(ts: number, limit: number): AutonomyWakeRecord[] {
    const boundedLimit = Number.isFinite(limit) ? Math.max(0, Math.floor(limit)) : 0;
    const rows = this.db
      .prepare(
        `
          SELECT id, ts, trigger_name, condition_name, session_id, wake_source_type, source_category,
                 outcome, outcome_detail
          FROM autonomy_wakes
          WHERE ts >= ?
          ORDER BY ts DESC, id DESC
          LIMIT ?
        `,
      )
      .all(ts, boundedLimit) as Record<string, unknown>[];

    return rows.map((row) => mapWakeRow(row));
  }

  recordOutcome(id: AutonomyWakeId, outcome: AutonomyWakeOutcome, detail?: string | null): void {
    this.db
      .prepare(
        "UPDATE autonomy_wakes SET outcome = ?, outcome_detail = ? WHERE id = ? AND outcome IS NULL",
      )
      .run(outcome, clampOutcomeDetail(detail), id);
  }

  interruptOrphanedWakesAtStartup(): number {
    const result = this.db
      .prepare(
        `
          UPDATE autonomy_wakes
          SET outcome = 'interrupted', outcome_detail = ?
          WHERE outcome IS NULL
        `,
      )
      .run(AUTONOMY_WAKE_STARTUP_INTERRUPTED_DETAIL);

    return result.changes;
  }

  /**
   * Distinct details recorded for one outcome over `ts >= cutoff`, most frequent
   * first, alongside the same-window bucket total and the number of rows in it
   * that carry no detail. The three are returned together deliberately: a reason
   * list alone reads as the whole bucket, and the difference between it and the
   * bucket count is the one thing a reader cannot recover from the list.
   */
  summarizeOutcomeDetailsSince(
    ts: number,
    outcome: AutonomyWakeOutcome,
  ): AutonomyWakeOutcomeDetailTally {
    const rows = this.db
      .prepare(
        `
          SELECT outcome_detail AS detail, COUNT(*) AS count
          FROM autonomy_wakes
          WHERE ts >= ? AND outcome = ?
          GROUP BY outcome_detail
          ORDER BY count DESC, detail ASC
        `,
      )
      .all(ts, outcome) as Array<{ detail: unknown; count: unknown }>;

    let total = 0;
    let withoutDetail = 0;
    const reasons: Array<{ detail: string; count: number }> = [];

    for (const row of rows) {
      const count = Number(row.count ?? 0);
      total += count;

      if (typeof row.detail !== "string" || row.detail.length === 0) {
        withoutDetail += count;
        continue;
      }

      reasons.push({ detail: row.detail, count });
    }

    return { total, without_detail: withoutDetail, reasons };
  }

  prune(olderThan: number): number {
    const result = this.db.prepare("DELETE FROM autonomy_wakes WHERE ts < ?").run(olderThan);

    return result.changes;
  }
}
