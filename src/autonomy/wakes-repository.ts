import { z } from "zod";

import { parseJsonArray } from "../storage/codecs.js";
import type { SqliteDatabase } from "../storage/sqlite/index.js";
import { SystemClock, type Clock } from "../util/clock.js";
import { StorageError } from "../util/errors.js";
import {
  autonomyWakeIdHelpers,
  createAutonomyWakeId,
  goalIdHelpers,
  isSessionId,
  parseAutonomyWakeId,
  parseGoalId,
  parseSessionId,
  type AutonomyWakeId,
  type GoalId,
  type SessionId,
} from "../util/ids.js";

import {
  AUTONOMY_CONDITION_NAMES,
  AUTONOMY_WAKE_OUTCOMES,
  AUTONOMY_WAKE_SOURCE_NAMES,
  type AutonomyConditionName,
  type AutonomyWakeOutcome,
  type AutonomyWakeOutcomeDetailTally,
  type AutonomyWakeOutcomeSpan,
  type AutonomyWakeSourceCategory,
  type AutonomyWakeSourceName,
  type AutonomyWakeSourceType,
} from "./types.js";

const autonomyWakeSourceTypeSchema = z.enum(["trigger", "condition"]);
const autonomyWakeSourceCategorySchema = z.enum(["contemplative", "operational"]);
const autonomyWakeSourceNameSchema = z.enum(AUTONOMY_WAKE_SOURCE_NAMES);
const autonomyConditionNameSchema = z.enum(AUTONOMY_CONDITION_NAMES);
const autonomyWakeOutcomeSchema = z.enum(AUTONOMY_WAKE_OUTCOMES);
const autonomyWakeHeadwayBasesSchema = z.array(z.string().min(1)).min(1);
const autonomyWakeExecutionCountsSchema = z
  .object({
    finalizer_rounds: z.number().int().nonnegative(),
    stall_retries: z.number().int().nonnegative(),
  })
  .strict();

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
  selected_goal_id: z
    .string()
    .refine((value) => goalIdHelpers.is(value), { message: "Invalid goal id" })
    .transform((value) => parseGoalId(value))
    .nullable()
    .optional(),
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
  headway_bases: autonomyWakeHeadwayBasesSchema.nullable(),
  finalizer_rounds: z.number().int().nonnegative().nullable(),
  stall_retries: z.number().int().nonnegative().nullable(),
  selected_goal_id: z
    .string()
    .refine((value) => goalIdHelpers.is(value), { message: "Invalid goal id" })
    .transform((value) => parseGoalId(value))
    .nullable(),
});

/**
 * Upper bound on a non-structural `outcome_detail`. Failure details can arrive
 * from an arbitrary layer below the scheduler, so their length is not ours to
 * predict. Structural headway bases are stored separately and their joined
 * display is never clipped mid-basis.
 */
export const AUTONOMY_WAKE_OUTCOME_DETAIL_MAX_LENGTH = 300;
export const AUTONOMY_WAKE_STARTUP_INTERRUPTED_GRACE_MS = 60 * 60 * 1_000;
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
   * Null means either the outcome had no detail or the row predates the column.
   * Callers that count details must state the undetailed remainder rather than
   * treat it as zero.
   */
  outcome_detail: string | null;
  /**
   * Ordered structural predicates that made this row headway. Null means the
   * row is not headway or predates structural basis storage.
   */
  headway_bases: string[] | null;
  /** Number of finalizer LLM rounds run by this wake's turn. */
  finalizer_rounds: number | null;
  /** Number of stall-class transport retries paid by this wake's turn. */
  stall_retries: number | null;
  selected_goal_id: GoalId | null;
};

export type AutonomyWakeExecutionCounts = z.infer<typeof autonomyWakeExecutionCountsSchema>;

export type AutonomyWakeRecordInput = {
  trigger_name: AutonomyWakeSourceName;
  condition_name?: AutonomyConditionName | null;
  session_id?: SessionId | null;
  wake_source_type: AutonomyWakeSourceType;
  source_category?: AutonomyWakeSourceCategory;
  selected_goal_id?: GoalId | null;
};

export type AutonomyWakesRepositoryOptions = {
  db: SqliteDatabase;
  clock?: Clock;
};

function mapWakeRow(row: Record<string, unknown>): AutonomyWakeRecord {
  const headwayBases =
    row.headway_bases_json === null || row.headway_bases_json === undefined
      ? null
      : typeof row.headway_bases_json === "string"
        ? parseJsonArray<unknown>(row.headway_bases_json, "autonomy wake headway bases", {
            errorCode: "AUTONOMY_WAKE_HEADWAY_BASES_INVALID",
            errorMessage: (label) => `${label} failed validation`,
          })
        : row.headway_bases_json;
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
    headway_bases: headwayBases,
    finalizer_rounds:
      row.finalizer_rounds === null || row.finalizer_rounds === undefined
        ? null
        : Number(row.finalizer_rounds),
    stall_retries:
      row.stall_retries === null || row.stall_retries === undefined
        ? null
        : Number(row.stall_retries),
    selected_goal_id: row.selected_goal_id === undefined ? null : (row.selected_goal_id ?? null),
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
      headway_bases: null,
      finalizer_rounds: null,
      stall_retries: null,
      selected_goal_id: parsed.selected_goal_id ?? null,
    };

    this.db
      .prepare(
        `
          INSERT INTO autonomy_wakes (
            id, ts, trigger_name, condition_name, session_id, wake_source_type, source_category,
            selected_goal_id
          ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
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
        record.selected_goal_id,
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
                 outcome, outcome_detail, headway_bases_json, finalizer_rounds, stall_retries,
                 selected_goal_id
          FROM autonomy_wakes
          WHERE ts >= ?
          ORDER BY ts DESC, id DESC
          LIMIT ?
        `,
      )
      .all(ts, boundedLimit) as Record<string, unknown>[];

    return rows.map((row) => mapWakeRow(row));
  }

  recordOutcome(
    id: AutonomyWakeId,
    outcome: AutonomyWakeOutcome,
    detail?: string | null,
    headwayBases?: readonly string[] | null,
    executionCounts?: AutonomyWakeExecutionCounts,
  ): void {
    const storedHeadwayBases =
      outcome === "headway" && headwayBases !== null && headwayBases !== undefined
        ? autonomyWakeHeadwayBasesSchema.parse(headwayBases)
        : null;
    const storedDetail =
      storedHeadwayBases === null ? clampOutcomeDetail(detail) : storedHeadwayBases.join("; ");
    const storedExecutionCounts =
      executionCounts === undefined
        ? null
        : autonomyWakeExecutionCountsSchema.parse(executionCounts);

    this.db
      .prepare(
        `UPDATE autonomy_wakes
         SET outcome = ?, outcome_detail = ?, headway_bases_json = ?,
             finalizer_rounds = ?, stall_retries = ?
         WHERE id = ? AND outcome IS NULL`,
      )
      .run(
        outcome,
        storedDetail,
        storedHeadwayBases === null ? null : JSON.stringify(storedHeadwayBases),
        storedExecutionCounts?.finalizer_rounds ?? null,
        storedExecutionCounts?.stall_retries ?? null,
        id,
      );
  }

  interruptOrphanedWakesAtStartup(): number {
    const orphanedBefore = this.clock.now() - AUTONOMY_WAKE_STARTUP_INTERRUPTED_GRACE_MS;
    const result = this.db
      .prepare(
        `
          UPDATE autonomy_wakes
          SET outcome = 'interrupted', outcome_detail = ?
          WHERE outcome IS NULL AND ts < ?
        `,
      )
      .run(AUTONOMY_WAKE_STARTUP_INTERRUPTED_DETAIL, orphanedBefore);

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
          SELECT outcome_detail AS detail, trigger_name AS trigger, COUNT(*) AS count
          FROM autonomy_wakes
          WHERE ts >= ? AND outcome = ?
          GROUP BY outcome_detail, trigger_name
          ORDER BY count DESC, detail ASC, trigger ASC
        `,
      )
      .all(ts, outcome) as Array<{ detail: unknown; trigger: unknown; count: unknown }>;

    let total = 0;
    let withoutDetail = 0;
    const reasons: AutonomyWakeOutcomeDetailTally["reasons"] = [];
    const byDetail = new Map<string, AutonomyWakeOutcomeDetailTally["reasons"][number]>();

    for (const row of rows) {
      const count = Number(row.count ?? 0);
      total += count;

      if (typeof row.detail !== "string" || row.detail.length === 0) {
        withoutDetail += count;
        continue;
      }

      const trigger = typeof row.trigger === "string" ? row.trigger : "";
      const existing = byDetail.get(row.detail);

      if (existing === undefined) {
        const reason = { detail: row.detail, count, triggers: [{ trigger, count }] };
        byDetail.set(row.detail, reason);
        reasons.push(reason);
        continue;
      }

      existing.count += count;
      existing.triggers.push({ trigger, count });
    }

    // The GROUP BY is by (detail, trigger), so a detail spread over several
    // triggers arrives as several rows and its folded count can pass a detail
    // ordered ahead of it. Re-sorting on the folded counts keeps the render cap
    // cutting the smallest details rather than the ones that happened to arrive
    // undivided.
    reasons.sort((a, b) => b.count - a.count || a.detail.localeCompare(b.detail));

    return { total, without_detail: withoutDetail, reasons };
  }

  /**
   * Where one outcome's rows sit among the window's other wakes: how many wakes
   * that did not land in the bucket fall strictly between its first and last row,
   * and whether the wake immediately before its first row carries the same
   * outcome. The second read deliberately ignores the window -- the table retains
   * more than the window covers, and a run clipped by the window edge is exactly
   * the case a count inside the window cannot distinguish from a run that began
   * there.
   */
  describeOutcomeSpanSince(
    ts: number,
    outcome: AutonomyWakeOutcome,
  ): AutonomyWakeOutcomeSpan | null {
    const bounds = this.db
      .prepare(
        `
          SELECT MIN(ts) AS first_ts, MAX(ts) AS last_ts, COUNT(*) AS count
          FROM autonomy_wakes
          WHERE ts >= ? AND outcome = ?
        `,
      )
      .get(ts, outcome) as { first_ts: unknown; last_ts: unknown; count: unknown } | undefined;

    const count = Number(bounds?.count ?? 0);

    if (count === 0 || typeof bounds?.first_ts !== "number" || typeof bounds.last_ts !== "number") {
      return null;
    }

    const between = this.db
      .prepare(
        `
          SELECT COUNT(*) AS count
          FROM autonomy_wakes
          WHERE ts > ? AND ts < ? AND (outcome IS NULL OR outcome != ?)
        `,
      )
      .get(bounds.first_ts, bounds.last_ts, outcome) as { count: unknown } | undefined;

    const previous = this.db
      .prepare(
        `
          SELECT outcome
          FROM autonomy_wakes
          WHERE ts < ?
          ORDER BY ts DESC
          LIMIT 1
        `,
      )
      .get(bounds.first_ts) as { outcome: unknown } | undefined;

    return {
      other_outcomes_between: Number(between?.count ?? 0),
      extends_before_window: previous === undefined ? null : previous.outcome === outcome,
    };
  }

  prune(olderThan: number): number {
    const result = this.db.prepare("DELETE FROM autonomy_wakes WHERE ts < ?").run(olderThan);

    return result.changes;
  }
}
