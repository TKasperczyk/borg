/*
 * Backfill terminal goal delete events stranded by aborted-turn rollback and
 * report (without repairing) live goals whose status differs from their latest
 * audited value.
 *
 * Run with every Borg writer stopped and take a verified backup first. Dry-run
 * is the default; --apply is required to write the missing delete events.
 *
 * Usage:
 *   pnpm tsx scripts/repair-goal-rollback-audit.ts --data-dir <bank-dir>
 *   pnpm tsx scripts/repair-goal-rollback-audit.ts --data-dir <bank-dir> --apply
 */
import { existsSync } from "node:fs";
import { join, resolve } from "node:path";
import { DatabaseSync } from "node:sqlite";
import { pathToFileURL } from "node:url";

import { z } from "zod";

import { IdentityEventRepository, type IdentityEvent } from "../src/memory/identity/index.js";
import { goalIdSchema, goalStatusSchema, type GoalStatus } from "../src/memory/self/index.js";
import { SqliteDatabase, SqliteRawDatabase } from "../src/storage/sqlite/index.js";
import type { GoalId } from "../src/util/ids.js";

export const GOAL_ROLLBACK_AUDIT_REPAIR_REASON =
  "operator repair: backfilled terminal delete for a goal create event stranded by turn rollback";
export const GOAL_ROLLBACK_AUDIT_REPAIR_PROVENANCE = {
  kind: "offline",
  process: "repair-goal-rollback-audit",
} as const;

const repairInputSchema = z
  .object({
    dataDir: z
      .string()
      .trim()
      .min(1, "--data-dir requires a non-empty path")
      .transform((value) => resolve(value)),
  })
  .strict();

const candidateIdRowSchema = z.object({ record_id: goalIdSchema }).strict();
const statusDriftRowSchema = z
  .object({
    id: goalIdSchema,
    current_status: goalStatusSchema,
    audited_status: goalStatusSchema,
    identity_event_id: z.number().int().positive(),
    identity_action: z.string().min(1),
  })
  .strict();

export type GoalRollbackAuditRepairCandidate = {
  goalId: GoalId;
  createEvent: IdentityEvent;
  latestSnapshotEvent: IdentityEvent;
  eventCount: number;
};

export type GoalStatusDrift = {
  goalId: GoalId;
  currentStatus: GoalStatus;
  auditedStatus: GoalStatus;
  identityEventId: number;
  identityAction: string;
};

export type GoalRollbackAuditRepairPlan = {
  dataDir: string;
  candidates: GoalRollbackAuditRepairCandidate[];
  statusDrifts: GoalStatusDrift[];
};

export type GoalRollbackAuditRepairFailure = {
  goalId: GoalId | "repair";
  message: string;
};

export type GoalRollbackAuditRepairReport = {
  dryRun: boolean;
  plan: GoalRollbackAuditRepairPlan;
  backfilled: GoalRollbackAuditRepairCandidate[];
  failures: GoalRollbackAuditRepairFailure[];
};

type GoalRollbackAuditRepairCliArgs =
  | { help: true }
  | { help: false; apply: boolean; dataDir: string };

type RepairOutput = {
  write(chunk: string): unknown;
};

export type GoalRollbackAuditRepairMainOptions = {
  stdout?: RepairOutput;
  stderr?: RepairOutput;
};

function describeZodError(error: z.ZodError): string {
  return error.issues
    .map((issue) => `${issue.path.join(".") || "input"}: ${issue.message}`)
    .join("; ");
}

function parseRepairInput(input: unknown): { dataDir: string } {
  const parsed = repairInputSchema.safeParse(input);

  if (!parsed.success) {
    throw new Error(`Invalid goal rollback audit repair input: ${describeZodError(parsed.error)}`);
  }

  return parsed.data;
}

function openRepairDatabase(path: string, readOnly: boolean): SqliteDatabase {
  let raw: SqliteRawDatabase | undefined;

  try {
    raw = new SqliteRawDatabase(
      new DatabaseSync(path, {
        enableDoubleQuotedStringLiterals: true,
        readOnly,
      }),
    );
    const db = new SqliteDatabase(raw);
    db.pragma("busy_timeout = 5000");
    db.pragma("foreign_keys = ON");

    if (readOnly) {
      db.pragma("query_only = ON");
    }

    return db;
  } catch (error) {
    try {
      raw?.close();
    } catch {
      // Preserve the original open failure.
    }

    throw error;
  }
}

function readRepairPlan(db: SqliteDatabase, dataDir: string): GoalRollbackAuditRepairPlan {
  const candidateIdRows = db
    .prepare(
      `
        WITH ordered_goal_events AS (
          SELECT
            id,
            record_id,
            action,
            ROW_NUMBER() OVER (PARTITION BY record_id ORDER BY ts ASC, id ASC) AS event_rank
          FROM identity_events
          WHERE record_type = 'goal'
        ),
        goal_event_chains AS (
          SELECT
            record_id,
            MAX(CASE WHEN event_rank = 1 THEN id END) AS first_event_id,
            MAX(CASE WHEN event_rank = 1 THEN action END) AS first_action,
            SUM(CASE WHEN action IN ('delete', 'forget') THEN 1 ELSE 0 END) AS terminal_count
          FROM ordered_goal_events
          GROUP BY record_id
        )
        SELECT chains.record_id
        FROM goal_event_chains AS chains
        LEFT JOIN goals ON goals.id = chains.record_id
        WHERE goals.id IS NULL
          AND chains.first_action = 'create'
          AND chains.terminal_count = 0
        ORDER BY chains.first_event_id ASC
      `,
    )
    .all() as Record<string, unknown>[];
  const candidateIds = candidateIdRows.map((row) => candidateIdRowSchema.parse(row).record_id);
  const goalEventCountRow = db
    .prepare("SELECT COUNT(*) AS count FROM identity_events WHERE record_type = 'goal'")
    .get() as { count: number } | undefined;
  const goalEventCount = Number(goalEventCountRow?.count ?? 0);
  const goalEvents =
    goalEventCount === 0
      ? []
      : new IdentityEventRepository({ db }).list({ recordType: "goal", limit: goalEventCount });
  const eventsByRecordId = new Map<string, IdentityEvent[]>();
  for (const event of goalEvents) {
    const events = eventsByRecordId.get(event.record_id) ?? [];
    events.push(event);
    eventsByRecordId.set(event.record_id, events);
  }
  const candidates = candidateIds.map((goalId) => {
    const events = eventsByRecordId.get(goalId) ?? [];
    const createEvent = events.at(-1);
    const latestSnapshotEvent = events.find((event) => event.new_value !== null);

    if (
      createEvent === undefined ||
      createEvent.action !== "create" ||
      createEvent.new_value === null ||
      latestSnapshotEvent === undefined
    ) {
      throw new Error(`Stranded goal event chain could not be validated: ${goalId}`);
    }

    return {
      goalId,
      createEvent,
      latestSnapshotEvent,
      eventCount: events.length,
    };
  });
  const statusDriftRows = db
    .prepare(
      `
        WITH ranked_goal_events AS (
          SELECT
            id,
            record_id,
            action,
            new_value_json,
            ROW_NUMBER() OVER (PARTITION BY record_id ORDER BY ts DESC, id DESC) AS event_rank
          FROM identity_events
          WHERE record_type = 'goal'
        )
        SELECT
          goals.id,
          goals.status AS current_status,
          json_extract(events.new_value_json, '$.status') AS audited_status,
          events.id AS identity_event_id,
          events.action AS identity_action
        FROM goals
        JOIN ranked_goal_events AS events
          ON events.record_id = goals.id AND events.event_rank = 1
        WHERE json_type(events.new_value_json, '$.status') = 'text'
          AND goals.status <> json_extract(events.new_value_json, '$.status')
        ORDER BY goals.id ASC
      `,
    )
    .all() as Record<string, unknown>[];
  const statusDrifts = statusDriftRows.map((row) => {
    const parsed = statusDriftRowSchema.parse({
      ...row,
      identity_event_id: Number(row.identity_event_id),
    });

    return {
      goalId: parsed.id,
      currentStatus: parsed.current_status,
      auditedStatus: parsed.audited_status,
      identityEventId: parsed.identity_event_id,
      identityAction: parsed.identity_action,
    };
  });

  return { dataDir, candidates, statusDrifts };
}

/** Build a read-only repair plan from an on-disk Borg bank. */
export function planGoalRollbackAuditRepair(input: unknown): GoalRollbackAuditRepairPlan {
  const parsed = parseRepairInput(input);
  const databasePath = join(parsed.dataDir, "borg.db");

  if (!existsSync(databasePath)) {
    throw new Error(`No borg.db found in data directory ${parsed.dataDir}`);
  }

  const db = openRepairDatabase(databasePath, true);

  try {
    return readRepairPlan(db, parsed.dataDir);
  } finally {
    db.close();
  }
}

function candidateStillStranded(
  db: SqliteDatabase,
  candidate: GoalRollbackAuditRepairCandidate,
): boolean {
  const goalExists = db.prepare("SELECT 1 FROM goals WHERE id = ?").get(candidate.goalId);

  if (goalExists !== undefined) {
    return false;
  }

  const eventRows = db
    .prepare(
      `
        SELECT id, action, new_value_json
        FROM identity_events
        WHERE record_type = 'goal' AND record_id = ?
        ORDER BY ts ASC, id ASC
      `,
    )
    .all(candidate.goalId) as Array<{
    id: number;
    action: string;
    new_value_json: string | null;
  }>;
  const firstEvent = eventRows[0];
  const latestSnapshotEvent = eventRows.findLast((event) => event.new_value_json !== null);

  return (
    eventRows.length === candidate.eventCount &&
    Number(firstEvent?.id) === candidate.createEvent.id &&
    firstEvent?.action === "create" &&
    eventRows.every((event) => event.action !== "delete" && event.action !== "forget") &&
    Number(latestSnapshotEvent?.id) === candidate.latestSnapshotEvent.id
  );
}

export function applyGoalRollbackAuditRepairPlan(
  db: SqliteDatabase,
  plan: GoalRollbackAuditRepairPlan,
): GoalRollbackAuditRepairReport {
  const report: GoalRollbackAuditRepairReport = {
    dryRun: false,
    plan,
    backfilled: [],
    failures: [],
  };
  const identityEvents = new IdentityEventRepository({ db });

  try {
    const backfilled = identityEvents.runInTransaction(() => {
      for (const candidate of plan.candidates) {
        if (!candidateStillStranded(db, candidate)) {
          throw new Error(`Goal audit changed after planning: ${candidate.goalId}`);
        }
      }

      for (const candidate of plan.candidates) {
        identityEvents.record({
          record_type: "goal",
          record_id: candidate.goalId,
          action: "delete",
          old_value: candidate.latestSnapshotEvent.new_value,
          new_value: null,
          reason: GOAL_ROLLBACK_AUDIT_REPAIR_REASON,
          provenance: GOAL_ROLLBACK_AUDIT_REPAIR_PROVENANCE,
        });
      }

      return [...plan.candidates];
    });
    report.backfilled.push(...backfilled);
  } catch (error) {
    report.failures.push({
      goalId: "repair",
      message: error instanceof Error ? error.message : String(error),
    });
  }

  return report;
}

export function formatGoalRollbackAuditRepairReport(report: GoalRollbackAuditRepairReport): string {
  const lines = [
    `mode=${report.dryRun ? "dry-run" : "apply"}`,
    `stranded_create_candidates=${report.plan.candidates.length}`,
  ];
  const backfilledIds = new Set(report.backfilled.map((candidate) => candidate.goalId));

  for (const candidate of report.plan.candidates) {
    const action = report.dryRun
      ? "would_backfill"
      : backfilledIds.has(candidate.goalId)
        ? "backfilled"
        : "not_backfilled";
    lines.push(
      `${action} goal=${candidate.goalId} create_identity_event=${candidate.createEvent.id} snapshot_identity_event=${candidate.latestSnapshotEvent.id} chain_events=${candidate.eventCount}`,
    );
  }

  lines.push(`status_drift_goals=${report.plan.statusDrifts.length}`);
  for (const drift of report.plan.statusDrifts) {
    lines.push(
      `status_drift goal=${drift.goalId} current=${drift.currentStatus} audited=${drift.auditedStatus} identity_event=${drift.identityEventId} action=${drift.identityAction} no_change=true`,
    );
  }

  if (!report.dryRun) {
    lines.push(`backfilled_total=${report.backfilled.length}`);
  }

  for (const failure of report.failures) {
    lines.push(`failure goal=${failure.goalId} message=${JSON.stringify(failure.message)}`);
  }
  lines.push(`failures=${report.failures.length}`);

  return `${lines.join("\n")}\n`;
}

export function goalRollbackAuditRepairExitCode(report: GoalRollbackAuditRepairReport): 0 | 1 {
  return report.failures.length === 0 ? 0 : 1;
}

export function parseGoalRollbackAuditRepairCliArgs(
  argv: readonly string[],
): GoalRollbackAuditRepairCliArgs {
  let dataDir: string | undefined;
  let apply = false;

  for (let index = 0; index < argv.length; index += 1) {
    const argument = argv[index];

    if (argument === "--help" || argument === "-h") {
      return { help: true };
    }

    if (argument === "--apply") {
      apply = true;
      continue;
    }

    if (argument === "--data-dir") {
      const value = argv[index + 1];

      if (value === undefined || value.startsWith("--")) {
        throw new Error("--data-dir requires a value");
      }

      dataDir = value;
      index += 1;
      continue;
    }

    throw new Error(`Unknown argument: ${argument ?? ""}`);
  }

  if (dataDir === undefined) {
    throw new Error("--data-dir is required");
  }

  return { help: false, apply, ...parseRepairInput({ dataDir }) };
}

function usage(): string {
  return [
    "Usage: pnpm tsx scripts/repair-goal-rollback-audit.ts --data-dir <bank-dir> [--apply]",
    "",
    "Dry-run is the default. Stop every Borg writer and take a verified backup before --apply.",
  ].join("\n");
}

export async function main(
  argv: readonly string[] = process.argv.slice(2),
  options: GoalRollbackAuditRepairMainOptions = {},
): Promise<0 | 1> {
  const args = parseGoalRollbackAuditRepairCliArgs(argv);
  const stdout = options.stdout ?? process.stdout;
  const stderr = options.stderr ?? process.stderr;

  if (args.help) {
    stdout.write(`${usage()}\n`);
    return 0;
  }

  stderr.write(
    "WARNING: this maintenance requires a verified backup and exclusive single-writer access.\n",
  );
  const plan = planGoalRollbackAuditRepair({ dataDir: args.dataDir });
  let report: GoalRollbackAuditRepairReport = {
    dryRun: !args.apply,
    plan,
    backfilled: [],
    failures: [],
  };

  if (args.apply && plan.candidates.length > 0) {
    const db = openRepairDatabase(join(args.dataDir, "borg.db"), false);

    try {
      report = applyGoalRollbackAuditRepairPlan(db, plan);
    } finally {
      db.close();
    }
  }

  stdout.write(formatGoalRollbackAuditRepairReport(report));
  const exitCode = goalRollbackAuditRepairExitCode(report);

  if (exitCode !== 0) {
    stderr.write("ERROR: goal rollback audit repair failed; inspect the report.\n");
  }

  return exitCode;
}

if (process.argv[1] !== undefined && import.meta.url === pathToFileURL(process.argv[1]).href) {
  main().then(
    (exitCode) => {
      process.exitCode = exitCode;
    },
    (error: unknown) => {
      process.stderr.write(`ERROR: ${error instanceof Error ? error.message : String(error)}\n`);
      process.exitCode = 1;
    },
  );
}
