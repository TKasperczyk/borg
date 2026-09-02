/*
 * Clear goal deadlines that were guessed by the harness rather than extracted
 * from a dated source.
 *
 * Run with every Borg writer stopped and take a verified backup first. Dry-run
 * is the default; --apply routes each write through Borg's public identity
 * service so the corresponding identity_events row is written atomically.
 *
 * Usage:
 *   pnpm tsx scripts/repair-goal-target-at.ts --data-dir <bank-dir> \
 *     --goal <goal-id,goal-id,...>
 *   pnpm tsx scripts/repair-goal-target-at.ts --data-dir <bank-dir> \
 *     --goal <goal-id,goal-id,...> --apply
 */
import { existsSync } from "node:fs";
import { join, resolve } from "node:path";
import { DatabaseSync } from "node:sqlite";
import { pathToFileURL } from "node:url";

import { z } from "zod";

import type { Borg, BorgOpenOptions } from "../src/index.js";
import {
  goalIdSchema,
  goalPatchSchema,
  goalStatusSchema,
  type GoalStatus,
} from "../src/memory/self/index.js";
import { SqliteDatabase, SqliteRawDatabase } from "../src/storage/sqlite/index.js";
import { withBorg } from "../src/cli/helpers/borg.js";
import type { GoalId } from "../src/util/ids.js";

export const GOAL_TARGET_AT_REPAIR_REASON =
  "operator repair: target_at was a harness guess with no dated source; cleared so the deadline latch reflects the record";

const repairInputSchema = z
  .object({
    dataDir: z
      .string()
      .trim()
      .min(1, "--data-dir requires a non-empty path")
      .transform((value) => resolve(value)),
    goalIds: z.array(goalIdSchema).min(1, "--goal requires at least one goal id"),
  })
  .strict();

const planningGoalRowSchema = z
  .object({
    id: goalIdSchema,
    record_version: z.number().int().positive(),
    description: z.string().min(1),
    status: goalStatusSchema,
    priority: z.number().finite(),
    created_at: z.number().finite(),
    target_at: z.number().finite().nullable(),
  })
  .strict();

export type GoalTargetAtRepairInput = {
  dataDir: string;
  goalIds: readonly GoalId[];
};

type GoalTargetAtRepairSnapshot = {
  id: GoalId;
  recordVersion: number;
  description: string;
  status: GoalStatus;
  priority: number;
  createdAt: number;
  currentTargetAt: number | null;
};

export type GoalTargetAtRepairCandidate = GoalTargetAtRepairSnapshot & {
  action: "clear";
  currentTargetAt: number;
  patch: { target_at: null };
};

export type GoalTargetAtRepairSkip = GoalTargetAtRepairSnapshot & {
  action: "skip";
  currentTargetAt: null;
  reason: "target_at is already NULL";
};

export type GoalTargetAtRepairEntry = GoalTargetAtRepairCandidate | GoalTargetAtRepairSkip;

export type GoalTargetAtRepairRefusal = {
  id: GoalId;
  message: "goal does not exist";
};

export type GoalTargetAtRepairPlan = {
  dataDir: string;
  requestedGoalIds: GoalId[];
  entries: GoalTargetAtRepairEntry[];
  candidates: GoalTargetAtRepairCandidate[];
  skipped: GoalTargetAtRepairSkip[];
  refusals: GoalTargetAtRepairRefusal[];
};

export type GoalTargetAtRepairFailure = {
  id: string;
  message: string;
};

export type GoalTargetAtRepairReport = {
  dryRun: boolean;
  plan: GoalTargetAtRepairPlan;
  applied: GoalTargetAtRepairCandidate[];
  failures: GoalTargetAtRepairFailure[];
};

type GoalTargetAtRepairCliArgs =
  | { help: true }
  | ({ help: false; apply: boolean } & GoalTargetAtRepairInput);

type RepairOutput = {
  write(chunk: string): unknown;
};

export type GoalTargetAtRepairMainOptions = {
  env?: NodeJS.ProcessEnv;
  openBorg?: (options: BorgOpenOptions) => Promise<Borg>;
  stdout?: RepairOutput;
  stderr?: RepairOutput;
};

function describeZodError(error: z.ZodError): string {
  return error.issues
    .map((issue) => `${issue.path.join(".") || "input"}: ${issue.message}`)
    .join("; ");
}

function parseRepairInput(input: unknown): GoalTargetAtRepairInput {
  const parsed = repairInputSchema.safeParse(input);

  if (!parsed.success) {
    throw new Error(`Invalid goal target_at repair input: ${describeZodError(parsed.error)}`);
  }

  return {
    ...parsed.data,
    goalIds: [...new Set(parsed.data.goalIds)],
  };
}

function openPlanningDatabase(path: string): SqliteDatabase {
  let raw: SqliteRawDatabase | undefined;

  try {
    raw = new SqliteRawDatabase(
      new DatabaseSync(path, {
        enableDoubleQuotedStringLiterals: true,
        readOnly: true,
      }),
    );
    const db = new SqliteDatabase(raw);
    db.pragma("busy_timeout = 5000");
    db.pragma("foreign_keys = ON");
    db.pragma("query_only = ON");
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

function snapshotFromPlanningRow(row: Record<string, unknown>): GoalTargetAtRepairSnapshot {
  const parsed = planningGoalRowSchema.parse({
    ...row,
    record_version: Number(row.record_version),
    priority: Number(row.priority),
    created_at: Number(row.created_at),
    target_at: row.target_at === null || row.target_at === undefined ? null : Number(row.target_at),
  });

  return {
    id: parsed.id,
    recordVersion: parsed.record_version,
    description: parsed.description,
    status: parsed.status,
    priority: parsed.priority,
    createdAt: parsed.created_at,
    currentTargetAt: parsed.target_at,
  };
}

/** Build a read-only repair plan from an on-disk Borg bank. */
export function planGoalTargetAtRepair(input: unknown): GoalTargetAtRepairPlan {
  const parsed = parseRepairInput(input);
  const databasePath = join(parsed.dataDir, "borg.db");

  if (!existsSync(databasePath)) {
    throw new Error(`No borg.db found in data directory ${parsed.dataDir}`);
  }

  const db = openPlanningDatabase(databasePath);

  try {
    const placeholders = parsed.goalIds.map(() => "?").join(", ");
    const rows = db
      .prepare(
        `
          SELECT id, record_version, description, status, priority, created_at, target_at
          FROM goals
          WHERE id IN (${placeholders})
        `,
      )
      .all(...parsed.goalIds) as Record<string, unknown>[];
    const rowsById = new Map(
      rows.map((row) => {
        const snapshot = snapshotFromPlanningRow(row);
        return [snapshot.id, snapshot] as const;
      }),
    );
    const entries: GoalTargetAtRepairEntry[] = [];
    const candidates: GoalTargetAtRepairCandidate[] = [];
    const skipped: GoalTargetAtRepairSkip[] = [];
    const refusals: GoalTargetAtRepairRefusal[] = [];

    for (const goalId of parsed.goalIds) {
      const snapshot = rowsById.get(goalId);

      if (snapshot === undefined) {
        refusals.push({ id: goalId, message: "goal does not exist" });
        continue;
      }

      if (snapshot.currentTargetAt === null) {
        const skip: GoalTargetAtRepairSkip = {
          ...snapshot,
          action: "skip",
          currentTargetAt: null,
          reason: "target_at is already NULL",
        };
        entries.push(skip);
        skipped.push(skip);
        continue;
      }

      const candidate: GoalTargetAtRepairCandidate = {
        ...snapshot,
        action: "clear",
        currentTargetAt: snapshot.currentTargetAt,
        patch: { target_at: null },
      };
      entries.push(candidate);
      candidates.push(candidate);
    }

    return {
      dataDir: parsed.dataDir,
      requestedGoalIds: [...parsed.goalIds],
      entries,
      candidates,
      skipped,
      refusals,
    };
  } finally {
    db.close();
  }
}

function failureFor(candidate: GoalTargetAtRepairCandidate, message: string) {
  return { id: candidate.id, message } satisfies GoalTargetAtRepairFailure;
}

function currentCandidateFailure(
  borg: Borg,
  candidate: GoalTargetAtRepairCandidate,
): GoalTargetAtRepairFailure | null {
  const current = borg.self.goals.get(candidate.id);

  if (current === null) {
    return failureFor(candidate, "goal disappeared after planning");
  }

  if (current.record_version !== candidate.recordVersion) {
    return failureFor(candidate, "goal changed after planning");
  }

  if (current.target_at !== candidate.currentTargetAt) {
    return failureFor(candidate, "goal target_at changed after planning");
  }

  return null;
}

function applyGoalTargetAtRepairPlan(
  borg: Borg,
  plan: GoalTargetAtRepairPlan,
): GoalTargetAtRepairReport {
  const report: GoalTargetAtRepairReport = {
    dryRun: false,
    plan,
    applied: [],
    failures: [],
  };

  for (const candidate of plan.candidates) {
    const failure = currentCandidateFailure(borg, candidate);

    if (failure !== null) {
      report.failures.push(failure);
      return report;
    }
  }

  for (const candidate of plan.candidates) {
    try {
      const result = borg.identity.updateGoal(
        candidate.id,
        goalPatchSchema.parse({ target_at: null }),
        { kind: "manual" },
        { throughReview: true, reason: GOAL_TARGET_AT_REPAIR_REASON },
      );

      if (result.status !== "applied") {
        report.failures.push(failureFor(candidate, "identity service returned requires_review"));
        break;
      }

      if (result.record.target_at !== null) {
        report.failures.push(
          failureFor(candidate, "identity service reported applied without clearing target_at"),
        );
        break;
      }

      report.applied.push(candidate);
    } catch (error) {
      report.failures.push(
        failureFor(candidate, error instanceof Error ? error.message : String(error)),
      );
      break;
    }
  }

  return report;
}

function normalizedDescription(value: string): string {
  return value.replaceAll("\r", " ").replaceAll("\n", " ").replaceAll("\t", " ");
}

function descriptionExcerpt(value: string): string {
  const characters = [...normalizedDescription(value)];
  return characters.length <= 80 ? characters.join("") : `${characters.slice(0, 79).join("")}…`;
}

function formatEntryTable(entries: readonly GoalTargetAtRepairEntry[]): string[] {
  const rows = entries.map((entry) => [
    entry.action,
    entry.id,
    entry.status,
    String(entry.priority),
    String(entry.createdAt),
    entry.currentTargetAt === null ? "NULL" : String(entry.currentTargetAt),
    JSON.stringify(descriptionExcerpt(entry.description)),
  ]);
  const headers = [
    "action",
    "id",
    "status",
    "priority",
    "created_at",
    "current target_at",
    "description",
  ];
  const widths = headers.map((header, column) =>
    Math.max(header.length, ...rows.map((row) => row[column]?.length ?? 0)),
  );
  const renderRow = (row: readonly string[]) =>
    row.map((cell, column) => cell.padEnd(widths[column] ?? cell.length)).join(" | ");

  return [
    renderRow(headers),
    widths.map((width) => "-".repeat(width)).join("-+-"),
    ...rows.map((row) => renderRow(row)),
  ];
}

export function formatGoalTargetAtRepairReport(report: GoalTargetAtRepairReport): string {
  const failureCount = report.plan.refusals.length + report.failures.length;
  const lines = [`mode=${report.dryRun ? "dry-run" : "apply"}`];

  if (report.plan.entries.length === 0) {
    lines.push("(no existing requested goals)");
  } else {
    lines.push(...formatEntryTable(report.plan.entries));
  }

  for (const skip of report.plan.skipped) {
    lines.push(`skipped id=${skip.id} reason=${JSON.stringify(skip.reason)}`);
  }

  for (const refusal of report.plan.refusals) {
    lines.push(`refused id=${refusal.id} reason=${JSON.stringify(refusal.message)}`);
  }

  lines.push(
    `requested=${report.plan.requestedGoalIds.length} selected=${report.plan.candidates.length} skipped=${report.plan.skipped.length} refused=${report.plan.refusals.length}`,
  );

  if (!report.dryRun) {
    for (const candidate of report.applied) {
      lines.push(`applied id=${candidate.id} target_at=NULL`);
    }
    lines.push(`applied_total=${report.applied.length}`);
  }

  for (const failure of report.failures) {
    lines.push(`failure id=${failure.id} message=${JSON.stringify(failure.message)}`);
  }

  lines.push(`failures=${failureCount}`);
  return `${lines.join("\n")}\n`;
}

export function goalTargetAtRepairExitCode(report: GoalTargetAtRepairReport): 0 | 1 {
  return report.plan.refusals.length === 0 && report.failures.length === 0 ? 0 : 1;
}

export function parseGoalTargetAtRepairCliArgs(argv: readonly string[]): GoalTargetAtRepairCliArgs {
  let dataDir: string | undefined;
  let goals: string | undefined;
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

    if (argument === "--data-dir" || argument === "--goal") {
      const value = argv[index + 1];

      if (value === undefined || value.startsWith("--")) {
        throw new Error(`${argument} requires a value`);
      }

      if (argument === "--data-dir") {
        dataDir = value;
      } else {
        goals = value;
      }

      index += 1;
      continue;
    }

    throw new Error(`Unknown argument: ${argument ?? ""}`);
  }

  if (dataDir === undefined) {
    throw new Error("--data-dir is required");
  }

  if (goals === undefined) {
    throw new Error("--goal is required");
  }

  return {
    help: false,
    apply,
    ...parseRepairInput({
      dataDir,
      goalIds: goals.split(",").map((value) => value.trim()),
    }),
  };
}

function usage(): string {
  return [
    "Usage: pnpm tsx scripts/repair-goal-target-at.ts --data-dir <bank-dir>",
    "       --goal <goal-id,goal-id,...> [--apply]",
    "",
    "Dry-run is the default. Stop every Borg writer and take a verified backup before --apply.",
  ].join("\n");
}

export async function main(
  argv: readonly string[] = process.argv.slice(2),
  options: GoalTargetAtRepairMainOptions = {},
): Promise<0 | 1> {
  const args = parseGoalTargetAtRepairCliArgs(argv);
  const stdout = options.stdout ?? process.stdout;
  const stderr = options.stderr ?? process.stderr;

  if (args.help) {
    stdout.write(`${usage()}\n`);
    return 0;
  }

  stderr.write(
    "WARNING: this maintenance requires a verified backup and exclusive single-writer access.\n",
  );
  const plan = planGoalTargetAtRepair({ dataDir: args.dataDir, goalIds: args.goalIds });
  let report: GoalTargetAtRepairReport = {
    dryRun: !args.apply,
    plan,
    applied: [],
    failures: [],
  };

  if (args.apply && plan.refusals.length === 0 && plan.candidates.length > 0) {
    try {
      await withBorg(
        {
          dataDir: args.dataDir,
          env: options.env,
          openBorg: options.openBorg,
        },
        async (borg) => {
          report = applyGoalTargetAtRepairPlan(borg, plan);
        },
      );
    } catch (error) {
      report.failures.push({
        id: "repair",
        message: error instanceof Error ? error.message : String(error),
      });
    }
  }

  stdout.write(formatGoalTargetAtRepairReport(report));
  const exitCode = goalTargetAtRepairExitCode(report);

  if (exitCode !== 0) {
    stderr.write("ERROR: goal target_at repair failed or was refused; inspect the report.\n");
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
