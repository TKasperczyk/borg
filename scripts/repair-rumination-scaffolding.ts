/*
 * Repair open-question rumination tensions that contain model tool-call
 * `<parameter>` wire scaffolding around the actual tension payload.
 *
 * Run with every Borg writer stopped and take a verified backup first. Dry-run
 * is the default; --apply updates recoverable rows and writes one
 * maintenance_audit row per repaired rumination. Rows whose wrappers cannot be
 * safely assigned to tensions are reported for manual decision and never
 * deleted or changed.
 *
 * Usage:
 *   pnpm tsx scripts/repair-rumination-scaffolding.ts --data-dir <bank-dir>
 *   pnpm tsx scripts/repair-rumination-scaffolding.ts --data-dir <bank-dir> --apply
 */
import { existsSync } from "node:fs";
import { join, resolve } from "node:path";
import { DatabaseSync } from "node:sqlite";
import { pathToFileURL } from "node:url";

import { z } from "zod";

import { unwrapTensionParameterScaffolding } from "../src/offline/ruminator/index.js";
import { AuditLog } from "../src/offline/audit-log.js";
import { SqliteDatabase, SqliteRawDatabase } from "../src/storage/sqlite/index.js";
import { SystemClock } from "../src/util/clock.js";
import { createMaintenanceRunId, type MaintenanceRunId } from "../src/util/ids.js";
import { serializeJsonValue } from "../src/util/json-value.js";

export const RUMINATION_SCAFFOLDING_REPAIR_AUDIT_ACTION = "repair_rumination_parameter_scaffolding";

const repairRuminationRowSchema = z.object({
  id: z.number().int().positive(),
  tensions: z.string(),
});
const storedTensionsSchema = z.array(z.string());
const repairedTensionsSchema = z.array(
  z
    .string()
    .min(1)
    .refine((value) => !value.includes("<parameter"), {
      message: "Repaired tension still contains parameter scaffolding",
    }),
);

export type RuminationScaffoldingRepairRow = z.infer<typeof repairRuminationRowSchema>;

export type RuminationScaffoldingRepairCandidate = {
  id: number;
  beforeTensions: string[];
  afterTensions: string[];
  beforeSerializedTensions: string;
  afterSerializedTensions: string;
  beforeLength: number;
  afterLength: number;
  beforeElementCount: number;
  afterElementCount: number;
};

export type RuminationScaffoldingManualDecision = {
  id: number;
  beforeLength: number;
  beforeElementCount: number;
  reason: string;
};

export type RuminationScaffoldingPlan = {
  scannedRows: number;
  candidates: RuminationScaffoldingRepairCandidate[];
  manualDecisions: RuminationScaffoldingManualDecision[];
};

export type RuminationScaffoldingRepairReport = RuminationScaffoldingPlan & {
  dryRun: boolean;
  repaired: RuminationScaffoldingRepairCandidate[];
};

type RepairDependencies = {
  db: SqliteDatabase;
  auditLog: Pick<AuditLog, "record">;
  runId: MaintenanceRunId;
};

function parseStoredTensions(row: RuminationScaffoldingRepairRow): string[] {
  let parsed: unknown;

  try {
    parsed = JSON.parse(row.tensions) as unknown;
  } catch (error) {
    throw new Error(`Rumination ${row.id} has invalid tensions JSON`, { cause: error });
  }

  const result = storedTensionsSchema.safeParse(parsed);

  if (!result.success) {
    throw new Error(`Rumination ${row.id} tensions failed array-of-strings validation`, {
      cause: result.error,
    });
  }

  return result.data;
}

function startsWithParameterScaffolding(value: string): boolean {
  return value.startsWith("<parameter");
}

function manualDecisionReason(error: unknown): string {
  return error instanceof Error ? error.message : "Unknown parameter scaffolding shape";
}

function totalTensionLength(tensions: readonly string[]): number {
  return tensions.reduce((total, tension) => total + tension.length, 0);
}

/**
 * Purely plan repairs from SQLite-shaped rows. Selection and transformation key
 * only on the tool-wire `<parameter` delimiter and the ruminator's exported
 * structural unwrap contract.
 */
export function planRuminationScaffoldingRepairs(
  rows: readonly unknown[],
): RuminationScaffoldingPlan {
  const candidates: RuminationScaffoldingRepairCandidate[] = [];
  const manualDecisions: RuminationScaffoldingManualDecision[] = [];

  for (const rawRow of rows) {
    const row = repairRuminationRowSchema.parse(rawRow);
    const beforeTensions = parseStoredTensions(row);

    if (!beforeTensions.some(startsWithParameterScaffolding)) {
      continue;
    }

    try {
      const afterTensions = repairedTensionsSchema.parse(
        beforeTensions.flatMap((tension) =>
          startsWithParameterScaffolding(tension)
            ? unwrapTensionParameterScaffolding(tension)
            : [tension],
        ),
      );
      const afterSerializedTensions = serializeJsonValue(afterTensions);

      candidates.push({
        id: row.id,
        beforeTensions,
        afterTensions,
        beforeSerializedTensions: row.tensions,
        afterSerializedTensions,
        beforeLength: totalTensionLength(beforeTensions),
        afterLength: totalTensionLength(afterTensions),
        beforeElementCount: beforeTensions.length,
        afterElementCount: afterTensions.length,
      });
    } catch (error) {
      manualDecisions.push({
        id: row.id,
        beforeLength: totalTensionLength(beforeTensions),
        beforeElementCount: beforeTensions.length,
        reason: manualDecisionReason(error),
      });
    }
  }

  return {
    scannedRows: rows.length,
    candidates,
    manualDecisions,
  };
}

function scanRuminationRows(db: SqliteDatabase): RuminationScaffoldingRepairRow[] {
  return db
    .prepare(
      `
        SELECT id, tensions
        FROM open_question_ruminations
        ORDER BY id ASC
      `,
    )
    .all()
    .map((row) => repairRuminationRowSchema.parse(row));
}

function inImmediateTransaction<T>(db: SqliteDatabase, operation: () => T): T {
  db.exec("BEGIN IMMEDIATE");

  try {
    const result = operation();
    db.exec("COMMIT");
    return result;
  } catch (error) {
    try {
      if (db.raw.inTransaction) {
        db.exec("ROLLBACK");
      }
    } catch {
      // Preserve the original failure.
    }

    throw error;
  }
}

export function repairRuminationScaffolding(
  dependencies: RepairDependencies,
  options: { apply?: boolean } = {},
): RuminationScaffoldingRepairReport {
  const plan = planRuminationScaffoldingRepairs(scanRuminationRows(dependencies.db));
  const apply = options.apply === true;
  const repaired: RuminationScaffoldingRepairCandidate[] = [];

  if (apply) {
    for (const candidate of plan.candidates) {
      inImmediateTransaction(dependencies.db, () => {
        const current = dependencies.db
          .prepare("SELECT id, tensions FROM open_question_ruminations WHERE id = ?")
          .get(candidate.id);

        if (current === undefined) {
          throw new Error(`Rumination ${candidate.id} disappeared before repair`);
        }

        const parsedCurrent = repairRuminationRowSchema.parse(current);

        if (parsedCurrent.tensions !== candidate.beforeSerializedTensions) {
          throw new Error(`Rumination ${candidate.id} changed after repair discovery`);
        }

        const result = dependencies.db
          .prepare(
            `
              UPDATE open_question_ruminations
              SET tensions = ?
              WHERE id = ? AND tensions = ?
            `,
          )
          .run(candidate.afterSerializedTensions, candidate.id, candidate.beforeSerializedTensions);

        if (result.changes !== 1) {
          throw new Error(`Rumination ${candidate.id} was not updated`);
        }

        dependencies.auditLog.record({
          run_id: dependencies.runId,
          process: "ruminator",
          action: RUMINATION_SCAFFOLDING_REPAIR_AUDIT_ACTION,
          targets: {
            rumination_id: candidate.id,
            before_length: candidate.beforeLength,
            after_length: candidate.afterLength,
            before_element_count: candidate.beforeElementCount,
            after_element_count: candidate.afterElementCount,
          },
          reversal: {
            no_reverser: true,
            previous_tensions: candidate.beforeTensions,
          },
        });
      });
      repaired.push(candidate);
    }
  }

  return {
    dryRun: !apply,
    ...plan,
    repaired,
  };
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

type RepairCliArgs =
  | { help: true }
  | {
      help: false;
      dataDir: string;
      apply: boolean;
    };

export function parseRepairCliArgs(argv: readonly string[]): RepairCliArgs {
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
        throw new Error("--data-dir requires a path");
      }

      dataDir = value;
      index += 1;
      continue;
    }

    if (argument !== undefined && !argument.startsWith("--") && dataDir === undefined) {
      dataDir = argument;
      continue;
    }

    throw new Error(`Unknown argument: ${argument ?? ""}`);
  }

  if (dataDir === undefined || dataDir.trim().length === 0) {
    throw new Error("A bank data directory is required");
  }

  return {
    help: false,
    dataDir: resolve(dataDir),
    apply,
  };
}

function usage(): string {
  return [
    "Usage: pnpm tsx scripts/repair-rumination-scaffolding.ts --data-dir <bank-dir> [--apply]",
    "",
    "Dry-run is the default. Stop every Borg writer and take a verified backup before --apply.",
  ].join("\n");
}

export function formatRepairReport(report: RuminationScaffoldingRepairReport): string {
  const lines = [
    `mode=${report.dryRun ? "dry-run" : "apply"}`,
    `scanned_rows=${report.scannedRows}`,
    `repair_candidates=${report.candidates.length}`,
    `repaired=${report.repaired.length}`,
    `manual_decisions=${report.manualDecisions.length}`,
  ];
  const repairedIds = new Set(report.repaired.map((candidate) => candidate.id));

  for (const candidate of report.candidates) {
    const action = report.dryRun
      ? "would_repair"
      : repairedIds.has(candidate.id)
        ? "repaired"
        : "not_repaired";
    lines.push(
      `${action} row=${candidate.id} before_length=${candidate.beforeLength} before_elements=${candidate.beforeElementCount} after_length=${candidate.afterLength} after_elements=${candidate.afterElementCount}`,
    );
  }

  for (const manual of report.manualDecisions) {
    lines.push(
      `manual_decision row=${manual.id} before_length=${manual.beforeLength} before_elements=${manual.beforeElementCount} after_length=unavailable after_elements=unavailable reason=${JSON.stringify(manual.reason)}`,
    );
  }

  return `${lines.join("\n")}\n`;
}

export function repairReportExitCode(report: RuminationScaffoldingRepairReport): 0 | 1 {
  return !report.dryRun && report.manualDecisions.length > 0 ? 1 : 0;
}

export async function main(argv: readonly string[] = process.argv.slice(2)): Promise<0 | 1> {
  const args = parseRepairCliArgs(argv);

  if (args.help) {
    process.stdout.write(`${usage()}\n`);
    return 0;
  }

  const databasePath = join(args.dataDir, "borg.db");

  if (!existsSync(databasePath)) {
    throw new Error(`No borg.db found in data directory ${args.dataDir}`);
  }

  process.stderr.write(
    "WARNING: this maintenance requires a verified backup and exclusive single-writer access.\n",
  );
  const db = openRepairDatabase(databasePath, !args.apply);
  const clock = new SystemClock();

  try {
    const report = repairRuminationScaffolding(
      {
        db,
        auditLog: new AuditLog({ db, clock }),
        runId: createMaintenanceRunId(),
      },
      { apply: args.apply },
    );
    process.stdout.write(formatRepairReport(report));
    const exitCode = repairReportExitCode(report);

    if (exitCode !== 0) {
      process.stderr.write(
        "ERROR: apply completed with rumination rows that need a manual decision; inspect the report.\n",
      );
    }

    return exitCode;
  } finally {
    db.close();
  }
}

if (process.argv[1] !== undefined && import.meta.url === pathToFileURL(process.argv[1]).href) {
  main().then(
    (exitCode) => {
      process.exitCode = exitCode;
    },
    (error: unknown) => {
      process.stderr.write(
        `${error instanceof Error ? (error.stack ?? error.message) : String(error)}\n`,
      );
      process.exitCode = 1;
    },
  );
}
