/*
 * Re-aim active commitments and goals from legacy BotArena thread audiences
 * to the durable audience entity for the continuous room.
 *
 * Run with every Borg writer stopped and take a verified backup first. Dry-run
 * is the default; --apply routes each write through Borg's public identity
 * service so the corresponding identity_events row is written atomically.
 * Stream entries, episodes, open questions, and inactive rows are not changed.
 *
 * Usage:
 *   pnpm tsx scripts/migrate-audience-scoping.ts --data-dir <bank-dir> \
 *     --from <entity-id,entity-id,...> --to <entity-id>
 *   pnpm tsx scripts/migrate-audience-scoping.ts --data-dir <bank-dir> \
 *     --from <entity-id,entity-id,...> --to <entity-id> --apply
 */
import { existsSync } from "node:fs";
import { join, resolve } from "node:path";
import { DatabaseSync } from "node:sqlite";
import { pathToFileURL } from "node:url";

import { z } from "zod";

import type { Borg, BorgOpenOptions } from "../src/index.js";
import {
  commitmentIdSchema,
  commitmentPatchSchema,
  entityIdSchema,
  entityKindSchema,
} from "../src/memory/commitments/index.js";
import { goalIdSchema, goalPatchSchema } from "../src/memory/self/index.js";
import { SqliteDatabase, SqliteRawDatabase } from "../src/storage/sqlite/index.js";
import type { CommitmentId, EntityId, GoalId } from "../src/util/ids.js";
import { withBorg } from "../src/cli/helpers/borg.js";

const migrationInputSchema = z
  .object({
    dataDir: z
      .string()
      .trim()
      .min(1, "--data-dir requires a non-empty path")
      .transform((value) => resolve(value)),
    fromEntityIds: z.array(entityIdSchema).min(1, "--from requires at least one entity id"),
    toEntityId: entityIdSchema,
  })
  .strict()
  .superRefine((input, context) => {
    if (input.fromEntityIds.some((entityId) => entityId === input.toEntityId)) {
      context.addIssue({
        code: "custom",
        message: `--from must not contain destination audience entity ${input.toEntityId}`,
        path: ["fromEntityIds"],
      });
    }
  });

const destinationRowSchema = z
  .object({
    id: entityIdSchema,
    canonical_name: z.string().min(1),
    kind: entityKindSchema.nullable(),
  })
  .strict();

const commitmentCandidateRowSchema = z
  .object({
    id: commitmentIdSchema,
    source_audience_entity_id: entityIdSchema,
    priority: z.number().int(),
    summary: z.string().min(1),
  })
  .strict();

const goalCandidateRowSchema = z
  .object({
    id: goalIdSchema,
    source_audience_entity_id: entityIdSchema,
    priority: z.number().finite(),
    summary: z.string().min(1),
  })
  .strict();

export type AudienceScopingMigrationInput = {
  dataDir: string;
  fromEntityIds: readonly EntityId[];
  toEntityId: EntityId;
};

export type AudienceScopingCommitmentCandidate = {
  kind: "commitment";
  id: CommitmentId;
  sourceAudienceEntityId: EntityId;
  priority: number;
  summary: string;
};

export type AudienceScopingGoalCandidate = {
  kind: "goal";
  id: GoalId;
  sourceAudienceEntityId: EntityId;
  priority: number;
  summary: string;
};

export type AudienceScopingMigrationCandidate =
  | AudienceScopingCommitmentCandidate
  | AudienceScopingGoalCandidate;

export type AudienceScopingMigrationPlan = {
  dataDir: string;
  fromEntityIds: EntityId[];
  toEntityId: EntityId;
  destinationName: string;
  candidates: AudienceScopingMigrationCandidate[];
};

export type AudienceScopingMigrationFailure = {
  kind: AudienceScopingMigrationCandidate["kind"] | "migration";
  id: string;
  sourceAudienceEntityId: EntityId | null;
  message: string;
};

export type AudienceScopingMigrationReport = {
  dryRun: boolean;
  plan: AudienceScopingMigrationPlan;
  applied: AudienceScopingMigrationCandidate[];
  failures: AudienceScopingMigrationFailure[];
};

type AudienceScopingCliArgs =
  | { help: true }
  | ({ help: false; apply: boolean } & AudienceScopingMigrationInput);

type MigrationOutput = {
  write(chunk: string): unknown;
};

export type AudienceScopingMainOptions = {
  env?: NodeJS.ProcessEnv;
  openBorg?: (options: BorgOpenOptions) => Promise<Borg>;
  stdout?: MigrationOutput;
  stderr?: MigrationOutput;
};

function describeZodError(error: z.ZodError): string {
  return error.issues
    .map((issue) => `${issue.path.join(".") || "input"}: ${issue.message}`)
    .join("; ");
}

function parseMigrationInput(input: unknown): AudienceScopingMigrationInput {
  const parsed = migrationInputSchema.safeParse(input);

  if (!parsed.success) {
    throw new Error(`Invalid audience migration input: ${describeZodError(parsed.error)}`);
  }

  return {
    ...parsed.data,
    fromEntityIds: [...new Set(parsed.data.fromEntityIds)],
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

function candidateKindRank(kind: AudienceScopingMigrationCandidate["kind"]): number {
  return kind === "commitment" ? 0 : 1;
}

/**
 * Build a read-only migration plan from an on-disk Borg bank.
 *
 * This function validates every supplied handle and selected row and never
 * mutates the bank. Apply mode consumes its result through the identity facade.
 */
export function planAudienceScopingMigration(input: unknown): AudienceScopingMigrationPlan {
  const parsed = parseMigrationInput(input);
  const databasePath = join(parsed.dataDir, "borg.db");

  if (!existsSync(databasePath)) {
    throw new Error(`No borg.db found in data directory ${parsed.dataDir}`);
  }

  const db = openPlanningDatabase(databasePath);

  try {
    const destinationRow = db
      .prepare(
        `
          SELECT id, canonical_name, kind
          FROM entities
          WHERE id = ?
        `,
      )
      .get(parsed.toEntityId) as Record<string, unknown> | undefined;

    if (destinationRow === undefined) {
      throw new Error(`Destination audience entity does not exist: ${parsed.toEntityId}`);
    }

    const destination = destinationRowSchema.parse(destinationRow);

    if (destination.kind !== "group") {
      throw new Error(
        `Destination audience entity ${destination.id} must have kind "group"; found ${JSON.stringify(destination.kind)}`,
      );
    }

    const placeholders = parsed.fromEntityIds.map(() => "?").join(", ");
    const commitmentRows = db
      .prepare(
        `
          SELECT id, restricted_audience AS source_audience_entity_id,
                 priority, directive AS summary
          FROM commitments
          WHERE revoked_at IS NULL
            AND expired_at IS NULL
            AND superseded_by IS NULL
            AND restricted_audience IN (${placeholders})
        `,
      )
      .all(...parsed.fromEntityIds) as Record<string, unknown>[];
    const goalRows = db
      .prepare(
        `
          SELECT id, audience_entity_id AS source_audience_entity_id,
                 priority, description AS summary
          FROM goals
          WHERE status = 'active'
            AND audience_entity_id IN (${placeholders})
        `,
      )
      .all(...parsed.fromEntityIds) as Record<string, unknown>[];
    const candidates: AudienceScopingMigrationCandidate[] = [
      ...commitmentRows.map((row): AudienceScopingCommitmentCandidate => {
        const candidate = commitmentCandidateRowSchema.parse({
          ...row,
          priority: Number(row.priority),
        });
        return {
          kind: "commitment",
          id: candidate.id,
          sourceAudienceEntityId: candidate.source_audience_entity_id,
          priority: candidate.priority,
          summary: candidate.summary,
        };
      }),
      ...goalRows.map((row): AudienceScopingGoalCandidate => {
        const candidate = goalCandidateRowSchema.parse({
          ...row,
          priority: Number(row.priority),
        });
        return {
          kind: "goal",
          id: candidate.id,
          sourceAudienceEntityId: candidate.source_audience_entity_id,
          priority: candidate.priority,
          summary: candidate.summary,
        };
      }),
    ];
    const sourceOrder = new Map(
      parsed.fromEntityIds.map((entityId, index) => [entityId, index] as const),
    );

    candidates.sort(
      (left, right) =>
        (sourceOrder.get(left.sourceAudienceEntityId) ?? Number.MAX_SAFE_INTEGER) -
          (sourceOrder.get(right.sourceAudienceEntityId) ?? Number.MAX_SAFE_INTEGER) ||
        candidateKindRank(left.kind) - candidateKindRank(right.kind) ||
        right.priority - left.priority ||
        left.id.localeCompare(right.id),
    );

    return {
      dataDir: parsed.dataDir,
      fromEntityIds: [...parsed.fromEntityIds],
      toEntityId: parsed.toEntityId,
      destinationName: destination.canonical_name,
      candidates,
    };
  } finally {
    db.close();
  }
}

function migrationReason(
  candidate: AudienceScopingMigrationCandidate,
  toEntityId: EntityId,
): string {
  return `BotArena continuous-room audience migration: ${candidate.sourceAudienceEntityId} -> ${toEntityId}`;
}

function failureFor(
  candidate: AudienceScopingMigrationCandidate,
  message: string,
): AudienceScopingMigrationFailure {
  return {
    kind: candidate.kind,
    id: candidate.id,
    sourceAudienceEntityId: candidate.sourceAudienceEntityId,
    message,
  };
}

function currentCandidateFailure(
  borg: Borg,
  candidate: AudienceScopingMigrationCandidate,
): AudienceScopingMigrationFailure | null {
  if (candidate.kind === "commitment") {
    const current = borg.commitments.get(candidate.id);

    if (current === null) {
      return failureFor(candidate, "commitment disappeared after planning");
    }

    if (
      current.revoked_at !== null ||
      current.expired_at !== null ||
      current.superseded_by !== null ||
      current.restricted_audience !== candidate.sourceAudienceEntityId
    ) {
      return failureFor(candidate, "commitment is no longer an active row at its planned audience");
    }

    return null;
  }

  const current = borg.self.goals.get(candidate.id);

  if (current === null) {
    return failureFor(candidate, "goal disappeared after planning");
  }

  if (
    current.status !== "active" ||
    current.audience_entity_id !== candidate.sourceAudienceEntityId
  ) {
    return failureFor(candidate, "goal is no longer active at its planned audience");
  }

  return null;
}

function applyAudienceScopingPlan(
  borg: Borg,
  plan: AudienceScopingMigrationPlan,
): AudienceScopingMigrationReport {
  const report: AudienceScopingMigrationReport = {
    dryRun: false,
    plan,
    applied: [],
    failures: [],
  };
  const destination = borg.entities.get(plan.toEntityId);

  if (destination === null) {
    report.failures.push({
      kind: "migration",
      id: plan.toEntityId,
      sourceAudienceEntityId: null,
      message: "destination audience entity disappeared after planning",
    });
    return report;
  }

  if (destination.kind !== "group") {
    report.failures.push({
      kind: "migration",
      id: plan.toEntityId,
      sourceAudienceEntityId: null,
      message: `destination audience entity is no longer kind "group"; found ${JSON.stringify(destination.kind)}`,
    });
    return report;
  }

  for (const candidate of plan.candidates) {
    const failure = currentCandidateFailure(borg, candidate);

    if (failure !== null) {
      report.failures.push(failure);
      return report;
    }
  }

  for (const candidate of plan.candidates) {
    try {
      const reason = migrationReason(candidate, plan.toEntityId);
      if (candidate.kind === "commitment") {
        const result = borg.identity.updateCommitment(
          candidate.id,
          commitmentPatchSchema.parse({ restricted_audience: plan.toEntityId }),
          { kind: "manual" },
          { throughReview: true, reason },
        );

        if (result.status !== "applied") {
          report.failures.push(failureFor(candidate, "identity service returned requires_review"));
          break;
        }

        if (result.record.restricted_audience !== plan.toEntityId) {
          report.failures.push(
            failureFor(
              candidate,
              "identity service reported applied without the destination audience",
            ),
          );
          break;
        }
      } else {
        const result = borg.identity.updateGoal(
          candidate.id,
          goalPatchSchema.parse({ audience_entity_id: plan.toEntityId }),
          { kind: "manual" },
          { throughReview: true, reason },
        );

        if (result.status !== "applied") {
          report.failures.push(failureFor(candidate, "identity service returned requires_review"));
          break;
        }

        if (result.record.audience_entity_id !== plan.toEntityId) {
          report.failures.push(
            failureFor(
              candidate,
              "identity service reported applied without the destination audience",
            ),
          );
          break;
        }
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

function normalizedSummary(value: string): string {
  return value.replaceAll("\r", " ").replaceAll("\n", " ").replaceAll("\t", " ");
}

function summaryExcerpt(value: string): string {
  const characters = [...normalizedSummary(value)];
  return characters.length <= 80 ? characters.join("") : `${characters.slice(0, 79).join("")}…`;
}

function countCandidates(candidates: readonly AudienceScopingMigrationCandidate[]): {
  commitments: number;
  goals: number;
  total: number;
} {
  const commitments = candidates.filter((candidate) => candidate.kind === "commitment").length;
  const goals = candidates.length - commitments;
  return { commitments, goals, total: candidates.length };
}

function formatCandidateTable(candidates: readonly AudienceScopingMigrationCandidate[]): string[] {
  const rows = candidates.map((candidate) => [
    candidate.kind,
    candidate.id,
    candidate.sourceAudienceEntityId,
    String(candidate.priority),
    JSON.stringify(summaryExcerpt(candidate.summary)),
  ]);
  const headers = ["kind", "id", "current audience", "priority", "directive/description"];
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

export function formatAudienceScopingMigrationReport(
  report: AudienceScopingMigrationReport,
): string {
  const lines = [
    `mode=${report.dryRun ? "dry-run" : "apply"}`,
    `destination=${report.plan.toEntityId} name=${JSON.stringify(report.plan.destinationName)}`,
  ];

  for (const sourceAudienceEntityId of report.plan.fromEntityIds) {
    const candidates = report.plan.candidates.filter(
      (candidate) => candidate.sourceAudienceEntityId === sourceAudienceEntityId,
    );
    const counts = countCandidates(candidates);
    lines.push("", `source_audience=${sourceAudienceEntityId}`);

    if (candidates.length === 0) {
      lines.push("(no selected rows)");
    } else {
      lines.push(...formatCandidateTable(candidates));
    }

    lines.push(
      `source_count audience=${sourceAudienceEntityId} commitments=${counts.commitments} goals=${counts.goals} total=${counts.total}`,
    );
  }

  const totals = countCandidates(report.plan.candidates);
  lines.push(
    "",
    `total commitments=${totals.commitments} goals=${totals.goals} rows=${totals.total}`,
  );

  if (!report.dryRun) {
    for (const sourceAudienceEntityId of report.plan.fromEntityIds) {
      const applied = report.applied.filter(
        (candidate) => candidate.sourceAudienceEntityId === sourceAudienceEntityId,
      );
      const counts = countCandidates(applied);
      lines.push(
        `applied_count audience=${sourceAudienceEntityId} commitments=${counts.commitments} goals=${counts.goals} total=${counts.total}`,
      );
    }

    const appliedTotals = countCandidates(report.applied);
    lines.push(
      `applied_total commitments=${appliedTotals.commitments} goals=${appliedTotals.goals} rows=${appliedTotals.total}`,
      `failures=${report.failures.length}`,
    );

    for (const failure of report.failures) {
      lines.push(
        `failure kind=${failure.kind} id=${failure.id} source_audience=${failure.sourceAudienceEntityId ?? "none"} message=${JSON.stringify(failure.message)}`,
      );
    }
  }

  return `${lines.join("\n")}\n`;
}

export function audienceScopingMigrationExitCode(report: AudienceScopingMigrationReport): 0 | 1 {
  return report.failures.length === 0 ? 0 : 1;
}

export function parseAudienceScopingCliArgs(argv: readonly string[]): AudienceScopingCliArgs {
  let dataDir: string | undefined;
  let from: string | undefined;
  let to: string | undefined;
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

    if (argument === "--data-dir" || argument === "--from" || argument === "--to") {
      const value = argv[index + 1];

      if (value === undefined || value.startsWith("--")) {
        throw new Error(`${argument} requires a value`);
      }

      if (argument === "--data-dir") {
        dataDir = value;
      } else if (argument === "--from") {
        from = value;
      } else {
        to = value;
      }

      index += 1;
      continue;
    }

    throw new Error(`Unknown argument: ${argument ?? ""}`);
  }

  if (dataDir === undefined) {
    throw new Error("--data-dir is required");
  }

  if (from === undefined) {
    throw new Error("--from is required");
  }

  if (to === undefined) {
    throw new Error("--to is required");
  }

  return {
    help: false,
    apply,
    ...parseMigrationInput({
      dataDir,
      fromEntityIds: from.split(",").map((value) => value.trim()),
      toEntityId: to,
    }),
  };
}

function usage(): string {
  return [
    "Usage: pnpm tsx scripts/migrate-audience-scoping.ts --data-dir <bank-dir>",
    "       --from <entity-id,entity-id,...> --to <entity-id> [--apply]",
    "",
    "Dry-run is the default. Stop every Borg writer and take a verified backup before --apply.",
  ].join("\n");
}

export async function main(
  argv: readonly string[] = process.argv.slice(2),
  options: AudienceScopingMainOptions = {},
): Promise<0 | 1> {
  const args = parseAudienceScopingCliArgs(argv);
  const stdout = options.stdout ?? process.stdout;
  const stderr = options.stderr ?? process.stderr;

  if (args.help) {
    stdout.write(`${usage()}\n`);
    return 0;
  }

  stderr.write(
    "WARNING: this maintenance requires a verified backup and exclusive single-writer access.\n",
  );
  const plan = planAudienceScopingMigration({
    dataDir: args.dataDir,
    fromEntityIds: args.fromEntityIds,
    toEntityId: args.toEntityId,
  });
  let report: AudienceScopingMigrationReport = {
    dryRun: !args.apply,
    plan,
    applied: [],
    failures: [],
  };

  if (args.apply && plan.candidates.length > 0) {
    try {
      await withBorg(
        {
          dataDir: args.dataDir,
          env: options.env,
          openBorg: options.openBorg,
        },
        async (borg) => {
          report = applyAudienceScopingPlan(borg, plan);
        },
      );
    } catch (error) {
      report.failures.push({
        kind: "migration",
        id: plan.toEntityId,
        sourceAudienceEntityId: null,
        message: error instanceof Error ? error.message : String(error),
      });
    }
  }

  stdout.write(formatAudienceScopingMigrationReport(report));
  const exitCode = audienceScopingMigrationExitCode(report);

  if (exitCode !== 0) {
    stderr.write(
      "ERROR: audience migration aborted after a failed identity update; inspect the report.\n",
    );
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
