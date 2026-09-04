/*
 * Clear legacy speaker-as-owner values written by goal promotion. Creation
 * identity events, rather than the goal row's mutable provenance, select the
 * affected path. Goals created elsewhere are never candidates.
 *
 * Run with every Borg writer stopped and take a verified backup first. Dry-run
 * is the default; --apply routes each write through Borg's public identity
 * service so the corresponding identity_events row is written atomically.
 *
 * Usage:
 *   pnpm tsx scripts/repair-goal-speaker-owner.ts --data-dir <bank-dir> [--apply]
 */
import { existsSync } from "node:fs";
import { join, resolve } from "node:path";
import { DatabaseSync } from "node:sqlite";
import { pathToFileURL } from "node:url";

import { z } from "zod";

import type { Borg, BorgOpenOptions } from "../src/index.js";
import {
  goalIdSchema,
  goalOwnerEntityIdSchema,
  goalPatchSchema,
} from "../src/memory/self/index.js";
import { SqliteDatabase, SqliteRawDatabase } from "../src/storage/sqlite/index.js";
import { withBorg } from "../src/cli/helpers/borg.js";
import type { EntityId, GoalId } from "../src/util/ids.js";

const GOAL_PROMOTION_PROCESS = "goal-promotion-extractor";

export const GOAL_SPEAKER_OWNER_REPAIR_REASON =
  "operator repair: clear legacy non-self speaker-as-owner written by goal promotion; extracted goals are self-owned and owner_entity_id is disclosure authority";

const repairInputSchema = z
  .object({
    dataDir: z
      .string()
      .trim()
      .min(1, "--data-dir requires a non-empty path")
      .transform((value) => resolve(value)),
  })
  .strict();

const planningGoalRowSchema = z
  .object({
    id: goalIdSchema,
    record_version: z.number().int().positive(),
    owner_entity_id: goalOwnerEntityIdSchema.nullable(),
    creation_event_id: z.number().int().positive().nullable(),
    creation_provenance_process: z.string().nullable(),
    creation_owner_value_type: z.enum(["null", "text"]).nullable(),
    creation_owner_entity_id: goalOwnerEntityIdSchema.nullable(),
    later_owner_change_event_id: z.number().int().positive().nullable(),
  })
  .strict();

type GoalSpeakerOwnerRepairSnapshot = {
  id: GoalId;
  recordVersion: number;
  currentOwnerEntityId: EntityId | null;
  creationEventId: number | null;
  creationProvenanceProcess: string | null;
  creationOwnerRecorded: boolean;
  creationOwnerEntityId: EntityId | null;
  laterOwnerChangeEventId: number | null;
};

export type GoalSpeakerOwnerRepairCandidate = GoalSpeakerOwnerRepairSnapshot & {
  action: "clear";
  currentOwnerEntityId: EntityId;
  patch: { owner_entity_id: null };
};

export type GoalSpeakerOwnerRepairSkip = GoalSpeakerOwnerRepairSnapshot & {
  action: "skip";
  reason:
    | "creation identity event is missing"
    | "creation provenance is not goal-promotion-extractor"
    | "creation identity event does not record owner_entity_id"
    | "a later identity event changed owner_entity_id"
    | "current owner_entity_id differs from creation owner_entity_id"
    | "owner_entity_id is already NULL"
    | "owner_entity_id is self";
};

export type GoalSpeakerOwnerRepairPlan = {
  dataDir: string;
  selfEntityId: EntityId;
  entries: Array<GoalSpeakerOwnerRepairCandidate | GoalSpeakerOwnerRepairSkip>;
  candidates: GoalSpeakerOwnerRepairCandidate[];
  skipped: GoalSpeakerOwnerRepairSkip[];
  counts: {
    total: number;
    selected: number;
    creationEventMissing: number;
    otherCreationPath: number;
    creationOwnerMissing: number;
    laterOwnerChange: number;
    currentOwnerMismatch: number;
    promotionOwnerNull: number;
    promotionOwnerSelf: number;
  };
};

export type GoalSpeakerOwnerRepairReport = {
  dryRun: boolean;
  plan: GoalSpeakerOwnerRepairPlan;
  applied: GoalSpeakerOwnerRepairCandidate[];
  failures: Array<{ id: string; message: string }>;
};

type GoalSpeakerOwnerRepairCliArgs =
  | { help: true }
  | { help: false; apply: boolean; dataDir: string };

type RepairOutput = { write(chunk: string): unknown };

export type GoalSpeakerOwnerRepairMainOptions = {
  env?: NodeJS.ProcessEnv;
  openBorg?: (options: BorgOpenOptions) => Promise<Borg>;
  stdout?: RepairOutput;
  stderr?: RepairOutput;
};

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

function planningSnapshot(row: Record<string, unknown>): GoalSpeakerOwnerRepairSnapshot {
  const parsed = planningGoalRowSchema.parse({
    ...row,
    record_version: Number(row.record_version),
    creation_event_id:
      row.creation_event_id === null || row.creation_event_id === undefined
        ? null
        : Number(row.creation_event_id),
    creation_provenance_process:
      row.creation_provenance_process === null || row.creation_provenance_process === undefined
        ? null
        : String(row.creation_provenance_process),
    creation_owner_value_type:
      row.creation_owner_value_type === null || row.creation_owner_value_type === undefined
        ? null
        : String(row.creation_owner_value_type),
    creation_owner_entity_id:
      row.creation_owner_entity_id === null || row.creation_owner_entity_id === undefined
        ? null
        : String(row.creation_owner_entity_id),
    later_owner_change_event_id:
      row.later_owner_change_event_id === null || row.later_owner_change_event_id === undefined
        ? null
        : Number(row.later_owner_change_event_id),
  });

  return {
    id: parsed.id,
    recordVersion: parsed.record_version,
    currentOwnerEntityId: parsed.owner_entity_id,
    creationEventId: parsed.creation_event_id,
    creationProvenanceProcess: parsed.creation_provenance_process,
    creationOwnerRecorded: parsed.creation_owner_value_type !== null,
    creationOwnerEntityId: parsed.creation_owner_entity_id,
    laterOwnerChangeEventId: parsed.later_owner_change_event_id,
  };
}

/** Build a read-only repair plan from creation identity events in an on-disk Borg bank. */
export function planGoalSpeakerOwnerRepair(input: unknown): GoalSpeakerOwnerRepairPlan {
  const parsed = repairInputSchema.parse(input);
  const databasePath = join(parsed.dataDir, "borg.db");

  if (!existsSync(databasePath)) {
    throw new Error(`No borg.db found in data directory ${parsed.dataDir}`);
  }

  const db = openPlanningDatabase(databasePath);

  try {
    const selfRow = db
      .prepare(
        `
          SELECT id
          FROM entities
          WHERE kind = 'self'
          ORDER BY created_at ASC, id ASC
          LIMIT 1
        `,
      )
      .get() as { id?: unknown } | undefined;
    const selfEntityId = goalOwnerEntityIdSchema.parse(selfRow?.id);
    const rows = db
      .prepare(
        `
          WITH creation_events AS (
            SELECT
              event.*,
              ROW_NUMBER() OVER (
                PARTITION BY event.record_id
                ORDER BY event.ts ASC, event.id ASC
              ) AS creation_rank
            FROM identity_events AS event
            WHERE event.record_type = 'goal'
              AND event.action = 'create'
          )
          SELECT
            goal.id,
            goal.record_version,
            goal.owner_entity_id,
            creation.id AS creation_event_id,
            creation.provenance_process AS creation_provenance_process,
            json_type(
              creation.new_value_json,
              '$.owner_entity_id'
            ) AS creation_owner_value_type,
            json_extract(
              creation.new_value_json,
              '$.owner_entity_id'
            ) AS creation_owner_entity_id,
            (
              SELECT later.id
              FROM identity_events AS later
              WHERE later.record_type = 'goal'
                AND later.record_id = goal.id
                AND (
                  later.ts > creation.ts
                  OR (later.ts = creation.ts AND later.id > creation.id)
                )
                AND (
                  json_type(later.old_value_json, '$.owner_entity_id')
                    IS NOT json_type(later.new_value_json, '$.owner_entity_id')
                  OR json_extract(later.old_value_json, '$.owner_entity_id')
                    IS NOT json_extract(later.new_value_json, '$.owner_entity_id')
                )
              ORDER BY later.ts ASC, later.id ASC
              LIMIT 1
            ) AS later_owner_change_event_id
          FROM goals AS goal
          LEFT JOIN creation_events AS creation
            ON creation.record_id = goal.id
           AND creation.creation_rank = 1
          ORDER BY goal.created_at ASC, goal.id ASC
        `,
      )
      .all() as Record<string, unknown>[];
    const entries: Array<GoalSpeakerOwnerRepairCandidate | GoalSpeakerOwnerRepairSkip> = [];
    const candidates: GoalSpeakerOwnerRepairCandidate[] = [];
    const skipped: GoalSpeakerOwnerRepairSkip[] = [];
    const counts = {
      total: rows.length,
      selected: 0,
      creationEventMissing: 0,
      otherCreationPath: 0,
      creationOwnerMissing: 0,
      laterOwnerChange: 0,
      currentOwnerMismatch: 0,
      promotionOwnerNull: 0,
      promotionOwnerSelf: 0,
    };

    for (const row of rows) {
      const snapshot = planningSnapshot(row);
      let skipReason: GoalSpeakerOwnerRepairSkip["reason"] | null = null;

      if (snapshot.creationEventId === null) {
        skipReason = "creation identity event is missing";
        counts.creationEventMissing += 1;
      } else if (snapshot.creationProvenanceProcess !== GOAL_PROMOTION_PROCESS) {
        skipReason = "creation provenance is not goal-promotion-extractor";
        counts.otherCreationPath += 1;
      } else if (!snapshot.creationOwnerRecorded) {
        skipReason = "creation identity event does not record owner_entity_id";
        counts.creationOwnerMissing += 1;
      } else if (snapshot.laterOwnerChangeEventId !== null) {
        skipReason = "a later identity event changed owner_entity_id";
        counts.laterOwnerChange += 1;
      } else if (snapshot.currentOwnerEntityId !== snapshot.creationOwnerEntityId) {
        skipReason = "current owner_entity_id differs from creation owner_entity_id";
        counts.currentOwnerMismatch += 1;
      } else if (snapshot.currentOwnerEntityId === null) {
        skipReason = "owner_entity_id is already NULL";
        counts.promotionOwnerNull += 1;
      } else if (snapshot.currentOwnerEntityId === selfEntityId) {
        skipReason = "owner_entity_id is self";
        counts.promotionOwnerSelf += 1;
      }

      if (skipReason !== null) {
        const skip: GoalSpeakerOwnerRepairSkip = {
          ...snapshot,
          action: "skip",
          reason: skipReason,
        };
        entries.push(skip);
        skipped.push(skip);
        continue;
      }

      const candidate: GoalSpeakerOwnerRepairCandidate = {
        ...snapshot,
        action: "clear",
        currentOwnerEntityId: snapshot.currentOwnerEntityId as EntityId,
        // The legacy owner is the turn speaker, which is provenance rather than
        // evidence of whom the responsibility runs toward. Do not copy it into
        // counterparty_entity_id; untouched legacy rows keep that field null/unknown.
        patch: { owner_entity_id: null },
      };
      entries.push(candidate);
      candidates.push(candidate);
      counts.selected += 1;
    }

    return {
      dataDir: parsed.dataDir,
      selfEntityId,
      entries,
      candidates,
      skipped,
      counts,
    };
  } finally {
    db.close();
  }
}

function candidateFailure(
  borg: Borg,
  plan: GoalSpeakerOwnerRepairPlan,
  candidate: GoalSpeakerOwnerRepairCandidate,
): { id: string; message: string } | null {
  if (borg.entities.getSelf()?.id !== plan.selfEntityId) {
    return { id: candidate.id, message: "self entity changed after planning" };
  }

  const current = borg.self.goals.get(candidate.id);

  if (current === null) {
    return { id: candidate.id, message: "goal disappeared after planning" };
  }
  if (current.record_version !== candidate.recordVersion) {
    return { id: candidate.id, message: "goal changed after planning" };
  }
  if (current.owner_entity_id !== candidate.currentOwnerEntityId) {
    return { id: candidate.id, message: "goal owner changed after planning" };
  }
  return null;
}

function applyGoalSpeakerOwnerRepairPlan(
  borg: Borg,
  plan: GoalSpeakerOwnerRepairPlan,
): GoalSpeakerOwnerRepairReport {
  const report: GoalSpeakerOwnerRepairReport = {
    dryRun: false,
    plan,
    applied: [],
    failures: [],
  };

  for (const candidate of plan.candidates) {
    const failure = candidateFailure(borg, plan, candidate);
    if (failure !== null) {
      report.failures.push(failure);
      return report;
    }
  }

  for (const candidate of plan.candidates) {
    try {
      const result = borg.identity.updateGoal(
        candidate.id,
        goalPatchSchema.parse(candidate.patch),
        { kind: "manual" },
        { throughReview: true, reason: GOAL_SPEAKER_OWNER_REPAIR_REASON },
      );

      if (result.status !== "applied") {
        report.failures.push({
          id: candidate.id,
          message: "identity service returned requires_review",
        });
        break;
      }
      if (result.record.owner_entity_id !== null) {
        report.failures.push({
          id: candidate.id,
          message: "identity service reported applied without clearing owner_entity_id",
        });
        break;
      }
      report.applied.push(candidate);
    } catch (error) {
      report.failures.push({
        id: candidate.id,
        message: error instanceof Error ? error.message : String(error),
      });
      break;
    }
  }

  return report;
}

export function formatGoalSpeakerOwnerRepairReport(report: GoalSpeakerOwnerRepairReport): string {
  const lines = [`mode=${report.dryRun ? "dry-run" : "apply"}`];

  for (const candidate of report.plan.candidates) {
    lines.push(
      `selected id=${candidate.id} owner_entity_id=${candidate.currentOwnerEntityId} creation_event_id=${candidate.creationEventId ?? "none"}`,
    );
  }

  for (const skip of report.plan.skipped) {
    lines.push(
      `skipped id=${skip.id} reason=${JSON.stringify(skip.reason)} current_owner_entity_id=${skip.currentOwnerEntityId ?? "NULL"} creation_owner_entity_id=${skip.creationOwnerRecorded ? (skip.creationOwnerEntityId ?? "NULL") : "not-recorded"} later_owner_change_event_id=${skip.laterOwnerChangeEventId ?? "none"}`,
    );
  }

  lines.push(
    [
      `total=${report.plan.counts.total}`,
      `selected=${report.plan.counts.selected}`,
      `creation_event_missing=${report.plan.counts.creationEventMissing}`,
      `other_creation_path=${report.plan.counts.otherCreationPath}`,
      `creation_owner_missing=${report.plan.counts.creationOwnerMissing}`,
      `later_owner_change=${report.plan.counts.laterOwnerChange}`,
      `current_owner_mismatch=${report.plan.counts.currentOwnerMismatch}`,
      `promotion_owner_null=${report.plan.counts.promotionOwnerNull}`,
      `promotion_owner_self=${report.plan.counts.promotionOwnerSelf}`,
    ].join(" "),
  );

  if (!report.dryRun) {
    for (const candidate of report.applied) {
      lines.push(`applied id=${candidate.id} owner_entity_id=NULL`);
    }
    lines.push(`applied_total=${report.applied.length}`);
  }

  for (const failure of report.failures) {
    lines.push(`failure id=${failure.id} message=${JSON.stringify(failure.message)}`);
  }
  lines.push(`failures=${report.failures.length}`);
  return `${lines.join("\n")}\n`;
}

export function parseGoalSpeakerOwnerRepairCliArgs(
  argv: readonly string[],
): GoalSpeakerOwnerRepairCliArgs {
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

  return {
    help: false,
    apply,
    ...repairInputSchema.parse({ dataDir }),
  };
}

function usage(): string {
  return [
    "Usage: pnpm tsx scripts/repair-goal-speaker-owner.ts --data-dir <bank-dir> [--apply]",
    "",
    "Dry-run is the default. Stop every Borg writer and take a verified backup before --apply.",
  ].join("\n");
}

export async function main(
  argv: readonly string[] = process.argv.slice(2),
  options: GoalSpeakerOwnerRepairMainOptions = {},
): Promise<0 | 1> {
  const args = parseGoalSpeakerOwnerRepairCliArgs(argv);
  const stdout = options.stdout ?? process.stdout;
  const stderr = options.stderr ?? process.stderr;

  if (args.help) {
    stdout.write(`${usage()}\n`);
    return 0;
  }

  stderr.write(
    "WARNING: this maintenance requires a verified backup and exclusive single-writer access.\n",
  );
  const plan = planGoalSpeakerOwnerRepair({ dataDir: args.dataDir });
  let report: GoalSpeakerOwnerRepairReport = {
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
          report = applyGoalSpeakerOwnerRepairPlan(borg, plan);
        },
      );
    } catch (error) {
      report.failures.push({
        id: "repair",
        message: error instanceof Error ? error.message : String(error),
      });
    }
  }

  stdout.write(formatGoalSpeakerOwnerRepairReport(report));
  if (report.failures.length > 0) {
    stderr.write("ERROR: goal speaker-owner repair failed; inspect the report.\n");
    return 1;
  }
  return 0;
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
