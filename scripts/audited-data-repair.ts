/** Offline repair infrastructure. Stop every writer before planning or applying. */
import {
  existsSync,
  mkdirSync,
  readFileSync,
  realpathSync,
  renameSync,
  readdirSync,
} from "node:fs";
import { join, resolve, sep } from "node:path";
import { pathToFileURL } from "node:url";
import { parseArgs, type ParseArgsConfig } from "node:util";
import { createMigrations } from "../src/borg/storage-setup.js";
import { AuditLog, type MaintenanceAuditRecordInput } from "../src/offline/audit-log.js";
import type { SqliteDatabase } from "../src/storage/sqlite/index.js";
import { FixedClock } from "../src/util/clock.js";
import { StorageError } from "../src/util/errors.js";
import { createMaintenanceRunId, type MaintenanceRunId } from "../src/util/ids.js";
import { syncDirectory, writeJsonFileAtomic } from "../src/util/atomic-write.js";
import { digest, openRepairDatabase } from "./repair-entity-graph-storage.js";

export type RepairKind =
  | "backfill-commitment-committers"
  | "repair-action-actor"
  | "apply-period-rollover";
const KINDS: RepairKind[] = [
  "backfill-commitment-committers",
  "repair-action-actor",
  "apply-period-rollover",
];
export type DataRepairPlan = {
  version: 1;
  kind: RepairKind;
  dataDir: string;
  plannedAt: number;
  runId: MaintenanceRunId | null;
  summary: Record<string, number>;
};
export type RepairCheckpoint = (point: string) => void | Promise<void>;
export type RepairDriver<P extends DataRepairPlan> = {
  preflight: (db: SqliteDatabase, plan: P) => void | Promise<void>;
  finish: (db: SqliteDatabase, plan: P, checkpoint?: RepairCheckpoint) => Promise<void>;
};

export function refuse(message: string): never {
  throw new StorageError(message, { code: "AUDITED_DATA_REPAIR_INVALID" });
}

// Check the lexical path BEFORE any filesystem lookup, then check symlink resolution.
function rejectLivePath(path: string): void {
  const parts = resolve(path).split(sep);
  for (let i = 0; i < parts.length - 2; i++) {
    if (parts[i] === "demo" && parts[i + 1] === "server" && parts[i + 2] === ".borg-data") {
      refuse("The live demo store is forbidden; supply a stopped rehearsal copy");
    }
  }
}

export function checkedPath(path: string): string {
  rejectLivePath(path);
  const result = realpathSync(resolve(path));
  rejectLivePath(result);
  return result;
}

export function assertMigrations(db: SqliteDatabase): void {
  const applied = db.listAppliedMigrations();
  const expected = createMigrations();
  if (
    expected.some((m) => !applied.some((a) => a.id === m.id && a.name === m.name)) ||
    applied.some((a) => !expected.some((m) => a.id === m.id && a.name === m.name))
  ) {
    refuse("Pending or unknown migrations; migrate and review a copy separately first");
  }
}

export function openDataRepairDatabase(dataDir: string, readOnly = true): SqliteDatabase {
  const bank = checkedPath(dataDir);
  checkedPath(join(bank, "borg.db"));
  const db = openRepairDatabase(bank, readOnly);
  try {
    assertMigrations(db);
    return db;
  } catch (error) {
    db.close();
    throw error;
  }
}

export function repairDirectory(dataDir: string, kind: RepairKind): string {
  return join(dataDir, "audited-data-repairs", kind);
}
export function repairPendingPath(dataDir: string, kind: RepairKind): string {
  return join(repairDirectory(dataDir, kind), "pending.json");
}
export function assertNoPending(dataDir: string, ownKind?: RepairKind): void {
  if (existsSync(join(dataDir, "entity-graph-repair", "pending.json")))
    refuse("Entity graph repair is pending");
  for (const kind of KINDS) {
    if (kind !== ownKind && existsSync(repairPendingPath(dataDir, kind)))
      refuse(`Repair pending: ${kind}; use --apply --resume`);
  }
}

export function newPlan(
  kind: RepairKind,
  dataDir: string,
  changed: number,
  plannedAt = Date.now(),
): DataRepairPlan {
  return {
    version: 1,
    kind,
    dataDir: checkedPath(dataDir),
    plannedAt,
    runId: changed === 0 ? null : createMaintenanceRunId(),
    summary: { changed },
  };
}

export function assertSame(actual: unknown, expected: unknown, label: string): void {
  // Object key order is immaterial; array order and every stored value are material.
  if (!sameSnapshot(actual, expected)) refuse(`Changed since planning: ${label}`);
}
export function sameSnapshot(actual: unknown, expected: unknown): boolean {
  return fingerprint(actual) === fingerprint(expected);
}
function stable(value: unknown): unknown {
  if (Array.isArray(value)) return value.map(stable);
  if (value !== null && typeof value === "object") {
    return Object.fromEntries(
      Object.entries(value)
        .sort(([a], [b]) => a.localeCompare(b))
        .map(([k, v]) => [k, stable(v)]),
    );
  }
  return value;
}
function fingerprint(value: unknown): string {
  return digest(JSON.stringify(stable(value)));
}

export function repairAudit(db: SqliteDatabase, plan: DataRepairPlan): AuditLog {
  return new AuditLog({ db, clock: new FixedClock(plan.plannedAt) });
}
export function hasStep(db: SqliteDatabase, plan: DataRepairPlan, step: string): boolean {
  return repairAudit(db, plan)
    .list({ run_id: plan.runId! })
    .some((a) => a.action === step);
}
export function auditInput(
  plan: DataRepairPlan,
  action: string,
  targets: Record<string, unknown>,
  reversal: Record<string, unknown>,
): MaintenanceAuditRecordInput {
  return { run_id: plan.runId!, process: "correction", action, targets, reversal };
}
export function completionAudit(plan: DataRepairPlan): MaintenanceAuditRecordInput {
  return auditInput(
    plan,
    `${plan.kind}:complete`,
    { summary: plan.summary },
    { kind: "run_summary", data_changes: false },
  );
}

export function formatDataRepairPlan(plan: DataRepairPlan): string {
  return JSON.stringify(
    {
      ...plan,
      completion_audit: plan.runId ? completionAudit(plan) : null,
      journal: plan.runId
        ? {
            pending: repairPendingPath(plan.dataDir, plan.kind),
            archive: join(repairDirectory(plan.dataDir, plan.kind), `${plan.runId}.json`),
          }
        : null,
    },
    null,
    2,
  );
}

type Journal<P> = { version: 1; plan: P; sha256: string };
export function loadRepairPlan<P extends DataRepairPlan>(dataDir: string, kind: RepairKind): P {
  const bank = checkedPath(dataDir);
  assertNoPending(bank, kind);
  let path = repairPendingPath(bank, kind);
  if (!existsSync(path)) {
    const dir = repairDirectory(bank, kind);
    const files = existsSync(dir) ? readdirSync(dir).filter((f) => f.endsWith(".json")) : [];
    const journals = files.map((f) => ({
      path: join(dir, f),
      value: JSON.parse(readFileSync(join(dir, f), "utf8")) as Journal<P>,
    }));
    journals.sort((a, b) => b.value.plan.plannedAt - a.value.plan.plannedAt);
    path = journals[0]?.path ?? refuse("No pending or completed repair journal");
  }
  const journal = JSON.parse(readFileSync(path, "utf8")) as Journal<P>;
  if (
    journal.version !== 1 ||
    journal.plan.version !== 1 ||
    journal.plan.dataDir !== bank ||
    journal.plan.kind !== kind ||
    fingerprint(journal.plan) !== journal.sha256
  )
    refuse("Invalid repair journal");
  return journal.plan;
}

export async function applyDataRepair<P extends DataRepairPlan>(
  plan: P,
  driver: RepairDriver<P>,
  options: { resume?: boolean; checkpoint?: RepairCheckpoint } = {},
): Promise<void> {
  assertSame(checkedPath(plan.dataDir), plan.dataDir, "bank path");
  assertNoPending(plan.dataDir, options.resume ? plan.kind : undefined);
  const preflight = openDataRepairDatabase(plan.dataDir);
  try {
    await driver.preflight(preflight, plan);
  } finally {
    preflight.close();
  }
  if (plan.runId === null) return;
  const pending = repairPendingPath(plan.dataDir, plan.kind);
  if (options.resume) assertSame(loadRepairPlan(plan.dataDir, plan.kind), plan, "journal plan");
  else {
    const root = join(plan.dataDir, "audited-data-repairs");
    mkdirSync(root, { recursive: true, mode: 0o700 });
    syncDirectory(plan.dataDir);
    mkdirSync(repairDirectory(plan.dataDir, plan.kind), { recursive: true, mode: 0o700 });
    syncDirectory(root);
    writeJsonFileAtomic(pending, { version: 1, plan, sha256: fingerprint(plan) }, { mode: 0o600 });
    await options.checkpoint?.("after_journal");
  }
  const db = openDataRepairDatabase(plan.dataDir, false);
  try {
    await driver.preflight(db, plan);
    await driver.finish(db, plan, options.checkpoint);
    await driver.preflight(db, plan);
    if (!hasStep(db, plan, `${plan.kind}:complete`))
      repairAudit(db, plan).record(completionAudit(plan));
    await options.checkpoint?.("after_completion");
  } finally {
    db.close();
  }
  if (existsSync(pending)) {
    renameSync(pending, join(repairDirectory(plan.dataDir, plan.kind), `${plan.runId}.json`));
    syncDirectory(repairDirectory(plan.dataDir, plan.kind));
  }
  await options.checkpoint?.("after_archive");
}

export function repairArgs(args: string[], extra: NonNullable<ParseArgsConfig["options"]> = {}) {
  const parsed: Record<string, string | string[] | boolean | undefined> = parseArgs({
    args,
    strict: true,
    allowPositionals: false,
    options: {
      "data-dir": { type: "string" },
      apply: { type: "boolean", default: false },
      resume: { type: "boolean", default: false },
      ...extra,
    },
  }).values;
  if (typeof parsed["data-dir"] !== "string")
    refuse("--data-dir is required; no default store is opened");
  if (parsed.resume && !parsed.apply) refuse("--resume requires --apply");
  return {
    values: parsed,
    dataDir: checkedPath(parsed["data-dir"]),
    apply: parsed.apply === true,
    resume: parsed.resume === true,
  };
}
export async function printPlan(plan: DataRepairPlan): Promise<void> {
  await new Promise<void>((resolve, reject) =>
    process.stdout.write(`${formatDataRepairPlan(plan)}\n`, (error) =>
      error ? reject(error) : resolve(),
    ),
  );
}
export function runRepairCli(url: string, main: () => Promise<void>): void {
  if (process.argv[1] && url === pathToFileURL(resolve(process.argv[1])).href) {
    void (async () => {
      try {
        await main();
      } catch (error) {
        console.error(error);
        process.exitCode = 1;
      }
    })();
  }
}
