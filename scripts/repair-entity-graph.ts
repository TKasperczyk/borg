import { quoteSqlString } from "../src/storage/codecs.js";
import { planEntityGraphReversal } from "./repair-entity-graph-reversal.js";
export { planEntityGraphReversal };
/** Offline only. Stop all writers. Source streams, working files and their index are immutable. */
import { existsSync, readFileSync, renameSync, readdirSync, mkdirSync } from "node:fs";
import { join, resolve } from "node:path";
import { pathToFileURL } from "node:url";
import { createMigrations } from "../src/borg/storage-setup.js";
import { SemanticNodeRepository } from "../src/memory/semantic/repository.js";
import { LanceDbTable } from "../src/storage/lancedb/index.js";
import { runMigrations, type SqliteDatabase } from "../src/storage/sqlite/index.js";
import { writeJsonFileAtomic, syncDirectory } from "../src/util/atomic-write.js";
import { parseSemanticNodeId, type MaintenanceRunId } from "../src/util/ids.js";
import {
  type RepairInput,
  type RepairPlan,
  type Merge,
  planEntityGraphRepair,
  protectedFiles,
} from "./repair-entity-graph-plan.js";
import {
  type Row,
  type SqlChange,
  fail,
  same,
  changedFields,
  openRepairDatabase,
  quoteIdentifier,
  applySqlChanges,
  openLance,
  sqlInventory,
  digest,
  preflightLance,
  lanceRowMatches,
} from "./repair-entity-graph-storage.js";
import { validateSqlChanges } from "./repair-entity-graph-validation.js";

export { planEntityGraphRepair };
export type { RepairInput, RepairPlan };
export type RepairCheckpoint =
  | "after_journal"
  | "after_sqlite"
  | `after_lance:${string}`
  | `during_outbox:${string}`
  | "before_completion"
  | "after_completion"
  | "before_archive"
  | "after_archive";
export type RepairOptions = { checkpoint?: (point: RepairCheckpoint) => void | Promise<void> };
type Journal = { version: 2; plan: RepairPlan };
const JOURNAL_DIRECTORY = "entity-graph-repair";
export function pendingPath(dataDir: string): string {
  return join(dataDir, JOURNAL_DIRECTORY, "pending.json");
}

function grouped<T extends { table: string }>(items: T[]): Map<string, T[]> {
  const result = new Map<string, T[]>();
  for (const item of items) {
    const group = result.get(item.table) ?? [];
    group.push(item);
    result.set(item.table, group);
  }
  return result;
}

function assertRows(db: SqliteDatabase, changes: SqlChange[], side: "before" | "after"): void {
  for (const change of changes) {
    const current =
      db
        .prepare(`SELECT * FROM ${quoteIdentifier(change.table)} WHERE rowid = ?`)
        .get(change.rowid) ?? null;
    if (!same(current, change[side]))
      fail(`SQL row changed since planning (${side}): ${change.table}/${change.rowid}`);
  }
}

function assertProtected(plan: RepairPlan): void {
  if (!same(protectedFiles(plan.input.dataDir), plan.protectedFiles))
    fail("Protected stream/working files changed since planning");
  const db = openRepairDatabase(plan.input.dataDir);
  try {
    if (
      digest(JSON.stringify(sqlInventory(db, ["stream_entry_index"]).get("stream_entry_index"))) !==
      plan.protectedIndexHash
    )
      fail("Protected stream entry index changed since planning");
  } finally {
    db.close();
  }
}

async function finishJournal(journal: Journal, options: RepairOptions = {}): Promise<void> {
  const { plan } = journal;
  const runId = plan.runId ?? fail("Empty repair cannot have a journal");
  assertProtected(plan);
  const db = openRepairDatabase(plan.input.dataDir, false);
  const connection = await openLance(plan.input.dataDir);
  try {
    const committed =
      db
        .prepare(
          "SELECT 1 FROM maintenance_audit WHERE run_id = ? AND action = 'entity_graph_sql_committed'",
        )
        .get(runId) !== undefined;
    await preflightLance(connection, plan.lance, committed);
    if (!committed) {
      validateSqlChanges([...plan.sql, ...plan.additionalSql, ...plan.completionSql]);
      db.transaction(() => {
        db.pragma("defer_foreign_keys = ON");
        const migrations = createMigrations().filter((item) =>
          plan.migrations.some((pending) => pending.id === item.id && pending.name === item.name),
        );
        runMigrations(db, migrations, plan.plannedAt);
        assertRows(db, plan.migrationSql, "after");
        assertRows(db, plan.sql, "before");
        applySqlChanges(db, plan.sql);
        assertRows(db, plan.sql, "after");
        assertRows(db, plan.additionalSql, "before");
        applySqlChanges(db, plan.additionalSql);
        assertRows(db, plan.additionalSql, "after");
        const violations = db.prepare("PRAGMA foreign_key_check").all();
        if (violations.length) fail(`Foreign key violations: ${JSON.stringify(violations)}`);
      }).immediate();
    }
    await options.checkpoint?.("after_sqlite");
    for (const [name, changes] of grouped(plan.lance)) {
      if (connection === null) fail("Lance directory disappeared");
      const table = new LanceDbTable(await connection.openTable(name));
      try {
        const current = new Map(
          (
            await table.list({
              where: `id IN (${changes.map((change) => quoteSqlString(change.id)).join(",")})`,
            })
          ).map((row) => [String(row.id), row]),
        );
        const updates: Row[] = [];
        for (const change of changes) {
          const row = current.get(change.id) ?? fail(`Missing Lance row ${change.id}`);
          if (lanceRowMatches(row, change, "after")) continue;
          if (!lanceRowMatches(row, change, "before"))
            fail(`Lance row changed since planning: ${name}/${change.id}`);
          updates.push({ ...row, ...change.after });
        }
        if (updates.length) await table.upsert(updates, { on: "id" });
      } finally {
        table.close();
      }
      await options.checkpoint?.(`after_lance:${name}`);
    }
    if (plan.touchedNodes.length) {
      if (connection === null) fail("Lance directory disappeared");
      const table = new LanceDbTable(await connection.openTable("semantic_nodes"));
      try {
        const repository = new SemanticNodeRepository({ db, table });
        for (const id of plan.touchedNodes) {
          const result = await repository.syncPendingVectorUpdates({
            nodeIds: [parseSemanticNodeId(id)],
            limit: 1,
          });
          if (result.failed.length || result.pending)
            fail(`Vector outbox not drained: ${JSON.stringify(result)}`);
          await options.checkpoint?.(`during_outbox:${id}`);
        }
      } finally {
        table.close();
      }
    }
    assertProtected(plan);
    await options.checkpoint?.("before_completion");
    if (
      db
        .prepare(
          "SELECT 1 FROM maintenance_audit WHERE run_id = ? AND action = 'entity_graph_complete'",
        )
        .get(runId) === undefined
    )
      db.transaction(() => {
        assertRows(db, plan.completionSql, "before");
        applySqlChanges(db, plan.completionSql);
        assertRows(db, plan.completionSql, "after");
      }).immediate();
    await options.checkpoint?.("after_completion");
    await options.checkpoint?.("before_archive");
    renameSync(
      pendingPath(plan.input.dataDir),
      join(plan.input.dataDir, JOURNAL_DIRECTORY, `${runId}.json`),
    );
    syncDirectory(join(plan.input.dataDir, JOURNAL_DIRECTORY));
    await options.checkpoint?.("after_archive");
  } finally {
    connection?.close();
    db.close();
  }
}

export async function applyEntityGraphRepair(
  plan: RepairPlan,
  options: RepairOptions = {},
): Promise<void> {
  // In particular: don't mkdir, create a journal, open a writable DB, or insert audit markers.
  if (plan.runId === null) return;
  if (existsSync(pendingPath(plan.input.dataDir)))
    fail("A repair is pending; use --resume while writers remain stopped");
  assertProtected(plan);
  const connection = await openLance(plan.input.dataDir);
  try {
    await preflightLance(connection, plan.lance);
  } finally {
    connection?.close();
  }
  // The journal's directory entry must be durable before committing SQLite.
  mkdirSync(join(plan.input.dataDir, JOURNAL_DIRECTORY), { recursive: true });
  syncDirectory(plan.input.dataDir);
  const journal: Journal = { version: 2, plan };
  writeJsonFileAtomic(pendingPath(plan.input.dataDir), journal, { space: 0, mode: 0o600 });
  await options.checkpoint?.("after_journal");
  await finishJournal(journal, options);
}

export async function resumeEntityGraphRepair(
  dataDir: string,
  options: RepairOptions = {},
): Promise<void> {
  const path = pendingPath(dataDir);
  if (!existsSync(path)) {
    // A crash just after atomic journal archival is already complete. Verify its
    // durable completion marker and return without creating another journal.
    const directory = join(dataDir, JOURNAL_DIRECTORY);
    const archived = existsSync(directory)
      ? readdirSync(directory).filter((file) => file.endsWith(".json"))
      : [];
    const db = openRepairDatabase(dataDir);
    try {
      if (
        archived.some((file) => {
          const journal = JSON.parse(readFileSync(join(directory, file), "utf8")) as Journal;
          return (
            journal.version === 2 &&
            journal.plan.runId !== null &&
            db
              .prepare(
                "SELECT 1 FROM maintenance_audit WHERE run_id = ? AND action = 'entity_graph_complete'",
              )
              .get(journal.plan.runId) !== undefined
          );
        })
      )
        return;
    } finally {
      db.close();
    }
    fail("No pending or completed repair journal");
  }
  const journal = JSON.parse(readFileSync(path, "utf8")) as Journal;
  if (journal.version !== 2 || journal.plan.input.dataDir !== resolve(dataDir))
    fail("Journal version or data directory mismatch");
  await finishJournal(journal, options);
}

/** Complete JSONL, including exact audit/outbox/event writes and both vector values when changed. */
export function formatRepairPlan(plan: RepairPlan): string {
  const lines: unknown[] = [
    {
      input: plan.input,
      run_id: plan.runId,
      embedding: plan.embedding,
      migrations: plan.migrations,
      sqlite_tables_scanned: Object.keys(plan.inventory.sqlite).length,
      lance_tables_scanned: Object.keys(plan.inventory.lance).length,
    },
  ];
  for (const [table, columns] of Object.entries(plan.inventory.sqlite)) {
    const rows = plan.sql.filter((row) => row.table === table);
    lines.push({
      store: "sqlite",
      table,
      scanned_columns: columns,
      updates: rows.filter((row) => row.after !== null && row.before !== null).length,
      deletes: rows.filter((row) => row.after === null).length,
      inserts: rows.filter((row) => row.before === null).length,
    });
    for (const row of rows)
      lines.push({
        store: "sqlite",
        ...row,
        changed_values:
          row.before !== null && row.after !== null
            ? changedFields(row.before, row.after)
            : row.after,
      });
  }
  for (const [table, columns] of Object.entries(plan.inventory.lance)) {
    lines.push({
      store: "lance",
      table,
      scanned_columns: columns,
      updates: plan.lance.filter((row) => row.table === table).length,
    });
    for (const row of plan.lance.filter((row) => row.table === table))
      lines.push({
        store: "lance",
        ...row,
        embedding_unchanged: same(row.before.embedding, row.after.embedding),
      });
  }
  for (const change of plan.schemaChanges) lines.push({ store: "sqlite_schema", ...change });
  for (const row of plan.migrationSql) lines.push({ store: "sqlite", phase: "migration", ...row });
  for (const row of plan.drainSql)
    lines.push({ store: "sqlite", phase: "outbox_drain_after_lance_commit", ...row });
  for (const row of plan.additionalSql)
    lines.push({ store: "sqlite", phase: "additional_transactional_write", ...row });
  for (const row of plan.completionSql)
    lines.push({ store: "sqlite", phase: "completion_after_external_commits", ...row });
  for (const file of plan.protectedFiles)
    lines.push({
      store: "file",
      path: file.path,
      changed_fields: [],
      before_sha256: file.sha256,
      after_sha256: file.sha256,
    });
  lines.push({
    additional_writes: {
      identity_events_inserts: plan.goals.length,
      semantic_node_vector_sync_outbox_enqueues_then_drains: plan.touchedNodes,
      outbox_mirror_upserts: plan.touchedNodes.map((id) => ({
        table: "semantic_nodes",
        id,
        after: plan.lance.find((row) => row.table === "semantic_nodes" && row.id === id)?.after,
      })),
      maintenance_audit_inserts: [...plan.additionalSql, ...plan.completionSql].filter(
        (row) => row.table === "maintenance_audit" && row.before === null,
      ).length,
      journal:
        plan.runId === null
          ? null
          : {
              create: pendingPath(plan.input.dataDir),
              archive: join(plan.input.dataDir, JOURNAL_DIRECTORY, `${plan.runId}.json`),
              format: "version 2; contains this complete plan",
            },
    },
  });
  for (const report of plan.reports)
    lines.push({ report: report.category, count: report.rows.length, rows: report.rows });
  lines.push({
    summary: {
      sqlite_changes: plan.sql.length,
      lance_changes: plan.lance.length,
      changed_files: 0,
      stream_changes: 0,
      working_changes: 0,
      stream_entry_index_changes: 0,
      semantic_reembeddings: plan.touchedNodes.length,
      goal_audit_backfills: plan.goals.length,
    },
  });
  return lines.map((line) => JSON.stringify(line)).join("\n") + "\n";
}
export function parseRepairArgs(
  argv: readonly string[],
): RepairInput & { apply: boolean; resume: boolean; reverse: string | null; help: boolean } {
  const result = {
    dataDir: "",
    merges: [] as Merge[],
    nodeMerges: [] as Merge[],
    apply: false,
    resume: false,
    reverse: null as string | null,
    help: false,
  };
  for (let i = 0; i < argv.length; i++) {
    const flag = argv[i]!;
    if (flag === "--help") {
      result.help = true;
      return result;
    }
    if (flag === "--apply") {
      result.apply = true;
      continue;
    }
    if (flag === "--resume") {
      result.resume = true;
      continue;
    }
    const value = argv[++i];
    if (value === undefined || value.startsWith("--")) fail(`${flag} requires a value`);
    if (flag === "--reverse") {
      result.reverse = value;
      continue;
    }
    if (flag === "--data-dir") {
      result.dataDir = resolve(value);
      continue;
    }
    if (flag === "--rename-into") {
      const merge = result.merges.at(-1) ?? fail("--rename-into must follow its --merge");
      if (merge.renameInto !== undefined) fail("Repeated --rename-into");
      merge.renameInto = value;
      continue;
    }
    if (flag === "--merge" || flag === "--merge-node") {
      const parts = value.split("=");
      if (parts.length !== 2 || !parts[0] || !parts[1]) fail(`${flag} requires from=into`);
      (flag === "--merge" ? result.merges : result.nodeMerges).push({
        from: parts[0],
        into: parts[1],
      });
      continue;
    }
    fail(`Unknown argument ${flag}`);
  }
  if (!result.dataDir) fail("--data-dir is required");
  if (!result.resume && result.reverse === null && result.merges.length === 0)
    fail("At least one --merge is required");
  if (result.resume && (!result.apply || result.merges.length > 0 || result.nodeMerges.length > 0))
    fail("Use --data-dir <dir> --apply --resume without merge arguments");
  if (
    result.reverse !== null &&
    (result.resume || result.merges.length || result.nodeMerges.length)
  )
    fail("--reverse takes a run ID without merge/resume flags");
  return result;
}

export async function main(argv: readonly string[] = process.argv.slice(2)): Promise<void> {
  const args = parseRepairArgs(argv);
  if (args.help) {
    process.stdout.write(
      "Usage: pnpm tsx scripts/repair-entity-graph.ts --data-dir <dir> --merge <from>=<into> [--rename-into <merged-canonical-name>] [--merge-node <from>=<into>] [--apply]\nDry-run by default. Stop all writers before applying. After interruption: --data-dir <dir> --apply --resume.\n",
    );
    return;
  }
  if (args.resume) {
    await resumeEntityGraphRepair(args.dataDir);
    process.stdout.write("Repair resumed and completed.\n");
    return;
  }
  if (existsSync(pendingPath(args.dataDir)))
    fail("A repair is pending; use --apply --resume before running the service");
  const input: RepairInput = {
    dataDir: args.dataDir,
    merges: args.merges,
    nodeMerges: args.nodeMerges,
  };
  const plan =
    args.reverse === null
      ? await planEntityGraphRepair(input)
      : await planEntityGraphReversal(args.dataDir, args.reverse);
  // Wait for the complete reviewable plan to reach stdout before any store mutation.
  await new Promise<void>((done, reject) =>
    process.stdout.write(
      JSON.stringify({ mode: args.apply ? "apply" : "dry-run" }) + "\n" + formatRepairPlan(plan),
      (error) => (error ? reject(error) : done()),
    ),
  );
  if (args.apply) {
    await applyEntityGraphRepair(plan);
    process.stdout.write(JSON.stringify({ complete: true, run_id: plan.runId }) + "\n");
  }
}

if (process.argv[1] !== undefined && import.meta.url === pathToFileURL(process.argv[1]).href) {
  try {
    await main();
  } catch (error) {
    process.stderr.write(`ERROR: ${error instanceof Error ? error.message : String(error)}\n`);
    process.exitCode = 1;
  }
}
