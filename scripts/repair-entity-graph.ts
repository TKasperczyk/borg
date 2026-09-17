/** Offline only: stop every writer first. Dry-run never opens Borg or runs migrations. */
import { existsSync, readFileSync, renameSync } from "node:fs";
import { join, resolve } from "node:path";
import { pathToFileURL } from "node:url";

import { createEmbeddingClient } from "../src/borg/clients.js";
import { loadConfig } from "../src/config/index.js";
import type { EmbeddingClient } from "../src/embeddings/index.js";
import { IdentityEventRepository } from "../src/memory/identity/repository.js";
import { buildNodeEmbeddingText } from "../src/memory/semantic/extractor.js";
import { SemanticNodeRepository } from "../src/memory/semantic/repository.js";
import { semanticObservationMetadataSchema } from "../src/memory/semantic/types.js";
import { AuditLog } from "../src/offline/audit-log.js";
import { LanceDbTable } from "../src/storage/lancedb/index.js";
import { quoteSqlString } from "../src/storage/codecs.js";
import { writeJsonFileAtomic, writeFileAtomic, syncDirectory } from "../src/util/atomic-write.js";
import {
  createMaintenanceRunId,
  parseSemanticNodeId,
  type MaintenanceRunId,
} from "../src/util/ids.js";
import type { JsonValue } from "../src/util/json-value.js";
import {
  type RepairInput,
  type RepairPlan,
  type Merge,
  planEntityGraphRepair,
} from "./repair-entity-graph-plan.js";
import {
  type Row,
  fail,
  same,
  changedFields,
  openRepairDatabase,
  quoteIdentifier,
  applySqlChanges,
  digest,
  openLance,
} from "./repair-entity-graph-storage.js";

export { planEntityGraphRepair };
export type { RepairInput, RepairPlan };

type Journal = { version: 1; runId: MaintenanceRunId; plan: RepairPlan };
const JOURNAL_DIRECTORY = "entity-graph-repair";

function pendingPath(dataDir: string): string {
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

async function prepareJournal(
  plan: RepairPlan,
  embeddingClient?: EmbeddingClient,
): Promise<Journal> {
  if (existsSync(pendingPath(plan.input.dataDir)))
    fail("A repair is pending; use --resume while writers remain stopped");
  const connection = await openLance(plan.input.dataDir);
  try {
    const client =
      embeddingClient ??
      (plan.touchedNodes.length === 0
        ? undefined
        : createEmbeddingClient(loadConfig({ dataDir: plan.input.dataDir })));
    for (const [name, changes] of grouped(plan.lance)) {
      if (connection === null) fail("Lance directory disappeared");
      const table = new LanceDbTable(await connection.openTable(name));
      try {
        for (const change of changes.filter((item) => item.reembed)) {
          const [row] = await table.list({ where: `id = ${quoteSqlString(change.id)}`, limit: 1 });
          if (row === undefined) fail(`Missing Lance row ${change.id}`);
          const { embedding, ...metadata } = row;
          if (!same(metadata, change.before))
            fail(`Lance row changed since planning: ${change.id}`);
          change.before.embedding = Array.from(embedding as Iterable<number>);
          const vector = await client!.embed(
            buildNodeEmbeddingText({
              label: String(change.after.label),
              description: String(change.after.description),
              aliases: JSON.parse(String(change.after.aliases)) as string[],
              observationMetadata: semanticObservationMetadataSchema
                .nullable()
                .parse(JSON.parse(String(change.after.observation_metadata ?? "null"))),
            }),
          );
          if (
            vector.length !== (change.before.embedding as number[]).length ||
            !vector.every(Number.isFinite)
          )
            fail(`Embedding shape changed for ${change.id}`);
          change.after.embedding = Array.from(vector);
        }
      } finally {
        table.close();
      }
    }
  } finally {
    connection?.close();
  }
  const journal: Journal = { version: 1, runId: createMaintenanceRunId(), plan };
  writeJsonFileAtomic(pendingPath(plan.input.dataDir), journal, { space: 0, mode: 0o600 });
  return journal;
}

async function finishJournal(journal: Journal): Promise<void> {
  const { plan, runId } = journal;
  const db = openRepairDatabase(plan.input.dataDir, false);
  const connection = await openLance(plan.input.dataDir);
  const audit = new AuditLog({ db });
  try {
    const committed = audit
      .list({ run_id: runId })
      .some((row) => row.action === "entity_graph_sql_committed");
    if (!committed) {
      db.transaction(() => {
        db.pragma("defer_foreign_keys = ON");
        for (const change of plan.sql) {
          const current = db
            .prepare(`SELECT * FROM ${quoteIdentifier(change.table)} WHERE rowid = ?`)
            .get(change.rowid);
          if (!same(current, change.before))
            fail(`SQL row changed since planning: ${change.table}/${change.rowid}`);
        }
        applySqlChanges(db, plan.sql);
        for (const [table, changes] of grouped(plan.sql)) {
          audit.record({
            run_id: runId,
            process: "correction",
            action: "entity_graph_sql_rows",
            targets: {
              table,
              rows: changes.map((row) => ({ rowid: row.rowid, after: row.after })),
            },
            reversal: {
              table,
              rows: changes.map((row) => ({ rowid: row.rowid, before: row.before })),
            },
          });
        }
        const identity = new IdentityEventRepository({ db });
        for (const gap of plan.goals) {
          const event = identity.record({
            record_type: "goal",
            record_id: gap.id,
            action: "repair_missing_update",
            old_value: gap.before as JsonValue,
            new_value: gap.after as JsonValue,
            reason: gap.reason,
            ts: gap.ts,
            provenance: { kind: "manual" },
          });
          audit.record({
            run_id: runId,
            process: "correction",
            action: "entity_graph_goal_audit",
            targets: { goal_id: gap.id, version: gap.version, identity_event_id: event.id },
            reversal: { delete_identity_event_id: event.id },
          });
        }
        // Same UPSERT/generation protocol as SemanticNodeRepository.enqueueVectorSyncRow.
        for (const nodeId of plan.touchedNodes) {
          const before =
            db
              .prepare("SELECT * FROM semantic_node_vector_sync_outbox WHERE node_id = ?")
              .get(nodeId) ?? null;
          db.prepare(
            `INSERT INTO semantic_node_vector_sync_outbox
            (node_id, reason, created_at, generation, attempts, last_attempt_at, last_error)
            VALUES (?, 'entity_graph_repair', ?, 1, 0, NULL, NULL)
            ON CONFLICT(node_id) DO UPDATE SET reason = excluded.reason, created_at = excluded.created_at,
              generation = semantic_node_vector_sync_outbox.generation + 1, attempts = 0,
              last_attempt_at = NULL, last_error = NULL`,
          ).run(nodeId, plan.plannedAt);
          audit.record({
            run_id: runId,
            process: "correction",
            action: "entity_graph_outbox",
            targets: { node_id: nodeId },
            reversal: { before },
          });
        }
        for (const [table, changes] of grouped(plan.lance))
          audit.record({
            run_id: runId,
            process: "correction",
            action: "entity_graph_lance_rows",
            targets: {
              table,
              rows: changes.map((row) => ({
                id: row.id,
                after: changedFields(row.before, row.after),
              })),
            },
            reversal: { table, rows: changes.map((row) => ({ id: row.id, before: row.before })) },
          });
        for (const file of plan.files)
          audit.record({
            run_id: runId,
            process: "correction",
            action: "entity_graph_file",
            targets: { path: file.path, hash: file.afterHash },
            reversal: { path: file.path, hash: file.hash, lines: file.lines },
          });
        audit.record({
          run_id: runId,
          process: "correction",
          action: "entity_graph_sql_committed",
          targets: { input: plan.input, journal: pendingPath(plan.input.dataDir) },
          reversal: { restore_run_id: runId },
        });
        const violations = db.prepare("PRAGMA foreign_key_check").all();
        if (violations.length > 0) fail(`Foreign key violations: ${JSON.stringify(violations)}`);
      }).immediate();
    }
    // Each external write is idempotent and checked against before/after. The
    // durable journal survives a kill between SQLite, file, and Lance commits.
    for (const file of plan.files) {
      const path = join(plan.input.dataDir, file.path);
      const raw = readFileSync(path, "utf8");
      if (digest(raw) === file.afterHash) continue;
      if (digest(raw) !== file.hash) fail(`File changed since planning: ${file.path}`);
      const separator = file.path.endsWith(".jsonl") ? "\n" : "";
      const lines = separator === "" ? [raw] : raw.split(separator);
      for (const line of file.lines) lines[line.index] = line.after;
      const after = lines.join(separator);
      if (digest(after) !== file.afterHash) fail(`Invalid file journal: ${file.path}`);
      writeFileAtomic(path, after, { mode: 0o600 });
    }
    for (const [name, changes] of grouped(plan.lance)) {
      if (connection === null) fail("Lance directory disappeared");
      const table = new LanceDbTable(await connection.openTable(name));
      try {
        // One mergeInsert per table, not one new Lance version per row.
        const current = new Map((await table.list()).map((row) => [String(row.id), row]));
        const updates: Row[] = [];
        for (const change of changes) {
          const row = current.get(change.id) ?? fail(`Missing Lance row ${change.id}`);
          const comparable = Object.fromEntries(
            Object.keys(change.before).map((key) => [
              key,
              key === "embedding" ? Array.from(row[key] as Iterable<number>) : row[key],
            ]),
          );
          if (same(comparable, change.after)) continue;
          if (!same(comparable, change.before))
            fail(`Lance row changed since planning: ${name}/${change.id}`);
          updates.push({ ...row, ...change.after });
        }
        await table.upsert(updates, { on: "id" });
        if (name === "semantic_nodes") {
          const repository = new SemanticNodeRepository({ db, table });
          const result = await repository.syncPendingVectorUpdates({
            nodeIds: plan.touchedNodes.map(parseSemanticNodeId),
            limit: Math.max(1, plan.touchedNodes.length),
          });
          if (result.failed.length > 0 || result.pending > 0)
            fail(`Vector outbox not drained: ${JSON.stringify(result)}`);
        }
      } finally {
        table.close();
      }
    }
    if (!audit.list({ run_id: runId }).some((row) => row.action === "entity_graph_complete"))
      audit.record({
        run_id: runId,
        process: "correction",
        action: "entity_graph_complete",
        targets: { input: plan.input },
        reversal: { restore_run_id: runId },
      });
    const completedPath = join(plan.input.dataDir, JOURNAL_DIRECTORY, `${runId}.json`);
    renameSync(pendingPath(plan.input.dataDir), completedPath);
    syncDirectory(join(plan.input.dataDir, JOURNAL_DIRECTORY));
  } finally {
    connection?.close();
    db.close();
  }
}

export async function applyEntityGraphRepair(
  plan: RepairPlan,
  embeddingClient?: EmbeddingClient,
): Promise<void> {
  const journal = await prepareJournal(plan, embeddingClient);
  await finishJournal(journal);
}

export async function resumeEntityGraphRepair(dataDir: string): Promise<void> {
  const journal = JSON.parse(readFileSync(pendingPath(dataDir), "utf8")) as Journal;
  if (journal.version !== 1 || journal.plan.input.dataDir !== resolve(dataDir))
    fail("Journal version or data directory mismatch");
  await finishJournal(journal);
}

export function formatRepairPlan(plan: RepairPlan): string {
  const lines: string[] = [];
  lines.push(
    JSON.stringify({
      input: plan.input,
      sqlite_tables_scanned: Object.keys(plan.inventory.sqlite).length,
      lance_tables_scanned: Object.keys(plan.inventory.lance).length,
    }),
  );
  for (const [table, columns] of Object.entries(plan.inventory.sqlite)) {
    const rows = plan.sql.filter((row) => row.table === table);
    const fields: Record<string, number> = {};
    for (const row of rows)
      for (const key of Object.keys(
        row.after === null ? row.before : changedFields(row.before, row.after),
      ))
        fields[key] = (fields[key] ?? 0) + 1;
    lines.push(
      JSON.stringify({
        store: "sqlite",
        table,
        scanned_columns: columns,
        updates: rows.filter((row) => row.after !== null).length,
        deletes: rows.filter((row) => row.after === null).length,
        fields,
      }),
    );
    if (["entities", "semantic_nodes", "semantic_edges"].includes(table)) {
      for (const row of rows)
        lines.push(
          JSON.stringify({ table, rowid: row.rowid, before: row.before, after: row.after }),
        );
    }
  }
  for (const [table, columns] of Object.entries(plan.inventory.lance))
    lines.push(
      JSON.stringify({
        store: "lance",
        table,
        scanned_columns: columns,
        updates: plan.lance.filter((row) => row.table === table).length,
      }),
    );
  for (const file of plan.files)
    lines.push(
      JSON.stringify({
        store: "file",
        path: file.path,
        changed_records: file.lines.length,
        before_sha256: file.hash,
        after_sha256: file.afterHash,
      }),
    );
  for (const gap of plan.goals) lines.push(JSON.stringify({ goal_audit_backfill: gap }));
  lines.push(
    JSON.stringify({
      additional_writes: {
        identity_events_inserts: plan.goals.length,
        semantic_node_vector_sync_outbox_enqueues_then_drains: plan.touchedNodes.length,
        maintenance_audit_inserts:
          grouped(plan.sql).size +
          plan.goals.length +
          plan.touchedNodes.length +
          grouped(plan.lance).size +
          plan.files.length +
          2,
        journal: pendingPath(plan.input.dataDir),
      },
    }),
  );
  for (const report of plan.reports)
    lines.push(
      JSON.stringify({ report: report.category, count: report.rows.length, rows: report.rows }),
    );
  lines.push(
    JSON.stringify({
      summary: {
        sqlite_changes: plan.sql.length,
        lance_changes: plan.lance.length,
        changed_files: plan.files.length,
        semantic_reembeddings: plan.touchedNodes.length,
        goal_audit_backfills: plan.goals.length,
      },
    }),
  );
  return `${lines.join("\n")}\n`;
}

export function parseRepairArgs(
  argv: readonly string[],
): RepairInput & { apply: boolean; resume: boolean; help: boolean } {
  const result = {
    dataDir: "",
    merges: [] as Merge[],
    nodeMerges: [] as Merge[],
    apply: false,
    resume: false,
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
  if (!result.resume && result.merges.length === 0) fail("At least one --merge is required");
  if (result.resume && (!result.apply || result.merges.length > 0 || result.nodeMerges.length > 0))
    fail("Use --data-dir <dir> --apply --resume without merge arguments");
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
  const plan = await planEntityGraphRepair(input);
  process.stdout.write(`mode=${args.apply ? "apply" : "dry-run"}\n${formatRepairPlan(plan)}`);
  if (args.apply) {
    await applyEntityGraphRepair(plan);
    process.stdout.write("Repair complete; semantic embeddings refreshed and outbox drained.\n");
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
