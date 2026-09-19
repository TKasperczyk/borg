import { existsSync } from "node:fs";
import { join } from "node:path";
import { loadConfig } from "../src/config/index.js";
import { AuditLog } from "../src/offline/audit-log.js";
import { createMaintenanceRunId, parseMaintenanceRunId } from "../src/util/ids.js";
import { ManualClock } from "../src/util/clock.js";
import { type RepairPlan, type RepairInput, protectedFiles } from "./repair-entity-graph-plan.js";
import {
  type Row,
  type SqlChange,
  type LanceChange,
  openRepairDatabase,
  fail,
  same,
  cloneDatabase,
  sqlInventory,
  applySqlChanges,
  quoteIdentifier,
  openLance,
  digest,
} from "./repair-entity-graph-storage.js";
import { validateSqlChanges, validateLanceChanges } from "./repair-entity-graph-validation.js";

/** Read reversal payloads from maintenance_audit, independently of the forward journal. */
export async function planEntityGraphReversal(
  dataDir: string,
  originalRunId: string,
): Promise<RepairPlan> {
  if (existsSync(join(dataDir, "entity-graph-repair", "pending.json")))
    fail("Resume the pending repair before reversal");
  const source = openRepairDatabase(dataDir);
  const db = cloneDatabase(source);
  const connection = await openLance(dataDir);
  try {
    const records = new AuditLog({ db }).list({ run_id: parseMaintenanceRunId(originalRunId) });
    const commit = records.find((row) => row.action === "entity_graph_sql_committed");
    if (commit === undefined || !records.some((row) => row.action === "entity_graph_complete"))
      fail("Reversal requires a completed audited repair");
    if (records.some((row) => row.reverted_at !== null)) fail("Repair has already been reversed");
    const input = commit.targets.input as RepairInput;
    const protectedIndexHash = digest(
      JSON.stringify(sqlInventory(db, ["stream_entry_index"]).get("stream_entry_index")),
    );
    const sqliteInventory = Object.fromEntries(
      db
        .prepare(
          "SELECT name FROM sqlite_master WHERE type = 'table' AND name NOT LIKE 'sqlite_%' ORDER BY name",
        )
        .all()
        .map(({ name }) => [
          String(name),
          db
            .prepare(`PRAGMA table_info(${quoteIdentifier(String(name))})`)
            .all()
            .map((column) => String(column.name)),
        ]),
    );
    const sql: SqlChange[] = [];
    const lance: LanceChange[] = [];
    for (const record of records) {
      if (record.action === "entity_graph_sql_rows") {
        const table = String(record.targets.table);
        const afterRows = record.targets.rows as Array<{ rowid: number; after: Row | null }>;
        for (const saved of record.reversal.rows as Array<{ rowid: number; before: Row | null }>) {
          const expected = afterRows.find((row) => row.rowid === saved.rowid)?.after;
          const current =
            db
              .prepare(`SELECT * FROM ${quoteIdentifier(table)} WHERE rowid = ?`)
              .get(saved.rowid) ?? null;
          // Forward completion drains exactly these queued generations. Any new
          // queue entry is concurrent work and must not be silently overwritten.
          if (
            table === "semantic_node_vector_sync_outbox"
              ? current !== null
              : !same(current, expected)
          )
            fail(`Cannot reverse modified row ${table}/${saved.rowid}`);
          if (!same(current, saved.before))
            sql.push({
              table,
              rowid: saved.rowid,
              before: current,
              after: saved.before,
              reason: `reverse audited run ${originalRunId}`,
            });
        }
      }
      if (record.action === "entity_graph_lance_rows") {
        const table = String(record.targets.table);
        const rows = record.targets.rows as Array<{ id: string; after: Row }>;
        for (const saved of record.reversal.rows as Array<{ id: string; before: Row }>)
          lance.push({
            table,
            id: saved.id,
            before: rows.find((row) => row.id === saved.id)!.after,
            after: saved.before,
          });
      }
    }
    validateSqlChanges(sql);
    db.transaction(() => {
      db.pragma("defer_foreign_keys = ON");
      applySqlChanges(db, sql);
    }).immediate();
    if (db.prepare("PRAGMA foreign_key_check").all().length)
      fail("Reversal projection violates foreign keys");
    validateLanceChanges(
      lance,
      new Map(
        [...sqlInventory(db, [...new Set(lance.map((row) => row.table))])].map(([table, rows]) => [
          table,
          new Map(rows.map(({ row }) => [String(row.id), row])),
        ]),
      ),
    );
    // The additive redirect migration remains installed. Reversal restores its
    // nullable data; removing a deployed schema migration would break current readers.
    const now = Date.now(),
      runId = createMaintenanceRunId();
    const config = loadConfig({ dataDir });
    const audit = new AuditLog({ db, clock: new ManualClock(now) });
    const additionalSql: SqlChange[] = [];
    const completionSql: SqlChange[] = [];
    const recordAudit = (input: Parameters<AuditLog["record"]>[0], destination: SqlChange[]) => {
      const record = audit.record(input);
      destination.push({
        table: "maintenance_audit",
        rowid: record.id,
        before: null,
        after: db.prepare("SELECT * FROM maintenance_audit WHERE id = ?").get(record.id)!,
        reason: "maintenance audit with reversal payload",
      });
    };
    for (const table of new Set(sql.map((change) => change.table))) {
      const changes = sql.filter((change) => change.table === table);
      recordAudit(
        {
          run_id: runId,
          process: "correction",
          action: "entity_graph_sql_rows",
          targets: { table, rows: changes.map(({ rowid, after }) => ({ rowid, after })) },
          reversal: {
            table,
            rows: changes.map(({ rowid, before: old }) => ({ rowid, before: old })),
          },
        },
        additionalSql,
      );
    }
    for (const table of new Set(lance.map((change) => change.table))) {
      const changes = lance.filter((change) => change.table === table);
      recordAudit(
        {
          run_id: runId,
          process: "correction",
          action: "entity_graph_lance_rows",
          targets: { table, rows: changes.map(({ id, after }) => ({ id, after })) },
          reversal: { table, rows: changes.map(({ id, before: old }) => ({ id, before: old })) },
        },
        additionalSql,
      );
    }
    recordAudit(
      {
        run_id: runId,
        process: "correction",
        action: "entity_graph_sql_committed",
        targets: { input: { ...input, dataDir }, reverses: originalRunId },
        reversal: { restore_run_id: runId },
      },
      additionalSql,
    );
    for (const record of records) {
      const before = db.prepare("SELECT * FROM maintenance_audit WHERE id = ?").get(record.id)!;
      const result = db
        .prepare(
          "UPDATE maintenance_audit SET reverted_at = ?, reverted_by = ? WHERE id = ? AND reverted_at IS NULL",
        )
        .run(now, runId, record.id);
      if (result.changes !== 1) fail(`Cannot mark audit ${record.id} reversed`);
      completionSql.push({
        table: "maintenance_audit",
        rowid: record.id,
        before,
        after: db.prepare("SELECT * FROM maintenance_audit WHERE id = ?").get(record.id)!,
        reason: "mark original audit reversed",
      });
    }
    recordAudit(
      {
        run_id: runId,
        process: "correction",
        action: "entity_graph_complete",
        targets: { input: { ...input, dataDir }, reverses: originalRunId },
        reversal: { restore_run_id: runId },
      },
      completionSql,
    );
    validateSqlChanges([...additionalSql, ...completionSql]);
    return {
      input: { ...input, dataDir },
      plannedAt: now,
      runId,
      migrations: [],
      migrationSql: [],
      schemaChanges: [],
      protectedIndexHash,
      drainSql: [],
      sql,
      additionalSql,
      completionSql,
      lance,
      files: [],
      protectedFiles: protectedFiles(dataDir),
      embedding: {
        model: config.embedding.model,
        dimensions: config.embedding.dims,
        endpoint: config.embedding.baseUrl,
        injectedClient: false,
      },
      goals: [],
      touchedNodes: [],
      inventory: {
        sqlite: sqliteInventory,
        lance: Object.fromEntries(
          await Promise.all(
            ((await connection?.tableNames()) ?? []).map(async (name) => {
              const table = await connection!.openTable(name);
              try {
                return [name, (await table.schema()).fields.map((field) => field.name)];
              } finally {
                table.close();
              }
            }),
          ),
        ),
      },
      reports: [
        {
          category: "reversal",
          rows: [{ originalRunId, retained_migration: "entity_identity_redirects" }],
        },
      ],
    };
  } finally {
    connection?.close();
    db.close();
    source.close();
  }
}
