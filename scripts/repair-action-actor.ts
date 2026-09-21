import { existsSync } from "node:fs";
import { join } from "node:path";
import { ActionRepository } from "../src/memory/actions/repository.js";
import type { ActionRecord } from "../src/memory/actions/types.js";
import type { MaintenanceAuditRecordInput } from "../src/offline/audit-log.js";
import { quoteSqlString } from "../src/storage/codecs.js";
import { LanceDbTable } from "../src/storage/lancedb/index.js";
import type { SqliteDatabase } from "../src/storage/sqlite/index.js";
import { parseEntityId, type EntityId } from "../src/util/ids.js";
import { openLance, type Row } from "./repair-entity-graph-storage.js";
import {
  applyDataRepair,
  assertNoPending,
  assertSame,
  auditInput,
  checkedPath,
  hasStep,
  loadRepairPlan,
  newPlan,
  openDataRepairDatabase,
  printPlan,
  refuse,
  repairArgs,
  repairAudit,
  repairDirectory,
  repairPendingPath,
  runRepairCli,
  sameSnapshot,
  type DataRepairPlan,
  type RepairCheckpoint,
} from "./audited-data-repair.js";

const KIND = "repair-action-actor";
const JOURNAL_VERSION = 1;
type ActorPlanRow = {
  before: ActionRecord;
  after: ActionRecord;
  mirror_before: Row | null;
  mirror_after: Row | null;
  audit: MaintenanceAuditRecordInput;
};
export type ActorRepairPlan = DataRepairPlan & {
  journal_version: typeof JOURNAL_VERSION;
  from: EntityId;
  to: EntityId;
  to_entity: Row;
  rows: ActorPlanRow[];
  mirror_table_present: boolean;
};

function assertJournalVersion(plan: ActorRepairPlan): void {
  if (plan.journal_version === JOURNAL_VERSION) return;
  const dataDir = checkedPath(plan.dataDir);
  const pending = repairPendingPath(dataDir, KIND);
  const path = existsSync(pending)
    ? pending
    : join(repairDirectory(dataDir, KIND), `${plan.runId}.json`);
  refuse(
    `Incompatible actor repair journal ${path}: journal_version ${plan.journal_version ?? "missing"}; expected ${JOURNAL_VERSION}. Preserve this journal and reconcile any partial apply before moving it aside; re-plan with the current script without --resume.`,
  );
}

function entityPreflight(db: SqliteDatabase, from: EntityId, to: EntityId): Row {
  if (db.prepare("SELECT id FROM entities WHERE id = ?").get(from))
    refuse("--from has an entities row; use the entity-graph redirect repair");
  return (
    db.prepare("SELECT * FROM entities WHERE id = ?").get(to) ?? refuse("--to has no entities row")
  );
}
async function withMirror<T>(
  dataDir: string,
  callback: (table: LanceDbTable | null) => Promise<T>,
): Promise<T> {
  if (existsSync(join(dataDir, "lancedb"))) checkedPath(join(dataDir, "lancedb"));
  const connection = await openLance(dataDir);
  let table: LanceDbTable | null = null;
  try {
    if (connection && (await connection.tableNames()).includes("action_records"))
      table = new LanceDbTable(await connection.openTable("action_records"));
    return await callback(table);
  } finally {
    table?.close();
    connection?.close();
  }
}
async function mirrorRow(table: LanceDbTable | null, id: string): Promise<Row | null> {
  if (!table) return null;
  const rows = await table.list({ where: `id = ${quoteSqlString(id)}` });
  if (rows.length > 1) refuse(`Duplicate action mirror ${id}`);
  const row = rows[0];
  return row ? { ...row, embedding: Array.from(row.embedding as Iterable<number>) } : null;
}

function sameMirrorSnapshot(current: Row | null, expected: Row | null): boolean {
  if (current === null || expected === null) return current === expected;
  // Identity and actor are mandatory assertions, even for an incomplete snapshot.
  if (typeof expected.id !== "string" || typeof expected.actor !== "string") return false;
  // Only additional, unrecorded fields may differ. Every recorded value remains strict.
  return sameSnapshot(
    Object.fromEntries(Object.keys(expected).map((key) => [key, current[key]])),
    expected,
  );
}

export async function planActionActorRepair(
  input: { dataDir: string; from: string; to: string },
  now = Date.now(),
): Promise<ActorRepairPlan> {
  const dataDir = checkedPath(input.dataDir),
    from = parseEntityId(input.from),
    to = parseEntityId(input.to);
  assertNoPending(dataDir);
  const db = openDataRepairDatabase(dataDir);
  try {
    const toEntity = entityPreflight(db, from, to);
    // Omit the limit so historical/terminal actions are all included.
    const records = new ActionRepository({ db }).list({ actor: from });
    const plan: ActorRepairPlan = {
      ...newPlan(KIND, dataDir, records.length, now),
      journal_version: JOURNAL_VERSION,
      from,
      to,
      to_entity: toEntity,
      rows: [],
      mirror_table_present: false,
    };
    await withMirror(dataDir, async (table) => {
      plan.mirror_table_present = table !== null;
      for (const before of records) {
        const after = { ...before, actor: to };
        const mirrorBefore = await mirrorRow(table, before.id);
        if (mirrorBefore && mirrorBefore.actor !== from)
          refuse(`Action mirror disagrees with SQL actor: ${before.id}`);
        // The mirror may lag SQL (including updated_at); this repair changes only actor.
        const mirrorAfter = mirrorBefore ? { ...mirrorBefore, actor: to } : null;
        plan.rows.push({
          before,
          after,
          mirror_before: mirrorBefore,
          mirror_after: mirrorAfter,
          audit: auditInput(
            plan,
            `${KIND}:${before.id}`,
            {
              action_id: before.id,
              before,
              after,
              mirror_before: mirrorBefore,
              mirror_after: mirrorAfter,
            },
            {
              actor: from,
              before,
              expected_after: after,
              mirror_before: mirrorBefore,
              mirror_expected_after: mirrorAfter,
            },
          ),
        });
      }
    });
    plan.summary = {
      changed: plan.rows.length,
      mirror_updates: plan.rows.filter((r) => r.mirror_before !== null).length,
    };
    return plan;
  } finally {
    db.close();
  }
}

function assertActor(
  current: ActionRecord | null,
  id: ActionRecord["id"],
  actor: EntityId,
): ActionRecord {
  assertSame(
    current === null ? null : { id: current.id, actor: current.actor },
    { id, actor },
    `action identity/actor ${id}`,
  );
  return current!;
}

function sqlPreflight(db: SqliteDatabase, plan: ActorRepairPlan): void {
  entityPreflight(db, plan.from, plan.to);
  const repo = new ActionRepository({ db });
  for (const row of plan.rows)
    assertActor(
      repo.get(row.before.id),
      row.before.id,
      hasStep(db, plan, row.audit.action) ? plan.to : plan.from,
    );
}
async function preflight(db: SqliteDatabase, plan: ActorRepairPlan): Promise<void> {
  sqlPreflight(db, plan);
  await withMirror(plan.dataDir, async (table) => {
    assertSame(table !== null, plan.mirror_table_present, "action mirror table");
    for (const row of plan.rows) {
      const current = await mirrorRow(table, row.before.id);
      const done = hasStep(db, plan, row.audit.action);
      // A journaled SQL commit may have either asserted mirror state.
      if (done && sameMirrorSnapshot(current, row.mirror_after)) continue;
      if (!sameMirrorSnapshot(current, row.mirror_before))
        refuse(`Changed since planning: action mirror ${row.before.id}`);
    }
  });
}
export async function applyActionActorRepair(
  plan: ActorRepairPlan,
  options: { resume?: boolean; checkpoint?: RepairCheckpoint } = {},
): Promise<void> {
  // Reject incompatible plans before opening a writable store or touching the journal.
  assertJournalVersion(plan);
  await applyDataRepair(
    plan,
    {
      preflight,
      async finish(db, plan, checkpoint) {
        db.transaction(() => {
          sqlPreflight(db, plan);
          const repo = new ActionRepository({ db });
          for (const row of plan.rows) {
            if (hasStep(db, plan, row.audit.action)) continue;
            // Earlier writes can update sibling metadata. Re-read immediately before
            // each write and guard only the handle and actor this repair depends on.
            const before = assertActor(repo.get(row.before.id), row.before.id, plan.from);
            const { id, created_at: _createdAt, ...currentFields } = before;
            // Zod defaults in the general patch schema fill omitted nullable fields.
            // Carry the fresh fields so an actor correction cannot clear archive,
            // session, or reference metadata. Never carry the stale plan snapshot.
            repo.update(id, { ...currentFields, actor: plan.to }, { skipSideEffects: true });
            const after = assertActor(repo.get(id), id, plan.to);
            repairAudit(db, plan).record({
              ...row.audit,
              targets: { ...row.audit.targets, before, after, repair_fields: ["actor"] },
              reversal: {
                ...row.audit.reversal,
                before,
                expected_after: after,
                repair_fields: ["actor"],
              },
            });
          }
          // Detect an actor change to an already-written sibling before committing.
          sqlPreflight(db, plan);
        }).immediate();
        await checkpoint?.("after_sqlite");
        await preflight(db, plan);
        await withMirror(plan.dataDir, async (table) => {
          if (!table) return;
          const repo = new ActionRepository({ db, table });
          for (const row of plan.rows) {
            if (row.mirror_before === null) continue;
            await repo.syncRepairedActorMirror(row.before.id, plan.from);
            if (!sameMirrorSnapshot(await mirrorRow(table, row.before.id), row.mirror_after))
              refuse(`Changed since planning: updated mirror ${row.before.id}`);
            await checkpoint?.(`after_mirror:${row.before.id}`);
          }
        });
      },
    },
    options,
  );
}
runRepairCli(import.meta.url, async () => {
  const args = repairArgs(process.argv.slice(2), {
    from: { type: "string" },
    to: { type: "string" },
  });
  let plan: ActorRepairPlan;
  if (args.resume) {
    plan = loadRepairPlan<ActorRepairPlan>(args.dataDir, KIND);
    assertJournalVersion(plan);
    if (args.values.from !== undefined) assertSame(args.values.from, plan.from, "--from");
    if (args.values.to !== undefined) assertSame(args.values.to, plan.to, "--to");
  } else {
    if (typeof args.values.from !== "string" || typeof args.values.to !== "string")
      refuse("--from and --to are required");
    plan = await planActionActorRepair({
      dataDir: args.dataDir,
      from: args.values.from,
      to: args.values.to,
    });
  }
  await printPlan(plan);
  if (args.apply) {
    await applyActionActorRepair(plan, { resume: args.resume });
    console.log(
      JSON.stringify({
        run_id: plan.runId,
        ...plan.summary,
        ...(plan.runId === null ? { status: "no changes; nothing pending" } : {}),
      }),
    );
  }
});
