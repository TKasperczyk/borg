import { CommitmentRepository, mapCommitmentRow } from "../src/memory/commitments/repository.js";
import type { CommitmentRecord } from "../src/memory/commitments/types.js";
import { IdentityEventRepository } from "../src/memory/identity/repository.js";
import { StreamEntryIndexRepository } from "../src/stream/entry-index.js";
import type { SqliteDatabase } from "../src/storage/sqlite/index.js";
import type { MaintenanceAuditRecordInput } from "../src/offline/audit-log.js";
import { FixedClock } from "../src/util/clock.js";
import type { EntityId } from "../src/util/ids.js";
import { cloneDatabase } from "./repair-entity-graph-storage.js";
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
  repairArgs,
  repairAudit,
  runRepairCli,
  type DataRepairPlan,
  type RepairCheckpoint,
} from "./audited-data-repair.js";

type JoinedEntry = { entry_id: string; indexed: boolean; sender_entity_id: EntityId | null };
type CommitmentPlanRow = {
  commitment_id: CommitmentRecord["id"];
  status: "single_sender" | "zero_senders" | "multiple_senders";
  resolved_sender: EntityId | null;
  distinct_senders: EntityId[];
  joined_entries: JoinedEntry[];
  before: CommitmentRecord;
  after: CommitmentRecord | null;
  audit: MaintenanceAuditRecordInput | null;
};
export type CommitterPlan = DataRepairPlan & {
  rows: CommitmentPlanRow[];
  identity_events: unknown[];
};
const KIND = "backfill-commitment-committers";
const reason = "Backfill directive source-entry sender from stream_entry_index";

function joinedEntries(db: SqliteDatabase, dataDir: string, row: CommitmentRecord): JoinedEntry[] {
  const index = new StreamEntryIndexRepository({ db, dataDir });
  const ids = row.source_stream_entry_ids ?? [];
  const found = index.lookupMany(ids);
  return ids.map((id) => ({
    entry_id: id,
    indexed: found.has(id),
    sender_entity_id: found.get(id)?.sender_entity_id ?? null,
  }));
}
function repository(db: SqliteDatabase, plannedAt: number): CommitmentRepository {
  const clock = new FixedClock(plannedAt);
  return new CommitmentRepository({
    db,
    clock,
    identityEventRepository: new IdentityEventRepository({ db, clock }),
  });
}
function update(db: SqliteDatabase, row: CommitmentPlanRow, now: number) {
  return repository(db, now).update(
    row.commitment_id,
    { committed_by_entity_id: row.resolved_sender },
    { kind: "manual" },
    { reason, expectedVersion: row.before.record_version },
  );
}
export function planCommitterBackfill(dataDir: string, now = Date.now()): CommitterPlan {
  dataDir = checkedPath(dataDir);
  assertNoPending(dataDir);
  const db = openDataRepairDatabase(dataDir);
  try {
    // list() materializes expiry: deliberately use the read-only query + repository decoder.
    const rows = db
      .prepare("SELECT * FROM commitments WHERE committed_by_entity_id IS NULL ORDER BY id")
      .all()
      .map(mapCommitmentRow)
      .map((before): CommitmentPlanRow => {
        const joined = joinedEntries(db, dataDir, before);
        const senders = [
          ...new Set(
            joined.flatMap((e) => (e.sender_entity_id === null ? [] : [e.sender_entity_id])),
          ),
        ].sort();
        return {
          commitment_id: before.id,
          status:
            senders.length === 1
              ? "single_sender"
              : senders.length === 0
                ? "zero_senders"
                : "multiple_senders",
          resolved_sender: senders.length === 1 ? senders[0]! : null,
          distinct_senders: senders,
          joined_entries: joined,
          before,
          after: null,
          audit: null,
        };
      });
    const changed = rows.filter((r) => r.status === "single_sender");
    const plan: CommitterPlan = {
      ...newPlan(KIND, dataDir, changed.length, now),
      rows,
      identity_events: [],
    };
    plan.summary = {
      scanned_null: rows.length,
      changed: changed.length,
      zero_senders: rows.filter((r) => r.status === "zero_senders").length,
      multiple_senders: rows.filter((r) => r.status === "multiple_senders").length,
    };
    if (changed.length) {
      const projection = cloneDatabase(db);
      try {
        const oldEventIds = new Set(
          projection
            .prepare("SELECT id FROM identity_events")
            .all()
            .map((r) => r.id),
        );
        for (const row of changed) {
          row.after = update(projection, row, now)!;
          row.audit = auditInput(
            plan,
            `${KIND}:${row.commitment_id}`,
            {
              commitment_id: row.commitment_id,
              joined_entries: row.joined_entries,
              before: row.before,
              after: row.after,
            },
            { committed_by_entity_id: null, before: row.before, expected_after: row.after },
          );
        }
        plan.identity_events = projection
          .prepare("SELECT * FROM identity_events ORDER BY id")
          .all()
          .filter((r) => !oldEventIds.has(r.id));
      } finally {
        projection.close();
      }
    }
    return plan;
  } finally {
    db.close();
  }
}

function preflight(db: SqliteDatabase, plan: CommitterPlan): void {
  const repo = repository(db, plan.plannedAt);
  for (const row of plan.rows) {
    const done = row.audit !== null && hasStep(db, plan, row.audit.action);
    assertSame(repo.get(row.commitment_id), done ? row.after : row.before, row.commitment_id);
    assertSame(
      joinedEntries(db, plan.dataDir, row.before),
      row.joined_entries,
      `index joins for ${row.commitment_id}`,
    );
  }
}
export async function applyCommitterBackfill(
  plan: CommitterPlan,
  options: { resume?: boolean; checkpoint?: RepairCheckpoint } = {},
): Promise<void> {
  await applyDataRepair(
    plan,
    {
      preflight,
      async finish(db, plan, checkpoint) {
        db.transaction(() => {
          preflight(db, plan);
          for (const row of plan.rows) {
            if (row.audit === null || hasStep(db, plan, row.audit.action)) continue;
            assertSame(update(db, row, plan.plannedAt), row.after, `updated ${row.commitment_id}`);
            repairAudit(db, plan).record(row.audit);
          }
        }).immediate();
        await checkpoint?.("after_sqlite");
      },
    },
    options,
  );
}
runRepairCli(import.meta.url, async () => {
  const args = repairArgs(process.argv.slice(2));
  const plan = args.resume
    ? loadRepairPlan<CommitterPlan>(args.dataDir, KIND)
    : planCommitterBackfill(args.dataDir);
  await printPlan(plan);
  if (args.apply) {
    await applyCommitterBackfill(plan, { resume: args.resume });
    console.log(
      JSON.stringify({
        run_id: plan.runId,
        ...plan.summary,
        ...(plan.runId === null ? { status: "no changes; nothing pending" } : {}),
      }),
    );
  }
});
