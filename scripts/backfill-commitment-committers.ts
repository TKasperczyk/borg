import { CommitmentRepository, mapCommitmentRow } from "../src/memory/commitments/repository.js";
import type { CommitmentRecord } from "../src/memory/commitments/types.js";
import { IdentityEventRepository } from "../src/memory/identity/repository.js";
import { StreamEntryIndexRepository } from "../src/stream/entry-index.js";
import type { SqliteDatabase } from "../src/storage/sqlite/index.js";
import type { MaintenanceAuditRecordInput } from "../src/offline/audit-log.js";
import { FixedClock } from "../src/util/clock.js";
import { parseCommitmentId, type EntityId } from "../src/util/ids.js";
import { cloneDatabase, type Row } from "./repair-entity-graph-storage.js";
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
type SourceAttribution = {
  status: "single_sender" | "zero_senders" | "multiple_senders" | "unresolved_source_entry";
  reason: "unresolved source entry" | null;
  resolved_sender: EntityId | null;
  distinct_senders: EntityId[];
};
type CommitmentPlanRow = SourceAttribution & {
  commitment_id: CommitmentRecord["id"];
  joined_entries: JoinedEntry[];
  before: Row;
  after: Row | null;
  audit: MaintenanceAuditRecordInput | null;
};
export type CommitterPlan = DataRepairPlan & {
  rows: CommitmentPlanRow[];
  identity_events: unknown[];
};
const KIND = "backfill-commitment-committers";
const reason = "Backfill directive source-entry sender from stream_entry_index";

function sourceAttribution(joined: JoinedEntry[]): SourceAttribution {
  const senders = [
    ...new Set(joined.flatMap((e) => (e.sender_entity_id === null ? [] : [e.sender_entity_id]))),
  ].sort();
  const unresolved = joined.some((e) => !e.indexed || e.sender_entity_id === null);
  const status = unresolved
    ? "unresolved_source_entry"
    : senders.length === 1
      ? "single_sender"
      : senders.length === 0
        ? "zero_senders"
        : "multiple_senders";
  return {
    status,
    reason: unresolved ? "unresolved source entry" : null,
    resolved_sender: status === "single_sender" ? senders[0]! : null,
    distinct_senders: senders,
  };
}

function joinedEntries(db: SqliteDatabase, dataDir: string, row: Row): JoinedEntry[] {
  const index = new StreamEntryIndexRepository({ db, dataDir });
  // Decode source handles only; retain raw column values in every snapshot.
  const ids = mapCommitmentRow(row).source_stream_entry_ids ?? [];
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
  return repository(db, now).backfillCommitter(row.commitment_id, row.resolved_sender!, reason);
}
export function planCommitterBackfill(dataDir: string, now = Date.now()): CommitterPlan {
  dataDir = checkedPath(dataDir);
  assertNoPending(dataDir);
  const db = openDataRepairDatabase(dataDir);
  try {
    // list() materializes expiry and the decoder normalizes legacy columns.
    // Snapshot raw storage so those values remain visible in plans and audits.
    const rows = db
      .prepare("SELECT * FROM commitments WHERE committed_by_entity_id IS NULL ORDER BY id")
      .all()
      .map((before): CommitmentPlanRow => {
        const joined = joinedEntries(db, dataDir, before);
        return {
          commitment_id: parseCommitmentId(String(before.id)),
          ...sourceAttribution(joined),
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
      unresolved_source_rows: rows.filter((r) => r.status === "unresolved_source_entry").length,
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
          row.after = update(projection, row, now);
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
  for (const row of plan.rows) {
    const done = row.audit !== null && hasStep(db, plan, row.audit.action);
    assertSame(
      db.prepare("SELECT * FROM commitments WHERE id = ?").get(row.commitment_id),
      done ? row.after : row.before,
      row.commitment_id,
    );
    const joined = joinedEntries(db, plan.dataDir, row.before);
    assertSame(joined, row.joined_entries, `index joins for ${row.commitment_id}`);
    // A journal from the older partial-join rule must not bypass full resolution on resume.
    assertSame(
      sourceAttribution(joined),
      {
        status: row.status,
        reason: row.reason ?? null,
        resolved_sender: row.resolved_sender,
        distinct_senders: row.distinct_senders,
      },
      `source attribution for ${row.commitment_id}`,
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
