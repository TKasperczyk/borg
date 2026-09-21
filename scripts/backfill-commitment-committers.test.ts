import { existsSync, readFileSync } from "node:fs";
import { join } from "node:path";
import { afterEach, describe, expect, it, vi } from "vitest";
import { AuditLog } from "../src/offline/audit-log.js";
import { CommitmentRepository, EntityRepository } from "../src/memory/commitments/repository.js";
import { StreamEntryIndexRepository } from "../src/stream/entry-index.js";
import { createStreamEntryId, DEFAULT_SESSION_ID, type StreamEntryId } from "../src/util/ids.js";
import {
  applyCommitterBackfill,
  planCommitterBackfill,
  type CommitterPlan,
} from "./backfill-commitment-committers.js";
import {
  checkedPath,
  formatDataRepairPlan,
  loadRepairPlan,
  repairArgs,
  repairPendingPath,
} from "./audited-data-repair.js";
import { repairFixture } from "./audited-data-repair-test-support.js";

const cleanups: Array<() => void> = [];
afterEach(() => {
  vi.restoreAllMocks();
  while (cleanups.length) cleanups.pop()!();
});
function fixture() {
  const f = repairFixture();
  cleanups.push(f.cleanup);
  const repo = new CommitmentRepository(f);
  const entities = new EntityRepository(f);
  const a = entities.add({ canonicalName: "A" }).id,
    b = entities.add({ canonicalName: "B" }).id;
  const index = new StreamEntryIndexRepository(f);
  const first = createStreamEntryId(),
    second = createStreamEntryId(),
    third = createStreamEntryId(),
    missing = createStreamEntryId();
  index.record(first, DEFAULT_SESSION_ID, 0, 0, 1, "user_msg", a);
  index.record(second, DEFAULT_SESSION_ID, 1, 1, 2, "user_msg", a);
  index.record(third, DEFAULT_SESSION_ID, 2, 2, 3, "user_msg", b);
  let count = 0;
  const add = (ids?: StreamEntryId[]) =>
    repo.add({
      type: "promise",
      directiveFamily: `fixture_${count++}`,
      directive: "Opaque directive",
      priority: 1,
      provenance: { kind: "manual" },
      sourceStreamEntryIds: ids,
    });
  const single = add([first, second]);
  const multiple = add([first, third]);
  const zero = add([missing]);
  const empty = add();
  const partial = add([first, missing]);
  return { ...f, repo, index, a, b, first, third, missing, single, multiple, zero, empty, partial };
}
describe("commitment committer backfill", () => {
  it("requires complete same-sender joins and lists unresolved sources separately", () => {
    const f = fixture();
    const bytes = readFileSync(join(f.dataDir, "borg.db"));
    const plan = planCommitterBackfill(f.dataDir, 2000);
    expect(plan.summary).toEqual({
      scanned_null: 5,
      changed: 1,
      zero_senders: 1,
      multiple_senders: 1,
      unresolved_source_rows: 2,
    });
    expect(plan.rows.find((r) => r.commitment_id === f.single.id)).toMatchObject({
      resolved_sender: f.a,
      joined_entries: [{ sender_entity_id: f.a }, { sender_entity_id: f.a }],
    });
    expect(plan.rows.find((r) => r.commitment_id === f.partial.id)).toMatchObject({
      status: "unresolved_source_entry",
      reason: "unresolved source entry",
      resolved_sender: null,
      distinct_senders: [f.a],
      joined_entries: [{ sender_entity_id: f.a }, { entry_id: f.missing, indexed: false }],
      after: null,
      audit: null,
    });
    expect(readFileSync(join(f.dataDir, "borg.db"))).toEqual(bytes);
    expect(f.audit.list()).toEqual([]);
    expect(formatDataRepairPlan(plan)).toContain(f.multiple.id);
    expect(existsSync(repairPendingPath(f.dataDir, plan.kind))).toBe(false);
  });
  it("applies via repository, audits prior NULL, preserves provenance and skips ambiguity", async () => {
    const f = fixture();
    const indexBefore = f.db.prepare("SELECT * FROM stream_entry_index").all();
    const plan = planCommitterBackfill(f.dataDir, 2000);
    await applyCommitterBackfill(plan);
    expect(f.repo.get(f.single.id)).toMatchObject({
      committed_by_entity_id: f.a,
      provenance: f.single.provenance,
      record_version: f.single.record_version,
    });
    for (const row of [f.multiple, f.zero, f.empty, f.partial])
      expect(f.repo.get(row.id)).toEqual(row);
    expect(f.audit.list({ process: "correction" })).toHaveLength(2);
    expect(
      f.audit.list().find((a) => a.action.endsWith(f.single.id))?.reversal.committed_by_entity_id,
    ).toBeNull();
    expect(f.db.prepare("SELECT * FROM stream_entry_index").all()).toEqual(indexBefore);
    expect(f.db.prepare("SELECT * FROM identity_events").all()).toEqual(plan.identity_events);
    const rerun = planCommitterBackfill(f.dataDir, 3000);
    expect(rerun.runId).toBeNull();
    await applyCommitterBackfill(rerun);
    expect(f.audit.list()).toHaveLength(2);
  });
  it.each(["apply", "after_sqlite"])(
    "preserves every raw legacy column and audits NULL enforcement (%s)",
    async (mode) => {
      const f = fixture();
      const sourceText = ` [\n${f.single.source_stream_entry_ids!.map((id) => JSON.stringify(id)).join(", ")}\n] `;
      f.db
        .prepare(
          `UPDATE commitments SET enforcement_class = NULL, updated_at = NULL,
        last_reinforced_at = NULL, source_stream_entry_ids = ?, source_episode_ids = '[ ]',
        provenance_episode_ids = '[ ]' WHERE id = ?`,
        )
        .run(sourceText, f.single.id);
      const read = () => f.db.prepare("SELECT * FROM commitments WHERE id = ?").get(f.single.id)!;
      const before = read();
      expect(f.repo.get(f.single.id)?.enforcement_class).toBe("advisory");
      const plan = planCommitterBackfill(f.dataDir, 2000);
      const planned = plan.rows.find((r) => r.commitment_id === f.single.id)!;
      expect(planned.before).toEqual(before);
      expect(planned.before.enforcement_class).toBeNull();
      expect(planned.after).toEqual({ ...before, committed_by_entity_id: f.a });
      expect(read()).toEqual(before);
      if (mode === "apply") await applyCommitterBackfill(plan);
      else {
        await expect(
          applyCommitterBackfill(plan, {
            checkpoint: (point) => {
              if (point === mode) throw new Error("interrupted");
            },
          }),
        ).rejects.toThrow("interrupted");
        await applyCommitterBackfill(loadRepairPlan<CommitterPlan>(f.dataDir, plan.kind), {
          resume: true,
        });
      }
      // Raw strings (including JSON whitespace), NULLs and every other column survive.
      expect(read()).toEqual({ ...before, committed_by_entity_id: f.a });
      const audit = f.audit.list().find((a) => a.action.endsWith(f.single.id))!;
      expect(audit.targets.before).toEqual(before);
      expect(audit.targets.after).toEqual(read());
      expect(audit.reversal.before).toEqual(before);
      expect(audit.reversal.expected_after).toEqual(read());
      const event = f.db
        .prepare("SELECT old_value_json, new_value_json FROM identity_events")
        .get()!;
      expect(JSON.parse(String(event.old_value_json))).toEqual(before);
      expect(JSON.parse(String(event.new_value_json))).toEqual(read());
      expect(f.audit.list()).toHaveLength(2);
    },
  );
  it("rejects changed raw columns even when the decoded value would match", async () => {
    const f = fixture();
    f.db.prepare("UPDATE commitments SET enforcement_class = NULL WHERE id = ?").run(f.single.id);
    const decoded = f.repo.get(f.single.id);
    const plan = planCommitterBackfill(f.dataDir, 2000);
    f.db
      .prepare("UPDATE commitments SET enforcement_class = 'advisory' WHERE id = ?")
      .run(f.single.id);
    expect(f.repo.get(f.single.id)).toEqual(decoded);
    await expect(applyCommitterBackfill(plan)).rejects.toThrow("Changed since planning");
    expect(f.audit.list()).toEqual([]);
    expect(existsSync(repairPendingPath(f.dataDir, plan.kind))).toBe(false);
  });
  it.each(["after_journal", "after_sqlite", "after_completion", "after_archive"])(
    "resumes interruption at %s exactly once",
    async (point) => {
      const f = fixture();
      const plan = planCommitterBackfill(f.dataDir, 2000);
      await expect(
        applyCommitterBackfill(plan, {
          checkpoint: (at) => {
            if (at === point) throw new Error("interrupted");
          },
        }),
      ).rejects.toThrow("interrupted");
      const loaded = loadRepairPlan<CommitterPlan>(f.dataDir, plan.kind);
      await applyCommitterBackfill(loaded, { resume: true });
      expect(f.repo.get(f.single.id)?.committed_by_entity_id).toBe(f.a);
      expect(f.audit.list()).toHaveLength(2);
      expect(f.db.prepare("SELECT * FROM identity_events").all()).toHaveLength(1);
    },
  );
  it("rejects stale source joins before journaling", async () => {
    const f = fixture();
    const plan = planCommitterBackfill(f.dataDir);
    f.index.record(f.first, DEFAULT_SESSION_ID, 0, 0, 1, "user_msg", f.b);
    await expect(applyCommitterBackfill(plan)).rejects.toThrow("Changed since planning");
    expect(f.audit.list()).toHaveLength(0);
    expect(existsSync(repairPendingPath(f.dataDir, plan.kind))).toBe(false);
  });
  it("counts NULL index senders as unresolved and never sweeps expired or already-attributed rows", async () => {
    const f = fixture();
    f.index.record(f.first, DEFAULT_SESSION_ID, 0, 0, 1, "user_msg", null);
    f.repo.update(f.single.id, { committed_by_entity_id: f.a }, { kind: "manual" });
    f.repo.update(f.zero.id, { expires_at: 1500, expired_at: null }, { kind: "manual" });
    const before = f.repo.get(f.single.id);
    const plan = planCommitterBackfill(f.dataDir, 2000);
    expect(plan.rows.some((r) => r.commitment_id === f.single.id)).toBe(false);
    expect(plan.rows.find((r) => r.commitment_id === f.multiple.id)).toMatchObject({
      distinct_senders: [f.b],
      status: "unresolved_source_entry",
      reason: "unresolved source entry",
      resolved_sender: null,
    });
    await applyCommitterBackfill(plan);
    expect(f.repo.get(f.single.id)).toEqual(before);
    expect(f.repo.get(f.multiple.id)).toEqual(f.multiple);
    expect(f.repo.get(f.zero.id)?.expired_at).toBeNull();
  });
  it.each(["missing_entry", "null_sender"])(
    "leaves [sender A, unresolved source] NULL without a per-row audit (%s)",
    async (kind) => {
      const f = fixture();
      if (kind === "null_sender")
        f.index.record(f.missing, DEFAULT_SESSION_ID, 3, 3, 4, "user_msg", null);
      const plan = planCommitterBackfill(f.dataDir, 2000);
      const partial = plan.rows.find((r) => r.commitment_id === f.partial.id)!;
      expect(partial.joined_entries[1]).toEqual({
        entry_id: f.missing,
        indexed: kind === "null_sender",
        sender_entity_id: null,
      });
      expect(partial.reason).toBe("unresolved source entry");
      expect(partial.resolved_sender).toBeNull();
      expect(partial.after).toBeNull();
      expect(partial.audit).toBeNull();
      expect(plan.summary.unresolved_source_rows).toBe(2);
      await applyCommitterBackfill(plan);
      expect(f.repo.get(f.partial.id)).toEqual(f.partial);
      expect(f.audit.list().some((a) => a.action.endsWith(f.partial.id))).toBe(false);
      expect(f.repo.get(f.single.id)?.committed_by_entity_id).toBe(f.a);
    },
  );
  it("classifies a missing source distinctly even with multiple known senders", () => {
    const f = fixture();
    f.repo.update(
      f.partial.id,
      { source_stream_entry_ids: [f.first, f.third, f.missing] },
      { kind: "manual" },
    );
    const plan = planCommitterBackfill(f.dataDir, 2000);
    expect(plan.rows.find((r) => r.commitment_id === f.partial.id)).toMatchObject({
      status: "unresolved_source_entry",
      reason: "unresolved source entry",
      distinct_senders: [f.a, f.b].sort(),
      resolved_sender: null,
      audit: null,
    });
  });
  it("refuses the older partial-join classification before journaling", async () => {
    const f = fixture();
    const plan = planCommitterBackfill(f.dataDir, 2000);
    const partial = plan.rows.find((r) => r.commitment_id === f.partial.id)!;
    partial.status = "single_sender";
    partial.reason = null;
    partial.resolved_sender = f.a;
    await expect(applyCommitterBackfill(plan)).rejects.toThrow("source attribution");
    expect(f.repo.get(f.partial.id)).toEqual(f.partial);
    expect(f.audit.list()).toEqual([]);
    expect(existsSync(repairPendingPath(f.dataDir, plan.kind))).toBe(false);
  });
  it("rolls back repository and identity writes when audit persistence fails", async () => {
    const f = fixture();
    const plan = planCommitterBackfill(f.dataDir, 2000);
    vi.spyOn(AuditLog.prototype, "record").mockImplementation(() => {
      throw new Error("audit failed");
    });
    await expect(applyCommitterBackfill(plan)).rejects.toThrow("audit failed");
    expect(f.repo.get(f.single.id)).toEqual(f.single);
    expect(f.db.prepare("SELECT * FROM identity_events").all()).toEqual([]);
    expect(f.audit.list()).toEqual([]);
    vi.restoreAllMocks();
    await applyCommitterBackfill(loadRepairPlan<CommitterPlan>(f.dataDir, plan.kind), {
      resume: true,
    });
    expect(f.repo.get(f.single.id)?.committed_by_entity_id).toBe(f.a);
  });
  it("rejects stale rows before journaling", async () => {
    const f = fixture();
    const plan = planCommitterBackfill(f.dataDir);
    f.repo.update(f.single.id, { directive: "Intervening change" }, { kind: "manual" });
    await expect(applyCommitterBackfill(plan)).rejects.toThrow("Changed since planning");
    expect(f.audit.list()).toHaveLength(0);
  });
  it("refuses pending migrations without applying them", () => {
    const f = fixture();
    f.db.prepare("DELETE FROM _migrations WHERE id = (SELECT MAX(id) FROM _migrations)").run();
    expect(() => planCommitterBackfill(f.dataDir)).toThrow("migrations");
    expect(f.audit.list()).toHaveLength(0);
  });
  it("requires an explicit bank and refuses the live path before opening it", () => {
    expect(() => repairArgs([])).toThrow("--data-dir");
    expect(() => checkedPath("demo/server/.borg-data/demo")).toThrow("forbidden");
    expect(() => repairArgs(["--data-dir", process.cwd(), "--resume"])).toThrow(
      "--resume requires --apply",
    );
  });
});
