import { cpSync, existsSync, mkdtempSync, rmSync } from "node:fs";
import { join } from "node:path";
import { spawnSync } from "node:child_process";
import { afterEach, describe, expect, it, vi } from "vitest";
import { ReviewQueueRepository } from "../src/memory/review-queue/review-queue.js";
import {
  AutobiographicalRepository,
  autobiographicalPeriodSchema,
} from "../src/memory/self/autobiographical.js";
import { IdentityService } from "../src/memory/identity/service.js";
import { AuditLog } from "../src/offline/audit-log.js";
import { createAutobiographicalPeriodId } from "../src/util/ids.js";
import {
  applyPeriodRollover,
  planPeriodRollover,
  verifyRolloverBackup,
  MANUAL_RESOLUTION_REF,
  type PeriodRolloverPlan,
} from "./apply-period-rollover.js";
import { formatDataRepairPlan, loadRepairPlan, repairPendingPath } from "./audited-data-repair.js";
import { repairFixture } from "./audited-data-repair-test-support.js";
import { sqlInventory } from "./repair-entity-graph-storage.js";

const cleanups: Array<() => void> = [];
afterEach(() => {
  vi.restoreAllMocks();
  while (cleanups.length) cleanups.pop()!();
});
function fixture() {
  const f = repairFixture();
  cleanups.push(f.cleanup);
  const repo = new AutobiographicalRepository(f),
    queue = new ReviewQueueRepository(f);
  const before = repo.upsertPeriod({
    label: "Earlier",
    start_ts: 100,
    narrative: "Original narrative",
    themes: [],
    provenance: { kind: "offline", process: "self-narrator" },
  });
  const next = autobiographicalPeriodSchema.parse({
    id: createAutobiographicalPeriodId(),
    label: "Successor",
    start_ts: 400,
    end_ts: null,
    narrative: "Successor narrative",
    key_episode_ids: [],
    themes: [],
    provenance: { kind: "offline", process: "self-narrator" },
    created_at: 1000,
    last_updated: 1000,
  });
  const refs = {
    target_type: "autobiographical_period",
    target_id: before.id,
    repair_op: "patch",
    patch: { end_ts: 300, narrative: "Reviewed predecessor narrative" },
    proposed_provenance: { kind: "offline", process: "self-narrator" },
    next_period_open_payload: next,
  };
  const addReview = (reviewRefs: Record<string, unknown>) =>
    queue.enqueue({
      kind: "identity_inconsistency",
      refs: reviewRefs,
      reason: "Opaque multilingual proposal: 閉鎖 / récit",
    });
  const main = addReview(refs);
  const duplicate = addReview({ ...refs, patch: { end_ts: 250 } });
  const narrative = addReview({
    target_type: "autobiographical_period",
    target_id: before.id,
    repair_op: "patch",
    patch: { narrative: "close period; duplicate; 新しい物語", themes: ["opaque"] },
  });
  const backup = mkdtempSync(join(process.cwd(), ".borg-test-rollover-backup-"));
  cleanups.push(() => rmSync(backup, { recursive: true, force: true }));
  cpSync(f.dataDir, backup, { recursive: true });
  const input = {
    dataDir: f.dataDir,
    reviewId: main.id,
    dismiss: [duplicate.id, narrative.id],
    operator: "Fixture operator",
    backupVerified: backup,
  };
  return {
    ...f,
    repo,
    queue,
    before,
    next,
    refs,
    addReview,
    main,
    duplicate,
    narrative,
    backup,
    input,
  };
}
describe("operator-applied period rollover", () => {
  it("plans one successor-start boundary and complete before/after rows without writes", async () => {
    const f = fixture();
    const before = sqlInventory(f.db);
    const plan = await planPeriodRollover({ ...f.input, backupVerified: undefined }, 2000);
    expect(plan.boundary).toBe(400);
    expect(plan.review.before.refs.patch).toMatchObject({ end_ts: 300 });
    expect(plan.predecessor.after).toMatchObject({
      end_ts: 400,
      provenance: { kind: "manual" },
      last_updated: 2000,
      record_version: 2,
    });
    expect(plan.successor.after).toMatchObject({
      start_ts: 400,
      end_ts: null,
      provenance: { kind: "manual" },
    });
    expect(plan.applied_refs.patch.provenance).toEqual({ kind: "manual" });
    expect(plan.identity_events).toHaveLength(2);
    expect(sqlInventory(f.db)).toEqual(before);
    const printed = JSON.parse(formatDataRepairPlan(plan));
    expect(printed.dismissals).toHaveLength(2);
    expect(printed.predecessor.before).toEqual(f.before);
    expect(printed.successor.before).toBeNull();
    expect(existsSync(repairPendingPath(f.dataDir, plan.kind))).toBe(false);
  });
  it("uses the review apply path, manual provenance on both rows and dismissal texts by shape", async () => {
    const f = fixture();
    const plan = await planPeriodRollover(f.input, 2000);
    const update = vi.spyOn(IdentityService.prototype, "updatePeriod");
    await applyPeriodRollover(plan, { backupVerified: f.backup });
    expect(update).toHaveBeenCalledWith(
      f.before.id,
      expect.objectContaining({ end_ts: 400, provenance: { kind: "manual" } }),
      { kind: "manual" },
      expect.objectContaining({ throughReview: true, reviewItemId: f.main.id }),
    );
    expect(f.repo.getPeriod(f.before.id)).toEqual(plan.predecessor.after);
    expect(f.repo.getPeriod(f.next.id)).toEqual(plan.successor.after);
    expect(f.queue.get(f.main.id)).toEqual(plan.review.after);
    const manual = f.queue.get(f.main.id)!.refs[MANUAL_RESOLUTION_REF];
    expect(manual).toMatchObject({
      operator: "Fixture operator",
      boundary: 400,
      text: expect.stringContaining("Fixture operator"),
    });
    expect(manual).toMatchObject({ text: expect.stringContaining("400") });
    expect(f.queue.get(f.duplicate.id)).toMatchObject({
      resolution: "dismiss",
      refs: { [MANUAL_RESOLUTION_REF]: { text: `dismissed as duplicate of ${f.main.id}` } },
    });
    expect(f.queue.get(f.narrative.id)).toMatchObject({
      resolution: "dismiss",
      refs: {
        [MANUAL_RESOLUTION_REF]: {
          text: `superseded by the operator-applied rollover ${f.main.id}; content after the boundary belongs to the successor`,
        },
      },
    });
    expect(f.audit.list()).toHaveLength(4);
    expect(f.audit.list().every((a) => a.process === "correction")).toBe(true);
    expect(f.db.prepare("SELECT * FROM identity_events ORDER BY id").all()).toEqual(
      plan.identity_events,
    );
    expect(
      f.audit.list().find((a) => a.action === `apply-period-rollover:review:${f.main.id}`)
        ?.reversal,
    ).toMatchObject({
      predecessor_before: f.before,
      successor_before: null,
      review_before: f.main,
    });
    await expect(planPeriodRollover(f.input)).rejects.toThrow("open identity_inconsistency");
  });
  it.each([50, 100])("refuses boundary %s at or before predecessor start", async (boundary) => {
    const f = fixture();
    const row = f.addReview({
      ...f.refs,
      next_period_open_payload: { ...f.next, start_ts: boundary },
    });
    await expect(planPeriodRollover({ ...f.input, reviewId: row.id })).rejects.toThrow(
      "strictly after",
    );
  });
  it("refuses multiple open periods", async () => {
    const f = fixture();
    const older = f.repo.upsertPeriod({
      label: "Older",
      start_ts: 0,
      end_ts: 100,
      narrative: "Prior",
      provenance: { kind: "manual" },
    });
    // Deliberately corrupt only this fixture to exercise defense against damaged stores.
    f.db.exec("DROP INDEX autobiographical_single_open");
    f.db.prepare("UPDATE autobiographical_periods SET end_ts = NULL WHERE id = ?").run(older.id);
    await expect(planPeriodRollover(f.input)).rejects.toThrow("Exactly one period");
  });
  it("refuses a target period that has already closed", async () => {
    const f = fixture();
    f.repo.upsertPeriod({
      label: "Another",
      start_ts: 500,
      narrative: "Later",
      provenance: { kind: "manual" },
    });
    await expect(planPeriodRollover(f.input)).rejects.toThrow("Target period is not open");
  });
  it.each([50, 150])("refuses an orphan or overlapping historical boundary at %s", async (end) => {
    const f = fixture();
    f.repo.upsertPeriod({
      label: "Older",
      start_ts: 0,
      end_ts: end,
      narrative: "Prior",
      provenance: { kind: "manual" },
    });
    await expect(planPeriodRollover(f.input)).rejects.toThrow(
      end === 50 ? "Orphan interval" : "Overlapping periods",
    );
    expect(f.audit.list()).toEqual([]);
  });
  it("refuses a closed successor, reused successor ID, or shifted predecessor start", async () => {
    const f = fixture();
    const closed = f.addReview({ ...f.refs, next_period_open_payload: { ...f.next, end_ts: 600 } });
    await expect(planPeriodRollover({ ...f.input, reviewId: closed.id })).rejects.toThrow(
      "Successor must be open",
    );
    const shifted = f.addReview({ ...f.refs, patch: { start_ts: 90, end_ts: 300 } });
    await expect(planPeriodRollover({ ...f.input, reviewId: shifted.id })).rejects.toThrow(
      "predecessor start",
    );
    f.repo.upsertPeriod({ ...f.next, end_ts: 600 });
    await expect(planPeriodRollover(f.input)).rejects.toThrow("already exists");
  });
  it("refuses resolved reviews and non-rollover proposals", async () => {
    const f = fixture();
    await expect(
      planPeriodRollover({ ...f.input, reviewId: f.narrative.id, dismiss: [] }),
    ).rejects.toThrow("closure and next_period_open_payload");
    f.db
      .prepare("UPDATE review_queue SET resolved_at = 1200, resolution = 'dismiss' WHERE id = ?")
      .run(f.main.id);
    await expect(planPeriodRollover(f.input)).rejects.toThrow("open identity_inconsistency");
  });
  it("refuses unrelated, reopening and repeated dismissal IDs", async () => {
    const f = fixture();
    const unrelated = f.addReview({ ...f.refs, target_id: createAutobiographicalPeriodId() });
    await expect(planPeriodRollover({ ...f.input, dismiss: [unrelated.id] })).rejects.toThrow(
      "another period",
    );
    const reopening = f.addReview({
      ...f.narrative.refs,
      patch: { end_ts: null, narrative: "Some narrative" },
    });
    await expect(planPeriodRollover({ ...f.input, dismiss: [reopening.id] })).rejects.toThrow(
      "neither a closure nor a narrative patch",
    );
    await expect(planPeriodRollover({ ...f.input, dismiss: [f.main.id] })).rejects.toThrow(
      "distinct",
    );
    await expect(
      planPeriodRollover({ ...f.input, dismiss: [f.duplicate.id, f.duplicate.id] }),
    ).rejects.toThrow("distinct");
  });
  it("requires an existing independent backup argument on every apply", async () => {
    const f = fixture();
    const plan = await planPeriodRollover(f.input);
    await expect(applyPeriodRollover(plan)).rejects.toThrow("--backup-verified");
    expect(() => verifyRolloverBackup(f.dataDir, f.dataDir)).toThrow("independent");
    expect(() => verifyRolloverBackup(f.dataDir, join(f.dataDir, "borg.db"))).toThrow(
      "independent",
    );
    rmSync(f.backup, { recursive: true });
    await expect(applyPeriodRollover(plan, { backupVerified: f.backup })).rejects.toThrow();
    expect(f.audit.list()).toEqual([]);
    expect(existsSync(repairPendingPath(f.dataDir, plan.kind))).toBe(false);
  });
  it.each([
    "after_journal",
    "rollover",
    "first_dismissal",
    "last_dismissal",
    "after_completion",
    "after_archive",
  ])("resumes %s without duplicate periods, resolutions or audits", async (boundary) => {
    const f = fixture();
    const plan = await planPeriodRollover(f.input, 2000);
    const point =
      (
        {
          rollover: `after_review:${f.main.id}`,
          first_dismissal: `after_review:${f.duplicate.id}`,
          last_dismissal: `after_review:${f.narrative.id}`,
        } as Record<string, string>
      )[boundary] ?? boundary;
    await expect(
      applyPeriodRollover(plan, {
        backupVerified: f.backup,
        checkpoint: (at) => {
          if (at === point) throw new Error("interrupted");
        },
      }),
    ).rejects.toThrow("interrupted");
    const loaded = loadRepairPlan<PeriodRolloverPlan>(f.dataDir, plan.kind);
    await applyPeriodRollover(loaded, { backupVerified: f.backup, resume: true });
    expect(f.repo.listPeriods()).toHaveLength(2);
    expect(f.repo.currentPeriod()?.id).toBe(f.next.id);
    expect(f.repo.getPeriod(f.before.id)?.end_ts).toBe(400);
    expect(f.audit.list()).toHaveLength(4);
    expect(f.db.prepare("SELECT * FROM identity_events").all()).toHaveLength(2);
    expect(f.queue.get(f.main.id)).toEqual(plan.review.after);
    expect(f.queue.get(f.narrative.id)).toEqual(plan.dismissals[1]!.after);
  });
  it("rolls back period and identity writes when the in-transaction audit fails", async () => {
    const f = fixture();
    const plan = await planPeriodRollover(f.input, 2000);
    const spy = vi.spyOn(AuditLog.prototype, "record").mockImplementationOnce(() => {
      throw new Error("audit failed");
    });
    await expect(applyPeriodRollover(plan, { backupVerified: f.backup })).rejects.toThrow(
      "audit failed",
    );
    expect(f.repo.getPeriod(f.before.id)).toEqual(f.before);
    expect(f.repo.getPeriod(f.next.id)).toBeNull();
    expect(f.queue.get(f.main.id)).toEqual(f.main);
    expect(f.db.prepare("SELECT * FROM identity_events").all()).toEqual([]);
    expect(f.audit.list()).toEqual([]);
    spy.mockRestore();
    await applyPeriodRollover(loadRepairPlan<PeriodRolloverPlan>(f.dataDir, plan.kind), {
      backupVerified: f.backup,
      resume: true,
    });
    expect(f.audit.list()).toHaveLength(4);
  });
  it("rejects changed timelines before creating a journal", async () => {
    const f = fixture();
    const plan = await planPeriodRollover(f.input);
    f.repo.upsertPeriod({ ...f.before, narrative: "Intervening narrative" });
    await expect(applyPeriodRollover(plan, { backupVerified: f.backup })).rejects.toThrow(
      "Changed since planning",
    );
    expect(f.queue.get(f.main.id)).toEqual(f.main);
    expect(existsSync(repairPendingPath(f.dataDir, plan.kind))).toBe(false);
  });
  it("runs the actual CLI as dry run, requires backup, and supports apply/resume", () => {
    const f = fixture();
    const before = sqlInventory(f.db);
    const args = [
      "--data-dir",
      f.dataDir,
      "--review-id",
      String(f.main.id),
      "--dismiss",
      String(f.duplicate.id),
      "--dismiss",
      String(f.narrative.id),
      "--operator",
      "CLI operator",
    ];
    const run = (flags: string[]) =>
      spawnSync(
        process.execPath,
        ["--import", "tsx", "scripts/apply-period-rollover.ts", ...flags],
        { cwd: process.cwd(), encoding: "utf8", timeout: 20_000 },
      );
    const dry = run(args);
    expect(dry.status, dry.stderr).toBe(0);
    const plan = JSON.parse(dry.stdout);
    expect(plan.boundary).toBe(400);
    expect(plan.dismissals).toHaveLength(2);
    expect(sqlInventory(f.db)).toEqual(before);
    const missingBackup = run([...args, "--apply"]);
    expect(missingBackup.status).toBe(1);
    expect(missingBackup.stderr).toContain("--backup-verified");
    expect(sqlInventory(f.db)).toEqual(before);
    const applied = run([...args, "--backup-verified", f.backup, "--apply"]);
    expect(applied.status, applied.stderr).toBe(0);
    expect(f.repo.currentPeriod()?.id).toBe(f.next.id);
    const resumed = run([
      "--data-dir",
      f.dataDir,
      "--backup-verified",
      f.backup,
      "--apply",
      "--resume",
    ]);
    expect(resumed.status, resumed.stderr).toBe(0);
    expect(f.audit.list()).toHaveLength(4);
  }, 30_000);
});
