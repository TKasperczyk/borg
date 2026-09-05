import { mkdtempSync, readFileSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";

import { afterEach, describe, expect, it } from "vitest";

import { createMigrations } from "../src/borg/storage-setup.js";
import { IdentityEventRepository } from "../src/memory/identity/index.js";
import { GoalsRepository } from "../src/memory/self/index.js";
import { openDatabase } from "../src/storage/sqlite/index.js";
import { FixedClock } from "../src/util/clock.js";
import {
  GOAL_ROLLBACK_AUDIT_REPAIR_PROVENANCE,
  GOAL_ROLLBACK_AUDIT_REPAIR_REASON,
  main,
  parseGoalRollbackAuditRepairCliArgs,
  planGoalRollbackAuditRepair,
} from "./repair-goal-rollback-audit.js";

function createOutputBuffer(): { output: { write(chunk: string): true }; read(): string } {
  let value = "";

  return {
    output: {
      write(chunk: string) {
        value += chunk;
        return true;
      },
    },
    read: () => value,
  };
}

describe("goal rollback audit repair", () => {
  const tempDirs: string[] = [];

  afterEach(() => {
    while (tempDirs.length > 0) {
      const tempDir = tempDirs.pop();

      if (tempDir !== undefined) {
        rmSync(tempDir, { recursive: true, force: true });
      }
    }
  });

  it("requires a named data directory and defaults to dry-run", () => {
    expect(parseGoalRollbackAuditRepairCliArgs(["--data-dir", "/tmp/example-bank"])).toMatchObject({
      help: false,
      apply: false,
    });
    expect(
      parseGoalRollbackAuditRepairCliArgs(["--data-dir", "/tmp/example-bank", "--apply"]),
    ).toMatchObject({ help: false, apply: true });
    expect(() => parseGoalRollbackAuditRepairCliArgs([])).toThrow("--data-dir is required");
    expect(() => parseGoalRollbackAuditRepairCliArgs(["/tmp/example-bank"])).toThrow(
      "Unknown argument",
    );
  });

  // Opens a real Borg instance and runs migrations twice; the default 15s test
  // timeout is not enough once the full suite is running these in parallel.
  it("backfills only stranded creates and reports status drift without changing it", async () => {
    const tempDir = mkdtempSync(join(tmpdir(), "borg-goal-rollback-audit-repair-"));
    tempDirs.push(tempDir);
    const databasePath = join(tempDir, "borg.db");
    const db = openDatabase(databasePath, { migrations: createMigrations() });
    const clock = new FixedClock(1_000);
    const identityEvents = new IdentityEventRepository({ db, clock });
    const goals = new GoalsRepository({ db, clock, identityEventRepository: identityEvents });
    const firstOrphan = goals.add({
      description: "First goal created by a turn that later aborted",
      priority: 9,
      provenance: { kind: "manual" },
    });
    const secondOrphan = goals.add({
      description: "Second goal created by a turn that later aborted",
      priority: 8,
      provenance: { kind: "offline", process: "reflector" },
    });
    const updatedOrphan = goals.add({
      description: "Goal updated before its turn later aborted",
      priority: 7.5,
      provenance: { kind: "manual" },
    });
    goals.updateStatus(updatedOrphan.id, "done", {
      kind: "system",
    });
    const updatedOrphanSnapshot = goals.get(updatedOrphan.id)!;
    const alreadyTerminal = goals.add({
      description: "Absent goal whose audit is already terminal",
      priority: 7,
      provenance: { kind: "manual" },
    });
    const firstDrift = goals.add({
      description: "Goal whose stored status moved without an audit",
      priority: 6,
      provenance: { kind: "manual" },
    });
    const secondDrift = goals.add({
      description: "Another goal whose stored status moved without an audit",
      priority: 5,
      provenance: { kind: "manual" },
    });
    const aligned = goals.add({
      description: "Goal whose status agrees with its audit",
      priority: 4,
      provenance: { kind: "manual" },
    });
    db.prepare("DELETE FROM goals WHERE id IN (?, ?, ?, ?)").run(
      firstOrphan.id,
      secondOrphan.id,
      updatedOrphan.id,
      alreadyTerminal.id,
    );
    identityEvents.record({
      record_type: "goal",
      record_id: alreadyTerminal.id,
      action: "delete",
      old_value: alreadyTerminal,
      new_value: null,
      reason: "existing terminal audit",
      provenance: { kind: "manual" },
    });
    db.prepare("UPDATE goals SET status = 'done' WHERE id = ?").run(firstDrift.id);
    db.prepare("UPDATE goals SET status = 'blocked' WHERE id = ?").run(secondDrift.id);
    db.close();

    const plan = planGoalRollbackAuditRepair({ dataDir: tempDir });
    expect(plan.candidates.map((candidate) => candidate.goalId)).toEqual([
      firstOrphan.id,
      secondOrphan.id,
      updatedOrphan.id,
    ]);
    expect(
      plan.candidates.find((candidate) => candidate.goalId === updatedOrphan.id),
    ).toMatchObject({
      eventCount: 2,
      latestSnapshotEvent: {
        action: "update",
        new_value: updatedOrphanSnapshot,
      },
    });
    expect(plan.statusDrifts).toHaveLength(2);
    expect(plan.statusDrifts).toEqual(
      expect.arrayContaining([
        expect.objectContaining({
          goalId: firstDrift.id,
          currentStatus: "done",
          auditedStatus: "active",
        }),
        expect.objectContaining({
          goalId: secondDrift.id,
          currentStatus: "blocked",
          auditedStatus: "active",
        }),
      ]),
    );

    const beforeDryRun = readFileSync(databasePath);
    const dryRunStdout = createOutputBuffer();
    await expect(
      main(["--data-dir", tempDir], {
        stdout: dryRunStdout.output,
        stderr: createOutputBuffer().output,
      }),
    ).resolves.toBe(0);
    expect(readFileSync(databasePath)).toEqual(beforeDryRun);
    expect(dryRunStdout.read()).toContain("mode=dry-run");
    expect(dryRunStdout.read()).toContain("stranded_create_candidates=3");
    expect(dryRunStdout.read()).toContain("status_drift_goals=2");
    expect(dryRunStdout.read()).toContain(`status_drift goal=${firstDrift.id}`);
    expect(dryRunStdout.read()).toContain("no_change=true");

    const applyStdout = createOutputBuffer();
    await expect(
      main(["--data-dir", tempDir, "--apply"], {
        stdout: applyStdout.output,
        stderr: createOutputBuffer().output,
      }),
    ).resolves.toBe(0);
    expect(applyStdout.read()).toContain("mode=apply");
    expect(applyStdout.read()).toContain("backfilled_total=3");
    expect(applyStdout.read()).toContain("status_drift_goals=2");

    const inspectionDb = openDatabase(databasePath);
    const inspectionEvents = new IdentityEventRepository({ db: inspectionDb });
    try {
      for (const [orphan, expectedSnapshot] of [
        [firstOrphan, firstOrphan],
        [secondOrphan, secondOrphan],
        [updatedOrphan, updatedOrphanSnapshot],
      ] as const) {
        const repairEvents = inspectionEvents
          .list({ recordType: "goal", recordId: orphan.id, limit: 10 })
          .filter((event) => event.reason === GOAL_ROLLBACK_AUDIT_REPAIR_REASON);
        expect(repairEvents).toHaveLength(1);
        expect(repairEvents[0]).toMatchObject({
          action: "delete",
          old_value: expectedSnapshot,
          new_value: null,
          provenance: GOAL_ROLLBACK_AUDIT_REPAIR_PROVENANCE,
        });
      }
      expect(
        inspectionEvents
          .list({ recordType: "goal", recordId: alreadyTerminal.id, limit: 10 })
          .filter((event) => event.reason === GOAL_ROLLBACK_AUDIT_REPAIR_REASON),
      ).toEqual([]);
      expect(
        inspectionDb.prepare("SELECT status FROM goals WHERE id = ?").get(firstDrift.id),
      ).toEqual({ status: "done" });
      expect(
        inspectionDb.prepare("SELECT status FROM goals WHERE id = ?").get(secondDrift.id),
      ).toEqual({ status: "blocked" });
      expect(inspectionDb.prepare("SELECT status FROM goals WHERE id = ?").get(aligned.id)).toEqual(
        { status: "active" },
      );
    } finally {
      inspectionDb.close();
    }

    const secondPlan = planGoalRollbackAuditRepair({ dataDir: tempDir });
    expect(secondPlan.candidates).toEqual([]);
    expect(secondPlan.statusDrifts).toHaveLength(2);
  }, 30_000);
});
