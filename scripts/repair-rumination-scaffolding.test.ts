import { describe, expect, it } from "vitest";

import { offlineMigrations } from "../src/offline/index.js";
import { AuditLog } from "../src/offline/audit-log.js";
import { openDatabase } from "../src/storage/sqlite/index.js";
import { FixedClock } from "../src/util/clock.js";
import { createMaintenanceRunId } from "../src/util/ids.js";
import {
  formatRepairReport,
  parseRepairCliArgs,
  planRuminationScaffoldingRepairs,
  repairRuminationScaffolding,
  RUMINATION_SCAFFOLDING_REPAIR_AUDIT_ACTION,
} from "./repair-rumination-scaffolding.js";

const SYNTHETIC_ROWS = [
  {
    id: 1,
    tensions: JSON.stringify([
      '<parameter name="tensions">["Synthetic alpha","Synthetic beta"]</parameter>',
    ]),
  },
  {
    id: 2,
    tensions: JSON.stringify(['<parameter name="0">Synthetic indexed payload']),
  },
  {
    id: 3,
    tensions: JSON.stringify(['<parameter name="">Synthetic empty-name payload</parameter>']),
  },
  {
    id: 4,
    tensions: JSON.stringify(["<parameter>Synthetic unnamed payload"]),
  },
  {
    id: 5,
    tensions: JSON.stringify(['<parameter name="item">Synthetic item payload</parameter>']),
  },
  {
    id: 6,
    tensions: JSON.stringify([
      '<parameter name="growth_marker">Synthetic marker payload</parameter>',
    ]),
  },
  {
    id: 7,
    tensions: JSON.stringify(["Synthetic clean tension"]),
  },
] as const;

describe("repair rumination scaffolding", () => {
  it("plans all six observed wrapper shapes and ignores a clean row", () => {
    const plan = planRuminationScaffoldingRepairs(SYNTHETIC_ROWS);

    expect(plan.scannedRows).toBe(7);
    expect(plan.candidates.map((candidate) => candidate.id)).toEqual([1, 2, 3, 4, 5]);
    expect(plan.candidates.map((candidate) => candidate.afterTensions)).toEqual([
      ["Synthetic alpha", "Synthetic beta"],
      ["Synthetic indexed payload"],
      ["Synthetic empty-name payload"],
      ["Synthetic unnamed payload"],
      ["Synthetic item payload"],
    ]);
    expect(plan.manualDecisions).toEqual([
      expect.objectContaining({
        id: 6,
        beforeElementCount: 1,
        reason: expect.stringContaining("growth_marker"),
      }),
    ]);

    const report = formatRepairReport({ dryRun: true, ...plan, repaired: [] });
    expect(report).toContain("would_repair row=1");
    expect(report).toContain("before_length=");
    expect(report).toContain("after_elements=2");
    expect(report).toContain("manual_decision row=6");
    expect(report).not.toContain("Synthetic alpha");
    expect(report).not.toContain("Synthetic marker payload");
  });

  it("updates and audits each recoverable row once while leaving manual rows untouched", () => {
    const db = openDatabase(":memory:", { migrations: offlineMigrations });
    const clock = new FixedClock(10_000);
    const auditLog = new AuditLog({ db, clock });

    try {
      db.exec(`
        CREATE TABLE open_question_ruminations (
          id INTEGER PRIMARY KEY,
          tensions TEXT NOT NULL
        )
      `);
      const insert = db.prepare(
        "INSERT INTO open_question_ruminations (id, tensions) VALUES (?, ?)",
      );
      insert.run(SYNTHETIC_ROWS[0].id, SYNTHETIC_ROWS[0].tensions);
      insert.run(SYNTHETIC_ROWS[5].id, SYNTHETIC_ROWS[5].tensions);

      const dependencies = {
        db,
        auditLog,
        runId: createMaintenanceRunId(),
      };
      const dryRun = repairRuminationScaffolding(dependencies);

      expect(dryRun).toMatchObject({
        dryRun: true,
        candidates: [expect.objectContaining({ id: 1 })],
        manualDecisions: [expect.objectContaining({ id: 6 })],
        repaired: [],
      });
      expect(
        db.prepare("SELECT tensions FROM open_question_ruminations WHERE id = 1").get(),
      ).toEqual({ tensions: SYNTHETIC_ROWS[0].tensions });

      const applied = repairRuminationScaffolding(dependencies, { apply: true });

      expect(applied.repaired.map((candidate) => candidate.id)).toEqual([1]);
      expect(
        db.prepare("SELECT tensions FROM open_question_ruminations WHERE id = 1").get(),
      ).toEqual({ tensions: JSON.stringify(["Synthetic alpha", "Synthetic beta"]) });
      expect(
        db.prepare("SELECT tensions FROM open_question_ruminations WHERE id = 6").get(),
      ).toEqual({ tensions: SYNTHETIC_ROWS[5].tensions });
      expect(auditLog.list({ process: "ruminator" })).toEqual([
        expect.objectContaining({
          action: RUMINATION_SCAFFOLDING_REPAIR_AUDIT_ACTION,
          targets: expect.objectContaining({ rumination_id: 1 }),
        }),
      ]);

      const secondApply = repairRuminationScaffolding(dependencies, { apply: true });

      expect(secondApply.candidates).toEqual([]);
      expect(secondApply.manualDecisions).toEqual([expect.objectContaining({ id: 6 })]);
      expect(secondApply.repaired).toEqual([]);
      expect(auditLog.list({ process: "ruminator" })).toHaveLength(1);
    } finally {
      db.close();
    }
  });

  it("defaults to dry-run and accepts flag or positional data directories", () => {
    expect(parseRepairCliArgs(["--data-dir", "/tmp/example-bank"])).toMatchObject({
      help: false,
      apply: false,
      dataDir: "/tmp/example-bank",
    });
    expect(parseRepairCliArgs(["/tmp/example-bank", "--apply"])).toMatchObject({
      help: false,
      apply: true,
      dataDir: "/tmp/example-bank",
    });
  });
});
