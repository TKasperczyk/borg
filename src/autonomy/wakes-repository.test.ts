import { mkdtempSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";

import { describe, expect, it } from "vitest";

import { openDatabase } from "../storage/sqlite/index.js";
import { ManualClock } from "../util/clock.js";
import { DEFAULT_SESSION_ID } from "../util/ids.js";

import { autonomyMigrations } from "./migrations.js";
import { AUTONOMY_CONDITION_NAMES, AUTONOMY_WAKE_SOURCE_NAMES } from "./types.js";
import { AutonomyWakesRepository } from "./wakes-repository.js";

describe("AutonomyWakesRepository", () => {
  it("records wakes and counts them since a cutoff", () => {
    const clock = new ManualClock(1_000);
    const db = openDatabase(":memory:", {
      migrations: autonomyMigrations,
    });
    const repository = new AutonomyWakesRepository({
      db,
      clock,
    });

    try {
      repository.record({
        trigger_name: "scheduled_reflection",
        condition_name: null,
        session_id: DEFAULT_SESSION_ID,
        wake_source_type: "trigger",
        source_category: "contemplative",
      });
      clock.set(2_000);
      repository.record({
        trigger_name: "commitment_revoked",
        condition_name: "commitment_revoked",
        session_id: DEFAULT_SESSION_ID,
        wake_source_type: "condition",
      });

      expect(repository.countSince(1_000)).toBe(2);
      expect(repository.countSince(1_500)).toBe(1);
      expect(repository.countSince(0, { sourceCategory: "contemplative" })).toBe(1);
      expect(repository.countSince(0, { sourceCategory: "operational" })).toBe(1);
      expect(repository.listSince(0, 10)[1]?.source_category).toBe("contemplative");
      expect(repository.listSince(0, 10).map((wake) => wake.trigger_name)).toEqual([
        "commitment_revoked",
        "scheduled_reflection",
      ]);
    } finally {
      db.close();
    }
  });

  it("records outcomes and filters counts without excluding legacy null outcomes", () => {
    const clock = new ManualClock(1_000);
    const db = openDatabase(":memory:", {
      migrations: autonomyMigrations,
    });
    const repository = new AutonomyWakesRepository({ db, clock });

    try {
      const headwayWake = repository.record({
        trigger_name: "goal_followup_due",
        session_id: DEFAULT_SESSION_ID,
        wake_source_type: "trigger",
      });
      clock.advance(1);
      const legacyNullWake = repository.record({
        trigger_name: "scheduled_reflection",
        session_id: DEFAULT_SESSION_ID,
        wake_source_type: "trigger",
        source_category: "contemplative",
      });

      repository.recordOutcome(headwayWake.id, "headway");

      expect(repository.countSince(0)).toBe(2);
      expect(repository.countSince(0, { outcome: "headway" })).toBe(1);
      expect(repository.countSince(0, { outcome: "silent" })).toBe(0);
      expect(
        repository.countSince(0, {
          sourceCategory: "operational",
          outcome: "headway",
        }),
      ).toBe(1);
      expect(repository.listSince(0, 10)).toEqual(
        expect.arrayContaining([
          expect.objectContaining({ id: headwayWake.id, outcome: "headway" }),
          expect.objectContaining({ id: legacyNullWake.id, outcome: null }),
        ]),
      );
    } finally {
      db.close();
    }
  });

  it("applies the additive outcome migration over legacy wake rows", () => {
    const tempDir = mkdtempSync(join(tmpdir(), "borg-autonomy-migration-"));
    const dbPath = join(tempDir, "borg.db");
    let db = openDatabase(dbPath, {
      migrations: autonomyMigrations.slice(0, 2),
    });

    try {
      db.prepare(
        `
          INSERT INTO autonomy_wakes (
            id, ts, trigger_name, condition_name, session_id, wake_source_type, source_category
          ) VALUES (?, ?, ?, ?, ?, ?, ?)
        `,
      ).run(
        "autonomy_wake_aaaaaaaaaaaaaaaa",
        1_000,
        "goal_followup_due",
        null,
        DEFAULT_SESSION_ID,
        "trigger",
        "operational",
      );
      db.close();

      db = openDatabase(dbPath, { migrations: autonomyMigrations });
      const repository = new AutonomyWakesRepository({ db, clock: new ManualClock(2_000) });
      const columns = db.prepare("PRAGMA table_info(autonomy_wakes)").all() as Array<{
        name: string;
      }>;

      expect(columns.map((column) => column.name)).toContain("outcome");
      expect(repository.listSince(0, 10)).toEqual([
        expect.objectContaining({
          id: "autonomy_wake_aaaaaaaaaaaaaaaa",
          outcome: null,
        }),
      ]);
    } finally {
      db.close();
      rmSync(tempDir, { recursive: true, force: true });
    }
  });

  it("prunes entries before the cutoff and leaves entries at or after it", () => {
    const clock = new ManualClock(100);
    const db = openDatabase(":memory:", {
      migrations: autonomyMigrations,
    });
    const repository = new AutonomyWakesRepository({
      db,
      clock,
    });

    try {
      const oldWake = repository.record({
        trigger_name: "scheduled_reflection",
        condition_name: null,
        session_id: DEFAULT_SESSION_ID,
        wake_source_type: "trigger",
      });
      clock.set(200);
      const boundaryWake = repository.record({
        trigger_name: "scheduled_reflection",
        condition_name: null,
        session_id: DEFAULT_SESSION_ID,
        wake_source_type: "trigger",
      });
      clock.set(300);
      const newWake = repository.record({
        trigger_name: "goal_followup_due",
        condition_name: null,
        session_id: DEFAULT_SESSION_ID,
        wake_source_type: "trigger",
      });

      expect(repository.prune(200)).toBe(1);
      const wakeIds = repository.listSince(0, 10).map((wake) => wake.id);
      expect(wakeIds).not.toContain(oldWake.id);
      expect(wakeIds).toContain(boundaryWake.id);
      expect(wakeIds).toContain(newWake.id);
    } finally {
      db.close();
    }
  });

  it("retains multiple records with the same timestamp", () => {
    const clock = new ManualClock(1_000);
    const db = openDatabase(":memory:", {
      migrations: autonomyMigrations,
    });
    const repository = new AutonomyWakesRepository({
      db,
      clock,
    });

    try {
      repository.record({
        trigger_name: "scheduled_reflection",
        condition_name: null,
        session_id: DEFAULT_SESSION_ID,
        wake_source_type: "trigger",
      });
      repository.record({
        trigger_name: "goal_followup_due",
        condition_name: null,
        session_id: DEFAULT_SESSION_ID,
        wake_source_type: "trigger",
      });
      repository.record({
        trigger_name: "open_question_urgency_bump",
        condition_name: "open_question_urgency_bump",
        session_id: DEFAULT_SESSION_ID,
        wake_source_type: "condition",
      });

      const wakes = repository.listSince(1_000, 10);
      expect(wakes).toHaveLength(3);
      expect(new Set(wakes.map((wake) => wake.id)).size).toBe(3);
      expect(wakes.every((wake) => wake.ts === 1_000)).toBe(true);
    } finally {
      db.close();
    }
  });

  it("accepts every declared wake source name (CHECK stays in sync with the name lists)", () => {
    const clock = new ManualClock(1_000);
    const db = openDatabase(":memory:", {
      migrations: autonomyMigrations,
    });
    const repository = new AutonomyWakesRepository({ db, clock });

    try {
      for (const name of AUTONOMY_WAKE_SOURCE_NAMES) {
        const isCondition = (AUTONOMY_CONDITION_NAMES as readonly string[]).includes(name);
        expect(() =>
          repository.record({
            trigger_name: name,
            condition_name: isCondition
              ? (name as (typeof AUTONOMY_CONDITION_NAMES)[number])
              : null,
            session_id: DEFAULT_SESSION_ID,
            wake_source_type: isCondition ? "condition" : "trigger",
          }),
        ).not.toThrow();
      }
    } finally {
      db.close();
    }
  });
});
