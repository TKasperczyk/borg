import { describe, expect, it } from "vitest";

import { openDatabase } from "../storage/sqlite/index.js";
import { ManualClock } from "../util/clock.js";
import { DEFAULT_SESSION_ID } from "../util/ids.js";

import { autonomyMigrations } from "./migrations.js";
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
      expect(repository.listSince(0, 10).map((wake) => wake.trigger_name)).toEqual([
        "commitment_revoked",
        "scheduled_reflection",
      ]);
    } finally {
      db.close();
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
});
