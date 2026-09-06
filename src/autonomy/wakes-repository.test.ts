import { mkdtempSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";

import { describe, expect, it } from "vitest";

import { openDatabase } from "../storage/sqlite/index.js";
import { ManualClock } from "../util/clock.js";
import { DEFAULT_SESSION_ID, parseGoalId } from "../util/ids.js";

import { autonomyMigrations } from "./migrations.js";
import {
  AUTONOMY_CONDITION_NAMES,
  AUTONOMY_WAKE_OUTCOMES,
  AUTONOMY_WAKE_SOURCE_NAMES,
} from "./types.js";
import {
  AUTONOMY_WAKE_OUTCOME_DETAIL_MAX_LENGTH,
  AUTONOMY_WAKE_STARTUP_INTERRUPTED_DETAIL,
  AUTONOMY_WAKE_STARTUP_INTERRUPTED_GRACE_MS,
  AutonomyWakesRepository,
} from "./wakes-repository.js";

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
    const selectedGoalId = parseGoalId("goal_aaaaaaaaaaaaaaaa");

    try {
      const headwayWake = repository.record({
        trigger_name: "goal_followup_due",
        session_id: DEFAULT_SESSION_ID,
        wake_source_type: "trigger",
        selected_goal_id: selectedGoalId,
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
          expect.objectContaining({
            id: headwayWake.id,
            outcome: "headway",
            selected_goal_id: selectedGoalId,
          }),
          expect.objectContaining({ id: legacyNullWake.id, outcome: null }),
        ]),
      );
    } finally {
      db.close();
    }
  });

  it("reconciles only old orphaned wakes and never overwrites a terminal outcome", () => {
    const clock = new ManualClock(1_000);
    const db = openDatabase(":memory:", { migrations: autonomyMigrations });
    const repository = new AutonomyWakesRepository({ db, clock });

    try {
      const firstOrphan = repository.record({
        trigger_name: "goal_followup_due",
        session_id: DEFAULT_SESSION_ID,
        wake_source_type: "trigger",
      });
      const secondOrphan = repository.record({
        trigger_name: "scheduled_reflection",
        session_id: DEFAULT_SESSION_ID,
        wake_source_type: "trigger",
        source_category: "contemplative",
      });
      clock.advance(AUTONOMY_WAKE_STARTUP_INTERRUPTED_GRACE_MS + 1);
      const recentInFlight = repository.record({
        trigger_name: "goal_followup_due",
        session_id: DEFAULT_SESSION_ID,
        wake_source_type: "trigger",
      });
      const completed = repository.record({
        trigger_name: "scheduled_wake",
        session_id: DEFAULT_SESSION_ID,
        wake_source_type: "trigger",
        source_category: "contemplative",
      });
      repository.recordOutcome(completed.id, "headway");

      expect(repository.interruptOrphanedWakesAtStartup()).toBe(2);
      expect(repository.interruptOrphanedWakesAtStartup()).toBe(0);
      repository.recordOutcome(completed.id, "interrupted", "must not replace headway");

      const wakes = repository.listSince(0, 10);
      expect(
        wakes.filter((wake) => wake.id === firstOrphan.id || wake.id === secondOrphan.id),
      ).toEqual(
        expect.arrayContaining([
          expect.objectContaining({
            outcome: "interrupted",
            outcome_detail: AUTONOMY_WAKE_STARTUP_INTERRUPTED_DETAIL,
          }),
          expect.objectContaining({
            outcome: "interrupted",
            outcome_detail: AUTONOMY_WAKE_STARTUP_INTERRUPTED_DETAIL,
          }),
        ]),
      );
      expect(wakes.find((wake) => wake.id === completed.id)).toMatchObject({
        outcome: "headway",
        outcome_detail: null,
      });
      expect(wakes.find((wake) => wake.id === recentInFlight.id)).toMatchObject({
        outcome: null,
        outcome_detail: null,
      });
      expect(repository.countSince(0, { outcome: "interrupted" })).toBe(2);
    } finally {
      db.close();
    }
  });

  it("tallies outcome details against the bucket total and names the undetailed remainder", () => {
    const clock = new ManualClock(1_000);
    const db = openDatabase(":memory:", { migrations: autonomyMigrations });
    const repository = new AutonomyWakesRepository({ db, clock });

    try {
      const recordErrored = (
        detail?: string | null,
        trigger: "goal_followup_due" | "scheduled_reflection" = "goal_followup_due",
      ) => {
        clock.advance(1);
        const wake = repository.record({
          trigger_name: trigger,
          session_id: DEFAULT_SESSION_ID,
          wake_source_type: "trigger",
          source_category: trigger === "scheduled_reflection" ? "contemplative" : "operational",
        });
        repository.recordOutcome(wake.id, "error", detail);
        return wake;
      };

      recordErrored("LLMError: Failed to complete Anthropic request");
      recordErrored("LLMError: Failed to complete Anthropic request", "scheduled_reflection");
      recordErrored("Anthropic connection failed after 3 attempts");
      // A pre-detail row: the outcome landed, the reason never did.
      recordErrored(null);
      clock.advance(1);
      const headwayWake = repository.record({
        trigger_name: "scheduled_reflection",
        session_id: DEFAULT_SESSION_ID,
        wake_source_type: "trigger",
        source_category: "contemplative",
      });
      repository.recordOutcome(headwayWake.id, "headway");

      const tally = repository.summarizeOutcomeDetailsSince(0, "error");

      // The bucket total is the same number countSince reports, so the split can
      // be checked against the count it claims to decompose.
      expect(tally.total).toBe(repository.countSince(0, { outcome: "error" }));
      expect(tally.total).toBe(4);
      expect(tally.without_detail).toBe(1);
      // Each reason carries its own trigger split. A detail spread over two
      // triggers arrives from the GROUP BY as two rows and is folded back to one
      // reason, so the count above it stays the detail's count and the render cap
      // still cuts the smallest details rather than the least divided ones.
      expect(tally.reasons).toEqual([
        {
          detail: "LLMError: Failed to complete Anthropic request",
          count: 2,
          triggers: [
            { trigger: "goal_followup_due", count: 1 },
            { trigger: "scheduled_reflection", count: 1 },
          ],
        },
        {
          detail: "Anthropic connection failed after 3 attempts",
          count: 1,
          triggers: [{ trigger: "goal_followup_due", count: 1 }],
        },
      ]);
      // reasons + without_detail always reconciles to total.
      expect(
        tally.reasons.reduce((sum, reason) => sum + reason.count, 0) + tally.without_detail,
      ).toBe(tally.total);
      // The headway row is a different bucket and contributes nothing here.
      expect(repository.summarizeOutcomeDetailsSince(0, "headway")).toEqual({
        total: 1,
        without_detail: 1,
        reasons: [],
      });
      // The window edge applies to the tally exactly as it does to the counts.
      expect(repository.summarizeOutcomeDetailsSince(1_003, "error").total).toBe(2);
    } finally {
      db.close();
    }
  });

  it("positions an outcome's rows against the window's other wakes and the wake before them", () => {
    const clock = new ManualClock(1_000);
    const db = openDatabase(":memory:", { migrations: autonomyMigrations });
    const repository = new AutonomyWakesRepository({ db, clock });

    try {
      const recordWake = (outcome: "error" | "headway") => {
        clock.advance(1);
        const wake = repository.record({
          trigger_name: "goal_followup_due",
          session_id: DEFAULT_SESSION_ID,
          wake_source_type: "trigger",
        });
        repository.recordOutcome(wake.id, outcome);
        return wake;
      };

      // ts 1001..1004: an unbroken run. ts 1005: the success that ends it.
      recordWake("error");
      recordWake("error");
      recordWake("error");
      recordWake("error");
      recordWake("headway");

      // Over the whole table the run is unbroken and nothing precedes it, so the
      // "did this start earlier" question has no evidence rather than a no.
      expect(repository.describeOutcomeSpanSince(0, "error")).toEqual({
        other_outcomes_between: 0,
        extends_before_window: null,
      });

      // A window whose edge falls inside the run sees only its tail. The count
      // inside the window cannot tell that apart from a run that began there --
      // the predecessor read is what does, and it deliberately ignores the edge.
      expect(repository.describeOutcomeSpanSince(1_003, "error")).toEqual({
        other_outcomes_between: 0,
        extends_before_window: true,
      });

      // ts 1006 error, 1007 headway, 1008 error: now the bucket is interleaved,
      // and the wake before its first row is the success at 1005.
      recordWake("error");
      recordWake("headway");
      recordWake("error");

      expect(repository.describeOutcomeSpanSince(1_006, "error")).toEqual({
        other_outcomes_between: 1,
        extends_before_window: false,
      });

      // An empty bucket has no position to report.
      expect(repository.describeOutcomeSpanSince(0, "busy")).toBeNull();
    } finally {
      db.close();
    }
  });

  it("clamps an outcome detail and marks it rather than storing an unbounded error", () => {
    const clock = new ManualClock(1_000);
    const db = openDatabase(":memory:", { migrations: autonomyMigrations });
    const repository = new AutonomyWakesRepository({ db, clock });

    try {
      const wake = repository.record({
        trigger_name: "goal_followup_due",
        session_id: DEFAULT_SESSION_ID,
        wake_source_type: "trigger",
      });
      repository.recordOutcome(wake.id, "error", `x${"y".repeat(1_000)}`);
      clock.advance(1);
      const blankWake = repository.record({
        trigger_name: "goal_followup_due",
        session_id: DEFAULT_SESSION_ID,
        wake_source_type: "trigger",
      });
      repository.recordOutcome(blankWake.id, "error", "   ");

      const stored = repository.listSince(0, 10).find((row) => row.id === wake.id);

      expect(stored?.outcome_detail).toHaveLength(
        AUTONOMY_WAKE_OUTCOME_DETAIL_MAX_LENGTH + "...".length,
      );
      expect(stored?.outcome_detail?.endsWith("...")).toBe(true);
      // A whitespace-only detail is no detail; it must not become a distinct
      // reason that looks like an attributed failure.
      expect(repository.summarizeOutcomeDetailsSince(0, "error").without_detail).toBe(1);
    } finally {
      db.close();
    }
  });

  it("stores every basis in a maximum goal batch without clipping the joined detail", () => {
    const clock = new ManualClock(1_000);
    const db = openDatabase(":memory:", { migrations: autonomyMigrations });
    const repository = new AutonomyWakesRepository({ db, clock });
    const goalIds = [
      "goal_aaaaaaaaaaaaaaaa",
      "goal_bbbbbbbbbbbbbbbb",
      "goal_cccccccccccccccc",
      "goal_dddddddddddddddd",
      "goal_eeeeeeeeeeeeeeee",
    ];
    const bases = goalIds.flatMap((goalId) => [
      `progress recorded on ${goalId}`,
      `goal ${goalId} retired by this turn`,
    ]);

    try {
      const wake = repository.record({
        trigger_name: "goal_followup_due",
        session_id: DEFAULT_SESSION_ID,
        wake_source_type: "trigger",
      });
      repository.recordOutcome(wake.id, "headway", null, bases);

      const stored = repository.listSince(0, 1)[0];
      const joinedBases = bases.join("; ");

      expect(joinedBases.length).toBeGreaterThan(AUTONOMY_WAKE_OUTCOME_DETAIL_MAX_LENGTH);
      expect(stored).toMatchObject({
        outcome: "headway",
        outcome_detail: joinedBases,
        headway_bases: bases,
      });
      expect(repository.summarizeOutcomeDetailsSince(0, "headway").reasons).toEqual([
        {
          detail: joinedBases,
          count: 1,
          triggers: [{ trigger: "goal_followup_due", count: 1 }],
        },
      ]);
      for (const goalId of goalIds) {
        expect(stored?.outcome_detail).toContain(goalId);
      }
    } finally {
      db.close();
    }
  });

  it("applies the complete wake schema chain in deterministic order", () => {
    const db = openDatabase(":memory:", { migrations: autonomyMigrations });

    try {
      expect(db.listAppliedMigrations().map(({ id, name }) => ({ id, name }))).toEqual([
        { id: 1, name: "autonomy_baseline" },
        { id: 2, name: "autonomy_wakes_source_category" },
        { id: 3, name: "autonomy_wakes_outcome" },
        { id: 4, name: "autonomy_wakes_outcome_detail" },
        { id: 5, name: "autonomy_wakes_interrupted_outcome" },
        { id: 6, name: "autonomy_wakes_selected_goal" },
        { id: 7, name: "autonomy_wakes_headway_bases" },
        { id: 8, name: "autonomy_wakes_execution_counts" },
      ]);

      const table = db
        .prepare("SELECT sql FROM sqlite_master WHERE type = 'table' AND name = 'autonomy_wakes'")
        .get() as { sql: string };
      expect(table.sql).toContain(
        "outcome IN ('headway', 'silent', 'error', 'busy', 'interrupted')",
      );

      const columns = db.prepare("PRAGMA table_info(autonomy_wakes)").all() as Array<{
        name: string;
        notnull: number;
      }>;
      expect(columns.find((column) => column.name === "selected_goal_id")).toMatchObject({
        notnull: 0,
      });
      expect(columns.find((column) => column.name === "headway_bases_json")).toMatchObject({
        notnull: 0,
      });
      expect(columns.find((column) => column.name === "finalizer_rounds")).toMatchObject({
        notnull: 0,
      });
      expect(columns.find((column) => column.name === "stall_retries")).toMatchObject({
        notnull: 0,
      });

      const indexes = db
        .prepare(
          `SELECT name, tbl_name AS table_name
           FROM sqlite_master
           WHERE type = 'index' AND sql IS NOT NULL
           ORDER BY name ASC`,
        )
        .all();
      expect(indexes).toEqual([
        { name: "idx_autonomy_wakes_ts", table_name: "autonomy_wakes" },
        { name: "idx_scheduled_wakes_due", table_name: "scheduled_wakes" },
      ]);

      const insert = db.prepare(
        `INSERT INTO autonomy_wakes (
           id, ts, trigger_name, condition_name, session_id, wake_source_type, source_category,
           outcome, outcome_detail, selected_goal_id, headway_bases_json
         ) VALUES (?, ?, 'goal_followup_due', NULL, ?, 'trigger', 'operational', ?, NULL, NULL, NULL)`,
      );
      for (const [index, outcome] of AUTONOMY_WAKE_OUTCOMES.entries()) {
        expect(() =>
          insert.run(
            `autonomy_wake_outcome_${String(index).padStart(8, "0")}`,
            1_000 + index,
            DEFAULT_SESSION_ID,
            outcome,
          ),
        ).not.toThrow();
      }
      expect(() =>
        insert.run("autonomy_wake_outcome_invalid", 2_000, DEFAULT_SESSION_ID, "not_an_outcome"),
      ).toThrow();
    } finally {
      db.close();
    }
  });

  it("applies additive outcome, selected-goal, and headway-basis migrations over legacy rows", () => {
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
      expect(columns.map((column) => column.name)).toContain("selected_goal_id");
      expect(columns.map((column) => column.name)).toContain("headway_bases_json");
      expect(columns.map((column) => column.name)).toContain("finalizer_rounds");
      expect(columns.map((column) => column.name)).toContain("stall_retries");
      expect(repository.listSince(0, 10)).toEqual([
        expect.objectContaining({
          id: "autonomy_wake_aaaaaaaaaaaaaaaa",
          outcome: null,
          selected_goal_id: null,
          headway_bases: null,
          finalizer_rounds: null,
          stall_retries: null,
        }),
      ]);
    } finally {
      db.close();
      rmSync(tempDir, { recursive: true, force: true });
    }
  });

  it("adds nullable execution counts over the existing wake schema without backfilling", () => {
    const tempDir = mkdtempSync(join(tmpdir(), "borg-autonomy-execution-count-migration-"));
    const dbPath = join(tempDir, "borg.db");
    let db = openDatabase(dbPath, { migrations: autonomyMigrations.slice(0, 7) });

    try {
      db.prepare(
        `
          INSERT INTO autonomy_wakes (
            id, ts, trigger_name, condition_name, session_id, wake_source_type, source_category,
            outcome, outcome_detail, selected_goal_id, headway_bases_json
          ) VALUES (?, ?, ?, NULL, ?, 'trigger', 'operational', 'silent', ?, NULL, NULL)
        `,
      ).run(
        "autonomy_wake_aaaaaaaaaaaaaaaa",
        1_000,
        "goal_followup_due",
        DEFAULT_SESSION_ID,
        "deliberate-silence: finalizer_no_output",
      );
      db.close();

      db = openDatabase(dbPath, { migrations: autonomyMigrations });
      const columns = db.prepare("PRAGMA table_info(autonomy_wakes)").all() as Array<{
        name: string;
        type: string;
        notnull: number;
      }>;
      const repository = new AutonomyWakesRepository({ db, clock: new ManualClock(2_000) });

      expect(columns.find((column) => column.name === "finalizer_rounds")).toMatchObject({
        type: "INTEGER",
        notnull: 0,
      });
      expect(columns.find((column) => column.name === "stall_retries")).toMatchObject({
        type: "INTEGER",
        notnull: 0,
      });
      expect(repository.listSince(0, 1)[0]).toMatchObject({
        outcome: "silent",
        finalizer_rounds: null,
        stall_retries: null,
      });
    } finally {
      db.close();
      rmSync(tempDir, { recursive: true, force: true });
    }
  });

  it("rebuilds the outcome CHECK without changing existing outcome details", () => {
    const tempDir = mkdtempSync(join(tmpdir(), "borg-autonomy-interrupted-migration-"));
    const dbPath = join(tempDir, "borg.db");
    let db = openDatabase(dbPath, { migrations: autonomyMigrations.slice(0, 4) });

    try {
      db.prepare(
        `
          INSERT INTO autonomy_wakes (
            id, ts, trigger_name, condition_name, session_id, wake_source_type, source_category,
            outcome, outcome_detail
          ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        `,
      ).run(
        "autonomy_wake_aaaaaaaaaaaaaaaa",
        1_000,
        "goal_followup_due",
        null,
        DEFAULT_SESSION_ID,
        "trigger",
        "operational",
        "error",
        "provider unavailable",
      );
      db.close();

      db = openDatabase(dbPath, { migrations: autonomyMigrations });
      const repository = new AutonomyWakesRepository({ db, clock: new ManualClock(2_000) });
      const preserved = repository.listSince(0, 10)[0];

      expect(preserved).toMatchObject({
        outcome: "error",
        outcome_detail: "provider unavailable",
      });
      const next = repository.record({
        trigger_name: "scheduled_reflection",
        session_id: DEFAULT_SESSION_ID,
        wake_source_type: "trigger",
        source_category: "contemplative",
      });
      expect(() =>
        repository.recordOutcome(next.id, "interrupted", "bookkeeping failed"),
      ).not.toThrow();
      expect(repository.listSince(0, 10).find((wake) => wake.id === next.id)).toMatchObject({
        outcome: "interrupted",
        outcome_detail: "bookkeeping failed",
      });
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
