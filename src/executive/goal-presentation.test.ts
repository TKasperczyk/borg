import { mkdtempSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { expect, it } from "vitest";
import { createMigrations } from "../borg/storage-setup.js";
import { EntityRepository } from "../memory/commitments/index.js";
import { GoalsRepository, selfMigrations } from "../memory/self/index.js";
import { StreamEntryIndexRepository, StreamWriter } from "../stream/index.js";
import { composeMigrations, openDatabase } from "../storage/sqlite/index.js";
import { StreamWatermarkRepository, streamWatermarkMigrations } from "../stream/watermark.js";
import { ManualClock } from "../util/clock.js";
import { createStreamEntryId, DEFAULT_SESSION_ID } from "../util/ids.js";
import { GoalPresentationStore } from "./goal-presentation.js";

it("persists generations across restart and reopens on explicit goal mutations or associated input", () => {
  const dir = mkdtempSync(join(tmpdir(), "borg-goal-presentation-"));
  const path = join(dir, "test.db");
  const migrations = composeMigrations(selfMigrations, streamWatermarkMigrations);
  const clock = new ManualClock(1_000);
  let db = openDatabase(path, { migrations });
  try {
    let watermarks = new StreamWatermarkRepository({ db, clock });
    let store = new GoalPresentationStore(watermarks);
    let goals = new GoalsRepository({ db, clock, watermarkRepository: watermarks });
    const first = goals.add({ description: "目的", priority: 5, provenance: { kind: "manual" } });
    const second = goals.add({
      description: "Drugie",
      priority: 4,
      provenance: { kind: "manual" },
    });
    const omitted = goals.add({
      description: "Not rendered",
      priority: 3,
      provenance: { kind: "manual" },
    });
    const snapshots = [first, second].map((goal) => store.snapshot(goal));
    expect(store.wasPresented(first)).toBe(false); // Preparation is read-only.
    store.presented(snapshots, "turn_silent_or_blocked");
    expect(store.wasPresented(omitted)).toBe(false);
    expect(store.hasNewInput(omitted)).toBe(false);
    store.associateInbound(omitted.id, createStreamEntryId());
    expect(store.hasNewInput(omitted)).toBe(true);
    goals.block(
      first.id,
      {
        blocker: { kind: "until", until: 2_000 },
        attempt_status: "attempted_unavailable",
        reason: "待機",
      },
      { kind: "manual" },
    );
    db.close();
    db = openDatabase(path, { migrations });
    watermarks = new StreamWatermarkRepository({ db, clock });
    store = new GoalPresentationStore(watermarks);
    goals = new GoalsRepository({ db, clock, watermarkRepository: watermarks });
    expect(store.wasPresented(goals.get(first.id)!)).toBe(true); // Blocking is an attempt, not input.
    expect(store.wasPresented(goals.get(second.id)!)).toBe(true);
    clock.advance(100_000);
    goals.reconcileBlocks();
    goals.update(
      first.id,
      { priority: 10, description: "Changed topology/score" },
      { kind: "manual" },
    );
    expect(store.wasPresented(goals.get(first.id)!)).toBe(false);
    goals.updateProgress(first.id, "新しい進捗", { kind: "manual" });
    const progressed = goals.get(first.id)!;
    expect(store.wasPresented(progressed)).toBe(false);
    store.presented([store.snapshot(progressed)], "turn_progress");
    // Two writes at an identical clock timestamp still create two generations.
    goals.updateProgress(first.id, "次の進捗", { kind: "manual" });
    expect(store.wasPresented(goals.get(first.id)!)).toBe(false);
    const beforeNewInput = store.snapshot(goals.get(first.id)!);
    const inbound = createStreamEntryId();
    store.associateInbound(first.id, inbound);
    store.presented([beforeNewInput], "turn_old_prompt");
    expect(store.wasPresented(goals.get(first.id)!)).toBe(false);
    store.presented([store.snapshot(goals.get(first.id)!)], "turn_input");
    store.associateInbound(first.id, inbound);
    expect(store.wasPresented(goals.get(first.id)!)).toBe(true);
    expect(store.wasPresented(goals.get(second.id)!)).toBe(true);
  } finally {
    db.close();
    rmSync(dir, { recursive: true, force: true });
  }
});

it("retains the newest 16 inputs, itemizes each goal write, and never relogs an evicted association", () => {
  const db = openDatabase(":memory:", {
    migrations: composeMigrations(selfMigrations, streamWatermarkMigrations),
  });
  const clock = new ManualClock(1_000);
  try {
    const watermarks = new StreamWatermarkRepository({ db, clock });
    const goals = new GoalsRepository({ db, clock, watermarkRepository: watermarks });
    const store = goals.presentation!;
    const goal = goals.add({ description: "目的", priority: 1, provenance: { kind: "manual" } });
    const inbound = createStreamEntryId();
    store.associateInbound(goal.id, inbound);
    store.associateInbound(goal.id, inbound);
    expect(store.admission(goals.get(goal.id)!, "executive").new_inputs).toEqual([
      { generation: 1, kind: "associated_inbound", entry_id: inbound },
    ]);
    for (let index = 0; index < 15; index++) {
      goals.updateProgress(goal.id, `進捗 ${index}`, { kind: "manual" });
    }
    const sharedSnapshot = store.snapshot(goals.get(goal.id)!);
    store.presented([sharedSnapshot], "shared");
    goals.update(goal.id, { priority: 2 }, { kind: "manual" });
    goals.updateStatus(goal.id, "done", { kind: "manual" });
    expect(store.admission(goals.get(goal.id)!, "followup")).toEqual({
      basis: "new_input",
      new_inputs: [
        { generation: 17, kind: "goal_write", write: "update", record_version: 17 },
        { generation: 18, kind: "goal_write", write: "update_status", record_version: 18 },
      ],
      unitemized_input_count: 0,
    });
    const admission = store.admission(goals.get(goal.id)!, "executive");
    expect(admission.new_inputs).toHaveLength(16);
    expect(admission.new_inputs[0]).toEqual({
      generation: 3,
      kind: "goal_write",
      write: "update_progress",
      record_version: 3,
    });
    expect(admission.unitemized_input_count).toBe(2);
    store.associateInbound(goal.id, inbound);
    expect(store.admission(goals.get(goal.id)!, "executive")).toEqual(admission);
    // A delayed acknowledgement cannot consume inputs that were not shown.
    store.presented([sharedSnapshot], "executive", new Set([goal.id]));
    expect(store.admission(goals.get(goal.id)!, "executive")).toEqual(
      store.admission(goals.get(goal.id)!, "followup"),
    );
  } finally {
    db.close();
  }
});

it("reads legacy generations as unitemized without backfilling or guessing their causes", () => {
  const db = openDatabase(":memory:", {
    migrations: composeMigrations(selfMigrations, streamWatermarkMigrations),
  });
  try {
    const watermarks = new StreamWatermarkRepository({ db });
    const goals = new GoalsRepository({ db, watermarkRepository: watermarks });
    const store = goals.presentation!;
    const goal = goals.add({ description: "目的", priority: 1, provenance: { kind: "manual" } });
    const key = `autonomy:goal-presentation:${goal.id}`;
    watermarks.set(key, DEFAULT_SESSION_ID, {
      lastTs: 7,
      lastEntryId: goal.id,
      metadata: {
        version: 1,
        inputGeneration: 7,
        lastProgressVersion: 8,
        presented: { inputGeneration: 3, progressAnchor: goal.created_at, turnId: "legacy" },
      },
    });
    const stored = watermarks.get(key, DEFAULT_SESSION_ID);
    expect(store.admission(goals.get(goal.id)!, "followup")).toEqual({
      basis: "new_input",
      new_inputs: [],
      unitemized_input_count: 4,
    });
    expect(store.admission(goals.get(goal.id)!, "executive")).toEqual({
      basis: "new_input",
      new_inputs: [],
      unitemized_input_count: 7,
    });
    expect(watermarks.get(key, DEFAULT_SESSION_ID)).toEqual(stored);
    store.associateInbound(goal.id, createStreamEntryId());
    expect(store.admission(goals.get(goal.id)!, "followup").unitemized_input_count).toBe(4);
  } finally {
    db.close();
  }
});

it("reports anchor-only admission in both lanes, including an executive acknowledgement seeded from the shared latch", () => {
  const db = openDatabase(":memory:", {
    migrations: composeMigrations(selfMigrations, streamWatermarkMigrations),
  });
  const clock = new ManualClock(1_000);
  try {
    const watermarks = new StreamWatermarkRepository({ db, clock });
    const goals = new GoalsRepository({ db, clock, watermarkRepository: watermarks });
    const store = goals.presentation!;
    const goal = goals.add({ description: "目的", priority: 1, provenance: { kind: "manual" } });
    for (const lane of ["executive", "followup"] as const) {
      expect(store.admission(goal, lane)).toEqual({
        basis: "first_presentation",
        new_inputs: [],
        unitemized_input_count: 0,
      });
    }
    store.presented([store.snapshot(goal)], "shared");
    const baseline = createMigrations().find((migration) => migration.id === 15_000_003)!;
    expect(baseline.name).toBe("executive_goal_presentation_baseline");
    db.exec(baseline.up as string);
    expect(store.wasExecutivePresented(goal)).toBe(true);
    // Restore is an existing write path that changes the anchor without
    // advancing inputGeneration. The migration also preserves old anchors.
    const restored = goals.restore({ ...goal, last_progress_ts: 1_500 });
    expect(store.snapshot(restored).inputGeneration).toBe(0);
    expect(store.wasPresented(restored)).toBe(false);
    expect(store.wasExecutivePresented(restored)).toBe(false);
    for (const lane of ["executive", "followup"] as const) {
      expect(store.admission(restored, lane)).toEqual({
        basis: "new_input",
        new_inputs: [],
        unitemized_input_count: 0,
        previous_progress_anchor: 1_000,
        current_progress_anchor: 1_500,
      });
    }
    store.presented([store.snapshot(restored)], "shared-updated");
    expect(store.admission(restored, "followup")).not.toHaveProperty("previous_progress_anchor");
    expect(store.admission(restored, "executive")).toMatchObject({
      previous_progress_anchor: 1_000,
      current_progress_anchor: 1_500,
    });
    store.presented([store.snapshot(restored)], "executive-updated", new Set([goal.id]));
    expect(store.admission(restored, "executive")).not.toHaveProperty("previous_progress_anchor");
    clock.advance(1_000);
    goals.updateProgress(goal.id, "進捗", { kind: "manual" });
    const progressed = goals.get(goal.id)!;
    for (const lane of ["executive", "followup"] as const) {
      expect(store.admission(progressed, lane)).toEqual({
        basis: "new_input",
        new_inputs: [
          { generation: 1, kind: "goal_write", write: "update_progress", record_version: 2 },
        ],
        unitemized_input_count: 0,
        previous_progress_anchor: 1_500,
        current_progress_anchor: 2_000,
      });
    }
  } finally {
    db.close();
  }
});

it.each(["until", "entity", "done", "superseded", "abandoned", "manual"] as const)(
  "records the %s unblock basis once while blocking only stamps interaction",
  async (cause) => {
    const dataDir = mkdtempSync(join(tmpdir(), "borg-admission-unblock-"));
    const db = openDatabase(":memory:", { migrations: createMigrations() });
    const clock = new ManualClock(1_000);
    const writer = new StreamWriter({
      dataDir,
      clock,
      entryIndex: new StreamEntryIndexRepository({ db, dataDir }),
    });
    try {
      const goals = new GoalsRepository({
        db,
        clock,
        watermarkRepository: new StreamWatermarkRepository({ db, clock }),
      });
      const store = goals.presentation!;
      const goal = goals.add({ description: "目的", priority: 1, provenance: { kind: "manual" } });
      const dependency = goals.add({
        description: "依存",
        priority: 1,
        provenance: { kind: "manual" },
      });
      const entity = new EntityRepository({ db, clock }).resolve("李", { kind: "person" });
      store.presented([store.snapshot(goal)], "shown", new Set([goal.id]));
      clock.advance(100);
      goals.block(
        goal.id,
        {
          blocker:
            cause === "entity"
              ? { kind: "entity", entity_id: entity }
              : cause === "until" || cause === "manual"
                ? { kind: "until", until: 2_000 }
                : { kind: "goal", goal_id: dependency.id },
          attempt_status: "attempted_unavailable",
          reason: "待機",
        },
        { kind: "manual" },
      );
      expect(store.snapshot(goal).inputGeneration).toBe(0);
      expect(store.wasPresented(goals.get(goal.id)!)).toBe(true);
      expect(store.wasExecutivePresented(goals.get(goal.id)!)).toBe(true);
      expect(goals.get(goal.id)?.last_interaction_at).toBe(1_100);
      clock.advance(400);
      let basis;
      if (cause === "until") {
        clock.advance(5_000);
        basis = { ended_by: "until_elapsed", until: 2_000, blocked_at: 1_100 };
      } else if (cause === "entity") {
        const entry = await writer.append({
          kind: "user_msg",
          sender_entity_id: entity,
          content: "戻りました",
        });
        basis = { ended_by: "entity_inbound", entity_id: entity, entry_id: entry.id };
      } else if (cause === "manual") {
        // Deliberately resembles another cause; the prose must not determine the basis.
        goals.unblock(goal.id, "until timestamp 2000 passed", { kind: "manual" });
        basis = { ended_by: "manual" };
      } else {
        goals.updateStatus(dependency.id, cause, { kind: "manual" });
        basis = { ended_by: "goal_terminal", goal_id: dependency.id, status: cause };
      }
      goals.reconcileBlocks();
      goals.reconcileBlocks();
      const unblocked = goals.get(goal.id)!;
      expect(unblocked.last_interaction_at).toBe(cause === "until" ? 2_000 : 1_500);
      for (const lane of ["executive", "followup"] as const) {
        expect(store.admission(goals.get(goal.id)!, lane)).toEqual({
          basis: "new_input",
          new_inputs: [
            {
              generation: 1,
              kind: "block_ended",
              record_version: 3,
              ...basis,
              disclosure_label: unblocked.block_history![0]!.disclosure_label,
            },
          ],
          unitemized_input_count: 0,
        });
      }
    } finally {
      writer.close();
      db.close();
      rmSync(dataDir, { recursive: true, force: true });
    }
  },
);

it("owns executive acknowledgements, deduplicates planner/finalizer and persists interaction without replay or rendering refresh", () => {
  const db = openDatabase(":memory:", {
    migrations: composeMigrations(selfMigrations, streamWatermarkMigrations),
  });
  const clock = new ManualClock(1_000);
  try {
    const watermarks = new StreamWatermarkRepository({ db, clock });
    const goals = new GoalsRepository({ db, clock, watermarkRepository: watermarks });
    const store = goals.presentation!;
    const goal = goals.add({ description: "Opaque", priority: 1, provenance: { kind: "manual" } });
    const shown = [store.snapshot(goal)];
    store.presented(shown, "reflection");
    expect(store.wasExecutivePresented(goal)).toBe(false);
    expect(goals.get(goal.id)?.last_interaction_at).toBeNull();
    store.presented(shown, "executive-1", new Set([goal.id]));
    expect(store.wasExecutivePresented(goal)).toBe(true);
    const key = `autonomy:goal-presentation:executive:${goal.id}`;
    const firstAcknowledgement = watermarks.get(key, DEFAULT_SESSION_ID);
    clock.advance(500);
    store.presented(shown, "executive-1", new Set([goal.id]));
    expect(store.wasExecutivePresented(goal)).toBe(true);
    expect(watermarks.get(key, DEFAULT_SESSION_ID)).toEqual(firstAcknowledgement);
    store.presented(shown, "followup");
    expect(store.wasExecutivePresented(goal)).toBe(true);
    expect(watermarks.get(key, DEFAULT_SESSION_ID)).toEqual(firstAcknowledgement);
    store.presented(shown, "executive-2", new Set([goal.id]));
    expect(store.wasExecutivePresented(goal)).toBe(true);
    const inbound = createStreamEntryId();
    store.associateInbound(goal.id, inbound);
    expect(goals.get(goal.id)?.last_interaction_at).toBe(1_500);
    expect(store.wasExecutivePresented(goal)).toBe(false);
    clock.advance(500);
    store.associateInbound(goal.id, inbound);
    expect(goals.get(goal.id)?.last_interaction_at).toBe(1_500);
    const current = goals.get(goal.id)!;
    store.presented([store.snapshot(current)], "new-input");
    expect(() =>
      goals.update(goal.id, { priority: 4 }, { kind: "manual" }, { expectedVersion: 999 }),
    ).toThrow();
    expect(store.wasPresented(goals.get(goal.id)!)).toBe(true);
    expect(goals.get(goal.id)?.last_interaction_at).toBe(1_500);
    goals.update(goal.id, {}, { kind: "manual" });
    expect(store.wasPresented(goals.get(goal.id)!)).toBe(true);
    expect(goals.get(goal.id)?.last_interaction_at).toBe(1_500);
    goals.updateStatus(goal.id, "done", { kind: "manual" });
    expect(store.wasPresented(goals.get(goal.id)!)).toBe(false);
    expect(goals.get(goal.id)?.last_interaction_at).toBe(2_000);
    goals.restore(current);
    expect(goals.get(goal.id)?.last_interaction_at).toBe(1_500);
    goals.block(
      goal.id,
      {
        blocker: { kind: "until", until: 3_000 },
        attempt_status: "attempted_unavailable",
        reason: "Waiting",
      },
      { kind: "manual" },
    );
    clock.advance(4_000_000_000);
    goals.reconcileBlocks();
    expect(goals.get(goal.id)?.last_interaction_at).toBe(3_000);
  } finally {
    db.close();
  }
});
