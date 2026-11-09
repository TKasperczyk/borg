import { mkdtempSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { expect, it } from "vitest";
import { GoalsRepository, selfMigrations } from "../memory/self/index.js";
import { composeMigrations, openDatabase } from "../storage/sqlite/index.js";
import { StreamWatermarkRepository, streamWatermarkMigrations } from "../stream/watermark.js";
import { ManualClock } from "../util/clock.js";
import { createStreamEntryId } from "../util/ids.js";
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
    expect(store.wasPresented(goals.get(first.id)!)).toBe(false);
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
    expect(store.executivePresentedAt(goal)).toBeNull();
    expect(goals.get(goal.id)?.last_interaction_at).toBeNull();
    store.presented(shown, "executive-1", new Set([goal.id]));
    expect(store.executivePresentedAt(goal)).toBe(1_000);
    clock.advance(500);
    store.presented(shown, "executive-1", new Set([goal.id]));
    expect(store.executivePresentedAt(goal)).toBe(1_000);
    store.presented(shown, "followup");
    expect(store.executivePresentedAt(goal)).toBe(1_000);
    store.presented(shown, "executive-2", new Set([goal.id]));
    expect(store.executivePresentedAt(goal)).toBe(1_500);
    const inbound = createStreamEntryId();
    store.associateInbound(goal.id, inbound);
    expect(goals.get(goal.id)?.last_interaction_at).toBe(1_500);
    expect(store.executivePresentedAt(goal)).toBeNull();
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
