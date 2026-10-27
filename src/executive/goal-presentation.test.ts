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

it("persists only presented generations across restart, and reopens only on new progress or an explicit inbound association", () => {
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
    expect(store.wasPresented(goals.get(first.id)!)).toBe(true);
    expect(store.wasPresented(goals.get(second.id)!)).toBe(true);
    clock.advance(100_000);
    goals.reconcileBlocks();
    goals.update(
      first.id,
      { priority: 10, description: "Changed topology/score" },
      { kind: "manual" },
    );
    expect(store.wasPresented(goals.get(first.id)!)).toBe(true);
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
