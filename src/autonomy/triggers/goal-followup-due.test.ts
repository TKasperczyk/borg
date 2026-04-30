import { afterEach, describe, expect, it } from "vitest";

import { createOfflineTestHarness } from "../../offline/test-support.js";
import { StreamWatermarkRepository } from "../../stream/index.js";
import { ManualClock } from "../../util/clock.js";

import { createGoalFollowupDueTrigger } from "./goal-followup-due.js";

describe("goal followup due trigger", () => {
  let cleanup: (() => Promise<void>) | undefined;

  afterEach(async () => {
    await cleanup?.();
    cleanup = undefined;
  });

  it("fires for deadline-approaching goals", async () => {
    const clock = new ManualClock(1_000_000);
    const harness = await createOfflineTestHarness({ clock });
    cleanup = harness.cleanup;
    const watermarkRepository = new StreamWatermarkRepository({
      db: harness.db,
      clock,
    });

    const goal = harness.goalsRepository.add({
      description: "Ship Sprint 11",
      priority: 9,
      provenance: { kind: "manual" },
      targetAt: clock.now() + 10_000,
    });
    const trigger = createGoalFollowupDueTrigger({
      goalsRepository: harness.goalsRepository,
      watermarkRepository,
      lookaheadMs: 20_000,
      staleMs: 14 * 24 * 60 * 60 * 1_000,
      clock,
    });

    const events = await trigger.scan();
    expect(events).toHaveLength(1);
    expect(events[0]?.payload).toMatchObject({
      goal_id: goal.id,
      reason: "deadline",
      target_at: clock.now() + 10_000,
    });
  });

  it("fires for stale goals with no recent progress", async () => {
    const clock = new ManualClock(2_000_000);
    const harness = await createOfflineTestHarness({ clock });
    cleanup = harness.cleanup;
    const watermarkRepository = new StreamWatermarkRepository({
      db: harness.db,
      clock,
    });

    const goal = harness.goalsRepository.add({
      description: "Write the autonomy tests",
      priority: 7,
      provenance: { kind: "manual" },
      createdAt: clock.now() - 20 * 24 * 60 * 60 * 1_000,
    });
    const trigger = createGoalFollowupDueTrigger({
      goalsRepository: harness.goalsRepository,
      watermarkRepository,
      lookaheadMs: 7 * 24 * 60 * 60 * 1_000,
      staleMs: 14 * 24 * 60 * 60 * 1_000,
      clock,
    });

    const events = await trigger.scan();
    expect(events).toHaveLength(1);
    expect(events[0]?.payload).toMatchObject({
      goal_id: goal.id,
      reason: "stale",
      last_progress_ts: null,
    });
    expect(events[0]?.payload.days_stale).toBe(20);
  });

  it("dedupes a combined event once and re-fires after the target changes", async () => {
    const clock = new ManualClock(3_000_000);
    const harness = await createOfflineTestHarness({ clock });
    cleanup = harness.cleanup;
    const watermarkRepository = new StreamWatermarkRepository({
      db: harness.db,
      clock,
    });

    const goal = harness.goalsRepository.add({
      description: "Keep the goal loop alive",
      priority: 10,
      progressNotes: "Started the work.",
      provenance: { kind: "manual" },
      createdAt: clock.now() - 21 * 24 * 60 * 60 * 1_000,
      targetAt: clock.now() + 10_000,
    });
    const trigger = createGoalFollowupDueTrigger({
      goalsRepository: harness.goalsRepository,
      watermarkRepository,
      lookaheadMs: 20_000,
      staleMs: 14 * 24 * 60 * 60 * 1_000,
      clock,
    });

    const firstScan = await trigger.scan();
    expect(firstScan).toHaveLength(1);
    expect(firstScan[0]?.payload.reason).toBe("both");

    watermarkRepository.set(firstScan[0]!.watermarkProcessName, "default" as never, {
      lastTs: clock.now(),
      lastEntryId: "watermark",
    });
    expect(await trigger.scan()).toEqual([]);

    harness.goalsRepository.update(
      goal.id,
      {
        target_at: clock.now() + 60 * 24 * 60 * 60 * 1_000,
      },
      { kind: "manual" },
    );

    const secondScan = await trigger.scan();
    expect(secondScan).toHaveLength(1);
    expect(secondScan[0]?.payload.reason).toBe("stale");
  });
});
