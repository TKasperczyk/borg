import { afterEach, describe, expect, it } from "vitest";

import { createOfflineTestHarness } from "../../offline/test-support.js";
import { StreamWatermarkRepository } from "../../stream/index.js";
import { ManualClock } from "../../util/clock.js";
import { DEFAULT_SESSION_ID } from "../../util/ids.js";
import { formatAutonomyTriggerContext } from "../../cognition/autonomy-trigger.js";
import { getExecutiveFocusGoalStaleBackoffProcessName } from "../executive-focus-stale-backoff.js";

import { createGoalFollowupDueTrigger } from "./goal-followup-due.js";

const STALE_BACKOFF = {
  baseCooldownMs: 1_000,
  multiplier: 2,
  maxCooldownMs: 60_000,
  dormancyCount: 3,
};

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
      staleBackoff: STALE_BACKOFF,
      respectStaleBackoff: true,
      clock,
    });

    const events = await trigger.scan();
    expect(events).toHaveLength(1);
    expect(events[0]?.payload).toMatchObject({
      goal_id: goal.id,
      selected_goal_id: goal.id,
      selected_goal: {
        id: goal.id,
        description: goal.description,
      },
      reason: "deadline",
      target_at: clock.now() + 10_000,
    });
    expect(events[0]?.stateTs).toBe(goal.created_at);
  });

  it("describes the next deadline or staleness threshold without firing", async () => {
    const clock = new ManualClock(1_000_000);
    const harness = await createOfflineTestHarness({ clock });
    cleanup = harness.cleanup;
    const watermarkRepository = new StreamWatermarkRepository({
      db: harness.db,
      clock,
    });

    harness.goalsRepository.add({
      description: "Future deadline",
      priority: 9,
      provenance: { kind: "manual" },
      targetAt: clock.now() + 120_000,
    });
    harness.goalsRepository.add({
      description: "Future stale goal",
      priority: 7,
      provenance: { kind: "manual" },
      createdAt: clock.now() - 20_000,
    });
    const trigger = createGoalFollowupDueTrigger({
      goalsRepository: harness.goalsRepository,
      watermarkRepository,
      lookaheadMs: 20_000,
      staleMs: 100_000,
      staleBackoff: STALE_BACKOFF,
      respectStaleBackoff: true,
      clock,
    });

    await expect(trigger.nextDueAt!()).resolves.toBe(clock.now() + 80_001);

    clock.advance(90_000);
    await expect(trigger.nextDueAt!()).resolves.toBe(clock.now());
  });

  it("returns null instead of scanning beyond the observability candidate cap", async () => {
    const clock = new ManualClock(1_000_000);
    const harness = await createOfflineTestHarness({ clock });
    cleanup = harness.cleanup;
    const watermarkRepository = new StreamWatermarkRepository({
      db: harness.db,
      clock,
    });

    for (let index = 0; index < 513; index += 1) {
      harness.goalsRepository.add({
        description: `Bounded follow-up goal ${index}`,
        priority: 1,
        provenance: { kind: "manual" },
        targetAt: clock.now() + 120_000 + index,
      });
    }
    const trigger = createGoalFollowupDueTrigger({
      goalsRepository: harness.goalsRepository,
      watermarkRepository,
      lookaheadMs: 20_000,
      staleMs: 100_000,
      staleBackoff: STALE_BACKOFF,
      respectStaleBackoff: true,
      clock,
    });

    await expect(trigger.nextDueAt!()).resolves.toBeNull();
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
      staleBackoff: STALE_BACKOFF,
      respectStaleBackoff: true,
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

  it("attaches the existing executive score order outside the model-facing payload", async () => {
    const clock = new ManualClock(2_100_000_000);
    const harness = await createOfflineTestHarness({ clock });
    cleanup = harness.cleanup;
    const watermarkRepository = new StreamWatermarkRepository({ db: harness.db, clock });
    const topGoal = harness.goalsRepository.add({
      description: "High-priority deadline goal",
      priority: 10,
      provenance: { kind: "manual" },
      createdAt: clock.now() - 200_000,
      targetAt: clock.now() + 1_000,
    });
    const otherGoal = harness.goalsRepository.add({
      description: "Lower-priority stale goal",
      priority: 1,
      provenance: { kind: "manual" },
      createdAt: clock.now() - 200_000,
    });
    const trigger = createGoalFollowupDueTrigger({
      goalsRepository: harness.goalsRepository,
      watermarkRepository,
      lookaheadMs: 20_000,
      staleMs: 100_000,
      staleBackoff: STALE_BACKOFF,
      respectStaleBackoff: true,
      executiveScoring: {
        embeddingClient: harness.embeddingClient,
        threshold: 0.45,
        deadlineLookaheadMs: 20_000,
        staleMs: 100_000,
      },
      clock,
    });

    const events = await trigger.scan();
    const topEvent = events.find((event) => event.payload.goal_id === topGoal.id);
    const otherEvent = events.find((event) => event.payload.goal_id === otherGoal.id);

    expect(topEvent?.executiveGoalRank).toBe(0);
    expect(otherEvent?.executiveGoalRank).toBe(1);
    expect(topEvent?.executiveGoalScore).toMatchObject({
      goal_id: topGoal.id,
      components: {
        priority: 1,
        deadline_pressure: expect.any(Number),
        context_fit: expect.any(Number),
        progress_debt: expect.any(Number),
      },
    });
    expect(topEvent?.payload).not.toHaveProperty("executiveGoalScore");
    expect(topEvent?.payload).not.toHaveProperty("executiveGoalRank");
  });

  it("excludes dormant goals from scan and nextDueAt", async () => {
    const clock = new ManualClock(2_500_000);
    const harness = await createOfflineTestHarness({ clock });
    cleanup = harness.cleanup;
    const watermarkRepository = new StreamWatermarkRepository({ db: harness.db, clock });
    const goal = harness.goalsRepository.add({
      description: "Dormant settled goal",
      priority: 7,
      provenance: { kind: "manual" },
      createdAt: clock.now() - 200_000,
    });
    watermarkRepository.set(
      getExecutiveFocusGoalStaleBackoffProcessName(goal.id),
      DEFAULT_SESSION_ID,
      {
        lastTs: clock.now(),
        lastEntryId: "empty-wake-3",
        metadata: { empty_count: 3 },
      },
    );
    const trigger = createGoalFollowupDueTrigger({
      goalsRepository: harness.goalsRepository,
      watermarkRepository,
      lookaheadMs: 20_000,
      staleMs: 100_000,
      staleBackoff: STALE_BACKOFF,
      respectStaleBackoff: true,
      clock,
    });

    await expect(trigger.scan()).resolves.toEqual([]);
    await expect(trigger.nextDueAt!()).resolves.toBeNull();
  });

  it("releases stale followup dormancy for a new structural action key", async () => {
    const clock = new ManualClock(2_550_000);
    const harness = await createOfflineTestHarness({ clock });
    cleanup = harness.cleanup;
    const watermarkRepository = new StreamWatermarkRepository({ db: harness.db, clock });
    const goal = harness.goalsRepository.add({
      description: "Dormant followup with a newly executable action",
      priority: 7,
      provenance: { kind: "manual" },
      createdAt: clock.now() - 200_000,
    });
    const processName = getExecutiveFocusGoalStaleBackoffProcessName(goal.id);
    const actionAvailabilityKey = "outbound_action_surface_v1:test";
    const staleLatchProcessName = `autonomy:goal-followup-due:${goal.id}:no-target:${goal.created_at}:stale`;
    watermarkRepository.set(processName, DEFAULT_SESSION_ID, {
      lastTs: clock.now(),
      lastEntryId: "empty-wake-3",
      metadata: { empty_count: 3 },
    });
    watermarkRepository.set(staleLatchProcessName, DEFAULT_SESSION_ID, {
      lastTs: clock.now(),
      lastEntryId: "pre-fix-stale-latch",
    });
    const trigger = createGoalFollowupDueTrigger({
      goalsRepository: harness.goalsRepository,
      watermarkRepository,
      lookaheadMs: 20_000,
      staleMs: 100_000,
      staleBackoff: STALE_BACKOFF,
      respectStaleBackoff: true,
      goalStaleBackoffActionAvailabilityKey: () => actionAvailabilityKey,
      clock,
    });

    const released = await trigger.scan();
    expect(released).toHaveLength(1);
    expect(released[0]?.watermarkProcessName).toBe(staleLatchProcessName);
    expect(released[0]?.goalStaleBackoffActionAvailabilityKey).toBe(actionAvailabilityKey);
    await expect(trigger.nextDueAt!()).resolves.toBe(clock.now());

    watermarkRepository.set(processName, DEFAULT_SESSION_ID, {
      lastTs: clock.now(),
      lastEntryId: "empty-wake-4",
      metadata: {
        empty_count: 3,
        action_availability_key: actionAvailabilityKey,
      },
    });
    await expect(trigger.scan()).resolves.toEqual([]);
    await expect(trigger.nextDueAt!()).resolves.toBeNull();
  });

  it("defers cooling goals until the shared backoff ends", async () => {
    const clock = new ManualClock(2_600_000);
    const harness = await createOfflineTestHarness({ clock });
    cleanup = harness.cleanup;
    const watermarkRepository = new StreamWatermarkRepository({ db: harness.db, clock });
    const goal = harness.goalsRepository.add({
      description: "Cooling settled goal",
      priority: 7,
      provenance: { kind: "manual" },
      createdAt: clock.now() - 200_000,
    });
    watermarkRepository.set(
      getExecutiveFocusGoalStaleBackoffProcessName(goal.id),
      DEFAULT_SESSION_ID,
      {
        lastTs: clock.now(),
        lastEntryId: "empty-wake-1",
        metadata: { empty_count: 1 },
      },
    );
    const trigger = createGoalFollowupDueTrigger({
      goalsRepository: harness.goalsRepository,
      watermarkRepository,
      lookaheadMs: 20_000,
      staleMs: 100_000,
      staleBackoff: STALE_BACKOFF,
      respectStaleBackoff: true,
      clock,
    });

    await expect(trigger.scan()).resolves.toEqual([]);
    await expect(trigger.nextDueAt!()).resolves.toBe(clock.now() + 2_000);

    clock.advance(2_001);
    await expect(trigger.scan()).resolves.toHaveLength(1);
  });

  it("lazy-clears the shared brake after genuine goal progress", async () => {
    const clock = new ManualClock(2_700_000);
    const harness = await createOfflineTestHarness({ clock });
    cleanup = harness.cleanup;
    const watermarkRepository = new StreamWatermarkRepository({ db: harness.db, clock });
    const goal = harness.goalsRepository.add({
      description: "Goal that later advanced",
      priority: 7,
      provenance: { kind: "manual" },
      createdAt: clock.now() - 200_000,
    });
    watermarkRepository.set(
      getExecutiveFocusGoalStaleBackoffProcessName(goal.id),
      DEFAULT_SESSION_ID,
      {
        lastTs: clock.now(),
        lastEntryId: "empty-wake-3",
        metadata: { empty_count: 3 },
      },
    );
    clock.advance(1);
    harness.goalsRepository.updateProgress(goal.id, "Structural progress", { kind: "manual" });
    clock.advance(200_000);
    const trigger = createGoalFollowupDueTrigger({
      goalsRepository: harness.goalsRepository,
      watermarkRepository,
      lookaheadMs: 20_000,
      staleMs: 100_000,
      staleBackoff: STALE_BACKOFF,
      respectStaleBackoff: true,
      clock,
    });

    const events = await trigger.scan();
    expect(events).toHaveLength(1);
    expect(events[0]?.payload.last_progress_ts).toBe(2_700_001);
  });

  it("lets a deadline-bearing stale goal pierce per-goal dormancy but not its own latch", async () => {
    const clock = new ManualClock(2_800_000);
    const harness = await createOfflineTestHarness({ clock });
    cleanup = harness.cleanup;
    const watermarkRepository = new StreamWatermarkRepository({ db: harness.db, clock });
    const goal = harness.goalsRepository.add({
      description: "Dormant goal with a live deadline",
      priority: 9,
      provenance: { kind: "manual" },
      createdAt: clock.now() - 200_000,
      targetAt: clock.now() + 10_000,
    });
    watermarkRepository.set(
      getExecutiveFocusGoalStaleBackoffProcessName(goal.id),
      DEFAULT_SESSION_ID,
      {
        lastTs: clock.now(),
        lastEntryId: "empty-wake-3",
        metadata: { empty_count: 3 },
      },
    );
    const trigger = createGoalFollowupDueTrigger({
      goalsRepository: harness.goalsRepository,
      watermarkRepository,
      lookaheadMs: 20_000,
      staleMs: 100_000,
      staleBackoff: STALE_BACKOFF,
      respectStaleBackoff: true,
      clock,
    });

    await expect(trigger.nextDueAt!()).resolves.toBe(clock.now());
    const events = await trigger.scan();
    expect(events).toHaveLength(1);
    expect(events[0]?.payload.reason).toBe("both");

    watermarkRepository.set(events[0]!.watermarkProcessName, DEFAULT_SESSION_ID, {
      lastTs: events[0]!.sortTs,
      lastEntryId: events[0]!.id,
    });
    await expect(trigger.scan()).resolves.toEqual([]);
  });

  it("consults stale and deadline phases once each for the same state tuple", async () => {
    const clock = new ManualClock(2_820_000);
    const harness = await createOfflineTestHarness({ clock });
    cleanup = harness.cleanup;
    const watermarkRepository = new StreamWatermarkRepository({ db: harness.db, clock });
    const goal = harness.goalsRepository.add({
      description: "Stale goal approaching its deadline phase",
      priority: 8,
      provenance: { kind: "manual" },
      createdAt: clock.now() - 200_000,
      targetAt: clock.now() + 120_000,
    });
    const trigger = createGoalFollowupDueTrigger({
      goalsRepository: harness.goalsRepository,
      watermarkRepository,
      lookaheadMs: 20_000,
      staleMs: 100_000,
      staleBackoff: STALE_BACKOFF,
      respectStaleBackoff: true,
      clock,
    });

    const staleEvents = await trigger.scan();
    expect(staleEvents).toHaveLength(1);
    expect(staleEvents[0]?.payload.reason).toBe("stale");
    expect(staleEvents[0]?.watermarkProcessName).toMatch(/:stale$/);
    watermarkRepository.set(staleEvents[0]!.watermarkProcessName, DEFAULT_SESSION_ID, {
      lastTs: staleEvents[0]!.sortTs,
      lastEntryId: staleEvents[0]!.id,
    });
    await expect(trigger.scan()).resolves.toEqual([]);

    clock.advance(100_001);
    const deadlineEvents = await trigger.scan();
    expect(deadlineEvents).toHaveLength(1);
    expect(deadlineEvents[0]?.payload).toMatchObject({
      goal_id: goal.id,
      reason: "both",
    });
    expect(deadlineEvents[0]?.watermarkProcessName).toMatch(/:deadline$/);
    expect(deadlineEvents[0]?.watermarkProcessName).not.toBe(staleEvents[0]?.watermarkProcessName);
  });

  it("treats pre-phase latch rows as authoritative for both phases", async () => {
    const clock = new ManualClock(2_830_000);
    const harness = await createOfflineTestHarness({ clock });
    cleanup = harness.cleanup;
    const watermarkRepository = new StreamWatermarkRepository({ db: harness.db, clock });
    const goal = harness.goalsRepository.add({
      description: "Goal already consulted by the legacy latch",
      priority: 8,
      provenance: { kind: "manual" },
      createdAt: clock.now() - 200_000,
      targetAt: clock.now() + 10_000,
    });
    const legacyProcessName = `autonomy:goal-followup-due:${goal.id}:${goal.target_at}:${goal.created_at}`;
    watermarkRepository.set(legacyProcessName, DEFAULT_SESSION_ID, {
      lastTs: clock.now(),
      lastEntryId: "legacy-followup-latch",
    });
    const trigger = createGoalFollowupDueTrigger({
      goalsRepository: harness.goalsRepository,
      watermarkRepository,
      lookaheadMs: 20_000,
      staleMs: 100_000,
      staleBackoff: STALE_BACKOFF,
      respectStaleBackoff: true,
      clock,
    });

    await expect(trigger.scan()).resolves.toEqual([]);
    await expect(trigger.nextDueAt!()).resolves.toBeNull();
  });

  it("describes the future deadline lookahead boundary through dormancy", async () => {
    const clock = new ManualClock(2_850_000);
    const harness = await createOfflineTestHarness({ clock });
    cleanup = harness.cleanup;
    const watermarkRepository = new StreamWatermarkRepository({ db: harness.db, clock });
    const goal = harness.goalsRepository.add({
      description: "Dormant goal with an approaching deadline",
      priority: 9,
      provenance: { kind: "manual" },
      createdAt: clock.now() - 200_000,
      targetAt: clock.now() + 120_000,
    });
    watermarkRepository.set(
      getExecutiveFocusGoalStaleBackoffProcessName(goal.id),
      DEFAULT_SESSION_ID,
      {
        lastTs: clock.now(),
        lastEntryId: "empty-wake-3",
        metadata: { empty_count: 3 },
      },
    );
    const trigger = createGoalFollowupDueTrigger({
      goalsRepository: harness.goalsRepository,
      watermarkRepository,
      lookaheadMs: 20_000,
      staleMs: 100_000,
      staleBackoff: STALE_BACKOFF,
      respectStaleBackoff: true,
      clock,
    });

    await expect(trigger.scan()).resolves.toEqual([]);
    await expect(trigger.nextDueAt!()).resolves.toBe(clock.now() + 100_001);
  });

  it("restores pre-sprint followup selection when stale-backoff respect is disabled", async () => {
    const clock = new ManualClock(2_900_000);
    const harness = await createOfflineTestHarness({ clock });
    cleanup = harness.cleanup;
    const watermarkRepository = new StreamWatermarkRepository({ db: harness.db, clock });
    const goal = harness.goalsRepository.add({
      description: "Rollback-visible dormant goal",
      priority: 7,
      provenance: { kind: "manual" },
      createdAt: clock.now() - 200_000,
    });
    watermarkRepository.set(
      getExecutiveFocusGoalStaleBackoffProcessName(goal.id),
      DEFAULT_SESSION_ID,
      {
        lastTs: clock.now(),
        lastEntryId: "empty-wake-3",
        metadata: { empty_count: 3 },
      },
    );
    const trigger = createGoalFollowupDueTrigger({
      goalsRepository: harness.goalsRepository,
      watermarkRepository,
      lookaheadMs: 20_000,
      staleMs: 100_000,
      staleBackoff: STALE_BACKOFF,
      respectStaleBackoff: false,
      clock,
    });

    await expect(trigger.scan()).resolves.toHaveLength(1);
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
      staleBackoff: STALE_BACKOFF,
      respectStaleBackoff: true,
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

  it("renders due goal disclosure labels in the autonomy payload", async () => {
    const clock = new ManualClock(4_000_000);
    const harness = await createOfflineTestHarness({ clock });
    cleanup = harness.cleanup;
    const watermarkRepository = new StreamWatermarkRepository({
      db: harness.db,
      clock,
    });
    const sam = harness.entityRepository.resolve("Sam");
    const goal = harness.goalsRepository.add({
      description: "Follow up on Sam's private goal thread",
      priority: 8,
      audienceEntityId: sam,
      provenance: { kind: "manual" },
      targetAt: clock.now() + 10_000,
    });
    const trigger = createGoalFollowupDueTrigger({
      goalsRepository: harness.goalsRepository,
      watermarkRepository,
      lookaheadMs: 20_000,
      staleMs: 14 * 24 * 60 * 60 * 1_000,
      staleBackoff: STALE_BACKOFF,
      respectStaleBackoff: true,
      clock,
    });

    const events = await trigger.scan();
    const turn = trigger.buildTurn(events[0]!);
    const rendered = formatAutonomyTriggerContext(turn.autonomyTrigger!);

    expect(events[0]?.payload).toMatchObject({
      goal_id: goal.id,
      disclosure_label: {
        disclosure_class: "relationship_private",
        private_to_entity_ids: [sam],
      },
    });
    expect(rendered).toContain("disclosure_label");
    expect(rendered).toContain("relationship_private");
    expect(rendered).toContain(sam);
  });
});
