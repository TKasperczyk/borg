import { afterEach, describe, expect, it, vi } from "vitest";

import { AutonomyScheduler } from "../scheduler.js";
import { AutonomyWakesRepository } from "../wakes-repository.js";
import { createOfflineTestHarness } from "../../offline/test-support.js";
import { StreamWatermarkRepository, StreamWriter } from "../../stream/index.js";
import { ToolDispatcher } from "../../tools/index.js";
import { ManualClock } from "../../util/clock.js";
import { SessionBusyError } from "../../util/errors.js";
import { DEFAULT_SESSION_ID } from "../../util/ids.js";
import type { AutonomyWakeSource } from "../types.js";

import { createExecutiveFocusDueTrigger } from "./executive-focus-due.js";
import { createGoalFollowupDueTrigger } from "./goal-followup-due.js";

describe("executive focus due trigger", () => {
  let cleanup: (() => Promise<void>) | undefined;

  afterEach(async () => {
    await cleanup?.();
    cleanup = undefined;
  });

  async function createHarness(start = 1_000_000) {
    const clock = new ManualClock(start);
    const harness = await createOfflineTestHarness({ clock });
    cleanup = harness.cleanup;
    const watermarkRepository = new StreamWatermarkRepository({
      db: harness.db,
      clock,
    });

    return {
      ...harness,
      clock,
      watermarkRepository,
    };
  }

  function createTrigger(
    harness: Awaited<ReturnType<typeof createHarness>>,
    overrides: Partial<Parameters<typeof createExecutiveFocusDueTrigger>[0]> = {},
  ) {
    return createExecutiveFocusDueTrigger({
      enabled: true,
      goalsRepository: harness.goalsRepository,
      executiveStepsRepository: harness.executiveStepsRepository,
      episodicRepository: harness.episodicRepository,
      watermarkRepository: harness.watermarkRepository,
      threshold: 0.45,
      stalenessMs: 86_400_000,
      dueLeadMs: 0,
      wakeCooldownMs: 3_600_000,
      deadlineLookaheadMs: 604_800_000,
      goalFollowupDue: {
        enabled: false,
        lookaheadMs: 604_800_000,
        staleMs: 1_209_600_000,
      },
      clock: harness.clock,
      ...overrides,
    });
  }

  function createScheduler(input: {
    harness: Awaited<ReturnType<typeof createHarness>>;
    trigger?: ReturnType<typeof createExecutiveFocusDueTrigger>;
    sources?: readonly AutonomyWakeSource[];
    maxWakesPerWindow?: number;
    turnRun?: ReturnType<typeof vi.fn>;
  }) {
    const wakeRepository = new AutonomyWakesRepository({
      db: input.harness.db,
      clock: input.harness.clock,
    });
    const turnOrchestrator = {
      run:
        input.turnRun ??
        vi.fn().mockResolvedValue({
          mode: "idle",
          path: "system_1",
          response: "Handled executive wake.",
          thoughts: [],
          usage: {
            input_tokens: 1,
            output_tokens: 1,
            stop_reason: "end_turn",
          },
          retrievedEpisodeIds: [],
          referencedEpisodeIds: [],
          intents: [],
          toolCalls: [],
          agentMessageId: "strm_agent",
        }),
    };

    return {
      wakeRepository,
      turnOrchestrator,
      scheduler: new AutonomyScheduler({
        enabled: true,
        intervalMs: 1_000,
        maxWakesPerWindow: input.maxWakesPerWindow ?? 6,
        budgetWindowMs: 60_000,
        clock: input.harness.clock,
        createStreamWriter: (sessionId) =>
          new StreamWriter({
            dataDir: input.harness.tempDir,
            sessionId,
            clock: input.harness.clock,
          }),
        watermarkRepository: input.harness.watermarkRepository,
        wakeRepository,
        turnOrchestrator,
        toolDispatcher: new ToolDispatcher({
          createStreamWriter: (sessionId) =>
            new StreamWriter({
              dataDir: input.harness.tempDir,
              sessionId,
              clock: input.harness.clock,
            }),
          clock: input.harness.clock,
        }),
        sources: input.sources ?? (input.trigger === undefined ? [] : [input.trigger]),
      }),
    };
  }

  function createGoalFollowupTrigger(harness: Awaited<ReturnType<typeof createHarness>>) {
    return createGoalFollowupDueTrigger({
      goalsRepository: harness.goalsRepository,
      watermarkRepository: harness.watermarkRepository,
      lookaheadMs: 604_800_000,
      staleMs: 1_209_600_000,
      clock: harness.clock,
    });
  }

  it("does not fire when disabled", async () => {
    const harness = await createHarness();
    const goal = harness.goalsRepository.add({
      description: "Ship executive focus",
      priority: 10,
      provenance: { kind: "manual" },
    });
    harness.executiveStepsRepository.add({
      goalId: goal.id,
      description: "Act on the overdue step",
      kind: "act",
      dueAt: harness.clock.now() - 1,
      provenance: { kind: "manual" },
    });
    const trigger = createTrigger(harness, {
      enabled: false,
    });

    await expect(trigger.scan()).resolves.toEqual([]);
  });

  it("fires when an open executive step is due", async () => {
    const harness = await createHarness();
    const goal = harness.goalsRepository.add({
      description: "Ship executive focus",
      priority: 10,
      provenance: { kind: "manual" },
    });
    const step = harness.executiveStepsRepository.add({
      goalId: goal.id,
      description: "Act on the overdue step",
      kind: "act",
      dueAt: harness.clock.now(),
      provenance: { kind: "manual" },
    });
    const trigger = createTrigger(harness);

    const events = await trigger.scan();

    expect(events).toHaveLength(1);
    expect(events[0]?.payload).toMatchObject({
      reason: "step_due",
      selected_goal_id: goal.id,
      force_executive_focus_goal_id: goal.id,
      top_open_step: {
        id: step.id,
        description: "Act on the overdue step",
      },
      due_step: {
        id: step.id,
      },
    });
  });

  it("fires when the scorer selects a stale goal", async () => {
    const harness = await createHarness();
    const goal = harness.goalsRepository.add({
      description: "Write the executive followup tests",
      priority: 10,
      provenance: { kind: "manual" },
      createdAt: harness.clock.now() - 90_000_000,
    });
    const trigger = createTrigger(harness, {
      stalenessMs: 86_400_000,
    });

    const events = await trigger.scan();

    expect(events).toHaveLength(1);
    expect(events[0]?.payload).toMatchObject({
      reason: "goal_stale",
      selected_goal_id: goal.id,
    });
    expect(events[0]?.payload.selected_score.components.progress_debt).toBe(1);
  });

  it("does not fire for stale goals when no goal clears the threshold", async () => {
    const harness = await createHarness();
    harness.goalsRepository.add({
      description: "Low-confidence stale goal",
      priority: 1,
      provenance: { kind: "manual" },
      createdAt: harness.clock.now() - 90_000_000,
    });
    const trigger = createTrigger(harness, {
      threshold: 0.99,
      stalenessMs: 86_400_000,
    });

    await expect(trigger.scan()).resolves.toEqual([]);
  });

  it("does not run the autonomous turn when the session is busy", async () => {
    const harness = await createHarness();
    const goal = harness.goalsRepository.add({
      description: "Ship executive focus",
      priority: 10,
      provenance: { kind: "manual" },
    });
    harness.executiveStepsRepository.add({
      goalId: goal.id,
      description: "Act on the overdue step",
      kind: "act",
      dueAt: harness.clock.now(),
      provenance: { kind: "manual" },
    });
    const trigger = createTrigger(harness);
    const turnRun = vi.fn().mockRejectedValue(
      new SessionBusyError("Session is busy", {
        code: "SESSION_TURN_BUSY",
      }),
    );
    const { scheduler } = createScheduler({
      harness,
      trigger,
      turnRun,
    });

    const result = await scheduler.tick();

    expect(result.firedEvents).toBe(0);
    expect(result.busySkipped).toBe(1);
    expect(result.events[0]?.status).toBe("busy_skipped");
  });

  it("does not run the autonomous turn when wake budget is exhausted", async () => {
    const harness = await createHarness();
    const goal = harness.goalsRepository.add({
      description: "Ship executive focus",
      priority: 10,
      provenance: { kind: "manual" },
    });
    harness.executiveStepsRepository.add({
      goalId: goal.id,
      description: "Act on the overdue step",
      kind: "act",
      dueAt: harness.clock.now(),
      provenance: { kind: "manual" },
    });
    const trigger = createTrigger(harness);
    const { scheduler, wakeRepository, turnOrchestrator } = createScheduler({
      harness,
      trigger,
      maxWakesPerWindow: 1,
    });
    wakeRepository.record({
      trigger_name: "scheduled_reflection",
      session_id: DEFAULT_SESSION_ID,
      wake_source_type: "trigger",
    });

    const result = await scheduler.tick();

    expect(result.firedEvents).toBe(0);
    expect(result.budgetSkipped).toBe(1);
    expect(result.events[0]?.status).toBe("budget_skipped");
    expect(turnOrchestrator.run).not.toHaveBeenCalled();
  });

  it("applies a per-goal cooldown after an executive wake and clears it on user progress", async () => {
    const harness = await createHarness(5_000_000);
    const goal = harness.goalsRepository.add({
      description: "Ship executive focus",
      priority: 10,
      provenance: { kind: "manual" },
    });
    harness.executiveStepsRepository.add({
      goalId: goal.id,
      description: "Act on the overdue step",
      kind: "act",
      dueAt: harness.clock.now(),
      provenance: { kind: "manual" },
    });
    const trigger = createTrigger(harness, {
      wakeCooldownMs: 3_600_000,
    });
    const { scheduler } = createScheduler({
      harness,
      trigger,
    });

    expect((await scheduler.tick()).firedEvents).toBe(1);

    harness.clock.advance(3_599_999);
    expect(await trigger.scan()).toEqual([]);

    harness.clock.advance(1);
    expect(await trigger.scan()).toHaveLength(1);

    expect((await scheduler.tick()).firedEvents).toBe(1);
    expect(await trigger.scan()).toEqual([]);

    harness.goalsRepository.updateProgress(goal.id, "User made progress.", {
      kind: "manual",
    });

    expect(await trigger.scan()).toHaveLength(1);
  });

  it("keeps stale-goal executive focus subordinate to goal followup across ticks", async () => {
    const harness = await createHarness(2_000_000_000);
    const goal = harness.goalsRepository.add({
      description: "Follow up on stale executive goal",
      priority: 10,
      provenance: { kind: "manual" },
      createdAt: harness.clock.now() - 15 * 24 * 60 * 60 * 1_000,
    });
    const goalFollowupTrigger = createGoalFollowupTrigger(harness);
    const executiveTrigger = createTrigger(harness, {
      stalenessMs: 86_400_000,
      goalFollowupDue: {
        enabled: true,
        lookaheadMs: 604_800_000,
        staleMs: 1_209_600_000,
      },
    });
    const { scheduler } = createScheduler({
      harness,
      sources: [goalFollowupTrigger, executiveTrigger],
    });

    const firstTick = await scheduler.tick();

    expect(firstTick.firedEvents).toBe(1);
    expect(firstTick.events).toHaveLength(1);
    expect(firstTick.events[0]).toMatchObject({
      sourceName: "goal_followup_due",
      payload: {
        goal_id: goal.id,
      },
    });

    harness.clock.advance(60_000);

    const secondTick = await scheduler.tick();

    expect(secondTick.firedEvents).toBe(0);
    expect(secondTick.events).toEqual([]);
  });

  it("still fires executive overdue steps when goal followup also matches", async () => {
    const harness = await createHarness(2_000_000_000);
    const goal = harness.goalsRepository.add({
      description: "Act on stale goal with a due executive step",
      priority: 10,
      provenance: { kind: "manual" },
      createdAt: harness.clock.now() - 15 * 24 * 60 * 60 * 1_000,
    });
    const step = harness.executiveStepsRepository.add({
      goalId: goal.id,
      description: "Take the overdue executive action",
      kind: "act",
      dueAt: harness.clock.now() - 1,
      provenance: { kind: "manual" },
    });
    const goalFollowupTrigger = createGoalFollowupTrigger(harness);
    const executiveTrigger = createTrigger(harness, {
      stalenessMs: 86_400_000,
      goalFollowupDue: {
        enabled: true,
        lookaheadMs: 604_800_000,
        staleMs: 1_209_600_000,
      },
    });
    const { scheduler } = createScheduler({
      harness,
      sources: [goalFollowupTrigger, executiveTrigger],
    });

    const result = await scheduler.tick();

    expect(result.firedEvents).toBe(2);
    expect(result.events.map((event) => event.sourceName)).toEqual(
      expect.arrayContaining(["goal_followup_due", "executive_focus_due"]),
    );
    expect(result.events.find((event) => event.sourceName === "executive_focus_due")).toMatchObject(
      {
        payload: {
          reason: "step_due",
          selected_goal_id: goal.id,
          due_step: {
            id: step.id,
          },
        },
      },
    );
  });
});
