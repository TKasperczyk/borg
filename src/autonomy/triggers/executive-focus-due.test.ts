import { afterEach, describe, expect, it, vi } from "vitest";

import { AutonomyScheduler } from "../scheduler.js";
import { AutonomyWakesRepository } from "../wakes-repository.js";
import { TestEmbeddingClient, createOfflineTestHarness } from "../../offline/test-support.js";
import type { TurnResult } from "../../cognition/index.js";
import type { EmbeddingClient } from "../../embeddings/index.js";
import { StreamWatermarkRepository, StreamWriter } from "../../stream/index.js";
import { ToolDispatcher } from "../../tools/index.js";
import { OUTBOUND_POST_TOOL_NAME } from "../../tools/internal/outbound-post-name.js";
import { ManualClock } from "../../util/clock.js";
import { SessionBusyError } from "../../util/errors.js";
import { DEFAULT_SESSION_ID } from "../../util/ids.js";
import type { AutonomyWakeSource } from "../types.js";
import {
  executiveFocusGoalStaleBackoffCooldownMs,
  getExecutiveFocusGoalStaleBackoffProcessName,
  readExecutiveFocusGoalStaleBackoffMetadata,
} from "../executive-focus-stale-backoff.js";

import { createExecutiveFocusDueTrigger } from "./executive-focus-due.js";
import { createGoalFollowupDueTrigger } from "./goal-followup-due.js";

describe("executive focus due trigger", () => {
  let cleanup: (() => Promise<void>) | undefined;

  afterEach(async () => {
    await cleanup?.();
    cleanup = undefined;
  });

  class FailingEmbeddingClient implements EmbeddingClient {
    async embed(): Promise<Float32Array> {
      throw new Error("embedding unavailable");
    }

    async embedBatch(): Promise<Float32Array[]> {
      throw new Error("embedding unavailable");
    }
  }

  class ApolloEmbeddingClient extends TestEmbeddingClient {
    async embed(text: string): Promise<Float32Array> {
      return text.includes("Apollo") ? Float32Array.from([1, 0, 0, 0]) : super.embed(text);
    }

    async embedBatch(texts: readonly string[]): Promise<Float32Array[]> {
      return Promise.all(texts.map((text) => this.embed(text)));
    }
  }

  async function createHarness(start = 1_000_000, embeddingClient?: EmbeddingClient) {
    const clock = new ManualClock(start);
    const harness = await createOfflineTestHarness({ clock, embeddingClient });
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
      embeddingClient: harness.embeddingClient,
      watermarkRepository: harness.watermarkRepository,
      threshold: 0.45,
      stalenessMs: 86_400_000,
      dueLeadMs: 0,
      wakeCooldownMs: 3_600_000,
      wakeCooldownBackoffMultiplier: 2,
      wakeCooldownMaxMs: 86_400_000,
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
      run: input.turnRun ?? vi.fn().mockResolvedValue(messageTurnResult()),
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
        goalsRepository: input.harness.goalsRepository,
        turnOrchestrator: turnOrchestrator as never,
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

  function baseTurnResult(overrides: Partial<TurnResult> = {}): TurnResult {
    return {
      turn_id: "turn_autonomous",
      mode: "idle",
      path: "system_1",
      response: "Handled executive wake.",
      emitted: true,
      emission: {
        kind: "message",
        content: "Handled executive wake.",
        agentMessageId: "strm_agent" as never,
      },
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
      ...overrides,
    };
  }

  function messageTurnResult(): TurnResult {
    return baseTurnResult();
  }

  function suppressedTurnResult(): TurnResult {
    return baseTurnResult({
      path: "suppressed",
      response: "",
      emitted: false,
      emission: {
        kind: "suppressed",
        reason: "finalizer_no_output",
      },
      agentMessageId: undefined,
    });
  }

  function continueThoughtTurnResult(): TurnResult {
    return baseTurnResult({
      path: "suppressed",
      response: "",
      emitted: false,
      emission: {
        kind: "continue_thought",
        markerEntryId: "strm_continue" as never,
      },
      agentMessageId: undefined,
    });
  }

  function observedTurnResult(): TurnResult {
    return baseTurnResult({
      path: "suppressed",
      response: "",
      emitted: false,
      emission: {
        kind: "observed",
        reason: "passive_observe",
        markerEntryId: "strm_observed" as never,
      },
      agentMessageId: undefined,
    });
  }

  function outboundPostTurnResult(): TurnResult {
    return baseTurnResult({
      path: "suppressed",
      response: "",
      emitted: false,
      emission: {
        kind: "suppressed",
        reason: "finalizer_no_output",
      },
      toolCalls: [
        {
          callId: "toolu_outbound",
          name: OUTBOUND_POST_TOOL_NAME,
          input: {},
          output: {
            outbound: {
              emitted: true,
            },
          },
          ok: true,
          durationMs: 1,
        },
      ],
      agentMessageId: undefined,
    });
  }

  function goalBackoffProcessName(goalId: string): string {
    return getExecutiveFocusGoalStaleBackoffProcessName(goalId);
  }

  function getEmptyCount(
    harness: Awaited<ReturnType<typeof createHarness>>,
    goalId: string,
  ): number {
    return readExecutiveFocusGoalStaleBackoffMetadata(
      harness.watermarkRepository.get(goalBackoffProcessName(goalId), DEFAULT_SESSION_ID),
    ).empty_count;
  }

  function setGoalStaleBackoff(input: {
    harness: Awaited<ReturnType<typeof createHarness>>;
    goalId: string;
    emptyCount: number;
  }): void {
    input.harness.watermarkRepository.set(
      goalBackoffProcessName(input.goalId),
      DEFAULT_SESSION_ID,
      {
        lastTs: input.harness.clock.now(),
        lastEntryId: "stale-backoff",
        metadata: {
          empty_count: input.emptyCount,
        },
      },
    );
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

  it("keeps stale-goal backoff cooldown at least the base cooldown", () => {
    expect(
      executiveFocusGoalStaleBackoffCooldownMs({
        baseCooldownMs: 3_600_000,
        multiplier: 2,
        maxCooldownMs: 1_000,
        emptyCount: 1,
      }),
    ).toBe(3_600_000);
  });

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
        disclosure_label: {
          disclosure_class: "self_private",
          private_to_entity_ids: [],
        },
      },
      due_step: {
        id: step.id,
        disclosure_label: {
          disclosure_class: "self_private",
          private_to_entity_ids: [],
        },
      },
    });
  });

  it("describes the next due executive step boundary without scoring", async () => {
    const harness = await createHarness();
    const goal = harness.goalsRepository.add({
      description: "Ship executive focus",
      priority: 10,
      provenance: { kind: "manual" },
    });
    harness.executiveStepsRepository.add({
      goalId: goal.id,
      description: "Act when the step enters its lead window",
      kind: "act",
      dueAt: harness.clock.now() + 120_000,
      provenance: { kind: "manual" },
    });
    const trigger = createTrigger(harness, {
      dueLeadMs: 30_000,
    });

    await expect(trigger.nextDueAt!()).resolves.toBe(harness.clock.now() + 90_000);

    harness.clock.advance(100_000);
    await expect(trigger.nextDueAt!()).resolves.toBe(harness.clock.now());

    harness.watermarkRepository.set(
      `autonomy:executive-focus-due:cooldown:${goal.id}`,
      "default" as never,
      {
        lastTs: harness.clock.now(),
        lastEntryId: "cooldown",
      },
    );
    await expect(trigger.nextDueAt!()).resolves.toBe(harness.clock.now() + 3_600_000);
  });

  it("returns null instead of scanning beyond the due-step observability candidate cap", async () => {
    const harness = await createHarness();

    for (let index = 0; index < 513; index += 1) {
      const goal = harness.goalsRepository.add({
        description: `Bounded executive goal ${index}`,
        priority: 1,
        provenance: { kind: "manual" },
      });
      harness.executiveStepsRepository.add({
        goalId: goal.id,
        description: `Bounded executive step ${index}`,
        kind: "act",
        dueAt: harness.clock.now() + 120_000 + index,
        provenance: { kind: "manual" },
      });
    }
    const trigger = createTrigger(harness, {
      dueLeadMs: 30_000,
    });

    await expect(trigger.nextDueAt!()).resolves.toBeNull();
  });

  it("uses embedding context fit for autonomy executive scoring", async () => {
    const harness = await createHarness(1_000_000, new ApolloEmbeddingClient());
    const goal = harness.goalsRepository.add({
      description: "Apollo launch plan",
      priority: 10,
      provenance: { kind: "manual" },
    });
    harness.executiveStepsRepository.add({
      goalId: goal.id,
      description: "Act on the Apollo launch checklist",
      kind: "act",
      dueAt: harness.clock.now(),
      provenance: { kind: "manual" },
    });
    const trigger = createTrigger(harness);

    const events = await trigger.scan();

    expect(events).toHaveLength(1);
    expect(events[0]?.payload.selected_score.components.context_fit).toBeGreaterThan(0);
  });

  it("falls back to zero context fit and emits degraded trace when embeddings fail", async () => {
    const harness = await createHarness(1_000_000, new FailingEmbeddingClient());
    const goal = harness.goalsRepository.add({
      description: "Apollo launch plan",
      priority: 10,
      provenance: { kind: "manual" },
    });
    harness.executiveStepsRepository.add({
      goalId: goal.id,
      description: "Act on the Apollo launch checklist",
      kind: "act",
      dueAt: harness.clock.now(),
      provenance: { kind: "manual" },
    });
    const emit = vi.fn();
    const trigger = createTrigger(harness, {
      tracer: {
        enabled: true,
        includePayloads: false,
        emit,
      },
    });

    const events = await trigger.scan();

    expect(events).toHaveLength(1);
    expect(events[0]?.payload.selected_score.components.context_fit).toBe(0);
    expect(emit).toHaveBeenCalledWith("retrieval.degraded", {
      turnId: "autonomy:executive_focus_due",
      subsystem: "scoring_features",
      reason: "embedding unavailable",
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

  it("does not reach past a fresh top eligible goal to fire on a lower stale goal", async () => {
    const harness = await createHarness();
    harness.goalsRepository.add({
      description: "Fresh deadline-shaped executive goal",
      priority: 10,
      provenance: { kind: "manual" },
      createdAt: harness.clock.now(),
      targetAt: harness.clock.now() + 60_000,
    });
    harness.goalsRepository.add({
      description: "Lower stale executive goal",
      priority: 9,
      provenance: { kind: "manual" },
      createdAt: harness.clock.now() - 90_000_000,
    });
    const trigger = createTrigger(harness, {
      stalenessMs: 86_400_000,
    });

    await expect(trigger.scan()).resolves.toEqual([]);
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

  it.each([
    [
      "busy",
      () =>
        new SessionBusyError("Session is busy", {
          code: "SESSION_TURN_BUSY",
        }),
      "busy_skipped",
    ],
    ["failed", () => new Error("turn failed"), "error"],
  ])(
    "does not increment stale-goal empty wake backoff when the turn is %s",
    async (_label, createError, expectedStatus) => {
      const harness = await createHarness(5_000_000);
      const goal = harness.goalsRepository.add({
        description: "Stale goal with unsuccessful autonomous turn",
        priority: 10,
        provenance: { kind: "manual" },
        createdAt: harness.clock.now() - 10_000,
      });
      const trigger = createTrigger(harness, {
        stalenessMs: 1_000,
        wakeCooldownMs: 1_000,
      });
      const { scheduler } = createScheduler({
        harness,
        trigger,
        turnRun: vi.fn().mockRejectedValue(createError()),
      });

      const result = await scheduler.tick();

      expect(result.firedEvents).toBe(0);
      expect(result.events[0]?.status).toBe(expectedStatus);
      expect(
        harness.watermarkRepository.get(goalBackoffProcessName(goal.id), DEFAULT_SESSION_ID),
      ).toBeNull();
    },
  );

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

  it("backs off repeated empty stale-goal wakes exponentially", async () => {
    const harness = await createHarness(5_000_000);
    const goal = harness.goalsRepository.add({
      description: "Keep reviewing the dormant priority",
      priority: 10,
      provenance: { kind: "manual" },
      createdAt: harness.clock.now() - 10_000,
    });
    const trigger = createTrigger(harness, {
      stalenessMs: 1_000,
      wakeCooldownMs: 1_000,
      wakeCooldownBackoffMultiplier: 2,
      wakeCooldownMaxMs: 60_000,
    });
    const { scheduler } = createScheduler({
      harness,
      trigger,
      turnRun: vi.fn().mockResolvedValue(suppressedTurnResult()),
    });

    expect((await scheduler.tick()).firedEvents).toBe(1);
    expect(getEmptyCount(harness, goal.id)).toBe(1);

    harness.clock.advance(1_999);
    expect(await trigger.scan()).toEqual([]);

    harness.clock.advance(1);
    expect(await trigger.scan()).toHaveLength(1);
    expect((await scheduler.tick()).firedEvents).toBe(1);
    expect(getEmptyCount(harness, goal.id)).toBe(2);

    harness.clock.advance(3_999);
    expect(await trigger.scan()).toEqual([]);

    harness.clock.advance(1);
    expect(await trigger.scan()).toHaveLength(1);
  });

  it("caps stale-goal empty wake backoff at the configured maximum", async () => {
    const harness = await createHarness(5_000_000);
    const goal = harness.goalsRepository.add({
      description: "Keep capped stale wake dampening",
      priority: 10,
      provenance: { kind: "manual" },
      createdAt: harness.clock.now() - 10_000,
    });
    const trigger = createTrigger(harness, {
      stalenessMs: 1_000,
      wakeCooldownMs: 1_000,
      wakeCooldownBackoffMultiplier: 10,
      wakeCooldownMaxMs: 2_500,
    });
    const { scheduler } = createScheduler({
      harness,
      trigger,
      turnRun: vi.fn().mockResolvedValue(suppressedTurnResult()),
    });

    expect((await scheduler.tick()).firedEvents).toBe(1);
    expect(getEmptyCount(harness, goal.id)).toBe(1);

    harness.clock.advance(2_499);
    expect(await trigger.scan()).toEqual([]);

    harness.clock.advance(1);
    expect(await trigger.scan()).toHaveLength(1);
  });

  it("resets stale-goal empty wake backoff on continued train-of-thought emission", async () => {
    const harness = await createHarness(5_000_000);
    const goal = harness.goalsRepository.add({
      description: "Continue private reasoning about a stale goal",
      priority: 10,
      provenance: { kind: "manual" },
      createdAt: harness.clock.now() - 10_000,
    });
    const trigger = createTrigger(harness, {
      stalenessMs: 1_000,
      wakeCooldownMs: 1_000,
      wakeCooldownBackoffMultiplier: 2,
      wakeCooldownMaxMs: 60_000,
    });
    const { scheduler } = createScheduler({
      harness,
      trigger,
      turnRun: vi
        .fn()
        .mockResolvedValueOnce(suppressedTurnResult())
        .mockResolvedValueOnce(continueThoughtTurnResult()),
    });

    expect((await scheduler.tick()).firedEvents).toBe(1);
    expect(getEmptyCount(harness, goal.id)).toBe(1);

    harness.clock.advance(2_000);
    expect((await scheduler.tick()).firedEvents).toBe(1);
    expect(
      harness.watermarkRepository.get(goalBackoffProcessName(goal.id), DEFAULT_SESSION_ID),
    ).toBeNull();

    harness.clock.advance(1_000);
    expect(await trigger.scan()).toHaveLength(1);
  });

  it("resets stale-goal empty wake backoff on successful outbound post emission", async () => {
    const harness = await createHarness(5_000_000);
    const goal = harness.goalsRepository.add({
      description: "Send an authorized outbound followup",
      priority: 10,
      provenance: { kind: "manual" },
      createdAt: harness.clock.now() - 10_000,
    });
    const trigger = createTrigger(harness, {
      stalenessMs: 1_000,
      wakeCooldownMs: 1_000,
      wakeCooldownBackoffMultiplier: 2,
      wakeCooldownMaxMs: 60_000,
    });
    const { scheduler } = createScheduler({
      harness,
      trigger,
      turnRun: vi
        .fn()
        .mockResolvedValueOnce(suppressedTurnResult())
        .mockResolvedValueOnce(outboundPostTurnResult()),
    });

    expect((await scheduler.tick()).firedEvents).toBe(1);
    expect(getEmptyCount(harness, goal.id)).toBe(1);

    harness.clock.advance(2_000);
    expect((await scheduler.tick()).firedEvents).toBe(1);
    expect(
      harness.watermarkRepository.get(goalBackoffProcessName(goal.id), DEFAULT_SESSION_ID),
    ).toBeNull();

    harness.clock.advance(1_000);
    expect(await trigger.scan()).toHaveLength(1);
  });

  it("counts observed emission as an empty stale-goal wake", async () => {
    const harness = await createHarness(5_000_000);
    const goal = harness.goalsRepository.add({
      description: "Passively observe without carrying goal work forward",
      priority: 10,
      provenance: { kind: "manual" },
      createdAt: harness.clock.now() - 10_000,
    });
    const trigger = createTrigger(harness, {
      stalenessMs: 1_000,
      wakeCooldownMs: 1_000,
      wakeCooldownBackoffMultiplier: 2,
      wakeCooldownMaxMs: 60_000,
    });
    const { scheduler } = createScheduler({
      harness,
      trigger,
      turnRun: vi
        .fn()
        .mockResolvedValueOnce(suppressedTurnResult())
        .mockResolvedValueOnce(observedTurnResult()),
    });

    expect((await scheduler.tick()).firedEvents).toBe(1);
    expect(getEmptyCount(harness, goal.id)).toBe(1);

    harness.clock.advance(2_000);
    expect((await scheduler.tick()).firedEvents).toBe(1);
    expect(getEmptyCount(harness, goal.id)).toBe(2);
  });

  it("resets stale-goal empty wake backoff when the focused goal progresses during the turn", async () => {
    const harness = await createHarness(5_000_000);
    const goal = harness.goalsRepository.add({
      description: "Advance the stale goal during the autonomous turn",
      priority: 10,
      provenance: { kind: "manual" },
      createdAt: harness.clock.now() - 10_000,
    });
    const trigger = createTrigger(harness, {
      stalenessMs: 1_000,
      wakeCooldownMs: 1_000,
      wakeCooldownBackoffMultiplier: 2,
      wakeCooldownMaxMs: 60_000,
    });
    const turnRun = vi
      .fn()
      .mockResolvedValueOnce(suppressedTurnResult())
      .mockImplementationOnce(async () => {
        harness.goalsRepository.updateProgress(goal.id, "Autonomous turn made headway.", {
          kind: "manual",
        });
        return suppressedTurnResult();
      });
    const { scheduler } = createScheduler({
      harness,
      trigger,
      turnRun,
    });

    expect((await scheduler.tick()).firedEvents).toBe(1);
    expect(getEmptyCount(harness, goal.id)).toBe(1);

    harness.clock.advance(2_000);
    expect((await scheduler.tick()).firedEvents).toBe(1);
    expect(
      harness.watermarkRepository.get(goalBackoffProcessName(goal.id), DEFAULT_SESSION_ID),
    ).toBeNull();

    harness.clock.advance(1_000);
    expect(await trigger.scan()).toHaveLength(1);
  });

  it("lets the next eligible stale candidate fire when the top stale goal is backed off", async () => {
    const harness = await createHarness(5_000_000);
    const backedOffGoal = harness.goalsRepository.add({
      description: "Top stale goal under empty-wake dampening",
      priority: 10,
      provenance: { kind: "manual" },
      createdAt: harness.clock.now() - 20_000,
    });
    const nextGoal = harness.goalsRepository.add({
      description: "Second stale goal still eligible",
      priority: 9,
      provenance: { kind: "manual" },
      createdAt: harness.clock.now() - 20_000,
    });
    const trigger = createTrigger(harness, {
      stalenessMs: 1_000,
      wakeCooldownMs: 1_000,
      wakeCooldownBackoffMultiplier: 4,
      wakeCooldownMaxMs: 60_000,
    });

    setGoalStaleBackoff({
      harness,
      goalId: backedOffGoal.id,
      emptyCount: 1,
    });

    const events = await trigger.scan();

    expect(events).toHaveLength(1);
    expect(events[0]?.payload).toMatchObject({
      reason: "goal_stale",
      selected_goal_id: nextGoal.id,
    });
  });

  it("does not delay step_due wakes with stale-goal empty wake backoff", async () => {
    const harness = await createHarness(5_000_000);
    const goal = harness.goalsRepository.add({
      description: "Due step must bypass stale-only dampening",
      priority: 10,
      provenance: { kind: "manual" },
      createdAt: harness.clock.now() - 20_000,
    });
    const step = harness.executiveStepsRepository.add({
      goalId: goal.id,
      description: "Act on the overdue step",
      kind: "act",
      dueAt: harness.clock.now(),
      provenance: { kind: "manual" },
    });
    const trigger = createTrigger(harness, {
      stalenessMs: 1_000,
      wakeCooldownMs: 1_000,
      wakeCooldownBackoffMultiplier: 4,
      wakeCooldownMaxMs: 60_000,
    });

    setGoalStaleBackoff({
      harness,
      goalId: goal.id,
      emptyCount: 1,
    });

    const events = await trigger.scan();

    expect(events).toHaveLength(1);
    expect(events[0]?.payload).toMatchObject({
      reason: "step_due",
      selected_goal_id: goal.id,
      due_step: {
        id: step.id,
      },
    });
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
