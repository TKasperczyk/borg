import { afterEach, describe, expect, it, vi } from "vitest";

import {
  DEFAULT_SESSION_ID,
  StreamReader,
  StreamWatermarkRepository,
  StreamWriter,
  ToolDispatcher,
  createCommitmentsListTool,
  createIdentityEventsListTool,
} from "../index.js";
import { ManualClock } from "../util/clock.js";
import { createOfflineTestHarness } from "../offline/test-support.js";
import { SessionBusyError } from "../util/errors.js";

import { createCommitmentExpiringTrigger, createScheduledReflectionTrigger } from "./index.js";
import { AutonomyScheduler } from "./scheduler.js";

describe("AutonomyScheduler", () => {
  let cleanup: (() => Promise<void>) | undefined;

  afterEach(async () => {
    vi.restoreAllMocks();
    await cleanup?.();
    cleanup = undefined;
  });

  it("fires due events once and respects trigger watermarks", async () => {
    const clock = new ManualClock(1_000_000);
    const harness = await createOfflineTestHarness({
      clock,
    });
    cleanup = harness.cleanup;
    const watermarkRepository = new StreamWatermarkRepository({
      db: harness.db,
      clock,
    });
    const dispatcher = new ToolDispatcher({
      createStreamWriter: (sessionId) =>
        new StreamWriter({
          dataDir: harness.tempDir,
          sessionId,
          clock,
        }),
      clock,
    });
    dispatcher.register(
      createIdentityEventsListTool({
        listEvents: (options) => harness.identityService.listEvents(options),
      }),
    );

    const trigger = createScheduledReflectionTrigger({
      watermarkRepository,
      intervalMs: 10_000,
      clock,
    });
    const turnRunner = {
      run: vi.fn().mockResolvedValue({
        mode: "idle",
        path: "system_1",
        response: "Reflected on recent changes.",
        thoughts: [],
        usage: {
          input_tokens: 1,
          output_tokens: 1,
          stop_reason: "end_turn",
        },
        retrievedEpisodeIds: [],
        intents: [],
        toolCalls: [],
        agentMessageId: "strm_agent_result",
      }),
    };
    const scheduler = new AutonomyScheduler({
      enabled: true,
      intervalMs: 1_000,
      maxWakesPerHour: 6,
      clock,
      createStreamWriter: (sessionId) =>
        new StreamWriter({
          dataDir: harness.tempDir,
          sessionId,
          clock,
        }),
      watermarkRepository,
      turnOrchestrator: turnRunner,
      toolDispatcher: dispatcher,
      sources: [trigger],
    });

    const firstTick = await scheduler.tick();
    expect(firstTick.firedEvents).toBe(1);
    expect(turnRunner.run).toHaveBeenCalledTimes(1);
    expect(
      watermarkRepository.get("autonomy:scheduled-reflection", DEFAULT_SESSION_ID),
    ).toMatchObject({
      lastTs: 1_000_000,
      lastEntryId: expect.any(String),
    });

    const secondTick = await scheduler.tick();
    expect(secondTick.firedEvents).toBe(0);
    expect(secondTick.dueEvents).toBe(0);

    const kinds = new StreamReader({
      dataDir: harness.tempDir,
      sessionId: DEFAULT_SESSION_ID,
    })
      .tail(4)
      .map((entry) => entry.kind);
    expect(kinds).toEqual(["internal_event", "tool_call", "tool_result", "internal_event"]);
  });

  it("respects maxWakesPerHour", async () => {
    const clock = new ManualClock(1_000_000);
    const harness = await createOfflineTestHarness({
      clock,
    });
    cleanup = harness.cleanup;
    const watermarkRepository = new StreamWatermarkRepository({
      db: harness.db,
      clock,
    });
    harness.commitmentRepository.add({
      type: "promise",
      directive: "First expiring commitment",
      priority: 5,
      provenance: { kind: "manual" },
      expiresAt: clock.now() + 5_000,
    });
    harness.commitmentRepository.add({
      type: "promise",
      directive: "Second expiring commitment",
      priority: 4,
      provenance: { kind: "manual" },
      expiresAt: clock.now() + 6_000,
    });

    const dispatcher = new ToolDispatcher({
      createStreamWriter: (sessionId) =>
        new StreamWriter({
          dataDir: harness.tempDir,
          sessionId,
          clock,
        }),
      clock,
    });
    dispatcher.register(
      createCommitmentsListTool({
        listCommitments: () =>
          harness.commitmentRepository.list({
            activeOnly: true,
          }),
      }),
    );
    const trigger = createCommitmentExpiringTrigger({
      commitmentRepository: harness.commitmentRepository,
      watermarkRepository,
      lookaheadMs: 20_000,
      clock,
    });
    const scheduler = new AutonomyScheduler({
      enabled: true,
      intervalMs: 1_000,
      maxWakesPerHour: 1,
      clock,
      createStreamWriter: (sessionId) =>
        new StreamWriter({
          dataDir: harness.tempDir,
          sessionId,
          clock,
        }),
      watermarkRepository,
      turnOrchestrator: {
        run: vi.fn().mockResolvedValue({
          mode: "idle",
          path: "system_1",
          response: "Processed one commitment.",
          thoughts: [],
          usage: {
            input_tokens: 1,
            output_tokens: 1,
            stop_reason: "end_turn",
          },
          retrievedEpisodeIds: [],
          intents: [],
          toolCalls: [],
          agentMessageId: "strm_agent_budget",
        }),
      },
      toolDispatcher: dispatcher,
      sources: [trigger],
    });

    const result = await scheduler.tick();
    expect(result.firedEvents).toBe(1);
    expect(result.budgetSkipped).toBe(1);
  });

  it("skips busy autonomous turns", async () => {
    const clock = new ManualClock(1_000_000);
    const harness = await createOfflineTestHarness({
      clock,
    });
    cleanup = harness.cleanup;
    const watermarkRepository = new StreamWatermarkRepository({
      db: harness.db,
      clock,
    });
    const dispatcher = new ToolDispatcher({
      createStreamWriter: (sessionId) =>
        new StreamWriter({
          dataDir: harness.tempDir,
          sessionId,
          clock,
        }),
      clock,
    });
    dispatcher.register(
      createIdentityEventsListTool({
        listEvents: (options) => harness.identityService.listEvents(options),
      }),
    );
    const trigger = createScheduledReflectionTrigger({
      watermarkRepository,
      intervalMs: 10_000,
      clock,
    });
    const turnRunner = {
      run: vi.fn().mockRejectedValue(new SessionBusyError("busy")),
    };
    const scheduler = new AutonomyScheduler({
      enabled: true,
      intervalMs: 1_000,
      maxWakesPerHour: 6,
      clock,
      createStreamWriter: (sessionId) =>
        new StreamWriter({
          dataDir: harness.tempDir,
          sessionId,
          clock,
        }),
      watermarkRepository,
      turnOrchestrator: turnRunner,
      toolDispatcher: dispatcher,
      sources: [trigger],
    });

    const result = await scheduler.tick();
    expect(result.busySkipped).toBe(1);
    expect(result.events[0]?.status).toBe("busy_skipped");
    expect(watermarkRepository.get("autonomy:scheduled-reflection", DEFAULT_SESSION_ID)).toBeNull();

    const secondResult = await scheduler.tick();
    expect(secondResult.busySkipped).toBe(0);
    expect(secondResult.events).toEqual([]);
    expect(turnRunner.run).toHaveBeenCalledTimes(1);

    clock.advance(30_000);
    const thirdResult = await scheduler.tick();
    expect(thirdResult.busySkipped).toBe(1);
    expect(thirdResult.events[0]?.status).toBe("busy_skipped");
    expect(turnRunner.run).toHaveBeenCalledTimes(2);
  });

  it("reuses scheduled reflection backoff within a due window and refreshes it in the next window", async () => {
    const clock = new ManualClock(1_000_000);
    const harness = await createOfflineTestHarness({
      clock,
    });
    cleanup = harness.cleanup;
    const watermarkRepository = new StreamWatermarkRepository({
      db: harness.db,
      clock,
    });
    const dispatcher = new ToolDispatcher({
      createStreamWriter: (sessionId) =>
        new StreamWriter({
          dataDir: harness.tempDir,
          sessionId,
          clock,
        }),
      clock,
    });
    dispatcher.register(
      createIdentityEventsListTool({
        listEvents: (options) => harness.identityService.listEvents(options),
      }),
    );
    const trigger = createScheduledReflectionTrigger({
      watermarkRepository,
      intervalMs: 60_000,
      clock,
    });
    const turnRunner = {
      run: vi.fn().mockRejectedValue(new SessionBusyError("busy")),
    };
    const scheduler = new AutonomyScheduler({
      enabled: true,
      intervalMs: 1_000,
      maxWakesPerHour: 6,
      clock,
      createStreamWriter: (sessionId) =>
        new StreamWriter({
          dataDir: harness.tempDir,
          sessionId,
          clock,
        }),
      watermarkRepository,
      turnOrchestrator: turnRunner,
      toolDispatcher: dispatcher,
      sources: [trigger],
    });

    const firstTick = await scheduler.tick();
    expect(firstTick.events[0]?.id).toBe("scheduled-reflection:1000000");
    expect(firstTick.busySkipped).toBe(1);
    expect(turnRunner.run).toHaveBeenCalledTimes(1);

    clock.advance(10_000);
    const secondTick = await scheduler.tick();
    expect(secondTick.events).toEqual([]);
    expect(turnRunner.run).toHaveBeenCalledTimes(1);

    clock.advance(19_999);
    const thirdTick = await scheduler.tick();
    expect(thirdTick.events).toEqual([]);
    expect(turnRunner.run).toHaveBeenCalledTimes(1);

    clock.advance(30_001);
    const fourthTick = await scheduler.tick();
    expect(fourthTick.events[0]?.id).toBe("scheduled-reflection:1060000");
    expect(fourthTick.busySkipped).toBe(1);
    expect(turnRunner.run).toHaveBeenCalledTimes(2);
  });

  it("leaves trigger watermarks untouched when an autonomous turn throws", async () => {
    const clock = new ManualClock(1_000_000);
    const harness = await createOfflineTestHarness({
      clock,
    });
    cleanup = harness.cleanup;
    const watermarkRepository = new StreamWatermarkRepository({
      db: harness.db,
      clock,
    });
    const dispatcher = new ToolDispatcher({
      createStreamWriter: (sessionId) =>
        new StreamWriter({
          dataDir: harness.tempDir,
          sessionId,
          clock,
        }),
      clock,
    });
    dispatcher.register(
      createIdentityEventsListTool({
        listEvents: (options) => harness.identityService.listEvents(options),
      }),
    );
    const trigger = createScheduledReflectionTrigger({
      watermarkRepository,
      intervalMs: 10_000,
      clock,
    });
    const turnRunner = {
      run: vi.fn().mockRejectedValue(new Error("turn failed")),
    };
    const scheduler = new AutonomyScheduler({
      enabled: true,
      intervalMs: 1_000,
      maxWakesPerHour: 6,
      clock,
      createStreamWriter: (sessionId) =>
        new StreamWriter({
          dataDir: harness.tempDir,
          sessionId,
          clock,
        }),
      watermarkRepository,
      turnOrchestrator: turnRunner,
      toolDispatcher: dispatcher,
      sources: [trigger],
    });

    const result = await scheduler.tick();
    expect(result.errorCount).toBe(1);
    expect(result.events[0]?.status).toBe("error");
    expect(watermarkRepository.get("autonomy:scheduled-reflection", DEFAULT_SESSION_ID)).toBeNull();

    const secondResult = await scheduler.tick();
    expect(secondResult.errorCount).toBe(0);
    expect(secondResult.events).toEqual([]);
    expect(turnRunner.run).toHaveBeenCalledTimes(1);

    clock.advance(30_000);
    const thirdResult = await scheduler.tick();
    expect(thirdResult.errorCount).toBe(1);
    expect(thirdResult.events[0]?.status).toBe("error");
    expect(turnRunner.run).toHaveBeenCalledTimes(2);
  });

  it("is inert when autonomy is disabled", async () => {
    const clock = new ManualClock(1_000_000);
    const harness = await createOfflineTestHarness({
      clock,
    });
    cleanup = harness.cleanup;
    const watermarkRepository = new StreamWatermarkRepository({
      db: harness.db,
      clock,
    });
    const setIntervalFn = vi.fn<typeof setInterval>();
    const scheduler = new AutonomyScheduler({
      enabled: false,
      intervalMs: 1_000,
      maxWakesPerHour: 6,
      clock,
      createStreamWriter: (sessionId) =>
        new StreamWriter({
          dataDir: harness.tempDir,
          sessionId,
          clock,
        }),
      watermarkRepository,
      turnOrchestrator: {
        run: vi.fn(),
      },
      toolDispatcher: new ToolDispatcher({
        createStreamWriter: (sessionId) =>
          new StreamWriter({
            dataDir: harness.tempDir,
            sessionId,
            clock,
          }),
        clock,
      }),
      sources: [],
      setIntervalFn,
      clearIntervalFn: vi.fn(),
    });

    scheduler.start();
    expect(setIntervalFn).not.toHaveBeenCalled();
    await expect(scheduler.tick()).resolves.toMatchObject({
      status: "disabled",
      firedEvents: 0,
    });
  });

  it("waits for an active tick to finish during graceful stop", async () => {
    const clock = new ManualClock(1_000_000);
    const harness = await createOfflineTestHarness({
      clock,
    });
    cleanup = harness.cleanup;
    const watermarkRepository = new StreamWatermarkRepository({
      db: harness.db,
      clock,
    });
    const dispatcher = new ToolDispatcher({
      createStreamWriter: (sessionId) =>
        new StreamWriter({
          dataDir: harness.tempDir,
          sessionId,
          clock,
        }),
      clock,
    });
    dispatcher.register(
      createIdentityEventsListTool({
        listEvents: (options) => harness.identityService.listEvents(options),
      }),
    );

    const trigger = createScheduledReflectionTrigger({
      watermarkRepository,
      intervalMs: 10_000,
      clock,
    });

    let intervalCallback: (() => void) | undefined;
    const setIntervalFn = vi.fn<typeof setInterval>((callback) => {
      intervalCallback = callback;
      return 1 as ReturnType<typeof setInterval>;
    });
    const clearIntervalFn = vi.fn<typeof clearInterval>();
    let resolveTurn:
      | ((value: {
          mode: "idle";
          path: "system_1";
          response: string;
          thoughts: [];
          usage: {
            input_tokens: number;
            output_tokens: number;
            stop_reason: "end_turn";
          };
          retrievedEpisodeIds: [];
          intents: [];
          toolCalls: [];
          agentMessageId: string;
        }) => void)
      | undefined;
    const turnCompletion = new Promise<{
      mode: "idle";
      path: "system_1";
      response: string;
      thoughts: [];
      usage: {
        input_tokens: number;
        output_tokens: number;
        stop_reason: "end_turn";
      };
      retrievedEpisodeIds: [];
      intents: [];
      toolCalls: [];
      agentMessageId: string;
    }>((resolve) => {
      resolveTurn = resolve;
    });
    const turnRunner = {
      run: vi.fn().mockReturnValue(turnCompletion),
    };

    const scheduler = new AutonomyScheduler({
      enabled: true,
      intervalMs: 1_000,
      maxWakesPerHour: 6,
      clock,
      createStreamWriter: (sessionId) =>
        new StreamWriter({
          dataDir: harness.tempDir,
          sessionId,
          clock,
        }),
      watermarkRepository,
      turnOrchestrator: turnRunner,
      toolDispatcher: dispatcher,
      sources: [trigger],
      setIntervalFn,
      clearIntervalFn,
    });

    scheduler.start();
    intervalCallback?.();
    await vi.waitFor(() => {
      expect(turnRunner.run).toHaveBeenCalledTimes(1);
    });

    let stopped = false;
    const stopPromise = scheduler.stop();
    void stopPromise.then(() => {
      stopped = true;
    });

    await Promise.resolve();
    expect(stopped).toBe(false);

    resolveTurn?.({
      mode: "idle",
      path: "system_1",
      response: "Finished reflective work.",
      thoughts: [],
      usage: {
        input_tokens: 1,
        output_tokens: 1,
        stop_reason: "end_turn",
      },
      retrievedEpisodeIds: [],
      intents: [],
      toolCalls: [],
      agentMessageId: "strm_stop_wait",
    });

    await stopPromise;
    expect(stopped).toBe(true);
    expect(clearIntervalFn).toHaveBeenCalledTimes(1);
  });

  it("waits for a direct tick to finish during graceful stop", async () => {
    const clock = new ManualClock(1_000_000);
    const harness = await createOfflineTestHarness({
      clock,
    });
    cleanup = harness.cleanup;
    const watermarkRepository = new StreamWatermarkRepository({
      db: harness.db,
      clock,
    });
    const dispatcher = new ToolDispatcher({
      createStreamWriter: (sessionId) =>
        new StreamWriter({
          dataDir: harness.tempDir,
          sessionId,
          clock,
        }),
      clock,
    });
    dispatcher.register(
      createIdentityEventsListTool({
        listEvents: (options) => harness.identityService.listEvents(options),
      }),
    );

    const trigger = createScheduledReflectionTrigger({
      watermarkRepository,
      intervalMs: 10_000,
      clock,
    });

    let resolveTurn:
      | ((value: {
          mode: "idle";
          path: "system_1";
          response: string;
          thoughts: [];
          usage: {
            input_tokens: number;
            output_tokens: number;
            stop_reason: "end_turn";
          };
          retrievedEpisodeIds: [];
          intents: [];
          toolCalls: [];
          agentMessageId: string;
        }) => void)
      | undefined;
    const turnCompletion = new Promise<{
      mode: "idle";
      path: "system_1";
      response: string;
      thoughts: [];
      usage: {
        input_tokens: number;
        output_tokens: number;
        stop_reason: "end_turn";
      };
      retrievedEpisodeIds: [];
      intents: [];
      toolCalls: [];
      agentMessageId: string;
    }>((resolve) => {
      resolveTurn = resolve;
    });
    const turnRunner = {
      run: vi.fn().mockReturnValue(turnCompletion),
    };

    const scheduler = new AutonomyScheduler({
      enabled: true,
      intervalMs: 1_000,
      maxWakesPerHour: 6,
      clock,
      createStreamWriter: (sessionId) =>
        new StreamWriter({
          dataDir: harness.tempDir,
          sessionId,
          clock,
        }),
      watermarkRepository,
      turnOrchestrator: turnRunner,
      toolDispatcher: dispatcher,
      sources: [trigger],
    });

    const tickPromise = scheduler.tick();
    await vi.waitFor(() => {
      expect(turnRunner.run).toHaveBeenCalledTimes(1);
    });

    let stopped = false;
    const stopPromise = scheduler.stop();
    void stopPromise.then(() => {
      stopped = true;
    });

    await Promise.resolve();
    expect(stopped).toBe(false);

    resolveTurn?.({
      mode: "idle",
      path: "system_1",
      response: "Finished reflective work.",
      thoughts: [],
      usage: {
        input_tokens: 1,
        output_tokens: 1,
        stop_reason: "end_turn",
      },
      retrievedEpisodeIds: [],
      intents: [],
      toolCalls: [],
      agentMessageId: "strm_direct_stop_wait",
    });

    await Promise.all([tickPromise, stopPromise]);
    expect(stopped).toBe(true);
  });

  it("reports watermark commit failures as errors and retries the source", async () => {
    const clock = new ManualClock(1_000_000);
    const harness = await createOfflineTestHarness({
      clock,
    });
    cleanup = harness.cleanup;
    const watermarkRepository = new StreamWatermarkRepository({
      db: harness.db,
      clock,
    });
    const dispatcher = new ToolDispatcher({
      createStreamWriter: (sessionId) =>
        new StreamWriter({
          dataDir: harness.tempDir,
          sessionId,
          clock,
        }),
      clock,
    });
    dispatcher.register(
      createIdentityEventsListTool({
        listEvents: (options) => harness.identityService.listEvents(options),
      }),
    );

    const trigger = createScheduledReflectionTrigger({
      watermarkRepository,
      intervalMs: 10_000,
      clock,
    });
    const turnRunner = {
      run: vi.fn().mockResolvedValue({
        mode: "idle",
        path: "system_1",
        response: "Reflected on recent changes.",
        thoughts: [],
        usage: {
          input_tokens: 1,
          output_tokens: 1,
          stop_reason: "end_turn",
        },
        retrievedEpisodeIds: [],
        intents: [],
        toolCalls: [],
        agentMessageId: "strm_agent_result",
      }),
    };
    vi.spyOn(watermarkRepository, "set").mockImplementationOnce(() => {
      throw new Error("watermark commit failed");
    });

    const scheduler = new AutonomyScheduler({
      enabled: true,
      intervalMs: 1_000,
      maxWakesPerHour: 6,
      clock,
      createStreamWriter: (sessionId) =>
        new StreamWriter({
          dataDir: harness.tempDir,
          sessionId,
          clock,
        }),
      watermarkRepository,
      turnOrchestrator: turnRunner,
      toolDispatcher: dispatcher,
      sources: [trigger],
    });

    const firstTick = await scheduler.tick();
    expect(firstTick.firedEvents).toBe(0);
    expect(firstTick.errorCount).toBe(1);
    expect(firstTick.events[0]).toMatchObject({
      status: "error",
      turnResultId: "strm_agent_result",
      error: "Error: watermark commit failed",
    });
    expect(firstTick.events[0]?.outcomeSummary).toContain("watermark commit failed");
    expect(watermarkRepository.get("autonomy:scheduled-reflection", DEFAULT_SESSION_ID)).toBeNull();

    const secondTick = await scheduler.tick();
    expect(secondTick.firedEvents).toBe(0);
    expect(secondTick.events).toEqual([]);
    expect(turnRunner.run).toHaveBeenCalledTimes(1);

    clock.advance(30_000);
    const thirdTick = await scheduler.tick();
    expect(thirdTick.firedEvents).toBe(1);
    expect(thirdTick.events[0]?.status).toBe("fired");
    expect(turnRunner.run).toHaveBeenCalledTimes(2);
    expect(
      watermarkRepository.get("autonomy:scheduled-reflection", DEFAULT_SESSION_ID),
    ).toMatchObject({
      lastTs: 1_030_000,
      lastEntryId: expect.any(String),
    });
  });

  it("dispatches mixed trigger and condition sources and records wake metadata", async () => {
    const clock = new ManualClock(1_000_000);
    const harness = await createOfflineTestHarness({
      clock,
    });
    cleanup = harness.cleanup;
    const watermarkRepository = new StreamWatermarkRepository({
      db: harness.db,
      clock,
    });
    const scheduler = new AutonomyScheduler({
      enabled: true,
      intervalMs: 1_000,
      maxWakesPerHour: 6,
      clock,
      createStreamWriter: (sessionId) =>
        new StreamWriter({
          dataDir: harness.tempDir,
          sessionId,
          clock,
        }),
      watermarkRepository,
      turnOrchestrator: {
        run: vi.fn().mockResolvedValue({
          mode: "idle",
          path: "system_1",
          response: "Handled the wake.",
          thoughts: [],
          usage: {
            input_tokens: 1,
            output_tokens: 1,
            stop_reason: "end_turn",
          },
          retrievedEpisodeIds: [],
          intents: [],
          toolCalls: [],
          agentMessageId: "strm_mixed_sources",
        }),
      },
      toolDispatcher: new ToolDispatcher({
        createStreamWriter: (sessionId) =>
          new StreamWriter({
            dataDir: harness.tempDir,
            sessionId,
            clock,
          }),
        clock,
      }),
      sources: [
        {
          name: "goal_followup_due",
          type: "trigger",
          async scan() {
            return [
              {
                id: "goal-1",
                sourceName: "goal_followup_due",
                sourceType: "trigger",
                watermarkProcessName: "autonomy:test:goal",
                sortTs: 1,
                payload: {
                  goal_id: "goal_aaaaaaaaaaaaaaaa",
                },
              },
            ];
          },
          buildTurn() {
            return {
              audience: "self",
              stakes: "low",
              userMessage: "Goal follow-up",
            };
          },
        },
        {
          name: "commitment_revoked",
          type: "condition",
          async scan() {
            return [
              {
                id: "condition-1",
                sourceName: "commitment_revoked",
                sourceType: "condition",
                watermarkProcessName: "autonomy:test:condition",
                sortTs: 2,
                payload: {
                  commitment_id: "cmt_aaaaaaaaaaaaaaaa",
                },
              },
            ];
          },
          buildTurn() {
            return {
              audience: "self",
              stakes: "low",
              userMessage: "Commitment reflection",
            };
          },
        },
      ],
    });

    const result = await scheduler.tick();
    expect(result.firedEvents).toBe(2);
    expect(result.events.map((event) => event.sourceName)).toEqual([
      "goal_followup_due",
      "commitment_revoked",
    ]);

    const wakeEntries = new StreamReader({
      dataDir: harness.tempDir,
      sessionId: DEFAULT_SESSION_ID,
    })
      .tail(8)
      .filter((entry) => entry.kind === "internal_event" && typeof entry.content === "object");

    expect(wakeEntries[0]?.content).toMatchObject({
      kind: "autonomous_wake",
      trigger_type: "trigger",
      source_name: "goal_followup_due",
    });
    expect(wakeEntries[2]?.content).toMatchObject({
      kind: "autonomous_wake",
      trigger_type: "condition",
      source_name: "commitment_revoked",
    });
  });

  it("logs a tick error and continues scheduling later ticks", async () => {
    const clock = new ManualClock(1_000_000);
    const harness = await createOfflineTestHarness({
      clock,
    });
    cleanup = harness.cleanup;

    let intervalCallback: (() => void) | undefined;
    const setIntervalFn = vi.fn<typeof setInterval>((callback) => {
      intervalCallback = callback;
      return 1 as ReturnType<typeof setInterval>;
    });
    const onTick = vi.fn();
    const onError = vi.fn();
    let scanCount = 0;

    const scheduler = new AutonomyScheduler({
      enabled: true,
      intervalMs: 1_000,
      maxWakesPerHour: 6,
      clock,
      createStreamWriter: (sessionId) =>
        new StreamWriter({
          dataDir: harness.tempDir,
          sessionId,
          clock,
        }),
      watermarkRepository: new StreamWatermarkRepository({
        db: harness.db,
        clock,
      }),
      turnOrchestrator: {
        run: vi.fn(),
      },
      toolDispatcher: new ToolDispatcher({
        createStreamWriter: (sessionId) =>
          new StreamWriter({
            dataDir: harness.tempDir,
            sessionId,
            clock,
          }),
        clock,
      }),
      sources: [
        {
          name: "scheduled_reflection",
          type: "trigger",
          scan: vi.fn().mockImplementation(async () => {
            scanCount += 1;

            if (scanCount === 1) {
              throw new Error("scan failed");
            }

            return [];
          }),
          buildTurn: vi.fn(),
        },
      ],
      setIntervalFn,
      clearIntervalFn: vi.fn(),
    });
    scheduler.setObserver({
      onTick,
      onError,
    });

    scheduler.start();
    intervalCallback?.();
    await vi.waitFor(() => {
      expect(onError).toHaveBeenCalledTimes(1);
    });

    intervalCallback?.();
    await vi.waitFor(() => {
      expect(onTick).toHaveBeenCalledTimes(1);
    });

    await scheduler.stop();
  });
});
