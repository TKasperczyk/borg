import { join } from "node:path";

import { afterEach, describe, expect, it, vi } from "vitest";

import {
  DEFAULT_SESSION_ID,
  StreamReader,
  StreamWatermarkRepository,
  StreamWriter,
  ToolDispatcher,
  autonomyMigrations,
  createCommitmentsListTool,
  createIdentityEventsListTool,
  streamWatermarkMigrations,
} from "../index.js";
import { ManualClock } from "../util/clock.js";
import { createOfflineTestHarness } from "../offline/test-support.js";
import { openDatabase, type SqliteDatabase } from "../storage/sqlite/index.js";
import { SessionBusyError } from "../util/errors.js";

import { createCommitmentExpiringTrigger, createScheduledReflectionTrigger } from "./index.js";
import { AutonomyScheduler, type AutonomySchedulerOptions } from "./scheduler.js";
import { AutonomyWakesRepository } from "./wakes-repository.js";

function createScheduler(
  options: Omit<AutonomySchedulerOptions, "budgetWindowMs" | "wakeRepository"> & {
    db: SqliteDatabase;
    budgetWindowMs?: number;
    wakeRepository?: AutonomyWakesRepository;
  },
): AutonomyScheduler {
  const { db, budgetWindowMs = 3_600_000, wakeRepository, ...schedulerOptions } = options;

  return new AutonomyScheduler({
    ...schedulerOptions,
    budgetWindowMs,
    wakeRepository:
      wakeRepository ??
      new AutonomyWakesRepository({
        db,
        clock: schedulerOptions.clock,
      }),
  });
}

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
        referencedEpisodeIds: [],
        intents: [],
        toolCalls: [],
        agentMessageId: "strm_agent_result",
      }),
    };
    const scheduler = createScheduler({
      db: harness.db,
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
    const scheduler = createScheduler({
      db: harness.db,
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
          referencedEpisodeIds: [],
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

  it("checks persisted wake history when a fresh scheduler enforces budget", async () => {
    const clock = new ManualClock(1_000_000);
    const harness = await createOfflineTestHarness({
      clock,
    });
    cleanup = harness.cleanup;
    const wakeRepository = new AutonomyWakesRepository({
      db: harness.db,
      clock,
    });
    wakeRepository.record({
      trigger_name: "scheduled_reflection",
      condition_name: null,
      session_id: DEFAULT_SESSION_ID,
      wake_source_type: "trigger",
    });
    const turnRunner = {
      run: vi.fn().mockResolvedValue({
        mode: "idle",
        path: "system_1",
        response: "Should not run.",
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
        agentMessageId: "strm_should_not_run",
      }),
    };
    const scheduler = createScheduler({
      db: harness.db,
      wakeRepository: new AutonomyWakesRepository({
        db: harness.db,
        clock,
      }),
      enabled: true,
      intervalMs: 1_000,
      maxWakesPerHour: 1,
      budgetWindowMs: 60_000,
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
      turnOrchestrator: turnRunner,
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
          async scan() {
            return [
              {
                id: "persisted-budget-event",
                sourceName: "scheduled_reflection",
                sourceType: "trigger",
                watermarkProcessName: "autonomy:test:persisted-budget",
                sortTs: clock.now(),
                payload: {},
              },
            ];
          },
          buildTurn() {
            return {
              audience: "self",
              stakes: "low",
              userMessage: "Reflect",
            };
          },
        },
      ],
    });

    const result = await scheduler.tick();
    expect(result.firedEvents).toBe(0);
    expect(result.budgetSkipped).toBe(1);
    expect(turnRunner.run).not.toHaveBeenCalled();
  });

  it("shares persisted budget across SQLite connections", async () => {
    const clock = new ManualClock(1_000_000);
    const harness = await createOfflineTestHarness({
      clock,
    });
    cleanup = harness.cleanup;
    const secondDb = openDatabase(join(harness.tempDir, "borg.db"), {
      migrations: [...autonomyMigrations, ...streamWatermarkMigrations],
    });

    try {
      const firstTurnRunner = {
        run: vi.fn().mockResolvedValue({
          mode: "idle",
          path: "system_1",
          response: "Process A wake.",
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
          agentMessageId: "strm_process_a",
        }),
      };
      const firstScheduler = createScheduler({
        db: harness.db,
        wakeRepository: new AutonomyWakesRepository({
          db: harness.db,
          clock,
        }),
        enabled: true,
        intervalMs: 1_000,
        maxWakesPerHour: 1,
        budgetWindowMs: 60_000,
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
        turnOrchestrator: firstTurnRunner,
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
                  id: "process-a-event",
                  sourceName: "goal_followup_due",
                  sourceType: "trigger",
                  watermarkProcessName: "autonomy:test:process-a",
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
                userMessage: "Process A",
              };
            },
          },
        ],
      });
      const firstResult = await firstScheduler.tick();
      expect(firstResult.firedEvents).toBe(1);

      const secondTurnRunner = {
        run: vi.fn(),
      };
      const secondScheduler = createScheduler({
        db: secondDb,
        wakeRepository: new AutonomyWakesRepository({
          db: secondDb,
          clock,
        }),
        enabled: true,
        intervalMs: 1_000,
        maxWakesPerHour: 1,
        budgetWindowMs: 60_000,
        clock,
        createStreamWriter: (sessionId) =>
          new StreamWriter({
            dataDir: harness.tempDir,
            sessionId,
            clock,
          }),
        watermarkRepository: new StreamWatermarkRepository({
          db: secondDb,
          clock,
        }),
        turnOrchestrator: secondTurnRunner,
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
                  id: "process-b-event",
                  sourceName: "goal_followup_due",
                  sourceType: "trigger",
                  watermarkProcessName: "autonomy:test:process-b",
                  sortTs: 2,
                  payload: {
                    goal_id: "goal_bbbbbbbbbbbbbbbb",
                  },
                },
              ];
            },
            buildTurn() {
              return {
                audience: "self",
                stakes: "low",
                userMessage: "Process B",
              };
            },
          },
        ],
      });

      const secondResult = await secondScheduler.tick();
      expect(secondResult.firedEvents).toBe(0);
      expect(secondResult.budgetSkipped).toBe(1);
      expect(secondTurnRunner.run).not.toHaveBeenCalled();
    } finally {
      secondDb.close();
    }
  });

  it("prunes wake records after each enabled tick", async () => {
    const clock = new ManualClock(10_000_000_000);
    const harness = await createOfflineTestHarness({
      clock,
    });
    cleanup = harness.cleanup;
    const budgetWindowMs = 60_000;
    const safetyBufferMs = 7 * 24 * 60 * 60 * 1_000;
    const pruneCutoff = clock.now() - budgetWindowMs - safetyBufferMs;
    const wakeRepository = new AutonomyWakesRepository({
      db: harness.db,
      clock,
    });

    clock.set(pruneCutoff - 1);
    const oldWake = wakeRepository.record({
      trigger_name: "scheduled_reflection",
      condition_name: null,
      session_id: DEFAULT_SESSION_ID,
      wake_source_type: "trigger",
    });
    clock.set(pruneCutoff);
    const retainedWake = wakeRepository.record({
      trigger_name: "scheduled_reflection",
      condition_name: null,
      session_id: DEFAULT_SESSION_ID,
      wake_source_type: "trigger",
    });
    clock.set(10_000_000_000);

    const scheduler = createScheduler({
      db: harness.db,
      wakeRepository,
      enabled: true,
      intervalMs: 1_000,
      maxWakesPerHour: 6,
      budgetWindowMs,
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
      sources: [],
    });

    await scheduler.tick();
    const wakeIds = wakeRepository.listSince(0, 10).map((wake) => wake.id);
    expect(wakeIds).not.toContain(oldWake.id);
    expect(wakeIds).toContain(retainedWake.id);
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
    const scheduler = createScheduler({
      db: harness.db,
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
    const scheduler = createScheduler({
      db: harness.db,
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
    const scheduler = createScheduler({
      db: harness.db,
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
    const scheduler = createScheduler({
      db: harness.db,
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
          referencedEpisodeIds: [];
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
      referencedEpisodeIds: [];
      intents: [];
      toolCalls: [];
      agentMessageId: string;
    }>((resolve) => {
      resolveTurn = resolve;
    });
    const turnRunner = {
      run: vi.fn().mockReturnValue(turnCompletion),
    };

    const scheduler = createScheduler({
      db: harness.db,
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
      referencedEpisodeIds: [],
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
          referencedEpisodeIds: [];
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
      referencedEpisodeIds: [];
      intents: [];
      toolCalls: [];
      agentMessageId: string;
    }>((resolve) => {
      resolveTurn = resolve;
    });
    const turnRunner = {
      run: vi.fn().mockReturnValue(turnCompletion),
    };

    const scheduler = createScheduler({
      db: harness.db,
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
      referencedEpisodeIds: [],
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
        referencedEpisodeIds: [],
        intents: [],
        toolCalls: [],
        agentMessageId: "strm_agent_result",
      }),
    };
    vi.spyOn(watermarkRepository, "set").mockImplementationOnce(() => {
      throw new Error("watermark commit failed");
    });

    const scheduler = createScheduler({
      db: harness.db,
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

  it("commits shared source watermarks to the processed event cursor", async () => {
    const clock = new ManualClock(1_000_000);
    const harness = await createOfflineTestHarness({
      clock,
    });
    cleanup = harness.cleanup;
    const watermarkRepository = new StreamWatermarkRepository({
      db: harness.db,
      clock,
    });
    const events = [
      {
        id: "event-a",
        sourceName: "goal_followup_due" as const,
        sourceType: "trigger" as const,
        watermarkProcessName: "autonomy:test:shared-cursor",
        sortTs: 100,
        payload: {
          goal_id: "goal_aaaaaaaaaaaaaaaa",
        },
      },
      {
        id: "event-b",
        sourceName: "goal_followup_due" as const,
        sourceType: "trigger" as const,
        watermarkProcessName: "autonomy:test:shared-cursor",
        sortTs: 100,
        payload: {
          goal_id: "goal_bbbbbbbbbbbbbbbb",
        },
      },
    ];
    const turnRunner = {
      run: vi
        .fn()
        .mockResolvedValueOnce({
          mode: "idle",
          path: "system_1",
          response: "Handled event A.",
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
          agentMessageId: "strm_event_a",
        })
        .mockRejectedValueOnce(new Error("event B failed"))
        .mockResolvedValueOnce({
          mode: "idle",
          path: "system_1",
          response: "Handled event B.",
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
          agentMessageId: "strm_event_b",
        }),
    };
    const scheduler = createScheduler({
      db: harness.db,
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
            const watermark = watermarkRepository.get(
              "autonomy:test:shared-cursor",
              DEFAULT_SESSION_ID,
            );

            return events.filter(
              (event) =>
                watermark === null ||
                event.sortTs > watermark.lastTs ||
                (event.sortTs === watermark.lastTs && event.id > (watermark.lastEntryId ?? "")),
            );
          },
          buildTurn(event) {
            return {
              audience: "self",
              stakes: "low",
              userMessage: `Handle ${event.id}`,
            };
          },
        },
      ],
    });

    const firstTick = await scheduler.tick();
    expect(firstTick.firedEvents).toBe(1);
    expect(firstTick.errorCount).toBe(1);
    expect(
      watermarkRepository.get("autonomy:test:shared-cursor", DEFAULT_SESSION_ID),
    ).toMatchObject({
      lastTs: 100,
      lastEntryId: "event-a",
    });

    clock.advance(30_000);
    const secondTick = await scheduler.tick();
    expect(secondTick.firedEvents).toBe(1);
    expect(secondTick.events[0]).toMatchObject({
      id: "event-b",
      status: "fired",
    });
    expect(
      watermarkRepository.get("autonomy:test:shared-cursor", DEFAULT_SESSION_ID),
    ).toMatchObject({
      lastTs: 100,
      lastEntryId: "event-b",
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
    const scheduler = createScheduler({
      db: harness.db,
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
          referencedEpisodeIds: [],
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

    const scheduler = createScheduler({
      db: harness.db,
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
