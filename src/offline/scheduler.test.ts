import { afterEach, describe, expect, it, vi } from "vitest";

import {
  StreamWriter,
  type StreamWatermark,
  type StreamWatermarkRepository,
} from "../stream/index.js";
import type { LanceDbOptimizeStorageResult } from "../storage/lancedb/index.js";
import { ManualClock } from "../util/clock.js";
import { DEFAULT_SESSION_ID, type SessionId } from "../util/ids.js";

import {
  MaintenanceScheduler,
  type MaintenanceCadence,
  type MaintenanceSchedulerOptions,
  type MaintenanceTickResult,
} from "./scheduler.js";
import { MaintenanceOrchestrator, type MaintenanceRunOptions } from "./orchestrator.js";
import { createOfflineTestHarness } from "./test-support.js";
import type {
  OfflineProcess,
  OfflineProcessName,
  OfflineProcessPlan,
  OfflineResult,
  OrchestratorResult,
} from "./types.js";
import type { MaintenancePlan } from "./plan-file.js";

type FakeOrchestratorSpy = {
  orchestrator: MaintenanceOrchestrator;
  runCalls: MaintenanceRunOptions[];
};

function createFakeOrchestrator(
  runImpl?: (options: MaintenanceRunOptions) => Promise<OrchestratorResult>,
): FakeOrchestratorSpy {
  const runCalls: MaintenanceRunOptions[] = [];
  const run = async (options: MaintenanceRunOptions): Promise<OrchestratorResult> => {
    runCalls.push(options);

    let result: OrchestratorResult;

    if (runImpl !== undefined) {
      result = await runImpl(options);
    } else {
      result = {
        run_id: "mrun_fake",
        dryRun: false,
        results: [],
        changes: [],
        tokens_used: 0,
        errors: [],
      } as unknown as OrchestratorResult;
    }

    await options.afterRun?.(result);
    return result;
  };

  const orchestrator = {
    plan: async () => ({}) as MaintenancePlan,
    preview: () => ({}) as OrchestratorResult,
    apply: async () => ({}) as OrchestratorResult,
    runMechanicalMaintenance: async <T>(operation: () => Promise<T>) => operation(),
    run,
  } satisfies Pick<
    MaintenanceOrchestrator,
    "plan" | "preview" | "apply" | "run" | "runMechanicalMaintenance"
  >;

  return {
    orchestrator: orchestrator as unknown as MaintenanceOrchestrator,
    runCalls,
  };
}

function createFakeProcessRegistry(): Record<OfflineProcessName, OfflineProcess> {
  const names: OfflineProcessName[] = [
    "consolidator",
    "reflector",
    "semantic-extractor",
    "curator",
    "overseer",
    "review-resolver",
    "ruminator",
    "self-narrator",
    "procedural-synthesizer",
    "belief-reviser",
    "creator-directive-reconciler",
    "commitment-reconciler",
  ];

  return names.reduce(
    (acc, name) => {
      acc[name] = {
        name,
        plan: async () => ({}) as never,
        preview: () => ({}) as never,
        apply: async () => ({}) as never,
        run: async () => ({}) as never,
      };

      return acc;
    },
    {} as Record<OfflineProcessName, OfflineProcess>,
  );
}

function createStorageOptimizationResult(): LanceDbOptimizeStorageResult {
  return {
    cleanupOlderThan: 1_000,
    durationMs: 12,
    tables: [
      {
        table: "episodes",
        status: "ok",
        fragmentsRemoved: 4,
        fragmentsAdded: 1,
        versionsPruned: 3,
        bytesRemoved: 128,
        durationMs: 11,
      },
    ],
  };
}

const MAINTENANCE_WATERMARK_SESSION_ID = "maintenance-global" as SessionId;

function cadenceWatermarkProcessName(cadence: MaintenanceCadence): string {
  return `maintenance:cadence:${cadence}`;
}

function watermarkKey(processName: string, sessionId: SessionId): string {
  return `${processName}\0${sessionId}`;
}

function createFakeCadenceWatermarks(clock: ManualClock): {
  repo: Pick<StreamWatermarkRepository, "get" | "set">;
  get: (cadence: MaintenanceCadence) => StreamWatermark | null;
  set: (cadence: MaintenanceCadence, lastTs: number, lastEntryId?: string) => StreamWatermark;
} {
  const records = new Map<string, StreamWatermark>();
  const repo: Pick<StreamWatermarkRepository, "get" | "set"> = {
    get: vi.fn((processName: string, sessionId: SessionId) => {
      return records.get(watermarkKey(processName, sessionId)) ?? null;
    }),
    set: vi.fn((processName: string, sessionId: SessionId, input) => {
      const record: StreamWatermark = {
        processName,
        sessionId,
        lastTs: input.lastTs,
        lastEntryId: input.lastEntryId,
        updatedAt: clock.now(),
      };
      records.set(watermarkKey(processName, sessionId), record);
      return record;
    }),
  };

  return {
    repo,
    get: (cadence) =>
      repo.get(cadenceWatermarkProcessName(cadence), MAINTENANCE_WATERMARK_SESSION_ID),
    set: (cadence, lastTs, lastEntryId = `${cadence}:${lastTs}`) =>
      repo.set(cadenceWatermarkProcessName(cadence), MAINTENANCE_WATERMARK_SESSION_ID, {
        lastTs,
        lastEntryId,
      }),
  };
}

function withSchedulerDefaults(
  options: Omit<
    MaintenanceSchedulerOptions,
    "busyRetryBaseMs" | "busyRetryMaxMs" | "cadenceWatermarkRepository" | "startupGraceMs"
  > &
    Partial<
      Pick<
        MaintenanceSchedulerOptions,
        "busyRetryBaseMs" | "busyRetryMaxMs" | "cadenceWatermarkRepository" | "startupGraceMs"
      >
    >,
): MaintenanceSchedulerOptions {
  const clock = options.clock ?? new ManualClock(1_000);

  return {
    busyRetryBaseMs: 1_000,
    busyRetryMaxMs: 10_000,
    cadenceWatermarkRepository: createFakeCadenceWatermarks(clock as ManualClock).repo,
    startupGraceMs: 0,
    ...options,
    clock,
  };
}

type ManualTimeout = {
  id: number;
  callback: () => void;
  delayMs: number;
  cleared: boolean;
};

function createManualTimeouts(): {
  setTimeoutFn: NonNullable<MaintenanceSchedulerOptions["setTimeoutFn"]>;
  clearTimeoutFn: NonNullable<MaintenanceSchedulerOptions["clearTimeoutFn"]>;
  activeTimers: () => ManualTimeout[];
  fire: (timer?: ManualTimeout) => Promise<void>;
} {
  const timers: ManualTimeout[] = [];
  let nextId = 1;
  const activeTimers = () => timers.filter((timer) => !timer.cleared);
  const setTimeoutFn: NonNullable<MaintenanceSchedulerOptions["setTimeoutFn"]> = (
    callback,
    delayMs,
  ) => {
    const timer: ManualTimeout = {
      id: nextId,
      callback,
      delayMs,
      cleared: false,
    };
    nextId += 1;
    timers.push(timer);
    return timer.id as unknown as ReturnType<typeof setTimeout>;
  };
  const clearTimeoutFn: NonNullable<MaintenanceSchedulerOptions["clearTimeoutFn"]> = (handle) => {
    const timer = timers.find((entry) => entry.id === (handle as unknown as number));

    if (timer !== undefined) {
      timer.cleared = true;
    }
  };
  const fire = async (timer = activeTimers()[0]): Promise<void> => {
    if (timer === undefined) {
      throw new Error("No active timer to fire.");
    }

    timer.cleared = true;
    timer.callback();
    await new Promise((resolve) => setImmediate(resolve));
  };

  return {
    setTimeoutFn,
    clearTimeoutFn,
    activeTimers,
    fire,
  };
}

describe("MaintenanceScheduler", () => {
  const cleanup: Array<() => Promise<void>> = [];

  afterEach(() => {
    vi.restoreAllMocks();
  });

  afterEach(async () => {
    while (cleanup.length > 0) {
      await cleanup.pop()?.();
    }
  });

  it("runs the configured light cadence on tick", async () => {
    const clock = new ManualClock(1_000);
    const spy = createFakeOrchestrator();
    const scheduler = new MaintenanceScheduler(
      withSchedulerDefaults({
        enabled: true,
        lightIntervalMs: 10_000,
        heavyIntervalMs: 60_000,
        lightProcesses: ["consolidator", "curator"],
        heavyProcesses: ["reflector"],
        orchestrator: spy.orchestrator,
        processRegistry: createFakeProcessRegistry(),
        clock,
      }),
    );

    const result = await scheduler.tick("light");

    expect(result.status).toBe("ok");
    expect(result.cadence).toBe("light");
    expect(result.processes).toEqual(["consolidator", "curator"]);
    expect(spy.runCalls).toHaveLength(1);
    expect(spy.runCalls[0]?.processes.map((process) => process.name)).toEqual([
      "consolidator",
      "curator",
    ]);
  });

  it("selects heavy processes when heavy cadence is requested", async () => {
    const clock = new ManualClock(1_000);
    const spy = createFakeOrchestrator();
    const scheduler = new MaintenanceScheduler(
      withSchedulerDefaults({
        enabled: true,
        lightIntervalMs: 10_000,
        heavyIntervalMs: 60_000,
        lightProcesses: ["consolidator"],
        heavyProcesses: ["reflector", "overseer", "self-narrator"],
        orchestrator: spy.orchestrator,
        processRegistry: createFakeProcessRegistry(),
        clock,
      }),
    );

    await scheduler.tick("heavy");

    expect(spy.runCalls[0]?.processes.map((process) => process.name)).toEqual([
      "reflector",
      "overseer",
      "self-narrator",
    ]);
  });

  it("runs storage optimization after enabled heavy maintenance", async () => {
    const clock = new ManualClock(1_000);
    const events: string[] = [];
    const traceEvents: Array<{ event: string; data: Record<string, unknown> }> = [];
    const spy = createFakeOrchestrator(async () => {
      events.push("orchestrator");

      return {
        run_id: "mrun_storage",
        dryRun: false,
        results: [],
        changes: [],
        tokens_used: 0,
        errors: [],
      } as unknown as OrchestratorResult;
    });
    const scheduler = new MaintenanceScheduler(
      withSchedulerDefaults({
        enabled: true,
        lightIntervalMs: 10_000,
        heavyIntervalMs: 60_000,
        lightProcesses: ["consolidator"],
        heavyProcesses: ["reflector"],
        orchestrator: spy.orchestrator,
        processRegistry: createFakeProcessRegistry(),
        optimizeStorage: true,
        storageOptimizer: async () => {
          events.push("storage");
          return createStorageOptimizationResult();
        },
        tracer: {
          enabled: true,
          includePayloads: true,
          emit: (event, data) => {
            traceEvents.push({ event, data });
          },
        },
        clock,
      }),
    );

    const result = await scheduler.tick("heavy");

    expect(events).toEqual(["orchestrator", "storage"]);
    expect(result.status).toBe("ok");
    expect(result.storageOptimization).toMatchObject({
      tables: [
        {
          table: "episodes",
          status: "ok",
          fragmentsRemoved: 4,
          fragmentsAdded: 1,
          versionsPruned: 3,
        },
      ],
    });
    expect(traceEvents).toEqual([
      {
        event: "storage.optimize.completed",
        data: expect.objectContaining({
          turnId: "mrun_storage",
          cadence: "heavy",
          table_count: 1,
          errors: 0,
          fragments_removed: 4,
          fragments_added: 1,
          versions_pruned: 3,
          duration_ms: 12,
        }),
      },
    ]);
  });

  it("skips storage optimization when disabled", async () => {
    const clock = new ManualClock(1_000);
    const spy = createFakeOrchestrator();
    const storageOptimizer = vi.fn(async () => createStorageOptimizationResult());
    const scheduler = new MaintenanceScheduler(
      withSchedulerDefaults({
        enabled: true,
        lightIntervalMs: 10_000,
        heavyIntervalMs: 60_000,
        lightProcesses: ["consolidator"],
        heavyProcesses: ["reflector"],
        orchestrator: spy.orchestrator,
        processRegistry: createFakeProcessRegistry(),
        optimizeStorage: false,
        storageOptimizer,
        clock,
      }),
    );

    const result = await scheduler.tick("heavy");

    expect(storageOptimizer).not.toHaveBeenCalled();
    expect(result.status).toBe("ok");
    expect(result.storageOptimization).toBeNull();
  });

  it("reports optimizer-wide failures without failing the heavy tick", async () => {
    const clock = new ManualClock(1_000);
    const traceEvents: Array<{ event: string; data: Record<string, unknown> }> = [];
    const spy = createFakeOrchestrator();
    const scheduler = new MaintenanceScheduler(
      withSchedulerDefaults({
        enabled: true,
        lightIntervalMs: 10_000,
        heavyIntervalMs: 60_000,
        lightProcesses: ["consolidator"],
        heavyProcesses: ["reflector"],
        orchestrator: spy.orchestrator,
        processRegistry: createFakeProcessRegistry(),
        optimizeStorage: true,
        storageOptimizer: async () => {
          throw new Error("table enumeration failed");
        },
        tracer: {
          enabled: true,
          includePayloads: true,
          emit: (event, data) => {
            traceEvents.push({ event, data });
          },
        },
        clock,
      }),
    );

    const result = await scheduler.tick("heavy");

    expect(result.status).toBe("ok");
    expect(result.result).not.toBeNull();
    expect(result.storageOptimization).toMatchObject({
      tables: [],
      error: {
        message: "table enumeration failed",
      },
    });
    expect(traceEvents).toEqual([
      {
        event: "storage.optimize.completed",
        data: expect.objectContaining({
          turnId: "mrun_fake",
          cadence: "heavy",
          table_count: 0,
          errors: 1,
          optimizer_error_message: "table enumeration failed",
        }),
      },
    ]);
  });

  it("runs heavy storage optimization without adding an offline process", async () => {
    const clock = new ManualClock(1_000);
    const spy = createFakeOrchestrator();
    const storageOptimizer = vi.fn(async () => createStorageOptimizationResult());
    const scheduler = new MaintenanceScheduler(
      withSchedulerDefaults({
        enabled: true,
        lightIntervalMs: 10_000,
        heavyIntervalMs: 60_000,
        lightProcesses: ["consolidator"],
        heavyProcesses: [],
        orchestrator: spy.orchestrator,
        processRegistry: createFakeProcessRegistry(),
        optimizeStorage: true,
        storageOptimizer,
        clock,
      }),
    );

    const result = await scheduler.tick("heavy");

    expect(result.status).toBe("ok");
    expect(result.processes).toEqual([]);
    expect(result.result).toBeNull();
    expect(result.storageOptimization).toEqual(createStorageOptimizationResult());
    expect(spy.runCalls).toHaveLength(0);
    expect(storageOptimizer).toHaveBeenCalledTimes(1);
  });

  it("reports disabled when the scheduler is off", async () => {
    const clock = new ManualClock(1_000);
    const spy = createFakeOrchestrator();
    const scheduler = new MaintenanceScheduler(
      withSchedulerDefaults({
        enabled: false,
        lightIntervalMs: 10_000,
        heavyIntervalMs: 60_000,
        lightProcesses: ["consolidator"],
        heavyProcesses: ["reflector"],
        orchestrator: spy.orchestrator,
        processRegistry: createFakeProcessRegistry(),
        clock,
      }),
    );

    const result = await scheduler.tick("light");

    expect(result.status).toBe("disabled");
    expect(spy.runCalls).toHaveLength(0);
  });

  it("skips when isBusy returns true", async () => {
    const clock = new ManualClock(1_000);
    const spy = createFakeOrchestrator();
    let busy = true;
    const scheduler = new MaintenanceScheduler(
      withSchedulerDefaults({
        enabled: true,
        lightIntervalMs: 10_000,
        heavyIntervalMs: 60_000,
        lightProcesses: ["consolidator"],
        heavyProcesses: ["reflector"],
        orchestrator: spy.orchestrator,
        processRegistry: createFakeProcessRegistry(),
        clock,
        isBusy: () => busy,
      }),
    );

    const busyResult = await scheduler.tick("light");
    expect(busyResult.status).toBe("skipped_busy");
    expect(spy.runCalls).toHaveLength(0);

    busy = false;
    const freeResult = await scheduler.tick("light");
    expect(freeResult.status).toBe("ok");
    expect(spy.runCalls).toHaveLength(1);
  });

  it("returns skipped_empty when the cadence has no processes", async () => {
    const clock = new ManualClock(1_000);
    const spy = createFakeOrchestrator();
    const scheduler = new MaintenanceScheduler(
      withSchedulerDefaults({
        enabled: true,
        lightIntervalMs: 10_000,
        heavyIntervalMs: 60_000,
        lightProcesses: [],
        heavyProcesses: ["reflector"],
        orchestrator: spy.orchestrator,
        processRegistry: createFakeProcessRegistry(),
        clock,
      }),
    );

    const result = await scheduler.tick("light");

    expect(result.status).toBe("skipped_empty");
    expect(spy.runCalls).toHaveLength(0);
  });

  it("schedules a boot catch-up after the grace delay when the marker is missing", async () => {
    const clock = new ManualClock(10_000);
    const spy = createFakeOrchestrator();
    const timeouts = createManualTimeouts();
    const watermarks = createFakeCadenceWatermarks(clock);
    const scheduler = new MaintenanceScheduler(
      withSchedulerDefaults({
        enabled: true,
        lightIntervalMs: 10_000,
        heavyIntervalMs: 60_000,
        lightProcesses: ["consolidator"],
        heavyProcesses: ["reflector"],
        orchestrator: spy.orchestrator,
        processRegistry: createFakeProcessRegistry(),
        cadenceWatermarkRepository: watermarks.repo,
        startupGraceMs: 250,
        clock,
        setTimeoutFn: timeouts.setTimeoutFn,
        clearTimeoutFn: timeouts.clearTimeoutFn,
      }),
    );

    scheduler.start();

    expect(timeouts.activeTimers().map((timer) => timer.delayMs)).toEqual([250, 250]);
    expect(spy.runCalls).toHaveLength(0);

    await timeouts.fire(timeouts.activeTimers()[0]);

    expect(spy.runCalls).toHaveLength(1);
    expect(watermarks.get("light")).toMatchObject({
      lastTs: 10_000,
      lastEntryId: "mrun_fake",
    });
  });

  it("does not boot-catch up when the marker is still inside the interval", () => {
    const clock = new ManualClock(10_000);
    const spy = createFakeOrchestrator();
    const timeouts = createManualTimeouts();
    const watermarks = createFakeCadenceWatermarks(clock);
    watermarks.set("light", 9_000);
    watermarks.set("heavy", 10_000);
    const scheduler = new MaintenanceScheduler(
      withSchedulerDefaults({
        enabled: true,
        lightIntervalMs: 10_000,
        heavyIntervalMs: 60_000,
        lightProcesses: ["consolidator"],
        heavyProcesses: ["reflector"],
        orchestrator: spy.orchestrator,
        processRegistry: createFakeProcessRegistry(),
        cadenceWatermarkRepository: watermarks.repo,
        startupGraceMs: 250,
        clock,
        setTimeoutFn: timeouts.setTimeoutFn,
        clearTimeoutFn: timeouts.clearTimeoutFn,
      }),
    );

    scheduler.start();

    expect(timeouts.activeTimers().map((timer) => timer.delayMs)).toEqual([9_000, 60_000]);
    expect(spy.runCalls).toHaveLength(0);
  });

  it("runs an overdue boot catch-up exactly once", async () => {
    const clock = new ManualClock(20_000);
    const spy = createFakeOrchestrator();
    const timeouts = createManualTimeouts();
    const watermarks = createFakeCadenceWatermarks(clock);
    watermarks.set("light", 0);
    watermarks.set("heavy", 20_000);
    const scheduler = new MaintenanceScheduler(
      withSchedulerDefaults({
        enabled: true,
        lightIntervalMs: 10_000,
        heavyIntervalMs: 60_000,
        lightProcesses: ["consolidator"],
        heavyProcesses: ["reflector"],
        orchestrator: spy.orchestrator,
        processRegistry: createFakeProcessRegistry(),
        cadenceWatermarkRepository: watermarks.repo,
        startupGraceMs: 50,
        clock,
        setTimeoutFn: timeouts.setTimeoutFn,
        clearTimeoutFn: timeouts.clearTimeoutFn,
      }),
    );

    scheduler.start();
    await timeouts.fire(timeouts.activeTimers()[0]);

    expect(spy.runCalls).toHaveLength(1);
    expect(watermarks.get("light")?.lastTs).toBe(20_000);
    expect(timeouts.activeTimers().map((timer) => timer.delayMs)).toEqual([60_000, 10_000]);
  });

  it("keeps start idempotent", async () => {
    const clock = new ManualClock(1_000);
    const spy = createFakeOrchestrator();
    const timeouts = createManualTimeouts();
    const scheduler = new MaintenanceScheduler(
      withSchedulerDefaults({
        enabled: true,
        lightIntervalMs: 10_000,
        heavyIntervalMs: 60_000,
        lightProcesses: ["consolidator"],
        heavyProcesses: ["reflector"],
        orchestrator: spy.orchestrator,
        processRegistry: createFakeProcessRegistry(),
        startupGraceMs: 25,
        clock,
        setTimeoutFn: timeouts.setTimeoutFn,
        clearTimeoutFn: timeouts.clearTimeoutFn,
      }),
    );

    scheduler.start();
    scheduler.start();

    expect(timeouts.activeTimers()).toHaveLength(2);

    await timeouts.fire(timeouts.activeTimers()[0]);

    expect(spy.runCalls).toHaveLength(1);
  });

  it("retries a busy boot catch-up without advancing the marker", async () => {
    const clock = new ManualClock(1_000);
    const spy = createFakeOrchestrator();
    const timeouts = createManualTimeouts();
    const watermarks = createFakeCadenceWatermarks(clock);
    let busy = true;
    const scheduler = new MaintenanceScheduler(
      withSchedulerDefaults({
        enabled: true,
        lightIntervalMs: 10_000,
        heavyIntervalMs: 60_000,
        lightProcesses: ["consolidator"],
        heavyProcesses: ["reflector"],
        orchestrator: spy.orchestrator,
        processRegistry: createFakeProcessRegistry(),
        cadenceWatermarkRepository: watermarks.repo,
        busyRetryBaseMs: 500,
        busyRetryMaxMs: 2_000,
        clock,
        isBusy: () => busy,
        setTimeoutFn: timeouts.setTimeoutFn,
        clearTimeoutFn: timeouts.clearTimeoutFn,
      }),
    );

    scheduler.start();
    await timeouts.fire(timeouts.activeTimers()[0]);

    expect(spy.runCalls).toHaveLength(0);
    expect(watermarks.get("light")).toBeNull();
    expect(timeouts.activeTimers().map((timer) => timer.delayMs)).toEqual([0, 500]);

    busy = false;
    await timeouts.fire(timeouts.activeTimers().find((timer) => timer.delayMs === 500));

    expect(spy.runCalls).toHaveLength(1);
    expect(watermarks.get("light")?.lastTs).toBe(1_000);
  });

  it("grows busy retry backoff and clamps it at the configured maximum", async () => {
    const clock = new ManualClock(1_000);
    const spy = createFakeOrchestrator();
    const timeouts = createManualTimeouts();
    const watermarks = createFakeCadenceWatermarks(clock);
    watermarks.set("heavy", 1_000);
    const scheduler = new MaintenanceScheduler(
      withSchedulerDefaults({
        enabled: true,
        lightIntervalMs: 10_000,
        heavyIntervalMs: 60_000,
        lightProcesses: ["consolidator"],
        heavyProcesses: ["reflector"],
        orchestrator: spy.orchestrator,
        processRegistry: createFakeProcessRegistry(),
        cadenceWatermarkRepository: watermarks.repo,
        busyRetryBaseMs: 100,
        busyRetryMaxMs: 250,
        clock,
        isBusy: () => true,
        setTimeoutFn: timeouts.setTimeoutFn,
        clearTimeoutFn: timeouts.clearTimeoutFn,
      }),
    );

    scheduler.start();

    await timeouts.fire(timeouts.activeTimers()[0]);
    expect(timeouts.activeTimers().map((timer) => timer.delayMs)).toEqual([60_000, 100]);

    await timeouts.fire(timeouts.activeTimers().find((timer) => timer.delayMs === 100));
    expect(timeouts.activeTimers().map((timer) => timer.delayMs)).toEqual([60_000, 200]);

    await timeouts.fire(timeouts.activeTimers().find((timer) => timer.delayMs === 200));
    expect(timeouts.activeTimers().map((timer) => timer.delayMs)).toEqual([60_000, 250]);

    await timeouts.fire(timeouts.activeTimers().find((timer) => timer.delayMs === 250));
    expect(timeouts.activeTimers().map((timer) => timer.delayMs)).toEqual([60_000, 250]);
    expect(watermarks.get("light")).toBeNull();
  });

  it("does not double-fire after a regular due run across restarts", async () => {
    const clock = new ManualClock(1_090);
    const spy = createFakeOrchestrator();
    const watermarks = createFakeCadenceWatermarks(clock);
    watermarks.set("light", 1_000);
    watermarks.set("heavy", 1_090);
    const firstTimeouts = createManualTimeouts();
    const firstScheduler = new MaintenanceScheduler(
      withSchedulerDefaults({
        enabled: true,
        lightIntervalMs: 100,
        heavyIntervalMs: 60_000,
        lightProcesses: ["consolidator"],
        heavyProcesses: ["reflector"],
        orchestrator: spy.orchestrator,
        processRegistry: createFakeProcessRegistry(),
        cadenceWatermarkRepository: watermarks.repo,
        startupGraceMs: 25,
        clock,
        setTimeoutFn: firstTimeouts.setTimeoutFn,
        clearTimeoutFn: firstTimeouts.clearTimeoutFn,
      }),
    );

    firstScheduler.start();
    expect(firstTimeouts.activeTimers().map((timer) => timer.delayMs)).toEqual([10, 60_000]);

    clock.advance(10);
    await firstTimeouts.fire(firstTimeouts.activeTimers()[0]);
    await firstScheduler.stop();

    expect(spy.runCalls).toHaveLength(1);
    expect(watermarks.get("light")?.lastTs).toBe(1_100);

    const secondTimeouts = createManualTimeouts();
    const secondScheduler = new MaintenanceScheduler(
      withSchedulerDefaults({
        enabled: true,
        lightIntervalMs: 100,
        heavyIntervalMs: 60_000,
        lightProcesses: ["consolidator"],
        heavyProcesses: ["reflector"],
        orchestrator: spy.orchestrator,
        processRegistry: createFakeProcessRegistry(),
        cadenceWatermarkRepository: watermarks.repo,
        startupGraceMs: 25,
        clock,
        setTimeoutFn: secondTimeouts.setTimeoutFn,
        clearTimeoutFn: secondTimeouts.clearTimeoutFn,
      }),
    );

    secondScheduler.start();

    expect(secondTimeouts.activeTimers().map((timer) => timer.delayMs)).toEqual([100, 59_990]);
    expect(spy.runCalls).toHaveLength(1);
  });

  it("re-reads the marker immediately before an overdue catch-up tick", async () => {
    const clock = new ManualClock(1_000);
    const spy = createFakeOrchestrator();
    const timeouts = createManualTimeouts();
    const watermarks = createFakeCadenceWatermarks(clock);
    const scheduler = new MaintenanceScheduler(
      withSchedulerDefaults({
        enabled: true,
        lightIntervalMs: 10_000,
        heavyIntervalMs: 60_000,
        lightProcesses: ["consolidator"],
        heavyProcesses: ["reflector"],
        orchestrator: spy.orchestrator,
        processRegistry: createFakeProcessRegistry(),
        cadenceWatermarkRepository: watermarks.repo,
        clock,
        setTimeoutFn: timeouts.setTimeoutFn,
        clearTimeoutFn: timeouts.clearTimeoutFn,
      }),
    );

    scheduler.start();
    watermarks.set("light", 1_000);
    await timeouts.fire(timeouts.activeTimers()[0]);

    expect(spy.runCalls).toHaveLength(0);
    expect(timeouts.activeTimers().map((timer) => timer.delayMs)).toEqual([0, 10_000]);
  });

  it("advances the heavy marker after a successful storage-only heavy tick", async () => {
    const clock = new ManualClock(1_000);
    const spy = createFakeOrchestrator();
    const watermarks = createFakeCadenceWatermarks(clock);
    const storageOptimizer = vi.fn(async () => createStorageOptimizationResult());
    const scheduler = new MaintenanceScheduler(
      withSchedulerDefaults({
        enabled: true,
        lightIntervalMs: 10_000,
        heavyIntervalMs: 60_000,
        lightProcesses: ["consolidator"],
        heavyProcesses: [],
        orchestrator: spy.orchestrator,
        processRegistry: createFakeProcessRegistry(),
        cadenceWatermarkRepository: watermarks.repo,
        optimizeStorage: true,
        storageOptimizer,
        clock,
      }),
    );

    const result = await scheduler.tick("heavy");

    expect(result.status).toBe("ok");
    expect(spy.runCalls).toHaveLength(0);
    expect(storageOptimizer).toHaveBeenCalledTimes(1);
    expect(watermarks.get("heavy")).toMatchObject({
      lastTs: 1_000,
      lastEntryId: "maintenance:heavy:1000",
    });
  });

  it("does not advance the marker for skipped, disabled, or thrown ticks", async () => {
    const skippedClock = new ManualClock(1_000);
    const skippedWatermarks = createFakeCadenceWatermarks(skippedClock);
    const skippedScheduler = new MaintenanceScheduler(
      withSchedulerDefaults({
        enabled: true,
        lightIntervalMs: 10_000,
        heavyIntervalMs: 60_000,
        lightProcesses: [],
        heavyProcesses: ["reflector"],
        orchestrator: createFakeOrchestrator().orchestrator,
        processRegistry: createFakeProcessRegistry(),
        cadenceWatermarkRepository: skippedWatermarks.repo,
        clock: skippedClock,
      }),
    );

    expect((await skippedScheduler.tick("light")).status).toBe("skipped_empty");
    expect(skippedWatermarks.get("light")).toBeNull();

    const disabledClock = new ManualClock(2_000);
    const disabledWatermarks = createFakeCadenceWatermarks(disabledClock);
    const disabledScheduler = new MaintenanceScheduler(
      withSchedulerDefaults({
        enabled: false,
        lightIntervalMs: 10_000,
        heavyIntervalMs: 60_000,
        lightProcesses: ["consolidator"],
        heavyProcesses: ["reflector"],
        orchestrator: createFakeOrchestrator().orchestrator,
        processRegistry: createFakeProcessRegistry(),
        cadenceWatermarkRepository: disabledWatermarks.repo,
        clock: disabledClock,
      }),
    );

    expect((await disabledScheduler.tick("light")).status).toBe("disabled");
    expect(disabledWatermarks.get("light")).toBeNull();

    const thrownClock = new ManualClock(3_000);
    const thrownWatermarks = createFakeCadenceWatermarks(thrownClock);
    const thrownScheduler = new MaintenanceScheduler(
      withSchedulerDefaults({
        enabled: true,
        lightIntervalMs: 10_000,
        heavyIntervalMs: 60_000,
        lightProcesses: ["consolidator"],
        heavyProcesses: ["reflector"],
        orchestrator: createFakeOrchestrator(async () => {
          throw new Error("maintenance failed");
        }).orchestrator,
        processRegistry: createFakeProcessRegistry(),
        cadenceWatermarkRepository: thrownWatermarks.repo,
        clock: thrownClock,
      }),
    );

    await expect(thrownScheduler.tick("light")).rejects.toThrow("maintenance failed");
    expect(thrownWatermarks.get("light")).toBeNull();
  });

  it("advances the marker for ok ticks with embedded process and optimizer errors", async () => {
    const clock = new ManualClock(1_000);
    const watermarks = createFakeCadenceWatermarks(clock);
    const processError = {
      process: "reflector",
      message: "invalid payload",
      code: "INVALID_PAYLOAD",
    } as const;
    const spy = createFakeOrchestrator(async () => {
      return {
        run_id: "mrun_with_embedded_errors",
        dryRun: false,
        results: [],
        changes: [],
        tokens_used: 0,
        errors: [processError],
      } as unknown as OrchestratorResult;
    });
    const storageOptimizer = vi.fn(async () => {
      throw new Error("optimize failed");
    });
    const scheduler = new MaintenanceScheduler(
      withSchedulerDefaults({
        enabled: true,
        lightIntervalMs: 10_000,
        heavyIntervalMs: 60_000,
        lightProcesses: ["consolidator"],
        heavyProcesses: ["reflector"],
        orchestrator: spy.orchestrator,
        processRegistry: createFakeProcessRegistry(),
        cadenceWatermarkRepository: watermarks.repo,
        optimizeStorage: true,
        storageOptimizer,
        clock,
      }),
    );

    const result = await scheduler.tick("heavy");

    expect(result.status).toBe("ok");
    expect(result.result?.errors).toEqual([processError]);
    expect(result.storageOptimization?.error?.message).toBe("optimize failed");
    expect(watermarks.get("heavy")).toMatchObject({
      lastTs: 1_000,
      lastEntryId: "mrun_with_embedded_errors",
    });
  });

  it("reschedules when the pre-run cadence watermark read throws", async () => {
    const clock = new ManualClock(1_000);
    const spy = createFakeOrchestrator();
    const timeouts = createManualTimeouts();
    const watermarks = createFakeCadenceWatermarks(clock);
    const readError = new Error("watermark read failed");
    let lightReads = 0;
    const cadenceWatermarkRepository: Pick<StreamWatermarkRepository, "get" | "set"> = {
      get: vi.fn((processName, sessionId) => {
        if (processName === cadenceWatermarkProcessName("light")) {
          lightReads += 1;

          if (lightReads > 1) {
            throw readError;
          }
        }

        return watermarks.repo.get(processName, sessionId);
      }),
      set: watermarks.repo.set,
    };
    const errors: unknown[] = [];
    const scheduler = new MaintenanceScheduler(
      withSchedulerDefaults({
        enabled: true,
        lightIntervalMs: 10_000,
        heavyIntervalMs: 60_000,
        lightProcesses: ["consolidator"],
        heavyProcesses: ["reflector"],
        orchestrator: spy.orchestrator,
        processRegistry: createFakeProcessRegistry(),
        cadenceWatermarkRepository,
        busyRetryBaseMs: 500,
        busyRetryMaxMs: 2_000,
        clock,
        setTimeoutFn: timeouts.setTimeoutFn,
        clearTimeoutFn: timeouts.clearTimeoutFn,
      }),
    );
    scheduler.setObserver({
      onError: (error) => {
        errors.push(error);
      },
    });

    scheduler.start();
    await timeouts.fire(timeouts.activeTimers()[0]);

    expect(errors).toEqual([readError]);
    expect(spy.runCalls).toHaveLength(0);
    expect(watermarks.get("light")).toBeNull();
    expect(timeouts.activeTimers().map((timer) => timer.delayMs)).toEqual([0, 500]);
  });

  it("rejects overlapping light and heavy process sets", () => {
    const clock = new ManualClock(1_000);
    const spy = createFakeOrchestrator();

    expect(
      () =>
        new MaintenanceScheduler(
          withSchedulerDefaults({
            enabled: true,
            lightIntervalMs: 10_000,
            heavyIntervalMs: 60_000,
            lightProcesses: ["consolidator", "reflector"],
            heavyProcesses: ["reflector"],
            orchestrator: spy.orchestrator,
            processRegistry: createFakeProcessRegistry(),
            clock,
          }),
        ),
    ).toThrow(/overlapping processes: reflector/);
  });

  it("coalesces same-cadence concurrent ticks without dropping different cadences", async () => {
    const clock = new ManualClock(1_000);
    const gates: Array<() => void> = [];
    const spy = createFakeOrchestrator(async () => {
      await new Promise<void>((resolve) => {
        gates.push(resolve);
      });
      return {
        run_id: "mrun_fake",
        dryRun: false,
        results: [],
        changes: [],
        tokens_used: 0,
        errors: [],
      } as unknown as OrchestratorResult;
    });
    const scheduler = new MaintenanceScheduler(
      withSchedulerDefaults({
        enabled: true,
        lightIntervalMs: 10_000,
        heavyIntervalMs: 60_000,
        lightProcesses: ["consolidator"],
        heavyProcesses: ["reflector"],
        orchestrator: spy.orchestrator,
        processRegistry: createFakeProcessRegistry(),
        clock,
      }),
    );

    const light1 = scheduler.tick("light");
    const light2 = scheduler.tick("light");
    const heavy = scheduler.tick("heavy");

    // Flush microtasks so both orchestrator.run invocations reach the gate.
    await new Promise((resolve) => setImmediate(resolve));

    // Same-cadence calls coalesce to a single run; distinct cadences are both submitted.
    expect(spy.runCalls).toHaveLength(2);

    for (const release of gates) {
      release();
    }

    const light1Result = await light1;
    const light2Result = await light2;
    const heavyResult = await heavy;

    expect(light1Result).toBe(light2Result);
    expect(light1Result.cadence).toBe("light");
    expect(heavyResult.cadence).toBe("heavy");
    expect(heavyResult).not.toBe(light1Result);
  });

  it("does not run light and heavy maintenance process applies concurrently", async () => {
    const harness = await createOfflineTestHarness();
    cleanup.push(harness.cleanup);

    const processNames: OfflineProcessName[] = [
      "consolidator",
      "reflector",
      "semantic-extractor",
      "curator",
      "overseer",
      "ruminator",
      "self-narrator",
    ];
    const events: string[] = [];
    const releases: Array<() => void> = [];
    let activeApplies = 0;
    let maxActiveApplies = 0;
    const resultFor = (name: OfflineProcessName): OfflineResult => ({
      process: name,
      dryRun: false,
      changes: [],
      tokens_used: 0,
      errors: [],
      budget_exhausted: false,
    });
    const processRegistry = processNames.reduce(
      (acc, name) => {
        acc[name] = {
          name,
          plan: async () =>
            ({
              process: name,
              items: [],
              errors: [],
              tokens_used: 0,
              budget_exhausted: false,
            }) as OfflineProcessPlan,
          preview: () => resultFor(name),
          apply: async () => {
            events.push(`${name}:start`);
            activeApplies += 1;
            maxActiveApplies = Math.max(maxActiveApplies, activeApplies);

            await new Promise<void>((resolve) => {
              releases.push(resolve);
            });

            activeApplies -= 1;
            events.push(`${name}:end`);
            return resultFor(name);
          },
          run: async () => resultFor(name),
        };

        return acc;
      },
      {} as Record<OfflineProcessName, OfflineProcess>,
    );
    const {
      runId: _runId,
      auditLog: _auditLog,
      streamWriter: _streamWriter,
      ...baseContext
    } = harness.createContext();
    const orchestrator = new MaintenanceOrchestrator({
      baseContext,
      auditLog: harness.auditLog,
      createStreamWriter: () =>
        new StreamWriter({
          dataDir: harness.tempDir,
          sessionId: DEFAULT_SESSION_ID,
          clock: harness.clock,
        }),
      processRegistry,
    });
    const scheduler = new MaintenanceScheduler(
      withSchedulerDefaults({
        enabled: true,
        lightIntervalMs: 10_000,
        heavyIntervalMs: 60_000,
        lightProcesses: ["curator"],
        heavyProcesses: ["reflector"],
        orchestrator,
        processRegistry,
        clock: harness.clock,
      }),
    );
    const flush = async () => {
      await new Promise((resolve) => setImmediate(resolve));
    };
    const waitForReleaseCount = async (count: number) => {
      for (let attempt = 0; attempt < 20 && releases.length < count; attempt += 1) {
        await flush();
      }

      expect(releases).toHaveLength(count);
    };

    const light = scheduler.tick("light");
    const heavy = scheduler.tick("heavy");

    await waitForReleaseCount(1);

    expect(events).toEqual(["curator:start"]);
    expect(maxActiveApplies).toBe(1);

    releases[0]?.();
    await waitForReleaseCount(2);

    expect(events).toEqual(["curator:start", "curator:end", "reflector:start"]);
    expect(maxActiveApplies).toBe(1);

    releases[1]?.();

    const [lightResult, heavyResult] = await Promise.all([light, heavy]);

    expect(lightResult.status).toBe("ok");
    expect(heavyResult.status).toBe("ok");
    expect(events).toEqual(["curator:start", "curator:end", "reflector:start", "reflector:end"]);
    expect(maxActiveApplies).toBe(1);
  });

  it("runs due timers when started and clears timers when stopped", async () => {
    const clock = new ManualClock(1_000);
    const spy = createFakeOrchestrator();
    const timeouts = createManualTimeouts();

    const scheduler = new MaintenanceScheduler(
      withSchedulerDefaults({
        enabled: true,
        lightIntervalMs: 10_000,
        heavyIntervalMs: 60_000,
        lightProcesses: ["consolidator"],
        heavyProcesses: ["reflector"],
        orchestrator: spy.orchestrator,
        processRegistry: createFakeProcessRegistry(),
        clock,
        setTimeoutFn: timeouts.setTimeoutFn,
        clearTimeoutFn: timeouts.clearTimeoutFn,
      }),
    );

    scheduler.start();
    expect(timeouts.activeTimers().map((timer) => timer.delayMs)).toEqual([0, 0]);

    await timeouts.fire(timeouts.activeTimers()[0]);
    expect(spy.runCalls).toHaveLength(1);
    expect(timeouts.activeTimers().map((timer) => timer.delayMs)).toEqual([0, 10_000]);

    await scheduler.stop();
    expect(timeouts.activeTimers()).toHaveLength(0);
  });

  it("graceful stop awaits in-flight ticks while non-graceful stop does not", async () => {
    const clock = new ManualClock(1_000);
    const releases: Array<() => void> = [];
    const spy = createFakeOrchestrator(async () => {
      await new Promise<void>((resolve) => {
        releases.push(resolve);
      });
      return {
        run_id: "mrun_fake",
        dryRun: false,
        results: [],
        changes: [],
        tokens_used: 0,
        errors: [],
      } as unknown as OrchestratorResult;
    });
    const scheduler = new MaintenanceScheduler(
      withSchedulerDefaults({
        enabled: true,
        lightIntervalMs: 10_000,
        heavyIntervalMs: 60_000,
        lightProcesses: ["consolidator"],
        heavyProcesses: ["reflector"],
        orchestrator: spy.orchestrator,
        processRegistry: createFakeProcessRegistry(),
        clock,
      }),
    );

    const activeTick = scheduler.tick("light");
    await new Promise((resolve) => setImmediate(resolve));

    let gracefulStopped = false;
    const gracefulStop = scheduler.stop().then(() => {
      gracefulStopped = true;
    });
    await new Promise((resolve) => setImmediate(resolve));

    expect(gracefulStopped).toBe(false);

    releases[0]?.();
    await activeTick;
    await gracefulStop;

    expect(gracefulStopped).toBe(true);

    const nextTick = scheduler.tick("light");
    await new Promise((resolve) => setImmediate(resolve));

    let nonGracefulStopped = false;
    await scheduler.stop({ graceful: false }).then(() => {
      nonGracefulStopped = true;
    });

    expect(nonGracefulStopped).toBe(true);

    releases[1]?.();
    await nextTick;
  });

  it("invokes observer onTick for scheduled runs and onError when orchestrator throws", async () => {
    const clock = new ManualClock(1_000);
    const error = new Error("boom");
    let shouldThrow = false;
    const spy = createFakeOrchestrator(async () => {
      if (shouldThrow) {
        throw error;
      }
      return {
        run_id: "mrun_fake",
        dryRun: false,
        results: [],
        changes: [],
        tokens_used: 0,
        errors: [],
      } as unknown as OrchestratorResult;
    });
    const timeouts = createManualTimeouts();

    const ticks: MaintenanceTickResult[] = [];
    const errors: unknown[] = [];

    const scheduler = new MaintenanceScheduler(
      withSchedulerDefaults({
        enabled: true,
        lightIntervalMs: 10_000,
        heavyIntervalMs: 60_000,
        lightProcesses: ["consolidator"],
        heavyProcesses: ["reflector"],
        orchestrator: spy.orchestrator,
        processRegistry: createFakeProcessRegistry(),
        clock,
        setTimeoutFn: timeouts.setTimeoutFn,
        clearTimeoutFn: timeouts.clearTimeoutFn,
      }),
    );

    scheduler.setObserver({
      onTick: (result) => {
        ticks.push(result);
      },
      onError: (err) => {
        errors.push(err);
      },
    });

    scheduler.start();
    await timeouts.fire(timeouts.activeTimers()[0]);
    expect(ticks).toHaveLength(1);
    expect(ticks[0]?.status).toBe("ok");

    shouldThrow = true;
    await timeouts.fire(timeouts.activeTimers()[0]);
    expect(errors).toHaveLength(1);
    expect(errors[0]).toBe(error);

    await scheduler.stop();
  });
});
