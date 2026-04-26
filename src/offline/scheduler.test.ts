import { afterEach, describe, expect, it, vi } from "vitest";

import { StreamWriter } from "../stream/index.js";
import { ManualClock } from "../util/clock.js";
import { DEFAULT_SESSION_ID } from "../util/ids.js";

import { MaintenanceScheduler, type MaintenanceTickResult } from "./scheduler.js";
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

    if (runImpl !== undefined) {
      return runImpl(options);
    }

    return {
      run_id: "mrun_fake",
      dryRun: false,
      results: [],
      changes: [],
      tokens_used: 0,
      errors: [],
    } as unknown as OrchestratorResult;
  };

  const orchestrator = {
    plan: async () => ({}) as MaintenancePlan,
    preview: () => ({}) as OrchestratorResult,
    apply: async () => ({}) as OrchestratorResult,
    run,
  } satisfies Pick<MaintenanceOrchestrator, "plan" | "preview" | "apply" | "run">;

  return {
    orchestrator: orchestrator as unknown as MaintenanceOrchestrator,
    runCalls,
  };
}

function createFakeProcessRegistry(): Record<OfflineProcessName, OfflineProcess> {
  const names: OfflineProcessName[] = [
    "consolidator",
    "reflector",
    "curator",
    "overseer",
    "ruminator",
    "self-narrator",
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
    const scheduler = new MaintenanceScheduler({
      enabled: true,
      lightIntervalMs: 10_000,
      heavyIntervalMs: 60_000,
      lightProcesses: ["consolidator", "curator"],
      heavyProcesses: ["reflector"],
      orchestrator: spy.orchestrator,
      processRegistry: createFakeProcessRegistry(),
      clock,
    });

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
    const scheduler = new MaintenanceScheduler({
      enabled: true,
      lightIntervalMs: 10_000,
      heavyIntervalMs: 60_000,
      lightProcesses: ["consolidator"],
      heavyProcesses: ["reflector", "overseer", "self-narrator"],
      orchestrator: spy.orchestrator,
      processRegistry: createFakeProcessRegistry(),
      clock,
    });

    await scheduler.tick("heavy");

    expect(spy.runCalls[0]?.processes.map((process) => process.name)).toEqual([
      "reflector",
      "overseer",
      "self-narrator",
    ]);
  });

  it("reports disabled when the scheduler is off", async () => {
    const clock = new ManualClock(1_000);
    const spy = createFakeOrchestrator();
    const scheduler = new MaintenanceScheduler({
      enabled: false,
      lightIntervalMs: 10_000,
      heavyIntervalMs: 60_000,
      lightProcesses: ["consolidator"],
      heavyProcesses: ["reflector"],
      orchestrator: spy.orchestrator,
      processRegistry: createFakeProcessRegistry(),
      clock,
    });

    const result = await scheduler.tick("light");

    expect(result.status).toBe("disabled");
    expect(spy.runCalls).toHaveLength(0);
  });

  it("skips when isBusy returns true", async () => {
    const clock = new ManualClock(1_000);
    const spy = createFakeOrchestrator();
    let busy = true;
    const scheduler = new MaintenanceScheduler({
      enabled: true,
      lightIntervalMs: 10_000,
      heavyIntervalMs: 60_000,
      lightProcesses: ["consolidator"],
      heavyProcesses: ["reflector"],
      orchestrator: spy.orchestrator,
      processRegistry: createFakeProcessRegistry(),
      clock,
      isBusy: () => busy,
    });

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
    const scheduler = new MaintenanceScheduler({
      enabled: true,
      lightIntervalMs: 10_000,
      heavyIntervalMs: 60_000,
      lightProcesses: [],
      heavyProcesses: ["reflector"],
      orchestrator: spy.orchestrator,
      processRegistry: createFakeProcessRegistry(),
      clock,
    });

    const result = await scheduler.tick("light");

    expect(result.status).toBe("skipped_empty");
    expect(spy.runCalls).toHaveLength(0);
  });

  it("rejects overlapping light and heavy process sets", () => {
    const clock = new ManualClock(1_000);
    const spy = createFakeOrchestrator();

    expect(
      () =>
        new MaintenanceScheduler({
          enabled: true,
          lightIntervalMs: 10_000,
          heavyIntervalMs: 60_000,
          lightProcesses: ["consolidator", "reflector"],
          heavyProcesses: ["reflector"],
          orchestrator: spy.orchestrator,
          processRegistry: createFakeProcessRegistry(),
          clock,
        }),
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
    const scheduler = new MaintenanceScheduler({
      enabled: true,
      lightIntervalMs: 10_000,
      heavyIntervalMs: 60_000,
      lightProcesses: ["consolidator"],
      heavyProcesses: ["reflector"],
      orchestrator: spy.orchestrator,
      processRegistry: createFakeProcessRegistry(),
      clock,
    });

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
    const scheduler = new MaintenanceScheduler({
      enabled: true,
      lightIntervalMs: 10_000,
      heavyIntervalMs: 60_000,
      lightProcesses: ["curator"],
      heavyProcesses: ["reflector"],
      orchestrator,
      processRegistry,
      clock: harness.clock,
    });
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

  it("runs on interval when started and stops when stopped", async () => {
    const clock = new ManualClock(1_000);
    const spy = createFakeOrchestrator();
    const intervalCallbacks: Array<() => void> = [];
    let nextHandle = 1;
    const handles = new Set<number>();
    const setIntervalFn = ((callback: () => void) => {
      intervalCallbacks.push(callback);
      const handle = nextHandle++;
      handles.add(handle);
      return handle as unknown as ReturnType<typeof setInterval>;
    }) as typeof setInterval;
    const clearIntervalFn = ((handle: ReturnType<typeof setInterval>) => {
      handles.delete(handle as unknown as number);
    }) as typeof clearInterval;

    const scheduler = new MaintenanceScheduler({
      enabled: true,
      lightIntervalMs: 10_000,
      heavyIntervalMs: 60_000,
      lightProcesses: ["consolidator"],
      heavyProcesses: ["reflector"],
      orchestrator: spy.orchestrator,
      processRegistry: createFakeProcessRegistry(),
      clock,
      setIntervalFn,
      clearIntervalFn,
    });

    scheduler.start();
    expect(handles.size).toBe(2);

    // Fire the light interval once.
    intervalCallbacks[0]?.();
    // Flush microtasks so the scheduled tick resolves.
    await new Promise((resolve) => setImmediate(resolve));
    expect(spy.runCalls).toHaveLength(1);

    await scheduler.stop();
    expect(handles.size).toBe(0);
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
    const intervalCallbacks: Array<() => void> = [];
    const setIntervalFn = ((callback: () => void) => {
      intervalCallbacks.push(callback);
      return intervalCallbacks.length as unknown as ReturnType<typeof setInterval>;
    }) as typeof setInterval;
    const clearIntervalFn = (() => {}) as typeof clearInterval;

    const ticks: MaintenanceTickResult[] = [];
    const errors: unknown[] = [];

    const scheduler = new MaintenanceScheduler({
      enabled: true,
      lightIntervalMs: 10_000,
      heavyIntervalMs: 60_000,
      lightProcesses: ["consolidator"],
      heavyProcesses: ["reflector"],
      orchestrator: spy.orchestrator,
      processRegistry: createFakeProcessRegistry(),
      clock,
      setIntervalFn,
      clearIntervalFn,
    });

    scheduler.setObserver({
      onTick: (result) => {
        ticks.push(result);
      },
      onError: (err) => {
        errors.push(err);
      },
    });

    scheduler.start();
    intervalCallbacks[0]?.();
    await new Promise((resolve) => setImmediate(resolve));
    expect(ticks).toHaveLength(1);
    expect(ticks[0]?.status).toBe("ok");

    shouldThrow = true;
    intervalCallbacks[0]?.();
    await new Promise((resolve) => setImmediate(resolve));
    expect(errors).toHaveLength(1);
    expect(errors[0]).toBe(error);

    await scheduler.stop();
  });
});
