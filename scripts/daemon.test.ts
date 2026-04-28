import { EventEmitter } from "node:events";

import { describe, expect, it, vi } from "vitest";

import {
  runDaemon,
  type DaemonBorg,
  type DaemonRunResult,
  type RunDaemonOptions,
} from "./daemon.ts";
import type { MaintenanceCadence, MaintenanceTickResult, TickResult } from "../src/index.ts";
import type { MaintenanceRunId } from "../src/util/ids.js";

type AutonomyObserver = Parameters<DaemonBorg["autonomy"]["scheduler"]["setObserver"]>[0];
type MaintenanceObserver = Parameters<DaemonBorg["maintenance"]["scheduler"]["setObserver"]>[0];

function autonomyTick(): TickResult {
  return {
    status: "ok",
    ts: 1,
    scannedSources: [],
    dueEvents: 0,
    firedEvents: 0,
    budgetSkipped: 0,
    busySkipped: 0,
    errorCount: 0,
    events: [],
  };
}

function maintenanceTick(cadence: MaintenanceCadence = "light"): MaintenanceTickResult {
  return {
    status: "ok",
    cadence,
    ts: 1,
    processes: [],
    result: {
      run_id: "test-run" as MaintenanceRunId,
      dryRun: false,
      results: [],
      changes: [],
      tokens_used: 0,
      errors: [],
    },
  };
}

function createAutonomyScheduler(
  enabled: boolean,
  stop: (options?: { graceful?: boolean }) => Promise<void> = async () => {},
) {
  let observer: AutonomyObserver = null;
  const scheduler: DaemonBorg["autonomy"]["scheduler"] = {
    isEnabled: vi.fn(() => enabled),
    start: vi.fn(),
    stop: vi.fn(stop),
    setObserver: vi.fn((next) => {
      observer = next;
    }),
  };

  return {
    scheduler,
    async emitTick(result: TickResult = autonomyTick()): Promise<void> {
      await observer?.onTick?.(result);
    },
  };
}

function createMaintenanceScheduler(
  enabled: boolean,
  stop: (options?: { graceful?: boolean }) => Promise<void> = async () => {},
) {
  let observer: MaintenanceObserver = null;
  const scheduler: DaemonBorg["maintenance"]["scheduler"] = {
    isEnabled: vi.fn(() => enabled),
    start: vi.fn(),
    stop: vi.fn(stop),
    setObserver: vi.fn((next) => {
      observer = next;
    }),
  };

  return {
    scheduler,
    async emitTick(result: MaintenanceTickResult = maintenanceTick()): Promise<void> {
      await observer?.onTick?.(result);
    },
  };
}

function createBorg(options: {
  autonomyEnabled: boolean;
  maintenanceEnabled: boolean;
  autonomyStop?: (options?: { graceful?: boolean }) => Promise<void>;
  maintenanceStop?: (options?: { graceful?: boolean }) => Promise<void>;
}) {
  const autonomy = createAutonomyScheduler(options.autonomyEnabled, options.autonomyStop);
  const maintenance = createMaintenanceScheduler(
    options.maintenanceEnabled,
    options.maintenanceStop,
  );
  const borg: DaemonBorg = {
    autonomy: {
      scheduler: autonomy.scheduler,
    },
    maintenance: {
      scheduler: maintenance.scheduler,
    },
    close: vi.fn(async () => {}),
  };

  return {
    autonomy,
    borg,
    maintenance,
  };
}

async function startDaemon(
  borg: DaemonBorg,
  overrides: Partial<RunDaemonOptions> = {},
): Promise<{
  exits: number[];
  logs: string[];
  result: DaemonRunResult;
  signals: EventEmitter;
}> {
  const logs: string[] = [];
  const exits: number[] = [];
  const signals = new EventEmitter();
  const result = await runDaemon({
    openBorg: async () => borg,
    writeStderr: (line) => {
      logs.push(line);
    },
    signalTarget: signals,
    exit: (code = 0) => {
      exits.push(code);
    },
    ...overrides,
  });

  return {
    exits,
    logs,
    result,
    signals,
  };
}

describe("daemon", () => {
  it("starts both schedulers and stops them cleanly when both are enabled", async () => {
    const { autonomy, borg, maintenance } = createBorg({
      autonomyEnabled: true,
      maintenanceEnabled: true,
    });
    const { logs, result } = await startDaemon(borg);

    expect(result.status).toBe("started");
    expect(autonomy.scheduler.start).toHaveBeenCalledTimes(1);
    expect(maintenance.scheduler.start).toHaveBeenCalledTimes(1);

    await autonomy.emitTick();
    await maintenance.emitTick(maintenanceTick("heavy"));

    expect(logs).toContain("[daemon] autonomy scheduler started");
    expect(logs).toContain("[daemon] maintenance scheduler started");
    expect(logs.some((line) => line.startsWith("[daemon] autonomy tick "))).toBe(true);
    expect(logs.some((line) => line.startsWith("[daemon] maintenance tick "))).toBe(true);

    await result.shutdown("test");

    expect(autonomy.scheduler.stop).toHaveBeenCalledWith({ graceful: true });
    expect(maintenance.scheduler.stop).toHaveBeenCalledWith({ graceful: true });
    expect(borg.close).toHaveBeenCalledTimes(1);
  });

  it("runs with only autonomy enabled without starting maintenance", async () => {
    const { autonomy, borg, maintenance } = createBorg({
      autonomyEnabled: true,
      maintenanceEnabled: false,
    });
    const { logs, result } = await startDaemon(borg);

    expect(result.status).toBe("started");
    expect(autonomy.scheduler.start).toHaveBeenCalledTimes(1);
    expect(maintenance.scheduler.start).not.toHaveBeenCalled();
    expect(logs).toContain("[daemon] maintenance scheduler disabled");

    await result.shutdown("test");
  });

  it("runs with only maintenance enabled without starting autonomy", async () => {
    const { autonomy, borg, maintenance } = createBorg({
      autonomyEnabled: false,
      maintenanceEnabled: true,
    });
    const { logs, result } = await startDaemon(borg);

    expect(result.status).toBe("started");
    expect(autonomy.scheduler.start).not.toHaveBeenCalled();
    expect(maintenance.scheduler.start).toHaveBeenCalledTimes(1);
    expect(logs).toContain("[daemon] autonomy scheduler disabled");

    await result.shutdown("test");
  });

  it("exits early with a clear message when both schedulers are disabled", async () => {
    const { autonomy, borg, maintenance } = createBorg({
      autonomyEnabled: false,
      maintenanceEnabled: false,
    });
    const { logs, result } = await startDaemon(borg);

    expect(result.status).toBe("disabled");
    expect(autonomy.scheduler.start).not.toHaveBeenCalled();
    expect(maintenance.scheduler.start).not.toHaveBeenCalled();
    expect(borg.close).toHaveBeenCalledTimes(1);
    expect(logs).toContain("[daemon] autonomy and maintenance disabled; exiting");
  });

  it("stops both schedulers during signal shutdown even when one stop throws", async () => {
    const stopFailure = new Error("autonomy stop failed");
    const { autonomy, borg, maintenance } = createBorg({
      autonomyEnabled: true,
      maintenanceEnabled: true,
      autonomyStop: async () => {
        throw stopFailure;
      },
    });
    let resolveExit: (() => void) | undefined;
    const exited = new Promise<void>((resolve) => {
      resolveExit = resolve;
    });
    const logs: string[] = [];
    const exits: number[] = [];
    const signals = new EventEmitter();

    await runDaemon({
      openBorg: async () => borg,
      writeStderr: (line) => {
        logs.push(line);
      },
      signalTarget: signals,
      exit: (code = 0) => {
        exits.push(code);
        resolveExit?.();
      },
    });

    signals.emit("SIGTERM");
    await exited;

    expect(autonomy.scheduler.stop).toHaveBeenCalledWith({ graceful: true });
    expect(maintenance.scheduler.stop).toHaveBeenCalledWith({ graceful: true });
    expect(borg.close).toHaveBeenCalledTimes(1);
    expect(exits).toEqual([1]);
    expect(logs).toContain("[daemon] autonomy scheduler stop failed Error: autonomy stop failed");
    expect(logs).toContain(
      "[daemon] shutdown-error AggregateError: Daemon shutdown completed with errors",
    );
  });
});
