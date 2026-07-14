import { describe, expect, it, vi } from "vitest";

import type { Borg } from "../borg.js";
import type { BorgDreamOptions } from "../borg/types.js";
import type {
  OfflineProcessError,
  OfflineProcessName,
  OrchestratorResult,
} from "../offline/types.js";
import type { MaintenanceRunId } from "../util/ids.js";
import {
  MemoryMaintenanceCoordinator,
  memorySelfNameFromEnv,
  type MemoryMaintenanceConfig,
} from "./memory-maintenance.js";

function deferred<T>() {
  let resolve!: (value: T) => void;
  const promise = new Promise<T>((done) => {
    resolve = done;
  });
  return { promise, resolve };
}

function resultFor(
  runId: MaintenanceRunId,
  process: OfflineProcessName,
  options: {
    tokens?: number;
    changes?: number;
    errors?: number;
    errorDetails?: OfflineProcessError[];
    budgetExhausted?: boolean;
  } = {},
): OrchestratorResult {
  const changes = Array.from({ length: options.changes ?? 0 }, (_, index) => ({
    process,
    action: `change-${index}`,
    targets: {},
  }));
  const errors =
    options.errorDetails ??
    Array.from({ length: options.errors ?? 0 }, (_, index) => ({
      process,
      message: `error-${index}`,
    }));
  const processResult = {
    process,
    dryRun: false,
    changes,
    tokens_used: options.tokens ?? 0,
    errors,
    budget_exhausted: options.budgetExhausted ?? false,
  };
  return {
    run_id: runId,
    dryRun: false,
    results: [processResult],
    changes,
    tokens_used: processResult.tokens_used,
    errors,
  };
}

function maintenanceConfig(
  overrides: Partial<MemoryMaintenanceConfig> = {},
): MemoryMaintenanceConfig {
  return {
    enabled: true,
    lightProcesses: ["consolidator"],
    heavyProcesses: ["consolidator", "reflector"],
    lightBudget: null,
    heavyBudget: null,
    processBudgets: {},
    ...overrides,
  };
}

describe("memory maintenance coordinator", () => {
  it("rejects disabled and shutting-down starts without scheduling work", () => {
    const scheduled: Array<() => void> = [];
    const pool = {
      withTenant: async () => {
        throw new Error("maintenance must not open a being");
      },
    };
    const disabled = new MemoryMaintenanceCoordinator({
      pool,
      config: maintenanceConfig({ enabled: false }),
      schedule: (task) => scheduled.push(task),
      writeReport: () => {},
    });
    expect(disabled.tryStart({ tenant: "alpha", mode: "light", dryRun: false })).toEqual({
      status: "disabled",
    });

    const shuttingDown = new MemoryMaintenanceCoordinator({
      pool,
      config: maintenanceConfig(),
      schedule: (task) => scheduled.push(task),
      writeReport: () => {},
    });
    expect(shuttingDown.beginShutdown()).toEqual([]);
    expect(shuttingDown.tryStart({ tenant: "alpha", mode: "light", dryRun: false })).toEqual({
      status: "shutting_down",
    });
    expect(scheduled).toEqual([]);
  });

  it("reserves synchronously and can clear a failed readiness admission without reporting a run", async () => {
    const scheduled: Array<() => void> = [];
    const lines: string[] = [];
    const coordinator = new MemoryMaintenanceCoordinator({
      pool: {
        withTenant: async () => {
          throw new Error("a reservation alone must not open a being");
        },
      },
      config: maintenanceConfig(),
      schedule: (task) => scheduled.push(task),
      writeReport: (line) => lines.push(line),
    });

    const reserved = coordinator.tryReserve({
      tenant: "alpha",
      mode: "light",
      dryRun: false,
    });
    if (reserved.status !== "accepted") {
      throw new Error("expected accepted reservation");
    }
    expect(coordinator.hasReservation("alpha", reserved.runId)).toBe(true);
    expect(coordinator.tryReserve({ tenant: "alpha", mode: "heavy", dryRun: false })).toEqual({
      status: "conflict",
      runId: reserved.runId,
    });
    expect(scheduled).toEqual([]);

    expect(coordinator.cancelReservation("alpha", reserved.runId)).toBe(true);
    expect((await reserved.completion).state).toBe("aborted");
    expect(coordinator.getStatus("alpha")).toEqual({ current: null, last: null });
    expect(lines).toEqual([]);
  });

  it("runs one exclusive reservation per process and then optimizes live heavy storage", async () => {
    const dreamCalls: BorgDreamOptions[] = [];
    const optimizeStorage = vi.fn(async () => ({ durationMs: 7, tables: [] }));
    const reservations: Array<{ tenant: string; exclusive: boolean }> = [];
    const yields: string[] = [];
    const lines: string[] = [];
    const borg = {
      dream: async (options: BorgDreamOptions = {}) => {
        dreamCalls.push(options);
        const process = options.processes![0]!;
        return resultFor(options.runId!, process, {
          tokens: process === "consolidator" ? 20 : 70,
          changes: 1,
        });
      },
      maintenance: { optimizeStorage },
    } as unknown as Borg;
    const coordinator = new MemoryMaintenanceCoordinator({
      pool: {
        withTenant: async (tenant, fn, opts) => {
          reservations.push({ tenant, exclusive: opts?.exclusive === true });
          return fn(borg);
        },
      },
      config: maintenanceConfig({
        heavyBudget: 100,
        processBudgets: { consolidator: 30, reflector: 80 },
      }),
      schedule: (task) => task(),
      yieldBetweenChunks: async () => {
        yields.push("yield");
      },
      writeReport: (line) => lines.push(line),
    });

    const started = coordinator.tryStart({ tenant: "alpha", mode: "heavy", dryRun: false });
    expect(started.status).toBe("accepted");
    if (started.status !== "accepted") {
      throw new Error("expected accepted run");
    }
    const report = await started.completion;

    expect(dreamCalls.map((call) => call.processes)).toEqual([["consolidator"], ["reflector"]]);
    expect(dreamCalls.map((call) => call.budget)).toEqual([30, 80]);
    expect(dreamCalls.every((call) => call.runId === started.runId)).toBe(true);
    expect(optimizeStorage).toHaveBeenCalledWith({ runId: started.runId });
    expect(reservations).toEqual([
      { tenant: "alpha", exclusive: true },
      { tenant: "alpha", exclusive: true },
      { tenant: "alpha", exclusive: true },
    ]);
    expect(yields).toEqual(["yield", "yield"]);
    expect(report.processes.map((process) => process.tokens_used)).toEqual([20, 70]);
    expect(report.requested_processes).toEqual(["consolidator", "reflector"]);
    expect(report.resolved_processes).toEqual(["consolidator", "reflector"]);
    expect(report.processes.map((process) => process.name)).toEqual(["consolidator", "reflector"]);
    expect(report.budget_remaining).toBe(10);
    expect(report.errors_total).toBe(0);
    expect(report.processes.every((process) => process.error_details.length === 0)).toBe(true);
    expect(report.storage_optimize).toMatchObject({ status: "completed", errors: 0 });
    expect(lines).toHaveLength(1);
    expect(JSON.parse(lines[0]!)).toEqual({
      evt: "memory_maintenance",
      tenant: "alpha",
      mode: "heavy",
      run_id: started.runId,
      dryRun: false,
      processes: report.processes.map((process) => ({
        name: process.name,
        changes: process.changes,
        errors: process.errors,
        error_details: [],
        tokens_used: process.tokens_used,
        budget_exhausted: process.budget_exhausted,
        duration_ms: process.duration_ms,
      })),
      storage_optimize: report.storage_optimize,
      total_duration_ms: report.total_duration_ms,
      errors_total: 0,
    });
  });

  it("keeps a heavy dry run non-mutating by skipping storage optimization exactly", async () => {
    const optimizeStorage = vi.fn();
    const borg = {
      dream: async (options: BorgDreamOptions = {}) =>
        resultFor(options.runId!, options.processes![0]!),
      maintenance: { optimizeStorage },
    } as unknown as Borg;
    const coordinator = new MemoryMaintenanceCoordinator({
      pool: { withTenant: async (_tenant, fn) => fn(borg) },
      config: maintenanceConfig({ heavyProcesses: ["consolidator"] }),
      schedule: (task) => task(),
      yieldBetweenChunks: async () => {},
      writeReport: () => {},
    });

    const started = coordinator.tryStart({ tenant: "alpha", mode: "heavy", dryRun: true });
    if (started.status !== "accepted") {
      throw new Error("expected accepted run");
    }
    const report = await started.completion;

    expect(optimizeStorage).not.toHaveBeenCalled();
    expect(report.storage_optimize).toEqual({ status: "skipped", reason: "dryRun" });
  });

  it("stops before the next process when the aggregate run budget is exhausted", async () => {
    const dream = vi.fn(async (options: BorgDreamOptions = {}) =>
      resultFor(options.runId!, options.processes![0]!, {
        tokens: 7,
        budgetExhausted: true,
      }),
    );
    const borg = { dream, maintenance: { optimizeStorage: vi.fn() } } as unknown as Borg;
    const lines: string[] = [];
    const coordinator = new MemoryMaintenanceCoordinator({
      pool: { withTenant: async (_tenant, fn) => fn(borg) },
      config: maintenanceConfig({
        lightProcesses: ["consolidator", "reflector"],
        lightBudget: 10,
        processBudgets: { consolidator: 100, reflector: 100 },
      }),
      schedule: (task) => task(),
      yieldBetweenChunks: async () => {},
      writeReport: (line) => lines.push(line),
    });

    const started = coordinator.tryStart({ tenant: "alpha", mode: "light", dryRun: false });
    if (started.status !== "accepted") {
      throw new Error("expected accepted run");
    }
    const report = await started.completion;

    expect(dream).toHaveBeenCalledTimes(1);
    expect(dream.mock.calls[0]?.[0]?.budget).toBe(10);
    expect(report.budget_remaining).toBe(0);
    expect(report.processes[1]).toMatchObject({
      name: "reflector",
      status: "skipped",
      skipped: "budget",
    });
    expect(report.resolved_processes).toEqual(["consolidator"]);
    expect(JSON.parse(lines[0]!)).toMatchObject({
      processes: [
        expect.objectContaining({ name: "consolidator", tokens_used: 7 }),
        { name: "reflector", skipped: "budget" },
      ],
    });
  });

  it("records an in-flight chunk before aborting the remaining run", async () => {
    const firstChunk = deferred<OrchestratorResult>();
    const dream = vi.fn(() => firstChunk.promise);
    const optimizeStorage = vi.fn();
    const borg = { dream, maintenance: { optimizeStorage } } as unknown as Borg;
    const lines: string[] = [];
    const coordinator = new MemoryMaintenanceCoordinator({
      pool: { withTenant: async (_tenant, fn) => fn(borg) },
      config: maintenanceConfig(),
      schedule: (task) => task(),
      yieldBetweenChunks: async () => {},
      writeReport: (line) => lines.push(line),
    });

    const started = coordinator.tryStart({ tenant: "alpha", mode: "heavy", dryRun: false });
    if (started.status !== "accepted") {
      throw new Error("expected accepted run");
    }
    const conflict = coordinator.tryStart({ tenant: "alpha", mode: "light", dryRun: false });
    expect(conflict).toEqual({ status: "conflict", runId: started.runId });
    expect(coordinator.getStatus("alpha")).toMatchObject({
      current: {
        run_id: started.runId,
        state: "running",
        requested_processes: ["consolidator", "reflector"],
        resolved_processes: ["consolidator"],
        errors_total: 0,
      },
      last: null,
    });

    expect(coordinator.beginShutdown()).toEqual([started.runId]);
    let settled = false;
    void started.completion.then(() => {
      settled = true;
    });
    await Promise.resolve();
    expect(settled).toBe(false);
    expect(lines).toEqual([]);

    firstChunk.resolve(resultFor(started.runId, "consolidator", { changes: 1 }));
    const report = await started.completion;

    expect(report.state).toBe("aborted");
    expect(report.processes[0]).toMatchObject({
      name: "consolidator",
      status: "completed",
      changes: 1,
    });
    expect(report.processes[1]).toMatchObject({
      name: "reflector",
      status: "skipped",
      skipped: "shutdown",
    });
    expect(report.storage_optimize).toEqual({ status: "skipped", reason: "shutdown" });
    expect(dream).toHaveBeenCalledTimes(1);
    expect(optimizeStorage).not.toHaveBeenCalled();
    expect(lines).toHaveLength(1);
    expect(coordinator.forceFinalizeAborted()).toEqual([]);
    expect(coordinator.beginShutdown()).toEqual([]);
    expect(coordinator.getStatus("alpha")).toMatchObject({
      current: null,
      last: { run_id: started.runId, state: "aborted" },
    });
  });

  it("checks shutdown inside a queued exclusive reservation before starting work", async () => {
    const acquired = deferred<void>();
    const release = deferred<void>();
    const dream = vi.fn();
    const borg = { dream, maintenance: { optimizeStorage: vi.fn() } } as unknown as Borg;
    const coordinator = new MemoryMaintenanceCoordinator({
      pool: {
        withTenant: async (_tenant, fn) => {
          acquired.resolve(undefined);
          await release.promise;
          return fn(borg);
        },
      },
      config: maintenanceConfig(),
      schedule: (task) => task(),
      writeReport: () => {},
    });

    const started = coordinator.tryStart({ tenant: "alpha", mode: "heavy", dryRun: false });
    if (started.status !== "accepted") {
      throw new Error("expected accepted run");
    }
    await acquired.promise;
    expect(coordinator.beginShutdown()).toEqual([started.runId]);
    release.resolve(undefined);
    const report = await started.completion;

    expect(dream).not.toHaveBeenCalled();
    expect(report.state).toBe("aborted");
    expect(report.resolved_processes).toEqual([]);
    expect(report.processes).toEqual([
      expect.objectContaining({ name: "consolidator", skipped: "shutdown" }),
      expect.objectContaining({ name: "reflector", skipped: "shutdown" }),
    ]);
  });

  it("force-finalizes an in-flight run once when the shared deadline expires", async () => {
    const chunk = deferred<OrchestratorResult>();
    const lines: string[] = [];
    const borg = {
      dream: () => chunk.promise,
      maintenance: { optimizeStorage: vi.fn() },
    } as unknown as Borg;
    const coordinator = new MemoryMaintenanceCoordinator({
      pool: { withTenant: async (_tenant, fn) => fn(borg) },
      config: maintenanceConfig(),
      schedule: (task) => task(),
      writeReport: (line) => lines.push(line),
    });
    const started = coordinator.tryStart({ tenant: "alpha", mode: "heavy", dryRun: false });
    if (started.status !== "accepted") {
      throw new Error("expected accepted run");
    }

    coordinator.beginShutdown();
    expect(coordinator.forceFinalizeAborted()).toEqual([started.runId]);
    const report = await started.completion;
    expect(report.state).toBe("aborted");
    expect(lines).toHaveLength(1);

    chunk.resolve(resultFor(started.runId, "consolidator", { changes: 1 }));
    await Promise.resolve();
    await Promise.resolve();
    expect(coordinator.forceFinalizeAborted()).toEqual([]);
    expect(lines).toHaveLength(1);
  });

  it("surfaces bounded process error details in status, stdout, and the error event", async () => {
    const lines: string[] = [];
    const errorLines: string[] = [];
    const longMessage = "x".repeat(350);
    const reportedErrors: OfflineProcessError[] = [
      { process: "consolidator", message: longMessage, code: "E_LONG" },
      { process: "consolidator", message: "second failure" },
      { process: "consolidator", message: "third failure", code: "E_THIRD" },
      { process: "consolidator", message: "fourth failure" },
    ];
    const expectedDetails = [
      { process: "consolidator", message: "x".repeat(300), code: "E_LONG" },
      { process: "consolidator", message: "second failure" },
      { process: "consolidator", message: "third failure", code: "E_THIRD" },
    ];
    const borg = {
      dream: async (options: BorgDreamOptions = {}) =>
        resultFor(options.runId!, options.processes![0]!, { errorDetails: reportedErrors }),
      maintenance: { optimizeStorage: vi.fn() },
    } as unknown as Borg;
    const coordinator = new MemoryMaintenanceCoordinator({
      pool: { withTenant: async (_tenant, fn) => fn(borg) },
      config: maintenanceConfig(),
      schedule: (task) => task(),
      writeReport: (line) => lines.push(line),
      writeError: (message) => errorLines.push(message),
    });

    const started = coordinator.tryStart({ tenant: "alpha", mode: "light", dryRun: false });
    if (started.status !== "accepted") {
      throw new Error("expected accepted run");
    }
    const report = await started.completion;

    expect(report.state).toBe("completed_with_errors");
    expect(report.errors_total).toBe(4);
    expect(report.processes[0]).toMatchObject({
      errors: 4,
      error_details: expectedDetails,
    });
    expect(report.processes[0]!.error_details[0]!.message).toHaveLength(300);
    expect(coordinator.getStatus("alpha").last?.processes[0]?.error_details).toEqual(
      expectedDetails,
    );
    expect(lines).toHaveLength(1);
    expect(JSON.parse(lines[0]!)).toMatchObject({
      run_id: started.runId,
      processes: [{ name: "consolidator", errors: 4, error_details: expectedDetails }],
    });
    expect(errorLines).toHaveLength(1);
    expect(JSON.parse(errorLines[0]!)).toEqual({
      evt: "memory_maintenance_error",
      tenant: "alpha",
      run_id: started.runId,
      mode: "light",
      errors_total: 4,
      error_details: expectedDetails,
    });
  });

  it("normalizes a thrown process chunk into the same error details", async () => {
    const errorLines: string[] = [];
    const chunkError = Object.assign(new Error("process chunk failed"), {
      code: "E_PROCESS_CHUNK",
    });
    const borg = {
      dream: async () => {
        throw chunkError;
      },
      maintenance: { optimizeStorage: vi.fn() },
    } as unknown as Borg;
    const coordinator = new MemoryMaintenanceCoordinator({
      pool: { withTenant: async (_tenant, fn) => fn(borg) },
      config: maintenanceConfig(),
      schedule: (task) => task(),
      writeReport: () => {},
      writeError: (line) => errorLines.push(line),
    });

    const started = coordinator.tryStart({ tenant: "alpha", mode: "light", dryRun: false });
    if (started.status !== "accepted") {
      throw new Error("expected accepted run");
    }
    const report = await started.completion;
    const expectedDetails = [
      {
        process: "consolidator",
        message: "process chunk failed",
        code: "E_PROCESS_CHUNK",
      },
    ];

    expect(report.processes[0]).toMatchObject({ errors: 1, error_details: expectedDetails });
    expect(JSON.parse(errorLines[0]!)).toMatchObject({ error_details: expectedDetails });
  });

  it("normalizes a thrown storage chunk into the storage report and error event", async () => {
    const lines: string[] = [];
    const errorLines: string[] = [];
    const storageError = Object.assign(new Error("storage chunk failed"), {
      code: "E_STORAGE_CHUNK",
    });
    const borg = {
      dream: vi.fn(),
      maintenance: {
        optimizeStorage: async () => {
          throw storageError;
        },
      },
    } as unknown as Borg;
    const coordinator = new MemoryMaintenanceCoordinator({
      pool: { withTenant: async (_tenant, fn) => fn(borg) },
      config: maintenanceConfig({ heavyProcesses: [] }),
      schedule: (task) => task(),
      writeReport: (line) => lines.push(line),
      writeError: (line) => errorLines.push(line),
    });

    const started = coordinator.tryStart({ tenant: "alpha", mode: "heavy", dryRun: false });
    if (started.status !== "accepted") {
      throw new Error("expected accepted run");
    }
    const report = await started.completion;
    const expectedDetails = [{ message: "storage chunk failed", code: "E_STORAGE_CHUNK" }];

    expect(report.storage_optimize).toMatchObject({
      status: "error",
      errors: 1,
      error_details: expectedDetails,
    });
    expect(JSON.parse(lines[0]!).storage_optimize).toMatchObject({
      error_details: expectedDetails,
    });
    expect(JSON.parse(errorLines[0]!)).toMatchObject({
      errors_total: 1,
      error_details: expectedDetails,
    });
  });

  it("caps completed tenant records at 64", async () => {
    const coordinator = new MemoryMaintenanceCoordinator({
      pool: {
        withTenant: async () => {
          throw new Error("empty maintenance must not open a being");
        },
      },
      config: maintenanceConfig({ lightProcesses: [] }),
      maxLastTenants: 100,
      schedule: (task) => task(),
      writeReport: () => {},
    });

    for (let index = 0; index < 65; index += 1) {
      const started = coordinator.tryStart({
        tenant: `tenant-${index}`,
        mode: "light",
        dryRun: false,
      });
      if (started.status !== "accepted") {
        throw new Error("expected accepted run");
      }
      await started.completion;
    }

    expect(coordinator.getStatus("tenant-0").last).toBeNull();
    expect(coordinator.getStatus("tenant-1").last).not.toBeNull();
    expect(coordinator.getStatus("tenant-64").last).not.toBeNull();
  });

  it("trims the configured sidecar self name and defaults blank input", () => {
    expect(memorySelfNameFromEnv({ BORG_MEMORY_SELF_NAME: "  Atlas Agent  " })).toBe("Atlas Agent");
    expect(memorySelfNameFromEnv({ BORG_MEMORY_SELF_NAME: "   " })).toBe("team-agent");
    expect(memorySelfNameFromEnv({})).toBe("team-agent");
  });
});
