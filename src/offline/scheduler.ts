// Schedules offline maintenance runs on two cadences (light/heavy).
// Separate from the autonomy scheduler: maintenance is housekeeping, not cognition,
// so it runs on its own interval loop with a busy-detection hook.

import { performance } from "node:perf_hooks";

import { SystemClock, type Clock } from "../util/clock.js";
import { ConfigError } from "../util/errors.js";
import {
  normalizeOptimizeError,
  type LanceDbOptimizeStorageResult,
} from "../storage/lancedb/index.js";
import type { TurnTracer } from "../tracing/tracer.js";

import type { MaintenanceOrchestrator } from "./orchestrator.js";
import type { OfflineProcess, OfflineProcessName, OrchestratorResult } from "./types.js";

type IntervalHandle = ReturnType<typeof setInterval>;

export type MaintenanceCadence = "light" | "heavy";

export type MaintenanceTickResult = {
  status: "ok" | "skipped_busy" | "skipped_empty" | "disabled";
  cadence: MaintenanceCadence;
  ts: number;
  processes: OfflineProcessName[];
  result: OrchestratorResult | null;
  storageOptimization?: LanceDbOptimizeStorageResult | null;
  reason?: string;
};

export type MaintenanceSchedulerObserver = {
  onTick?(result: MaintenanceTickResult): void | Promise<void>;
  onError?(error: unknown, cadence: MaintenanceCadence): void | Promise<void>;
};

export type MaintenanceSchedulerStopOptions = {
  graceful?: boolean;
};

export type MaintenanceSchedulerOptions = {
  enabled: boolean;
  lightIntervalMs: number;
  heavyIntervalMs: number;
  lightProcesses: readonly OfflineProcessName[];
  heavyProcesses: readonly OfflineProcessName[];
  orchestrator: MaintenanceOrchestrator;
  processRegistry: Record<OfflineProcessName, OfflineProcess>;
  optimizeStorage?: boolean;
  storageOptimizer?: () => Promise<LanceDbOptimizeStorageResult>;
  tracer?: TurnTracer;
  clock?: Clock;
  isBusy?: () => boolean;
  setIntervalFn?: typeof setInterval;
  clearIntervalFn?: typeof clearInterval;
};

export class MaintenanceScheduler {
  private readonly clock: Clock;
  private readonly setIntervalFn: typeof setInterval;
  private readonly clearIntervalFn: typeof clearInterval;
  private lightHandle: IntervalHandle | null = null;
  private heavyHandle: IntervalHandle | null = null;
  private readonly activeTicks: Record<MaintenanceCadence, Promise<MaintenanceTickResult> | null> =
    {
      light: null,
      heavy: null,
    };
  private observer: MaintenanceSchedulerObserver | null = null;

  constructor(private readonly options: MaintenanceSchedulerOptions) {
    // Defense in depth: the orchestrator serializes maintenance work because
    // processes share stores. The cadence split should still avoid scheduling
    // the same process through two interval groups.
    const overlappingProcesses = options.lightProcesses.filter((process) =>
      options.heavyProcesses.includes(process),
    );

    if (overlappingProcesses.length > 0) {
      throw new ConfigError(
        `Maintenance light/heavy process sets must be disjoint; overlapping processes: ${[
          ...new Set(overlappingProcesses),
        ].join(", ")}`,
        {
          code: "MAINTENANCE_PROCESS_CADENCE_OVERLAP",
        },
      );
    }

    this.clock = options.clock ?? new SystemClock();
    this.setIntervalFn = options.setIntervalFn ?? setInterval;
    this.clearIntervalFn = options.clearIntervalFn ?? clearInterval;
  }

  setObserver(observer: MaintenanceSchedulerObserver | null): void {
    this.observer = observer;
  }

  isEnabled(): boolean {
    return this.options.enabled;
  }

  start(): void {
    if (!this.options.enabled) {
      return;
    }

    if (this.lightHandle === null) {
      this.lightHandle = this.setIntervalFn(() => {
        void this.runScheduledTick("light");
      }, this.options.lightIntervalMs);
    }

    if (this.heavyHandle === null) {
      this.heavyHandle = this.setIntervalFn(() => {
        void this.runScheduledTick("heavy");
      }, this.options.heavyIntervalMs);
    }
  }

  async stop(options: MaintenanceSchedulerStopOptions = {}): Promise<void> {
    if (this.lightHandle !== null) {
      this.clearIntervalFn(this.lightHandle);
      this.lightHandle = null;
    }

    if (this.heavyHandle !== null) {
      this.clearIntervalFn(this.heavyHandle);
      this.heavyHandle = null;
    }

    if (options.graceful === false) {
      return;
    }

    for (const cadence of ["light", "heavy"] as const) {
      const activeTick = this.activeTicks[cadence];

      if (activeTick !== null) {
        try {
          await activeTick;
        } catch {
          // Active tick errors were already surfaced via the observer;
          // stop() must not propagate them.
        }
      }
    }
  }

  async tick(cadence: MaintenanceCadence): Promise<MaintenanceTickResult> {
    return this.runTrackedTick(cadence, { notifyObserver: false });
  }

  private processNamesFor(cadence: MaintenanceCadence): readonly OfflineProcessName[] {
    return cadence === "light" ? this.options.lightProcesses : this.options.heavyProcesses;
  }

  private shouldOptimizeStorage(cadence: MaintenanceCadence): boolean {
    return (
      cadence === "heavy" &&
      this.options.optimizeStorage === true &&
      this.options.storageOptimizer !== undefined
    );
  }

  private emitStorageOptimizationCompleted(input: {
    cadence: MaintenanceCadence;
    ts: number;
    runId?: string;
    result: LanceDbOptimizeStorageResult;
  }): void {
    if (this.options.tracer?.enabled !== true) {
      return;
    }

    const successfulTables = input.result.tables.filter((table) => table.status === "ok");
    const errorCount =
      input.result.tables.length -
      successfulTables.length +
      (input.result.error === undefined ? 0 : 1);
    const tables: Array<Record<string, number | string>> = input.result.tables.map((table) => {
      if (table.status === "ok") {
        return {
          table: table.table,
          status: table.status,
          fragments_removed: table.fragmentsRemoved,
          fragments_added: table.fragmentsAdded,
          versions_pruned: table.versionsPruned,
          bytes_removed: table.bytesRemoved,
          duration_ms: table.durationMs,
        };
      }

      const errorTable: Record<string, number | string> = {
        table: table.table,
        status: table.status,
        duration_ms: table.durationMs,
        error_message: table.error.message,
      };

      if (table.error.code !== undefined) {
        errorTable.error_code = table.error.code;
      }

      return errorTable;
    });

    this.options.tracer.emit("storage.optimize.completed", {
      turnId: input.runId ?? `maintenance_storage_${input.cadence}_${input.ts}`,
      cadence: input.cadence,
      table_count: input.result.tables.length,
      errors: errorCount,
      fragments_removed: successfulTables.reduce((sum, table) => sum + table.fragmentsRemoved, 0),
      fragments_added: successfulTables.reduce((sum, table) => sum + table.fragmentsAdded, 0),
      versions_pruned: successfulTables.reduce((sum, table) => sum + table.versionsPruned, 0),
      duration_ms: input.result.durationMs,
      tables,
      ...(input.result.error === undefined
        ? {}
        : {
            optimizer_error_message: input.result.error.message,
            ...(input.result.error.code === undefined
              ? {}
              : { optimizer_error_code: input.result.error.code }),
          }),
    });
  }

  private createStorageOptimizationFailureResult(input: {
    error: unknown;
    startedAt: number;
  }): LanceDbOptimizeStorageResult {
    return {
      durationMs: Math.round(performance.now() - input.startedAt),
      tables: [],
      error: normalizeOptimizeError(input.error),
    };
  }

  private async optimizeStorageAfterHeavy(input: {
    cadence: MaintenanceCadence;
    ts: number;
    runId?: string;
  }): Promise<LanceDbOptimizeStorageResult | null> {
    if (!this.shouldOptimizeStorage(input.cadence)) {
      return null;
    }

    const startedAt = performance.now();
    let result: LanceDbOptimizeStorageResult;

    try {
      result = await this.options.storageOptimizer!();
    } catch (error) {
      result = this.createStorageOptimizationFailureResult({
        error,
        startedAt,
      });
    }

    this.emitStorageOptimizationCompleted({
      cadence: input.cadence,
      ts: input.ts,
      runId: input.runId,
      result,
    });
    return result;
  }

  private async tickOnce(cadence: MaintenanceCadence): Promise<MaintenanceTickResult> {
    const ts = this.clock.now();
    const processes = this.processNamesFor(cadence);
    const shouldOptimizeStorage = this.shouldOptimizeStorage(cadence);

    if (!this.options.enabled) {
      return {
        status: "disabled",
        cadence,
        ts,
        processes: [...processes],
        result: null,
        storageOptimization: null,
        reason: "Maintenance scheduler is disabled.",
      };
    }

    if (processes.length === 0 && !shouldOptimizeStorage) {
      return {
        status: "skipped_empty",
        cadence,
        ts,
        processes: [],
        result: null,
        storageOptimization: null,
        reason: `No processes configured for the ${cadence} cadence.`,
      };
    }

    if (this.options.isBusy?.() === true) {
      return {
        status: "skipped_busy",
        cadence,
        ts,
        processes: [...processes],
        result: null,
        storageOptimization: null,
        reason: "Skipped because the system is busy.",
      };
    }

    let result: OrchestratorResult | null = null;
    let storageOptimization: LanceDbOptimizeStorageResult | null = null;

    if (processes.length > 0) {
      const offlineProcesses = processes
        .map((name) => this.options.processRegistry[name])
        .filter((process): process is OfflineProcess => process !== undefined);
      result = await this.options.orchestrator.run({
        processes: offlineProcesses,
        ...(shouldOptimizeStorage
          ? {
              afterRun: async (runResult) => {
                storageOptimization = await this.optimizeStorageAfterHeavy({
                  cadence,
                  ts,
                  runId: runResult.run_id,
                });
              },
            }
          : {}),
      });
    } else if (shouldOptimizeStorage) {
      storageOptimization = await this.options.orchestrator.runMechanicalMaintenance(() =>
        this.optimizeStorageAfterHeavy({
          cadence,
          ts,
        }),
      );
    }

    return {
      status: "ok",
      cadence,
      ts,
      processes: [...processes],
      result,
      storageOptimization,
    };
  }

  private runTrackedTick(
    cadence: MaintenanceCadence,
    options: { notifyObserver: boolean },
  ): Promise<MaintenanceTickResult> {
    const existing = this.activeTicks[cadence];

    if (existing !== null) {
      return existing;
    }

    const promise = (async () => {
      try {
        const result = await this.tickOnce(cadence);

        if (options.notifyObserver) {
          await this.notifyTick(result);
        }

        return result;
      } catch (error) {
        if (options.notifyObserver) {
          await this.notifyError(error, cadence);
        }

        throw error;
      }
    })().finally(() => {
      if (this.activeTicks[cadence] === promise) {
        this.activeTicks[cadence] = null;
      }
    });

    this.activeTicks[cadence] = promise;
    return promise;
  }

  private async runScheduledTick(cadence: MaintenanceCadence): Promise<void> {
    // Guard only against duplicate same-cadence interval callbacks here.
    // Cross-cadence work is queued by the orchestrator so heavy and light
    // ticks both run without sharing stores concurrently.
    if (this.activeTicks[cadence] !== null) {
      return;
    }

    try {
      await this.runTrackedTick(cadence, { notifyObserver: true });
    } catch {
      // Scheduled ticks report failures through notifyError; the interval loop
      // must not surface an unhandled rejection.
    }
  }

  private async notifyTick(result: MaintenanceTickResult): Promise<void> {
    try {
      await this.observer?.onTick?.(result);
    } catch (error) {
      await this.notifyError(error, result.cadence);
    }
  }

  private async notifyError(error: unknown, cadence: MaintenanceCadence): Promise<void> {
    try {
      await this.observer?.onError?.(error, cadence);
    } catch {
      // Observer failures must not stop the scheduler loop.
    }
  }
}
