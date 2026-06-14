// Schedules offline maintenance runs on two cadences (light/heavy).
// Separate from the autonomy scheduler: maintenance is housekeeping, not cognition,
// so it runs on its own durable timer loop with a busy-detection hook.

import { performance } from "node:perf_hooks";

import { SystemClock, type Clock } from "../util/clock.js";
import { ConfigError } from "../util/errors.js";
import {
  normalizeOptimizeError,
  type LanceDbOptimizeStorageResult,
} from "../storage/lancedb/index.js";
import type { StreamWatermark, StreamWatermarkRepository } from "../stream/index.js";
import type { TurnTracer } from "../tracing/tracer.js";
import type { SessionId } from "../util/ids.js";

import type { MaintenanceOrchestrator } from "./orchestrator.js";
import type { OfflineProcess, OfflineProcessName, OrchestratorResult } from "./types.js";

type TimeoutHandle = ReturnType<typeof setTimeout>;
type SetTimeoutFn = (callback: () => void, delayMs: number) => TimeoutHandle;
type ClearTimeoutFn = (handle: TimeoutHandle) => void;

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
  cadenceWatermarkRepository: Pick<StreamWatermarkRepository, "get" | "set">;
  startupGraceMs: number;
  busyRetryBaseMs: number;
  busyRetryMaxMs: number;
  tracer?: TurnTracer;
  clock?: Clock;
  isBusy?: () => boolean;
  setTimeoutFn?: SetTimeoutFn;
  clearTimeoutFn?: ClearTimeoutFn;
};

type CadenceTimerState = {
  timer: TimeoutHandle | null;
  retryDelayMs: number | null;
};

const MAINTENANCE_CADENCE_WATERMARK_SESSION_ID = "maintenance-global" as SessionId;

function cadenceWatermarkProcessName(cadence: MaintenanceCadence): string {
  return `maintenance:cadence:${cadence}`;
}

export class MaintenanceScheduler {
  private readonly clock: Clock;
  private readonly setTimeoutFn: SetTimeoutFn;
  private readonly clearTimeoutFn: ClearTimeoutFn;
  private readonly timers: Record<MaintenanceCadence, CadenceTimerState> = {
    light: {
      timer: null,
      retryDelayMs: null,
    },
    heavy: {
      timer: null,
      retryDelayMs: null,
    },
  };
  private readonly activeTicks: Record<MaintenanceCadence, Promise<MaintenanceTickResult> | null> =
    {
      light: null,
      heavy: null,
    };
  private observer: MaintenanceSchedulerObserver | null = null;
  private started = false;
  private stopping = false;

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
    this.setTimeoutFn =
      options.setTimeoutFn ?? ((callback, delayMs) => setTimeout(callback, delayMs));
    this.clearTimeoutFn = options.clearTimeoutFn ?? ((handle) => clearTimeout(handle));
  }

  setObserver(observer: MaintenanceSchedulerObserver | null): void {
    this.observer = observer;
  }

  isEnabled(): boolean {
    return this.options.enabled;
  }

  start(): void {
    if (!this.options.enabled || this.started) {
      return;
    }

    this.started = true;
    this.stopping = false;
    this.scheduleCadence("light", "startup");
    this.scheduleCadence("heavy", "startup");
  }

  async stop(options: MaintenanceSchedulerStopOptions = {}): Promise<void> {
    this.started = false;
    this.stopping = true;

    for (const cadence of ["light", "heavy"] as const) {
      this.clearCadenceTimer(cadence);
      this.timers[cadence].retryDelayMs = null;
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

  private intervalMsFor(cadence: MaintenanceCadence): number {
    return cadence === "light" ? this.options.lightIntervalMs : this.options.heavyIntervalMs;
  }

  private readCadenceWatermark(cadence: MaintenanceCadence): StreamWatermark | null {
    return this.options.cadenceWatermarkRepository.get(
      cadenceWatermarkProcessName(cadence),
      MAINTENANCE_CADENCE_WATERMARK_SESSION_ID,
    );
  }

  private isCadenceDue(cadence: MaintenanceCadence, nowMs: number): boolean {
    const watermark = this.readCadenceWatermark(cadence);

    return watermark === null || nowMs - watermark.lastTs >= this.intervalMsFor(cadence);
  }

  private nextDueAt(cadence: MaintenanceCadence, nowMs: number): number {
    const watermark = this.readCadenceWatermark(cadence);

    return watermark === null ? nowMs : watermark.lastTs + this.intervalMsFor(cadence);
  }

  private clearCadenceTimer(cadence: MaintenanceCadence): void {
    const timer = this.timers[cadence].timer;

    if (timer === null) {
      return;
    }

    this.clearTimeoutFn(timer);
    this.timers[cadence].timer = null;
  }

  private nextBusyRetryDelay(cadence: MaintenanceCadence): number {
    const previousDelayMs = this.timers[cadence].retryDelayMs;
    const baseDelayMs = this.options.busyRetryBaseMs;
    const nextDelayMs =
      previousDelayMs === null ? baseDelayMs : Math.max(baseDelayMs, previousDelayMs * 2);
    const boundedDelayMs = Math.min(nextDelayMs, this.options.busyRetryMaxMs);

    this.timers[cadence].retryDelayMs = boundedDelayMs;
    return boundedDelayMs;
  }

  private scheduleCadence(
    cadence: MaintenanceCadence,
    reason: "startup" | "next" | "busy_retry",
  ): void {
    if (!this.started || this.stopping || !this.options.enabled) {
      return;
    }

    this.clearCadenceTimer(cadence);

    const nowMs = this.clock.now();
    const delayMs =
      reason === "busy_retry"
        ? this.nextBusyRetryDelay(cadence)
        : this.delayUntilNextDue(cadence, nowMs, reason);

    this.scheduleCadenceAfter(cadence, delayMs);
  }

  private scheduleCadenceAfter(cadence: MaintenanceCadence, delayMs: number): void {
    if (!this.started || this.stopping || !this.options.enabled) {
      return;
    }

    this.clearCadenceTimer(cadence);

    this.timers[cadence].timer = this.setTimeoutFn(() => {
      this.timers[cadence].timer = null;

      if (!this.started || this.stopping) {
        return;
      }

      void this.runDueTick(cadence);
    }, delayMs);
  }

  private delayUntilNextDue(
    cadence: MaintenanceCadence,
    nowMs: number,
    reason: "startup" | "next",
  ): number {
    const delayMs = Math.max(0, this.nextDueAt(cadence, nowMs) - nowMs);

    if (delayMs === 0 && reason === "startup") {
      return this.options.startupGraceMs;
    }

    return delayMs;
  }

  private scheduleAfterResult(result: MaintenanceTickResult): void {
    if (!this.started || this.stopping) {
      return;
    }

    if (result.status === "skipped_busy") {
      this.scheduleCadence(result.cadence, "busy_retry");
      return;
    }

    if (result.status !== "ok") {
      this.timers[result.cadence].retryDelayMs = null;
      this.scheduleCadenceAfter(result.cadence, this.intervalMsFor(result.cadence));
      return;
    }

    this.timers[result.cadence].retryDelayMs = null;
    this.scheduleCadence(result.cadence, "next");
  }

  private async runDueTick(cadence: MaintenanceCadence): Promise<void> {
    try {
      if (!this.started || this.stopping) {
        return;
      }

      const activeTick = this.activeTicks[cadence];

      if (activeTick !== null) {
        const result = await activeTick;
        this.scheduleAfterResult(result);
        return;
      }

      // Re-read the durable marker immediately before doing work so a restart
      // inside the same interval observes a successful prior run instead of
      // blindly executing another catch-up tick.
      if (!this.isCadenceDue(cadence, this.clock.now())) {
        this.timers[cadence].retryDelayMs = null;
        this.scheduleCadence(cadence, "next");
        return;
      }

      const result = await this.runTrackedTick(cadence, { notifyObserver: false });
      await this.notifyTick(result);
      this.scheduleAfterResult(result);
    } catch (error) {
      await this.notifyError(error, cadence);
      this.scheduleCadence(cadence, "busy_retry");
    }
  }

  private watermarkEntryIdFor(result: MaintenanceTickResult): string {
    return result.result?.run_id ?? `maintenance:${result.cadence}:${result.ts}`;
  }

  private recordSuccessfulTick(result: MaintenanceTickResult): void {
    if (result.status !== "ok") {
      return;
    }

    // Borg runs one maintenance scheduler per data dir; this last-run anchor is not a distributed lock.
    // "ok" means the cadence executed; embedded errors are surfaced via result/tracer and retry next cadence.
    this.options.cadenceWatermarkRepository.set(
      cadenceWatermarkProcessName(result.cadence),
      MAINTENANCE_CADENCE_WATERMARK_SESSION_ID,
      {
        lastTs: this.clock.now(),
        lastEntryId: this.watermarkEntryIdFor(result),
      },
    );
    this.timers[result.cadence].retryDelayMs = null;
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
        this.recordSuccessfulTick(result);

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
