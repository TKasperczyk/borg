// Schedules offline maintenance runs on two cadences (light/heavy).
// Separate from the autonomy scheduler: maintenance is housekeeping, not cognition,
// so it runs on its own interval loop with a busy-detection hook.

import { SystemClock, type Clock } from "../util/clock.js";
import { ConfigError } from "../util/errors.js";

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

  private async tickOnce(cadence: MaintenanceCadence): Promise<MaintenanceTickResult> {
    const ts = this.clock.now();
    const processes = this.processNamesFor(cadence);

    if (!this.options.enabled) {
      return {
        status: "disabled",
        cadence,
        ts,
        processes: [...processes],
        result: null,
        reason: "Maintenance scheduler is disabled.",
      };
    }

    if (processes.length === 0) {
      return {
        status: "skipped_empty",
        cadence,
        ts,
        processes: [],
        result: null,
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
        reason: "Skipped because the system is busy.",
      };
    }

    const offlineProcesses = processes
      .map((name) => this.options.processRegistry[name])
      .filter((process): process is OfflineProcess => process !== undefined);
    const result = await this.options.orchestrator.run({
      processes: offlineProcesses,
    });

    return {
      status: "ok",
      cadence,
      ts,
      processes: [...processes],
      result,
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
    // Guard only against overlap with the same cadence; light and heavy can
    // run in parallel since they operate on disjoint processes and have
    // independent cadences. Coalescing across cadences would cause the heavy
    // cycle to be skipped forever whenever both interval timers fire together.
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
