// Builds the offline maintenance scheduler that fires consolidator/curator/reflector/... on cadences.
// Separate from autonomy because maintenance is housekeeping, not cognition.

import type { Config } from "../config/index.js";
import {
  MaintenanceScheduler,
  type MaintenanceOrchestrator,
  type OfflineProcess,
  type OfflineProcessName,
} from "../offline/index.js";
import type { LanceDbStore } from "../storage/lancedb/index.js";
import type { StreamWatermarkRepository } from "../stream/index.js";
import type { TurnTracer } from "../tracing/tracer.js";
import type { Clock } from "../util/clock.js";

export type BuildMaintenanceSchedulerOptions = {
  config: Config;
  lance: LanceDbStore;
  orchestrator: MaintenanceOrchestrator;
  processRegistry: Record<OfflineProcessName, OfflineProcess>;
  cadenceWatermarkRepository: Pick<StreamWatermarkRepository, "get" | "set">;
  clock: Clock;
  tracer?: TurnTracer;
  isBusy?: () => boolean;
};

export function buildMaintenanceScheduler(
  options: BuildMaintenanceSchedulerOptions,
): MaintenanceScheduler {
  return new MaintenanceScheduler({
    enabled: options.config.maintenance.enabled,
    lightIntervalMs: options.config.maintenance.lightIntervalMs,
    heavyIntervalMs: options.config.maintenance.heavyIntervalMs,
    startupGraceMs: options.config.maintenance.startupGraceMs,
    busyRetryBaseMs: options.config.maintenance.busyRetryBaseMs,
    busyRetryMaxMs: options.config.maintenance.busyRetryMaxMs,
    lightProcesses: options.config.maintenance.lightProcesses,
    heavyProcesses: options.config.maintenance.heavyProcesses,
    orchestrator: options.orchestrator,
    processRegistry: options.processRegistry,
    cadenceWatermarkRepository: options.cadenceWatermarkRepository,
    optimizeStorage: options.config.maintenance.optimizeStorage,
    storageOptimizer: () => options.lance.optimizeStorage(),
    tracer: options.tracer,
    clock: options.clock,
    isBusy: options.isBusy,
  });
}
