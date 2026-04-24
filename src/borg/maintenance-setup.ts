// Builds the offline maintenance scheduler that fires consolidator/curator/reflector/... on cadences.
// Separate from autonomy because maintenance is housekeeping, not cognition.

import type { Config } from "../config/index.js";
import {
  MaintenanceScheduler,
  type MaintenanceOrchestrator,
  type OfflineProcess,
  type OfflineProcessName,
} from "../offline/index.js";
import type { Clock } from "../util/clock.js";

export type BuildMaintenanceSchedulerOptions = {
  config: Config;
  orchestrator: MaintenanceOrchestrator;
  processRegistry: Record<OfflineProcessName, OfflineProcess>;
  clock: Clock;
  isBusy?: () => boolean;
};

export function buildMaintenanceScheduler(
  options: BuildMaintenanceSchedulerOptions,
): MaintenanceScheduler {
  return new MaintenanceScheduler({
    enabled: options.config.maintenance.enabled,
    lightIntervalMs: options.config.maintenance.lightIntervalMs,
    heavyIntervalMs: options.config.maintenance.heavyIntervalMs,
    lightProcesses: options.config.maintenance.lightProcesses,
    heavyProcesses: options.config.maintenance.heavyProcesses,
    orchestrator: options.orchestrator,
    processRegistry: options.processRegistry,
    clock: options.clock,
    isBusy: options.isBusy,
  });
}
