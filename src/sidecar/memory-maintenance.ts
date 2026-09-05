import type { Borg } from "../borg.js";
import type { Config } from "../config/index.js";
import {
  formatOfflineProcessErrorMessage,
  MAX_OFFLINE_ERROR_DETAILS,
  MAX_OFFLINE_ERROR_MESSAGE_LENGTH,
} from "../offline/process-errors.js";
import { storageOptimizationErrorCount } from "../offline/storage-optimization.js";
import type { OfflineProcessError, OfflineProcessName } from "../offline/types.js";
import type { LanceDbOptimizeStorageResult } from "../storage/lancedb/index.js";
import { createMaintenanceRunId, type MaintenanceRunId } from "../util/ids.js";

export type MemoryMaintenanceMode = "light" | "heavy";
export type MemoryMaintenanceSkipReason = "budget" | "shutdown";

export type MemoryMaintenanceErrorDetail = {
  process?: OfflineProcessName;
  message: string;
  code?: string;
};

export type MemoryMaintenanceProcessReport = {
  name: OfflineProcessName;
  status: "pending" | "completed" | "skipped";
  changes: number;
  errors: number;
  tokens_used: number;
  budget_exhausted: boolean;
  duration_ms: number;
  error_details: MemoryMaintenanceErrorDetail[];
  skipped?: MemoryMaintenanceSkipReason;
  error?: string;
};

export type MemoryMaintenanceStorageReport =
  | { status: "pending" }
  | { status: "skipped"; reason: "dryRun" | "modeLight" | "shutdown" }
  | {
      status: "completed" | "error";
      errors: number;
      error_details: MemoryMaintenanceErrorDetail[];
      duration_ms: number;
      result: LanceDbOptimizeStorageResult;
    }
  | {
      status: "error";
      errors: 1;
      error_details: MemoryMaintenanceErrorDetail[];
      duration_ms: number;
      error: string;
    };

export type MemoryMaintenanceRunReport = {
  evt: "memory_maintenance";
  tenant: string;
  mode: MemoryMaintenanceMode;
  run_id: MaintenanceRunId;
  dryRun: boolean;
  state: "running" | "completed" | "completed_with_errors" | "aborted";
  started_at: number;
  finished_at: number | null;
  requested_processes: OfflineProcessName[];
  resolved_processes: OfflineProcessName[];
  budget: number | null;
  budget_remaining: number | null;
  processes: MemoryMaintenanceProcessReport[];
  storage_optimize: MemoryMaintenanceStorageReport;
  total_duration_ms: number;
  errors_total: number;
};

type MemoryMaintenanceStdoutProcess =
  | {
      name: OfflineProcessName;
      changes: number;
      errors: number;
      error_details: MemoryMaintenanceErrorDetail[];
      tokens_used: number;
      budget_exhausted: boolean;
      duration_ms: number;
    }
  | {
      name: OfflineProcessName;
      skipped: MemoryMaintenanceSkipReason;
      error_details: MemoryMaintenanceErrorDetail[];
    };

type MemoryMaintenanceStdoutReport = {
  evt: "memory_maintenance";
  tenant: string;
  mode: MemoryMaintenanceMode;
  run_id: MaintenanceRunId;
  dryRun: boolean;
  processes: MemoryMaintenanceStdoutProcess[];
  storage_optimize: MemoryMaintenanceStorageReport;
  total_duration_ms: number;
  errors_total: number;
};

type MemoryMaintenanceErrorReport = {
  evt: "memory_maintenance_error";
  tenant: string;
  run_id: MaintenanceRunId;
  mode: MemoryMaintenanceMode;
  errors_total: number;
  error_details: MemoryMaintenanceErrorDetail[];
};

export type MemoryMaintenanceStatus = {
  current: MemoryMaintenanceRunReport | null;
  last: MemoryMaintenanceRunReport | null;
};

export type MemoryMaintenanceConfig = {
  enabled: boolean;
  lightProcesses: readonly OfflineProcessName[];
  heavyProcesses: readonly OfflineProcessName[];
  lightBudget: number | null;
  heavyBudget: number | null;
  processBudgets: Partial<Record<OfflineProcessName, number | null>>;
};

export type MemoryMaintenancePool = {
  withTenant<T>(
    tenantId: string,
    fn: (borg: Borg) => T | Promise<T>,
    opts?: { exclusive?: boolean },
  ): Promise<T>;
};

type ActiveRun = {
  report: MemoryMaintenanceRunReport;
  finalized: boolean;
  abortRequested: boolean;
  running: boolean;
  phase: "reserved" | "scheduled" | "executing";
  accepted: boolean;
  resolveCompletion: (report: MemoryMaintenanceRunReport) => void;
};

export type MemoryMaintenanceStartResult =
  | {
      status: "accepted";
      runId: MaintenanceRunId;
      completion: Promise<MemoryMaintenanceRunReport>;
    }
  | { status: "conflict"; runId: MaintenanceRunId }
  | { status: "disabled" }
  | { status: "shutting_down" };

export type MemoryMaintenanceStartInput = {
  tenant: string;
  mode: MemoryMaintenanceMode;
  dryRun: boolean;
};

export type MemoryMaintenanceCoordinatorOptions = {
  pool: MemoryMaintenancePool;
  config: MemoryMaintenanceConfig;
  maxLastTenants?: number;
  now?: () => number;
  schedule?: (task: () => void) => void;
  yieldBetweenChunks?: () => Promise<void>;
  writeReport?: (line: string) => void;
  writeError?: (message: string) => void;
};

const DEFAULT_MAX_LAST_TENANTS = 64;

function errorMessage(error: unknown): string {
  return formatOfflineProcessErrorMessage(error);
}

function errorCode(error: unknown): string | undefined {
  if (error === null || typeof error !== "object" || !("code" in error)) {
    return undefined;
  }
  return typeof error.code === "string" ? error.code : undefined;
}

function normalizeErrorDetail(input: {
  process?: OfflineProcessName;
  message: string;
  code?: string;
}): MemoryMaintenanceErrorDetail {
  return {
    ...(input.process === undefined ? {} : { process: input.process }),
    message: input.message.slice(0, MAX_OFFLINE_ERROR_MESSAGE_LENGTH),
    ...(input.code === undefined ? {} : { code: input.code }),
  };
}

function offlineErrorDetails(
  errors: readonly OfflineProcessError[],
): MemoryMaintenanceErrorDetail[] {
  return errors.slice(0, MAX_OFFLINE_ERROR_DETAILS).map((error) =>
    normalizeErrorDetail({
      process: error.process,
      message: error.message,
      ...(error.code === undefined ? {} : { code: error.code }),
    }),
  );
}

function thrownErrorDetail(
  error: unknown,
  process?: OfflineProcessName,
): MemoryMaintenanceErrorDetail {
  const code = errorCode(error);
  return normalizeErrorDetail({
    ...(process === undefined ? {} : { process }),
    message: errorMessage(error),
    ...(code === undefined ? {} : { code }),
  });
}

function storageErrorDetails(result: LanceDbOptimizeStorageResult): MemoryMaintenanceErrorDetail[] {
  const details: MemoryMaintenanceErrorDetail[] = [];
  if (result.error !== undefined) {
    details.push(
      normalizeErrorDetail({
        message: result.error.message,
        ...(result.error.code === undefined ? {} : { code: result.error.code }),
      }),
    );
  }
  for (const table of result.tables) {
    if (details.length >= MAX_OFFLINE_ERROR_DETAILS) {
      break;
    }
    if (table.status === "error") {
      details.push(
        normalizeErrorDetail({
          message: table.error.message,
          ...(table.error.code === undefined ? {} : { code: table.error.code }),
        }),
      );
    }
  }
  return details;
}

function cloneReport(report: MemoryMaintenanceRunReport): MemoryMaintenanceRunReport {
  return structuredClone(report);
}

function skippedProcess(
  name: OfflineProcessName,
  reason: MemoryMaintenanceSkipReason,
): MemoryMaintenanceProcessReport {
  return {
    name,
    status: "skipped",
    changes: 0,
    errors: 0,
    tokens_used: 0,
    budget_exhausted: reason === "budget",
    duration_ms: 0,
    error_details: [],
    skipped: reason,
  };
}

function stdoutReport(report: MemoryMaintenanceRunReport): MemoryMaintenanceStdoutReport {
  return {
    evt: report.evt,
    tenant: report.tenant,
    mode: report.mode,
    run_id: report.run_id,
    dryRun: report.dryRun,
    processes: report.processes.map((process) =>
      process.skipped === undefined
        ? {
            name: process.name,
            changes: process.changes,
            errors: process.errors,
            error_details: process.error_details,
            tokens_used: process.tokens_used,
            budget_exhausted: process.budget_exhausted,
            duration_ms: process.duration_ms,
          }
        : {
            name: process.name,
            skipped: process.skipped,
            error_details: process.error_details,
          },
    ),
    storage_optimize: report.storage_optimize,
    total_duration_ms: report.total_duration_ms,
    errors_total: report.errors_total,
  };
}

function errorReport(report: MemoryMaintenanceRunReport): MemoryMaintenanceErrorReport {
  const storageDetails =
    report.storage_optimize.status === "completed" || report.storage_optimize.status === "error"
      ? report.storage_optimize.error_details
      : [];
  return {
    evt: "memory_maintenance_error",
    tenant: report.tenant,
    run_id: report.run_id,
    mode: report.mode,
    errors_total: report.errors_total,
    error_details: [
      ...report.processes.flatMap((process) => process.error_details),
      ...storageDetails,
    ].slice(0, MAX_OFFLINE_ERROR_DETAILS),
  };
}

function sidecarProcessBudgets(config: Config): MemoryMaintenanceConfig["processBudgets"] {
  return {
    consolidator: config.offline.consolidator.budget,
    reflector: config.offline.reflector.budget,
    associator: config.offline.associator.budget,
    "semantic-extractor": config.offline.semanticExtractor.budget,
    curator: null,
    overseer: config.offline.overseer.budget,
    "review-resolver": config.offline.reviewResolver.budget,
    ruminator: config.offline.ruminator.budget,
    "self-narrator": config.offline.selfNarrator.budget,
    "lived-experience-day-summarizer": config.offline.livedExperienceDaySummarizer.budget,
    "procedural-synthesizer": config.offline.proceduralSynthesizer.budget,
    "belief-reviser": null,
    "creator-directive-reconciler": config.offline.creatorDirectiveReconciler.budget,
    "commitment-reconciler": config.offline.commitmentReconciler.budget,
  };
}

export function memoryMaintenanceConfigFromConfig(config: Config): MemoryMaintenanceConfig {
  return {
    enabled: config.maintenance.enabled,
    lightProcesses: config.maintenance.lightProcesses,
    heavyProcesses: config.maintenance.heavyProcesses,
    lightBudget: config.maintenance.lightBudget,
    heavyBudget: config.maintenance.heavyBudget,
    processBudgets: sidecarProcessBudgets(config),
  };
}

export function memorySelfNameFromEnv(env: NodeJS.ProcessEnv): string {
  return env.BORG_MEMORY_SELF_NAME?.trim() || "team-agent";
}

export class MemoryMaintenanceCoordinator {
  private readonly pool: MemoryMaintenancePool;
  private readonly config: MemoryMaintenanceConfig;
  private readonly maxLastTenants: number;
  private readonly now: () => number;
  private readonly schedule: (task: () => void) => void;
  private readonly yieldBetweenChunks: () => Promise<void>;
  private readonly writeReport: (line: string) => void;
  private readonly writeError: (message: string) => void;

  private readonly activeByTenant = new Map<string, ActiveRun>();
  private readonly lastByTenant = new Map<string, MemoryMaintenanceRunReport>();
  private shuttingDown = false;

  constructor(options: MemoryMaintenanceCoordinatorOptions) {
    this.pool = options.pool;
    this.config = options.config;
    const requestedLastTenantCap = options.maxLastTenants ?? DEFAULT_MAX_LAST_TENANTS;
    this.maxLastTenants = Number.isFinite(requestedLastTenantCap)
      ? Math.max(0, Math.min(DEFAULT_MAX_LAST_TENANTS, Math.floor(requestedLastTenantCap)))
      : DEFAULT_MAX_LAST_TENANTS;
    this.now = options.now ?? Date.now;
    this.schedule =
      options.schedule ??
      ((task) => {
        setImmediate(task);
      });
    this.yieldBetweenChunks =
      options.yieldBetweenChunks ??
      (() =>
        new Promise<void>((resolve) => {
          setImmediate(resolve);
        }));
    this.writeReport = options.writeReport ?? ((line) => console.log(line));
    this.writeError = options.writeError ?? ((message) => console.error(message));
  }

  tryStart(input: MemoryMaintenanceStartInput): MemoryMaintenanceStartResult {
    const reserved = this.tryReserve(input);
    if (reserved.status === "accepted") {
      this.startReserved(input.tenant, reserved.runId);
    }
    return reserved;
  }

  tryReserve(input: MemoryMaintenanceStartInput): MemoryMaintenanceStartResult {
    if (this.shuttingDown) {
      return { status: "shutting_down" };
    }
    if (!this.config.enabled) {
      return { status: "disabled" };
    }

    const active = this.activeByTenant.get(input.tenant);
    if (active !== undefined) {
      return { status: "conflict", runId: active.report.run_id };
    }

    const runId = createMaintenanceRunId();
    const processNames =
      input.mode === "light" ? this.config.lightProcesses : this.config.heavyProcesses;
    const budget = input.mode === "light" ? this.config.lightBudget : this.config.heavyBudget;
    const startedAt = this.now();
    let resolveCompletion!: (report: MemoryMaintenanceRunReport) => void;
    const completion = new Promise<MemoryMaintenanceRunReport>((resolve) => {
      resolveCompletion = resolve;
    });
    const state: ActiveRun = {
      finalized: false,
      abortRequested: false,
      running: false,
      phase: "reserved",
      accepted: false,
      resolveCompletion,
      report: {
        evt: "memory_maintenance",
        tenant: input.tenant,
        mode: input.mode,
        run_id: runId,
        dryRun: input.dryRun,
        state: "running",
        started_at: startedAt,
        finished_at: null,
        requested_processes: [...processNames],
        resolved_processes: [],
        budget,
        budget_remaining: budget,
        processes: processNames.map((name) => ({
          name,
          status: "pending",
          changes: 0,
          errors: 0,
          tokens_used: 0,
          budget_exhausted: false,
          duration_ms: 0,
          error_details: [],
        })),
        storage_optimize: input.dryRun
          ? { status: "skipped", reason: "dryRun" }
          : input.mode === "light"
            ? { status: "skipped", reason: "modeLight" }
            : { status: "pending" },
        total_duration_ms: 0,
        errors_total: 0,
      },
    };
    this.activeByTenant.set(input.tenant, state);

    return { status: "accepted", runId, completion };
  }

  startReserved(tenant: string, runId: MaintenanceRunId): boolean {
    const state = this.activeByTenant.get(tenant);
    if (
      state === undefined ||
      state.report.run_id !== runId ||
      state.finalized ||
      state.phase !== "reserved"
    ) {
      return false;
    }
    state.phase = "scheduled";
    state.accepted = true;

    this.schedule(() => {
      if (state.finalized) {
        return;
      }
      state.phase = "executing";
      void this.execute(state).catch((error: unknown) => {
        if (state.finalized) {
          return;
        }
        if (state.abortRequested) {
          this.finishAborted(state, 0);
          return;
        }
        const pending = state.report.processes.find((process) => process.status === "pending");
        if (pending !== undefined) {
          Object.assign(pending, {
            status: "completed",
            errors: 1,
            error_details: [thrownErrorDetail(error, pending.name)],
            error: errorMessage(error),
          });
        }
        this.finalize(state, "completed_with_errors");
      });
    });

    return true;
  }

  hasReservation(tenant: string, runId: MaintenanceRunId): boolean {
    const state = this.activeByTenant.get(tenant);
    return (
      state !== undefined &&
      state.report.run_id === runId &&
      !state.finalized &&
      state.phase === "reserved"
    );
  }

  cancelReservation(tenant: string, runId: MaintenanceRunId): boolean {
    const state = this.activeByTenant.get(tenant);
    if (
      state === undefined ||
      state.report.run_id !== runId ||
      state.finalized ||
      state.phase !== "reserved"
    ) {
      return false;
    }
    state.finalized = true;
    state.report.state = "aborted";
    state.report.finished_at = this.now();
    state.report.total_duration_ms = Math.max(
      0,
      state.report.finished_at - state.report.started_at,
    );
    this.activeByTenant.delete(tenant);
    state.resolveCompletion(cloneReport(state.report));
    return true;
  }

  getStatus(tenant: string): MemoryMaintenanceStatus {
    const active = this.activeByTenant.get(tenant);
    const current = active === undefined ? null : cloneReport(active.report);
    if (current !== null) {
      current.errors_total = this.errorCount(current);
      current.total_duration_ms = Math.max(0, this.now() - current.started_at);
    }

    return {
      current,
      last:
        this.lastByTenant.get(tenant) === undefined
          ? null
          : cloneReport(this.lastByTenant.get(tenant)!),
    };
  }

  beginShutdown(): MaintenanceRunId[] {
    if (this.shuttingDown) {
      return [];
    }
    this.shuttingDown = true;

    const abortRequested: MaintenanceRunId[] = [];
    for (const state of [...this.activeByTenant.values()]) {
      state.abortRequested = true;
      abortRequested.push(state.report.run_id);
    }
    return abortRequested;
  }

  forceFinalizeAborted(): MaintenanceRunId[] {
    const finalized: MaintenanceRunId[] = [];
    for (const state of [...this.activeByTenant.values()]) {
      if (!state.abortRequested || state.finalized) {
        continue;
      }
      finalized.push(state.report.run_id);
      this.finishAborted(state, 0);
    }
    return finalized;
  }

  private async execute(state: ActiveRun): Promise<void> {
    for (let index = 0; index < state.report.processes.length; index += 1) {
      if (state.finalized) {
        return;
      }
      if (state.abortRequested) {
        this.finishAborted(state, index);
        return;
      }

      const process = state.report.processes[index]!;
      const remaining = state.report.budget_remaining;
      if (remaining !== null && remaining <= 0) {
        this.skipRemainingForBudget(state, index);
        break;
      }

      const configuredProcessBudget = this.config.processBudgets[process.name] ?? null;
      const chunkBudget =
        remaining === null
          ? configuredProcessBudget
          : configuredProcessBudget === null
            ? remaining
            : Math.min(remaining, configuredProcessBudget);
      let startedAt: number | null = null;

      try {
        // One reservation per process is the intentional scheduling boundary.
        // Recall stays non-exclusive and may overlap this chunk; append/remember
        // are exclusive and can acquire the tenant between chunks.
        const reserved = await this.pool.withTenant(
          state.report.tenant,
          async (borg) => {
            // Shutdown may arrive while this exclusive reservation is queued.
            // Recheck only after acquisition so queued work never starts late.
            if (state.abortRequested || state.finalized) {
              return { started: false as const };
            }
            state.running = true;
            startedAt = this.now();
            state.report.resolved_processes.push(process.name);
            try {
              return {
                started: true as const,
                result: await borg.dream({
                  runId: state.report.run_id,
                  processes: [process.name],
                  dryRun: state.report.dryRun,
                  ...(chunkBudget === null ? {} : { budget: chunkBudget }),
                }),
              };
            } finally {
              state.running = false;
            }
          },
          { exclusive: true },
        );

        if (state.finalized) {
          return;
        }
        if (!reserved.started) {
          this.finishAborted(state, index);
          return;
        }
        const result = reserved.result;
        const processResult = result.results[0];
        const reportedErrors = processResult?.errors ?? result.errors;
        process.status = "completed";
        process.changes = processResult?.changes.length ?? result.changes.length;
        process.errors = reportedErrors.length;
        process.error_details = offlineErrorDetails(reportedErrors);
        process.tokens_used = processResult?.tokens_used ?? result.tokens_used;
        process.budget_exhausted = processResult?.budget_exhausted ?? false;
        process.duration_ms = Math.max(0, this.now() - (startedAt ?? this.now()));

        if (remaining !== null) {
          const nextRemaining = Math.max(0, remaining - process.tokens_used);
          state.report.budget_remaining =
            process.budget_exhausted && chunkBudget === remaining ? 0 : nextRemaining;
        }
      } catch (error) {
        if (state.finalized) {
          return;
        }
        if (state.abortRequested && startedAt === null) {
          this.finishAborted(state, index);
          return;
        }
        process.status = "completed";
        process.errors = 1;
        process.error_details = [thrownErrorDetail(error, process.name)];
        process.error = errorMessage(error);
        process.duration_ms = Math.max(0, this.now() - (startedAt ?? this.now()));
      }

      // A chunk that was already running at shutdown is allowed to settle and
      // its result is retained. Only subsequent chunks are skipped.
      if (state.abortRequested) {
        this.finishAborted(state, index + 1);
        return;
      }

      let budgetStopped = false;
      if (
        index + 1 < state.report.processes.length &&
        state.report.budget_remaining !== null &&
        state.report.budget_remaining <= 0
      ) {
        this.skipRemainingForBudget(state, index + 1);
        budgetStopped = true;
      }
      if (
        (!budgetStopped && index + 1 < state.report.processes.length) ||
        state.report.storage_optimize.status === "pending"
      ) {
        await this.yieldBetweenChunks();
      }
      if (budgetStopped) {
        break;
      }
    }

    if (state.finalized) {
      return;
    }
    if (state.abortRequested) {
      this.finishAborted(state, state.report.processes.length);
      return;
    }

    if (state.report.storage_optimize.status === "pending") {
      let startedAt: number | null = null;
      try {
        const reserved = await this.pool.withTenant(
          state.report.tenant,
          async (borg) => {
            if (state.abortRequested || state.finalized) {
              return { started: false as const };
            }
            state.running = true;
            startedAt = this.now();
            try {
              return {
                started: true as const,
                result: await borg.maintenance.optimizeStorage({ runId: state.report.run_id }),
              };
            } finally {
              state.running = false;
            }
          },
          { exclusive: true },
        );
        if (state.finalized) {
          return;
        }
        if (!reserved.started) {
          this.finishAborted(state, state.report.processes.length);
          return;
        }
        const result = reserved.result;
        const errors = storageOptimizationErrorCount(result);
        state.report.storage_optimize = {
          status: errors === 0 ? "completed" : "error",
          errors,
          error_details: storageErrorDetails(result),
          duration_ms: Math.max(0, this.now() - (startedAt ?? this.now())),
          result,
        };
      } catch (error) {
        if (state.finalized) {
          return;
        }
        if (state.abortRequested && startedAt === null) {
          this.finishAborted(state, state.report.processes.length);
          return;
        }
        state.report.storage_optimize = {
          status: "error",
          errors: 1,
          error_details: [thrownErrorDetail(error)],
          duration_ms: Math.max(0, this.now() - (startedAt ?? this.now())),
          error: errorMessage(error),
        };
      }
    }

    if (state.abortRequested) {
      this.finishAborted(state, state.report.processes.length);
      return;
    }

    const errors = this.errorCount(state.report);
    this.finalize(state, errors === 0 ? "completed" : "completed_with_errors");
  }

  private finishAborted(state: ActiveRun, fromIndex: number): void {
    for (let index = fromIndex; index < state.report.processes.length; index += 1) {
      const process = state.report.processes[index]!;
      if (process.status === "pending") {
        state.report.processes[index] = skippedProcess(process.name, "shutdown");
      }
    }
    if (state.report.storage_optimize.status === "pending") {
      state.report.storage_optimize = { status: "skipped", reason: "shutdown" };
    }
    this.finalize(state, "aborted");
  }

  private skipRemainingForBudget(state: ActiveRun, fromIndex: number): void {
    for (let index = fromIndex; index < state.report.processes.length; index += 1) {
      const process = state.report.processes[index]!;
      if (process.status === "pending") {
        state.report.processes[index] = skippedProcess(process.name, "budget");
      }
    }
  }

  private errorCount(report: MemoryMaintenanceRunReport): number {
    const processErrors = report.processes.reduce((sum, process) => sum + process.errors, 0);
    const storageErrors =
      report.storage_optimize.status === "completed" || report.storage_optimize.status === "error"
        ? report.storage_optimize.errors
        : 0;
    return processErrors + storageErrors;
  }

  private finalize(
    state: ActiveRun,
    finalState: Exclude<MemoryMaintenanceRunReport["state"], "running">,
  ): boolean {
    if (state.finalized) {
      return false;
    }
    state.finalized = true;
    const finishedAt = this.now();
    state.report.state = finalState;
    state.report.finished_at = finishedAt;
    state.report.total_duration_ms = Math.max(0, finishedAt - state.report.started_at);
    state.report.errors_total = this.errorCount(state.report);

    if (this.activeByTenant.get(state.report.tenant) === state) {
      this.activeByTenant.delete(state.report.tenant);
    }
    const finalReport = cloneReport(state.report);
    if (state.accepted) {
      this.lastByTenant.delete(state.report.tenant);
      this.lastByTenant.set(state.report.tenant, cloneReport(finalReport));
      while (this.lastByTenant.size > this.maxLastTenants) {
        const oldest = this.lastByTenant.keys().next().value as string | undefined;
        if (oldest === undefined) {
          break;
        }
        this.lastByTenant.delete(oldest);
      }

      this.writeReport(JSON.stringify(stdoutReport(finalReport)));
      const errors = finalReport.errors_total;
      if (errors > 0) {
        this.writeError(JSON.stringify(errorReport(finalReport)));
      }
    }
    state.resolveCompletion(finalReport);
    return true;
  }
}
