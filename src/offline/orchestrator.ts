import { performance } from "node:perf_hooks";

import { StreamWriter } from "../stream/index.js";
import { createMaintenanceRunId } from "../util/ids.js";
import { BorgError } from "../util/errors.js";

import type { AuditLog } from "./audit-log.js";
import { maintenancePlanSchema, type MaintenancePlan } from "./plan-file.js";
import { offlineProcessError } from "./process-errors.js";
import {
  type OfflineContext,
  type OfflineProcess,
  type OfflineProcessError,
  type OfflineProcessName,
  type OfflineResult,
  type OrchestratorResult,
} from "./types.js";

export type MaintenanceOrchestratorOptions = {
  baseContext: Omit<OfflineContext, "runId" | "streamWriter" | "auditLog">;
  auditLog: AuditLog;
  createStreamWriter: () => StreamWriter;
  processRegistry: Record<OfflineProcessName, OfflineProcess>;
};

export type MaintenanceRunOptions = {
  processes: OfflineProcess[];
  opts?: {
    dryRun?: boolean;
    budget?: number;
    processOverrides?: Partial<
      Record<
        OfflineProcessName,
        {
          dryRun?: boolean;
          budget?: number;
          params?: Record<string, unknown>;
        }
      >
    >;
  };
};

function traceErrorDetails(errors: readonly OfflineProcessError[]) {
  return errors.map((error) => ({
    message: error.message,
    ...(error.code === undefined ? {} : { code: error.code }),
    ...(error.target_type === undefined ? {} : { target_type: error.target_type }),
    ...(error.target_id === undefined ? {} : { target_id: error.target_id }),
  }));
}

export class MaintenanceOrchestrator {
  private operationQueue: Promise<void> = Promise.resolve();

  constructor(private readonly options: MaintenanceOrchestratorOptions) {}

  private async runExclusive<T>(operation: () => Promise<T>): Promise<T> {
    const previous = this.operationQueue;
    let release: () => void = () => {};

    this.operationQueue = new Promise<void>((resolve) => {
      release = resolve;
    });

    await previous;

    try {
      return await operation();
    } finally {
      release();
    }
  }

  private createContext(
    runId: ReturnType<typeof createMaintenanceRunId>,
    streamWriter: StreamWriter,
  ) {
    return {
      ...this.options.baseContext,
      runId,
      auditLog: this.options.auditLog,
      streamWriter,
    } satisfies OfflineContext;
  }

  private createFailureResult(
    processName: OfflineProcessName,
    dryRun: boolean,
    error: unknown,
  ): OfflineResult {
    const borgError = error instanceof BorgError ? error : undefined;

    return {
      process: processName,
      dryRun,
      changes: [],
      tokens_used:
        error instanceof Error && "tokens_used" in error && typeof error.tokens_used === "number"
          ? error.tokens_used
          : 0,
      errors: [offlineProcessError(processName, error, { code: borgError?.code })],
      budget_exhausted: borgError?.code === "OFFLINE_BUDGET_EXCEEDED",
    };
  }

  private summarizeResults(
    runId: ReturnType<typeof createMaintenanceRunId>,
    results: OfflineResult[],
  ) {
    return {
      run_id: runId,
      dryRun: results.every((result) => result.dryRun),
      results,
      changes: results.flatMap((result) => result.changes),
      tokens_used: results.reduce((sum, result) => sum + result.tokens_used, 0),
      errors: results.flatMap((result) => result.errors),
    } satisfies OrchestratorResult;
  }

  private emitProcessCompleted(input: {
    context: OfflineContext;
    result: OfflineResult;
    durationMs: number;
  }): void {
    if (input.context.tracer?.enabled !== true) {
      return;
    }

    const stats = input.result.candidate_stats ?? {
      proposed: input.result.changes.length,
      accepted: input.result.changes.length,
      rejected: input.result.errors.length,
    };

    input.context.tracer.emit("offline_process.completed", {
      turnId: input.context.runId,
      process_name: input.result.process,
      candidates_proposed: stats.proposed,
      candidates_accepted: stats.accepted,
      candidates_rejected: stats.rejected,
      errors: input.result.errors.length,
      error_details: traceErrorDetails(input.result.errors),
      tokens_used: input.result.tokens_used,
      budget_exhausted: input.result.budget_exhausted,
      duration_ms: Math.round(input.durationMs),
    });
  }

  private async planUnlocked(input: MaintenanceRunOptions): Promise<MaintenancePlan> {
    const runId = createMaintenanceRunId();
    const streamWriter = this.options.createStreamWriter();

    try {
      const plans = [];

      for (const process of input.processes) {
        const override = input.opts?.processOverrides?.[process.name];
        const context = this.createContext(runId, streamWriter);
        const plan = await process.plan(context, {
          dryRun: override?.dryRun ?? input.opts?.dryRun,
          budget: override?.budget ?? input.opts?.budget,
          params: override?.params,
        });
        plans.push(plan);
      }

      return maintenancePlanSchema.parse({
        kind: "borg_maintenance_plan",
        version: 1,
        created_at: this.options.baseContext.clock.now(),
        processes: plans,
      });
    } finally {
      streamWriter.close();
    }
  }

  async plan(input: MaintenanceRunOptions): Promise<MaintenancePlan> {
    return this.runExclusive(() => this.planUnlocked(input));
  }

  preview(rawPlan: MaintenancePlan): OrchestratorResult {
    const plan = maintenancePlanSchema.parse(rawPlan);
    const runId = createMaintenanceRunId();
    const results = plan.processes.map((processPlan) => {
      const process = this.options.processRegistry[processPlan.process];
      return process.preview(processPlan);
    });

    return this.summarizeResults(runId, results);
  }

  private async applyUnlocked(rawPlan: MaintenancePlan): Promise<OrchestratorResult> {
    const plan = maintenancePlanSchema.parse(rawPlan);
    const runId = createMaintenanceRunId();
    const streamWriter = this.options.createStreamWriter();
    const results: OfflineResult[] = [];

    try {
      for (const processPlan of plan.processes) {
        const process = this.options.processRegistry[processPlan.process];
        const context = this.createContext(runId, streamWriter);
        const startedAt = performance.now();
        let result: OfflineResult | null = null;

        try {
          result = await process.apply(context, processPlan);
        } catch (error) {
          result = this.createFailureResult(process.name, false, error);
        } finally {
          if (result !== null) {
            results.push(result);
            this.emitProcessCompleted({
              context,
              result,
              durationMs: performance.now() - startedAt,
            });
          }
          await context.reviewQueueRepository.flushEnqueueHooks();
        }
      }

      const output = this.summarizeResults(runId, results);
      const budgetExhaustedProcesses = results
        .filter((result) => result.budget_exhausted)
        .map((result) => result.process);

      await streamWriter.append({
        kind: "dream_report",
        content: {
          run_id: output.run_id,
          processes: results.map((result) => result.process),
          dry_run: output.dryRun,
          planned_at: plan.created_at,
          changes: output.changes.length,
          tokens_used: output.tokens_used,
          errors: output.errors,
          budget_exhausted_processes: budgetExhaustedProcesses,
          notes:
            budgetExhaustedProcesses.length === 0
              ? []
              : [`Budget exhausted: ${budgetExhaustedProcesses.join(", ")}`],
        },
      });

      return output;
    } finally {
      streamWriter.close();
    }
  }

  async apply(rawPlan: MaintenancePlan): Promise<OrchestratorResult> {
    return this.runExclusive(() => this.applyUnlocked(rawPlan));
  }

  async run(input: MaintenanceRunOptions): Promise<OrchestratorResult> {
    return this.runExclusive(async () => {
      const plan = await this.planUnlocked(input);
      return input.opts?.dryRun === true ? this.preview(plan) : this.applyUnlocked(plan);
    });
  }
}
