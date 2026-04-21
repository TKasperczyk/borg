import { StreamWriter } from "../stream/index.js";
import { createMaintenanceRunId } from "../util/ids.js";
import { BorgError } from "../util/errors.js";

import type { AuditLog } from "./audit-log.js";
import { maintenancePlanSchema, type MaintenancePlan } from "./plan-file.js";
import {
  OFFLINE_PROCESS_NAMES,
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
        }
      >
    >;
  };
};

export class MaintenanceOrchestrator {
  constructor(private readonly options: MaintenanceOrchestratorOptions) {}

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
      errors: [
        {
          process: processName,
          message: error instanceof Error ? error.message : String(error),
          code: borgError?.code,
        },
      ],
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

  async plan(input: MaintenanceRunOptions): Promise<MaintenancePlan> {
    const runId = createMaintenanceRunId();
    const streamWriter = this.options.createStreamWriter();

    try {
      const plans = [];

      for (const process of input.processes) {
        const override = input.opts?.processOverrides?.[process.name];
        const context = this.createContext(runId, streamWriter);
        const plan = await process.plan(context, {
          budget: override?.budget ?? input.opts?.budget,
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

  preview(rawPlan: MaintenancePlan): OrchestratorResult {
    const plan = maintenancePlanSchema.parse(rawPlan);
    const runId = createMaintenanceRunId();
    const results = plan.processes.map((processPlan) => {
      const process = this.options.processRegistry[processPlan.process];
      return process.preview(processPlan);
    });

    return this.summarizeResults(runId, results);
  }

  async apply(rawPlan: MaintenancePlan): Promise<OrchestratorResult> {
    const plan = maintenancePlanSchema.parse(rawPlan);
    const runId = createMaintenanceRunId();
    const streamWriter = this.options.createStreamWriter();
    const results: OfflineResult[] = [];

    try {
      for (const processPlan of plan.processes) {
        const process = this.options.processRegistry[processPlan.process];
        const context = this.createContext(runId, streamWriter);

        try {
          results.push(await process.apply(context, processPlan));
        } catch (error) {
          results.push(this.createFailureResult(process.name, false, error));
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

  async run(input: MaintenanceRunOptions): Promise<OrchestratorResult> {
    const plan = await this.plan(input);
    return input.opts?.dryRun === true ? this.preview(plan) : this.apply(plan);
  }
}
