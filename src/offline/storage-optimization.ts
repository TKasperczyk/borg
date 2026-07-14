import { performance } from "node:perf_hooks";

import {
  normalizeOptimizeError,
  type LanceDbOptimizeStorageResult,
} from "../storage/lancedb/index.js";
import type { TurnTracer } from "../tracing/tracer.js";

export function storageOptimizationErrorCount(result: LanceDbOptimizeStorageResult): number {
  return (
    result.tables.filter((table) => table.status === "error").length +
    (result.error === undefined ? 0 : 1)
  );
}

function emitStorageOptimizationCompleted(input: {
  cadence: "heavy";
  ts: number;
  runId?: string;
  result: LanceDbOptimizeStorageResult;
  tracer?: TurnTracer;
}): void {
  if (input.tracer?.enabled !== true) {
    return;
  }

  const successfulTables = input.result.tables.filter((table) => table.status === "ok");
  const tables = input.result.tables.map((table): Record<string, number | string> => {
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

    return {
      table: table.table,
      status: table.status,
      duration_ms: table.durationMs,
      error_message: table.error.message,
      ...(table.error.code === undefined ? {} : { error_code: table.error.code }),
    };
  });

  input.tracer.emit("storage.optimize.completed", {
    turnId: input.runId ?? `maintenance_storage_${input.cadence}_${input.ts}`,
    cadence: input.cadence,
    table_count: input.result.tables.length,
    errors: storageOptimizationErrorCount(input.result),
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

export async function runStorageOptimization(input: {
  optimizer: () => Promise<LanceDbOptimizeStorageResult>;
  ts: number;
  runId?: string;
  tracer?: TurnTracer;
}): Promise<LanceDbOptimizeStorageResult> {
  const startedAt = performance.now();
  let result: LanceDbOptimizeStorageResult;

  try {
    result = await input.optimizer();
  } catch (error) {
    result = {
      durationMs: Math.round(performance.now() - startedAt),
      tables: [],
      error: normalizeOptimizeError(error),
    };
  }

  emitStorageOptimizationCompleted({
    cadence: "heavy",
    ts: input.ts,
    runId: input.runId,
    result,
    tracer: input.tracer,
  });
  return result;
}
