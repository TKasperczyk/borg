// Lifecycle helpers for orderly shutdown and partial-open cleanup.

import type { LanceDbStore } from "../storage/lancedb/index.js";
import type { SqliteDatabase } from "../storage/sqlite/index.js";
import type { BorgDependencies } from "./types.js";

export async function closeBestEffort(
  sqlite: SqliteDatabase | undefined,
  lance: LanceDbStore | undefined,
): Promise<boolean> {
  let closed = true;
  if (sqlite !== undefined) {
    try {
      sqlite.close();
    } catch {
      closed = false;
      // Keep the bank lease if a store may still be open.
    }
  }

  if (lance !== undefined) {
    try {
      await lance.close();
    } catch {
      closed = false;
      // Keep the bank lease if a store may still be open.
    }
  }
  return closed;
}

function collectCloseError(errors: unknown[], label: string, error: unknown): void {
  errors.push(error);
  console.error(`Failed to close ${label}`, error);
}

export async function drainBorgDependencies(
  deps: BorgDependencies,
  admittedOperations: readonly Promise<unknown>[] = [],
): Promise<void> {
  const errors: unknown[] = [];

  const schedulerStops = [
    {
      label: "autonomy scheduler",
      close: () =>
        deps.autonomyScheduler.stop({
          graceful: true,
        }),
    },
    {
      label: "maintenance scheduler",
      close: () =>
        deps.maintenanceScheduler.stop({
          graceful: true,
        }),
    },
    {
      label: "chat response catch-up worker",
      close: () =>
        deps.chatResponseCatchUpWorker.stop({
          graceful: true,
        }),
    },
  ] as const;

  const schedulerStopResults = await Promise.allSettled(
    schedulerStops.map(async (stop) => await stop.close()),
  );

  schedulerStopResults.forEach((result, index) => {
    if (result.status === "fulfilled") {
      return;
    }

    const schedulerStop = schedulerStops[index];

    if (schedulerStop !== undefined) {
      collectCloseError(errors, schedulerStop.label, result.reason);
    }
  });

  await Promise.allSettled(admittedOperations);
  try {
    await deps.streamIngestionCoordinator?.close();
  } catch (error) {
    collectCloseError(errors, "stream ingestion coordinator", error);
  }

  // Startup work may enqueue repository writes; wait for it before snapshotting
  // repository queues so those tails remain covered by the bank lease.
  const startupResults = await Promise.allSettled(deps.pendingStartupTasks ?? []);
  try {
    await deps.semanticReviewService?.flushDuplicateReviews?.();
  } catch (error) {
    collectCloseError(errors, "semantic duplicate reviews", error);
  }
  try {
    await deps.reviewQueueRepository?.flushEnqueueHooks?.();
  } catch (error) {
    collectCloseError(errors, "review queue enqueue hooks", error);
  }
  const pendingResults = await Promise.allSettled([
    deps.actionRepository?.waitForPendingEmbeddings(),
    deps.openQuestionsRepository?.waitForPendingEmbeddings(),
    deps.observedEventRepository?.waitForPendingEmbeddings(),
  ]);
  for (const result of [...startupResults, ...pendingResults]) {
    if (result.status === "rejected") {
      collectCloseError(errors, "pending embeddings/startup work", result.reason);
    }
  }
  try {
    await deps.failureReports?.flush();
  } catch (error) {
    collectCloseError(errors, "stream failure reports", error);
  }
  if (errors.length > 0) {
    throw new AggregateError(errors, "One or more Borg dependencies failed to drain");
  }
}

export async function closeBorgDependencies(
  deps: BorgDependencies,
  options: { drain?: false | (() => Promise<void>); retainBankLease?: boolean } = {},
): Promise<void> {
  const errors: unknown[] = [];
  if (options.drain !== false) {
    try {
      await (options.drain?.() ?? drainBorgDependencies(deps));
    } catch (error) {
      collectCloseError(errors, "runtime", error);
    }
  }
  try {
    deps.sqlite.close();
  } catch (error) {
    collectCloseError(errors, "SQLite database", error);
  }
  try {
    await deps.lance.close();
  } catch (error) {
    collectCloseError(errors, "LanceDB store", error);
  }

  // A failed/timed-out host drain cannot prove that background writers drained,
  // even if Borg's own drain completed. Retain the lease until process exit.
  if (errors.length === 0 && options.drain !== false && options.retainBankLease !== true) {
    await deps.releaseEmbeddingBankAccess?.();
  }

  if (errors.length > 0) {
    throw new AggregateError(errors, "One or more Borg dependencies failed to close");
  }
}
