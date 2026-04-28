// Lifecycle helpers for orderly shutdown and partial-open cleanup.

import type { LanceDbStore } from "../storage/lancedb/index.js";
import type { SqliteDatabase } from "../storage/sqlite/index.js";
import type { BorgDependencies } from "./types.js";

export async function closeBestEffort(
  sqlite: SqliteDatabase | undefined,
  lance: LanceDbStore | undefined,
): Promise<void> {
  if (sqlite !== undefined) {
    try {
      sqlite.close();
    } catch {
      // Best-effort cleanup after a partial Borg.open failure.
    }
  }

  if (lance !== undefined) {
    try {
      await lance.close();
    } catch {
      // Best-effort cleanup after a partial Borg.open failure.
    }
  }
}

export async function closeBorgDependencies(deps: BorgDependencies): Promise<void> {
  const errors: unknown[] = [];
  const collectCloseError = (label: string, error: unknown): void => {
    errors.push(error);
    console.error(`Failed to close ${label}`, error);
  };

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
  ] as const;

  const schedulerStopResults = await Promise.allSettled(
    schedulerStops.map((stop) => Promise.resolve().then(() => stop.close())),
  );

  schedulerStopResults.forEach((result, index) => {
    if (result.status === "fulfilled") {
      return;
    }

    const schedulerStop = schedulerStops[index];

    if (schedulerStop !== undefined) {
      collectCloseError(schedulerStop.label, result.reason);
    }
  });

  try {
    await deps.streamIngestionCoordinator?.close();
  } catch (error) {
    collectCloseError("stream ingestion coordinator", error);
  } finally {
    try {
      deps.sqlite.close();
    } catch (error) {
      collectCloseError("SQLite database", error);
    }

    try {
      await deps.lance.close();
    } catch (error) {
      collectCloseError("LanceDB store", error);
    }
  }

  if (errors.length > 0) {
    throw new AggregateError(errors, "One or more Borg dependencies failed to close");
  }
}
