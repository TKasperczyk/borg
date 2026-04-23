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
  try {
    await deps.autonomyScheduler.stop({
      graceful: true,
    });
    await deps.streamIngestionCoordinator?.close();
  } finally {
    deps.sqlite.close();
    await deps.lance.close();
  }
}
