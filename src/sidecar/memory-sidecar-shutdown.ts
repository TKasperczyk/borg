import type { MaintenanceRunId } from "../util/ids.js";

export type MemorySidecarShutdownPhase =
  | { status: "completed" }
  | { status: "timed_out" }
  | { status: "error"; error: unknown };

export type MemorySidecarShutdownResult = {
  abandonedRunIds: MaintenanceRunId[];
  http: MemorySidecarShutdownPhase;
  pool: MemorySidecarShutdownPhase;
};

export type DrainMemorySidecarOptions = {
  timeoutMs: number;
  beginShutdown: () => MaintenanceRunId[];
  forceFinalizeMaintenance?: () => MaintenanceRunId[];
  closeIdleConnections: () => void;
  closeHttp: () => Promise<void>;
  shutdownPool: () => Promise<void>;
  onAbandoned?: (runIds: readonly MaintenanceRunId[]) => void;
  now?: () => number;
};

async function settleBeforeDeadline(
  work: () => Promise<void>,
  deadlineReached: Promise<void>,
): Promise<MemorySidecarShutdownPhase> {
  let completed: Promise<MemorySidecarShutdownPhase>;
  try {
    completed = work().then<MemorySidecarShutdownPhase, MemorySidecarShutdownPhase>(
      () => ({ status: "completed" }),
      (error: unknown) => ({ status: "error", error }),
    );
  } catch (error) {
    completed = Promise.resolve({ status: "error", error });
  }
  return Promise.race([
    completed,
    deadlineReached.then<MemorySidecarShutdownPhase>(() => ({ status: "timed_out" })),
  ]);
}

export async function drainMemorySidecar(
  options: DrainMemorySidecarOptions,
): Promise<MemorySidecarShutdownResult> {
  const now = options.now ?? Date.now;
  const deadline = now() + Math.max(0, options.timeoutMs);

  // Request a cooperative stop first. Running chunks may finish and report
  // until the one shared absolute deadline force-finalizes anything left.
  const shutdownRunIds = options.beginShutdown();
  let abandonedRunIds: MaintenanceRunId[] = [];
  let didReachDeadline = false;
  let deadlineTimer: ReturnType<typeof setTimeout> | undefined;
  const deadlineReached = new Promise<void>((resolve) => {
    deadlineTimer = setTimeout(
      () => {
        didReachDeadline = true;
        abandonedRunIds = options.forceFinalizeMaintenance?.() ?? shutdownRunIds;
        options.onAbandoned?.(abandonedRunIds);
        resolve();
      },
      Math.max(0, deadline - now()),
    );
    deadlineTimer.unref?.();
  });

  const http = await settleBeforeDeadline(async () => {
    options.closeIdleConnections();
    await options.closeHttp();
  }, deadlineReached);
  // Let accepted HTTP requests drain before closing their Borg instances. The
  // pool receives only the time left on the same absolute process deadline.
  const pool = await settleBeforeDeadline(options.shutdownPool, deadlineReached);

  if (!didReachDeadline && deadlineTimer !== undefined) {
    clearTimeout(deadlineTimer);
  }

  return {
    abandonedRunIds,
    http,
    pool,
  };
}
