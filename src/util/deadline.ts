export type DeadlineSettlement =
  | { status: "completed" }
  | { status: "timed_out" }
  | { status: "error"; error: unknown };

/** A hard caller deadline even when an adapter ignores AbortSignal. */
export async function raceWithSignal<T>(work: () => Promise<T>, signal?: AbortSignal): Promise<T> {
  signal?.throwIfAborted();
  if (signal === undefined) return await work();
  let onAbort: () => void = () => {};
  const aborted = new Promise<never>((_, reject) => {
    onAbort = () => reject(signal.reason);
    signal.addEventListener("abort", onAbort, { once: true });
  });
  try {
    // Promise.race observes late failures as well as the winning settlement.
    return await Promise.race([(async () => await work())(), aborted]);
  } finally {
    signal.removeEventListener("abort", onAbort);
  }
}

export async function runWithDeadline<T>(options: {
  timeoutMs: number;
  signal?: AbortSignal | null;
  timeoutError: () => Error;
  run: (signal: AbortSignal) => Promise<T>;
}): Promise<T> {
  const startedAt = performance.now();
  const controller = new AbortController();
  const signal = options.signal
    ? AbortSignal.any([options.signal, controller.signal])
    : controller.signal;
  const timer = setTimeout(() => controller.abort(options.timeoutError()), options.timeoutMs);
  timer.unref?.();
  try {
    const value = await raceWithSignal(() => options.run(signal), signal);
    if (performance.now() - startedAt >= options.timeoutMs) {
      controller.abort(options.timeoutError());
      throw controller.signal.reason;
    }
    return value;
  } finally {
    clearTimeout(timer);
  }
}

/** Observe work until a caller-owned deadline; late rejections remain handled. */
export async function settleBeforeDeadline(
  work: () => Promise<void>,
  deadlineReached: Promise<void>,
): Promise<DeadlineSettlement> {
  const completed = (async (): Promise<DeadlineSettlement> => {
    try {
      await work();
      return { status: "completed" };
    } catch (error) {
      return { status: "error", error };
    }
  })();
  const deadline = (async (): Promise<DeadlineSettlement> => {
    await deadlineReached;
    return { status: "timed_out" };
  })();
  return await Promise.race([completed, deadline]);
}
