import { SystemClock, type Clock } from "./clock.js";
import { raceWithSignal } from "./deadline.js";

/** Infrastructure state only; no inference from tool arguments or model output. */
export type TimeoutLoadSignals = {
  maintenanceActive?: boolean;
  sessionBusy?: boolean;
  embeddingRetryAt?: number | null;
  embeddingRequestStartedAt?: number | null;
};

export type TimeoutBudget = { timeoutMs: number; maxTimeoutMs: number };
export type TimeoutObservation = {
  budgetMs: number;
  elapsedMs: number;
  extensionReasons: string[];
  outcome: "selected" | "extended" | "completed" | "timed_out" | "failed";
};

export class TimeoutBudgetExceeded extends Error {
  constructor(readonly timeoutMs: number) {
    super(`Timed out after ${timeoutMs}ms`);
    this.name = "TimeoutBudgetExceeded";
  }
}

/** One operation, one start, at most one extension. A caller deadline always wins. */
export async function runWithTimeoutPolicy<T>(
  options: TimeoutBudget & {
    deadlineAt?: number;
    signal?: AbortSignal;
    clock?: Clock;
    loadSignals?: () => TimeoutLoadSignals;
    observe?: (observation: TimeoutObservation) => void;
    run: (signal: AbortSignal) => Promise<T>;
  },
): Promise<T> {
  const clock = options.clock ?? new SystemClock();
  const startedAt = performance.now();
  const remaining =
    options.deadlineAt === undefined ? Infinity : Math.max(0, options.deadlineAt - clock.now());
  const maximum = Math.min(options.maxTimeoutMs, remaining);
  const base = Math.min(options.timeoutMs, maximum);
  const controller = new AbortController();
  const signal =
    options.signal === undefined
      ? controller.signal
      : AbortSignal.any([options.signal, controller.signal]);
  let budgetMs = base;
  let extensionReasons: string[] = [];
  let timer: ReturnType<typeof setTimeout> | undefined;
  const elapsed = () => Math.max(0, performance.now() - startedAt);
  const observe = (outcome: TimeoutObservation["outcome"]) =>
    options.observe?.({ budgetMs, elapsedMs: elapsed(), extensionReasons, outcome });
  const extend = () => {
    if (budgetMs >= maximum) return false;
    const state = options.loadSignals?.() ?? {};
    const now = clock.now();
    const reasons = [
      ...(state.maintenanceActive ? ["maintenance_active"] : []),
      ...(state.sessionBusy ? ["session_busy"] : []),
      ...((state.embeddingRetryAt ?? 0) > now ? ["embedding_backoff"] : []),
      ...(state.embeddingRequestStartedAt != null &&
      now - state.embeddingRequestStartedAt >= base / 2
        ? ["embedding_slow"]
        : []),
    ];
    if (reasons.length === 0) return false;
    extensionReasons = reasons;
    budgetMs = maximum;
    return true;
  };
  const expire = () => {
    if (elapsed() < maximum && extend()) {
      observe("extended");
      timer = setTimeout(expire, Math.max(0, budgetMs - elapsed()));
      timer.unref?.();
    } else {
      controller.abort(new TimeoutBudgetExceeded(budgetMs));
    }
  };
  extend();
  observe("selected");
  const remainingMs = budgetMs - elapsed();
  if (remainingMs <= 0) controller.abort(new TimeoutBudgetExceeded(budgetMs));
  else {
    timer = setTimeout(expire, remainingMs);
    timer.unref?.();
  }
  try {
    const value = await raceWithSignal(() => options.run(signal), signal);
    // A synchronous event-loop hold can deliver a result before an overdue timer.
    if (elapsed() >= budgetMs) {
      controller.abort(new TimeoutBudgetExceeded(budgetMs));
      throw controller.signal.reason;
    }
    observe("completed");
    return value;
  } catch (error) {
    observe(error instanceof TimeoutBudgetExceeded ? "timed_out" : "failed");
    throw error;
  } finally {
    if (timer !== undefined) clearTimeout(timer);
  }
}
