import { AsyncLocalStorage } from "node:async_hooks";

import type {
  LLMClient,
  LLMCompleteOptions,
  LLMCompleteStreamOptions,
  LLMConverseOptions,
  LLMConverseStreamOptions,
  LLMTransportRetryEvent,
} from "../llm/index.js";

export type TurnExecutionMetrics = {
  finalizer_rounds: number;
  stall_retries: number;
};

type RetryObservableOptions = {
  budget: string;
  onTransportRetry?: (event: LLMTransportRetryEvent) => void;
};

const metricsByTurnError = new WeakMap<object, TurnExecutionMetrics>();
const observedClients = new WeakMap<LLMClient, LLMClient>();
export const turnExecutionMetricsStorage = new AsyncLocalStorage<TurnExecutionMetrics>();

export function createTurnExecutionMetrics(): TurnExecutionMetrics {
  return {
    finalizer_rounds: 0,
    stall_retries: 0,
  };
}

export function snapshotTurnExecutionMetrics(metrics: TurnExecutionMetrics): TurnExecutionMetrics {
  return { ...metrics };
}

function isFinalizerBudget(budget: string): boolean {
  return budget === "cognition-system-1" || budget === "cognition-system-2";
}

function observedOptions<T extends RetryObservableOptions>(options: T): T {
  const metrics = turnExecutionMetricsStorage.getStore();
  if (metrics === undefined) {
    return options;
  }
  recordFinalizerRound(options, metrics);
  const priorObserver = options.onTransportRetry;

  return {
    ...options,
    onTransportRetry: (event: LLMTransportRetryEvent) => {
      if (event.kind === "stall") {
        metrics.stall_retries += 1;
      }
      priorObserver?.(event);
    },
  };
}

function recordFinalizerRound(
  options: RetryObservableOptions,
  metrics: TurnExecutionMetrics,
): void {
  if (isFinalizerBudget(options.budget)) {
    metrics.finalizer_rounds += 1;
  }
}

/**
 * Observes the same request and retry boundaries that feed the LLM trace while
 * leaving the caller's existing retry observer intact. A finalizer round is one
 * logical LLM request; an in-place transport retry adds only to stall_retries.
 */
export function observeTurnLlmClient(client: LLMClient): LLMClient {
  const existing = observedClients.get(client);
  if (existing !== undefined) {
    return existing;
  }
  const streamComplete = client.streamComplete?.bind(client);
  const streamConverse = client.streamConverse?.bind(client);

  // Return each underlying promise directly. This observer must not add an
  // async microtask boundary that reorders concurrently emitted trace events.
  const observed: LLMClient = {
    complete: (options: LLMCompleteOptions) => {
      return client.complete(observedOptions(options));
    },
    converse: (options: LLMConverseOptions) => {
      return client.converse(observedOptions(options));
    },
    ...(streamComplete === undefined
      ? {}
      : {
          streamComplete: (options: LLMCompleteStreamOptions) => {
            return streamComplete(observedOptions(options));
          },
        }),
    ...(streamConverse === undefined
      ? {}
      : {
          streamConverse: (options: LLMConverseStreamOptions) => {
            return streamConverse(observedOptions(options));
          },
        }),
  };
  observedClients.set(client, observed);
  observedClients.set(observed, observed);
  return observed;
}

export function associateTurnExecutionMetricsWithError(
  error: unknown,
  metrics: TurnExecutionMetrics,
): void {
  if ((typeof error === "object" && error !== null) || typeof error === "function") {
    metricsByTurnError.set(error, snapshotTurnExecutionMetrics(metrics));
  }
}

export function turnExecutionMetricsFromError(error: unknown): TurnExecutionMetrics | null {
  if ((typeof error !== "object" || error === null) && typeof error !== "function") {
    return null;
  }

  return metricsByTurnError.get(error) ?? null;
}
