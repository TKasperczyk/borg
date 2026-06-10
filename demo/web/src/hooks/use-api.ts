import { useCallback, useEffect, useRef, useState } from "react";

import { ApiError } from "../api/client";

export const API_RETRY_INITIAL_DELAY_MS = 500;
export const API_RETRY_MAX_DELAY_MS = 30_000;
export const API_RETRY_JITTER_MS = 250;

type ApiRetryOptions = {
  /** Enables automatic retry for retryable transport failures. */
  enabled?: boolean;
  initialDelayMs?: number;
  maxDelayMs?: number;
  jitterMs?: number;
  /** Default retries network failures and 5xx API errors; 4xx API errors are not retried. */
  shouldRetry?: (error: Error) => boolean;
};

export type UseApiOptions = {
  /** Optional resilience policy. Omitted/false preserves the legacy no-auto-retry behavior. */
  retry?: boolean | ApiRetryOptions;
  /** Changing this value asks the hook to revalidate and coalesces with any in-flight resilient request. */
  revalidateKey?: unknown;
};

export type ApiHookState<T> = {
  /** Last successful payload. Retained during revalidation and failed refreshes. */
  data: T | null;
  loading: boolean;
  /** Last refresh error. Cleared only after a later refresh succeeds. */
  error: Error | null;
  refetch: () => Promise<void>;
  retry: () => Promise<void>;
  /** True when data is present but a refresh is loading or has failed. */
  isStale: boolean;
  /**
   * Refresh-health flag: true after any failed refresh until a later refresh succeeds.
   * This includes non-retryable 4xx responses; when data is present, shown data may be stale.
   */
  degraded: boolean;
  /** True only while an automatic retry for a retryable error is scheduled or in flight. */
  retrying: boolean;
};

type ResolvedRetryOptions = Required<
  Pick<ApiRetryOptions, "enabled" | "initialDelayMs" | "maxDelayMs" | "jitterMs" | "shouldRetry">
>;

function isRetryableTransportError(error: Error): boolean {
  if (error instanceof ApiError) {
    return error.status >= 500;
  }
  return true;
}

function resolveRetryOptions(input: UseApiOptions["retry"]): ResolvedRetryOptions {
  if (input === undefined || input === false) {
    return {
      enabled: false,
      initialDelayMs: API_RETRY_INITIAL_DELAY_MS,
      maxDelayMs: API_RETRY_MAX_DELAY_MS,
      jitterMs: API_RETRY_JITTER_MS,
      shouldRetry: isRetryableTransportError,
    };
  }

  if (input === true) {
    return {
      enabled: true,
      initialDelayMs: API_RETRY_INITIAL_DELAY_MS,
      maxDelayMs: API_RETRY_MAX_DELAY_MS,
      jitterMs: API_RETRY_JITTER_MS,
      shouldRetry: isRetryableTransportError,
    };
  }

  return {
    enabled: input.enabled ?? true,
    initialDelayMs: input.initialDelayMs ?? API_RETRY_INITIAL_DELAY_MS,
    maxDelayMs: input.maxDelayMs ?? API_RETRY_MAX_DELAY_MS,
    jitterMs: input.jitterMs ?? API_RETRY_JITTER_MS,
    shouldRetry: input.shouldRetry ?? isRetryableTransportError,
  };
}

function retryDelay(attempt: number, options: ResolvedRetryOptions): number {
  const base = options.initialDelayMs * 2 ** attempt;
  const jitter = Math.floor(Math.random() * options.jitterMs);
  return Math.min(options.maxDelayMs, base + jitter);
}

type RequestOptions = {
  resetBackoff?: boolean;
  fromRetry?: boolean;
};

export function useApi<T>(
  loader: () => Promise<T>,
  deps: readonly unknown[] = [],
  options: UseApiOptions = {},
): ApiHookState<T> {
  const [data, setData] = useState<T | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<Error | null>(null);
  const [degraded, setDegraded] = useState(false);
  const [retrying, setRetrying] = useState(false);
  const requestSeqRef = useRef(0);
  const mountedRef = useRef(true);
  const retryTimerRef = useRef<number | null>(null);
  const retryAttemptRef = useRef(0);
  const loaderRef = useRef(loader);
  const inFlightRef = useRef<Promise<void> | null>(null);
  const guardInFlightRef = useRef(false);
  const trailingRequestRef = useRef(false);
  const trailingResetBackoffRef = useRef(false);
  const retryOptionsRef = useRef(resolveRetryOptions(options.retry));
  const revalidateKeyRef = useRef(options.revalidateKey);
  loaderRef.current = loader;
  retryOptionsRef.current = resolveRetryOptions(options.retry);
  guardInFlightRef.current = retryOptionsRef.current.enabled || options.revalidateKey !== undefined;

  const clearRetryTimer = useCallback((updateState = true) => {
    if (retryTimerRef.current !== null) {
      window.clearTimeout(retryTimerRef.current);
      retryTimerRef.current = null;
    }
    if (updateState && mountedRef.current) {
      setRetrying(false);
    }
  }, []);

  useEffect(() => {
    return () => {
      mountedRef.current = false;
      if (retryTimerRef.current !== null) {
        window.clearTimeout(retryTimerRef.current);
        retryTimerRef.current = null;
      }
    };
  }, []);

  const runRequest = useCallback(
    async (requestOptions: RequestOptions = {}) => {
      if (guardInFlightRef.current && inFlightRef.current !== null) {
        trailingRequestRef.current = true;
        trailingResetBackoffRef.current =
          trailingResetBackoffRef.current || requestOptions.resetBackoff === true;
        clearRetryTimer(!requestOptions.fromRetry);
        return inFlightRef.current;
      }

      if (requestOptions.resetBackoff === true) {
        retryAttemptRef.current = 0;
      }

      clearRetryTimer(!requestOptions.fromRetry);
      const requestSeq = requestSeqRef.current + 1;
      requestSeqRef.current = requestSeq;
      if (mountedRef.current) {
        setLoading(true);
      }

      let request!: Promise<void>;
      request = (async () => {
        try {
          const result = await loaderRef.current();
          if (mountedRef.current && requestSeqRef.current === requestSeq) {
            setData(result);
            setError(null);
            setDegraded(false);
            setRetrying(false);
            retryAttemptRef.current = 0;
          }
        } catch (caught) {
          if (mountedRef.current && requestSeqRef.current === requestSeq) {
            const nextError = caught instanceof Error ? caught : new Error(String(caught));
            setError(nextError);
            setDegraded(true);
            const retryOptions = retryOptionsRef.current;
            if (retryOptions.enabled && retryOptions.shouldRetry(nextError)) {
              const delay = retryDelay(retryAttemptRef.current, retryOptions);
              retryAttemptRef.current += 1;
              setRetrying(true);
              retryTimerRef.current = window.setTimeout(() => {
                retryTimerRef.current = null;
                if (mountedRef.current) {
                  void runRequest({ fromRetry: true });
                }
              }, delay);
            } else {
              setRetrying(false);
            }
          }
        } finally {
          if (mountedRef.current && requestSeqRef.current === requestSeq) {
            setLoading(false);
          }
          if (inFlightRef.current === request) {
            inFlightRef.current = null;
          }
          if (mountedRef.current && trailingRequestRef.current) {
            const resetBackoff = trailingResetBackoffRef.current;
            trailingRequestRef.current = false;
            trailingResetBackoffRef.current = false;
            void runRequest({ resetBackoff });
          }
        }
      })();

      if (guardInFlightRef.current) {
        inFlightRef.current = request;
      }
      return request;
    },
    [clearRetryTimer],
  );

  const refetch = useCallback(async () => {
    await runRequest({ resetBackoff: true });
  }, [runRequest]);

  useEffect(() => {
    requestSeqRef.current += 1;
    trailingRequestRef.current = false;
    trailingResetBackoffRef.current = false;
    retryAttemptRef.current = 0;
    void runRequest({ resetBackoff: true });

    return () => {
      requestSeqRef.current += 1;
      clearRetryTimer(false);
    };
  }, [clearRetryTimer, runRequest, ...deps]);

  useEffect(() => {
    const previous = revalidateKeyRef.current;
    revalidateKeyRef.current = options.revalidateKey;
    if (options.revalidateKey === undefined || Object.is(previous, options.revalidateKey)) {
      return;
    }
    void runRequest({ resetBackoff: true });
  }, [options.revalidateKey, runRequest]);

  return {
    data,
    loading,
    error,
    refetch,
    retry: refetch,
    isStale: data !== null && (loading || degraded),
    degraded,
    retrying,
  };
}
