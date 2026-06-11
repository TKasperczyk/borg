import { useCallback, useEffect, useRef, useState } from "react";

type QueryState<T> = {
  data: T | undefined;
  error: Error | undefined;
  loading: boolean;
};

type QueryRegistration = {
  key: string;
  refetch: () => void;
};

const activeQueries = new Set<QueryRegistration>();

function asError(error: unknown): Error {
  return error instanceof Error ? error : new Error(String(error));
}

export function invalidateQueries(prefix: string): void {
  for (const query of activeQueries) {
    if (query.key.startsWith(prefix)) {
      query.refetch();
    }
  }
}

export function useQuery<T>(
  key: string,
  fn: () => Promise<T>,
): QueryState<T> & { refetch: () => void } {
  const fnRef = useRef(fn);
  const mountedRef = useRef(false);
  const runIdRef = useRef(0);
  const [state, setState] = useState<QueryState<T>>({
    data: undefined,
    error: undefined,
    loading: true,
  });

  useEffect(() => {
    fnRef.current = fn;
  }, [fn]);

  const refetch = useCallback(() => {
    const runId = runIdRef.current + 1;
    runIdRef.current = runId;
    setState((current) => ({ ...current, error: undefined, loading: true }));

    void fnRef
      .current()
      .then((data) => {
        if (mountedRef.current && runIdRef.current === runId) {
          setState({ data, error: undefined, loading: false });
        }
      })
      .catch((error: unknown) => {
        if (mountedRef.current && runIdRef.current === runId) {
          setState((current) => ({
            ...current,
            error: asError(error),
            loading: false,
          }));
        }
      });
  }, []);

  useEffect(() => {
    mountedRef.current = true;
    const registration: QueryRegistration = { key, refetch };
    activeQueries.add(registration);
    refetch();

    return () => {
      mountedRef.current = false;
      activeQueries.delete(registration);
    };
  }, [key, refetch]);

  return { ...state, refetch };
}
