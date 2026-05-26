import { useCallback, useEffect, useRef, useState } from "react";

export type ApiHookState<T> = {
  data: T | null;
  loading: boolean;
  error: Error | null;
  refetch: () => Promise<void>;
};

export function useApi<T>(
  loader: () => Promise<T>,
  deps: readonly unknown[] = [],
): ApiHookState<T> {
  const [data, setData] = useState<T | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<Error | null>(null);
  const requestSeqRef = useRef(0);
  const mountedRef = useRef(true);

  useEffect(
    () => () => {
      mountedRef.current = false;
    },
    [],
  );

  const refetch = useCallback(async () => {
    const requestSeq = requestSeqRef.current + 1;
    requestSeqRef.current = requestSeq;
    setLoading(true);
    setError(null);
    try {
      const result = await loader();
      if (mountedRef.current && requestSeqRef.current === requestSeq) {
        setData(result);
      }
    } catch (caught) {
      if (mountedRef.current && requestSeqRef.current === requestSeq) {
        setError(caught instanceof Error ? caught : new Error(String(caught)));
      }
    } finally {
      if (mountedRef.current && requestSeqRef.current === requestSeq) {
        setLoading(false);
      }
    }
  }, deps);

  useEffect(() => {
    let cancelled = false;
    const requestSeq = requestSeqRef.current + 1;
    requestSeqRef.current = requestSeq;
    setLoading(true);
    setError(null);

    void loader()
      .then((result) => {
        if (!cancelled && requestSeqRef.current === requestSeq) {
          setData(result);
        }
      })
      .catch((caught: unknown) => {
        if (!cancelled && requestSeqRef.current === requestSeq) {
          setError(caught instanceof Error ? caught : new Error(String(caught)));
        }
      })
      .finally(() => {
        if (!cancelled && requestSeqRef.current === requestSeq) {
          setLoading(false);
        }
      });

    return () => {
      cancelled = true;
    };
  }, deps);

  return { data, loading, error, refetch };
}
