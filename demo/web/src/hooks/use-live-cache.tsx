import {
  createContext,
  useCallback,
  useContext,
  useEffect,
  useMemo,
  useRef,
  useState,
  type ReactNode,
} from "react";

import { getSessions, getState } from "../api/client";
import type {
  DreamProcessStartedFrame,
  MaintenanceTickFrame,
  SessionsResponse,
  StateSnapshot,
  WsState,
} from "../api/types";
import { useApi, type ApiHookState } from "./use-api";
import { useLiveEventsContext } from "./live-context";

const STREAM_REFETCH_DEBOUNCE_MS = 250;

export type DreamActivity = Pick<DreamProcessStartedFrame, "process" | "phase" | "run_id">;

export type LiveCacheValue = {
  stateApi: ApiHookState<StateSnapshot>;
  counts: StateSnapshot["counts"] | null;
  sessionsApi: ApiHookState<SessionsResponse>;
  sessionActivity: ReadonlyMap<string, number>;
  lastMaintenanceTick: MaintenanceTickFrame | null;
  dreamActivity: DreamActivity | null;
  wsState: WsState;
  connectionCount: number;
};

export type LiveCacheProviderProps = {
  sessionId: string;
  children: ReactNode;
};

const LiveCacheContext = createContext<LiveCacheValue | null>(null);

export function LiveCacheProvider({ sessionId, children }: LiveCacheProviderProps) {
  const live = useLiveEventsContext();
  const stateApi = useApi(() => getState({ session: sessionId }), [sessionId]);
  const sessionsApi = useApi(getSessions, []);
  const refetchState = stateApi.refetch;
  const refetchSessions = sessionsApi.refetch;
  const [sessionActivity, setSessionActivity] = useState<ReadonlyMap<string, number>>(
    () => new Map(),
  );
  const [lastMaintenanceTick, setLastMaintenanceTick] = useState<MaintenanceTickFrame | null>(null);
  const [dreamActivity, setDreamActivity] = useState<DreamActivity | null>(null);
  const streamRefetchTimerRef = useRef<number | null>(null);
  const previousConnectionCountRef = useRef(live.connectionCount);

  const clearStreamRefetchTimer = useCallback(() => {
    if (streamRefetchTimerRef.current !== null) {
      window.clearTimeout(streamRefetchTimerRef.current);
      streamRefetchTimerRef.current = null;
    }
  }, []);

  const scheduleStreamRefetch = useCallback(() => {
    clearStreamRefetchTimer();
    streamRefetchTimerRef.current = window.setTimeout(() => {
      streamRefetchTimerRef.current = null;
      void refetchState();
      void refetchSessions();
    }, STREAM_REFETCH_DEBOUNCE_MS);
  }, [clearStreamRefetchTimer, refetchSessions, refetchState]);

  useEffect(() => {
    const sessions = sessionsApi.data?.sessions;
    if (sessions === undefined) {
      return;
    }

    setSessionActivity((current) => {
      const next = new Map<string, number>();
      for (const session of sessions) {
        const previous = current.get(session.session_id);
        next.set(
          session.session_id,
          previous === undefined
            ? session.last_activity_at
            : Math.max(previous, session.last_activity_at),
        );
      }
      return next;
    });
  }, [sessionsApi.data]);

  useEffect(() => {
    const unsubscribe = live.subscribe((frame) => {
      if (frame.type === "stream:append") {
        if (frame.entries.some((entry) => entry.session_id === sessionId)) {
          setSessionActivity((current) => {
            const next = new Map(current);
            next.set(sessionId, Math.max(current.get(sessionId) ?? 0, Date.now(), frame.ts));
            return next;
          });
        }
        scheduleStreamRefetch();
        return;
      }

      if (frame.type === "maintenance:tick") {
        setLastMaintenanceTick(frame);
        void refetchState();
        return;
      }

      if (frame.type === "dream:process:started") {
        setDreamActivity({
          process: frame.process,
          phase: frame.phase,
          run_id: frame.run_id,
        });
        return;
      }

      if (frame.type === "dream:process:completed") {
        setDreamActivity((current) => {
          if (
            current !== null &&
            current.process === frame.process &&
            current.run_id === frame.run_id &&
            current.phase === frame.phase
          ) {
            return null;
          }
          return current;
        });
        return;
      }

      if (frame.type === "borg:reset") {
        window.location.reload();
      }
    });
    return () => {
      unsubscribe();
      clearStreamRefetchTimer();
    };
  }, [clearStreamRefetchTimer, live, refetchState, scheduleStreamRefetch, sessionId]);

  useEffect(
    () => () => {
      clearStreamRefetchTimer();
    },
    [clearStreamRefetchTimer],
  );

  useEffect(() => {
    const previousConnectionCount = previousConnectionCountRef.current;
    previousConnectionCountRef.current = live.connectionCount;

    if (live.connectionCount <= 1 || live.connectionCount === previousConnectionCount) {
      return;
    }

    void refetchState();
    void refetchSessions();
  }, [live.connectionCount, refetchSessions, refetchState]);

  const value = useMemo<LiveCacheValue>(
    () => ({
      stateApi,
      counts: stateApi.data?.counts ?? null,
      sessionsApi,
      sessionActivity,
      lastMaintenanceTick,
      dreamActivity,
      wsState: live.wsState,
      connectionCount: live.connectionCount,
    }),
    [
      dreamActivity,
      lastMaintenanceTick,
      live.connectionCount,
      live.wsState,
      sessionActivity,
      sessionsApi,
      stateApi,
    ],
  );

  return <LiveCacheContext.Provider value={value}>{children}</LiveCacheContext.Provider>;
}

export function useLiveCache(): LiveCacheValue {
  const value = useContext(LiveCacheContext);
  if (value === null) {
    throw new Error("LiveCacheContext is not available");
  }
  return value;
}
