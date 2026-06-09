import { useCallback, useEffect, useMemo, useRef, useState } from "react";

import { liveUrl } from "../api/client";
import type { LiveFrame, WsState } from "../api/types";
import { DEFAULT_DEMO_SESSION_ID } from "./use-session";

export type LiveEventHandler = (frame: LiveFrame) => void;

export type LiveEvents = {
  wsState: WsState;
  connectionCount: number;
  subscribe: (handler: LiveEventHandler) => () => void;
};

export const DOWN_AFTER_FAILED_ATTEMPTS = 5;
const SOCKET_OPEN = 1;

function isLiveFrame(value: unknown): value is LiveFrame {
  return (
    typeof value === "object" &&
    value !== null &&
    "type" in value &&
    typeof (value as { type?: unknown }).type === "string" &&
    "ts" in value &&
    typeof (value as { ts?: unknown }).ts === "number"
  );
}

function reconnectDelay(attempt: number): number {
  const base = Math.min(5_000, 250 * 2 ** attempt);
  return base + Math.floor(Math.random() * 250);
}

export function useLiveEvents(
  input: { onReconnected?: () => void; sessionId?: string } = {},
): LiveEvents {
  const [wsState, setWsState] = useState<WsState>("reconnecting");
  const [connectionCount, setConnectionCount] = useState(0);
  const handlersRef = useRef(new Set<LiveEventHandler>());
  const reconnectTimerRef = useRef<number | null>(null);
  const onReconnectedRef = useRef(input.onReconnected);
  const socketRef = useRef<WebSocket | null>(null);
  const sessionIdRef = useRef(input.sessionId ?? DEFAULT_DEMO_SESSION_ID);
  const subscribedSessionRef = useRef<string | null>(null);

  onReconnectedRef.current = input.onReconnected;
  sessionIdRef.current = input.sessionId ?? DEFAULT_DEMO_SESSION_ID;

  const subscribe = useCallback((handler: LiveEventHandler) => {
    handlersRef.current.add(handler);
    return () => {
      handlersRef.current.delete(handler);
    };
  }, []);

  const sendSessionSubscription = useCallback(
    (type: "subscribe" | "unsubscribe", sessionId: string) => {
      const socket = socketRef.current;
      if (socket === null || socket.readyState !== SOCKET_OPEN) {
        return;
      }
      socket.send(JSON.stringify({ type, session_id: sessionId }));
    },
    [],
  );

  useEffect(() => {
    let disposed = false;
    let socket: WebSocket | null = null;
    let attempt = 0;
    let openedOnce = false;

    const clearReconnectTimer = () => {
      if (reconnectTimerRef.current !== null) {
        window.clearTimeout(reconnectTimerRef.current);
        reconnectTimerRef.current = null;
      }
    };

    const connect = () => {
      clearReconnectTimer();

      if (disposed) {
        return;
      }

      setWsState(attempt >= DOWN_AFTER_FAILED_ATTEMPTS ? "down" : "reconnecting");
      socket = new WebSocket(liveUrl());
      socketRef.current = socket;

      socket.addEventListener("open", () => {
        if (disposed) {
          return;
        }
        const reconnected = openedOnce;
        openedOnce = true;
        attempt = 0;
        setWsState("live");
        setConnectionCount((count) => count + 1);
        const sessionId = sessionIdRef.current;
        subscribedSessionRef.current = sessionId;
        socket?.send(JSON.stringify({ type: "subscribe", session_id: sessionId }));
        if (reconnected) {
          onReconnectedRef.current?.();
        }
      });

      socket.addEventListener("message", (event) => {
        let parsed: unknown;
        try {
          parsed = JSON.parse(String(event.data));
        } catch {
          return;
        }

        if (!isLiveFrame(parsed)) {
          return;
        }

        for (const handler of handlersRef.current) {
          handler(parsed);
        }
      });

      const scheduleReconnect = () => {
        if (disposed) {
          return;
        }
        if (socketRef.current === socket) {
          socketRef.current = null;
        }
        subscribedSessionRef.current = null;
        const delay = reconnectDelay(attempt);
        attempt += 1;
        setWsState(attempt >= DOWN_AFTER_FAILED_ATTEMPTS ? "down" : "reconnecting");
        reconnectTimerRef.current = window.setTimeout(connect, delay);
      };

      socket.addEventListener("close", scheduleReconnect);
      socket.addEventListener("error", () => {
        socket?.close();
      });
    };

    connect();

    return () => {
      disposed = true;
      clearReconnectTimer();
      socketRef.current = null;
      subscribedSessionRef.current = null;
      socket?.close();
    };
  }, []);

  useEffect(() => {
    const nextSessionId = input.sessionId ?? DEFAULT_DEMO_SESSION_ID;
    const previousSessionId = subscribedSessionRef.current;

    if (previousSessionId === nextSessionId) {
      return;
    }

    if (previousSessionId !== null) {
      sendSessionSubscription("unsubscribe", previousSessionId);
    }
    sendSessionSubscription("subscribe", nextSessionId);

    if (socketRef.current?.readyState === SOCKET_OPEN) {
      subscribedSessionRef.current = nextSessionId;
    }
  }, [input.sessionId, sendSessionSubscription]);

  return useMemo(
    () => ({ wsState, connectionCount, subscribe }),
    [connectionCount, subscribe, wsState],
  );
}
