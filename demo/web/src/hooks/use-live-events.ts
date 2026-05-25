import { useCallback, useEffect, useRef, useState } from "react";

import { liveUrl } from "../api/client";
import type { LiveFrame, WsState } from "../api/types";

export type LiveEventHandler = (frame: LiveFrame) => void;

export type LiveEvents = {
  wsState: WsState;
  connectionCount: number;
  subscribe: (handler: LiveEventHandler) => () => void;
};

const DOWN_AFTER_FAILED_ATTEMPTS = 5;

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

export function useLiveEvents(input: { onReconnected?: () => void } = {}): LiveEvents {
  const [wsState, setWsState] = useState<WsState>("reconnecting");
  const [connectionCount, setConnectionCount] = useState(0);
  const handlersRef = useRef(new Set<LiveEventHandler>());
  const reconnectTimerRef = useRef<number | null>(null);
  const onReconnectedRef = useRef(input.onReconnected);

  onReconnectedRef.current = input.onReconnected;

  const subscribe = useCallback((handler: LiveEventHandler) => {
    handlersRef.current.add(handler);
    return () => {
      handlersRef.current.delete(handler);
    };
  }, []);

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

      socket.addEventListener("open", () => {
        if (disposed) {
          return;
        }
        const reconnected = openedOnce;
        openedOnce = true;
        attempt = 0;
        setWsState("live");
        setConnectionCount((count) => count + 1);
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
      socket?.close();
    };
  }, []);

  return { wsState, connectionCount, subscribe };
}
