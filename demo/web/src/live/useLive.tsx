import {
  createContext,
  type ReactNode,
  useCallback,
  useContext,
  useEffect,
  useMemo,
  useRef,
  useState,
} from "react";

import type { LiveFrame, LiveFrameType } from "../api/types";
import { invalidateQueries } from "../api/useQuery";

export type LiveStatus = "connecting" | "open" | "closed";
type LiveFrameHandler = (frame: LiveFrame) => void;
type FrameHandlerKey = LiveFrameType | "*";

type LiveContextValue = {
  status: LiveStatus;
  sendJson: (payload: unknown) => void;
  subscribeSession: (sessionId: string) => void;
  unsubscribeSession: (sessionId: string) => void;
  onFrame: (type: FrameHandlerKey, handler: LiveFrameHandler) => () => void;
};

const LiveContext = createContext<LiveContextValue | null>(null);

const INVALIDATE_BY_FRAME: Partial<Record<LiveFrameType, string[]>> = {
  "turn:terminal": ["state", "turns", "stream"],
  "stream:append": ["stream", "turns"],
  "maintenance:tick": ["state", "dream"],
  "dream:process:completed": ["dream"],
  "borg:reset": [""],
};

function liveUrl(): string {
  const protocol = window.location.protocol === "https:" ? "wss:" : "ws:";
  return `${protocol}//${window.location.host}/api/live`;
}

function isLiveFrame(value: unknown): value is LiveFrame {
  return (
    typeof value === "object" &&
    value !== null &&
    "type" in value &&
    typeof (value as { type?: unknown }).type === "string"
  );
}

function reportFrameHandlerError(frame: LiveFrame, error: unknown): void {
  console.error("Live frame handler failed", {
    type: frame.type,
    cause: error instanceof Error ? error.message : String(error),
  });
}

export function LiveProvider({ children }: { children: ReactNode }) {
  const [status, setStatus] = useState<LiveStatus>("connecting");
  const socketRef = useRef<WebSocket | null>(null);
  const closedRef = useRef(false);
  const reconnectTimerRef = useRef<number | null>(null);
  const retryRef = useRef(0);
  const subscriptionsRef = useRef(new Map<string, number>());
  const handlersRef = useRef(new Map<FrameHandlerKey, Set<LiveFrameHandler>>());
  const invalidationTimersRef = useRef(new Map<string, number>());

  const sendJson = useCallback((payload: unknown) => {
    const socket = socketRef.current;
    if (socket?.readyState === WebSocket.OPEN) {
      socket.send(JSON.stringify(payload));
    }
  }, []);

  const debounceInvalidation = useCallback((prefix: string) => {
    const timers = invalidationTimersRef.current;
    const existing = timers.get(prefix);
    if (existing !== undefined) {
      window.clearTimeout(existing);
    }

    const timer = window.setTimeout(() => {
      timers.delete(prefix);
      invalidateQueries(prefix);
    }, 300);
    timers.set(prefix, timer);
  }, []);

  const dispatchFrame = useCallback(
    (frame: LiveFrame) => {
      const prefixes = INVALIDATE_BY_FRAME[frame.type] ?? [];
      for (const prefix of prefixes) {
        debounceInvalidation(prefix);
      }

      for (const handler of handlersRef.current.get(frame.type) ?? []) {
        try {
          handler(frame);
        } catch (error) {
          reportFrameHandlerError(frame, error);
        }
      }
      for (const handler of handlersRef.current.get("*") ?? []) {
        try {
          handler(frame);
        } catch (error) {
          reportFrameHandlerError(frame, error);
        }
      }
    },
    [debounceInvalidation],
  );

  const connect = useCallback((registerSocket: (socket: WebSocket) => void) => {
    if (closedRef.current) {
      return;
    }

    if (typeof WebSocket === "undefined") {
      setStatus("closed");
      return;
    }

    setStatus("connecting");
    const socket = new WebSocket(liveUrl());
    registerSocket(socket);
    socketRef.current = socket;

    socket.addEventListener("open", () => {
      if (socketRef.current !== socket) {
        return;
      }

      retryRef.current = 0;
      setStatus("open");
      for (const [sessionId, count] of subscriptionsRef.current) {
        if (count > 0) {
          socket.send(JSON.stringify({ type: "subscribe", session_id: sessionId }));
        }
      }
    });

    socket.addEventListener("message", (event: MessageEvent) => {
      if (socketRef.current !== socket) {
        return;
      }

      try {
        const parsed = JSON.parse(String(event.data)) as unknown;
        if (isLiveFrame(parsed)) {
          dispatchFrame(parsed);
        }
      } catch {
        // Ignore malformed frames; the live channel is observational.
      }
    });

    const scheduleReconnect = () => {
      if (
        socketRef.current !== socket ||
        closedRef.current ||
        reconnectTimerRef.current !== null
      ) {
        return;
      }

      setStatus("closed");
      const retry = retryRef.current;
      retryRef.current += 1;
      const delay = Math.min(8_000, 500 * 2 ** retry) + Math.floor(Math.random() * 250);
      reconnectTimerRef.current = window.setTimeout(() => {
        reconnectTimerRef.current = null;
        connect(registerSocket);
      }, delay);
    };

    socket.addEventListener("close", scheduleReconnect);
    socket.addEventListener("error", scheduleReconnect);
  }, [dispatchFrame]);

  useEffect(() => {
    const ownedSockets = new Set<WebSocket>();
    const registerSocket = (socket: WebSocket) => {
      ownedSockets.add(socket);
    };

    closedRef.current = false;
    connect(registerSocket);

    return () => {
      closedRef.current = true;
      if (reconnectTimerRef.current !== null) {
        window.clearTimeout(reconnectTimerRef.current);
        reconnectTimerRef.current = null;
      }
      for (const timer of invalidationTimersRef.current.values()) {
        window.clearTimeout(timer);
      }
      invalidationTimersRef.current.clear();
      for (const socket of ownedSockets) {
        if (socketRef.current === socket) {
          socketRef.current = null;
        }
        socket.close();
      }
    };
  }, [connect]);

  const subscribeSession = useCallback(
    (sessionId: string) => {
      const counts = subscriptionsRef.current;
      const count = counts.get(sessionId) ?? 0;
      counts.set(sessionId, count + 1);
      if (count === 0) {
        sendJson({ type: "subscribe", session_id: sessionId });
      }
    },
    [sendJson],
  );

  const unsubscribeSession = useCallback(
    (sessionId: string) => {
      const counts = subscriptionsRef.current;
      const count = counts.get(sessionId) ?? 0;
      if (count <= 1) {
        counts.delete(sessionId);
        sendJson({ type: "unsubscribe", session_id: sessionId });
        return;
      }

      counts.set(sessionId, count - 1);
    },
    [sendJson],
  );

  const onFrame = useCallback((type: FrameHandlerKey, handler: LiveFrameHandler) => {
    const handlers = handlersRef.current.get(type) ?? new Set<LiveFrameHandler>();
    handlers.add(handler);
    handlersRef.current.set(type, handlers);

    return () => {
      handlers.delete(handler);
      if (handlers.size === 0) {
        handlersRef.current.delete(type);
      }
    };
  }, []);

  const value = useMemo<LiveContextValue>(
    () => ({
      status,
      sendJson,
      subscribeSession,
      unsubscribeSession,
      onFrame,
    }),
    [onFrame, sendJson, status, subscribeSession, unsubscribeSession],
  );

  return <LiveContext.Provider value={value}>{children}</LiveContext.Provider>;
}

export function useLive(): LiveContextValue {
  const live = useContext(LiveContext);
  if (live === null) {
    throw new Error("useLive must be used within LiveProvider");
  }

  return live;
}
