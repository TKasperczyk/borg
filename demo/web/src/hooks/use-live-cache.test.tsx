import { act, render, screen, waitFor } from "@testing-library/react";
import { afterEach, describe, expect, it, vi } from "vitest";

import type { LiveFrame, SessionRecord, StateSnapshot, StreamEntry, WsState } from "../api/types";
import { LiveEventsProvider } from "./live-context";
import type { LiveEventHandler, LiveEvents } from "./use-live-events";
import { LiveCacheProvider, useLiveCache } from "./use-live-cache";

const realLocation = window.location;

function jsonResponse(body: unknown): Response {
  return new Response(JSON.stringify(body), {
    status: 200,
    headers: { "Content-Type": "application/json" },
  });
}

function requestPath(request: RequestInfo | URL): string {
  return new URL(String(request), "http://test.invalid").pathname;
}

function callsFor(fetchMock: ReturnType<typeof vi.fn>, path: string): number {
  return fetchMock.mock.calls.filter(
    ([request]) => requestPath(request as RequestInfo | URL) === path,
  ).length;
}

function makeLiveSource(): {
  emit: (frame: LiveFrame) => void;
  live: (connectionCount?: number, wsState?: WsState) => LiveEvents;
} {
  const handlers = new Set<LiveEventHandler>();
  return {
    live: (connectionCount = 1, wsState = "live") => ({
      wsState,
      connectionCount,
      subscribe: (handler) => {
        handlers.add(handler);
        return () => {
          handlers.delete(handler);
        };
      },
    }),
    emit: (frame) => {
      for (const handler of handlers) {
        handler(frame);
      }
    },
  };
}

function stateSnapshot(input: Partial<StateSnapshot> = {}): StateSnapshot {
  return {
    active_session: "default",
    audiences: ["alice"],
    counts: {
      turns: 1,
      commitments: 2,
      open_qs: 3,
      open_reviews: 4,
      dream_audit_rows: 5,
    },
    current_mood: {
      session_id: "default",
      valence: 0,
      arousal: 0,
      updated_at: 1_000,
      half_life_hours: 12,
      recent_triggers: [],
    },
    version: "test",
    ...input,
  };
}

function session(input: Partial<SessionRecord> & Pick<SessionRecord, "session_id">): SessionRecord {
  return {
    source_type: "demo",
    source_external_id: null,
    source_url: null,
    label: input.session_id,
    audience_label: "alice",
    audience_entity_id: null,
    conversation_kind: "demo",
    created_at: 1_000,
    last_activity_at: 1_000,
    last_turn_id: null,
    message_count: 0,
    status: "active",
    privacy_level: "payload_off",
    participation_policy: "active",
    audience_role: "participant",
    ...input,
  };
}

function streamEntry(input: Partial<StreamEntry> & Pick<StreamEntry, "session_id">): StreamEntry {
  return {
    id: `strm_${input.session_id}`,
    timestamp: Date.now(),
    kind: "user_msg",
    content: { text: "hello" },
    sender_entity_id: null,
    reply_target_entity_id: null,
    compressed: false,
    ...input,
  };
}

function streamAppend(sessionId: string, ts = Date.now()): LiveFrame {
  return {
    type: "stream:append",
    ts,
    entries: [streamEntry({ session_id: sessionId })],
  };
}

function CacheProbe() {
  const cache = useLiveCache();

  return (
    <>
      <output data-testid="turns">{cache.counts?.turns ?? "none"}</output>
      <output data-testid="session-count">
        {cache.sessionsApi.data?.sessions.length ?? "none"}
      </output>
      <output data-testid="default-activity">
        {cache.sessionActivity.get("default") ?? "none"}
      </output>
      <output data-testid="other-activity">{cache.sessionActivity.get("other") ?? "none"}</output>
      <output data-testid="tick">
        {cache.lastMaintenanceTick === null
          ? "none"
          : `${cache.lastMaintenanceTick.cadence}:${cache.lastMaintenanceTick.changes}`}
      </output>
      <output data-testid="dream">
        {cache.dreamActivity === null
          ? "none"
          : `${cache.dreamActivity.process}:${cache.dreamActivity.phase}:${
              cache.dreamActivity.run_id ?? "null"
            }`}
      </output>
      <output data-testid="ws">
        {cache.wsState}:{cache.connectionCount}
      </output>
    </>
  );
}

function CacheHarness({
  live,
  connectionCount = 1,
  wsState = "live",
  sessionId = "default",
}: {
  live: ReturnType<typeof makeLiveSource>;
  connectionCount?: number;
  wsState?: WsState;
  sessionId?: string;
}) {
  return (
    <LiveEventsProvider value={live.live(connectionCount, wsState)}>
      <LiveCacheProvider sessionId={sessionId}>
        <CacheProbe />
      </LiveCacheProvider>
    </LiveEventsProvider>
  );
}

afterEach(() => {
  Object.defineProperty(window, "location", {
    configurable: true,
    value: realLocation,
  });
  vi.useRealTimers();
  vi.restoreAllMocks();
  vi.unstubAllGlobals();
});

describe("useLiveCache", () => {
  it("stores maintenance ticks and debounces stream append state/session refetches", async () => {
    const live = makeLiveSource();
    const fetchMock = vi.fn((request: RequestInfo | URL) => {
      const path = requestPath(request);
      if (path === "/api/state") {
        return Promise.resolve(jsonResponse(stateSnapshot()));
      }
      if (path === "/api/sessions") {
        return Promise.resolve(jsonResponse({ sessions: [session({ session_id: "default" })] }));
      }
      return Promise.reject(new Error(`unexpected fetch ${path}`));
    });
    vi.stubGlobal("fetch", fetchMock);

    render(<CacheHarness live={live} />);

    await waitFor(() => expect(screen.getByTestId("turns")).toHaveTextContent("1"));
    expect(callsFor(fetchMock, "/api/state")).toBe(1);
    expect(callsFor(fetchMock, "/api/sessions")).toBe(1);

    vi.useFakeTimers();
    vi.setSystemTime(10_000);

    act(() => {
      live.emit({
        type: "maintenance:tick",
        ts: 10_000,
        cadence: "light",
        status: "ok",
        processes: ["belief-reviser"],
        changed: true,
        changes: 2,
        errors: 0,
      });
    });

    expect(screen.getByTestId("tick")).toHaveTextContent("light:2");
    expect(callsFor(fetchMock, "/api/state")).toBe(2);
    expect(callsFor(fetchMock, "/api/sessions")).toBe(1);

    act(() => {
      live.emit(streamAppend("default", 10_001));
      live.emit(streamAppend("default", 10_002));
    });

    expect(callsFor(fetchMock, "/api/state")).toBe(2);
    expect(callsFor(fetchMock, "/api/sessions")).toBe(1);

    act(() => {
      vi.advanceTimersByTime(249);
    });
    expect(callsFor(fetchMock, "/api/state")).toBe(2);
    expect(callsFor(fetchMock, "/api/sessions")).toBe(1);

    act(() => {
      vi.advanceTimersByTime(1);
    });
    expect(callsFor(fetchMock, "/api/state")).toBe(3);
    expect(callsFor(fetchMock, "/api/sessions")).toBe(2);
  });

  it("tracks dream activity and ignores out-of-order completions", async () => {
    const live = makeLiveSource();
    const fetchMock = vi.fn((request: RequestInfo | URL) => {
      const path = requestPath(request);
      if (path === "/api/state") {
        return Promise.resolve(jsonResponse(stateSnapshot()));
      }
      if (path === "/api/sessions") {
        return Promise.resolve(jsonResponse({ sessions: [session({ session_id: "default" })] }));
      }
      return Promise.reject(new Error(`unexpected fetch ${path}`));
    });
    vi.stubGlobal("fetch", fetchMock);

    render(<CacheHarness live={live} />);
    await waitFor(() => expect(screen.getByTestId("turns")).toHaveTextContent("1"));

    act(() => {
      live.emit({
        type: "dream:process:completed",
        ts: 1,
        process: "belief-reviser",
        run_id: "run_old",
        phase: "plan",
        errors: 0,
        candidates_accepted: 0,
      });
    });
    expect(screen.getByTestId("dream")).toHaveTextContent("none");

    act(() => {
      live.emit({
        type: "dream:process:started",
        ts: 2,
        process: "belief-reviser",
        run_id: "run_1",
        phase: "plan",
      });
      live.emit({
        type: "dream:process:completed",
        ts: 3,
        process: "belief-reviser",
        run_id: "run_old",
        phase: "plan",
        errors: 0,
        candidates_accepted: 0,
      });
    });
    expect(screen.getByTestId("dream")).toHaveTextContent("belief-reviser:plan:run_1");

    act(() => {
      live.emit({
        type: "dream:process:started",
        ts: 4,
        process: "belief-reviser",
        run_id: "run_2",
        phase: "apply",
      });
      live.emit({
        type: "dream:process:completed",
        ts: 5,
        process: "belief-reviser",
        run_id: "run_1",
        phase: "plan",
        errors: 0,
        candidates_accepted: 0,
      });
    });
    expect(screen.getByTestId("dream")).toHaveTextContent("belief-reviser:apply:run_2");

    act(() => {
      live.emit({
        type: "dream:process:completed",
        ts: 6,
        process: "belief-reviser",
        run_id: "run_2",
        phase: "apply",
        errors: 0,
        candidates_accepted: 0,
      });
    });
    expect(screen.getByTestId("dream")).toHaveTextContent("none");
  });

  it("seeds session recency from last_activity_at and bumps the active session immediately", async () => {
    const live = makeLiveSource();
    let sessions = [
      session({ session_id: "default", last_activity_at: 1_000 }),
      session({ session_id: "other", last_activity_at: 2_000 }),
    ];
    const fetchMock = vi.fn((request: RequestInfo | URL) => {
      const path = requestPath(request);
      if (path === "/api/state") {
        return Promise.resolve(jsonResponse(stateSnapshot()));
      }
      if (path === "/api/sessions") {
        return Promise.resolve(jsonResponse({ sessions }));
      }
      return Promise.reject(new Error(`unexpected fetch ${path}`));
    });
    vi.stubGlobal("fetch", fetchMock);

    render(<CacheHarness live={live} />);

    await waitFor(() => expect(screen.getByTestId("default-activity")).toHaveTextContent("1000"));
    expect(screen.getByTestId("other-activity")).toHaveTextContent("2000");

    vi.useFakeTimers();
    vi.setSystemTime(20_000);
    sessions = [
      session({ session_id: "default", last_activity_at: 1_000 }),
      session({ session_id: "other", last_activity_at: 12_000 }),
    ];

    act(() => {
      live.emit(streamAppend("default", 19_000));
    });

    expect(screen.getByTestId("default-activity")).toHaveTextContent("20000");
    expect(screen.getByTestId("other-activity")).toHaveTextContent("2000");

    await act(async () => {
      await vi.advanceTimersByTimeAsync(250);
    });

    expect(screen.getByTestId("default-activity")).toHaveTextContent("20000");
    expect(screen.getByTestId("other-activity")).toHaveTextContent("12000");
  });

  it("reloads the page on borg reset", async () => {
    const live = makeLiveSource();
    const reload = vi.fn();
    Object.defineProperty(window, "location", {
      configurable: true,
      value: { ...window.location, reload },
    });
    const fetchMock = vi.fn((request: RequestInfo | URL) => {
      const path = requestPath(request);
      if (path === "/api/state") {
        return Promise.resolve(jsonResponse(stateSnapshot()));
      }
      if (path === "/api/sessions") {
        return Promise.resolve(jsonResponse({ sessions: [session({ session_id: "default" })] }));
      }
      return Promise.reject(new Error(`unexpected fetch ${path}`));
    });
    vi.stubGlobal("fetch", fetchMock);

    render(<CacheHarness live={live} />);
    await waitFor(() => expect(screen.getByTestId("turns")).toHaveTextContent("1"));

    act(() => {
      live.emit({ type: "borg:reset", ts: Date.now() });
    });

    expect(reload).toHaveBeenCalledTimes(1);
  });

  it("refetches state and sessions when the live connection count increases", async () => {
    const live = makeLiveSource();
    const fetchMock = vi.fn((request: RequestInfo | URL) => {
      const path = requestPath(request);
      if (path === "/api/state") {
        return Promise.resolve(jsonResponse(stateSnapshot()));
      }
      if (path === "/api/sessions") {
        return Promise.resolve(jsonResponse({ sessions: [session({ session_id: "default" })] }));
      }
      return Promise.reject(new Error(`unexpected fetch ${path}`));
    });
    vi.stubGlobal("fetch", fetchMock);

    const { rerender } = render(<CacheHarness live={live} connectionCount={1} />);

    await waitFor(() => expect(screen.getByTestId("ws")).toHaveTextContent("live:1"));
    expect(callsFor(fetchMock, "/api/state")).toBe(1);
    expect(callsFor(fetchMock, "/api/sessions")).toBe(1);

    rerender(<CacheHarness live={live} connectionCount={2} wsState="live" />);

    await waitFor(() => expect(screen.getByTestId("ws")).toHaveTextContent("live:2"));
    await waitFor(() => expect(callsFor(fetchMock, "/api/state")).toBe(2));
    expect(callsFor(fetchMock, "/api/sessions")).toBe(2);
  });
});
