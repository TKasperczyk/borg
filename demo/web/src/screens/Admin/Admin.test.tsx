import { act, fireEvent, render, screen, waitFor } from "@testing-library/react";
import { useState } from "react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import { RESET_CONFIRM_TOKEN } from "../../api/client";
import type { LiveFrame, SessionRecord, StateSnapshot, WsState } from "../../api/types";
import { AppErrorBoundary } from "../../components/AppErrorBoundary";
import { CommandPalette } from "../../components/CommandPalette/CommandPalette";
import { InspectorProvider } from "../../components/Inspector/inspector-context";
import { ResetButton } from "../../components/ResetButton";
import { LiveEventsProvider } from "../../hooks/live-context";
import { LiveCacheProvider, useLiveCache } from "../../hooks/use-live-cache";
import type { LiveEventHandler, LiveEvents } from "../../hooks/use-live-events";
import { clearClientErrors, recordClientError } from "../../lib/client-error-log";
import { AdminScreen, type AdminRefetchResult } from "./index";

const realLocation = window.location;

function jsonResponse(body: unknown, status = 200): Response {
  return new Response(JSON.stringify(body), {
    status,
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

function stateSnapshot(input: Partial<StateSnapshot> = {}): StateSnapshot {
  return {
    active_session: "default",
    audiences: ["alice"],
    counts: {
      turns: 4,
      commitments: 3,
      open_qs: 2,
      open_reviews: 1,
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
    version: "demo-v16",
    ...input,
  };
}

function session(input: Partial<SessionRecord> = {}): SessionRecord {
  return {
    session_id: "default",
    source_type: "demo",
    source_external_id: null,
    source_url: null,
    label: "demo (default)",
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

function setupFetch(input: { stateStatus?: number; stateMessage?: string } = {}) {
  const fetchMock = vi.fn((request: RequestInfo | URL, init?: RequestInit) => {
    const path = requestPath(request);

    if (path === "/api/state") {
      if (input.stateStatus !== undefined && input.stateStatus >= 400) {
        return Promise.resolve(
          jsonResponse(
            { error: { message: input.stateMessage ?? "state failed" } },
            input.stateStatus,
          ),
        );
      }
      return Promise.resolve(jsonResponse(stateSnapshot()));
    }

    if (path === "/api/sessions" && init?.method !== "POST") {
      return Promise.resolve(jsonResponse({ sessions: [session()] }));
    }

    if (path === "/api/admin/reset" && init?.method === "POST") {
      return Promise.resolve(jsonResponse({ ok: true }));
    }

    if (path === "/api/memory/bands/episodic") {
      return Promise.resolve(
        jsonResponse({ band: "episodic", mode: "search", query: "", items: [], next_cursor: null }),
      );
    }

    if (path === "/api/memory/bands/semantic") {
      return Promise.resolve(
        jsonResponse({
          band: "semantic",
          mode: "search",
          query: "",
          nodes: [],
          edges: [],
          next_cursor: null,
        }),
      );
    }

    if (path === "/api/memory/bands/procedural") {
      return Promise.resolve(
        jsonResponse({
          band: "procedural",
          mode: "search",
          query: "",
          items: [],
          next_cursor: null,
        }),
      );
    }

    return Promise.reject(new Error(`unexpected fetch ${path}`));
  });
  vi.stubGlobal("fetch", fetchMock);
  return fetchMock;
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

function AdminAdapter({
  resetCaches,
  onOpenResetConfirm,
}: {
  resetCaches: () => boolean;
  onOpenResetConfirm: () => void;
}) {
  const cache = useLiveCache();

  const refetchAll = async (): Promise<AdminRefetchResult> => {
    await Promise.all([cache.stateApi.refetch(), cache.sessionsApi.refetch()]);
    return { turnCachesReset: resetCaches() };
  };

  return (
    <AdminScreen
      route="admin"
      sessionId="default"
      onRefetchAll={refetchAll}
      onOpenResetConfirm={onOpenResetConfirm}
    />
  );
}

function AdminHarness({
  live,
  wsState = "live",
  connectionCount = 1,
  resetCaches = () => true,
}: {
  live: ReturnType<typeof makeLiveSource>;
  wsState?: WsState;
  connectionCount?: number;
  resetCaches?: () => boolean;
}) {
  const [resetOpen, setResetOpen] = useState(false);

  return (
    <LiveEventsProvider value={live.live(connectionCount, wsState)}>
      <LiveCacheProvider sessionId="default">
        <AdminAdapter resetCaches={resetCaches} onOpenResetConfirm={() => setResetOpen(true)} />
        <ResetButton open={resetOpen} onOpenChange={setResetOpen} showTrigger={false} />
      </LiveCacheProvider>
    </LiveEventsProvider>
  );
}

function AdminPaletteHarness({ live }: { live: ReturnType<typeof makeLiveSource> }) {
  const [resetOpen, setResetOpen] = useState(false);
  const [paletteOpen, setPaletteOpen] = useState(false);

  return (
    <LiveEventsProvider value={live.live()}>
      <LiveCacheProvider sessionId="default">
        <InspectorProvider
          setView={() => undefined}
          setSessionId={() => undefined}
          sessionId="default"
          audience="alice"
        >
          <AdminAdapter resetCaches={() => true} onOpenResetConfirm={() => setResetOpen(true)} />
          <button type="button" onClick={() => setPaletteOpen(true)}>
            open palette
          </button>
          <CommandPalette
            open={paletteOpen}
            onOpenChange={setPaletteOpen}
            setView={() => undefined}
            setSessionId={() => undefined}
            onOpenReset={() => setResetOpen(true)}
          />
          <ResetButton open={resetOpen} onOpenChange={setResetOpen} showTrigger={false} />
        </InspectorProvider>
      </LiveCacheProvider>
    </LiveEventsProvider>
  );
}

function resetDialogCount(): number {
  return screen
    .getAllByRole("dialog")
    .filter((dialog) => dialog.textContent?.includes("reset borg to clean slate")).length;
}

function Crash() {
  throw new Error("render failed");
  return null;
}

beforeEach(() => {
  clearClientErrors();
});

afterEach(() => {
  Object.defineProperty(window, "location", {
    configurable: true,
    value: realLocation,
  });
  vi.restoreAllMocks();
  vi.unstubAllGlobals();
});

describe("AdminScreen", () => {
  it("renders runtime fields, live cache activity, and route diagnostics", async () => {
    setupFetch();
    const live = makeLiveSource();

    render(<AdminHarness live={live} wsState="down" connectionCount={2} />);

    expect(await screen.findAllByText("demo-v16")).toHaveLength(2);
    expect(screen.getByText("same-origin")).toBeInTheDocument();
    expect(screen.getByTestId("ws-state")).toHaveTextContent("down");
    expect(screen.getByText("2 connections")).toBeInTheDocument();
    expect(screen.getByText("down after 5 failed reconnect attempts")).toBeInTheDocument();
    expect(screen.getByText("none this session")).toBeInTheDocument();
    expect(screen.getByText("none")).toBeInTheDocument();
    expect(screen.getByText("alt+9")).toBeInTheDocument();
    expect(screen.getByText("commit")).toBeInTheDocument();
    expect(screen.getByText("directives")).toBeInTheDocument();

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
      live.emit({
        type: "dream:process:started",
        ts: 10_100,
        process: "belief-reviser",
        run_id: "run_1",
        phase: "plan",
      });
    });

    expect(screen.getByText("light 1p 2chg")).toBeInTheDocument();
    expect(screen.getByText("session-only")).toBeInTheDocument();
    expect(screen.getByText("belief-reviser · plan · run_1")).toBeInTheDocument();
  });

  it("records API failures and boundary failures, then clears the client log", async () => {
    setupFetch({ stateStatus: 500, stateMessage: "state failed" });
    const live = makeLiveSource();

    render(<AdminHarness live={live} />);

    expect(await screen.findByText(/state snapshot unavailable: state failed/)).toBeInTheDocument();
    expect(await screen.findByTestId("client-error-row")).toHaveTextContent("500");
    expect(screen.getByTestId("client-error-row")).toHaveTextContent("/api/state");

    const consoleError = vi.spyOn(console, "error").mockImplementation(() => undefined);
    const preventExpectedErrorReport = (event: ErrorEvent) => event.preventDefault();
    window.addEventListener("error", preventExpectedErrorReport);

    render(
      <AppErrorBoundary
        onError={(error) =>
          recordClientError({
            source: "boundary",
            boundarySource: "test boundary",
            message: error.message,
          })
        }
      >
        <Crash />
      </AppErrorBoundary>,
    );

    await waitFor(() => expect(screen.getAllByTestId("client-error-row")).toHaveLength(2));
    expect(screen.getByText("test boundary")).toBeInTheDocument();
    expect(screen.getAllByText("render failed").length).toBeGreaterThan(0);

    fireEvent.click(screen.getByRole("button", { name: "clear log" }));

    expect(screen.queryByTestId("client-error-row")).not.toBeInTheDocument();
    expect(screen.getByText("no client-captured errors this session")).toBeInTheDocument();

    window.removeEventListener("error", preventExpectedErrorReport);
    consoleError.mockRestore();
  });

  it("keeps danger-zone reset behind the RESET token", async () => {
    const fetchMock = setupFetch();
    const reload = vi.fn();
    Object.defineProperty(window, "location", {
      configurable: true,
      value: { ...window.location, reload },
    });
    const live = makeLiveSource();

    render(<AdminHarness live={live} />);
    await screen.findAllByText("demo-v16");

    fireEvent.click(screen.getByRole("button", { name: "reset" }));
    expect(screen.getByRole("button", { name: "reset borg" })).toBeDisabled();
    expect(callsFor(fetchMock, "/api/admin/reset")).toBe(0);

    fireEvent.change(screen.getByPlaceholderText(RESET_CONFIRM_TOKEN), {
      target: { value: RESET_CONFIRM_TOKEN },
    });
    fireEvent.click(screen.getByRole("button", { name: "reset borg" }));

    await waitFor(() => expect(callsFor(fetchMock, "/api/admin/reset")).toBe(1));
    await waitFor(() => expect(reload).toHaveBeenCalled());
  });

  it("keeps one reset dialog when Admin and palette reset entry points both fire", async () => {
    setupFetch();
    const live = makeLiveSource();

    render(<AdminPaletteHarness live={live} />);
    await screen.findAllByText("demo-v16");

    fireEvent.click(screen.getByRole("button", { name: "reset" }));
    expect(resetDialogCount()).toBe(1);

    fireEvent.click(screen.getByRole("button", { name: "open palette" }));
    const input = await screen.findByRole("searchbox", { name: "Command palette search" });
    fireEvent.change(input, { target: { value: "reset" } });
    fireEvent.click(await screen.findByText("Reset demo"));

    expect(resetDialogCount()).toBe(1);
  });

  it("refetches live cache state and sessions through refetch all", async () => {
    const fetchMock = setupFetch();
    const live = makeLiveSource();
    const resetCaches = vi.fn(() => true);

    render(<AdminHarness live={live} resetCaches={resetCaches} />);

    await screen.findAllByText("demo-v16");
    expect(callsFor(fetchMock, "/api/state")).toBe(1);
    expect(callsFor(fetchMock, "/api/sessions")).toBe(1);

    fireEvent.click(screen.getByRole("button", { name: "refetch all" }));

    await waitFor(() => expect(callsFor(fetchMock, "/api/state")).toBe(2));
    expect(callsFor(fetchMock, "/api/sessions")).toBe(2);
    expect(resetCaches).toHaveBeenCalledTimes(1);
    expect(screen.getByText("client caches refetched")).toBeInTheDocument();
  });
});
