import { fireEvent, render, screen, waitFor } from "@testing-library/react";
import { useState } from "react";
import { afterEach, describe, expect, it, vi } from "vitest";

import { RESET_CONFIRM_TOKEN } from "../../api/client";
import type { MemoryBandDetail, SessionRecord, StateSnapshot } from "../../api/types";
import { LiveEventsProvider } from "../../hooks/live-context";
import { LiveCacheProvider } from "../../hooks/use-live-cache";
import { usePaletteHotkey } from "../../hooks/use-palette-hotkey";
import type { RouteId } from "../../routes";
import { Inspector } from "../Inspector/Inspector";
import { InspectorProvider, useInspector } from "../Inspector/inspector-context";
import { ResetButton } from "../ResetButton";
import { CommandPalette } from "./CommandPalette";

const realLocation = window.location;
const realScrollIntoView = window.HTMLElement.prototype.scrollIntoView;

function jsonResponse(body: unknown, status = 200): Response {
  return new Response(JSON.stringify(body), {
    status,
    headers: { "Content-Type": "application/json" },
  });
}

function requestUrl(request: RequestInfo | URL): URL {
  return new URL(String(request), "http://test.invalid");
}

function callsForPath(fetchMock: ReturnType<typeof vi.fn>, path: string): number {
  return fetchMock.mock.calls.filter(
    ([request]) => requestUrl(request as RequestInfo | URL).pathname === path,
  ).length;
}

function stateSnapshot(input: Partial<StateSnapshot> = {}): StateSnapshot {
  return {
    active_session: "default",
    audiences: ["alice"],
    counts: {
      turns: 0,
      commitments: 0,
      open_qs: 0,
      open_reviews: 0,
      dream_audit_rows: 0,
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

function session(
  input: Partial<SessionRecord> & Pick<SessionRecord, "session_id" | "label">,
): SessionRecord {
  return {
    source_type: "demo",
    source_external_id: null,
    source_url: null,
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

function emptyMemoryDetail(band: "episodic" | "semantic" | "procedural"): MemoryBandDetail {
  if (band === "episodic") {
    return { band, mode: "search", query: "", items: [], next_cursor: null };
  }
  if (band === "semantic") {
    return { band, mode: "search", query: "", nodes: [], edges: [], next_cursor: null };
  }
  return { band, mode: "search", query: "", items: [], next_cursor: null };
}

function setupFetch({
  sessions = [session({ session_id: "default", label: "demo (default)" })],
  memory = {},
}: {
  sessions?: SessionRecord[];
  memory?: Partial<Record<"episodic" | "semantic" | "procedural", MemoryBandDetail>>;
} = {}) {
  const fetchMock = vi.fn((request: RequestInfo | URL, init?: RequestInit) => {
    const url = requestUrl(request);

    if (url.pathname === "/api/state") {
      return Promise.resolve(jsonResponse(stateSnapshot()));
    }
    if (url.pathname === "/api/sessions" && init?.method !== "POST") {
      return Promise.resolve(jsonResponse({ sessions }));
    }
    if (url.pathname === "/api/memory/bands/episodic") {
      return Promise.resolve(jsonResponse(memory.episodic ?? emptyMemoryDetail("episodic")));
    }
    if (url.pathname === "/api/memory/bands/semantic") {
      return Promise.resolve(jsonResponse(memory.semantic ?? emptyMemoryDetail("semantic")));
    }
    if (url.pathname === "/api/memory/bands/procedural") {
      return Promise.resolve(jsonResponse(memory.procedural ?? emptyMemoryDetail("procedural")));
    }
    if (url.pathname === "/api/admin/reset" && init?.method === "POST") {
      return Promise.resolve(jsonResponse({ ok: true }));
    }

    return Promise.reject(new Error(`unexpected fetch ${url.pathname}`));
  });
  vi.stubGlobal("fetch", fetchMock);
  return fetchMock;
}

function liveEvents() {
  return {
    wsState: "live" as const,
    connectionCount: 1,
    subscribe: () => () => undefined,
  };
}

function InspectorTargetProbe() {
  const inspector = useInspector();
  return (
    <>
      <button
        type="button"
        onClick={() => inspector.openObject({ type: "session", id: "default" })}
      >
        open inspector
      </button>
      <output data-testid="inspector-target">
        {inspector.target === null ? "none" : `${inspector.target.type}:${inspector.target.id}`}
      </output>
    </>
  );
}

function PaletteHarness({
  setView = vi.fn(),
  setSessionId = vi.fn(),
}: {
  setView?: (view: RouteId) => void;
  setSessionId?: (sessionId: string) => void;
}) {
  const palette = usePaletteHotkey();
  const [resetOpen, setResetOpen] = useState(false);

  return (
    <LiveEventsProvider value={liveEvents()}>
      <LiveCacheProvider sessionId="default">
        <InspectorProvider
          setView={setView}
          setSessionId={setSessionId}
          sessionId="default"
          audience="alice"
        >
          <CommandPalette
            open={palette.open}
            onOpenChange={palette.setOpen}
            setView={setView}
            setSessionId={setSessionId}
            onOpenReset={() => setResetOpen(true)}
          />
          <ResetButton open={resetOpen} onOpenChange={setResetOpen} showTrigger={false} />
          <InspectorTargetProbe />
          <Inspector />
        </InspectorProvider>
      </LiveCacheProvider>
    </LiveEventsProvider>
  );
}

function renderPalette(
  props: {
    setView?: (view: RouteId) => void;
    setSessionId?: (sessionId: string) => void;
  } = {},
) {
  return render(<PaletteHarness {...props} />);
}

async function openWithMeta(): Promise<HTMLInputElement> {
  fireEvent.keyDown(window, { key: "k", metaKey: true });
  return screen.findByRole("searchbox", { name: "Command palette search" });
}

afterEach(() => {
  Object.defineProperty(window, "location", {
    configurable: true,
    value: realLocation,
  });
  if (realScrollIntoView === undefined) {
    Reflect.deleteProperty(window.HTMLElement.prototype, "scrollIntoView");
  } else {
    Object.defineProperty(window.HTMLElement.prototype, "scrollIntoView", {
      configurable: true,
      value: realScrollIntoView,
    });
  }
  vi.unstubAllGlobals();
  vi.restoreAllMocks();
});

describe("CommandPalette", () => {
  it("opens on Command-K and Control-K and autofocuses the search field", async () => {
    setupFetch();
    renderPalette();

    const metaInput = await openWithMeta();
    expect(metaInput).toHaveFocus();

    fireEvent.keyDown(metaInput, { key: "Escape" });
    await waitFor(() => {
      expect(
        screen.queryByRole("searchbox", { name: "Command palette search" }),
      ).not.toBeInTheDocument();
    });

    fireEvent.keyDown(window, { key: "k", ctrlKey: true });
    const ctrlInput = await screen.findByRole("searchbox", { name: "Command palette search" });
    expect(ctrlInput).toHaveFocus();
  });

  it("toggles closed on a second Command-K press", async () => {
    setupFetch();
    renderPalette();

    await openWithMeta();
    fireEvent.keyDown(window, { key: "k", metaKey: true });

    await waitFor(() => {
      expect(
        screen.queryByRole("searchbox", { name: "Command palette search" }),
      ).not.toBeInTheDocument();
    });
  });

  it("closes on Escape and backdrop mouse down", async () => {
    setupFetch();
    const { container } = renderPalette();

    const input = await openWithMeta();
    fireEvent.keyDown(input, { key: "Escape" });
    await waitFor(() => {
      expect(screen.queryByRole("dialog", { name: "command palette" })).not.toBeInTheDocument();
    });

    await openWithMeta();
    const backdrop = container.querySelector(".cmdp-backdrop");
    expect(backdrop).not.toBeNull();
    fireEvent.mouseDown(backdrop!);

    await waitFor(() => {
      expect(screen.queryByRole("dialog", { name: "command palette" })).not.toBeInTheDocument();
    });
  });

  it("keeps Inspector open when Escape closes the palette, then lets the next Escape close Inspector", async () => {
    setupFetch();
    renderPalette();

    fireEvent.click(screen.getByRole("button", { name: "open inspector" }));
    expect(screen.getByTestId("inspector-target")).toHaveTextContent("session:default");
    expect(await screen.findByRole("dialog", { name: "Session inspector" })).toBeInTheDocument();

    const input = await openWithMeta();
    fireEvent.keyDown(input, { key: "Escape" });

    await waitFor(() => {
      expect(screen.queryByRole("dialog", { name: "command palette" })).not.toBeInTheDocument();
    });
    expect(screen.getByTestId("inspector-target")).toHaveTextContent("session:default");
    expect(screen.getByRole("dialog", { name: "Session inspector" })).toBeInTheDocument();

    fireEvent.keyDown(window, { key: "Escape" });

    await waitFor(() => {
      expect(screen.getByTestId("inspector-target")).toHaveTextContent("none");
    });
  });

  it("moves the active row with arrows and runs the selected screen command with Enter", async () => {
    setupFetch();
    const setView = vi.fn();
    renderPalette({ setView });

    const input = await openWithMeta();
    fireEvent.keyDown(input, { key: "ArrowDown" });
    fireEvent.keyDown(input, { key: "Enter" });

    expect(setView).toHaveBeenCalledWith("cognition");
    await waitFor(() => {
      expect(screen.queryByRole("dialog", { name: "command palette" })).not.toBeInTheDocument();
    });
  });

  it("keeps the active row scrolled into view during arrow navigation", async () => {
    setupFetch();
    const scrollIntoView = vi.fn();
    Object.defineProperty(window.HTMLElement.prototype, "scrollIntoView", {
      configurable: true,
      value: scrollIntoView,
    });
    renderPalette();

    const input = await openWithMeta();
    fireEvent.keyDown(input, { key: "ArrowDown" });

    expect(scrollIntoView).toHaveBeenCalledWith({ block: "nearest" });
  });

  it("picks up the admin screen command from rail items", async () => {
    setupFetch();
    const setView = vi.fn();
    renderPalette({ setView });

    const input = await openWithMeta();
    fireEvent.change(input, { target: { value: "admin" } });
    fireEvent.click(await screen.findByText("Go to admin"));

    expect(setView).toHaveBeenCalledWith("admin");
  });

  it("switches sessions from live cache rows", async () => {
    const targetSession = session({
      session_id: "sess_research000000",
      label: "research chat",
      audience_label: "bob",
    });
    setupFetch({
      sessions: [session({ session_id: "default", label: "demo (default)" }), targetSession],
    });
    const setSessionId = vi.fn();
    renderPalette({ setSessionId });

    await openWithMeta();
    const row = await screen.findByText("Switch session: research chat");
    fireEvent.click(row);

    expect(setSessionId).toHaveBeenCalledWith(targetSession.session_id);
  });

  it("opens a prefixed object id through the inspector", async () => {
    setupFetch();
    renderPalette();

    const input = await openWithMeta();
    fireEvent.change(input, { target: { value: "ep_abc123456789" } });

    expect(await screen.findByText(/Open Episode/)).toBeInTheDocument();
    fireEvent.keyDown(input, { key: "Enter" });

    expect(screen.getByTestId("inspector-target")).toHaveTextContent("episode:ep_abc123456789");
  });

  it("shows an honest degraded hint for non-prefixed ids without guessing an object type", async () => {
    const fetchMock = setupFetch();
    renderPalette();

    const input = await openWithMeta();
    fireEvent.change(input, { target: { value: "42" } });

    expect(await screen.findByText("Object ID not resolvable")).toBeInTheDocument();
    fireEvent.keyDown(input, { key: "Enter" });

    expect(screen.getByTestId("inspector-target")).toHaveTextContent("none");
    expect(callsForPath(fetchMock, "/api/memory/bands/episodic")).toBe(0);
  });

  it("uses server memory search results and opens the active memory object", async () => {
    const episodeId = "ep_meaning0000001";
    const fetchMock = setupFetch({
      memory: {
        episodic: {
          band: "episodic",
          mode: "search",
          query: "meaning",
          items: [
            {
              id: episodeId,
              title: "Meaning episode",
              narrative: "Returned by server search.",
              participants: [],
              location: null,
              start_time: 1_000,
              end_time: 2_000,
              audience: "alice",
              significance: 0.8,
              confidence: 0.9,
              tags: [],
              source_stream_ids: ["strm_source000001"],
              source_count: 1,
              lineage: { derived_from: [], supersedes: [] },
              emotional_arc: null,
              vector_dims: 4,
              created_at: 1_000,
              updated_at: 2_000,
            },
          ],
          next_cursor: null,
        },
      },
    });
    renderPalette();

    const input = await openWithMeta();
    fireEvent.change(input, { target: { value: "meaning" } });

    expect(await screen.findByText("Open Meaning episode")).toBeInTheDocument();
    fireEvent.keyDown(input, { key: "Enter" });

    expect(screen.getByTestId("inspector-target")).toHaveTextContent(`episode:${episodeId}`);
    expect(callsForPath(fetchMock, "/api/memory/bands/episodic")).toBeGreaterThanOrEqual(1);
    expect(callsForPath(fetchMock, "/api/memory/bands/semantic")).toBeGreaterThanOrEqual(1);
    expect(callsForPath(fetchMock, "/api/memory/bands/procedural")).toBeGreaterThanOrEqual(1);
  });

  it("routes Reset demo through the RESET confirmation before posting", async () => {
    const fetchMock = setupFetch();
    const reload = vi.fn();
    Object.defineProperty(window, "location", {
      configurable: true,
      value: { ...window.location, reload },
    });
    renderPalette();

    const input = await openWithMeta();
    fireEvent.change(input, { target: { value: "reset" } });
    fireEvent.click(await screen.findByText("Reset demo"));

    expect(await screen.findByPlaceholderText(RESET_CONFIRM_TOKEN)).toBeInTheDocument();
    expect(callsForPath(fetchMock, "/api/admin/reset")).toBe(0);

    fireEvent.change(screen.getByPlaceholderText(RESET_CONFIRM_TOKEN), {
      target: { value: RESET_CONFIRM_TOKEN },
    });
    fireEvent.click(screen.getByRole("button", { name: "reset borg" }));

    await waitFor(() => expect(callsForPath(fetchMock, "/api/admin/reset")).toBe(1));
    await waitFor(() => expect(reload).toHaveBeenCalled());
  });

  it("navigates action commands to their existing flows", async () => {
    setupFetch();
    const setView = vi.fn();
    renderPalette({ setView });

    await openWithMeta();
    fireEvent.click(screen.getByText("Create commitment"));
    expect(setView).toHaveBeenCalledWith("governance", { governanceTab: "commitments" });

    await openWithMeta();
    fireEvent.click(screen.getByText("Run dream plan"));
    expect(setView).toHaveBeenCalledWith("dream");

    await openWithMeta();
    fireEvent.click(screen.getByText("Open assembled prompt"));
    expect(setView).toHaveBeenCalledWith("prompts");
  });
});
