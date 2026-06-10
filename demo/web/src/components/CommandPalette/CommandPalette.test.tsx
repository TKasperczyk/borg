import { act, fireEvent, render, screen, waitFor } from "@testing-library/react";
import { useState } from "react";
import { afterEach, describe, expect, it, vi } from "vitest";

import { RESET_CONFIRM_TOKEN } from "../../api/client";
import type {
  CommitmentItem,
  MaintenanceAuditRow,
  MemoryBandDetail,
  ReviewRow,
  SessionRecord,
  StateSnapshot,
} from "../../api/types";
import { LiveEventsProvider } from "../../hooks/live-context";
import { LiveCacheProvider } from "../../hooks/use-live-cache";
import { usePaletteHotkey } from "../../hooks/use-palette-hotkey";
import type { RouteId } from "../../routes";
import { Inspector } from "../Inspector/Inspector";
import { InspectorProvider, useInspector } from "../Inspector/inspector-context";
import { ResetButton } from "../ResetButton";
import { ShortcutLegend } from "../ShortcutLegend";
import { shortId } from "../../screens/screen-utils";
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
  memoryErrors = {},
  commitments = [],
  reviews = [],
  dreamRows = [],
}: {
  sessions?: SessionRecord[];
  memory?: Partial<Record<"episodic" | "semantic" | "procedural", MemoryBandDetail>>;
  memoryErrors?: Partial<Record<"episodic" | "semantic" | "procedural", string>>;
  commitments?: CommitmentItem[];
  reviews?: ReviewRow[];
  dreamRows?: MaintenanceAuditRow[];
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
      if (memoryErrors.episodic !== undefined) {
        return Promise.reject(new Error(memoryErrors.episodic));
      }
      return Promise.resolve(jsonResponse(memory.episodic ?? emptyMemoryDetail("episodic")));
    }
    if (url.pathname === "/api/memory/bands/semantic") {
      if (memoryErrors.semantic !== undefined) {
        return Promise.reject(new Error(memoryErrors.semantic));
      }
      return Promise.resolve(jsonResponse(memory.semantic ?? emptyMemoryDetail("semantic")));
    }
    if (url.pathname === "/api/memory/bands/procedural") {
      if (memoryErrors.procedural !== undefined) {
        return Promise.reject(new Error(memoryErrors.procedural));
      }
      return Promise.resolve(jsonResponse(memory.procedural ?? emptyMemoryDetail("procedural")));
    }
    if (url.pathname === "/api/commitments") {
      return Promise.resolve(jsonResponse({ commitments }));
    }
    if (url.pathname === "/api/reviews") {
      return Promise.resolve(jsonResponse({ rows: reviews }));
    }
    if (url.pathname === "/api/dream/audit") {
      return Promise.resolve(jsonResponse({ rows: dreamRows }));
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
  const [legendOpen, setLegendOpen] = useState(false);
  const palette = usePaletteHotkey({
    onRouteChord: setView,
    onHelpChord: () => setLegendOpen(true),
  });
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
          <input aria-label="editable target" />
          <ShortcutLegend open={legendOpen} onClose={() => setLegendOpen(false)} />
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
  const searchbox = screen.queryByRole("searchbox", {
    name: "Command palette search",
  }) as HTMLInputElement | null;
  if (searchbox !== null) {
    return searchbox;
  }
  return screen.findByRole("searchbox", { name: "Command palette search" });
}

async function advancePaletteTimers(ms: number): Promise<void> {
  await act(async () => {
    vi.advanceTimersByTime(ms);
    await Promise.resolve();
    await Promise.resolve();
  });
}

async function flushMicrotasks(): Promise<void> {
  await act(async () => {
    await Promise.resolve();
    await Promise.resolve();
  });
}

afterEach(() => {
  vi.useRealTimers();
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

  it("routes implemented alt-digit chords only when the palette is closed", async () => {
    setupFetch();
    const setView = vi.fn();
    renderPalette({ setView });

    fireEvent.keyDown(window, { key: "x", code: "Digit3", altKey: true });
    expect(setView).toHaveBeenCalledWith("memory");

    fireEvent.keyDown(window, { key: "x", code: "Digit4", altKey: true, ctrlKey: true });
    expect(setView).not.toHaveBeenCalledWith("identity");

    await openWithMeta();
    expect(await screen.findByText("alt+3")).toBeInTheDocument();
    fireEvent.keyDown(window, { key: "x", code: "Digit9", altKey: true });
    expect(setView).not.toHaveBeenCalledWith("admin");
  });

  it("does not route alt-digit chords from editable targets", () => {
    setupFetch();
    const setView = vi.fn();
    renderPalette({ setView });

    fireEvent.keyDown(screen.getByLabelText("editable target"), {
      key: "x",
      code: "Digit3",
      altKey: true,
    });

    expect(setView).not.toHaveBeenCalled();
  });

  it("opens the shortcut legend on question mark and closes it with Escape", async () => {
    setupFetch();
    renderPalette();

    fireEvent.keyDown(window, { key: "?", code: "Slash", shiftKey: true });

    expect(await screen.findByRole("dialog", { name: "shortcuts" })).toBeInTheDocument();
    expect(screen.getByText("command palette")).toBeInTheDocument();
    expect(screen.getByText("ctrl+K")).toBeInTheDocument();

    fireEvent.keyDown(window, { key: "Escape" });

    await waitFor(() => {
      expect(screen.queryByRole("dialog", { name: "shortcuts" })).not.toBeInTheDocument();
    });
  });

  it("does not open the shortcut legend from editable targets", () => {
    setupFetch();
    renderPalette();

    fireEvent.keyDown(screen.getByLabelText("editable target"), {
      key: "?",
      code: "Slash",
      shiftKey: true,
    });

    expect(screen.queryByRole("dialog", { name: "shortcuts" })).not.toBeInTheDocument();
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
    vi.useFakeTimers();
    const episodeId = "ep_abc1234567890000";
    const fetchMock = setupFetch({
      memory: {
        episodic: {
          band: "episodic",
          mode: "browse",
          items: [
            {
              id: episodeId,
              title: "Exact episode",
              narrative: "Resolved through the registry list path.",
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
    fireEvent.change(input, { target: { value: episodeId } });

    expect(screen.getByText("Checking object id")).toBeInTheDocument();
    expect(callsForPath(fetchMock, "/api/memory/bands/episodic")).toBe(0);

    await advancePaletteTimers(200);

    expect(screen.getByText(/Open Episode/)).toBeInTheDocument();
    fireEvent.keyDown(input, { key: "Enter" });

    expect(screen.getByTestId("inspector-target")).toHaveTextContent(`episode:${episodeId}`);
  });

  it("flushes a pending full-id verification on immediate Enter", async () => {
    vi.useFakeTimers();
    const episodeId = "ep_enterflush000001";
    const fetchMock = setupFetch({
      memory: {
        episodic: {
          band: "episodic",
          mode: "browse",
          items: [
            {
              id: episodeId,
              title: "Enter flush episode",
              narrative: "Resolved by immediate Enter.",
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
    fireEvent.change(input, { target: { value: episodeId } });

    expect(screen.getByText("Checking object id")).toBeInTheDocument();
    expect(callsForPath(fetchMock, "/api/memory/bands/episodic")).toBe(0);

    fireEvent.keyDown(input, { key: "Enter" });
    await flushMicrotasks();

    const callsAfterEnter = callsForPath(fetchMock, "/api/memory/bands/episodic");
    expect(callsAfterEnter).toBeGreaterThanOrEqual(1);
    expect(screen.getByTestId("inspector-target")).toHaveTextContent(`episode:${episodeId}`);

    await advancePaletteTimers(200);
    expect(callsForPath(fetchMock, "/api/memory/bands/episodic")).toBe(callsAfterEnter);
  });

  it("resolves numeric review and dream rows from their list endpoints", async () => {
    vi.useFakeTimers();
    const review: ReviewRow = {
      id: 42,
      kind: "contradiction",
      refs: {},
      reason: "numeric review",
      created_at: 1,
      resolved_at: null,
      resolution: null,
    };
    const dreamRow: MaintenanceAuditRow = {
      id: 42,
      run_id: "run_numeric",
      process: "curator",
      action: "applied",
      targets: {},
      reversal: {},
      applied_at: 1,
      reverted_at: null,
      reverted_by: null,
    };
    const fetchMock = setupFetch({ reviews: [review], dreamRows: [dreamRow] });
    renderPalette();

    const input = await openWithMeta();
    fireEvent.change(input, { target: { value: "4" } });
    await advancePaletteTimers(199);
    expect(callsForPath(fetchMock, "/api/reviews")).toBe(0);
    expect(callsForPath(fetchMock, "/api/dream/audit")).toBe(0);

    fireEvent.change(input, { target: { value: "42" } });
    await advancePaletteTimers(200);

    expect(screen.getByText("Open Review 42")).toBeInTheDocument();
    expect(screen.getByText("Open Dream audit row 42")).toBeInTheDocument();
    expect(callsForPath(fetchMock, "/api/reviews")).toBe(1);
    expect(callsForPath(fetchMock, "/api/dream/audit")).toBe(1);

    fireEvent.change(input, { target: { value: "43" } });
    await advancePaletteTimers(200);
    expect(callsForPath(fetchMock, "/api/reviews")).toBe(1);
    expect(callsForPath(fetchMock, "/api/dream/audit")).toBe(1);

    fireEvent.change(input, { target: { value: "42" } });
    await advancePaletteTimers(200);
    fireEvent.keyDown(input, { key: "Enter" });

    expect(screen.getByTestId("inspector-target")).toHaveTextContent("review:42");
  });

  it("shows an honest degraded hint for non-prefixed ids without guessing an object type", async () => {
    const fetchMock = setupFetch();
    renderPalette();

    const input = await openWithMeta();
    fireEvent.change(input, { target: { value: "review_42" } });

    expect(await screen.findByText("Object ID not resolvable")).toBeInTheDocument();
    fireEvent.keyDown(input, { key: "Enter" });

    expect(screen.getByTestId("inspector-target")).toHaveTextContent("none");
    expect(callsForPath(fetchMock, "/api/memory/bands/episodic")).toBe(0);
  });

  it("uses server memory search results and opens the active memory object", async () => {
    vi.useFakeTimers();
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

    await advancePaletteTimers(200);

    expect(screen.getByText("Open Meaning episode")).toBeInTheDocument();
    fireEvent.keyDown(input, { key: "Enter" });

    expect(screen.getByTestId("inspector-target")).toHaveTextContent(`episode:${episodeId}`);
    expect(callsForPath(fetchMock, "/api/memory/bands/episodic")).toBeGreaterThanOrEqual(1);
    expect(callsForPath(fetchMock, "/api/memory/bands/semantic")).toBeGreaterThanOrEqual(1);
    expect(callsForPath(fetchMock, "/api/memory/bands/procedural")).toBeGreaterThanOrEqual(1);
  });

  it("debounces memory fan-out and flushes the pending search on Enter", async () => {
    vi.useFakeTimers();
    const episodeId = "ep_debounce000001";
    const fetchMock = setupFetch({
      memory: {
        episodic: {
          band: "episodic",
          mode: "search",
          query: "meaning",
          items: [
            {
              id: episodeId,
              title: "Debounced episode",
              narrative: "Returned after debounce.",
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

    expect(callsForPath(fetchMock, "/api/memory/bands/episodic")).toBe(0);
    await advancePaletteTimers(199);
    expect(callsForPath(fetchMock, "/api/memory/bands/episodic")).toBe(0);

    await advancePaletteTimers(1);

    expect(screen.getByText("Open Debounced episode")).toBeInTheDocument();
    const searchesAfterDebounce = callsForPath(fetchMock, "/api/memory/bands/episodic");

    fireEvent.change(input, { target: { value: "zzflush" } });
    expect(callsForPath(fetchMock, "/api/memory/bands/episodic")).toBe(searchesAfterDebounce);

    fireEvent.keyDown(input, { key: "Enter" });
    await flushMicrotasks();

    expect(callsForPath(fetchMock, "/api/memory/bands/episodic")).toBeGreaterThan(
      searchesAfterDebounce,
    );
  });

  it("keeps memory hits when one searchable band fails and shows the real partial state", async () => {
    vi.useFakeTimers();
    const episodeId = "ep_partial0000001";
    setupFetch({
      memory: {
        episodic: {
          band: "episodic",
          mode: "search",
          query: "partial",
          items: [
            {
              id: episodeId,
              title: "Partial episode",
              narrative: "Returned despite a semantic failure.",
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
      memoryErrors: {
        semantic: "semantic unavailable",
      },
    });
    renderPalette();

    const input = await openWithMeta();
    fireEvent.change(input, { target: { value: "partial" } });

    await advancePaletteTimers(200);

    expect(screen.getByText("Open Partial episode")).toBeInTheDocument();
    expect(screen.getByText("Partial memory results")).toBeInTheDocument();
    expect(screen.getByText(/semantic unavailable/)).toBeInTheDocument();
  });

  it("does not offer a dead opener for truncated prefixed ids", async () => {
    vi.useFakeTimers();
    const fetchMock = setupFetch();
    renderPalette();

    const input = await openWithMeta();
    fireEvent.change(input, { target: { value: "cmt_p48r" } });

    await advancePaletteTimers(250);

    expect(callsForPath(fetchMock, "/api/commitments")).toBe(0);
    expect(screen.queryByText(/Open Commitment/)).not.toBeInTheDocument();
  });

  it("does not registry-check long partial prefixed ids outside the strict id shape", async () => {
    vi.useFakeTimers();
    const fetchMock = setupFetch();
    renderPalette();

    const input = await openWithMeta();
    fireEvent.change(input, { target: { value: "cmt_1234567890" } });

    await advancePaletteTimers(250);

    expect(callsForPath(fetchMock, "/api/commitments")).toBe(0);
    expect(screen.queryByText(/Open Commitment/)).not.toBeInTheDocument();
  });

  it("resolves pasted ellipsized ids only from loaded object caches", async () => {
    vi.useFakeTimers();
    const episodeId = "ep_cachedellipsis0001";
    setupFetch({
      memory: {
        episodic: {
          band: "episodic",
          mode: "search",
          query: "cached",
          items: [
            {
              id: episodeId,
              title: "Cached ellipsis episode",
              narrative: "Loaded before the shortened id was pasted.",
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
    fireEvent.change(input, { target: { value: "cached" } });
    await advancePaletteTimers(200);
    expect(screen.getByText("Open Cached ellipsis episode")).toBeInTheDocument();

    fireEvent.change(input, { target: { value: shortId(episodeId) } });

    expect(screen.getByText(`Open Episode ${shortId(episodeId)}`)).toBeInTheDocument();
    fireEvent.keyDown(input, { key: "Enter" });
    expect(screen.getByTestId("inspector-target")).toHaveTextContent(`episode:${episodeId}`);
  });

  it("gives a full-id hint for pasted ellipsized ids that are not loaded", async () => {
    vi.useFakeTimers();
    const fetchMock = setupFetch();
    renderPalette();

    const input = await openWithMeta();
    fireEvent.change(input, { target: { value: "cmt_p48r…abcd" } });

    expect(screen.getByText("Paste the full id")).toBeInTheDocument();
    await advancePaletteTimers(250);
    expect(screen.queryByText(/Open Commitment/)).not.toBeInTheDocument();
    expect(callsForPath(fetchMock, "/api/commitments")).toBe(0);
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
