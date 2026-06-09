import { fireEvent, render, screen } from "@testing-library/react";
import { afterEach, describe, expect, it, vi } from "vitest";

import type { SessionRecord, StateSnapshot } from "./api/types";

const mocks = vi.hoisted(() => ({
  liveSubscribe: vi.fn(() => () => undefined),
  runTurn: vi.fn(),
  crashMemory: false,
}));

vi.mock("./hooks/use-live-events", () => ({
  useLiveEvents: () => ({
    wsState: "live",
    connectionCount: 1,
    subscribe: mocks.liveSubscribe,
  }),
}));

vi.mock("./hooks/use-turn-stream", () => ({
  useTurnStream: () => ({
    activeTurnId: null,
    running: false,
    phases: [],
    tokenTextByPhase: new Map(),
    detailByPhase: new Map(),
    terminalOutcome: null,
    delibPath: null,
    finalAttempt: 1,
    flowSnapshotByTurn: new Map(),
    eventTail: [],
    ledgerByTurn: new Map(),
    lastPhase: "idle",
    runTurn: mocks.runTurn,
    resetForReconnect: vi.fn(),
    replaceTailFromEntries: vi.fn(),
  }),
}));

vi.mock("./screens/Cognition", () => ({
  CognitionScreen: () => <div data-testid="cognition-screen" />,
}));

vi.mock("./screens/Stream", () => ({
  StreamScreen: () => <div data-testid="stream-screen" />,
}));

vi.mock("./screens/Memory", () => ({
  MemoryScreen: ({ onOpenReview }: { onOpenReview: () => void }) => {
    if (mocks.crashMemory) {
      throw new Error("memory exploded");
    }
    return (
      <div data-testid="memory-screen">
        <button type="button" onClick={onOpenReview}>
          open review
        </button>
      </div>
    );
  },
}));

vi.mock("./screens/MissionControl", () => ({
  MissionControlScreen: () => <div data-testid="mission-screen" />,
}));

vi.mock("./screens/Identity", () => ({
  IdentityScreen: () => <div data-testid="identity-screen" />,
}));

vi.mock("./screens/Governance", () => ({
  GovernanceScreen: ({ activeTab }: { activeTab: string }) => (
    <div data-testid="governance-screen">{activeTab}</div>
  ),
}));

vi.mock("./screens/Review", () => ({
  ReviewScreen: () => <div data-testid="review-screen" />,
}));

vi.mock("./screens/Dream", () => ({
  DreamScreen: ({ onOpenReview }: { onOpenReview: () => void }) => (
    <div data-testid="dream-screen">
      <button type="button" onClick={onOpenReview}>
        open review
      </button>
    </div>
  ),
}));

vi.mock("./screens/Prompts", () => ({
  PromptsScreen: () => <div data-testid="prompts-screen" />,
}));

import { App } from "./App";

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

function jsonResponse(body: unknown, status = 200): Response {
  return new Response(JSON.stringify(body), {
    status,
    headers: { "Content-Type": "application/json" },
  });
}

afterEach(() => {
  vi.restoreAllMocks();
  mocks.crashMemory = false;
  window.history.replaceState(null, "", "/");
});

describe("App", () => {
  it("shows the mark-creator affordance when operator chat returns 409", async () => {
    const fetchMock = vi.fn((input: RequestInfo | URL, init?: RequestInit) => {
      const url = new URL(String(input), "http://localhost");

      if (url.pathname === "/api/state") {
        return Promise.resolve(jsonResponse(stateSnapshot()));
      }
      if (url.pathname === "/api/sessions" && init?.method !== "POST") {
        return Promise.resolve(jsonResponse({ sessions: [session()] }));
      }
      if (url.pathname === "/api/entities/creator") {
        return Promise.resolve(jsonResponse(null));
      }
      if (url.pathname === "/api/sessions/operator" && init?.method === "POST") {
        return Promise.resolve(jsonResponse({ error: { message: "no creator entity set" } }, 409));
      }

      return Promise.reject(new Error(`unexpected fetch ${url.pathname}`));
    });
    vi.stubGlobal("fetch", fetchMock);

    render(<App />);

    expect(screen.queryByRole("button", { name: "graph" })).not.toBeInTheDocument();
    expect(screen.queryByRole("button", { name: "shared" })).not.toBeInTheDocument();
    expect(screen.getByRole("button", { name: "identity" })).toHaveAttribute(
      "title",
      "identity (⌘4)",
    );
    expect(screen.getByRole("button", { name: "prompts" })).toHaveAttribute(
      "title",
      "prompts (⌘8)",
    );

    fireEvent.click(await screen.findByRole("button", { name: "operator chat" }));

    expect(await screen.findByText("mark a creator first")).toBeInTheDocument();
    expect(fetchMock).toHaveBeenCalledWith(
      "/api/sessions/operator",
      expect.objectContaining({ method: "POST" }),
    );
  });

  it("deep-links the shell view and updates the view URL", async () => {
    const fetchMock = vi.fn((input: RequestInfo | URL, init?: RequestInit) => {
      const url = new URL(String(input), "http://localhost");

      if (url.pathname === "/api/state") {
        return Promise.resolve(jsonResponse(stateSnapshot()));
      }
      if (url.pathname === "/api/sessions" && init?.method !== "POST") {
        return Promise.resolve(jsonResponse({ sessions: [session()] }));
      }
      if (url.pathname === "/api/entities/creator") {
        return Promise.resolve(jsonResponse(null));
      }

      return Promise.reject(new Error(`unexpected fetch ${url.pathname}`));
    });
    vi.stubGlobal("fetch", fetchMock);
    window.history.replaceState(null, "", "/?view=memory");

    render(<App />);

    expect(await screen.findByTestId("memory-screen")).toBeInTheDocument();

    fireEvent.click(screen.getByRole("button", { name: "review" }));

    expect(await screen.findByTestId("review-screen")).toBeInTheDocument();
    expect(new URL(window.location.href).searchParams.get("view")).toBe("review");
  });

  it("normalizes legacy directive deep-links into the Governance directives tab", async () => {
    const fetchMock = vi.fn((input: RequestInfo | URL, init?: RequestInit) => {
      const url = new URL(String(input), "http://localhost");

      if (url.pathname === "/api/state") {
        return Promise.resolve(jsonResponse(stateSnapshot()));
      }
      if (url.pathname === "/api/sessions" && init?.method !== "POST") {
        return Promise.resolve(jsonResponse({ sessions: [session()] }));
      }
      if (url.pathname === "/api/entities/creator") {
        return Promise.resolve(jsonResponse(null));
      }

      return Promise.reject(new Error(`unexpected fetch ${url.pathname}`));
    });
    vi.stubGlobal("fetch", fetchMock);
    window.history.replaceState(null, "", "/?view=directives");

    render(<App />);

    expect(await screen.findByTestId("governance-screen")).toHaveTextContent("shared_state");
    const url = new URL(window.location.href);
    expect(url.searchParams.get("view")).toBe("governance");
    expect(url.searchParams.get("tab")).toBe("shared_state");
  });

  it("keeps chrome mounted and recovers from a screen crash when the view changes", async () => {
    const consoleError = vi.spyOn(console, "error").mockImplementation(() => undefined);
    const preventExpectedErrorReport = (event: ErrorEvent) => event.preventDefault();
    const fetchMock = vi.fn((input: RequestInfo | URL, init?: RequestInit) => {
      const url = new URL(String(input), "http://localhost");

      if (url.pathname === "/api/state") {
        return Promise.resolve(jsonResponse(stateSnapshot()));
      }
      if (url.pathname === "/api/sessions" && init?.method !== "POST") {
        return Promise.resolve(jsonResponse({ sessions: [session()] }));
      }
      if (url.pathname === "/api/entities/creator") {
        return Promise.resolve(jsonResponse(null));
      }

      return Promise.reject(new Error(`unexpected fetch ${url.pathname}`));
    });
    vi.stubGlobal("fetch", fetchMock);
    mocks.crashMemory = true;
    window.history.replaceState(null, "", "/?view=memory");
    window.addEventListener("error", preventExpectedErrorReport);

    render(<App />);

    expect(await screen.findByRole("alert")).toHaveTextContent("screen crashed");
    expect(screen.getByRole("button", { name: "reload" })).toBeInTheDocument();
    expect(screen.getByRole("button", { name: "retry" })).toBeInTheDocument();
    expect(screen.getByRole("button", { name: "review" })).toBeInTheDocument();

    mocks.crashMemory = false;
    fireEvent.click(screen.getByRole("button", { name: "review" }));

    expect(screen.getByTestId("review-screen")).toBeInTheDocument();
    expect(screen.queryByRole("alert")).not.toBeInTheDocument();
    expect(new URL(window.location.href).searchParams.get("view")).toBe("review");
    window.removeEventListener("error", preventExpectedErrorReport);
    consoleError.mockRestore();
  });
});
