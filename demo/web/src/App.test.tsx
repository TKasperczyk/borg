import { fireEvent, render, screen } from "@testing-library/react";
import { afterEach, describe, expect, it, vi } from "vitest";

import type { SessionRecord, StateSnapshot } from "./api/types";

const mocks = vi.hoisted(() => ({
  liveSubscribe: vi.fn(() => () => undefined),
  runTurn: vi.fn(),
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

    fireEvent.click(await screen.findByRole("button", { name: "operator chat" }));

    expect(await screen.findByText("mark a creator first")).toBeInTheDocument();
    expect(fetchMock).toHaveBeenCalledWith(
      "/api/sessions/operator",
      expect.objectContaining({ method: "POST" }),
    );
  });
});
