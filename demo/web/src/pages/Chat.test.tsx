import { act, fireEvent, render, screen, waitFor } from "@testing-library/react";

import { installMockWebSocket } from "../__tests__/mock-websocket";
import type { ApiState, EvidenceLedger, SessionRecord, StreamEntry } from "../api/types";
import { LiveProvider } from "../live/useLive";
import { StateProvider } from "../state/app-state";
import { MoodProvider } from "../state/mood";
import { ChatPage } from "./Chat";

const baseSession: SessionRecord = {
  session_id: "s_default",
  source_type: "demo",
  source_external_id: null,
  source_url: null,
  label: "default chat",
  audience_label: "operator",
  audience_entity_id: null,
  conversation_kind: "demo",
  created_at: Date.UTC(2026, 5, 11, 10),
  last_activity_at: Date.UTC(2026, 5, 11, 10),
  last_turn_id: null,
  message_count: 0,
  status: "active",
  privacy_level: "payload_on",
  participation_policy: "active",
  audience_role: "participant",
};

const operatorSession: SessionRecord = {
  ...baseSession,
  session_id: "s_operator",
  label: "operator chat",
  audience_role: "operator",
};

function state(): ApiState {
  return {
    active_session: "s_default",
    audiences: [],
    counts: {
      turns: 0,
      commitments: 0,
      open_qs: 0,
      open_reviews: 0,
      dream_audit_rows: 0,
    },
    current_mood: null,
    version: "0.1.0",
  };
}

function json(body: unknown, status = 200): Response {
  return new Response(JSON.stringify(body), {
    status,
    headers: { "Content-Type": "application/json" },
  });
}

function streamEntry(input: Partial<StreamEntry> & Pick<StreamEntry, "id" | "kind" | "timestamp" | "content">): StreamEntry {
  return {
    sender_entity_id: null,
    reply_target_entity_id: null,
    session_id: "s_default",
    sender_label: null,
    session_label: null,
    audience_label: null,
    ...input,
  };
}

function renderChat(options: { turnStatus?: number; turnMessage?: string } = {}) {
  const ws = installMockWebSocket();
  vi.spyOn(HTMLCanvasElement.prototype, "getContext").mockImplementation(() => null);
  let sessions = [baseSession];
  let postedTurn: unknown = null;
  const fetchMock = vi.spyOn(globalThis, "fetch").mockImplementation(async (input, init) => {
    const url = String(input);
    const method = init?.method ?? "GET";

    if (url === "/api/state") {
      return json(state());
    }
    if (url === "/api/sessions" && method === "GET") {
      return json({ sessions });
    }
    if (url === "/api/sessions/operator" && method === "POST") {
      sessions = [baseSession, operatorSession];
      return json(operatorSession);
    }
    if (url.startsWith("/api/stream?")) {
      return json({ entries: [], next_cursor: null });
    }
    if (url.startsWith("/api/turns?")) {
      return json({ rows: [], next_cursor: null });
    }
    if (url === "/api/turn" && method === "POST") {
      postedTurn = JSON.parse(String(init?.body));
      if (options.turnStatus !== undefined && options.turnStatus >= 400) {
        return json({ message: options.turnMessage ?? "turn failed" }, options.turnStatus);
      }

      return json({ ok: true, status: "queued", stream_entry_id: "strm_1" });
    }

    return json({ message: "not found" }, 404);
  });

  render(
    <LiveProvider>
      <StateProvider>
        <MoodProvider>
          <ChatPage />
        </MoodProvider>
      </StateProvider>
    </LiveProvider>,
  );

  return { fetchMock, postedTurn: () => postedTurn, ws };
}

describe("Chat page", () => {
  afterEach(() => {
    vi.restoreAllMocks();
    vi.unstubAllGlobals();
  });

  it("posts composer messages with the real body shape", async () => {
    const { postedTurn } = renderChat();

    const input = await screen.findByPlaceholderText("message the entity…");
    fireEvent.change(input, { target: { value: "hello" } });
    fireEvent.click(screen.getByRole("button", { name: /SEND/i }));

    await waitFor(() => expect(postedTurn()).not.toBeNull());
    expect(postedTurn()).toMatchObject({
      message: "hello",
      session: "s_default",
    });
    expect((postedTurn() as { external_message_id?: unknown }).external_message_id).toEqual(
      expect.any(String),
    );
  });

  it("surfaces server error status and message under the composer", async () => {
    renderChat({ turnStatus: 409, turnMessage: "audience is required for unknown sessions" });

    const input = await screen.findByPlaceholderText("message the entity…");
    fireEvent.change(input, { target: { value: "hello" } });
    fireEvent.click(screen.getByRole("button", { name: /SEND/i }));

    expect(await screen.findByText("409 audience is required for unknown sessions")).toBeTruthy();
  });

  it("ensures an operator session and selects the returned session", async () => {
    renderChat();

    fireEvent.click(await screen.findByRole("button", { name: /\+ ENSURE OPERATOR SESSION/i }));

    expect(await screen.findAllByText("operator chat")).toHaveLength(2);
  });

  it("does not render a previous session stream while the selected session is loading", async () => {
    installMockWebSocket();
    vi.spyOn(HTMLCanvasElement.prototype, "getContext").mockImplementation(() => null);
    let sessions = [baseSession];
    const pendingOperatorStream = new Promise<Response>(() => {});

    vi.spyOn(globalThis, "fetch").mockImplementation(async (input, init) => {
      const url = String(input);
      const method = init?.method ?? "GET";

      if (url === "/api/state") {
        return json(state());
      }
      if (url === "/api/sessions" && method === "GET") {
        return json({ sessions });
      }
      if (url === "/api/sessions/operator" && method === "POST") {
        sessions = [baseSession, operatorSession];
        return json(operatorSession);
      }
      if (url.startsWith("/api/stream?")) {
        if (url.includes("s_operator")) {
          return pendingOperatorStream;
        }

        return json({
          entries: [
            streamEntry({
              id: "u_default",
              kind: "user_msg",
              timestamp: Date.UTC(2026, 5, 11, 10),
              content: "default-only",
              sender_label: "operator",
            }),
          ],
          next_cursor: null,
        });
      }
      if (url.startsWith("/api/turns?")) {
        return json({ rows: [], next_cursor: null });
      }

      return json({ message: "not found" }, 404);
    });

    render(
      <LiveProvider>
        <StateProvider>
          <MoodProvider>
            <ChatPage />
          </MoodProvider>
        </StateProvider>
      </LiveProvider>,
    );

    expect(await screen.findByText("default-only")).toBeTruthy();

    fireEvent.click(await screen.findByRole("button", { name: /\+ ENSURE OPERATOR SESSION/i }));
    await screen.findAllByText("operator chat");

    expect(screen.queryByText("default-only")).toBeNull();
    expect(screen.getByText("loading stream…")).toBeTruthy();
  });

  it("resets current-turn cognition display when a new in-flight turn starts", async () => {
    const { ws } = renderChat();
    const ledger: EvidenceLedger = {
      sections: [
        {
          id: "episodes",
          label: "Episodes",
          entries: [{ id: "episode:1", source_type: "episode" }],
        },
      ],
    };

    await screen.findByPlaceholderText("message the entity…");
    act(() => {
      ws.instances[0]!.open();
    });
    await waitFor(() =>
      expect(ws.instances[0]!.sent).toContain(
        JSON.stringify({ type: "subscribe", session_id: "s_default" }),
      ),
    );

    await act(async () => {
      ws.instances[0]!.receive({
        type: "turn:phase:started",
        ts: 1,
        event: "turn_phase.started",
        data: {
          turnId: "t1",
          turn_id: "t1",
          session_id: "s_default",
          phase: "delib",
          ts: 1,
        },
      });
      ws.instances[0]!.receive({
        type: "turn:delib_path",
        ts: 2,
        turn_id: "t1",
        session_id: "s_default",
        path: "system_2",
      });
      ws.instances[0]!.receive({
        type: "evidence_ledger:built",
        ts: 3,
        turn_id: "t1",
        session_id: "s_default",
        ledger,
      });
    });

    expect(await screen.findByText("SYS_2 DELIB")).toBeTruthy();
    expect(await screen.findByText("EPI")).toBeTruthy();
    expect(await screen.findByText("t1")).toBeTruthy();

    await act(async () => {
      ws.instances[0]!.receive({
        type: "turn:phase:started",
        ts: 4,
        event: "turn_phase.started",
        data: {
          turnId: "t2",
          turn_id: "t2",
          session_id: "s_default",
          phase: "ingest",
          ts: 4,
        },
      });
    });

    expect(await screen.findByText("PATH …")).toBeTruthy();
    await waitFor(() => expect(screen.queryByText("EPI")).toBeNull());
    expect(await screen.findByText("no ledger for current turn yet")).toBeTruthy();
  });
});
