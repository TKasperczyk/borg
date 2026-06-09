import { act, fireEvent, screen, waitFor, within } from "@testing-library/react";
import { afterEach, describe, expect, it, vi } from "vitest";

import type {
  EvidenceLedger,
  LiveFrame,
  SessionRecord,
  StreamEntry,
  TurnHistoryRow,
  TurnTerminalOutcome,
  TurnPhaseName,
  WsState,
} from "../../api/types";
import { LiveEventsProvider } from "../../hooks/live-context";
import type { LiveEventHandler, LiveEvents } from "../../hooks/use-live-events";
import { useTurnStream } from "../../hooks/use-turn-stream";
import { renderWithInspector } from "../../test/inspector";
import { ChatStream } from "./ChatStream";
import { CognitionScreen } from "./index";
import { LedgerView } from "./LedgerView";

function jsonResponse(body: unknown): Response {
  return new Response(JSON.stringify(body), {
    status: 200,
    headers: { "Content-Type": "application/json" },
  });
}

function deferredResponse(): {
  promise: Promise<Response>;
  resolve: (response: Response) => void;
} {
  let resolve!: (response: Response) => void;
  const promise = new Promise<Response>((innerResolve) => {
    resolve = innerResolve;
  });
  return { promise, resolve };
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

function streamEntry(
  input: Partial<StreamEntry> & Pick<StreamEntry, "id" | "kind" | "content">,
): StreamEntry {
  return {
    timestamp: 1,
    turn_id: "turn_1",
    audience: "alice",
    sender_entity_id: null,
    reply_target_entity_id: null,
    session_id: "default",
    compressed: false,
    ...input,
  };
}

function sessionRecord(input: Partial<SessionRecord> = {}): SessionRecord {
  return {
    session_id: "default",
    source_type: "demo",
    source_external_id: null,
    source_url: null,
    label: "demo",
    audience_label: "alice",
    audience_entity_id: null,
    conversation_kind: "demo",
    created_at: 1,
    last_activity_at: 1,
    last_turn_id: null,
    message_count: 0,
    status: "active",
    privacy_level: "payload_off",
    participation_policy: "active",
    audience_role: "participant",
    ...input,
  };
}

function turnRow(input: Partial<TurnHistoryRow> & Pick<TurnHistoryRow, "turn_id">): TurnHistoryRow {
  return {
    started_at: 1,
    audience: "alice",
    outcome: "emitted",
    suppression_reason: null,
    ...input,
  };
}

function installCognitionFetch(
  input: {
    streamEntries?: StreamEntry[][];
    turnRows?: TurnHistoryRow[];
    sharedState?: unknown;
    commitments?: unknown;
    identity?: unknown;
    prompt?: unknown;
    ledgerResponses?: Record<string, Response | Promise<Response>>;
    turnResponse?: Response | Promise<Response>;
    turnResponses?: Array<Response | Promise<Response>>;
  } = {},
): {
  fetchMock: ReturnType<typeof vi.fn>;
  streamCalls: () => number;
} {
  const streamResponses = input.streamEntries ?? [[]];
  let streamCallCount = 0;
  let turnCallCount = 0;
  const fetchMock = vi.fn((request: RequestInfo | URL, init?: RequestInit) => {
    const url = new URL(String(request), "http://test.invalid");
    if (url.pathname === "/api/stream") {
      const entries = streamResponses[Math.min(streamCallCount, streamResponses.length - 1)];
      streamCallCount += 1;
      return Promise.resolve(jsonResponse({ entries, next_cursor: null }));
    }
    if (url.pathname === "/api/turns") {
      return Promise.resolve(jsonResponse({ rows: input.turnRows ?? [], next_cursor: null }));
    }
    if (url.pathname.startsWith("/api/turns/") && url.pathname.endsWith("/ledger")) {
      const turnId = decodeURIComponent(
        url.pathname.replace("/api/turns/", "").replace("/ledger", ""),
      );
      return Promise.resolve(
        input.ledgerResponses?.[turnId] ??
          new Response(JSON.stringify({ error: { message: "ledger missing" } }), {
            status: 404,
            headers: { "Content-Type": "application/json" },
          }),
      );
    }
    if (url.pathname === "/api/shared-state") {
      return Promise.resolve(jsonResponse(input.sharedState ?? { audience: "alice", entries: [] }));
    }
    if (url.pathname === "/api/commitments") {
      return Promise.resolve(jsonResponse(input.commitments ?? { commitments: [] }));
    }
    if (url.pathname === "/api/identity") {
      return Promise.resolve(
        jsonResponse(
          input.identity ?? {
            values: [],
            goals: [],
            traits: [],
            open_questions: [],
            growth_markers: [],
            periods: [],
            open_question_events: [],
          },
        ),
      );
    }
    if (url.pathname === "/api/prompts/assembled") {
      return Promise.resolve(jsonResponse(input.prompt ?? { sections: [], text: "" }));
    }
    if (
      url.pathname.includes("/api/sessions/") &&
      url.pathname.endsWith("/participation") &&
      init?.method === "POST"
    ) {
      const body = JSON.parse(String(init.body)) as {
        policy: SessionRecord["participation_policy"];
      };
      const sessionId = decodeURIComponent(
        url.pathname.split("/api/sessions/")[1]?.replace("/participation", "") ?? "default",
      );

      return Promise.resolve(
        jsonResponse(
          sessionRecord({
            session_id: sessionId,
            participation_policy: body.policy,
          }),
        ),
      );
    }
    if (url.pathname === "/api/turn" && init?.method === "POST") {
      const response =
        input.turnResponses?.[Math.min(turnCallCount, input.turnResponses.length - 1)] ??
        input.turnResponse ??
        jsonResponse({ ok: true, status: "enqueued", stream_entry_id: "strm_user_abc" });
      turnCallCount += 1;
      return Promise.resolve(response);
    }
    return Promise.resolve(new Response("{}", { status: 404 }));
  });
  vi.stubGlobal("fetch", fetchMock);
  return {
    fetchMock,
    streamCalls: () => streamCallCount,
  };
}

function reflectFrame(turnId = "turn_abc"): LiveFrame {
  return {
    type: "turn:phase:completed",
    event: "turn_phase.completed",
    ts: Date.now(),
    session_id: "default",
    data: {
      turnId,
      turn_id: turnId,
      session_id: "default",
      phase: "reflect",
      duration_ms: 12,
      sub: "done",
    },
  };
}

function phaseFrame(
  type: "turn:phase:started" | "turn:phase:completed" | "turn:phase:failed",
  phase: TurnPhaseName,
  turnId = "turn_abc",
): LiveFrame {
  const event =
    type === "turn:phase:started"
      ? "turn_phase.started"
      : type === "turn:phase:completed"
        ? "turn_phase.completed"
        : "turn_phase.failed";

  return {
    type,
    event,
    ts: Date.now(),
    session_id: "default",
    data: {
      turnId,
      turn_id: turnId,
      session_id: "default",
      phase,
      duration_ms: type === "turn:phase:started" ? undefined : 12,
      sub: type === "turn:phase:started" ? "running" : "done",
    },
  };
}

function terminalFrame(
  turnId = "turn_abc",
  outcome: TurnTerminalOutcome = "suppressed_action",
): LiveFrame {
  return {
    type: "turn:terminal",
    event: "turn.terminal",
    ts: Date.now(),
    session_id: "default",
    data: {
      turnId,
      turn_id: turnId,
      session_id: "default",
      outcome,
      duration_ms: 42,
    },
  };
}

function ledgerWithText(text: string): EvidenceLedger {
  return {
    sections: [
      {
        id: "episodes",
        label: "episodes",
        entries: [
          {
            id: `entry_${text}`,
            source_type: "episode",
            session_scope: "current_session",
            actor: "memory",
            trust_rank: 1,
            text,
          },
        ],
      },
    ],
    sharedState: null,
    transcriptIncluded: true,
    transcriptCompacted: false,
    originalTranscriptTokenEstimate: 1,
    compactedTranscriptEntryCount: 0,
    rawPreservedUserTranscriptEntryCount: 1,
    estimatedTokens: 1,
  };
}

function chatUserBodies(): string[] {
  return [...document.querySelectorAll(".chat-msg.user .body")].map(
    (element) => element.textContent ?? "",
  );
}

function turnPostCalls(fetchMock: ReturnType<typeof vi.fn>) {
  return fetchMock.mock.calls.filter(
    ([request, init]) => String(request).endsWith("/api/turn") && init?.method === "POST",
  );
}

function fetchPathCalls(fetchMock: ReturnType<typeof vi.fn>, pathname: string): unknown[][] {
  return fetchMock.mock.calls.filter(
    ([request]) => new URL(String(request), "http://test.invalid").pathname === pathname,
  );
}

function Harness({
  live,
  sessionId = "default",
  session = sessionRecord({ session_id: sessionId }),
  onSessionPolicyChanged = async () => undefined,
}: {
  live: LiveEvents;
  sessionId?: string;
  session?: SessionRecord | null;
  onSessionPolicyChanged?: () => Promise<void>;
}) {
  const turnStream = useTurnStream(live, { sessionId });
  return (
    <LiveEventsProvider value={live}>
      <CognitionScreen
        sessionId={sessionId}
        audience="alice"
        turnStream={turnStream}
        session={session}
        onSessionPolicyChanged={onSessionPolicyChanged}
      />
    </LiveEventsProvider>
  );
}

afterEach(() => {
  vi.restoreAllMocks();
  vi.unstubAllGlobals();
});

describe("cognition screen", () => {
  it("renders image attachments without turn ids under their parent user turn", () => {
    const entries: StreamEntry[] = [
      streamEntry({
        id: "strm_zuser",
        timestamp: 1,
        entry_index: 1,
        kind: "user_msg",
        content: "see this",
      }),
      streamEntry({
        id: "strm_aatt",
        timestamp: 1,
        entry_index: 2,
        kind: "user_image_attachment",
        turn_id: undefined,
        content: {
          type: "image_ref",
          attachment_id: "att_123",
          media_type: "image/png",
          parent_entry_id: "strm_zuser",
        },
      }),
    ];

    renderWithInspector(
      <ChatStream entries={entries} sessionId="default" audience="alice" running={false} />,
    );

    expect(screen.getByText("see this")).toBeInTheDocument();
    expect(screen.getByText("[att:att_123]")).toBeInTheDocument();
  });

  it("renders suppressed and observed stream entries as diagnostic markers", () => {
    const entries: StreamEntry[] = [
      streamEntry({
        id: "strm_user_marker",
        timestamp: 1,
        kind: "user_msg",
        turn_id: "turn_marker",
        content: "marker source",
      }),
      streamEntry({
        id: "strm_suppressed_marker",
        timestamp: 2,
        kind: "agent_suppressed",
        turn_id: "turn_marker",
        content: {
          reason: "finalizer_no_output",
          user_entry_ids: ["strm_user_marker"],
          turn_id: "turn_marker",
          primary_no_output_reason: "low_value_echo",
          no_output_categories: ["closure"],
          structural_no_output_flags: ["with_open_question"],
          finalizer_invalid_tool: {
            tool_name: "EmitAnswer",
            reason: "invalid schema",
            attempt: "regenerate",
          },
        },
      }),
      streamEntry({
        id: "strm_observed_marker",
        timestamp: 3,
        kind: "agent_observed",
        turn_id: "turn_observed",
        content: {
          reason: "observer policy",
          user_entry_id: "strm_user_marker",
          turn_id: "turn_observed",
        },
      }),
    ];

    renderWithInspector(
      <ChatStream entries={entries} sessionId="default" audience="alice" running={false} />,
    );

    expect(screen.getByText("deliberate silence")).toBeInTheDocument();
    expect(screen.getByText("observed")).toBeInTheDocument();

    const marker = screen.getByText("deliberate silence").closest("details");
    expect(marker).not.toBeNull();
    fireEvent.click(within(marker as HTMLElement).getByText("deliberate silence"));

    expect(within(marker as HTMLElement).getByText("low_value_echo")).toBeInTheDocument();
    expect(within(marker as HTMLElement).getByText("closure")).toBeInTheDocument();
    expect(within(marker as HTMLElement).getByText("with_open_question")).toBeInTheDocument();
    expect(within(marker as HTMLElement).getByText(/EmitAnswer/)).toBeInTheDocument();
    expect(
      within(marker as HTMLElement).getByRole("button", { name: "jump to turn_marker" }),
    ).toBeInTheDocument();
    expect(
      within(marker as HTMLElement).getByRole("button", { name: "jump to strm_user_marker" }),
    ).toBeInTheDocument();
  });

  it("renders message audience chips and response source-entry refs", () => {
    const entries: StreamEntry[] = [
      streamEntry({
        id: "strm_user_source",
        timestamp: 1,
        kind: "user_msg",
        content: "source question",
      }),
      streamEntry({
        id: "strm_agent_response",
        timestamp: 2,
        kind: "agent_msg",
        content: "source answer",
        response_to: {
          source_entry_ids: ["strm_user_source"],
        },
      }),
    ];

    renderWithInspector(
      <ChatStream entries={entries} sessionId="default" audience="alice" running={false} />,
    );

    expect(screen.getAllByText("aud alice").length).toBeGreaterThan(0);
    expect(screen.getByRole("button", { name: "jump to strm_user_source" })).toBeInTheDocument();
  });

  it("does not render a stale cached ledger after switching turns", () => {
    const fetchMock = vi.fn(() => new Promise<Response>(() => undefined));
    vi.stubGlobal("fetch", fetchMock);
    const { rerender } = renderWithInspector(
      <LedgerView
        turnId="turn_a"
        cachedLedger={ledgerWithText("ledger A")}
        active
        audience="alice"
      />,
    );

    expect(screen.getByText("ledger A")).toBeInTheDocument();

    rerender(<LedgerView turnId="turn_b" cachedLedger={undefined} active audience="alice" />);

    expect(screen.queryByText("ledger A")).not.toBeInTheDocument();
    expect(screen.getByText("ledger not loaded yet")).toBeInTheDocument();
  });

  it("renders every backend ledger section dynamically", () => {
    renderWithInspector(
      <LedgerView
        turnId="turn_sections"
        cachedLedger={{
          ...ledgerWithText("episode kept"),
          sections: [
            {
              id: "current_session_transcript",
              label: "current session transcript",
              entries: [
                {
                  id: "ledger_transcript",
                  source_type: "current_session_stream",
                  session_scope: "current_session",
                  actor: "user",
                  trust_rank: 1,
                  text: "transcript kept",
                },
              ],
            },
            {
              id: "action_states",
              label: "action states",
              entries: [
                {
                  id: "ledger_action",
                  source_type: "action_record",
                  session_scope: "current_session",
                  actor: "memory",
                  trust_rank: 2,
                  text: "action kept",
                },
              ],
            },
            {
              id: "attribution_matrix",
              label: "attribution matrix",
              entries: [
                {
                  id: "ledger_attr",
                  source_type: "system_metadata",
                  session_scope: "current_session",
                  actor: "system",
                  trust_rank: 3,
                  text: "attribution kept",
                },
              ],
            },
            {
              id: "open_questions",
              label: "open questions",
              entries: [
                {
                  id: "ledger_question",
                  source_type: "system_metadata",
                  session_scope: "current_session",
                  actor: "memory",
                  trust_rank: 4,
                  text: "question kept",
                },
              ],
            },
          ],
        }}
        active
        audience="alice"
      />,
    );

    expect(screen.getByText("current session transcript")).toBeInTheDocument();
    expect(screen.getByText("transcript kept")).toBeInTheDocument();
    expect(screen.getByText("action states")).toBeInTheDocument();
    expect(screen.getByText("action kept")).toBeInTheDocument();
    expect(screen.getByText("attribution matrix")).toBeInTheDocument();
    expect(screen.getByText("attribution kept")).toBeInTheDocument();
    expect(screen.getByText("open questions")).toBeInTheDocument();
    expect(screen.getByText("question kept")).toBeInTheDocument();
  });

  it("opens the inspector from a namespaced ledger source handle raw id", async () => {
    const episodeId = "ep_ledger111111111";
    vi.stubGlobal(
      "fetch",
      vi.fn((request: RequestInfo | URL) => {
        const url = new URL(String(request), "http://test.invalid");
        if (url.pathname === "/api/memory/bands/episodic") {
          return Promise.resolve(
            jsonResponse({
              band: "episodic",
              items: [
                {
                  id: episodeId,
                  title: "Ledger-linked episode",
                  narrative: "Opened through the ledger source handle.",
                  participants: [],
                  location: null,
                  start_time: 1,
                  end_time: 2,
                  audience: "alice",
                  significance: 0.5,
                  confidence: 0.8,
                  tags: [],
                  source_stream_ids: ["strm_ledger111111"],
                  source_count: 1,
                  lineage: { derived_from: [], supersedes: [] },
                  emotional_arc: null,
                  vector_dims: 4,
                  created_at: 1,
                  updated_at: 2,
                },
              ],
              next_cursor: null,
            }),
          );
        }
        return Promise.resolve(new Response("not found", { status: 404 }));
      }),
    );

    renderWithInspector(
      <LedgerView
        turnId="turn_ledger"
        cachedLedger={{
          ...ledgerWithText("episode source kept"),
          sections: [
            {
              id: "episodes",
              label: "episodes",
              entries: [
                {
                  id: `episode:${episodeId}`,
                  source_type: "episode",
                  session_scope: "current_session",
                  actor: "memory",
                  trust_rank: 1,
                  text: "episode source kept",
                },
              ],
            },
          ],
        }}
        active
        audience="alice"
      />,
      { inspector: true },
    );

    fireEvent.click(screen.getByRole("button", { name: `jump to ${episodeId}` }));

    expect(await screen.findByRole("dialog", { name: "Episode inspector" })).toBeInTheDocument();
    expect(await screen.findByText("Ledger-linked episode")).toBeInTheDocument();
  });

  it("keeps ledger source handles inert when no single raw id is recoverable", () => {
    renderWithInspector(
      <LedgerView
        turnId="turn_ledger"
        cachedLedger={{
          ...ledgerWithText("unrecoverable source kept"),
          sections: [
            {
              id: "episodes",
              label: "episodes",
              entries: [
                {
                  id: "retrieved_evidence:not_prefixed",
                  source_type: "episode",
                  session_scope: "current_session",
                  actor: "memory",
                  trust_rank: 1,
                  text: "unrecoverable source kept",
                },
              ],
            },
          ],
        }}
        active
        audience="alice"
      />,
    );

    expect(screen.getByText("[episodes:retrieved_evidence:not_prefixed]")).toBeInTheDocument();
    expect(screen.queryByRole("button", { name: /retrieved_evidence:not_prefixed/ })).toBeNull();
  });

  it("shows an optimistic queued user bubble immediately and marks it sent after ack", async () => {
    const source = makeLiveSource();
    const pendingTurn = deferredResponse();
    installCognitionFetch({
      turnResponse: pendingTurn.promise,
    });

    renderWithInspector(<Harness live={source.live()} />);

    fireEvent.change(screen.getByPlaceholderText("send a turn"), {
      target: { value: "hello queued" },
    });
    fireEvent.click(screen.getByRole("button", { name: "send" }));

    await waitFor(() => expect(chatUserBodies()).toContain("hello queued"));
    expect(screen.getByText("queued")).toBeInTheDocument();

    await act(async () => {
      pendingTurn.resolve(
        jsonResponse({
          ok: true,
          status: "enqueued",
          stream_entry_id: "strm_user_queued",
        }),
      );
      await pendingTurn.promise;
    });

    expect(screen.getByText("sent")).toBeInTheDocument();
  });

  it("adopts the live batch turn frame instead of the POST ack for visualization", async () => {
    const source = makeLiveSource();
    const { fetchMock } = installCognitionFetch({
      turnResponse: jsonResponse({
        ok: true,
        status: "enqueued",
        stream_entry_id: "strm_user_ack",
      }),
    });

    renderWithInspector(<Harness live={source.live()} />);

    fireEvent.change(screen.getByPlaceholderText("send a turn"), {
      target: { value: "hello batch" },
    });
    fireEvent.click(screen.getByRole("button", { name: "send" }));

    await waitFor(() =>
      expect(
        fetchMock.mock.calls.some(
          ([request, init]) => String(request).endsWith("/api/turn") && init?.method === "POST",
        ),
      ).toBe(true),
    );

    expect(document.querySelector(".turn-id")?.textContent).toContain("idle");
    expect(screen.queryByText("strm_user_ack")).not.toBeInTheDocument();

    act(() => {
      source.emit(phaseFrame("turn:phase:started", "ingest", "turn_batch"));
    });

    expect(screen.getByText("turn_batch")).toBeInTheDocument();
    expect(screen.getByTestId("phase-ingest")).toHaveClass("fc-node-running");
  });

  it("releases the active turn after terminal so a later batch from rapid sends is visualized", async () => {
    const source = makeLiveSource();
    const { fetchMock } = installCognitionFetch({
      turnResponses: [
        jsonResponse({
          ok: true,
          status: "enqueued",
          stream_entry_id: "strm_user_a",
        }),
        jsonResponse({
          ok: true,
          status: "enqueued",
          stream_entry_id: "strm_user_b",
        }),
      ],
    });

    renderWithInspector(<Harness live={source.live()} />);

    fireEvent.change(screen.getByPlaceholderText("send a turn"), {
      target: { value: "message A" },
    });
    fireEvent.click(screen.getByRole("button", { name: "send" }));
    await waitFor(() => expect(turnPostCalls(fetchMock)).toHaveLength(1));

    fireEvent.change(screen.getByPlaceholderText("send a turn"), {
      target: { value: "message B" },
    });
    fireEvent.click(screen.getByRole("button", { name: "send" }));
    await waitFor(() => expect(turnPostCalls(fetchMock)).toHaveLength(2));

    act(() => {
      source.emit(phaseFrame("turn:phase:started", "ingest", "turn_a"));
    });
    expect(screen.getByText("turn_a")).toBeInTheDocument();
    expect(screen.getByTestId("phase-ingest")).toHaveClass("fc-node-running");

    act(() => {
      source.emit({
        type: "stream:append",
        ts: Date.now(),
        entries: [
          streamEntry({
            id: "strm_agent_a",
            kind: "agent_msg",
            turn_id: "turn_a",
            content: "answer A",
            response_to: {
              kind: "stream_backlog",
              source_entry_ids: ["strm_user_a"],
            },
          }),
        ],
      });
      source.emit(terminalFrame("turn_a", "reflected"));
      source.emit(phaseFrame("turn:phase:started", "ingest", "turn_b"));
    });

    expect(screen.getByText("turn_b")).toBeInTheDocument();
    expect(screen.getByTestId("phase-ingest")).toHaveClass("fc-node-running");
  });

  it("does not add a second optimistic bubble for a duplicate ack", async () => {
    const source = makeLiveSource();
    vi.spyOn(globalThis.crypto, "randomUUID").mockReturnValue(
      "00000000-0000-4000-8000-000000000001",
    );
    const { fetchMock } = installCognitionFetch({
      turnResponses: [
        jsonResponse({
          ok: true,
          status: "enqueued",
          stream_entry_id: "strm_user_original",
        }),
        jsonResponse({
          ok: true,
          status: "duplicate",
          stream_entry_id: "strm_user_original",
        }),
      ],
    });

    renderWithInspector(<Harness live={source.live()} />);

    fireEvent.change(screen.getByPlaceholderText("send a turn"), {
      target: { value: "hello once" },
    });
    fireEvent.click(screen.getByRole("button", { name: "send" }));
    await waitFor(() => expect(screen.getByText("sent")).toBeInTheDocument());

    fireEvent.change(screen.getByPlaceholderText("send a turn"), {
      target: { value: "hello duplicate" },
    });
    fireEvent.click(screen.getByRole("button", { name: "send" }));

    await waitFor(() =>
      expect(
        fetchMock.mock.calls.filter(
          ([request, init]) => String(request).endsWith("/api/turn") && init?.method === "POST",
        ),
      ).toHaveLength(2),
    );

    expect(document.querySelectorAll(".chat-msg.user")).toHaveLength(1);
    expect(screen.getByText("hello once")).toBeInTheDocument();
    expect(screen.queryByText("hello duplicate")).not.toBeInTheDocument();
  });

  it("does not enqueue the same draft twice on double submit", async () => {
    const source = makeLiveSource();
    const pendingTurn = deferredResponse();
    const { fetchMock } = installCognitionFetch({
      turnResponse: pendingTurn.promise,
    });

    renderWithInspector(<Harness live={source.live()} />);

    fireEvent.change(screen.getByPlaceholderText("send a turn"), {
      target: { value: "same draft" },
    });
    const sendButton = screen.getByRole("button", { name: "send" });
    fireEvent.click(sendButton);
    fireEvent.click(sendButton);

    await waitFor(() => expect(turnPostCalls(fetchMock)).toHaveLength(1));
    const body = JSON.parse(String((turnPostCalls(fetchMock)[0]?.[1] as RequestInit).body)) as {
      external_message_id: string;
      message: string;
    };
    expect(body).toMatchObject({
      message: "same draft",
      external_message_id: expect.any(String),
    });

    await act(async () => {
      pendingTurn.resolve(
        jsonResponse({
          ok: true,
          status: "enqueued",
          stream_entry_id: "strm_same_draft",
        }),
      );
      await pendingTurn.promise;
    });
  });

  it("keeps the composer enabled so a second send can enqueue while one is running", async () => {
    const source = makeLiveSource();
    const firstTurn = deferredResponse();
    const secondTurn = deferredResponse();
    const { fetchMock } = installCognitionFetch({
      turnResponses: [firstTurn.promise, secondTurn.promise],
    });

    renderWithInspector(<Harness live={source.live()} />);

    fireEvent.change(screen.getByPlaceholderText("send a turn"), {
      target: { value: "first queued" },
    });
    fireEvent.click(screen.getByRole("button", { name: "send" }));

    await waitFor(() => expect(chatUserBodies()).toContain("first queued"));
    act(() => {
      source.emit(phaseFrame("turn:phase:started", "ingest", "turn_running"));
    });

    const sendButton = screen.getByRole("button", { name: "send" });
    fireEvent.change(screen.getByPlaceholderText("send a turn"), {
      target: { value: "second queued" },
    });
    expect(sendButton).toBeEnabled();
    fireEvent.click(sendButton);

    await waitFor(() =>
      expect(
        fetchMock.mock.calls.filter(
          ([request, init]) => String(request).endsWith("/api/turn") && init?.method === "POST",
        ),
      ).toHaveLength(2),
    );
    expect(chatUserBodies()).toContain("second queued");

    await act(async () => {
      firstTurn.resolve(
        jsonResponse({
          ok: true,
          status: "enqueued",
          stream_entry_id: "strm_first",
        }),
      );
      secondTurn.resolve(
        jsonResponse({
          ok: true,
          status: "enqueued",
          stream_entry_id: "strm_second",
        }),
      );
      await Promise.all([firstTurn.promise, secondTurn.promise]);
    });
  });

  it("keeps the in-flight placeholder after POST resolves until reflect completion", async () => {
    const source = makeLiveSource();
    installCognitionFetch({
      turnResponse: jsonResponse({
        ok: true,
        status: "enqueued",
        stream_entry_id: "strm_user_abc",
      }),
    });

    renderWithInspector(<Harness live={source.live()} />);

    fireEvent.change(screen.getByPlaceholderText("send a turn"), {
      target: { value: "hello borg" },
    });
    fireEvent.click(screen.getByRole("button", { name: "send" }));

    expect(await screen.findByText(/borg is thinking/)).toBeInTheDocument();

    await act(async () => {
      await Promise.resolve();
      await Promise.resolve();
    });

    expect(screen.getByText(/borg is thinking/)).toBeInTheDocument();

    act(() => {
      source.emit(reflectFrame());
    });

    await waitFor(() => expect(screen.queryByText(/borg is thinking/)).not.toBeInTheDocument());
  });

  it("scopes stream fetches and turn posts to the active session", async () => {
    const source = makeLiveSource();
    const { fetchMock } = installCognitionFetch({
      turnResponse: jsonResponse({
        ok: true,
        status: "enqueued",
        stream_entry_id: "strm_user_abc",
      }),
    });

    renderWithInspector(<Harness live={source.live()} sessionId="sess_custom" />);

    await waitFor(() => {
      const streamCall = fetchMock.mock.calls.find(([request]) =>
        String(request).includes("/api/stream"),
      );
      expect(streamCall).toBeDefined();
      const streamUrl = new URL(String(streamCall?.[0]), "http://test.invalid");
      expect(streamUrl.searchParams.get("session")).toBe("sess_custom");
    });

    fireEvent.change(screen.getByPlaceholderText("send a turn"), {
      target: { value: "hello borg" },
    });
    fireEvent.click(screen.getByRole("button", { name: "send" }));

    await waitFor(() => {
      const turnCall = fetchMock.mock.calls.find(
        ([request, init]) => String(request).endsWith("/api/turn") && init?.method === "POST",
      );
      expect(turnCall).toBeDefined();
      const init = turnCall?.[1] as RequestInit;
      expect(JSON.parse(String(init.body))).toEqual({
        message: "hello borg",
        external_message_id: expect.any(String),
        audience: "alice",
        session: "sess_custom",
      });
    });
  });

  it("renders the current participation policy", () => {
    const source = makeLiveSource();
    installCognitionFetch();

    renderWithInspector(
      <Harness live={source.live()} session={sessionRecord({ participation_policy: "muted" })} />,
    );

    expect(screen.getByRole("button", { name: "participation policy muted" })).toHaveTextContent(
      "muted",
    );
    expect(
      screen.getByText(
        (_, element) =>
          element?.classList.contains("participation-policy-line") === true &&
          element.textContent?.includes("will stay silent") === true,
      ),
    ).toBeInTheDocument();
  });

  it("renders recent turn rows in the workbench strip", async () => {
    const source = makeLiveSource();
    installCognitionFetch({
      turnRows: [
        turnRow({
          turn_id: "turn_recent",
          started_at: 2,
          outcome: "guard-blocked",
          suppression_reason: "commitment_violation",
        }),
      ],
    });

    renderWithInspector(<Harness live={source.live()} />);

    expect(await screen.findByText("turn_recent")).toBeInTheDocument();
    expect(screen.getByText("guard-blocked")).toBeInTheDocument();
    expect(screen.getByText("commitment_violation")).toBeInTheDocument();
    expect(screen.getByRole("button", { name: "live" })).toHaveAttribute("aria-pressed", "true");
  });

  it("fetches workbench tab data lazily on first activation and caches it", async () => {
    const source = makeLiveSource();
    const { fetchMock } = installCognitionFetch();

    renderWithInspector(<Harness live={source.live()} />);

    await waitFor(() => expect(fetchPathCalls(fetchMock, "/api/stream")).toHaveLength(1));
    expect(fetchPathCalls(fetchMock, "/api/shared-state")).toHaveLength(0);
    expect(fetchPathCalls(fetchMock, "/api/commitments")).toHaveLength(0);
    expect(fetchPathCalls(fetchMock, "/api/identity")).toHaveLength(0);
    expect(fetchPathCalls(fetchMock, "/api/prompts/assembled")).toHaveLength(0);

    fireEvent.click(screen.getByRole("tab", { name: "shared state" }));
    await waitFor(() => expect(fetchPathCalls(fetchMock, "/api/shared-state")).toHaveLength(1));

    fireEvent.click(screen.getByRole("tab", { name: "flow" }));
    fireEvent.click(screen.getByRole("tab", { name: "shared state" }));
    await act(async () => {
      await Promise.resolve();
    });
    expect(fetchPathCalls(fetchMock, "/api/shared-state")).toHaveLength(1);

    fireEvent.click(screen.getByRole("tab", { name: "commitments" }));
    await waitFor(() => expect(fetchPathCalls(fetchMock, "/api/commitments")).toHaveLength(1));

    fireEvent.click(screen.getByRole("tab", { name: "open qs" }));
    await waitFor(() => expect(fetchPathCalls(fetchMock, "/api/identity")).toHaveLength(1));

    fireEvent.click(screen.getByRole("tab", { name: "prompt" }));
    await waitFor(() =>
      expect(fetchPathCalls(fetchMock, "/api/prompts/assembled")).toHaveLength(1),
    );
  });

  it("selects a cached historical turn for replay with its cached ledger", async () => {
    const source = makeLiveSource();
    installCognitionFetch({
      turnRows: [turnRow({ turn_id: "turn_cached", started_at: 2 })],
    });

    renderWithInspector(<Harness live={source.live()} />);

    act(() => {
      source.emit(phaseFrame("turn:phase:started", "ingest", "turn_cached"));
      source.emit({
        type: "evidence_ledger:built",
        ts: Date.now(),
        session_id: "default",
        turn_id: "turn_cached",
        ledger: ledgerWithText("cached replay ledger"),
      });
      source.emit(terminalFrame("turn_cached", "reflected"));
      source.emit(phaseFrame("turn:phase:started", "final", "turn_live"));
    });

    await screen.findByText("turn_cached");
    expect(screen.getByText("turn_live")).toBeInTheDocument();

    fireEvent.click(screen.getByRole("button", { name: /turn_cached/ }));

    expect(screen.getByLabelText("Replay turn metadata")).toHaveTextContent("replay");
    expect(screen.getAllByText("turn_cached").length).toBeGreaterThan(0);

    fireEvent.click(screen.getByRole("tab", { name: "ledger" }));

    expect(screen.getByText("cached replay ledger")).toBeInTheDocument();
  });

  it("shows unavailable trace copy for an uncached replay and lets the ledger 404 degrade", async () => {
    const source = makeLiveSource();
    installCognitionFetch({
      turnRows: [turnRow({ turn_id: "turn_uncached", started_at: 2 })],
    });

    renderWithInspector(<Harness live={source.live()} />);

    fireEvent.click(await screen.findByRole("button", { name: /turn_uncached/ }));

    expect(screen.getByText("trace unavailable this browser session")).toBeInTheDocument();

    fireEvent.click(screen.getByRole("tab", { name: "ledger" }));

    expect(await screen.findByText(/ledger not retained \(pre-restart\)/)).toBeInTheDocument();
  });

  it("returns from replay to live state without blocking later live updates", async () => {
    const source = makeLiveSource();
    installCognitionFetch({
      turnRows: [turnRow({ turn_id: "turn_cached", started_at: 2 })],
    });

    renderWithInspector(<Harness live={source.live()} />);

    act(() => {
      source.emit(phaseFrame("turn:phase:started", "ingest", "turn_cached"));
      source.emit(terminalFrame("turn_cached", "reflected"));
    });

    fireEvent.click(await screen.findByRole("button", { name: /turn_cached/ }));
    expect(screen.getByLabelText("Replay turn metadata")).toBeInTheDocument();

    act(() => {
      source.emit(phaseFrame("turn:phase:started", "final", "turn_after_replay"));
    });

    expect(screen.getByLabelText("Replay turn metadata")).toHaveTextContent("turn_cached");
    expect(screen.queryByText("turn_after_replay")).not.toBeInTheDocument();

    fireEvent.click(screen.getByRole("button", { name: "live" }));

    expect(screen.queryByLabelText("Replay turn metadata")).not.toBeInTheDocument();
    expect(screen.getByText("turn_after_replay")).toBeInTheDocument();
    expect(screen.getByTestId("phase-final")).toHaveClass("fc-node-running");
  });

  it("posts participation policy changes with a reason", async () => {
    const source = makeLiveSource();
    const { fetchMock } = installCognitionFetch();
    const onSessionPolicyChanged = vi.fn(async () => undefined);

    renderWithInspector(
      <Harness live={source.live()} onSessionPolicyChanged={onSessionPolicyChanged} />,
    );

    fireEvent.click(screen.getByRole("button", { name: "participation policy active" }));
    fireEvent.change(screen.getByLabelText("participation policy selection"), {
      target: { value: "observing" },
    });
    fireEvent.change(screen.getByLabelText("participation policy reason"), {
      target: { value: "  needs space  " },
    });
    fireEvent.click(screen.getByRole("button", { name: "apply" }));

    await waitFor(() => expect(onSessionPolicyChanged).toHaveBeenCalledTimes(1));
    const postCall = fetchMock.mock.calls.find(
      ([request, init]) =>
        String(request).endsWith("/api/sessions/default/participation") && init?.method === "POST",
    );
    expect(postCall).toBeDefined();
    expect(JSON.parse(String((postCall?.[1] as RequestInit).body))).toEqual({
      policy: "observing",
      reason: "needs space",
    });
  });

  it("stages uploaded images and posts turns as multipart form data", async () => {
    const source = makeLiveSource();
    const { fetchMock } = installCognitionFetch({
      turnResponse: jsonResponse({
        ok: true,
        status: "enqueued",
        stream_entry_id: "strm_user_abc",
      }),
    });
    const file = new File([new Uint8Array([1, 2, 3])], "pixel.png", { type: "image/png" });

    renderWithInspector(<Harness live={source.live()} />);

    fireEvent.click(screen.getByRole("button", { name: "+ attach" }));
    fireEvent.change(screen.getByTestId("attachment-file-input"), {
      target: { files: [file] },
    });

    expect(screen.getAllByText("pixel.png").length).toBeGreaterThan(0);

    fireEvent.change(screen.getByPlaceholderText("send a turn"), {
      target: { value: "see this image" },
    });
    fireEvent.click(screen.getByRole("button", { name: "send" }));

    await waitFor(() => {
      const call = fetchMock.mock.calls.find(
        ([request, init]) => String(request).endsWith("/api/turn") && init?.method === "POST",
      );
      expect(call).toBeDefined();
      const init = call?.[1] as RequestInit;
      expect(init.body).toBeInstanceOf(FormData);
      const body = init.body as FormData;
      expect(body.get("message")).toBe("see this image");
      expect(body.get("external_message_id")).toEqual(expect.any(String));
      expect(body.get("audience")).toBe("alice");
      expect(body.get("session")).toBe("default");
      expect(body.getAll("attachments[]")).toEqual([file]);
    });

    await waitFor(() => expect(screen.queryByText("pixel.png")).not.toBeInTheDocument());
  });

  it("clears the in-flight placeholder on terminal turn frames", async () => {
    const source = makeLiveSource();
    installCognitionFetch({
      turnResponse: jsonResponse({
        ok: true,
        status: "enqueued",
        stream_entry_id: "strm_user_abc",
      }),
    });

    renderWithInspector(<Harness live={source.live()} />);

    fireEvent.change(screen.getByPlaceholderText("send a turn"), {
      target: { value: "hello borg" },
    });
    fireEvent.click(screen.getByRole("button", { name: "send" }));

    expect(await screen.findByText(/borg is thinking/)).toBeInTheDocument();

    await act(async () => {
      await Promise.resolve();
      await Promise.resolve();
    });

    act(() => {
      source.emit({
        type: "stream:append",
        ts: Date.now(),
        entries: [
          streamEntry({
            id: "strm_agent_abc",
            kind: "agent_msg",
            turn_id: "turn_abc",
            content: "answer",
            response_to: {
              kind: "stream_backlog",
              source_entry_ids: ["strm_user_abc"],
            },
          }),
        ],
      });
      source.emit(terminalFrame());
    });

    await waitFor(() => expect(screen.queryByText(/borg is thinking/)).not.toBeInTheDocument());
  });

  it("renders all flow phases upfront and tracks state transitions", async () => {
    const source = makeLiveSource();
    installCognitionFetch();

    renderWithInspector(<Harness live={source.live()} />);

    expect(screen.getByTestId("phase-ingest")).toHaveClass("fc-node-queue");
    expect(screen.getByTestId("phase-audience")).toHaveClass("fc-node-queue");
    expect(screen.getByTestId("phase-final")).toHaveClass("fc-node-queue");

    fireEvent.change(screen.getByPlaceholderText("send a turn"), {
      target: { value: "hello borg" },
    });
    fireEvent.click(screen.getByRole("button", { name: "send" }));
    expect(await screen.findByText(/borg is thinking/)).toBeInTheDocument();

    act(() => {
      source.emit(phaseFrame("turn:phase:started", "ingest"));
    });
    expect(screen.getByTestId("phase-ingest")).toHaveClass("fc-node-running");

    act(() => {
      source.emit(phaseFrame("turn:phase:completed", "ingest"));
      source.emit(phaseFrame("turn:phase:started", "audience"));
      source.emit(phaseFrame("turn:phase:failed", "final"));
    });

    expect(screen.getByTestId("phase-ingest")).toHaveClass("fc-node-done");
    expect(screen.getByTestId("phase-audience")).toHaveClass("fc-node-running");
    expect(screen.getByTestId("phase-final")).toHaveClass("fc-node-fail");
  });

  it("renders running non-LLM phase details inside the active stream pane", async () => {
    const source = makeLiveSource();
    installCognitionFetch();

    renderWithInspector(<Harness live={source.live()} />);

    fireEvent.change(screen.getByPlaceholderText("send a turn"), {
      target: { value: "hello borg" },
    });
    fireEvent.click(screen.getByRole("button", { name: "send" }));
    expect(await screen.findByText(/borg is thinking/)).toBeInTheDocument();

    act(() => {
      source.emit(phaseFrame("turn:phase:started", "retrieval"));
      source.emit({
        type: "turn:phase:detail",
        ts: Date.now(),
        turn_id: "turn_abc",
        session_id: "default",
        phase: "retrieval",
        event: "retrieval.completed",
        summary: "episodeCount=2 semanticHits=4 confidence=0.82",
      });
    });

    expect(document.querySelector(".flow-active-head")?.textContent).toMatch(/retrieval/i);
    expect(document.querySelector(".flow-active-body")?.textContent).toContain(
      "retrieval.completed · episodeCount=2 semanticHits=4 confidence=0.82",
    );
  });

  it("renders accumulated token text inside the active streaming phase", async () => {
    const source = makeLiveSource();
    installCognitionFetch();

    renderWithInspector(<Harness live={source.live()} />);

    fireEvent.change(screen.getByPlaceholderText("send a turn"), {
      target: { value: "hello borg" },
    });
    fireEvent.click(screen.getByRole("button", { name: "send" }));
    expect(await screen.findByText(/borg is thinking/)).toBeInTheDocument();

    act(() => {
      source.emit(phaseFrame("turn:phase:started", "final"));
      source.emit({
        type: "turn:token",
        ts: Date.now(),
        turn_id: "turn_abc",
        phase: "final",
        chunk_text: "Hello ",
        sequence: 1,
      });
      source.emit({
        type: "turn:token",
        ts: Date.now(),
        turn_id: "turn_abc",
        phase: "final",
        chunk_text: "world",
        sequence: 2,
      });
    });

    // Token text now lives in the active-stream pane below the pipeline,
    // not inside the phase pill itself. The active phase's name shows in
    // the head label so we can disambiguate which phase is being streamed.
    expect(document.querySelector(".flow-active-body")?.textContent).toContain("Hello world");
    expect(document.querySelector(".flow-active-head")?.textContent).toMatch(/finalizer/i);
  });

  it("keeps the active-stream pane showing final text after the phase completes", async () => {
    const source = makeLiveSource();
    installCognitionFetch();

    renderWithInspector(<Harness live={source.live()} />);

    fireEvent.change(screen.getByPlaceholderText("send a turn"), {
      target: { value: "hello borg" },
    });
    fireEvent.click(screen.getByRole("button", { name: "send" }));
    expect(await screen.findByText(/borg is thinking/)).toBeInTheDocument();

    act(() => {
      source.emit(phaseFrame("turn:phase:started", "final"));
      source.emit({
        type: "turn:token",
        ts: Date.now(),
        turn_id: "turn_abc",
        phase: "final",
        chunk_text: "Settled answer.",
        sequence: 1,
      });
      source.emit(phaseFrame("turn:phase:completed", "final"));
    });

    expect(document.querySelector(".flow-active-body")?.textContent).toContain("Settled answer.");
    expect(document.querySelector(".flow-active-body")?.className).toMatch(/muted/);
  });

  it("ignores stale turn frames after a newer turn starts", async () => {
    const source = makeLiveSource();
    installCognitionFetch({
      turnResponse: jsonResponse({
        ok: true,
        status: "enqueued",
        stream_entry_id: "strm_user_new",
      }),
    });

    renderWithInspector(<Harness live={source.live()} />);

    fireEvent.change(screen.getByPlaceholderText("send a turn"), {
      target: { value: "hello borg" },
    });
    fireEvent.click(screen.getByRole("button", { name: "send" }));
    expect(await screen.findByText(/borg is thinking/)).toBeInTheDocument();

    act(() => {
      source.emit(phaseFrame("turn:phase:started", "final", "turn_new"));
      source.emit({
        type: "turn:token",
        ts: Date.now(),
        turn_id: "turn_new",
        phase: "final",
        chunk_text: "fresh",
        sequence: 1,
      });
      source.emit({
        type: "turn:token",
        ts: Date.now(),
        turn_id: "turn_old",
        phase: "final",
        chunk_text: "stale",
        sequence: 2,
      });
      source.emit(phaseFrame("turn:phase:failed", "final", "turn_old"));
      source.emit(terminalFrame("turn_old", "error"));
    });

    expect(document.querySelector(".flow-active-body")?.textContent).toContain("fresh");
    expect(document.querySelector(".flow-active-body")?.textContent).not.toContain("stale");
    expect(screen.getByTestId("phase-final")).toHaveClass("fc-node-running");
    expect(screen.queryByText(/terminal error/)).not.toBeInTheDocument();
  });

  it("merges initial fetch results over live stream appends", async () => {
    const source = makeLiveSource();
    let resolveStream: ((response: Response) => void) | undefined;
    const streamResponse = new Promise<Response>((resolve) => {
      resolveStream = resolve;
    });
    const fetchMock = vi.fn((request: RequestInfo | URL) => {
      const url = String(request);
      if (url.includes("/api/stream")) {
        return streamResponse;
      }
      if (url.endsWith("/api/shared-state?audience=alice")) {
        return Promise.resolve(jsonResponse({ audience: "alice", entries: [] }));
      }
      return Promise.resolve(new Response("{}", { status: 404 }));
    });
    vi.stubGlobal("fetch", fetchMock);
    const liveEntry = streamEntry({
      id: "strm_live",
      timestamp: 2,
      entry_index: 2,
      kind: "agent_msg",
      content: "live kept",
    });

    renderWithInspector(<Harness live={source.live()} />);

    await act(async () => {
      await Promise.resolve();
    });
    act(() => {
      source.emit({
        type: "stream:append",
        ts: 2,
        entries: [liveEntry],
      });
    });

    expect(await screen.findByText("live kept")).toBeInTheDocument();

    await act(async () => {
      resolveStream?.(
        jsonResponse({
          entries: [
            streamEntry({
              id: "strm_snapshot",
              timestamp: 1,
              entry_index: 1,
              kind: "user_msg",
              content: "snapshot",
            }),
          ],
          next_cursor: null,
        }),
      );
      await Promise.resolve();
    });

    expect(screen.getByText("live kept")).toBeInTheDocument();
    expect(screen.getByText("snapshot")).toBeInTheDocument();
  });

  it("does not refetch the stream on the first WebSocket connection", async () => {
    const source = makeLiveSource();
    const { streamCalls } = installCognitionFetch();

    const { rerender } = renderWithInspector(<Harness live={source.live(0, "reconnecting")} />);

    await waitFor(() => expect(streamCalls()).toBe(1));

    rerender(<Harness live={source.live(1, "live")} />);

    await act(async () => {
      await Promise.resolve();
      await Promise.resolve();
    });

    expect(streamCalls()).toBe(1);
  });

  it("rebuilds the tail from stream entries on WebSocket reconnect", async () => {
    const source = makeLiveSource();
    const rebuiltEntry = streamEntry({
      id: "strm_rebuilt",
      timestamp: 10,
      kind: "agent_msg",
      content: "tail rebuilt",
    });
    const { streamCalls } = installCognitionFetch({
      streamEntries: [[], [rebuiltEntry]],
    });

    const { rerender } = renderWithInspector(<Harness live={source.live(1, "live")} />);

    await waitFor(() => expect(streamCalls()).toBe(1));

    rerender(<Harness live={source.live(2, "live")} />);

    await waitFor(() => expect(streamCalls()).toBe(2));
    expect(await screen.findByText("tail rebuilt")).toBeInTheDocument();
  });
});
