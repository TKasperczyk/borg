import { act, fireEvent, render, screen, waitFor } from "@testing-library/react";
import { afterEach, describe, expect, it, vi } from "vitest";

import type {
  EvidenceLedger,
  LiveFrame,
  StreamEntry,
  TurnTerminalOutcome,
  TurnPhaseName,
  WsState,
} from "../../api/types";
import { LiveEventsProvider } from "../../hooks/live-context";
import type { LiveEventHandler, LiveEvents } from "../../hooks/use-live-events";
import { useTurnStream } from "../../hooks/use-turn-stream";
import { ChatStream } from "./ChatStream";
import { CognitionScreen } from "./index";
import { LedgerView } from "./LedgerView";

function jsonResponse(body: unknown): Response {
  return new Response(JSON.stringify(body), {
    status: 200,
    headers: { "Content-Type": "application/json" },
  });
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

function installCognitionFetch(
  input: {
    streamEntries?: StreamEntry[][];
    turnResponse?: Response | Promise<Response>;
  } = {},
): { streamCalls: () => number } {
  const streamResponses = input.streamEntries ?? [[]];
  let streamCallCount = 0;
  const fetchMock = vi.fn((request: RequestInfo | URL, init?: RequestInit) => {
    const url = String(request);
    if (url.includes("/api/stream")) {
      const entries = streamResponses[Math.min(streamCallCount, streamResponses.length - 1)];
      streamCallCount += 1;
      return Promise.resolve(jsonResponse({ entries, next_cursor: null }));
    }
    if (url.endsWith("/api/shared-state?audience=alice")) {
      return Promise.resolve(jsonResponse({ audience: "alice", entries: [] }));
    }
    if (url.endsWith("/api/turn") && init?.method === "POST") {
      return Promise.resolve(input.turnResponse ?? jsonResponse({ ok: true, turn_id: "turn_abc" }));
    }
    return Promise.resolve(new Response("{}", { status: 404 }));
  });
  vi.stubGlobal("fetch", fetchMock);
  return {
    streamCalls: () => streamCallCount,
  };
}

function reflectFrame(turnId = "turn_abc"): LiveFrame {
  return {
    type: "turn:phase:completed",
    event: "turn_phase.completed",
    ts: Date.now(),
    data: {
      turnId,
      turn_id: turnId,
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
    data: {
      turnId,
      turn_id: turnId,
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
    data: {
      turnId,
      turn_id: turnId,
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

function Harness({ live }: { live: LiveEvents }) {
  const turnStream = useTurnStream(live);
  return (
    <LiveEventsProvider value={live}>
      <CognitionScreen sessionId="default" audience="alice" turnStream={turnStream} />
    </LiveEventsProvider>
  );
}

afterEach(() => {
  vi.unstubAllGlobals();
});

describe("cognition screen", () => {
  it("renders existing image attachments as chips inside the user turn", () => {
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
        content: { type: "image_ref", attachment_id: "att_123", media_type: "image/png" },
      }),
    ];

    render(<ChatStream entries={entries} sessionId="default" audience="alice" running={false} />);

    expect(screen.getByText("see this")).toBeInTheDocument();
    expect(screen.getByText("[att:att_123]")).toBeInTheDocument();
  });

  it("does not render a stale cached ledger after switching turns", () => {
    const fetchMock = vi.fn(() => new Promise<Response>(() => undefined));
    vi.stubGlobal("fetch", fetchMock);
    const { rerender } = render(
      <LedgerView turnId="turn_a" cachedLedger={ledgerWithText("ledger A")} active />,
    );

    expect(screen.getByText("ledger A")).toBeInTheDocument();

    rerender(<LedgerView turnId="turn_b" cachedLedger={undefined} active />);

    expect(screen.queryByText("ledger A")).not.toBeInTheDocument();
    expect(screen.getByText("ledger not loaded yet")).toBeInTheDocument();
  });

  it("keeps the in-flight placeholder after POST resolves until reflect completion", async () => {
    const source = makeLiveSource();
    installCognitionFetch({
      turnResponse: jsonResponse({ ok: true, turn_id: "turn_abc" }),
    });

    render(<Harness live={source.live()} />);

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

  it("clears the in-flight placeholder on terminal turn frames", async () => {
    const source = makeLiveSource();
    installCognitionFetch({
      turnResponse: jsonResponse({ ok: true, turn_id: "turn_abc" }),
    });

    render(<Harness live={source.live()} />);

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
      source.emit(terminalFrame());
    });

    await waitFor(() => expect(screen.queryByText(/borg is thinking/)).not.toBeInTheDocument());
  });

  it("renders all flow phases upfront and tracks state transitions", () => {
    const source = makeLiveSource();
    installCognitionFetch();

    render(<Harness live={source.live()} />);

    expect(screen.getByTestId("phase-ingest")).toHaveClass("queue");
    expect(screen.getByTestId("phase-audience")).toHaveClass("queue");
    expect(screen.getByTestId("phase-final")).toHaveClass("queue");

    act(() => {
      source.emit(phaseFrame("turn:phase:started", "ingest"));
    });
    expect(screen.getByTestId("phase-ingest")).toHaveClass("running");

    act(() => {
      source.emit(phaseFrame("turn:phase:completed", "ingest"));
      source.emit(phaseFrame("turn:phase:started", "audience"));
      source.emit(phaseFrame("turn:phase:failed", "final"));
    });

    expect(screen.getByTestId("phase-ingest")).toHaveClass("done");
    expect(screen.getByTestId("phase-audience")).toHaveClass("running");
    expect(screen.getByTestId("phase-final")).toHaveClass("fail");
  });

  it("renders accumulated token text inside the active streaming phase", () => {
    const source = makeLiveSource();
    installCognitionFetch();

    render(<Harness live={source.live()} />);

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

    render(<Harness live={source.live()} />);

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

    const { rerender } = render(<Harness live={source.live(0, "reconnecting")} />);

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

    const { rerender } = render(<Harness live={source.live(1, "live")} />);

    await waitFor(() => expect(streamCalls()).toBe(1));

    rerender(<Harness live={source.live(2, "live")} />);

    await waitFor(() => expect(streamCalls()).toBe(2));
    expect(await screen.findByText("tail rebuilt")).toBeInTheDocument();
  });
});
