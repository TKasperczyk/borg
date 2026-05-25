import { act, fireEvent, render, screen, waitFor } from "@testing-library/react";
import { afterEach, describe, expect, it, vi } from "vitest";

import type { LiveFrame, StreamEntry, WsState } from "../../api/types";
import { LiveEventsProvider } from "../../hooks/live-context";
import type { LiveEventHandler, LiveEvents } from "../../hooks/use-live-events";
import { useTurnStream } from "../../hooks/use-turn-stream";
import { ChatStream } from "./ChatStream";
import { CognitionScreen } from "./index";

function jsonResponse(body: unknown): Response {
  return new Response(JSON.stringify(body), {
    status: 200,
    headers: { "Content-Type": "application/json" }
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
      }
    }),
    emit: (frame) => {
      for (const handler of handlers) {
        handler(frame);
      }
    }
  };
}

function streamEntry(input: Partial<StreamEntry> & Pick<StreamEntry, "id" | "kind" | "content">): StreamEntry {
  return {
    timestamp: 1,
    turn_id: "turn_1",
    audience: "alice",
    sender_entity_id: null,
    reply_target_entity_id: null,
    session_id: "default",
    compressed: false,
    ...input
  };
}

function installCognitionFetch(input: {
  streamEntries?: StreamEntry[][];
  turnResponse?: Response | Promise<Response>;
} = {}): { streamCalls: () => number } {
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
    streamCalls: () => streamCallCount
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
      sub: "done"
    }
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
        id: "strm_user",
        timestamp: 1,
        kind: "user_msg",
        content: "see this"
      }),
      streamEntry({
        id: "strm_att",
        timestamp: 2,
        kind: "user_image_attachment",
        content: { type: "image_ref", attachment_id: "att_123", media_type: "image/png" }
      })
    ];

    render(<ChatStream entries={entries} sessionId="default" audience="alice" running={false} />);

    expect(screen.getByText("see this")).toBeInTheDocument();
    expect(screen.getByText("[att:att_123]")).toBeInTheDocument();
  });

  it("keeps the in-flight placeholder after POST resolves until reflect completion", async () => {
    const source = makeLiveSource();
    installCognitionFetch({
      turnResponse: jsonResponse({ ok: true, turn_id: "turn_abc" })
    });

    render(<Harness live={source.live()} />);

    fireEvent.change(screen.getByPlaceholderText("send a turn"), {
      target: { value: "hello borg" }
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
      content: "tail rebuilt"
    });
    const { streamCalls } = installCognitionFetch({
      streamEntries: [[], [rebuiltEntry]]
    });

    const { rerender } = render(<Harness live={source.live(1, "live")} />);

    await waitFor(() => expect(streamCalls()).toBe(1));

    rerender(<Harness live={source.live(2, "live")} />);

    await waitFor(() => expect(streamCalls()).toBe(2));
    fireEvent.click(screen.getByText("tail"));

    expect(await screen.findByText("agent_msg · tail rebuilt")).toBeInTheDocument();
  });
});
