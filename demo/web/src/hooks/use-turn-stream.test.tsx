import { act, fireEvent, render, screen, waitFor } from "@testing-library/react";
import { afterEach, describe, expect, it, vi } from "vitest";

import type { LiveFrame, WsState } from "../api/types";
import type { LiveEventHandler, LiveEvents } from "./use-live-events";
import { useTurnStream } from "./use-turn-stream";

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

function DetailProbe({ live }: { live: LiveEvents }) {
  const turnStream = useTurnStream(live, { sessionId: "default" });
  const detailLines = [...turnStream.detailByPhase.values()].flat();

  return (
    <>
      <button
        type="button"
        onClick={() => {
          void turnStream.runTurn({
            message: "hello borg",
            external_message_id: "detail-test-message",
            audience: "alice",
            session: "default",
          });
        }}
      >
        start
      </button>
      <pre data-testid="details">{detailLines.join("\n")}</pre>
    </>
  );
}

afterEach(() => {
  vi.restoreAllMocks();
  vi.unstubAllGlobals();
});

describe("useTurnStream", () => {
  it("accumulates active turn phase detail frames by phase", async () => {
    const source = makeLiveSource();
    const fetchMock = vi.fn(() =>
      Promise.resolve(
        jsonResponse({ ok: true, status: "enqueued", stream_entry_id: "strm_user_abc" }),
      ),
    );
    vi.stubGlobal("fetch", fetchMock);

    render(<DetailProbe live={source.live()} />);

    fireEvent.click(screen.getByRole("button", { name: "start" }));
    await waitFor(() => expect(fetchMock).toHaveBeenCalled());

    act(() => {
      source.emit({
        type: "turn:phase:detail",
        ts: Date.now(),
        turn_id: "turn_hook",
        session_id: "default",
        phase: "retrieval",
        event: "retrieval.completed",
        summary: "episodeCount=2 semanticHits=4 confidence=0.82",
      });
    });

    await waitFor(() =>
      expect(screen.getByTestId("details")).toHaveTextContent(
        "retrieval.completed · episodeCount=2 semanticHits=4 confidence=0.82",
      ),
    );
  });
});
