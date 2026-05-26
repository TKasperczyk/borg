import { act, render, screen } from "@testing-library/react";
import { afterEach, describe, expect, it, vi } from "vitest";

import { useLiveEvents } from "./use-live-events";

class FakeWebSocket extends EventTarget {
  static sockets: FakeWebSocket[] = [];

  constructor(readonly url: string) {
    super();
    FakeWebSocket.sockets.push(this);
  }

  open(): void {
    this.dispatchEvent(new Event("open"));
  }

  close(): void {
    this.dispatchEvent(new Event("close"));
  }
}

function installFakeWebSocket(): void {
  FakeWebSocket.sockets = [];
  vi.stubGlobal("WebSocket", FakeWebSocket);
  vi.spyOn(Math, "random").mockReturnValue(0);
}

function Probe({
  onLive,
  onReconnected,
}: {
  onLive?: (live: ReturnType<typeof useLiveEvents>) => void;
  onReconnected?: () => void;
}) {
  const live = useLiveEvents({ onReconnected });
  onLive?.(live);
  return (
    <div data-testid="ws-state">
      {live.wsState}:{live.connectionCount}
    </div>
  );
}

afterEach(() => {
  vi.unstubAllGlobals();
  vi.restoreAllMocks();
  vi.useRealTimers();
});

describe("useLiveEvents", () => {
  it("increments connectionCount and treats the first open as initial connect", () => {
    vi.useFakeTimers();
    installFakeWebSocket();
    const onReconnected = vi.fn();

    render(<Probe onReconnected={onReconnected} />);

    expect(screen.getByTestId("ws-state")).toHaveTextContent("reconnecting:0");

    act(() => {
      FakeWebSocket.sockets[0]?.open();
    });

    expect(screen.getByTestId("ws-state")).toHaveTextContent("live:1");
    expect(onReconnected).not.toHaveBeenCalled();

    act(() => {
      FakeWebSocket.sockets[0]?.close();
    });

    expect(screen.getByTestId("ws-state")).toHaveTextContent("reconnecting:1");

    act(() => {
      vi.advanceTimersByTime(250);
      FakeWebSocket.sockets[1]?.open();
    });

    expect(screen.getByTestId("ws-state")).toHaveTextContent("live:2");
    expect(onReconnected).toHaveBeenCalledTimes(1);
  });

  it("marks the socket down after repeated failed reconnect attempts", () => {
    vi.useFakeTimers();
    installFakeWebSocket();

    render(<Probe />);

    for (let failure = 1; failure <= 5; failure += 1) {
      act(() => {
        FakeWebSocket.sockets.at(-1)?.close();
      });

      if (failure < 5) {
        expect(screen.getByTestId("ws-state")).toHaveTextContent("reconnecting:0");
        act(() => {
          vi.advanceTimersByTime(10_000);
        });
      }
    }

    expect(screen.getByTestId("ws-state")).toHaveTextContent("down:0");
  });

  it("returns a stable object across unrelated rerenders", () => {
    vi.useFakeTimers();
    installFakeWebSocket();
    const seen: ReturnType<typeof useLiveEvents>[] = [];
    const { rerender } = render(<Probe onLive={(live) => seen.push(live)} />);

    rerender(<Probe onLive={(live) => seen.push(live)} />);

    expect(seen[1]).toBe(seen[0]);

    act(() => {
      FakeWebSocket.sockets[0]?.open();
    });

    expect(seen.at(-1)).not.toBe(seen[0]);
  });
});
