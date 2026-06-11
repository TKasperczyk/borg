import { act, render } from "@testing-library/react";
import { StrictMode, useEffect } from "react";

import type { LiveFrame } from "../api/types";
import { useQuery } from "../api/useQuery";
import { installMockWebSocket } from "../__tests__/mock-websocket";
import { LiveProvider, useLive, type LiveStatus } from "./useLive";

function LiveProbe({
  onReady,
  onToken,
}: {
  onReady?: (live: ReturnType<typeof useLive>) => void;
  onToken?: (frame: LiveFrame) => void;
}) {
  const live = useLive();

  useEffect(() => {
    onReady?.(live);
  }, [live, onReady]);

  useEffect(() => {
    if (onToken === undefined) {
      return undefined;
    }

    return live.onFrame("turn:token", onToken);
  }, [live, onToken]);

  return <div>{live.status}</div>;
}

function QueryProbe({ fn }: { fn: () => Promise<number> }) {
  const query = useQuery("state", fn);
  return <div>{query.data}</div>;
}

function MultiHandlerProbe({
  first,
  second,
  wildcard,
}: {
  first: (frame: LiveFrame) => void;
  second: (frame: LiveFrame) => void;
  wildcard: (frame: LiveFrame) => void;
}) {
  const live = useLive();

  useEffect(() => {
    const unsubscribeFirst = live.onFrame("turn:token", first);
    const unsubscribeSecond = live.onFrame("turn:token", second);
    const unsubscribeWildcard = live.onFrame("*", wildcard);

    return () => {
      unsubscribeFirst();
      unsubscribeSecond();
      unsubscribeWildcard();
    };
  }, [first, live, second, wildcard]);

  return null;
}

describe("LiveProvider", () => {
  beforeEach(() => {
    vi.spyOn(Math, "random").mockReturnValue(0);
  });

  afterEach(() => {
    vi.useRealTimers();
    vi.restoreAllMocks();
    vi.unstubAllGlobals();
  });

  it("fans out frames by type", () => {
    const { instances } = installMockWebSocket();
    const onToken = vi.fn();

    render(
      <LiveProvider>
        <LiveProbe onToken={onToken} />
      </LiveProvider>,
    );

    act(() => {
      instances[0]!.open();
      instances[0]!.receive({
        type: "turn:token",
        ts: 1,
        turn_id: "t1",
        phase: "final",
        chunk_text: "hi",
        sequence: 1,
      });
    });

    expect(onToken).toHaveBeenCalledWith(
      expect.objectContaining({ type: "turn:token", chunk_text: "hi" }),
    );
  });

  it("continues fan-out when one frame handler throws", () => {
    const { instances } = installMockWebSocket();
    const first = vi.fn(() => {
      throw new Error("handler failed");
    });
    const second = vi.fn();
    const wildcard = vi.fn();
    const errorSpy = vi.spyOn(console, "error").mockImplementation(() => {});
    const frame: LiveFrame = {
      type: "turn:token",
      ts: 1,
      turn_id: "t1",
      phase: "final",
      chunk_text: "hi",
      sequence: 1,
    };

    render(
      <LiveProvider>
        <MultiHandlerProbe first={first} second={second} wildcard={wildcard} />
      </LiveProvider>,
    );

    act(() => {
      instances[0]!.open();
      instances[0]!.receive(frame);
    });

    expect(first).toHaveBeenCalledWith(frame);
    expect(second).toHaveBeenCalledWith(frame);
    expect(wildcard).toHaveBeenCalledWith(frame);
    expect(errorSpy).toHaveBeenCalledWith(
      "Live frame handler failed",
      expect.objectContaining({ type: "turn:token" }),
    );
  });

  it("ignores stale close events from a superseded StrictMode socket", () => {
    vi.useFakeTimers();
    const { instances } = installMockWebSocket();
    let status: LiveStatus | undefined;

    render(
      <StrictMode>
        <LiveProvider>
          <LiveProbe onReady={(live) => (status = live.status)} />
        </LiveProvider>
      </StrictMode>,
    );

    expect(instances).toHaveLength(2);

    act(() => {
      instances[1]!.open();
    });
    expect(status).toBe("open");

    act(() => {
      instances[0]!.close();
      vi.advanceTimersByTime(1_000);
    });

    expect(status).toBe("open");
    expect(instances).toHaveLength(2);
  });

  it("debounces invalidations for terminal bursts", async () => {
    vi.useFakeTimers();
    const { instances } = installMockWebSocket();
    let count = 0;
    const fn = vi.fn(async () => {
      count += 1;
      return count;
    });

    render(
      <LiveProvider>
        <QueryProbe fn={fn} />
      </LiveProvider>,
    );

    await act(async () => {});
    expect(fn).toHaveBeenCalledTimes(1);

    act(() => {
      instances[0]!.receive({
        type: "turn:terminal",
        ts: 1,
        event: "turn.terminal",
        data: {
          turnId: "t1",
          turn_id: "t1",
          session_id: "default",
          outcome: "reflected",
          ts: 1,
          duration_ms: 10,
        },
      });
      instances[0]!.receive({
        type: "turn:terminal",
        ts: 2,
        event: "turn.terminal",
        data: {
          turnId: "t2",
          turn_id: "t2",
          session_id: "default",
          outcome: "reflected",
          ts: 2,
          duration_ms: 11,
        },
      });
      vi.advanceTimersByTime(299);
    });
    expect(fn).toHaveBeenCalledTimes(1);

    await act(async () => {
      vi.advanceTimersByTime(1);
    });
    expect(fn).toHaveBeenCalledTimes(2);
  });

  it("does not invalidate on token frames", async () => {
    vi.useFakeTimers();
    const { instances } = installMockWebSocket();
    let count = 0;
    const fn = vi.fn(async () => {
      count += 1;
      return count;
    });

    render(
      <LiveProvider>
        <QueryProbe fn={fn} />
      </LiveProvider>,
    );

    await act(async () => {});
    expect(fn).toHaveBeenCalledTimes(1);

    act(() => {
      instances[0]!.receive({
        type: "turn:token",
        ts: 1,
        turn_id: "t1",
        phase: "delib",
        chunk_text: "x",
        sequence: 1,
      });
      vi.advanceTimersByTime(500);
    });

    expect(fn).toHaveBeenCalledTimes(1);
  });

  it("resubscribes held sessions after reconnect", () => {
    vi.useFakeTimers();
    const { instances } = installMockWebSocket();
    let live: ReturnType<typeof useLive> | undefined;

    render(
      <LiveProvider>
        <LiveProbe onReady={(value) => (live = value)} />
      </LiveProvider>,
    );

    act(() => {
      live!.subscribeSession("s_demo");
      instances[0]!.open();
    });
    expect(instances[0]!.sent).toContain(JSON.stringify({ type: "subscribe", session_id: "s_demo" }));

    act(() => {
      instances[0]!.close();
      vi.advanceTimersByTime(500);
    });
    expect(instances).toHaveLength(2);

    act(() => {
      instances[1]!.open();
    });
    expect(instances[1]!.sent).toContain(JSON.stringify({ type: "subscribe", session_id: "s_demo" }));
  });

  it("exposes closed status while reconnecting", () => {
    vi.useFakeTimers();
    const { instances } = installMockWebSocket();
    let status: LiveStatus | undefined;

    render(
      <LiveProvider>
        <LiveProbe onReady={(live) => (status = live.status)} />
      </LiveProvider>,
    );

    act(() => {
      instances[0]!.close();
    });

    expect(status).toBe("closed");
  });
});
