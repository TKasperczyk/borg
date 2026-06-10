import { act, render, screen } from "@testing-library/react";
import { afterEach, describe, expect, it, vi } from "vitest";

import { ApiError } from "../api/client";
import { useApi, type UseApiOptions } from "./use-api";

type Deferred<T> = {
  promise: Promise<T>;
  resolve: (value: T) => void;
  reject: (reason?: unknown) => void;
};

function deferred<T>(): Deferred<T> {
  let resolve: (value: T) => void = () => undefined;
  let reject: (reason?: unknown) => void = () => undefined;
  const promise = new Promise<T>((promiseResolve, promiseReject) => {
    resolve = promiseResolve;
    reject = promiseReject;
  });

  return { promise, resolve, reject };
}

function Probe({ loader, options }: { loader: () => Promise<string>; options?: UseApiOptions }) {
  const api = useApi(loader, [loader], options);

  return (
    <div>
      <div data-testid="data">{api.data ?? "none"}</div>
      <div data-testid="error">{api.error?.message ?? "none"}</div>
      <div data-testid="stale">{api.isStale ? "stale" : "fresh"}</div>
      <div data-testid="degraded">{api.degraded ? "degraded" : "healthy"}</div>
      <div data-testid="retrying">{api.retrying ? "retrying" : "idle"}</div>
      <button onClick={() => void api.refetch()}>refetch</button>
    </div>
  );
}

function ParamProbe({
  loader,
  options,
  param,
}: {
  loader: (param: string) => Promise<string>;
  options?: UseApiOptions;
  param: string;
}) {
  const api = useApi(() => loader(param), [loader, param], options);

  return (
    <div>
      <div data-testid="data">{api.data ?? "none"}</div>
      <div data-testid="error">{api.error?.message ?? "none"}</div>
    </div>
  );
}

function KeyProbe({
  loader,
  revalidateKey,
}: {
  loader: () => Promise<string>;
  revalidateKey: number;
}) {
  const api = useApi(loader, [loader], {
    retry: { initialDelayMs: 500, jitterMs: 0 },
    revalidateKey,
  });

  return (
    <div>
      <div data-testid="data">{api.data ?? "none"}</div>
      <div data-testid="error">{api.error?.message ?? "none"}</div>
    </div>
  );
}

afterEach(() => {
  vi.useRealTimers();
  vi.restoreAllMocks();
});

describe("useApi", () => {
  it("coalesces overlapping refetches into one trailing request", async () => {
    const first = deferred<string>();
    const second = deferred<string>();
    const third = deferred<string>();
    const requests = [first, second, third];
    const loader = vi.fn(() => requests.shift()?.promise ?? Promise.resolve("fallback"));

    render(<Probe loader={loader} options={{ retry: true }} />);

    act(() => {
      first.resolve("initial");
    });
    expect(await screen.findByText("initial")).toBeInTheDocument();

    act(() => {
      screen.getByRole("button", { name: "refetch" }).click();
      screen.getByRole("button", { name: "refetch" }).click();
    });

    expect(loader).toHaveBeenCalledTimes(2);

    await act(async () => {
      second.resolve("intermediate");
      await Promise.resolve();
    });

    expect(loader).toHaveBeenCalledTimes(3);

    await act(async () => {
      third.resolve("newer");
      await Promise.resolve();
    });

    expect(screen.getByTestId("data")).toHaveTextContent("newer");
  });

  it("retries retryable transport errors and keeps degraded stale data during recovery", async () => {
    vi.useFakeTimers();
    vi.spyOn(Math, "random").mockReturnValue(0);
    const recovered = deferred<string>();
    const loader = vi
      .fn<() => Promise<string>>()
      .mockResolvedValueOnce("initial")
      .mockRejectedValueOnce(new ApiError({ status: 502, message: "Bad Gateway" }))
      .mockReturnValueOnce(recovered.promise);

    render(<Probe loader={loader} options={{ retry: { initialDelayMs: 500, jitterMs: 0 } }} />);

    await act(async () => {
      await Promise.resolve();
    });

    expect(screen.getByTestId("data")).toHaveTextContent("initial");

    await act(async () => {
      screen.getByRole("button", { name: "refetch" }).click();
      await Promise.resolve();
    });

    expect(screen.getByTestId("data")).toHaveTextContent("initial");
    expect(screen.getByTestId("error")).toHaveTextContent("Bad Gateway");
    expect(screen.getByTestId("stale")).toHaveTextContent("stale");
    expect(screen.getByTestId("degraded")).toHaveTextContent("degraded");
    expect(screen.getByTestId("retrying")).toHaveTextContent("retrying");

    await act(async () => {
      await vi.advanceTimersByTimeAsync(500);
    });

    expect(loader).toHaveBeenCalledTimes(3);
    expect(screen.getByTestId("data")).toHaveTextContent("initial");
    expect(screen.getByTestId("error")).toHaveTextContent("Bad Gateway");
    expect(screen.getByTestId("degraded")).toHaveTextContent("degraded");
    expect(screen.getByTestId("retrying")).toHaveTextContent("retrying");

    await act(async () => {
      recovered.resolve("recovered");
      await Promise.resolve();
    });

    expect(screen.getByTestId("data")).toHaveTextContent("recovered");
    expect(screen.getByTestId("error")).toHaveTextContent("none");
    expect(screen.getByTestId("stale")).toHaveTextContent("fresh");
    expect(screen.getByTestId("degraded")).toHaveTextContent("healthy");
    expect(screen.getByTestId("retrying")).toHaveTextContent("idle");
    expect(loader).toHaveBeenCalledTimes(3);
  });

  it("does not retry 4xx api errors", async () => {
    vi.useFakeTimers();
    vi.spyOn(Math, "random").mockReturnValue(0);
    const loader = vi
      .fn<() => Promise<string>>()
      .mockRejectedValue(new ApiError({ status: 404, message: "Not Found" }));

    render(<Probe loader={loader} options={{ retry: { initialDelayMs: 500, jitterMs: 0 } }} />);

    await act(async () => {
      await Promise.resolve();
    });

    expect(screen.getByTestId("error")).toHaveTextContent("Not Found");
    expect(screen.getByTestId("degraded")).toHaveTextContent("degraded");
    expect(screen.getByTestId("retrying")).toHaveTextContent("idle");

    await act(async () => {
      await vi.advanceTimersByTimeAsync(5_000);
    });

    expect(loader).toHaveBeenCalledTimes(1);
  });

  it("backs off exponentially and caps retry delay", async () => {
    vi.useFakeTimers();
    vi.spyOn(Math, "random").mockReturnValue(0);
    const loader = vi
      .fn<() => Promise<string>>()
      .mockRejectedValue(new ApiError({ status: 503, message: "Unavailable" }));

    render(
      <Probe
        loader={loader}
        options={{ retry: { initialDelayMs: 10, maxDelayMs: 20, jitterMs: 0 } }}
      />,
    );

    await act(async () => {
      await Promise.resolve();
    });

    expect(loader).toHaveBeenCalledTimes(1);

    await act(async () => {
      await vi.advanceTimersByTimeAsync(10);
    });
    expect(loader).toHaveBeenCalledTimes(2);

    await act(async () => {
      await vi.advanceTimersByTimeAsync(19);
    });
    expect(loader).toHaveBeenCalledTimes(2);

    await act(async () => {
      await vi.advanceTimersByTimeAsync(1);
    });
    expect(loader).toHaveBeenCalledTimes(3);

    await act(async () => {
      await vi.advanceTimersByTimeAsync(20);
    });
    expect(loader).toHaveBeenCalledTimes(4);
  });

  it("caps the final jittered retry delay", async () => {
    vi.useFakeTimers();
    vi.spyOn(Math, "random").mockReturnValue(0.99);
    const loader = vi
      .fn<() => Promise<string>>()
      .mockRejectedValue(new ApiError({ status: 503, message: "Unavailable" }));

    render(
      <Probe
        loader={loader}
        options={{ retry: { initialDelayMs: 10, maxDelayMs: 20, jitterMs: 50 } }}
      />,
    );

    await act(async () => {
      await Promise.resolve();
    });

    await act(async () => {
      await vi.advanceTimersByTimeAsync(19);
    });
    expect(loader).toHaveBeenCalledTimes(1);

    await act(async () => {
      await vi.advanceTimersByTimeAsync(1);
    });
    expect(loader).toHaveBeenCalledTimes(2);
  });

  it("cancels a pending retry on param change and fetches fresh params once", async () => {
    vi.useFakeTimers();
    vi.spyOn(Math, "random").mockReturnValue(0);
    const loader = vi.fn((param: string) =>
      param === "old"
        ? Promise.reject(new ApiError({ status: 503, message: "Unavailable" }))
        : Promise.resolve(`fresh-${param}`),
    );

    const { rerender } = render(
      <ParamProbe
        loader={loader}
        param="old"
        options={{ retry: { initialDelayMs: 500, jitterMs: 0 } }}
      />,
    );

    await act(async () => {
      await Promise.resolve();
    });

    expect(screen.getByTestId("error")).toHaveTextContent("Unavailable");

    act(() => {
      rerender(
        <ParamProbe
          loader={loader}
          param="new"
          options={{ retry: { initialDelayMs: 500, jitterMs: 0 } }}
        />,
      );
    });

    await act(async () => {
      await Promise.resolve();
    });

    expect(screen.getByTestId("data")).toHaveTextContent("fresh-new");
    expect(loader.mock.calls.map(([param]) => param)).toEqual(["old", "new"]);

    await act(async () => {
      await vi.advanceTimersByTimeAsync(500);
    });

    expect(loader).toHaveBeenCalledTimes(2);
  });

  it("coalesces revalidate bumps during an in-flight error into one follow-up request", async () => {
    vi.useFakeTimers();
    vi.spyOn(Math, "random").mockReturnValue(0);
    const first = deferred<string>();
    const second = deferred<string>();
    const loader = vi
      .fn<() => Promise<string>>()
      .mockReturnValueOnce(first.promise)
      .mockReturnValueOnce(second.promise);

    const { rerender } = render(<KeyProbe loader={loader} revalidateKey={1} />);

    expect(loader).toHaveBeenCalledTimes(1);

    act(() => {
      rerender(<KeyProbe loader={loader} revalidateKey={2} />);
      rerender(<KeyProbe loader={loader} revalidateKey={3} />);
    });

    expect(loader).toHaveBeenCalledTimes(1);

    await act(async () => {
      first.reject(new ApiError({ status: 502, message: "Bad Gateway" }));
      await Promise.resolve();
    });

    expect(loader).toHaveBeenCalledTimes(2);

    await act(async () => {
      await vi.advanceTimersByTimeAsync(500);
    });

    expect(loader).toHaveBeenCalledTimes(2);

    await act(async () => {
      second.resolve("recovered");
      await Promise.resolve();
    });

    expect(screen.getByTestId("data")).toHaveTextContent("recovered");
    expect(screen.getByTestId("error")).toHaveTextContent("none");
  });
});
