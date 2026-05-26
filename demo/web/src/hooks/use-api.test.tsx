import { act, render, screen, waitFor } from "@testing-library/react";
import { describe, expect, it } from "vitest";

import { useApi } from "./use-api";

type Deferred<T> = {
  promise: Promise<T>;
  resolve: (value: T) => void;
};

function deferred<T>(): Deferred<T> {
  let resolve: (value: T) => void = () => undefined;
  const promise = new Promise<T>((promiseResolve) => {
    resolve = promiseResolve;
  });

  return { promise, resolve };
}

function Probe({ loader }: { loader: () => Promise<string> }) {
  const api = useApi(loader, [loader]);

  return (
    <div>
      <div data-testid="data">{api.data ?? "none"}</div>
      <button onClick={() => void api.refetch()}>refetch</button>
    </div>
  );
}

describe("useApi", () => {
  it("does not let an older refetch overwrite a newer response", async () => {
    const first = deferred<string>();
    const second = deferred<string>();
    const third = deferred<string>();
    const requests = [first, second, third];
    const loader = () => requests.shift()?.promise ?? Promise.resolve("fallback");

    render(<Probe loader={loader} />);

    act(() => {
      first.resolve("initial");
    });
    expect(await screen.findByText("initial")).toBeInTheDocument();

    act(() => {
      screen.getByRole("button", { name: "refetch" }).click();
      screen.getByRole("button", { name: "refetch" }).click();
    });

    await act(async () => {
      third.resolve("newer");
      await Promise.resolve();
    });
    expect(screen.getByTestId("data")).toHaveTextContent("newer");

    await act(async () => {
      second.resolve("older");
      await Promise.resolve();
    });

    await waitFor(() => {
      expect(screen.getByTestId("data")).toHaveTextContent("newer");
    });
  });
});
