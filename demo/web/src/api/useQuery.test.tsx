import { StrictMode } from "react";
import { act, render, screen } from "@testing-library/react";

import { invalidateQueries, useQuery } from "./useQuery";

function ValueProbe({ queryKey, fn }: { queryKey: string; fn: () => Promise<string | number> }) {
  const query = useQuery(queryKey, fn);
  return <div data-testid="value">{query.loading ? "loading" : query.data}</div>;
}

describe("useQuery", () => {
  it("resolves data", async () => {
    render(<ValueProbe queryKey="state" fn={async () => "ready"} />);

    expect(await screen.findByText("ready")).toBeTruthy();
  });

  it("resolves under StrictMode double mount", async () => {
    const fn = vi.fn(async () => "strict-ready");

    render(
      <StrictMode>
        <ValueProbe queryKey="state" fn={fn} />
      </StrictMode>,
    );

    expect(await screen.findByText("strict-ready")).toBeTruthy();
  });

  it("refetches matching keys on invalidation", async () => {
    let count = 0;
    const fn = vi.fn(async () => {
      count += 1;
      return count;
    });

    render(<ValueProbe queryKey="state" fn={fn} />);
    expect(await screen.findByText("1")).toBeTruthy();

    act(() => {
      invalidateQueries("sta");
    });

    expect(await screen.findByText("2")).toBeTruthy();
    expect(fn).toHaveBeenCalledTimes(2);
  });
});
