import { act, fireEvent, render, screen } from "@testing-library/react";
import { afterEach, describe, expect, it } from "vitest";

import { useView } from "./use-view";

function Probe() {
  const { view, setView } = useView();
  return (
    <div>
      <span data-testid="view">{view}</span>
      <button type="button" onClick={() => setView("memory")}>
        switch
      </button>
      <button type="button" onClick={() => setView("mission")}>
        default
      </button>
      <button type="button" onClick={() => setView("cognition")}>
        cognition
      </button>
    </div>
  );
}

afterEach(() => {
  window.history.replaceState(null, "", "/");
});

describe("useView", () => {
  it("reads the view query parameter", () => {
    window.history.replaceState(null, "", "/?view=stream");

    render(<Probe />);

    expect(screen.getByTestId("view")).toHaveTextContent("stream");
  });

  it("updates the URL when switching views", () => {
    window.history.replaceState(null, "", "/");

    render(<Probe />);

    fireEvent.click(screen.getByRole("button", { name: "switch" }));

    expect(screen.getByTestId("view")).toHaveTextContent("memory");
    expect(new URL(window.location.href).searchParams.get("view")).toBe("memory");
  });

  it("falls back to default and clears an invalid view query parameter", () => {
    window.history.replaceState(null, "", "/?view=!!!invalid!!!");

    render(<Probe />);

    expect(screen.getByTestId("view")).toHaveTextContent("mission");
    expect(new URL(window.location.href).searchParams.has("view")).toBe(false);
  });

  it("clears the default view query parameter", () => {
    window.history.replaceState(null, "", "/?view=memory");

    render(<Probe />);

    fireEvent.click(screen.getByRole("button", { name: "default" }));

    expect(screen.getByTestId("view")).toHaveTextContent("mission");
    expect(new URL(window.location.href).searchParams.has("view")).toBe(false);
  });

  it("keeps cognition as an explicit view query parameter", () => {
    window.history.replaceState(null, "", "/");

    render(<Probe />);

    fireEvent.click(screen.getByRole("button", { name: "cognition" }));

    expect(screen.getByTestId("view")).toHaveTextContent("cognition");
    expect(new URL(window.location.href).searchParams.get("view")).toBe("cognition");
  });

  it("re-reads the view on popstate", () => {
    window.history.replaceState(null, "", "/?view=stream");

    render(<Probe />);

    act(() => {
      window.history.pushState(null, "", "/?view=review");
      window.dispatchEvent(new Event("popstate"));
    });

    expect(screen.getByTestId("view")).toHaveTextContent("review");
  });
});
