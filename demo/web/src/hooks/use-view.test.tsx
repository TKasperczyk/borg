import { act, fireEvent, render, screen } from "@testing-library/react";
import { afterEach, describe, expect, it } from "vitest";

import { useView } from "./use-view";

function Probe() {
  const { view, governanceTab, memoryBand, dreamProcess, setView, setGovernanceTab } = useView();
  return (
    <div>
      <span data-testid="view">{view}</span>
      <span data-testid="governance-tab">{governanceTab}</span>
      <span data-testid="memory-band">{memoryBand ?? ""}</span>
      <span data-testid="dream-process">{dreamProcess ?? ""}</span>
      <button type="button" onClick={() => setView("memory")}>
        switch
      </button>
      <button type="button" onClick={() => setView("memory", { memoryBand: "semantic" })}>
        semantic band
      </button>
      <button type="button" onClick={() => setView("dream", { dreamProcess: "ruminator" })}>
        ruminator process
      </button>
      <button type="button" onClick={() => setView("mission")}>
        default
      </button>
      <button type="button" onClick={() => setView("cognition")}>
        cognition
      </button>
      <button
        type="button"
        onClick={() => setView("governance", { governanceTab: "shared_state" })}
      >
        directives tab
      </button>
      <button type="button" onClick={() => setGovernanceTab("scope")}>
        scope tab
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

  it("normalizes legacy commit and directives view aliases to governance", () => {
    window.history.replaceState(null, "", "/?view=commit");

    const { unmount } = render(<Probe />);

    expect(screen.getByTestId("view")).toHaveTextContent("governance");
    expect(screen.getByTestId("governance-tab")).toHaveTextContent("commitments");
    expect(new URL(window.location.href).searchParams.get("view")).toBe("governance");
    expect(new URL(window.location.href).searchParams.has("tab")).toBe(false);

    unmount();
    window.history.replaceState(null, "", "/?view=directives");

    render(<Probe />);

    expect(screen.getByTestId("view")).toHaveTextContent("governance");
    expect(screen.getByTestId("governance-tab")).toHaveTextContent("shared_state");
    expect(new URL(window.location.href).searchParams.get("view")).toBe("governance");
    expect(new URL(window.location.href).searchParams.get("tab")).toBe("shared_state");
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

  it("writes and clears governance tab query parameters", () => {
    window.history.replaceState(null, "", "/");

    render(<Probe />);

    fireEvent.click(screen.getByRole("button", { name: "directives tab" }));

    expect(screen.getByTestId("view")).toHaveTextContent("governance");
    expect(screen.getByTestId("governance-tab")).toHaveTextContent("shared_state");
    expect(new URL(window.location.href).searchParams.get("view")).toBe("governance");
    expect(new URL(window.location.href).searchParams.get("tab")).toBe("shared_state");

    fireEvent.click(screen.getByRole("button", { name: "scope tab" }));

    expect(screen.getByTestId("governance-tab")).toHaveTextContent("scope");
    expect(new URL(window.location.href).searchParams.get("tab")).toBe("scope");

    fireEvent.click(screen.getByRole("button", { name: "default" }));

    expect(screen.getByTestId("view")).toHaveTextContent("mission");
    expect(new URL(window.location.href).searchParams.has("view")).toBe(false);
    expect(new URL(window.location.href).searchParams.has("tab")).toBe(false);
  });

  it("writes memory band and dream process route options", () => {
    window.history.replaceState(null, "", "/");

    render(<Probe />);

    fireEvent.click(screen.getByRole("button", { name: "semantic band" }));

    expect(screen.getByTestId("view")).toHaveTextContent("memory");
    expect(screen.getByTestId("memory-band")).toHaveTextContent("semantic");
    expect(new URL(window.location.href).searchParams.get("view")).toBe("memory");
    expect(new URL(window.location.href).searchParams.get("band")).toBe("semantic");

    fireEvent.click(screen.getByRole("button", { name: "ruminator process" }));

    expect(screen.getByTestId("view")).toHaveTextContent("dream");
    expect(screen.getByTestId("dream-process")).toHaveTextContent("ruminator");
    expect(new URL(window.location.href).searchParams.get("view")).toBe("dream");
    expect(new URL(window.location.href).searchParams.get("process")).toBe("ruminator");
    expect(new URL(window.location.href).searchParams.has("band")).toBe(false);
  });

  it("reads valid route options and clears invalid route options", () => {
    window.history.replaceState(null, "", "/?view=memory&band=episodic&process=not-real");

    const { unmount } = render(<Probe />);

    expect(screen.getByTestId("memory-band")).toHaveTextContent("episodic");
    expect(new URL(window.location.href).searchParams.get("band")).toBe("episodic");
    expect(new URL(window.location.href).searchParams.has("process")).toBe(false);

    unmount();
    window.history.replaceState(null, "", "/?view=dream&process=commitment-reconciler&band=bad");

    render(<Probe />);

    expect(screen.getByTestId("dream-process")).toHaveTextContent("commitment-reconciler");
    expect(new URL(window.location.href).searchParams.get("process")).toBe("commitment-reconciler");
    expect(new URL(window.location.href).searchParams.has("band")).toBe(false);
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
