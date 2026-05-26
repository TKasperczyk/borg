import { fireEvent, render, screen } from "@testing-library/react";
import { afterEach, describe, expect, it } from "vitest";

import { useSession } from "./use-session";

function Probe() {
  const { sessionId, setSessionId } = useSession();
  return (
    <div>
      <span data-testid="session">{sessionId}</span>
      <button type="button" onClick={() => setSessionId("sess_bbbbbbbbbbbbbbbb")}>
        switch
      </button>
    </div>
  );
}

afterEach(() => {
  window.history.replaceState(null, "", "/");
});

describe("useSession", () => {
  it("reads the session query parameter", () => {
    window.history.replaceState(null, "", "/?session=sess_aaaaaaaaaaaaaaaa");

    render(<Probe />);

    expect(screen.getByTestId("session")).toHaveTextContent("sess_aaaaaaaaaaaaaaaa");
  });

  it("updates the URL when switching sessions", () => {
    window.history.replaceState(null, "", "/");

    render(<Probe />);

    fireEvent.click(screen.getByRole("button", { name: "switch" }));

    expect(screen.getByTestId("session")).toHaveTextContent("sess_bbbbbbbbbbbbbbbb");
    expect(new URL(window.location.href).searchParams.get("session")).toBe(
      "sess_bbbbbbbbbbbbbbbb",
    );
  });

  it("falls back to default and clears an invalid session query parameter", () => {
    window.history.replaceState(null, "", "/?session=!!!invalid!!!");

    render(<Probe />);

    expect(screen.getByTestId("session")).toHaveTextContent("default");
    expect(new URL(window.location.href).searchParams.has("session")).toBe(false);
  });
});
