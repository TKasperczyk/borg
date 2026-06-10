import { fireEvent, render, screen, waitFor } from "@testing-library/react";
import { afterEach, describe, expect, it, vi } from "vitest";

import { RESET_CONFIRM_TOKEN } from "../api/client";
import { ResetButton } from "./ResetButton";

function requestPath(request: RequestInfo | URL): string {
  return new URL(String(request), "http://test.invalid").pathname;
}

afterEach(() => {
  vi.unstubAllGlobals();
  vi.restoreAllMocks();
});

describe("ResetButton", () => {
  it("opens the modal and keeps the confirm button disabled until the user types RESET", () => {
    render(<ResetButton />);

    fireEvent.click(screen.getByRole("button", { name: "reset" }));

    const confirm = screen.getByRole("button", { name: "reset borg" });
    expect(confirm).toBeDisabled();
    expect(confirm).toHaveClass("danger");

    const field = screen.getByPlaceholderText(RESET_CONFIRM_TOKEN);
    fireEvent.change(field, { target: { value: "reset" } });
    expect(confirm).toBeDisabled();

    fireEvent.change(field, { target: { value: RESET_CONFIRM_TOKEN } });
    expect(confirm).not.toBeDisabled();
  });

  it("posts to /api/admin/reset with the confirm token when the operator confirms", async () => {
    const fetchMock = vi.fn((_request: RequestInfo | URL, _init?: RequestInit) =>
      Promise.resolve(
        new Response(JSON.stringify({ ok: true }), {
          status: 200,
          headers: { "Content-Type": "application/json" },
        }),
      ),
    );
    vi.stubGlobal("fetch", fetchMock);
    const reload = vi.fn();
    Object.defineProperty(window, "location", {
      configurable: true,
      value: { ...window.location, reload },
    });

    render(<ResetButton />);
    fireEvent.click(screen.getByRole("button", { name: "reset" }));
    fireEvent.change(screen.getByPlaceholderText(RESET_CONFIRM_TOKEN), {
      target: { value: RESET_CONFIRM_TOKEN },
    });
    fireEvent.click(screen.getByRole("button", { name: "reset borg" }));

    await waitFor(() => {
      expect(fetchMock).toHaveBeenCalledTimes(1);
    });

    const [request, init] = fetchMock.mock.calls[0]!;
    expect(requestPath(request as RequestInfo | URL)).toBe("/api/admin/reset");
    expect(init?.method).toBe("POST");
    expect(JSON.parse(String(init?.body))).toEqual({ confirm: RESET_CONFIRM_TOKEN });
    await waitFor(() => expect(reload).toHaveBeenCalled());
  });

  it("shows the server error message and re-enables the form when reset fails", async () => {
    const fetchMock = vi.fn((_request: RequestInfo | URL, _init?: RequestInit) =>
      Promise.resolve(
        new Response(JSON.stringify({ error: { message: "boom" } }), {
          status: 500,
          headers: { "Content-Type": "application/json" },
        }),
      ),
    );
    vi.stubGlobal("fetch", fetchMock);

    render(<ResetButton />);
    fireEvent.click(screen.getByRole("button", { name: "reset" }));
    fireEvent.change(screen.getByPlaceholderText(RESET_CONFIRM_TOKEN), {
      target: { value: RESET_CONFIRM_TOKEN },
    });
    fireEvent.click(screen.getByRole("button", { name: "reset borg" }));

    await waitFor(() => {
      expect(screen.getByRole("alert").textContent).toContain("boom");
    });

    expect(screen.getByRole("button", { name: "reset borg" })).not.toBeDisabled();
  });
});
