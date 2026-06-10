import { fireEvent, render, screen, waitFor, within } from "@testing-library/react";
import { afterEach, describe, expect, it, vi } from "vitest";

import { renderWithInspector } from "../test/inspector";
import { JsonValueView } from "./JsonValueView";
import { Modal } from "./Modal";
import { ProvenanceEvidence } from "./ProvenanceEvidence";

function jsonResponse(body: unknown, status = 200): Response {
  return new Response(JSON.stringify(body), {
    status,
    headers: { "Content-Type": "application/json" },
  });
}

function requestPath(request: RequestInfo | URL): string {
  return new URL(String(request), "http://test.invalid").pathname;
}

describe("ProvenanceEvidence", () => {
  afterEach(() => {
    vi.unstubAllGlobals();
  });

  it("renders provenance first with ids, event diffs, and collapsed raw record", async () => {
    const id = "val_abcdefghijklmnop";
    const fetchMock = vi.fn((request: RequestInfo | URL) => {
      expect(requestPath(request)).toBe(`/api/correction/${id}/why`);
      return Promise.resolve(
        jsonResponse({
          target_type: "value",
          source_stream_ids: ["strm_abcdefghijklmnop"],
          identity_events: [
            {
              id: 1,
              action: "correction_apply",
              record_type: "value",
              record_id: id,
              ts: 1_700_000_000_000,
              old_value: {
                description: "old description",
                unchanged: "same",
              },
              new_value: {
                description: "new description",
                unchanged: "same",
              },
              reason: "operator correction",
            },
          ],
          record: {
            id,
            label: "clarity",
            description: "new description",
            updated_at: 1_700_000_000_000,
            ignored_null: null,
          },
        }),
      );
    });
    vi.stubGlobal("fetch", fetchMock);

    renderWithInspector(<ProvenanceEvidence id={id} />);

    expect((await screen.findAllByText("value")).length).toBeGreaterThan(0);
    expect(screen.getAllByRole("button", { name: `jump to ${id}` }).length).toBeGreaterThan(0);
    expect(
      screen.getByRole("button", { name: "jump to strm_abcdefghijklmnop" }),
    ).toBeInTheDocument();
    expect(screen.getAllByText("correction_apply").length).toBeGreaterThan(0);
    expect(screen.getAllByText("old description").length).toBeGreaterThan(0);
    expect(screen.getAllByText("new description").length).toBeGreaterThan(0);
    const rawEvent = screen.getByText("raw event").closest("details");
    expect(rawEvent).not.toBeNull();
    expect(rawEvent).not.toHaveAttribute("open");
    expect(within(rawEvent as HTMLElement).getAllByText(/unchanged/).length).toBeGreaterThan(0);
    expect(screen.queryByText("ignored null")).not.toBeInTheDocument();
    expect(screen.getByText("record").closest("details")).not.toHaveAttribute("open");
    await waitFor(() => expect(fetchMock).toHaveBeenCalledTimes(1));
  });

  it("uses the shared not-found vocabulary", async () => {
    vi.stubGlobal(
      "fetch",
      vi.fn(() =>
        Promise.resolve(jsonResponse({ error: { status: 404, message: "missing" } }, 404)),
      ),
    );

    renderWithInspector(<ProvenanceEvidence id="val_missing1111111" />);

    expect(await screen.findByText("no provenance retained")).toBeInTheDocument();
  });
});

describe("JsonValueView", () => {
  it("summarizes large collections until explicitly expanded", () => {
    render(<JsonValueView value={Array.from({ length: 60 }, (_, index) => index)} />);

    expect(screen.getByText("60 numbers")).toBeInTheDocument();
    expect(screen.queryByText(/59/)).not.toBeInTheDocument();

    fireEvent.click(screen.getByRole("button", { name: "expand" }));

    expect(screen.getByText(/59/)).toBeInTheDocument();
  });
});

describe("Modal", () => {
  it("renders a visible close affordance and keeps Escape closing", () => {
    const onClose = vi.fn();

    render(
      <Modal open title="operator modal" onClose={onClose}>
        body
      </Modal>,
    );

    expect(screen.getByText("esc")).toBeInTheDocument();
    fireEvent.click(screen.getByRole("button", { name: "close dialog" }));
    expect(onClose).toHaveBeenCalledTimes(1);

    fireEvent.keyDown(window, { key: "Escape" });
    expect(onClose).toHaveBeenCalledTimes(2);
  });
});
