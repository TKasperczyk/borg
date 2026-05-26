import { render, screen, waitFor } from "@testing-library/react";
import { afterEach, describe, expect, it, vi } from "vitest";

import { WhyDrawer } from "./WhyDrawer";

function jsonResponse(body: unknown): Response {
  return new Response(JSON.stringify(body), {
    status: 200,
    headers: { "Content-Type": "application/json" },
  });
}

function requestPath(request: RequestInfo | URL): string {
  return new URL(String(request), "http://test.invalid").pathname;
}

describe("WhyDrawer", () => {
  afterEach(() => {
    vi.unstubAllGlobals();
  });

  it("fetches and renders structured provenance JSON when opened", async () => {
    const fetchMock = vi.fn((request: RequestInfo | URL) => {
      expect(requestPath(request)).toBe("/api/correction/ep_test/why");
      return Promise.resolve(
        jsonResponse({
          target_type: "episode",
          record: { id: "ep_test", title: "seed memory" },
          source_stream_ids: ["strm_one"],
        }),
      );
    });
    vi.stubGlobal("fetch", fetchMock);

    render(<WhyDrawer open id="ep_test" onClose={() => undefined} />);

    expect(await screen.findByText("target_type")).toBeInTheDocument();
    expect(screen.getByText("episode")).toBeInTheDocument();
    await waitFor(() => {
      expect(fetchMock).toHaveBeenCalledTimes(1);
    });
    expect(screen.getByText(/seed memory/)).toBeInTheDocument();
    expect(screen.getByText(/strm_one/)).toBeInTheDocument();
  });
});
