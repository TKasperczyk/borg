import { fireEvent, render, screen, waitFor, within } from "@testing-library/react";
import { afterEach, describe, expect, it, vi } from "vitest";

import { MemoryScreen } from ".";

const EPISODE_ID = "ep_aaaaaaaaaaaaaaaa";

function jsonResponse(body: unknown): Response {
  return new Response(JSON.stringify(body), {
    status: 200,
    headers: { "Content-Type": "application/json" },
  });
}

function requestPath(request: RequestInfo | URL): string {
  return new URL(String(request), "http://test.invalid").pathname;
}

function memoryBandsResponse() {
  return {
    bands: [
      {
        id: "episodic",
        n: "01",
        name: "episodic",
        desc: "what happened",
        count: 1,
        growth: [1],
        stats: [{ k: "items", v: 1 }],
      },
      {
        id: "semantic",
        n: "02",
        name: "semantic",
        desc: "beliefs",
        count: 0,
        growth: [1],
        stats: [],
      },
      {
        id: "procedural",
        n: "03",
        name: "procedural",
        desc: "skills",
        count: 0,
        growth: [1],
        stats: [],
      },
      {
        id: "affective",
        n: "04",
        name: "affective",
        desc: "mood",
        count: 0,
        growth: [1],
        stats: [],
      },
      {
        id: "self",
        n: "05",
        name: "self",
        desc: "identity",
        count: 0,
        growth: [1],
        stats: [],
      },
      {
        id: "commitments",
        n: "06",
        name: "commitments",
        desc: "rules",
        count: 0,
        growth: [1],
        stats: [],
      },
      {
        id: "social",
        n: "07",
        name: "social",
        desc: "people",
        count: 0,
        growth: [1],
        stats: [],
      },
      {
        id: "relational",
        n: "08",
        name: "relational",
        desc: "slots",
        count: 0,
        growth: [1],
        stats: [],
      },
    ],
  };
}

function episodeBandResponse() {
  return {
    band: "episodic",
    items: [
      {
        id: EPISODE_ID,
        title: "Episode one",
        narrative: "remembered narrative",
        participants: ["operator"],
        location: null,
        start_time: 1,
        end_time: 1,
        audience: null,
        significance: 0.5,
        confidence: 0.8,
        tags: ["test"],
        source_stream_ids: ["strm_one"],
        source_count: 1,
        lineage: { derived_from: [], supersedes: [] },
        emotional_arc: null,
        vector_dims: 4,
        created_at: 1,
        updated_at: 1,
      },
    ],
    nextCursor: null,
  };
}

describe("Memory correction actions", () => {
  afterEach(() => {
    vi.unstubAllGlobals();
  });

  it("posts forget for an episode row and refetches memory data", async () => {
    const fetchMock = vi.fn((request: RequestInfo | URL, init?: RequestInit) => {
      const path = requestPath(request);
      if (path === "/api/memory/bands") {
        return Promise.resolve(jsonResponse(memoryBandsResponse()));
      }
      if (path === "/api/reviews") {
        return Promise.resolve(jsonResponse({ rows: [] }));
      }
      if (path === "/api/memory/bands/episodic") {
        return Promise.resolve(jsonResponse(episodeBandResponse()));
      }
      if (path === `/api/correction/${EPISODE_ID}/forget` && init?.method === "POST") {
        return Promise.resolve(
          jsonResponse({
            id: EPISODE_ID,
            target_type: "episode",
            archived: true,
            provenance: { kind: "manual" },
          }),
        );
      }
      return Promise.resolve(new Response("not found", { status: 404 }));
    });
    vi.stubGlobal("fetch", fetchMock);

    render(<MemoryScreen sessionId="default" />);

    const episodicLabels = await screen.findAllByText("episodic");
    fireEvent.click(episodicLabels[0]?.closest(".band-card") ?? episodicLabels[0]!);
    expect((await screen.findAllByText("Episode one")).length).toBeGreaterThan(0);

    fireEvent.click(screen.getByRole("button", { name: "forget" }));
    fireEvent.click(within(screen.getByRole("dialog")).getByRole("button", { name: "forget" }));

    await waitFor(() => {
      expect(
        fetchMock.mock.calls.some(
          ([request, init]) =>
            requestPath(request) === `/api/correction/${EPISODE_ID}/forget` &&
            init?.method === "POST",
        ),
      ).toBe(true);
    });
    await waitFor(() => {
      expect(
        fetchMock.mock.calls.filter(([request]) => requestPath(request) === "/api/memory/bands"),
      ).toHaveLength(2);
      expect(
        fetchMock.mock.calls.filter(
          ([request]) => requestPath(request) === "/api/memory/bands/episodic",
        ),
      ).toHaveLength(2);
    });
  });

  it("shows correction review count and routes to the unified review screen", async () => {
    const fetchMock = vi.fn((request: RequestInfo | URL, init?: RequestInit) => {
      const path = requestPath(request);
      if (path === "/api/memory/bands") {
        return Promise.resolve(jsonResponse(memoryBandsResponse()));
      }
      if (path === "/api/reviews") {
        return Promise.resolve(
          jsonResponse({
            rows: [
              {
                id: 7,
                kind: "correction",
                refs: {
                  target_type: "episode",
                  target_id: EPISODE_ID,
                  prompt_summary: "user proposed changing episode",
                  operator_reason: "operator supplied correction reason",
                  patch: { title: "Updated episode" },
                },
                reason: "queued",
                created_at: 1,
                resolved_at: null,
                resolution: null,
              },
            ],
          }),
        );
      }
      return Promise.resolve(new Response("not found", { status: 404 }));
    });
    vi.stubGlobal("fetch", fetchMock);
    const openReview = vi.fn();

    render(<MemoryScreen sessionId="default" onOpenReview={openReview} />);

    expect(await screen.findByText("1 pending correction review rows.")).toBeInTheDocument();
    fireEvent.click(screen.getByRole("button", { name: "open review" }));

    expect(openReview).toHaveBeenCalledTimes(1);
    expect(
      fetchMock.mock.calls.some(
        ([request, init]) =>
          requestPath(request) === "/api/correction/reviews/7" && init?.method === "PATCH",
      ),
    ).toBe(false);
  });
});
