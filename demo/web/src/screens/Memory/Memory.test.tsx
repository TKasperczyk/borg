import { fireEvent, render, screen, waitFor, within } from "@testing-library/react";
import { afterEach, describe, expect, it, vi } from "vitest";

import { LiveEventsProvider } from "../../hooks/live-context";
import type { LiveEvents } from "../../hooks/use-live-events";
import { MemoryScreen } from ".";

const EPISODE_ID = "ep_aaaaaaaaaaaaaaaa";
const LOADED_SEMANTIC_NODE_ID = "semn_loaded0000000";
const OFF_PAGE_SEMANTIC_NODE_ID = "semn_offpage000000";

function jsonResponse(body: unknown): Response {
  return new Response(JSON.stringify(body), {
    status: 200,
    headers: { "Content-Type": "application/json" },
  });
}

function requestPath(request: RequestInfo | URL): string {
  return new URL(String(request), "http://test.invalid").pathname;
}

function requestUrl(request: RequestInfo | URL): URL {
  return new URL(String(request), "http://test.invalid");
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
        count_is_lower_bound: false,
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
    items: [episodeItem(EPISODE_ID, "Episode one")],
    next_cursor: null,
  };
}

function episodeItem(
  id: string,
  title: string,
  input: { audience?: string | null; tags?: string[]; ts?: number; search_score?: number } = {},
) {
  const ts = input.ts ?? 1;
  return {
    id,
    title,
    narrative: `${title} narrative`,
    participants: ["operator"],
    location: null,
    start_time: ts,
    end_time: ts,
    audience: input.audience ?? null,
    significance: 0.5,
    confidence: 0.8,
    tags: input.tags ?? ["test"],
    source_stream_ids: ["strm_one"],
    source_count: 1,
    lineage: { derived_from: [], supersedes: [] },
    emotional_arc: null,
    vector_dims: 4,
    created_at: ts,
    updated_at: ts,
    ...(input.search_score === undefined ? {} : { search_score: input.search_score }),
  };
}

function semanticNode(id: string, label: string, description: string) {
  return {
    id,
    kind: "entity",
    label,
    description,
    domain: null,
    aliases: [],
    confidence: 0.8,
    status: "active",
    source_episode_ids: ["ep_source000000000"],
    source_count: 1,
    created_at: 1,
    updated_at: 2,
  };
}

function testLiveEvents(): LiveEvents {
  return {
    wsState: "live",
    connectionCount: 1,
    subscribe: vi.fn(() => () => undefined),
  };
}

describe("Memory correction actions", () => {
  afterEach(() => {
    vi.restoreAllMocks();
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

  it("loads more episodic memory rows with the returned cursor", async () => {
    const fetchMock = vi.fn((request: RequestInfo | URL) => {
      const url = requestUrl(request);
      if (url.pathname === "/api/memory/bands") {
        const bands = memoryBandsResponse();
        bands.bands[0]!.count = 2;
        return Promise.resolve(jsonResponse(bands));
      }
      if (url.pathname === "/api/reviews") {
        return Promise.resolve(jsonResponse({ rows: [] }));
      }
      if (url.pathname === "/api/memory/bands/episodic") {
        if (url.searchParams.get("cursor") === "cursor-1") {
          return Promise.resolve(
            jsonResponse({
              band: "episodic",
              mode: "browse",
              items: [episodeItem("ep_bbbbbbbbbbbbbbbb", "Episode two", { ts: 2 })],
              next_cursor: null,
            }),
          );
        }
        return Promise.resolve(
          jsonResponse({
            band: "episodic",
            mode: "browse",
            items: [episodeItem(EPISODE_ID, "Episode one", { ts: 3 })],
            next_cursor: "cursor-1",
          }),
        );
      }
      return Promise.resolve(new Response("not found", { status: 404 }));
    });
    vi.stubGlobal("fetch", fetchMock);

    render(<MemoryScreen sessionId="default" />);

    const episodicLabels = await screen.findAllByText("episodic");
    fireEvent.click(episodicLabels[0]?.closest(".band-card") ?? episodicLabels[0]!);
    expect(await screen.findByText("loaded 1 of 2")).toBeInTheDocument();

    fireEvent.click(screen.getByRole("button", { name: "load more" }));

    expect(await screen.findByText("Episode two")).toBeInTheDocument();
    expect(screen.getByText("loaded 2 of 2")).toBeInTheDocument();
    expect(
      fetchMock.mock.calls.some(
        ([request]) => requestUrl(request).searchParams.get("cursor") === "cursor-1",
      ),
    ).toBe(true);
  });

  it("shows capped band totals as a lower bound", async () => {
    const fetchMock = vi.fn((request: RequestInfo | URL) => {
      const url = requestUrl(request);
      if (url.pathname === "/api/memory/bands") {
        const bands = memoryBandsResponse();
        bands.bands[0]!.count = 500;
        bands.bands[0]!.count_is_lower_bound = true;
        return Promise.resolve(jsonResponse(bands));
      }
      if (url.pathname === "/api/reviews") {
        return Promise.resolve(jsonResponse({ rows: [] }));
      }
      if (url.pathname === "/api/memory/bands/episodic") {
        return Promise.resolve(
          jsonResponse({
            band: "episodic",
            mode: "browse",
            items: [episodeItem(EPISODE_ID, "Episode one")],
            next_cursor: "cursor-1",
          }),
        );
      }
      return Promise.resolve(new Response("not found", { status: 404 }));
    });
    vi.stubGlobal("fetch", fetchMock);

    render(<MemoryScreen sessionId="default" />);

    expect(await screen.findByText("500+")).toBeInTheDocument();
    const episodicLabels = await screen.findAllByText("episodic");
    fireEvent.click(episodicLabels[0]?.closest(".band-card") ?? episodicLabels[0]!);

    expect(await screen.findByText("loaded 1 of 500+")).toBeInTheDocument();
  });

  it("keeps browse rows when submitting an empty search from browse mode", async () => {
    const fetchMock = vi.fn((request: RequestInfo | URL) => {
      const url = requestUrl(request);
      if (url.pathname === "/api/memory/bands") {
        return Promise.resolve(jsonResponse(memoryBandsResponse()));
      }
      if (url.pathname === "/api/reviews") {
        return Promise.resolve(jsonResponse({ rows: [] }));
      }
      if (url.pathname === "/api/memory/bands/episodic") {
        return Promise.resolve(
          jsonResponse({
            band: "episodic",
            mode: "browse",
            items: [episodeItem(EPISODE_ID, "Episode one")],
            next_cursor: null,
          }),
        );
      }
      return Promise.resolve(new Response("not found", { status: 404 }));
    });
    vi.stubGlobal("fetch", fetchMock);
    const { container } = render(<MemoryScreen sessionId="default" />);

    const episodicLabels = await screen.findAllByText("episodic");
    fireEvent.click(episodicLabels[0]?.closest(".band-card") ?? episodicLabels[0]!);
    await waitFor(() => {
      expect(
        [...container.querySelectorAll(".list-row .ttl")].map((item) => item.textContent),
      ).toEqual(["Episode one"]);
    });
    const detailCallsBefore = fetchMock.mock.calls.filter(
      ([request]) => requestUrl(request).pathname === "/api/memory/bands/episodic",
    ).length;

    fireEvent.click(screen.getByRole("button", { name: "search" }));

    expect(
      [...container.querySelectorAll(".list-row .ttl")].map((item) => item.textContent),
    ).toEqual(["Episode one"]);
    expect(
      fetchMock.mock.calls.filter(
        ([request]) => requestUrl(request).pathname === "/api/memory/bands/episodic",
      ),
    ).toHaveLength(detailCallsBefore);
    expect(
      fetchMock.mock.calls.some(([request]) => requestUrl(request).searchParams.has("query")),
    ).toBe(false);
  });

  it("sorts and filters accumulated rows by structural fields only", async () => {
    const fetchMock = vi.fn((request: RequestInfo | URL) => {
      const url = requestUrl(request);
      if (url.pathname === "/api/memory/bands") {
        const bands = memoryBandsResponse();
        bands.bands[0]!.count = 2;
        return Promise.resolve(jsonResponse(bands));
      }
      if (url.pathname === "/api/reviews") {
        return Promise.resolve(jsonResponse({ rows: [] }));
      }
      if (url.pathname === "/api/memory/bands/episodic") {
        return Promise.resolve(
          jsonResponse({
            band: "episodic",
            mode: "browse",
            items: [
              episodeItem("ep_bbbbbbbbbbbbbbbb", "Beta episode", {
                audience: "Alice",
                tags: ["beta"],
                ts: 20,
              }),
              episodeItem("ep_cccccccccccccccc", "Alpha episode", {
                audience: "global",
                tags: ["alpha"],
                ts: 10,
              }),
            ],
            next_cursor: null,
          }),
        );
      }
      return Promise.resolve(new Response("not found", { status: 404 }));
    });
    vi.stubGlobal("fetch", fetchMock);
    const { container } = render(<MemoryScreen sessionId="default" />);

    const episodicLabels = await screen.findAllByText("episodic");
    fireEvent.click(episodicLabels[0]?.closest(".band-card") ?? episodicLabels[0]!);
    await waitFor(() => {
      expect(
        [...container.querySelectorAll(".list-row .ttl")].map((item) => item.textContent),
      ).toEqual(["Beta episode", "Alpha episode"]);
    });

    fireEvent.click(screen.getByText("oldest"));
    const titles = [...container.querySelectorAll(".list-row .ttl")].map(
      (item) => item.textContent,
    );
    expect(titles).toEqual(["Alpha episode", "Beta episode"]);

    fireEvent.click(screen.getByText("beta"));
    const filteredTitles = [...container.querySelectorAll(".list-row .ttl")].map(
      (item) => item.textContent,
    );
    expect(filteredTitles).toEqual(["Beta episode"]);
    expect(screen.queryByText("Alpha episode")).not.toBeInTheDocument();
  });

  it("renders server-ranked search results from the band detail query path", async () => {
    const fetchMock = vi.fn((request: RequestInfo | URL) => {
      const url = requestUrl(request);
      if (url.pathname === "/api/memory/bands") {
        const bands = memoryBandsResponse();
        bands.bands[0]!.count = 2;
        return Promise.resolve(jsonResponse(bands));
      }
      if (url.pathname === "/api/reviews") {
        return Promise.resolve(jsonResponse({ rows: [] }));
      }
      if (url.pathname === "/api/memory/bands/episodic") {
        if (url.searchParams.get("query") === "meaning") {
          return Promise.resolve(
            jsonResponse({
              band: "episodic",
              mode: "search",
              query: "meaning",
              items: [
                episodeItem("ep_dddddddddddddddd", "Meaning result", {
                  ts: 30,
                  search_score: 0.92,
                }),
              ],
              next_cursor: null,
            }),
          );
        }
        return Promise.resolve(
          jsonResponse({
            band: "episodic",
            mode: "browse",
            items: [episodeItem(EPISODE_ID, "Episode one")],
            next_cursor: null,
          }),
        );
      }
      return Promise.resolve(new Response("not found", { status: 404 }));
    });
    vi.stubGlobal("fetch", fetchMock);

    render(<MemoryScreen sessionId="default" />);

    const episodicLabels = await screen.findAllByText("episodic");
    fireEvent.click(episodicLabels[0]?.closest(".band-card") ?? episodicLabels[0]!);
    fireEvent.change(await screen.findByPlaceholderText("search episodic"), {
      target: { value: "meaning" },
    });
    fireEvent.click(screen.getByRole("button", { name: "search" }));

    expect((await screen.findAllByText("Meaning result")).length).toBeGreaterThan(0);
    expect(screen.getByText("1 results")).toBeInTheDocument();
    expect(screen.getAllByText(/score 0.92/).length).toBeGreaterThan(0);
    expect(
      fetchMock.mock.calls.some(
        ([request]) => requestUrl(request).searchParams.get("query") === "meaning",
      ),
    ).toBe(true);
  });

  it("selects and inspects a topology semantic node outside the loaded browser page", async () => {
    vi.spyOn(window, "requestAnimationFrame").mockImplementation((callback) =>
      window.setTimeout(() => callback(performance.now()), 16),
    );
    vi.spyOn(window, "cancelAnimationFrame").mockImplementation((id) => {
      window.clearTimeout(id);
    });

    const fetchMock = vi.fn((request: RequestInfo | URL) => {
      const url = requestUrl(request);
      if (url.pathname === "/api/memory/bands") {
        const bands = memoryBandsResponse();
        bands.bands[1]!.count = 2;
        return Promise.resolve(jsonResponse(bands));
      }
      if (url.pathname === "/api/reviews") {
        return Promise.resolve(jsonResponse({ rows: [] }));
      }
      if (url.pathname === "/api/memory/bands/semantic") {
        return Promise.resolve(
          jsonResponse({
            band: "semantic",
            mode: "browse",
            nodes: [
              semanticNode(
                LOADED_SEMANTIC_NODE_ID,
                "Loaded semantic node",
                "Loaded semantic node description",
              ),
            ],
            edges: [],
            next_cursor: null,
          }),
        );
      }
      if (url.pathname === "/api/semantic/graph") {
        return Promise.resolve(
          jsonResponse({
            nodes: [
              {
                id: LOADED_SEMANTIC_NODE_ID,
                label: "Loaded semantic node",
                status: "active",
                kind: "entity",
                edge_count: 1,
              },
              {
                id: OFF_PAGE_SEMANTIC_NODE_ID,
                label: "Off page semantic node",
                status: "active",
                kind: "entity",
                edge_count: 1,
              },
            ],
            edges: [
              {
                id: "seme_related00000",
                source: LOADED_SEMANTIC_NODE_ID,
                target: OFF_PAGE_SEMANTIC_NODE_ID,
                type: "related_to",
                weight: 0.7,
              },
            ],
            total_nodes: 2,
            total_edges: 1,
            rendered: { nodes: 2, edges: 1 },
          }),
        );
      }
      if (url.pathname === `/api/semantic/nodes/${OFF_PAGE_SEMANTIC_NODE_ID}`) {
        return Promise.resolve(
          jsonResponse({
            node: semanticNode(
              OFF_PAGE_SEMANTIC_NODE_ID,
              "Off page semantic node",
              "Off page full semantic description",
            ),
          }),
        );
      }
      return Promise.resolve(new Response("not found", { status: 404 }));
    });
    vi.stubGlobal("fetch", fetchMock);

    render(
      <LiveEventsProvider value={testLiveEvents()}>
        <MemoryScreen sessionId="default" />
      </LiveEventsProvider>,
    );

    const semanticLabels = await screen.findAllByText("semantic");
    fireEvent.click(semanticLabels[0]?.closest(".band-card") ?? semanticLabels[0]!);
    expect((await screen.findAllByText("Loaded semantic node")).length).toBeGreaterThan(0);

    fireEvent.click(screen.getByRole("tab", { name: "topology" }));
    expect(await screen.findByTestId("semantic-topology-svg")).toBeInTheDocument();
    fireEvent.click(await screen.findByRole("button", { name: /Off page semantic node/ }));

    expect(await screen.findByText("Off page full semantic description")).toBeInTheDocument();
    expect(screen.queryByText("Loaded semantic node description")).not.toBeInTheDocument();
    expect(
      fetchMock.mock.calls.some(
        ([request]) =>
          requestUrl(request).pathname === `/api/semantic/nodes/${OFF_PAGE_SEMANTIC_NODE_ID}`,
      ),
    ).toBe(true);
  });

  it("does not feed semantic edge row selection into topology highlight state", async () => {
    vi.spyOn(window, "requestAnimationFrame").mockImplementation((callback) =>
      window.setTimeout(() => callback(performance.now()), 16),
    );
    vi.spyOn(window, "cancelAnimationFrame").mockImplementation((id) => {
      window.clearTimeout(id);
    });

    const edgeId = "seme_edge00000000";
    const fetchMock = vi.fn((request: RequestInfo | URL) => {
      const url = requestUrl(request);
      if (url.pathname === "/api/memory/bands") {
        const bands = memoryBandsResponse();
        bands.bands[1]!.count = 2;
        return Promise.resolve(jsonResponse(bands));
      }
      if (url.pathname === "/api/reviews") {
        return Promise.resolve(jsonResponse({ rows: [] }));
      }
      if (url.pathname === "/api/memory/bands/semantic") {
        return Promise.resolve(
          jsonResponse({
            band: "semantic",
            mode: "browse",
            nodes: [
              semanticNode(
                LOADED_SEMANTIC_NODE_ID,
                "Loaded semantic node",
                "Loaded semantic node description",
              ),
            ],
            edges: [
              {
                id: edgeId,
                from_node_id: LOADED_SEMANTIC_NODE_ID,
                to_node_id: OFF_PAGE_SEMANTIC_NODE_ID,
                relation: "related_to",
                confidence: 0.7,
                evidence_episode_ids: ["ep_source000000000"],
                source_count: 1,
                valid_from: 1,
                valid_to: null,
                invalidated_at: null,
                invalidated_by_edge_id: null,
                invalidated_by_review_id: null,
                invalidated_by_process: null,
                invalidated_reason: null,
              },
            ],
            next_cursor: null,
          }),
        );
      }
      if (url.pathname === "/api/semantic/graph") {
        return Promise.resolve(
          jsonResponse({
            nodes: [
              {
                id: LOADED_SEMANTIC_NODE_ID,
                label: "Loaded semantic node",
                status: "active",
                kind: "entity",
                edge_count: 1,
              },
              {
                id: OFF_PAGE_SEMANTIC_NODE_ID,
                label: "Off page semantic node",
                status: "active",
                kind: "entity",
                edge_count: 1,
              },
            ],
            edges: [
              {
                id: edgeId,
                source: LOADED_SEMANTIC_NODE_ID,
                target: OFF_PAGE_SEMANTIC_NODE_ID,
                type: "related_to",
                weight: 0.7,
              },
            ],
            total_nodes: 2,
            total_edges: 1,
            rendered: { nodes: 2, edges: 1 },
          }),
        );
      }
      return Promise.resolve(new Response("not found", { status: 404 }));
    });
    vi.stubGlobal("fetch", fetchMock);

    render(
      <LiveEventsProvider value={testLiveEvents()}>
        <MemoryScreen sessionId="default" />
      </LiveEventsProvider>,
    );

    const semanticLabels = await screen.findAllByText("semantic");
    fireEvent.click(semanticLabels[0]?.closest(".band-card") ?? semanticLabels[0]!);
    fireEvent.click(await screen.findByText(/--related_to->/));
    fireEvent.click(screen.getByRole("tab", { name: "topology" }));

    expect(await screen.findByTestId("semantic-topology-svg")).toBeInTheDocument();
    expect(screen.queryByText(/selected seme_/)).not.toBeInTheDocument();
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
