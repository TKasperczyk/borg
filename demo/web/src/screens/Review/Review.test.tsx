import { act, fireEvent, screen, waitFor } from "@testing-library/react";
import { afterEach, describe, expect, it, vi } from "vitest";

import type { LiveFrame, ReviewKind, ReviewRow, WsState } from "../../api/types";
import { LiveEventsProvider } from "../../hooks/live-context";
import type { LiveEventHandler, LiveEvents } from "../../hooks/use-live-events";
import { renderWithInspector } from "../../test/inspector";
import { ReviewScreen } from ".";

function jsonResponse(body: unknown): Response {
  return new Response(JSON.stringify(body), {
    status: 200,
    headers: { "Content-Type": "application/json" },
  });
}

function requestPath(request: RequestInfo | URL): string {
  return new URL(String(request), "http://test.invalid").pathname;
}

function requestBody(init: RequestInit | undefined): unknown {
  return JSON.parse(String(init?.body ?? "{}")) as unknown;
}

function makeLiveSource(): {
  emit: (frame: LiveFrame) => void;
  live: (connectionCount?: number, wsState?: WsState) => LiveEvents;
} {
  const handlers = new Set<LiveEventHandler>();
  return {
    live: (connectionCount = 1, wsState = "live") => ({
      wsState,
      connectionCount,
      subscribe: (handler) => {
        handlers.add(handler);
        return () => {
          handlers.delete(handler);
        };
      },
    }),
    emit: (frame) => {
      for (const handler of handlers) {
        handler(frame);
      }
    },
  };
}

function reviewRow(
  input: Partial<ReviewRow> & Pick<ReviewRow, "id" | "kind" | "reason">,
): ReviewRow {
  return {
    refs: {},
    created_at: Date.now(),
    resolved_at: null,
    resolution: null,
    ...input,
  };
}

function installReviewFetch(rows: ReviewRow[] | (() => ReviewRow[])): ReturnType<typeof vi.fn> {
  const fetchMock = vi.fn((request: RequestInfo | URL, init?: RequestInit) => {
    const url = new URL(String(request), "http://test.invalid");
    if (url.pathname === "/api/reviews" && init?.method === undefined) {
      const sourceRows = typeof rows === "function" ? rows() : rows;
      return Promise.resolve(jsonResponse({ rows: sourceRows }));
    }
    if (url.pathname === "/api/creator-directives") {
      return Promise.resolve(jsonResponse({ directives: [] }));
    }
    if (url.pathname === "/api/commitments") {
      return Promise.resolve(jsonResponse({ commitments: [] }));
    }
    return Promise.resolve(new Response("not found", { status: 404 }));
  });
  vi.stubGlobal("fetch", fetchMock);
  return fetchMock;
}

function clickQueueRow(reason: string): void {
  const row = screen
    .getAllByText(reason)
    .map((element) => element.closest(".review-queue-row"))
    .find((element): element is HTMLElement => element !== null);
  if (row === undefined) {
    throw new Error(`queue row not found for ${reason}`);
  }
  fireEvent.click(row);
}

function renderReview(live = makeLiveSource(), options: { inspector?: boolean } = {}) {
  return renderWithInspector(
    <LiveEventsProvider value={live.live()}>
      <ReviewScreen />
    </LiveEventsProvider>,
    options,
  );
}

afterEach(() => {
  vi.unstubAllGlobals();
});

describe("Review & Repair", () => {
  it("filters the loaded queue by kind, created-at age, and structurally derived affected type", async () => {
    const now = Date.now();
    const openRows = [
      reviewRow({
        id: 1,
        kind: "correction",
        reason: "open commitment repair",
        refs: { target_id: "cmt_filter111111111", target_type: "commitment" },
        created_at: now - 2 * 24 * 60 * 60 * 1000,
      }),
      reviewRow({
        id: 2,
        kind: "contradiction",
        reason: "fresh semantic review",
        refs: { node_ids: ["semn_filter11111111", "semn_filter22222222"] },
        created_at: now - 10 * 60 * 1000,
      }),
    ];
    const allRows = [
      ...openRows,
      reviewRow({
        id: 3,
        kind: "correction",
        reason: "resolved commitment repair",
        refs: { target_id: "cmt_filter333333333", target_type: "commitment" },
        created_at: now - 2 * 24 * 60 * 60 * 1000,
        resolved_at: now,
        resolution: "reject",
      }),
    ];
    const fetchMock = vi.fn((request: RequestInfo | URL) => {
      const url = new URL(String(request), "http://test.invalid");
      if (url.pathname === "/api/reviews") {
        return Promise.resolve(
          jsonResponse({
            rows: url.searchParams.get("open_only") === "false" ? allRows : openRows,
          }),
        );
      }
      if (url.pathname === "/api/creator-directives") {
        return Promise.resolve(jsonResponse({ directives: [] }));
      }
      if (url.pathname === "/api/commitments") {
        return Promise.resolve(jsonResponse({ commitments: [] }));
      }
      return Promise.resolve(new Response("not found", { status: 404 }));
    });
    vi.stubGlobal("fetch", fetchMock);

    renderReview();

    expect((await screen.findAllByText("open commitment repair")).length).toBeGreaterThan(0);
    expect(screen.getByText("fresh semantic review")).toBeInTheDocument();

    fireEvent.change(screen.getByLabelText("kind filter"), { target: { value: "correction" } });
    fireEvent.change(screen.getByLabelText("age filter"), { target: { value: "week" } });
    fireEvent.change(screen.getByLabelText("affected type filter"), {
      target: { value: "commitment" },
    });

    await waitFor(() => {
      expect(screen.queryByText("fresh semantic review")).not.toBeInTheDocument();
    });
    expect(screen.getAllByText("open commitment repair").length).toBeGreaterThan(0);

    fireEvent.click(screen.getByLabelText("open only"));
    expect(await screen.findByText("resolved commitment repair")).toBeInTheDocument();
    expect(screen.getAllByText("reject").length).toBeGreaterThan(0);
    expect(
      fetchMock.mock.calls.some(([request]) => String(request).includes("open_only=false")),
    ).toBe(true);
  });

  it("does not derive affected type from id-looking diagnostic text", async () => {
    const row = reviewRow({
      id: 5,
      kind: "correction",
      reason: "diagnostic prefix mention",
      refs: {
        target_id: "cmt_struct11111111",
        target_type: "commitment",
        reason: "diagnostic text mentions seme_notaref111111 and semn_notaref111111",
        diagnostic: { message: "another ep_notaref111111 value" },
      },
    });
    installReviewFetch([row]);

    renderReview(makeLiveSource(), { inspector: true });

    expect(await screen.findByText("diagnostic prefix mention")).toBeInTheDocument();
    const filter = screen.getByLabelText("affected type filter") as HTMLSelectElement;
    const values = Array.from(filter.options).map((option) => option.value);

    expect(values).toContain("commitment");
    expect(values).not.toContain("semantic_edge");
    expect(values).not.toContain("semantic_node");
    expect(values).not.toContain("episode");
  });

  it("routes each review kind through its sanctioned endpoint", async () => {
    const rows = [
      reviewRow({ id: 1, kind: "correction", reason: "correction route" }),
      reviewRow({ id: 2, kind: "belief_revision", reason: "belief route" }),
      reviewRow({
        id: 3,
        kind: "creator_directive_reconciliation",
        reason: "directive route",
        refs: { directive_ids: ["cdir_route1111111"] },
      }),
      reviewRow({ id: 4, kind: "new_insight", reason: "generic route" }),
    ];
    const fetchMock = vi.fn((request: RequestInfo | URL, init?: RequestInit) => {
      const url = new URL(String(request), "http://test.invalid");
      if (url.pathname === "/api/reviews" && init?.method === undefined) {
        return Promise.resolve(jsonResponse({ rows }));
      }
      if (url.pathname === "/api/creator-directives") {
        return Promise.resolve(jsonResponse({ directives: [] }));
      }
      if (url.pathname === "/api/commitments") {
        return Promise.resolve(jsonResponse({ commitments: [] }));
      }
      if (
        url.pathname === "/api/correction/reviews/1" ||
        url.pathname === "/api/dream/review/2" ||
        url.pathname === "/api/reviews/3/creator-directive-reconciliation" ||
        url.pathname === "/api/reviews/4"
      ) {
        return Promise.resolve(jsonResponse({ ...rows[0], resolved_at: Date.now() }));
      }
      return Promise.resolve(new Response("not found", { status: 404 }));
    });
    vi.stubGlobal("fetch", fetchMock);

    renderReview();

    expect(await screen.findByText("correction route")).toBeInTheDocument();
    fireEvent.click(screen.getByRole("button", { name: "accept" }));
    await waitFor(() => {
      expect(
        fetchMock.mock.calls.some(
          ([request, init]) =>
            requestPath(request) === "/api/correction/reviews/1" &&
            init?.method === "PATCH" &&
            (requestBody(init) as { action?: string }).action === "accept",
        ),
      ).toBe(true);
    });

    clickQueueRow("belief route");
    fireEvent.click(screen.getByRole("button", { name: "dismiss" }));
    await waitFor(() => {
      expect(
        fetchMock.mock.calls.some(
          ([request, init]) =>
            requestPath(request) === "/api/dream/review/2" &&
            init?.method === "PATCH" &&
            (requestBody(init) as { action?: string }).action === "dismiss",
        ),
      ).toBe(true);
    });

    clickQueueRow("directive route");
    fireEvent.click(screen.getByRole("button", { name: "keep both" }));
    await waitFor(() => {
      expect(
        fetchMock.mock.calls.some(
          ([request, init]) =>
            requestPath(request) === "/api/reviews/3/creator-directive-reconciliation" &&
            init?.method === "POST" &&
            (requestBody(init) as { action?: string }).action === "keep",
        ),
      ).toBe(true);
    });

    clickQueueRow("generic route");
    fireEvent.click(screen.getByRole("button", { name: "accept" }));
    await waitFor(() => {
      expect(
        fetchMock.mock.calls.some(
          ([request, init]) =>
            requestPath(request) === "/api/reviews/4" &&
            init?.method === "PATCH" &&
            (requestBody(init) as { action?: string }).action === "accept",
        ),
      ).toBe(true);
    });
  });

  it("requires confirmation before destructive review actions", async () => {
    const row = reviewRow({
      id: 10,
      kind: "contradiction",
      reason: "destructive confirm route",
      refs: { node_ids: ["semn_confirm111111", "semn_confirm222222"] },
    });
    const fetchMock = installReviewFetch([row]);

    renderReview();

    expect(await screen.findByText("destructive confirm route")).toBeInTheDocument();
    fireEvent.click(screen.getByRole("button", { name: "invalidate" }));

    expect(
      fetchMock.mock.calls.some(
        ([request, init]) => requestPath(request) === "/api/reviews/10" && init?.method === "PATCH",
      ),
    ).toBe(false);

    fireEvent.click(screen.getByRole("button", { name: "confirm invalidate" }));

    await waitFor(() => {
      const call = fetchMock.mock.calls.find(
        ([request, init]) => requestPath(request) === "/api/reviews/10" && init?.method === "PATCH",
      );
      expect(call).toBeDefined();
      expect(requestBody(call?.[1])).toMatchObject({
        action: "invalidate",
        winner_node_id: "semn_confirm111111",
      });
    });
  });

  it("keeps the Inspector open when Escape closes a Review confirmation modal", async () => {
    const nodeId = "semn_escape11111111";
    const row = reviewRow({
      id: 11,
      kind: "correction",
      reason: "escape confirm route",
      refs: { target_id: nodeId, target_type: "semantic_node" },
    });
    const fetchMock = vi.fn((request: RequestInfo | URL, init?: RequestInit) => {
      const url = new URL(String(request), "http://test.invalid");
      if (url.pathname === "/api/reviews" && init?.method === undefined) {
        return Promise.resolve(jsonResponse({ rows: [row] }));
      }
      if (url.pathname === "/api/creator-directives") {
        return Promise.resolve(jsonResponse({ directives: [] }));
      }
      if (url.pathname === "/api/commitments") {
        return Promise.resolve(jsonResponse({ commitments: [] }));
      }
      if (url.pathname === `/api/semantic/nodes/${nodeId}`) {
        return Promise.resolve(
          jsonResponse({
            node: {
              id: nodeId,
              kind: "proposition",
              label: "Escape node",
              description: "Node kept open while confirm closes.",
              domain: "test",
              aliases: [],
              confidence: 0.8,
              status: "active",
              source_episode_ids: [],
              source_count: 0,
              created_at: 1,
              updated_at: 1,
            },
          }),
        );
      }
      return Promise.resolve(new Response("not found", { status: 404 }));
    });
    vi.stubGlobal("fetch", fetchMock);

    renderReview(makeLiveSource(), { inspector: true });

    expect(await screen.findByText("escape confirm route")).toBeInTheDocument();
    fireEvent.click(screen.getByRole("button", { name: `jump to ${nodeId}` }));
    expect(
      await screen.findByRole("dialog", { name: "Semantic node inspector" }),
    ).toBeInTheDocument();

    fireEvent.click(screen.getByRole("button", { name: "reject" }));
    expect(screen.getByText(/Confirm reject for review 11/)).toBeInTheDocument();

    fireEvent.keyDown(window, { key: "Escape" });

    await waitFor(() =>
      expect(screen.queryByText(/Confirm reject for review 11/)).not.toBeInTheDocument(),
    );
    expect(screen.getByRole("dialog", { name: "Semantic node inspector" })).toBeInTheDocument();
  });

  it("renders commitment reconciliation comparison with structural mismatch highlighting", async () => {
    const row = reviewRow({
      id: 20,
      kind: "commitment_reconciliation",
      reason: "commitments need comparison",
      refs: { commitment_ids: ["cmt_compare111111", "cmt_compare222222"] },
    });
    const fetchMock = vi.fn((request: RequestInfo | URL, init?: RequestInit) => {
      const url = new URL(String(request), "http://test.invalid");
      if (url.pathname === "/api/reviews" && init?.method === undefined) {
        return Promise.resolve(jsonResponse({ rows: [row] }));
      }
      if (url.pathname === "/api/creator-directives") {
        return Promise.resolve(jsonResponse({ directives: [] }));
      }
      if (url.pathname === "/api/commitments") {
        return Promise.resolve(
          jsonResponse({
            commitments: [
              commitment("cmt_compare111111", "Critical launch boundary", "critical", "alice"),
              commitment("cmt_compare222222", "Advisory launch boundary", "advisory", "bob"),
            ],
          }),
        );
      }
      return Promise.resolve(new Response("not found", { status: 404 }));
    });
    vi.stubGlobal("fetch", fetchMock);

    renderReview();

    expect(await screen.findByText("Critical launch boundary")).toBeInTheDocument();
    expect(screen.getByText("Advisory launch boundary")).toBeInTheDocument();
    expect(screen.getByText(/commitment comparison is read-only context/)).toBeInTheDocument();
    expect(
      screen.getAllByText("enforcement").some((element) => element.classList.contains("warn")),
    ).toBe(true);
    expect(screen.getAllByText("critical").length).toBeGreaterThan(0);
    expect(screen.getAllByText("advisory").length).toBeGreaterThan(0);
  });

  it("runs the Correction Lab why, correct, forget, and invalidate flows", async () => {
    let reviewRows: ReviewRow[] = [];
    const queuedReview = reviewRow({
      id: 77,
      kind: "correction",
      reason: "queued lab correction",
      refs: { target_id: "goal_lab11111111", target_type: "goal" },
    });
    const fetchMock = vi.fn((request: RequestInfo | URL, init?: RequestInit) => {
      const url = new URL(String(request), "http://test.invalid");
      if (url.pathname === "/api/reviews" && init?.method === undefined) {
        return Promise.resolve(jsonResponse({ rows: reviewRows }));
      }
      if (url.pathname === "/api/creator-directives") {
        return Promise.resolve(jsonResponse({ directives: [] }));
      }
      if (url.pathname === "/api/commitments") {
        return Promise.resolve(jsonResponse({ commitments: [] }));
      }
      if (url.pathname.endsWith("/why")) {
        return Promise.resolve(jsonResponse({ provenance: { source: "lab test" } }));
      }
      if (url.pathname === "/api/correction/goal_lab11111111/correct" && init?.method === "POST") {
        reviewRows = [queuedReview];
        return Promise.resolve(jsonResponse(queuedReview));
      }
      if (url.pathname === "/api/correction/ep_forget1111111/forget" && init?.method === "POST") {
        return Promise.resolve(
          jsonResponse({
            id: "ep_forget1111111",
            target_type: "episode",
            archived: true,
            provenance: { kind: "manual" },
          }),
        );
      }
      if (
        url.pathname === "/api/correction/semantic-edges/seme_labedge111111/invalidate" &&
        init?.method === "POST"
      ) {
        return Promise.resolve(jsonResponse({ id: "seme_labedge111111", invalidated: true }));
      }
      return Promise.resolve(new Response("not found", { status: 404 }));
    });
    vi.stubGlobal("fetch", fetchMock);

    renderReview();

    fireEvent.click(await screen.findByRole("tab", { name: "lab" }));
    fireEvent.change(screen.getByLabelText("object id"), { target: { value: "not_correctable" } });
    expect(screen.getByText("not a correctable id")).toBeInTheDocument();

    fireEvent.change(screen.getByLabelText("object id"), { target: { value: "goal_lab11111111" } });
    expect(await screen.findByText("provenance")).toBeInTheDocument();
    fireEvent.change(screen.getByLabelText("json patch"), {
      target: { value: '{"description":"updated"}' },
    });
    fireEvent.change(screen.getByLabelText("reason"), { target: { value: "operator reason" } });
    fireEvent.click(screen.getByRole("button", { name: "queue correction" }));

    expect((await screen.findAllByText("queued lab correction")).length).toBeGreaterThan(0);
    expect(
      screen
        .getAllByText("queued lab correction")
        .some((element) => element.closest(".review-evidence")),
    ).toBe(true);
    expect(
      fetchMock.mock.calls.some(
        ([request, init]) =>
          requestPath(request) === "/api/correction/goal_lab11111111/correct" &&
          init?.method === "POST" &&
          (requestBody(init) as { patch?: unknown }).patch !== undefined,
      ),
    ).toBe(true);

    fireEvent.click(screen.getByRole("tab", { name: "lab" }));
    fireEvent.change(screen.getByLabelText("object id"), { target: { value: "ep_forget1111111" } });
    expect(await screen.findByText("correctable episode")).toBeInTheDocument();
    fireEvent.click(screen.getByRole("button", { name: "forget" }));
    expect(
      fetchMock.mock.calls.some(
        ([request, init]) =>
          requestPath(request) === "/api/correction/ep_forget1111111/forget" &&
          init?.method === "POST",
      ),
    ).toBe(false);
    fireEvent.click(screen.getByRole("button", { name: "confirm forget" }));
    await waitFor(() => {
      expect(
        fetchMock.mock.calls.some(
          ([request, init]) =>
            requestPath(request) === "/api/correction/ep_forget1111111/forget" &&
            init?.method === "POST",
        ),
      ).toBe(true);
    });

    fireEvent.change(screen.getByLabelText("object id"), {
      target: { value: "seme_labedge111111" },
    });
    expect(await screen.findByText("correctable semantic edge")).toBeInTheDocument();
    fireEvent.click(screen.getByRole("button", { name: "invalidate" }));
    expect(
      fetchMock.mock.calls.some(
        ([request, init]) =>
          requestPath(request) === "/api/correction/semantic-edges/seme_labedge111111/invalidate" &&
          init?.method === "POST",
      ),
    ).toBe(false);
    fireEvent.click(screen.getByRole("button", { name: "confirm invalidate" }));
    await waitFor(() => {
      expect(
        fetchMock.mock.calls.some(
          ([request, init]) =>
            requestPath(request) ===
              "/api/correction/semantic-edges/seme_labedge111111/invalidate" &&
            init?.method === "POST",
        ),
      ).toBe(true);
    });
  });

  it("clears conflicting queue filters when the Correction Lab links a returned review", async () => {
    let reviewRows: ReviewRow[] = [
      reviewRow({
        id: 70,
        kind: "contradiction",
        reason: "active contradiction filter",
        refs: { node_ids: ["semn_filtervisible1", "semn_filtervisible2"] },
      }),
    ];
    const queuedReview = reviewRow({
      id: 78,
      kind: "correction",
      reason: "linked lab correction",
      refs: { target_id: "goal_link11111111", target_type: "goal" },
    });
    const fetchMock = vi.fn((request: RequestInfo | URL, init?: RequestInit) => {
      const url = new URL(String(request), "http://test.invalid");
      if (url.pathname === "/api/reviews" && init?.method === undefined) {
        return Promise.resolve(jsonResponse({ rows: reviewRows }));
      }
      if (url.pathname === "/api/creator-directives") {
        return Promise.resolve(jsonResponse({ directives: [] }));
      }
      if (url.pathname === "/api/commitments") {
        return Promise.resolve(jsonResponse({ commitments: [] }));
      }
      if (url.pathname === "/api/correction/goal_link11111111/why") {
        return Promise.resolve(jsonResponse({ provenance: { source: "link test" } }));
      }
      if (url.pathname === "/api/correction/goal_link11111111/correct" && init?.method === "POST") {
        reviewRows = [queuedReview];
        return Promise.resolve(jsonResponse(queuedReview));
      }
      return Promise.resolve(new Response("not found", { status: 404 }));
    });
    vi.stubGlobal("fetch", fetchMock);

    renderReview();

    expect(await screen.findByText("active contradiction filter")).toBeInTheDocument();
    fireEvent.change(screen.getByLabelText("kind filter"), { target: { value: "contradiction" } });
    fireEvent.click(screen.getByRole("tab", { name: "lab" }));
    fireEvent.change(screen.getByLabelText("object id"), {
      target: { value: "goal_link11111111" },
    });
    expect(await screen.findByText("provenance")).toBeInTheDocument();
    fireEvent.change(screen.getByLabelText("json patch"), {
      target: { value: '{"description":"linked"}' },
    });
    fireEvent.click(screen.getByRole("button", { name: "queue correction" }));

    expect((await screen.findAllByText("linked lab correction")).length).toBeGreaterThan(0);
    expect((screen.getByLabelText("kind filter") as HTMLSelectElement).value).toBe("all");
    expect(
      screen
        .getAllByText("linked lab correction")
        .some((element) => element.closest(".review-evidence")),
    ).toBe(true);
  });

  it("refetches reviews on dream completion and borg reset frames", async () => {
    const live = makeLiveSource();
    const fetchMock = installReviewFetch([]);

    renderWithInspector(
      <LiveEventsProvider value={live.live()}>
        <ReviewScreen />
      </LiveEventsProvider>,
    );

    await waitFor(() => {
      expect(
        fetchMock.mock.calls.filter(([request]) => requestPath(request) === "/api/reviews"),
      ).toHaveLength(1);
    });

    act(() => {
      live.emit({
        type: "dream:process:completed",
        ts: 1,
        process: "curator",
        run_id: "run_review",
        phase: "apply",
        errors: 0,
        candidates_accepted: 0,
      });
    });

    await waitFor(() => {
      expect(
        fetchMock.mock.calls.filter(([request]) => requestPath(request) === "/api/reviews"),
      ).toHaveLength(2);
    });

    act(() => {
      live.emit({ type: "borg:reset", ts: 2 });
    });

    await waitFor(() => {
      expect(
        fetchMock.mock.calls.filter(([request]) => requestPath(request) === "/api/reviews"),
      ).toHaveLength(3);
    });
  });
});

function commitment(
  id: string,
  text: string,
  enforcement: "critical" | "advisory",
  audience: string,
) {
  return {
    id,
    text,
    type: "rule",
    kind: "boundary",
    enforcement_class: enforcement,
    critical_domain: enforcement === "critical" ? "audience_scope" : null,
    state: "active",
    priority: enforcement === "critical" ? 10 : 1,
    directive_family: "launch",
    audience,
    made_to: null,
    about: null,
    committed_by: null,
    source: "manual",
    source_stream_entry_ids: [],
    created_at: 1,
    expires_at: null,
    expired_at: null,
    revoked_at: null,
    revoked_reason: null,
    superseded_by_id: null,
    canonicalized_by_artifact_entry_id: null,
    last_reinforced_at: 1,
  };
}
