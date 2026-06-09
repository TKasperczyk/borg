import { fireEvent, screen, waitFor, within } from "@testing-library/react";
import { renderWithInspector } from "../test/inspector";
import { afterEach, describe, expect, it, vi } from "vitest";

import type { LiveEvents, LiveEventHandler } from "../hooks/use-live-events";
import { LiveEventsProvider } from "../hooks/live-context";
import { DreamScreen } from "./Dream";
import { IdentityScreen } from "./Identity";

function jsonResponse(body: unknown): Response {
  return new Response(JSON.stringify(body), {
    status: 200,
    headers: { "Content-Type": "application/json" },
  });
}

function identityResponse() {
  return {
    values: [],
    goals: [
      {
        id: "goal_1111111111111111",
        description: "ship the operator surface",
        priority: 1,
        status: "active",
        progress_notes: null,
        created_at: 1,
        target_at: null,
      },
    ],
    traits: [],
    open_questions: [
      {
        id: "oq_1111111111111111",
        question: "what should Tom inspect next?",
        urgency: 0.7,
        status: "open",
        goal_id: null,
        source: "ruminator",
        created_at: 1,
        last_touched: 2,
        resolved_at: null,
        abandoned_at: null,
        abandoned_reason: null,
        resolution_note: null,
        unresolved_rumination_ticks: 1,
        last_ruminated_at: null,
      },
    ],
    growth_markers: [],
    periods: [],
    open_question_events: [],
  };
}

function dreamStateResponse() {
  return {
    processes: [
      {
        name: "belief-reviser",
        description: "invalidate, weaken, contradict",
        last_run_at: 4,
        last_status: "ok",
        last_audit_id: 1,
        budget: null,
        enabled: true,
      },
    ],
    schedule: [],
    audit_rows: [],
    belief_revision_rows: [
      {
        id: 1,
        kind: "belief_revision",
        refs: { target_type: "semantic_node", target_id: "semn_1111111111111111" },
        reason: "dependency invalidated",
        created_at: 4,
        resolved_at: null,
        resolution: null,
      },
    ],
    scheduler: {
      enabled: true,
      light_interval_ms: 1,
      heavy_interval_ms: 1,
      light_processes: [],
      heavy_processes: ["belief-reviser"],
      process_budgets: {},
    },
  };
}

function liveSource(): LiveEvents {
  const handlers = new Set<LiveEventHandler>();
  return {
    wsState: "live",
    connectionCount: 1,
    subscribe: (handler) => {
      handlers.add(handler);
      return () => handlers.delete(handler);
    },
  };
}

function requestPath(request: RequestInfo | URL): string {
  return new URL(String(request), "http://test.invalid").pathname;
}

afterEach(() => {
  vi.unstubAllGlobals();
});

describe("operator actions", () => {
  it("blocks a direct identity create until risk acknowledgment, then refetches identity", async () => {
    const fetchMock = vi.fn((request: RequestInfo | URL, init?: RequestInit) => {
      const path = requestPath(request);
      if (path === "/api/identity" && init?.method === undefined) {
        return Promise.resolve(jsonResponse(identityResponse()));
      }
      if (path === "/api/identity/values" && init?.method === "POST") {
        return Promise.resolve(
          jsonResponse({
            id: "val_1111111111111111",
            label: "care",
            description: "care about clean operator workflows",
            priority: 0,
            created_at: 1,
            last_affirmed: null,
            state: "candidate",
            confidence: 0.5,
            support_count: 0,
            contradiction_count: 0,
            evidence_episode_ids: [],
          }),
        );
      }
      return Promise.resolve(new Response("{}", { status: 404 }));
    });
    vi.stubGlobal("fetch", fetchMock);

    renderWithInspector(<IdentityScreen />);

    fireEvent.click(await screen.findByLabelText("add value"));
    fireEvent.change(screen.getByLabelText("name"), { target: { value: "care" } });
    fireEvent.change(screen.getByLabelText("description"), {
      target: { value: "care about clean operator workflows" },
    });
    const writeButton = screen.getByRole("button", { name: "write live self-band" });
    expect(writeButton).toBeDisabled();

    fireEvent.click(screen.getByLabelText(/acknowledge this direct live self-band write/i));
    expect(writeButton).not.toBeDisabled();
    fireEvent.click(writeButton);

    await waitFor(() => {
      expect(
        fetchMock.mock.calls.some((call) => requestPath(call[0]) === "/api/identity/values"),
      ).toBe(true);
      expect(
        fetchMock.mock.calls.filter((call) => requestPath(call[0]) === "/api/identity"),
      ).toHaveLength(2);
    });
  });

  it("renders open question events newest-first and collapses empty identity sections", async () => {
    const fetchMock = vi.fn((request: RequestInfo | URL, init?: RequestInit) => {
      const path = requestPath(request);
      if (path === "/api/identity" && init?.method === undefined) {
        return Promise.resolve(
          jsonResponse({
            ...identityResponse(),
            open_question_events: [
              {
                id: 1,
                record_type: "open_question",
                record_id: "oq_old",
                action: "create",
                old_value: null,
                new_value: {
                  status: "open",
                  urgency: 0.4,
                  question: "older question",
                },
                reason: null,
                provenance: { kind: "online", process: "reflector" },
                review_item_id: null,
                overwrite_without_review: false,
                ts: 10,
              },
              {
                id: 2,
                record_type: "open_question",
                record_id: "oq_merge",
                action: "update",
                old_value: {
                  status: "open",
                  urgency: 0.4,
                  question: "duplicate question",
                },
                new_value: {
                  status: "open",
                  urgency: 0.7,
                  question: "duplicate question",
                },
                reason: "open_question_duplicate_merge",
                provenance: { kind: "offline", process: "ruminator" },
                review_item_id: null,
                overwrite_without_review: false,
                ts: 20,
              },
              {
                id: 3,
                record_type: "open_question",
                record_id: "oq_new",
                action: "resolve",
                old_value: {
                  status: "open",
                  urgency: 0.8,
                  question: "newer question",
                },
                new_value: {
                  status: "resolved",
                  urgency: 0.8,
                  question: "newer question",
                },
                reason: "operator resolution",
                provenance: { kind: "manual" },
                review_item_id: 42,
                overwrite_without_review: true,
                ts: 30,
              },
            ],
          }),
        );
      }
      return Promise.resolve(new Response("{}", { status: 404 }));
    });
    vi.stubGlobal("fetch", fetchMock);

    renderWithInspector(<IdentityScreen />);

    fireEvent.click(await screen.findByRole("button", { name: "all events" }));

    const eventSection = await screen.findByLabelText("open question events history");
    const rows = within(eventSection).getAllByTestId("identity-event-row");
    expect(rows).toHaveLength(3);
    expect(rows[0]).toHaveTextContent("oq_new");
    expect(rows[0]).toHaveTextContent("without review gate");
    expect(rows[0]).toHaveTextContent("review 42");
    expect(rows[1]).toHaveTextContent("oq_merge");
    expect(rows[1]).toHaveTextContent("duplicate merge");
    expect(rows[1]).toHaveTextContent("open_question_duplicate_merge");
    expect(rows[2]).toHaveTextContent("oq_old");

    expect(screen.getByLabelText("established values")).toHaveTextContent("no established values");
    expect(screen.getByLabelText("candidate values")).toHaveTextContent("no candidate values");
    expect(screen.getByLabelText("empty growth markers")).toHaveTextContent(
      "no growth markers recorded",
    );
    expect(screen.getByLabelText("empty autobiographical periods")).toHaveTextContent(
      "no autobiographical periods recorded",
    );
  });

  it("keeps the correction path queued without direct-write acknowledgment", async () => {
    const fetchMock = vi.fn((request: RequestInfo | URL, init?: RequestInit) => {
      const path = requestPath(request);
      if (path === "/api/identity" && init?.method === undefined) {
        return Promise.resolve(jsonResponse(identityResponse()));
      }
      if (path === "/api/correction/goal_1111111111111111/correct" && init?.method === "POST") {
        return Promise.resolve(
          jsonResponse({
            id: 1,
            kind: "correction",
            refs: {},
            reason: "queued correction",
            created_at: 1,
            resolved_at: null,
            resolution: null,
          }),
        );
      }
      return Promise.resolve(new Response("{}", { status: 404 }));
    });
    vi.stubGlobal("fetch", fetchMock);

    renderWithInspector(<IdentityScreen />);

    fireEvent.click((await screen.findAllByRole("button", { name: "correct" }))[0]!);
    expect(screen.queryByLabelText(/acknowledge this direct live self-band write/i)).toBeNull();
    fireEvent.click(screen.getByRole("button", { name: "queue" }));

    await waitFor(() => {
      const correctionCall = fetchMock.mock.calls.find(
        (call) => requestPath(call[0]) === "/api/correction/goal_1111111111111111/correct",
      );
      expect(correctionCall).toBeDefined();
      expect(JSON.parse(String((correctionCall?.[1] as RequestInit | undefined)?.body))).toEqual({
        patch: { description: "ship the operator surface" },
      });
    });
  });

  it("bumps an open question from the identity row action", async () => {
    const fetchMock = vi.fn((request: RequestInfo | URL, init?: RequestInit) => {
      const path = requestPath(request);
      if (path === "/api/identity" && init?.method === undefined) {
        return Promise.resolve(jsonResponse(identityResponse()));
      }
      if (path === "/api/identity/open-questions/oq_1111111111111111" && init?.method === "PATCH") {
        return Promise.resolve(
          jsonResponse({ ...identityResponse().open_questions[0], urgency: 0.8 }),
        );
      }
      return Promise.resolve(new Response("{}", { status: 404 }));
    });
    vi.stubGlobal("fetch", fetchMock);

    renderWithInspector(<IdentityScreen />);

    fireEvent.click(await screen.findByRole("button", { name: "bump" }));

    await waitFor(() => {
      const patchCall = fetchMock.mock.calls.find(
        (call) => requestPath(call[0]) === "/api/identity/open-questions/oq_1111111111111111",
      );
      expect(patchCall).toBeDefined();
      expect(JSON.parse(String((patchCall?.[1] as RequestInit | undefined)?.body))).toEqual({
        action: "bump",
      });
    });
  });

  it("applies dream maintenance directly from the header (no upstream plan)", async () => {
    const fetchMock = vi.fn((request: RequestInfo | URL, init?: RequestInit) => {
      const path = requestPath(request);
      if (path === "/api/dream/state" && init?.method === undefined) {
        return Promise.resolve(jsonResponse(dreamStateResponse()));
      }
      if (path === "/api/dream/plan" && init?.method === "POST") {
        return Promise.resolve(
          jsonResponse({
            plan_id: "demo_plan_1",
            processes: [
              {
                name: "belief-reviser",
                would_change: true,
                summary: "2 changes",
                budget_used: 12,
                changes: [
                  {
                    process: "belief-reviser",
                    action: "enqueue_review",
                    targets: { target_id: "semn_1" },
                  },
                  {
                    process: "belief-reviser",
                    action: "regrade_belief_revision",
                    targets: { review_id: 1 },
                  },
                ],
                errors: [],
                budget_exhausted: false,
              },
            ],
            total_budget_used: 12,
            changes: 2,
          }),
        );
      }
      if (path === "/api/dream/apply" && init?.method === "POST") {
        return Promise.resolve(
          jsonResponse({
            run_id: "run_1",
            applied: [{ name: "belief-reviser", audit_id: 1, audit_ids: [1], changes: 2 }],
            failed: [],
            duration_ms: 10,
            total_budget_used: 12,
          }),
        );
      }
      if (path === "/api/dream/audit" && init?.method === undefined) {
        return Promise.resolve(jsonResponse({ rows: [] }));
      }
      return Promise.resolve(new Response("{}", { status: 404 }));
    });
    vi.stubGlobal("fetch", fetchMock);

    renderWithInspector(
      <LiveEventsProvider value={liveSource()}>
        <DreamScreen />
      </LiveEventsProvider>,
    );

    fireEvent.click(await screen.findByLabelText("apply dream"));
    // Apply opens the confirm modal IMMEDIATELY with a generic "run the
    // dream cycle?" prompt rather than blocking on a dry-run plan call
    // (which takes ~1 minute and made the modal feel non-responsive).
    expect(await screen.findByText(/Run the dream cycle\?/i)).toBeInTheDocument();
    fireEvent.click(within(screen.getByRole("dialog")).getByRole("button", { name: "apply" }));

    await waitFor(() => {
      expect(fetchMock.mock.calls.some((call) => requestPath(call[0]) === "/api/dream/apply")).toBe(
        true,
      );
      expect(fetchMock.mock.calls.some((call) => requestPath(call[0]) === "/api/dream/audit")).toBe(
        true,
      );
    });
    const applyCall = fetchMock.mock.calls.find(
      (call) => requestPath(call[0]) === "/api/dream/apply",
    );
    expect(JSON.parse(String((applyCall?.[1] as RequestInit | undefined)?.body))).toEqual({});
    // The plan endpoint must NOT have been hit by the apply-from-header path.
    expect(fetchMock.mock.calls.some((call) => requestPath(call[0]) === "/api/dream/plan")).toBe(
      false,
    );
  });
});
