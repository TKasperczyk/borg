import { fireEvent, render, screen, waitFor, within } from "@testing-library/react";
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
  it("adds a value from the identity modal and refetches identity", async () => {
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

    render(<IdentityScreen />);

    fireEvent.click(await screen.findByLabelText("add value"));
    fireEvent.change(screen.getByLabelText("name"), { target: { value: "care" } });
    fireEvent.change(screen.getByLabelText("description"), {
      target: { value: "care about clean operator workflows" },
    });
    fireEvent.click(screen.getByRole("button", { name: "save" }));

    await waitFor(() => {
      expect(
        fetchMock.mock.calls.some((call) => requestPath(call[0]) === "/api/identity/values"),
      ).toBe(true);
      expect(
        fetchMock.mock.calls.filter((call) => requestPath(call[0]) === "/api/identity"),
      ).toHaveLength(2);
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

    render(<IdentityScreen />);

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

  it("plans then applies dream maintenance from the header button", async () => {
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

    render(
      <LiveEventsProvider value={liveSource()}>
        <DreamScreen />
      </LiveEventsProvider>,
    );

    fireEvent.click(await screen.findByLabelText("apply dream"));
    expect(await screen.findByText(/Apply 2 changes from 1 processes/)).toBeInTheDocument();
    fireEvent.click(within(screen.getByRole("dialog")).getByRole("button", { name: "apply" }));

    await waitFor(() => {
      expect(fetchMock.mock.calls.some((call) => requestPath(call[0]) === "/api/dream/apply")).toBe(
        true,
      );
      expect(fetchMock.mock.calls.some((call) => requestPath(call[0]) === "/api/dream/audit")).toBe(
        true,
      );
    });
  });
});
