import { act, fireEvent, screen, waitFor, within } from "@testing-library/react";
import { afterEach, describe, expect, it, vi } from "vitest";

import type { DreamStateResponse, LiveFrame, MaintenanceAuditRow, WsState } from "../../api/types";
import { LiveEventsProvider } from "../../hooks/live-context";
import type { LiveEventHandler, LiveEvents } from "../../hooks/use-live-events";
import { renderWithInspector } from "../../test/inspector";
import { DreamScreen } from ".";

function jsonResponse(body: unknown): Response {
  return new Response(JSON.stringify(body), {
    status: 200,
    headers: { "Content-Type": "application/json" },
  });
}

function requestPath(request: RequestInfo | URL): string {
  return new URL(String(request), "http://test.invalid").pathname;
}

function requestBody(call: [RequestInfo | URL, RequestInit | undefined]): unknown {
  return JSON.parse(String(call[1]?.body));
}

function deferred<T>(): { promise: Promise<T>; resolve: (value: T) => void } {
  let resolve!: (value: T) => void;
  const promise = new Promise<T>((nextResolve) => {
    resolve = nextResolve;
  });
  return { promise, resolve };
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

function dreamState(input: Partial<DreamStateResponse> = {}): DreamStateResponse {
  return {
    processes: [],
    pending_extraction_episodes: 0,
    schedule: [],
    dream_reports: [],
    audit_rows: [],
    belief_revision_rows: [],
    scheduler: {
      enabled: true,
      light_interval_ms: 1000,
      heavy_interval_ms: 60000,
      light_processes: ["consolidator"],
      heavy_processes: ["belief-reviser"],
      process_budgets: {},
    },
    ...input,
  };
}

function installDreamFetch(
  state: DreamStateResponse,
  extra?: (url: URL, init?: RequestInit) => Promise<Response> | undefined,
): ReturnType<typeof vi.fn> {
  const fetchMock = vi.fn((request: RequestInfo | URL, init?: RequestInit) => {
    const url = new URL(String(request), "http://test.invalid");
    if (url.pathname === "/api/dream/state") {
      return Promise.resolve(jsonResponse(state));
    }

    const extraResponse = extra?.(url, init);
    if (extraResponse !== undefined) {
      return extraResponse;
    }

    return Promise.resolve(new Response("not found", { status: 404 }));
  });
  vi.stubGlobal("fetch", fetchMock);
  return fetchMock;
}

afterEach(() => {
  vi.unstubAllGlobals();
});

describe("DreamScreen Dream Ops", () => {
  it("scrolls and focuses the process selected by the route", async () => {
    const live = makeLiveSource();
    const scrollIntoView = vi.fn();
    const originalScrollIntoView = window.HTMLElement.prototype.scrollIntoView;
    Object.defineProperty(window.HTMLElement.prototype, "scrollIntoView", {
      configurable: true,
      value: scrollIntoView,
    });
    vi.stubGlobal(
      "matchMedia",
      vi.fn((query: string) => ({
        matches: query === "(prefers-reduced-motion: reduce)",
        media: query,
        onchange: null,
        addEventListener: vi.fn(),
        removeEventListener: vi.fn(),
        addListener: vi.fn(),
        removeListener: vi.fn(),
        dispatchEvent: vi.fn(),
      })),
    );
    installDreamFetch(
      dreamState({
        processes: [
          {
            name: "curator",
            description: "Curates stale memories.",
            last_run_at: null,
            last_status: "ok",
            last_audit_id: null,
            budget: null,
            enabled: true,
          },
        ],
      }),
    );

    try {
      renderWithInspector(
        <LiveEventsProvider value={live.live()}>
          <DreamScreen initialProcess="curator" />
        </LiveEventsProvider>,
      );

      const card = await screen.findByTestId("dream-process-card-curator");
      await waitFor(() =>
        expect(scrollIntoView).toHaveBeenCalledWith({
          block: "center",
          inline: "nearest",
          behavior: "auto",
        }),
      );
      expect(card).toHaveFocus();
    } finally {
      Object.defineProperty(window.HTMLElement.prototype, "scrollIntoView", {
        configurable: true,
        value: originalScrollIntoView,
      });
    }
  });

  it("plans a selected process subset with budget, applies the plan id, and renders apply results", async () => {
    const live = makeLiveSource();
    const fetchMock = installDreamFetch(dreamState(), (url, init) => {
      if (url.pathname === "/api/dream/plan" && init?.method === "POST") {
        return Promise.resolve(
          jsonResponse({
            plan_id: "plan_1",
            processes: [
              {
                name: "consolidator",
                would_change: true,
                summary: "would consolidate one family",
                budget_used: 80,
                changes: [
                  {
                    process: "consolidator",
                    action: "merge_family",
                    targets: { episode_id: "ep_plan11111111" },
                    preview: { family_id: "cfam_plan111111" },
                  },
                ],
                errors: [],
                budget_exhausted: false,
              },
              {
                name: "curator",
                would_change: false,
                summary: "stopped at budget",
                budget_used: 43,
                changes: [],
                errors: [
                  {
                    process: "curator",
                    message: "budget stopped curator",
                    code: "BUDGET_EXHAUSTED",
                  },
                ],
                budget_exhausted: true,
              },
            ],
            total_budget_used: 123,
            changes: 1,
          }),
        );
      }

      if (url.pathname === "/api/dream/apply" && init?.method === "POST") {
        return Promise.resolve(
          jsonResponse({
            run_id: "run_plan11111111",
            applied: [
              {
                name: "consolidator",
                audit_id: 7,
                audit_ids: [7, 8],
                changes: 2,
              },
            ],
            failed: [{ name: "curator", message: "curator failed", code: "CURATOR_FAIL" }],
            duration_ms: 1500,
            total_budget_used: 123,
          }),
        );
      }

      if (url.pathname === "/api/dream/audit") {
        return Promise.resolve(jsonResponse({ rows: [] }));
      }

      return undefined;
    });

    renderWithInspector(
      <LiveEventsProvider value={live.live()}>
        <DreamScreen />
      </LiveEventsProvider>,
    );

    await screen.findByText("plan/apply workbench");
    fireEvent.click(screen.getByRole("button", { name: "clear" }));
    fireEvent.click(screen.getByLabelText("include consolidator"));
    fireEvent.click(screen.getByLabelText("include curator"));
    fireEvent.change(screen.getByLabelText("budget"), { target: { value: "123" } });
    fireEvent.click(screen.getByRole("button", { name: "plan dream" }));

    expect(await screen.findByText("would consolidate one family")).toBeInTheDocument();
    expect(screen.getByText("total budget used: 123")).toBeInTheDocument();
    expect(screen.getAllByText("would change").length).toBeGreaterThan(0);
    expect(screen.getAllByText("budget exhausted").length).toBeGreaterThan(0);
    expect(screen.getByText("budget stopped curator")).toBeInTheDocument();

    const planCall = fetchMock.mock.calls.find(
      (call) => requestPath(call[0]) === "/api/dream/plan",
    );
    expect(planCall).toBeDefined();
    expect(requestBody(planCall as [RequestInfo | URL, RequestInit | undefined])).toEqual({
      processes: ["consolidator", "curator"],
      budget: 123,
    });

    fireEvent.click(screen.getByRole("button", { name: "apply plan" }));
    expect(await screen.findByText(/Apply 1 changes from 2 processes/i)).toBeInTheDocument();
    const confirmDialog = screen.getAllByRole("dialog").at(-1);
    expect(confirmDialog).toBeDefined();
    fireEvent.click(within(confirmDialog as HTMLElement).getByRole("button", { name: "apply" }));

    await waitFor(() => {
      expect(fetchMock.mock.calls.some((call) => requestPath(call[0]) === "/api/dream/apply")).toBe(
        true,
      );
    });

    const applyCall = fetchMock.mock.calls.find(
      (call) => requestPath(call[0]) === "/api/dream/apply",
    );
    expect(applyCall).toBeDefined();
    expect(requestBody(applyCall as [RequestInfo | URL, RequestInit | undefined])).toEqual({
      plan_id: "plan_1",
    });

    expect(await screen.findByText("apply result")).toBeInTheDocument();
    expect(screen.getByRole("button", { name: "jump to run_plan11111111" })).toBeInTheDocument();
    expect(screen.getByText("2 changes")).toBeInTheDocument();
    expect(screen.getByRole("button", { name: "jump to audit 7" })).toBeInTheDocument();
    expect(screen.getByRole("button", { name: "jump to audit 8" })).toBeInTheDocument();
    expect(screen.getByText("curator failed")).toBeInTheDocument();
    expect(screen.getByText(/code CURATOR_FAIL/)).toBeInTheDocument();
    expect(screen.getByText("123 budget used")).toBeInTheDocument();
    expect(screen.getAllByText("1500 ms").length).toBeGreaterThan(0);
  });

  it("marks a loaded plan stale after maintenance state changes and clears staleness on re-plan", async () => {
    const live = makeLiveSource();
    let planCalls = 0;
    installDreamFetch(dreamState(), (url, init) => {
      if (url.pathname === "/api/dream/plan" && init?.method === "POST") {
        planCalls += 1;
        return Promise.resolve(
          jsonResponse({
            plan_id: `plan_${planCalls}`,
            processes: [
              {
                name: "consolidator",
                would_change: true,
                summary: `fresh plan ${planCalls}`,
                budget_used: 10,
                changes: [
                  {
                    process: "consolidator",
                    action: "merge_family",
                    targets: { episode_id: "ep_stale111111" },
                  },
                ],
                errors: [],
                budget_exhausted: false,
              },
            ],
            total_budget_used: 10,
            changes: 1,
          }),
        );
      }

      return undefined;
    });

    renderWithInspector(
      <LiveEventsProvider value={live.live()}>
        <DreamScreen />
      </LiveEventsProvider>,
    );

    await screen.findByText("plan/apply workbench");
    fireEvent.click(screen.getByRole("button", { name: "clear" }));
    fireEvent.click(screen.getByLabelText("include consolidator"));
    fireEvent.click(screen.getByRole("button", { name: "plan dream" }));

    expect(await screen.findByText("fresh plan 1")).toBeInTheDocument();
    expect(screen.getByRole("button", { name: "apply plan" })).not.toBeDisabled();

    act(() => {
      live.emit({
        type: "maintenance:tick",
        ts: 8,
        cadence: "manual",
        status: "ok",
        processes: ["consolidator"],
        changed: true,
        changes: 1,
        errors: 0,
      });
    });

    expect(
      await screen.findByText("state changed since this plan -- re-plan to apply"),
    ).toBeInTheDocument();
    expect(screen.getByRole("button", { name: "apply plan" })).toBeDisabled();

    fireEvent.click(screen.getByRole("button", { name: "plan dream" }));

    expect(await screen.findByText("fresh plan 2")).toBeInTheDocument();
    expect(
      screen.queryByText("state changed since this plan -- re-plan to apply"),
    ).not.toBeInTheDocument();
    expect(screen.getByRole("button", { name: "apply plan" })).not.toBeDisabled();
  });

  it("uses the selected plan process count for apply progress", async () => {
    const live = makeLiveSource();
    const applyDeferred = deferred<Response>();
    installDreamFetch(dreamState(), (url, init) => {
      if (url.pathname === "/api/dream/plan" && init?.method === "POST") {
        return Promise.resolve(
          jsonResponse({
            plan_id: "plan_subset",
            processes: [
              {
                name: "consolidator",
                would_change: true,
                summary: "subset consolidator",
                budget_used: 5,
                changes: [],
                errors: [],
                budget_exhausted: false,
              },
              {
                name: "curator",
                would_change: false,
                summary: "subset curator",
                budget_used: 4,
                changes: [],
                errors: [],
                budget_exhausted: false,
              },
            ],
            total_budget_used: 9,
            changes: 0,
          }),
        );
      }

      if (url.pathname === "/api/dream/apply" && init?.method === "POST") {
        return applyDeferred.promise;
      }

      if (url.pathname === "/api/dream/audit") {
        return Promise.resolve(jsonResponse({ rows: [] }));
      }

      return undefined;
    });

    renderWithInspector(
      <LiveEventsProvider value={live.live()}>
        <DreamScreen />
      </LiveEventsProvider>,
    );

    await screen.findByText("plan/apply workbench");
    fireEvent.click(screen.getByRole("button", { name: "clear" }));
    fireEvent.click(screen.getByLabelText("include consolidator"));
    fireEvent.click(screen.getByLabelText("include curator"));
    fireEvent.click(screen.getByRole("button", { name: "plan dream" }));
    expect(await screen.findByText("subset consolidator")).toBeInTheDocument();

    fireEvent.click(screen.getByRole("button", { name: "apply plan" }));
    const confirmDialog = screen.getAllByRole("dialog").at(-1);
    expect(confirmDialog).toBeDefined();
    fireEvent.click(within(confirmDialog as HTMLElement).getByRole("button", { name: "apply" }));

    expect(await screen.findByText("planning 0/2")).toBeInTheDocument();

    applyDeferred.resolve(
      jsonResponse({
        run_id: "run_subset111111",
        applied: [],
        failed: [],
        duration_ms: 1,
        total_budget_used: 0,
      }),
    );

    expect(await screen.findByText("apply result")).toBeInTheDocument();
  });

  it("guards workbench validation and omits blank budget from plan requests", async () => {
    const live = makeLiveSource();
    const planBodies: unknown[] = [];
    installDreamFetch(dreamState(), (url, init) => {
      if (url.pathname === "/api/dream/plan" && init?.method === "POST") {
        planBodies.push(JSON.parse(String(init.body)));
        return Promise.resolve(
          jsonResponse({
            plan_id: "plan_valid",
            processes: [
              {
                name: "consolidator",
                would_change: false,
                summary: "valid blank budget plan",
                budget_used: 0,
                changes: [],
                errors: [],
                budget_exhausted: false,
              },
            ],
            total_budget_used: 0,
            changes: 0,
          }),
        );
      }

      return undefined;
    });

    renderWithInspector(
      <LiveEventsProvider value={live.live()}>
        <DreamScreen />
      </LiveEventsProvider>,
    );

    await screen.findByText("plan/apply workbench");
    fireEvent.click(screen.getByRole("button", { name: "clear" }));
    fireEvent.click(screen.getByRole("button", { name: "plan dream" }));

    expect(await screen.findByText("Select at least one process to plan.")).toBeInTheDocument();
    expect(planBodies).toHaveLength(0);

    fireEvent.click(screen.getByLabelText("include consolidator"));
    fireEvent.change(screen.getByLabelText("budget"), { target: { value: "0" } });
    fireEvent.click(screen.getByRole("button", { name: "plan dream" }));

    expect(await screen.findByText("Budget must be a positive integer.")).toBeInTheDocument();
    expect(planBodies).toHaveLength(0);

    fireEvent.change(screen.getByLabelText("budget"), { target: { value: "1.5" } });
    fireEvent.click(screen.getByRole("button", { name: "plan dream" }));

    expect(await screen.findByText("Budget must be a positive integer.")).toBeInTheDocument();
    expect(planBodies).toHaveLength(0);

    fireEvent.change(screen.getByLabelText("budget"), { target: { value: "" } });
    fireEvent.click(screen.getByRole("button", { name: "plan dream" }));

    expect(await screen.findByText("valid blank budget plan")).toBeInTheDocument();
    expect(planBodies).toEqual([{ processes: ["consolidator"] }]);
  });

  it("renders maintenance health cards and recent runs with related IdRefs", async () => {
    const live = makeLiveSource();
    installDreamFetch(
      dreamState({
        processes: [
          {
            name: "belief-reviser",
            description: "invalidate beliefs",
            last_run_at: 4,
            last_status: "error",
            last_audit_id: 3,
            budget: null,
            enabled: true,
          },
          {
            name: "curator",
            description: "decay episodes",
            last_run_at: 5,
            last_status: "ok",
            last_audit_id: 4,
            budget: 10,
            enabled: true,
          },
        ],
        pending_extraction_episodes: 5,
        schedule: [
          {
            process: "curator",
            scheduled_at: 4,
            source: "stream",
            stream_entry_id: "strm_sched111111",
          },
          {
            process: "belief-reviser",
            scheduled_at: 6,
            source: "audit",
            audit_id: 9,
          },
        ],
        belief_revision_rows: [
          {
            id: 1,
            kind: "belief_revision",
            refs: { target_type: "semantic_node", target_id: "semn_111111111111" },
            reason: "needs revision",
            created_at: 1,
            resolved_at: null,
            resolution: null,
          },
          {
            id: 2,
            kind: "belief_revision",
            refs: { target_type: "semantic_node", target_id: "semn_222222222222" },
            reason: "still open",
            created_at: 2,
            resolved_at: null,
            resolution: null,
          },
        ],
        scheduler: {
          enabled: true,
          light_interval_ms: 1000,
          heavy_interval_ms: 60000,
          light_processes: ["consolidator", "curator"],
          heavy_processes: ["belief-reviser"],
          process_budgets: {},
        },
      }),
    );

    renderWithInspector(
      <LiveEventsProvider value={live.live()}>
        <DreamScreen />
      </LiveEventsProvider>,
    );

    expect(await screen.findByText("scheduler")).toBeInTheDocument();
    expect(screen.getByTitle(/light: consolidator, curator/)).toBeInTheDocument();
    expect(screen.getByText("light 1s / heavy 1m")).toBeInTheDocument();
    expect(screen.getByText("pending extraction").closest(".dream-health-card")).toHaveTextContent(
      "5",
    );
    expect(screen.getByText("belief revision").closest(".dream-health-card")).toHaveTextContent(
      "2",
    );
    expect(screen.getByText("recent errors").closest(".dream-health-card")).toHaveTextContent("1");
    expect(screen.getByText("last tick this session")).toBeInTheDocument();
    expect(screen.getByText("live frame only")).toBeInTheDocument();

    expect(screen.getByText("recent runs")).toBeInTheDocument();
    expect(screen.getByText("ran at")).toBeInTheDocument();
    expect(screen.getAllByText("curator").length).toBeGreaterThan(0);
    expect(screen.getByRole("button", { name: "jump to strm_sched111111" })).toBeInTheDocument();
    expect(screen.getByRole("button", { name: "jump to audit 9" })).toBeInTheDocument();

    act(() => {
      live.emit({
        type: "maintenance:tick",
        ts: 8,
        cadence: "manual",
        status: "ok",
        processes: ["curator", "belief-reviser"],
        changed: true,
        changes: 2,
        errors: 0,
        pending_extraction_episodes: 3,
      });
    });

    expect(await screen.findByText(/last manual/i)).toBeInTheDocument();
    expect(screen.getByText(/2 processes/i)).toBeInTheDocument();
  });

  it("shows revert confirmation as current to after-revert payload panels", async () => {
    const live = makeLiveSource();
    const row: MaintenanceAuditRow = {
      id: 7,
      run_id: "run_revert111111",
      process: "curator",
      action: "decay",
      targets: { episode_ids: ["ep_revert111111"] },
      reversal: {
        decay: [
          {
            episode_id: "ep_revert111111",
            old_salience: 0.8,
            new_salience: 0.4,
          },
        ],
      },
      applied_at: 4,
      reverted_at: null,
      reverted_by: null,
    };
    installDreamFetch(dreamState({ audit_rows: [row] }));

    renderWithInspector(
      <LiveEventsProvider value={live.live()}>
        <DreamScreen />
      </LiveEventsProvider>,
    );

    fireEvent.click(await screen.findByRole("button", { name: "revert audit 7" }));
    const dialog = screen.getByRole("dialog");

    expect(dialog).toHaveTextContent("current / audited target");
    expect(dialog).toHaveTextContent("after revert / reversal payload");
    expect(dialog).toHaveTextContent("raw current target JSON");
    expect(dialog).toHaveTextContent("raw after-revert JSON");
    expect(dialog).toHaveTextContent("ep_revert111111");
  });
});
