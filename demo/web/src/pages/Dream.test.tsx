import { fireEvent, render, screen, waitFor } from "@testing-library/react";

import { installMockWebSocket } from "../__tests__/mock-websocket";
import { LiveProvider } from "../live/useLive";
import { DreamPage, sameProcessSet } from "./Dream";
import {
  appendDreamRequest,
  EMPTY_DREAM_RUN_FEED,
  reduceDreamRunFeed,
} from "./dream/runFeed";
import type {
  DreamStateResponse,
  MaintenanceAuditRow,
  OfflineProcessName,
} from "../api/types";

const now = Date.UTC(2026, 5, 11, 12);

function auditRow(input: Partial<MaintenanceAuditRow> & Pick<MaintenanceAuditRow, "id" | "reversal" | "reverted_at">): MaintenanceAuditRow {
  return {
    id: input.id,
    run_id: "run_demo",
    process: "creator-directive-reconciler",
    action: "creator_directive_merge",
    targets: ["cd_1"],
    reversal: input.reversal,
    applied_at: now,
    reverted_at: input.reverted_at,
    reverted_by: null,
  };
}

function dreamState(input: Partial<DreamStateResponse> = {}): DreamStateResponse {
  return {
    processes: [
      {
        name: "consolidator",
        description: "Merge raw episodes into families.",
        last_run_at: null,
        last_status: null,
        last_audit_id: null,
        budget: 120,
        enabled: true,
      },
      {
        name: "reflector",
        description: "Distill reflective insights.",
        last_run_at: now,
        last_status: "ok",
        last_audit_id: 7,
        budget: null,
        enabled: true,
      },
    ],
    pending_extraction_episodes: 2,
    schedule: [],
    dream_reports: [
      {
        run_id: "run_report",
        processes: ["belief-reviser"],
        dry_run: false,
        planned_at: now,
        changes: 1,
        tokens_used: 55,
        errors: [{ process: "belief-reviser", message: "old failure" }],
        budget_exhausted_processes: [],
        notes: ["Budget exhausted: belief-reviser"],
      },
    ],
    audit_rows: [],
    belief_revision_rows: [],
    scheduler: {
      enabled: true,
      light_interval_ms: 60_000,
      heavy_interval_ms: 3_600_000,
      optimize_storage: true,
      light_processes: ["consolidator"],
      heavy_processes: ["reflector"],
      process_budgets: { consolidator: 120 },
    },
    ...input,
  };
}

function json(body: unknown, status = 200): Response {
  return new Response(JSON.stringify(body), {
    status,
    headers: { "Content-Type": "application/json" },
  });
}

function createDeferred<T>() {
  let resolve!: (value: T) => void;
  let reject!: (reason?: unknown) => void;
  const promise = new Promise<T>((innerResolve, innerReject) => {
    resolve = innerResolve;
    reject = innerReject;
  });
  return { promise, resolve, reject };
}

function planResponse(): Response {
  return json({
    plan_id: "plan_1",
    processes: [
      {
        name: "consolidator",
        would_change: true,
        summary: "1 change",
        budget_used: 10,
        changes: [{ id: "change" }],
        errors: [],
        budget_exhausted: false,
      },
    ],
    total_budget_used: 10,
    changes: 1,
  });
}

function renderDream(options: { auditRows?: MaintenanceAuditRow[]; planDeferred?: ReturnType<typeof createDeferred<Response>> } = {}) {
  installMockWebSocket();
  const requests: Array<{ url: string; method: string; body: unknown }> = [];
  vi.spyOn(globalThis, "fetch").mockImplementation(async (input, init) => {
    const url = String(input);
    const method = init?.method ?? "GET";
    requests.push({
      url,
      method,
      body: init?.body === undefined ? null : JSON.parse(String(init.body)),
    });

    if (url === "/api/dream/state") {
      return json(dreamState({ audit_rows: options.auditRows ?? [] }));
    }
    if (url === "/api/dream/audit?limit=50") {
      return json({ rows: options.auditRows ?? [] });
    }
    if (url === "/api/dream/plan" && method === "POST") {
      return options.planDeferred?.promise ?? planResponse();
    }
    if (url === "/api/dream/apply" && method === "POST") {
      return json({
        run_id: "run_apply",
        applied: [{ name: "consolidator", audit_id: 12, audit_ids: [12], changes: 1 }],
        failed: [],
        duration_ms: 50,
        total_budget_used: 10,
      });
    }
    if (url.includes("/api/dream/audit/") && method === "POST") {
      return json({ ...(options.auditRows?.[0] ?? auditRow({ id: 1, reversal: {}, reverted_at: null })), reverted_at: now });
    }

    return json({ message: "not found" }, 404);
  });

  render(
    <LiveProvider>
      <DreamPage />
    </LiveProvider>,
  );
  return { requests };
}

describe("Dream page", () => {
  afterEach(() => {
    vi.restoreAllMocks();
  });

  it("renders pending extraction only when nonzero and handles selection toolbar", async () => {
    renderDream();

    expect(await screen.findByText("2 episodes pending extraction")).toBeTruthy();
    expect(screen.getByText("0 selected")).toBeTruthy();
    expect((screen.getByRole("button", { name: "PLAN" }) as HTMLButtonElement).disabled).toBe(true);

    fireEvent.click(screen.getByText("CONSOLIDATOR"));

    expect(screen.getByText("1 selected")).toBeTruthy();
    expect((screen.getByRole("button", { name: "PLAN" }) as HTMLButtonElement).disabled).toBe(false);
  });

  it("plans selected processes and applies by plan_id while the selection still matches", async () => {
    const { requests } = renderDream();

    fireEvent.click(await screen.findByText("CONSOLIDATOR"));
    fireEvent.click(screen.getByRole("button", { name: "PLAN" }));

    await waitFor(() =>
      expect(requests.some((request) => request.url === "/api/dream/plan" && request.method === "POST")).toBe(true),
    );
    expect(requests.find((request) => request.url === "/api/dream/plan")?.body).toEqual({
      processes: ["consolidator"],
    });

    fireEvent.click(await screen.findByRole("button", { name: "▶ APPLY" }));

    await waitFor(() =>
      expect(requests.some((request) => request.url === "/api/dream/apply" && request.method === "POST")).toBe(true),
    );
    expect(requests.find((request) => request.url === "/api/dream/apply")?.body).toEqual({
      plan_id: "plan_1",
    });
  });

  it("applies selected processes instead of stale plan_id after selection changes", async () => {
    const { requests } = renderDream();

    fireEvent.click(await screen.findByText("CONSOLIDATOR"));
    fireEvent.click(screen.getByRole("button", { name: "PLAN" }));
    await screen.findByText("PLAN plan_1");

    fireEvent.click(screen.getByText("REFLECTOR"));
    fireEvent.click(screen.getByRole("button", { name: "▶ APPLY" }));

    await waitFor(() =>
      expect(requests.filter((request) => request.url === "/api/dream/apply")).toHaveLength(1),
    );
    expect(requests.find((request) => request.url === "/api/dream/apply")?.body).toEqual({
      processes: ["consolidator", "reflector"],
    });
  });

  it("gates audit revert by structural reversal fields and posts the revert route", async () => {
    const rows = [
      auditRow({ id: 101, reversal: { loser_id: "cd_2" }, reverted_at: null }),
      auditRow({ id: 102, reversal: {}, reverted_at: null }),
      auditRow({ id: 103, reversal: { loser_id: "cd_3" }, reverted_at: now }),
    ];
    const { requests } = renderDream({ auditRows: rows });

    expect(await screen.findByText("#101")).toBeTruthy();
    expect(screen.getAllByRole("button", { name: "REVERT" })).toHaveLength(1);

    fireEvent.click(screen.getByRole("button", { name: "REVERT" }));
    fireEvent.click(screen.getByRole("button", { name: "CONFIRM" }));

    await waitFor(() =>
      expect(requests.some((request) => request.url === "/api/dream/audit/101/revert")).toBe(true),
    );
  });

  it("renders dream report errors from real report fields", async () => {
    renderDream();

    expect(await screen.findByText("1 errors")).toBeTruthy();
    expect(screen.getByText("belief-reviser · old failure")).toBeTruthy();
  });

  it("locks all process selection controls while planning is pending", async () => {
    const planDeferred = createDeferred<Response>();
    renderDream({ planDeferred });

    fireEvent.click(await screen.findByText("CONSOLIDATOR"));
    fireEvent.click(screen.getByRole("button", { name: "PLAN" }));

    await waitFor(() => expect((screen.getByRole("button", { name: "PLAN" }) as HTMLButtonElement).disabled).toBe(true));
    expect((screen.getByRole("button", { name: "[ ] ALL" }) as HTMLButtonElement).disabled).toBe(true);
    expect((screen.getByRole("button", { name: "[ ] NONE" }) as HTMLButtonElement).disabled).toBe(true);
    expect((screen.getByRole("button", { name: /REFLECTOR/ }) as HTMLButtonElement).disabled).toBe(true);

    fireEvent.click(screen.getByRole("button", { name: "[ ] NONE" }));
    fireEvent.click(screen.getByRole("button", { name: /REFLECTOR/ }));
    expect(screen.getByText("1 selected")).toBeTruthy();

    planDeferred.resolve(planResponse());
    expect(await screen.findByText("PLAN plan_1")).toBeTruthy();
  });
});

describe("Dream run feed reducer", () => {
  it("uses order-independent set equality for plan reuse", () => {
    expect(sameProcessSet(["reflector", "consolidator"], ["consolidator", "reflector"])).toBe(true);
    expect(sameProcessSet(["reflector"], ["reflector", "consolidator"])).toBe(false);
  });

  it("aggregates started/completed/tick frames with error counts and honest summary", () => {
    const selected: OfflineProcessName[] = ["consolidator"];
    let state = appendDreamRequest(EMPTY_DREAM_RUN_FEED, {
      action: "apply",
      processes: selected,
      ts: now,
    });
    state = reduceDreamRunFeed(state, {
      type: "dream:process:started",
      ts: now + 1,
      process: "consolidator",
      run_id: "run_1",
      phase: "apply",
    });
    state = reduceDreamRunFeed(state, {
      type: "dream:process:completed",
      ts: now + 2,
      process: "consolidator",
      run_id: "run_1",
      phase: "apply",
      duration_ms: 25,
      errors: 1,
      candidates_accepted: 2,
    });
    state = reduceDreamRunFeed(state, {
      type: "maintenance:tick",
      ts: now + 3,
      cadence: "manual",
      status: "ok",
      processes: ["consolidator"],
      changed: true,
      changes: 2,
      errors: 1,
      duration_ms: 30,
      run_id: "run_1",
    });

    expect(state.inFlight).toBe(false);
    expect(state.entries.map((entry) => entry.kind)).toEqual(["request", "started", "completed", "tick"]);
    expect(state.summary).toEqual({
      key: "apply:run_1",
      phase: "apply",
      run_id: "run_1",
      changes: 2,
      errors: 1,
      duration_ms: 30,
    });
  });

  it("settles a plan-phase run without a maintenance tick", () => {
    let state = reduceDreamRunFeed(EMPTY_DREAM_RUN_FEED, {
      type: "dream:process:started",
      ts: now + 1,
      process: "curator",
      run_id: "plan_run",
      phase: "plan",
    });
    expect(state.inFlight).toBe(true);

    state = reduceDreamRunFeed(state, {
      type: "dream:process:completed",
      ts: now + 2,
      process: "curator",
      run_id: "plan_run",
      phase: "plan",
      duration_ms: 11,
      errors: 0,
      candidates_accepted: 3,
    });

    expect(state.inFlight).toBe(false);
    expect(state.summary).toEqual({
      key: "plan:plan_run",
      phase: "plan",
      run_id: "plan_run",
      changes: 3,
      errors: 0,
      duration_ms: 11,
    });
  });

  it("partitions summaries by run and lets manual ticks overwrite apply totals", () => {
    let state = reduceDreamRunFeed(EMPTY_DREAM_RUN_FEED, {
      type: "dream:process:completed",
      ts: now + 1,
      process: "consolidator",
      run_id: "run_a",
      phase: "apply",
      duration_ms: 10,
      errors: 0,
      candidates_accepted: 1,
    });
    state = reduceDreamRunFeed(state, {
      type: "dream:process:completed",
      ts: now + 2,
      process: "reflector",
      run_id: "run_b",
      phase: "apply",
      duration_ms: 20,
      errors: 1,
      candidates_accepted: 5,
    });

    expect(state.summary).toMatchObject({
      key: "apply:run_b",
      changes: 5,
      errors: 1,
      duration_ms: 20,
    });

    state = reduceDreamRunFeed(state, {
      type: "maintenance:tick",
      ts: now + 3,
      cadence: "manual",
      status: "ok",
      processes: ["reflector"],
      changed: true,
      changes: 4,
      errors: 0,
      duration_ms: 31,
      run_id: "run_b",
    });

    expect(state.summary).toMatchObject({
      key: "apply:run_b",
      changes: 4,
      errors: 0,
      duration_ms: 31,
    });
  });
});
