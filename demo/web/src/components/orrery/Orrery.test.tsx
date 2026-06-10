import { act, fireEvent, screen, waitFor } from "@testing-library/react";
import { afterEach, describe, expect, it, vi } from "vitest";

import type {
  CommitmentEnforcement,
  CommitmentItem,
  CreatorDirectiveItem,
  DreamProcessName,
  DreamStateResponse,
  LiveFrame,
  MemoryBandId,
  MemoryBandsResponse,
  ReviewRow,
  SessionRecord,
  StateSnapshot,
  WsState,
} from "../../api/types";
import { LiveEventsProvider } from "../../hooks/live-context";
import type { LiveEventHandler, LiveEvents } from "../../hooks/use-live-events";
import { LiveCacheProvider } from "../../hooks/use-live-cache";
import { MissionControlScreen } from "../../screens/MissionControl";
import { renderWithInspector } from "../../test/inspector";
import { MiniOrrery } from "./MiniOrrery";
import { Orrery, type OrreryProps } from "./Orrery";
import { useOrreryData, type OrreryTurnInput, type OrreryViewModel } from "./useOrreryData";

const realLocation = window.location;

const BAND_IDS: MemoryBandId[] = [
  "episodic",
  "semantic",
  "procedural",
  "affective",
  "self",
  "commitments",
  "social",
  "relational",
];

function jsonResponse(body: unknown, status = 200): Response {
  return new Response(JSON.stringify(body), {
    status,
    headers: { "Content-Type": "application/json" },
  });
}

function requestPath(request: RequestInfo | URL): string {
  return new URL(String(request), "http://test.invalid").pathname;
}

function callsFor(fetchMock: ReturnType<typeof vi.fn>, path: string): number {
  return fetchMock.mock.calls.filter(
    ([request]) => requestPath(request as RequestInfo | URL) === path,
  ).length;
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

function stateSnapshot(input: Partial<StateSnapshot> = {}): StateSnapshot {
  return {
    active_session: "default",
    audiences: ["alice"],
    counts: {
      turns: 2,
      commitments: 2,
      open_qs: 0,
      open_reviews: 1,
      dream_audit_rows: 0,
    },
    current_mood: {
      session_id: "default",
      valence: 0,
      arousal: 0,
      updated_at: 1_000,
      half_life_hours: 12,
      recent_triggers: [],
    },
    version: "test",
    ...input,
  };
}

function session(input: Partial<SessionRecord> = {}): SessionRecord {
  return {
    session_id: "default",
    source_type: "demo",
    source_external_id: null,
    source_url: null,
    label: "demo",
    audience_label: "alice",
    audience_entity_id: null,
    conversation_kind: "demo",
    created_at: 1_000,
    last_activity_at: 1_000,
    last_turn_id: null,
    message_count: 0,
    status: "active",
    privacy_level: "payload_off",
    participation_policy: "active",
    audience_role: "participant",
    ...input,
  };
}

function memoryBands(): MemoryBandsResponse {
  return {
    bands: BAND_IDS.map((id, index) => ({
      id,
      name: id,
      count: (index + 1) * 11,
      count_is_lower_bound: id === "semantic",
      stats: [],
    })),
  };
}

function dreamProcess(name: DreamProcessName) {
  return {
    name,
    description: `${name} process`,
    last_run_at: null,
    last_status: null,
    last_audit_id: null,
    budget: null,
    enabled: true,
  };
}

function dreamState(): DreamStateResponse {
  return {
    processes: [dreamProcess("consolidator"), dreamProcess("ruminator")],
    schedule: [],
    dream_reports: [],
    audit_rows: [],
    belief_revision_rows: [],
    scheduler: {
      enabled: true,
      light_interval_ms: 1_000,
      heavy_interval_ms: 2_000,
      light_processes: ["consolidator"],
      heavy_processes: ["ruminator"],
      process_budgets: {},
    },
  };
}

function reviewRow(id = 7): ReviewRow {
  return {
    id,
    kind: "belief_revision",
    refs: {},
    reason: "needs operator review",
    created_at: 1_000,
    resolved_at: null,
    resolution: null,
  };
}

function commitment(id: string, enforcement: CommitmentEnforcement): CommitmentItem {
  return {
    id,
    text: `${enforcement} commitment`,
    type: "rule",
    kind: "assistant_commitment",
    enforcement_class: enforcement,
    critical_domain: enforcement === "critical" ? "safety" : null,
    state: "active",
    priority: 1,
    directive_family: "test",
    audience: null,
    made_to: null,
    about: null,
    committed_by: null,
    source: "test",
    source_stream_entry_ids: [],
    created_at: 1_000,
    expires_at: null,
    expired_at: null,
    revoked_at: null,
    revoked_reason: null,
    superseded_by_id: null,
    canonicalized_by_artifact_entry_id: null,
    last_reinforced_at: 1_000,
  };
}

function directive(id = "cdir_test111111"): CreatorDirectiveItem {
  return {
    id,
    kind: "response_policy",
    text: "test directive",
    source_session_id: "default",
    authorization_stream_entry_ids: [],
    content_source_stream_entry_ids: [],
    canonical_fact: null,
    operational_directive: "test",
    activation_scope: "public",
    activation_allowed_entity_ids: [],
    activation_excluded_entity_ids: [],
    content_scope: "public",
    mention_policy: "answer_if_asked",
    status: "active",
    subject_kind: "system",
    subject_entity_id: null,
    subject_entity_name: null,
    priority: 1,
    superseded_by_id: null,
    revoked_reason: null,
    created_at: 1_000,
    updated_at: 1_000,
  };
}

function installFetch(
  input: {
    bands?: MemoryBandsResponse;
    dreams?: DreamStateResponse;
    reviews?: ReviewRow[];
    failSubstrate?: boolean;
  } = {},
) {
  const fetchMock = vi.fn((request: RequestInfo | URL) => {
    const path = requestPath(request);

    if (path === "/api/state") {
      return Promise.resolve(jsonResponse(stateSnapshot()));
    }
    if (path === "/api/sessions") {
      return Promise.resolve(jsonResponse({ sessions: [session()] }));
    }
    if (path === "/api/memory/bands") {
      return input.failSubstrate
        ? Promise.resolve(jsonResponse({ error: { message: "bands failed" } }, 500))
        : Promise.resolve(jsonResponse(input.bands ?? memoryBands()));
    }
    if (path === "/api/dream/state") {
      return Promise.resolve(jsonResponse(input.dreams ?? dreamState()));
    }
    if (path === "/api/reviews") {
      return Promise.resolve(jsonResponse({ rows: input.reviews ?? [reviewRow()] }));
    }
    if (path === "/api/commitments") {
      return Promise.resolve(
        jsonResponse({
          commitments: [
            commitment("cmt_critical111111", "critical"),
            commitment("cmt_adv111111", "advisory"),
          ],
        }),
      );
    }
    if (path === "/api/creator-directives") {
      return Promise.resolve(jsonResponse({ directives: [directive()] }));
    }
    if (path === "/api/stream") {
      return Promise.resolve(jsonResponse({ entries: [], next_cursor: null }));
    }
    if (path === "/api/prompts") {
      return Promise.resolve(jsonResponse({ blocks: [] }));
    }

    return Promise.reject(new Error(`unexpected fetch ${path}`));
  });
  vi.stubGlobal("fetch", fetchMock);
  return fetchMock;
}

const idleTurn: OrreryTurnInput = {
  activeTurnId: null,
  lastPhase: "idle",
  running: false,
  terminalOutcome: null,
};

function HookedOrrery({
  live,
  turn = idleTurn,
  onNavigate,
  onInspect,
}: {
  live: ReturnType<typeof makeLiveSource>;
  turn?: OrreryTurnInput;
  onNavigate: OrreryProps["onNavigate"];
  onInspect: OrreryProps["onInspect"];
}) {
  return (
    <LiveEventsProvider value={live.live()}>
      <LiveCacheProvider sessionId="default">
        <HookedOrreryInner turn={turn} onNavigate={onNavigate} onInspect={onInspect} />
      </LiveCacheProvider>
    </LiveEventsProvider>
  );
}

function HookedOrreryInner({
  turn,
  onNavigate,
  onInspect,
}: {
  turn: OrreryTurnInput;
  onNavigate: OrreryProps["onNavigate"];
  onInspect: OrreryProps["onInspect"];
}) {
  const data = useOrreryData(turn);
  return <Orrery size="full" data={data} onNavigate={onNavigate} onInspect={onInspect} />;
}

function emptyViewModel(input: Partial<OrreryViewModel> = {}): OrreryViewModel {
  return {
    loading: false,
    error: null,
    memoryBands: [],
    dream: { satellites: [], runningCount: 0 },
    governance: {
      commitments: { critical: 0, advisory: 0, total: 0 },
      directives: { active: 0, total: 0 },
    },
    reviews: { openCount: 0, severity: "idle", faults: [] },
    stream: { ...idleTurn, state: "idle" },
    runtime: {
      wsState: "live",
      connectionCount: 1,
      counts: null,
      dreamActivity: null,
      lastMaintenanceTick: null,
    },
    ...input,
  };
}

afterEach(() => {
  Object.defineProperty(window, "location", {
    configurable: true,
    value: realLocation,
  });
  vi.restoreAllMocks();
  vi.unstubAllGlobals();
});

describe("Orrery", () => {
  it("renders substrate rings, dream satellites, and review faults with navigation", async () => {
    installFetch();
    const live = makeLiveSource();
    const onNavigate = vi.fn();
    const onInspect = vi.fn();
    const { container } = renderWithInspector(
      <HookedOrrery live={live} onNavigate={onNavigate} onInspect={onInspect} />,
    );

    await screen.findByTestId("orr-memory-ring-episodic");

    expect(container.querySelectorAll("[data-testid^='orr-memory-ring-']")).toHaveLength(8);
    expect(screen.getByTestId("orr-dream-satellite-consolidator")).toBeInTheDocument();
    expect(screen.getByTestId("orr-dream-satellite-ruminator")).toBeInTheDocument();
    expect(screen.getByTestId("orr-fault-node")).toBeInTheDocument();

    fireEvent.click(screen.getByTestId("orr-memory-ring-episodic"));
    expect(onNavigate).toHaveBeenCalledWith("memory");

    fireEvent.click(screen.getByTestId("orr-dream-satellite-consolidator"));
    expect(onNavigate).toHaveBeenCalledWith("dream");

    fireEvent.click(screen.getByTestId("orr-governance-commitments"));
    expect(onNavigate).toHaveBeenCalledWith("governance", { governanceTab: "commitments" });

    fireEvent.click(screen.getByTestId("orr-governance-directives"));
    expect(onNavigate).toHaveBeenCalledWith("governance", { governanceTab: "shared_state" });

    fireEvent.click(screen.getByTestId("orr-fault-node"));
    expect(onNavigate).toHaveBeenCalledWith("review");
  });

  it("omits governance arcs when there are no active constraints", () => {
    const onNavigate = vi.fn();
    const onInspect = vi.fn();

    renderWithInspector(
      <Orrery data={emptyViewModel()} size="full" onNavigate={onNavigate} onInspect={onInspect} />,
    );

    expect(screen.queryByTestId("orr-governance-commitments")).not.toBeInTheDocument();
    expect(screen.queryByTestId("orr-governance-directives")).not.toBeInTheDocument();
    expect(screen.getByText("cmt none")).toBeInTheDocument();
    expect(screen.getByText("dir none")).toBeInTheDocument();
  });

  it("scales governance arcs from active counts and marks only critical commitments as red", () => {
    const onNavigate = vi.fn();
    const onInspect = vi.fn();
    const { rerender } = renderWithInspector(
      <Orrery
        data={emptyViewModel({
          governance: {
            commitments: { critical: 0, advisory: 1, total: 1 },
            directives: { active: 1, total: 1 },
          },
        })}
        size="full"
        onNavigate={onNavigate}
        onInspect={onInspect}
      />,
    );

    const initialCommitments = screen.getByTestId("orr-governance-commitments");
    const initialPath = initialCommitments.querySelector("path");
    expect(initialCommitments).toHaveClass("orr-governance-advisory");
    expect(initialCommitments).not.toHaveClass("orr-governance-critical");
    const initialPathD = initialPath?.getAttribute("d");

    rerender(
      <Orrery
        data={emptyViewModel({
          governance: {
            commitments: { critical: 2, advisory: 6, total: 8 },
            directives: { active: 5, total: 7 },
          },
        })}
        size="full"
        onNavigate={onNavigate}
        onInspect={onInspect}
      />,
    );

    const scaledCommitments = screen.getByTestId("orr-governance-commitments");
    const scaledPath = scaledCommitments.querySelector("path");
    expect(scaledCommitments).toHaveClass("orr-governance-critical");
    expect(scaledPath?.getAttribute("d")).not.toBe(initialPathD);
    expect(screen.getByText("cmt 2/6")).toBeInTheDocument();
    expect(screen.getByText("dir 5/7")).toBeInTheDocument();
  });

  it("marks the active turn pulse and inspects the active turn", async () => {
    installFetch();
    const live = makeLiveSource();
    const onNavigate = vi.fn();
    const onInspect = vi.fn();

    renderWithInspector(
      <HookedOrrery
        live={live}
        onNavigate={onNavigate}
        onInspect={onInspect}
        turn={{
          activeTurnId: "turn_active111111",
          lastPhase: "retrieval",
          running: true,
          terminalOutcome: null,
        }}
      />,
    );

    const pulse = await screen.findByTestId("orr-active-turn-pulse");
    expect(pulse).toHaveAttribute("data-active", "true");
    expect(pulse).toHaveAttribute("data-running", "true");
    expect(screen.getByText("retrieval")).toBeInTheDocument();

    fireEvent.click(pulse);

    expect(onInspect).toHaveBeenCalledWith(
      expect.objectContaining({ type: "turn", id: "turn_active111111" }),
    );
    expect(onNavigate).not.toHaveBeenCalledWith("cognition");
  });

  it("routes the idle stream pulse to cognition", () => {
    const onNavigate = vi.fn();
    const onInspect = vi.fn();

    renderWithInspector(
      <Orrery size="full" data={emptyViewModel()} onNavigate={onNavigate} onInspect={onInspect} />,
    );

    fireEvent.click(screen.getByTestId("orr-active-turn-pulse"));

    expect(onNavigate).toHaveBeenCalledWith("cognition");
    expect(onInspect).not.toHaveBeenCalled();
  });

  it("refetches on global live frames without crashing", async () => {
    const reload = vi.fn();
    Object.defineProperty(window, "location", {
      configurable: true,
      value: { ...window.location, reload },
    });
    const fetchMock = installFetch();
    const live = makeLiveSource();
    const onNavigate = vi.fn();
    const onInspect = vi.fn();

    renderWithInspector(<HookedOrrery live={live} onNavigate={onNavigate} onInspect={onInspect} />);
    await screen.findByTestId("orr-memory-ring-episodic");
    expect(callsFor(fetchMock, "/api/memory/bands")).toBe(1);

    act(() => {
      live.emit({
        type: "maintenance:tick",
        ts: 2_000,
        cadence: "light",
        status: "ok",
        processes: ["consolidator"],
        changed: true,
        changes: 1,
        errors: 0,
      });
      live.emit({
        type: "dream:process:started",
        ts: 2_001,
        process: "consolidator",
        run_id: "run_1",
        phase: "plan",
      });
      live.emit({
        type: "dream:process:completed",
        ts: 2_002,
        process: "consolidator",
        run_id: "run_1",
        phase: "plan",
        errors: 0,
        candidates_accepted: 0,
      });
      live.emit({ type: "borg:reset", ts: 2_003 });
    });

    await waitFor(() => expect(callsFor(fetchMock, "/api/memory/bands")).toBeGreaterThanOrEqual(3));
    expect(reload).toHaveBeenCalledTimes(1);
    expect(screen.getByTestId("orrery")).toBeInTheDocument();
  });

  it("does not crash for empty, loading, or error view-models", () => {
    const onNavigate = vi.fn();
    const onInspect = vi.fn();
    const { rerender } = renderWithInspector(
      <Orrery
        size="compact"
        data={emptyViewModel({ loading: true })}
        onNavigate={onNavigate}
        onInspect={onInspect}
      />,
    );

    expect(screen.getByTestId("orr-loading")).toBeInTheDocument();
    expect(screen.getByTestId("orr-active-turn-pulse")).toHaveAttribute("data-active", "false");

    rerender(
      <Orrery
        size="full"
        data={emptyViewModel({ error: "substrate unavailable" })}
        onNavigate={onNavigate}
        onInspect={onInspect}
      />,
    );

    expect(screen.getByTestId("orr-error")).toHaveTextContent("substrate unavailable");
  });

  it("reflects mini status states", () => {
    const { rerender } = renderWithInspector(
      <MiniOrrery wsState="live" dreamRunning={false} openReviews={0} />,
    );

    expect(screen.getByTestId("mini-orrery")).toHaveAttribute("data-ws-state", "live");
    expect(screen.getByTestId("mini-orrery")).toHaveAttribute("data-dream-running", "false");
    expect(screen.getByTestId("mini-orrery")).toHaveAttribute("data-open-reviews", "0");

    rerender(<MiniOrrery wsState="down" dreamRunning={true} openReviews={3} />);

    expect(screen.getByTestId("mini-orrery")).toHaveAttribute("data-ws-state", "down");
    expect(screen.getByTestId("mini-orrery")).toHaveAttribute("data-dream-running", "true");
    expect(screen.getByTestId("mini-orrery")).toHaveAttribute("data-open-reviews", "3");
  });

  it("renders the Mission Control screen Orrery", async () => {
    installFetch();
    const live = makeLiveSource();

    renderWithInspector(
      <LiveEventsProvider value={live.live()}>
        <LiveCacheProvider sessionId="default">
          <MissionControlScreen sessionId="default" turnStream={idleTurn} onNavigate={vi.fn()} />
        </LiveCacheProvider>
      </LiveEventsProvider>,
    );

    expect(await screen.findByTestId("mission-control-screen")).toBeInTheDocument();
    expect(await screen.findByTestId("orrery")).toBeInTheDocument();
  });
});
