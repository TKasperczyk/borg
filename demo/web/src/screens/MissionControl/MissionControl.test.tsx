import { act, fireEvent, screen, waitFor, within } from "@testing-library/react";
import { afterEach, describe, expect, it, vi } from "vitest";

import type {
  CommitmentItem,
  CreatorDirectiveItem,
  DreamStateResponse,
  LiveFrame,
  MemoryBandsResponse,
  PromptBlockView,
  ReviewRow,
  SessionRecord,
  StateSnapshot,
  StreamEntry,
  WsState,
} from "../../api/types";
import { useInspector } from "../../components/Inspector/inspector-context";
import { LiveEventsProvider } from "../../hooks/live-context";
import type { LiveEventHandler, LiveEvents } from "../../hooks/use-live-events";
import { LiveCacheProvider } from "../../hooks/use-live-cache";
import type { RouteId } from "../../routes";
import { renderWithInspector } from "../../test/inspector";
import { MissionControlScreen } from ".";
import type { OrreryTurnInput } from "../../components/orrery/useOrreryData";

function jsonResponse(body: unknown): Response {
  return new Response(JSON.stringify(body), {
    status: 200,
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
      turns: 8,
      commitments: 3,
      open_qs: 1,
      open_reviews: 5,
      dream_audit_rows: 2,
    },
    current_mood: {
      session_id: "default",
      valence: 0.25,
      arousal: 0.75,
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
    bands: [
      {
        id: "episodic",
        name: "episodic",
        count: 4,
        stats: [],
      },
      {
        id: "semantic",
        name: "semantic",
        count: 7,
        stats: [],
      },
    ],
  };
}

function reviewRow(input: Partial<ReviewRow> & Pick<ReviewRow, "id" | "kind">): ReviewRow {
  return {
    refs: {},
    reason: "needs operator attention",
    created_at: 1_000,
    resolved_at: null,
    resolution: null,
    ...input,
  };
}

function commitment(input: Partial<CommitmentItem> & Pick<CommitmentItem, "id">): CommitmentItem {
  return {
    text: "Keep answers direct.",
    type: "rule",
    kind: "process_norm",
    enforcement_class: "advisory",
    critical_domain: null,
    state: "active",
    priority: 5,
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
    ...input,
  };
}

function directive(
  input: Partial<CreatorDirectiveItem> & Pick<CreatorDirectiveItem, "id">,
): CreatorDirectiveItem {
  return {
    kind: "response_policy",
    text: "Prefer concise answers.",
    source_session_id: "default",
    authorization_stream_entry_ids: [],
    content_source_stream_entry_ids: [],
    canonical_fact: null,
    operational_directive: "Prefer concise answers.",
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
    ...input,
  };
}

function dreamState(beliefRows: ReviewRow[]): DreamStateResponse {
  return {
    processes: [],
    pending_extraction_episodes: 3,
    schedule: [],
    dream_reports: [],
    audit_rows: [],
    belief_revision_rows: beliefRows,
    scheduler: {
      enabled: true,
      light_interval_ms: 1_000,
      heavy_interval_ms: 2_000,
      light_processes: [],
      heavy_processes: [],
      process_budgets: {},
    },
  };
}

function streamEntry(
  input: Partial<StreamEntry> & Pick<StreamEntry, "id" | "kind" | "content">,
): StreamEntry {
  return {
    timestamp: 1_000,
    turn_id: "turn_1",
    sender_entity_id: null,
    reply_target_entity_id: null,
    session_id: "default",
    compressed: false,
    ...input,
  };
}

function promptBlock(
  input: Partial<PromptBlockView> & Pick<PromptBlockView, "key">,
): PromptBlockView {
  return {
    label: "Voice",
    description: "Voice and posture",
    default_text: "default",
    current_text: "override",
    current_text_kind: "stored_override",
    overridden: true,
    updated_at: 1_000,
    ...input,
  };
}

const idleTurn: OrreryTurnInput = {
  activeTurnId: null,
  lastPhase: "idle",
  running: false,
  terminalOutcome: null,
};

const activeTurn: OrreryTurnInput = {
  activeTurnId: "turn_status111111",
  lastPhase: "retrieval",
  running: true,
  terminalOutcome: "reflected",
};

function TargetProbe() {
  const inspector = useInspector();
  return (
    <output data-testid="inspector-target">
      {inspector.target === null ? "none" : `${inspector.target.type}:${inspector.target.id}`}
    </output>
  );
}

function MissionHarness({
  live,
  turn = idleTurn,
  onNavigate = vi.fn(),
}: {
  live: ReturnType<typeof makeLiveSource>;
  turn?: OrreryTurnInput;
  onNavigate?: (view: RouteId) => void;
}) {
  return (
    <LiveEventsProvider value={live.live()}>
      <LiveCacheProvider sessionId="default">
        <MissionControlScreen sessionId="default" turnStream={turn} onNavigate={onNavigate} />
        <TargetProbe />
      </LiveCacheProvider>
    </LiveEventsProvider>
  );
}

function installFetch(): ReturnType<typeof vi.fn> {
  const openReviews = [
    reviewRow({ id: 11, kind: "contradiction" }),
    reviewRow({ id: 12, kind: "belief_revision" }),
  ];
  const conflictReview = reviewRow({
    id: 13,
    kind: "creator_directive_reconciliation",
    refs: {
      subkind: "conflict",
      directive_ids: ["cdir_one111111", "cdir_two222222"],
    },
  });
  const beliefRows = [reviewRow({ id: 21, kind: "belief_revision" })];
  const commitments = [
    commitment({
      id: "cmt_critical111111",
      text: "Never reveal internal identifiers.",
      enforcement_class: "critical",
      critical_domain: "privacy",
    }),
    commitment({ id: "cmt_advisory111111", text: "Prefer direct answers." }),
  ];
  const entries = [
    streamEntry({
      id: "strm_suppressed111",
      kind: "agent_suppressed",
      content: { reason: "finalizer_failed" },
    }),
    streamEntry({
      id: "strm_observed1111",
      kind: "agent_observed",
      content: { reason: "operator observed" },
    }),
  ];
  const prompts = [
    promptBlock({ key: "voice_and_posture", label: "Voice" }),
    promptBlock({
      key: "host_capabilities",
      label: "Host capabilities",
      current_text_kind: "static_default",
      overridden: false,
    }),
  ];

  const fetchMock = vi.fn((request: RequestInfo | URL) => {
    const url = new URL(String(request), "http://test.invalid");

    if (url.pathname === "/api/state") {
      return Promise.resolve(jsonResponse(stateSnapshot()));
    }
    if (url.pathname === "/api/sessions") {
      return Promise.resolve(jsonResponse({ sessions: [session()] }));
    }
    if (url.pathname === "/api/memory/bands") {
      return Promise.resolve(jsonResponse(memoryBands()));
    }
    if (url.pathname === "/api/reviews") {
      if (url.searchParams.get("kind") === "creator_directive_reconciliation") {
        return Promise.resolve(jsonResponse({ rows: [conflictReview] }));
      }
      return Promise.resolve(jsonResponse({ rows: [...openReviews, conflictReview] }));
    }
    if (url.pathname === "/api/commitments") {
      return Promise.resolve(jsonResponse({ commitments }));
    }
    if (url.pathname === "/api/creator-directives") {
      return Promise.resolve(jsonResponse({ directives: [directive({ id: "cdir_one111111" })] }));
    }
    if (url.pathname === "/api/dream/state") {
      return Promise.resolve(jsonResponse(dreamState(beliefRows)));
    }
    if (url.pathname === "/api/stream") {
      return Promise.resolve(jsonResponse({ entries, next_cursor: "strm_before" }));
    }
    if (url.pathname === "/api/prompts") {
      return Promise.resolve(jsonResponse({ blocks: prompts }));
    }

    return Promise.reject(new Error(`unexpected fetch ${url.pathname}`));
  });
  vi.stubGlobal("fetch", fetchMock);
  return fetchMock;
}

afterEach(() => {
  vi.restoreAllMocks();
  vi.unstubAllGlobals();
});

describe("MissionControlScreen", () => {
  it("renders every attention card from the sanctioned endpoints", async () => {
    installFetch();
    const live = makeLiveSource();

    renderWithInspector(<MissionHarness live={live} />);

    const reviews = within(await screen.findByTestId("attention-reviews"));
    expect(reviews.getByLabelText("open reviews: 5")).toBeInTheDocument();
    expect(reviews.getAllByText("contradiction").length).toBeGreaterThan(0);
    expect(reviews.getAllByText("belief revision").length).toBeGreaterThan(0);

    const commitments = within(screen.getByTestId("attention-commitments"));
    expect(commitments.getByLabelText("active commitments: 3")).toBeInTheDocument();
    expect(commitments.getAllByText("critical").length).toBeGreaterThan(0);
    expect(commitments.getAllByText("advisory").length).toBeGreaterThan(0);

    const directives = within(screen.getByTestId("attention-directives"));
    expect(directives.getByLabelText("creator-directive conflicts: 1")).toBeInTheDocument();
    expect(directives.getByText("2 directives")).toBeInTheDocument();

    const dream = within(screen.getByTestId("attention-dream"));
    expect(
      dream.getByLabelText("dream extraction and belief revision work: 4"),
    ).toBeInTheDocument();
    expect(dream.getByText("extraction episodes")).toBeInTheDocument();
    expect(dream.getByText("belief revisions")).toBeInTheDocument();

    const outcomes = within(screen.getByTestId("attention-outcomes"));
    expect(
      outcomes.getByLabelText("recent suppressed and observed outcomes: 2"),
    ).toBeInTheDocument();
    expect(outcomes.getByText("2+ recent window")).toBeInTheDocument();
    expect(outcomes.getAllByText("emission failed").length).toBeGreaterThan(0);
    expect(outcomes.getAllByText("observed").length).toBeGreaterThan(0);

    const prompts = within(screen.getByTestId("attention-prompts"));
    expect(prompts.getByLabelText("prompt overrides: 1")).toBeInTheDocument();
    expect(prompts.getByText("Voice")).toBeInTheDocument();

    const attachments = within(screen.getByTestId("attention-attachments"));
    expect(attachments.getByText(/needs backend/i)).toBeInTheDocument();
    expect(attachments.getByText(/requires explicit attachment ids/i)).toBeInTheDocument();
  });

  it("routes open actions to the relevant screens", async () => {
    installFetch();
    const live = makeLiveSource();
    const onNavigate = vi.fn();

    renderWithInspector(<MissionHarness live={live} onNavigate={onNavigate} />);
    await screen.findByTestId("attention-reviews");

    fireEvent.click(
      within(screen.getByTestId("attention-reviews")).getByRole("button", { name: "open" }),
    );
    fireEvent.click(
      within(screen.getByTestId("attention-commitments")).getByRole("button", { name: "open" }),
    );
    fireEvent.click(
      within(screen.getByTestId("attention-directives")).getByRole("button", { name: "open" }),
    );
    fireEvent.click(
      within(screen.getByTestId("attention-dream")).getByRole("button", { name: "open" }),
    );
    fireEvent.click(
      within(screen.getByTestId("attention-outcomes")).getByRole("button", { name: "open" }),
    );
    fireEvent.click(
      within(screen.getByTestId("attention-prompts")).getByRole("button", { name: "open" }),
    );

    expect(onNavigate).toHaveBeenCalledWith("review");
    expect(onNavigate).toHaveBeenCalledWith("commit");
    expect(onNavigate).toHaveBeenCalledWith("directives");
    expect(onNavigate).toHaveBeenCalledWith("dream");
    expect(onNavigate).toHaveBeenCalledWith("stream");
    expect(onNavigate).toHaveBeenCalledWith("prompts");
  });

  it("opens row inspections through the inspector context", async () => {
    installFetch();
    const live = makeLiveSource();

    renderWithInspector(<MissionHarness live={live} />);
    await screen.findByTestId("attention-reviews");

    const inspectButtons = within(screen.getByTestId("attention-reviews")).getAllByRole("button", {
      name: "inspect",
    });
    expect(inspectButtons.length).toBeGreaterThan(0);
    fireEvent.click(inspectButtons[0]!);

    expect(screen.getByTestId("inspector-target")).toHaveTextContent("review:11");

    fireEvent.click(
      within(screen.getByTestId("attention-commitments")).getAllByRole("button", {
        name: "inspect",
      })[0]!,
    );
    expect(screen.getByTestId("inspector-target")).toHaveTextContent(
      "commitment:cmt_critical111111",
    );

    fireEvent.click(
      within(screen.getByTestId("attention-outcomes")).getAllByRole("button", {
        name: "inspect",
      })[0]!,
    );
    expect(screen.getByTestId("inspector-target")).toHaveTextContent(
      "stream_entry:strm_suppressed111",
    );

    fireEvent.click(
      within(screen.getByTestId("attention-prompts")).getByRole("button", {
        name: "inspect",
      }),
    );
    expect(screen.getByTestId("inspector-target")).toHaveTextContent(
      "prompt_block:voice_and_posture",
    );
  });

  it("renders turn, mood, and dream status in the strip", async () => {
    installFetch();
    const live = makeLiveSource();

    renderWithInspector(<MissionHarness live={live} turn={activeTurn} />);

    const strip = within(await screen.findByTestId("mission-status-strip"));
    expect(strip.getByRole("button", { name: "jump to turn_status111111" })).toBeInTheDocument();
    expect(strip.getByText("retrieval")).toBeInTheDocument();
    expect(strip.getByText("reflected")).toBeInTheDocument();
    expect(await strip.findByText(/v 0.25 · a 0.75/)).toBeInTheDocument();

    act(() => {
      live.emit({
        type: "dream:process:started",
        ts: 2_000,
        process: "belief-reviser",
        run_id: "run_1",
        phase: "apply",
      });
    });

    expect(strip.getByText("belief-reviser apply")).toBeInTheDocument();
  });

  it("refetches attention data on maintenance and stream frames", async () => {
    const fetchMock = installFetch();
    const live = makeLiveSource();

    renderWithInspector(<MissionHarness live={live} />);

    await waitFor(() => expect(callsFor(fetchMock, "/api/prompts")).toBe(1));

    act(() => {
      live.emit({
        type: "maintenance:tick",
        ts: 2_000,
        cadence: "light",
        status: "ok",
        processes: ["belief-reviser"],
        changed: true,
        changes: 1,
        errors: 0,
      });
      live.emit({
        type: "stream:append",
        ts: 2_001,
        entries: [
          streamEntry({
            id: "strm_new111111",
            kind: "agent_observed",
            content: { reason: "new outcome" },
          }),
        ],
      });
    });

    await waitFor(() => expect(callsFor(fetchMock, "/api/prompts")).toBeGreaterThanOrEqual(3));
  });
});
