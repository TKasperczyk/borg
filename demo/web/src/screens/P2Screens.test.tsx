import { act, fireEvent, screen, waitFor, within } from "@testing-library/react";
import { renderWithInspector } from "../test/inspector";
import { afterEach, describe, expect, it, vi } from "vitest";

import type { LiveFrame, MaintenanceAuditRow, StreamEntry, WsState } from "../api/types";
import { LiveEventsProvider } from "../hooks/live-context";
import type { LiveEventHandler, LiveEvents } from "../hooks/use-live-events";
import { CommitScreen } from "./Commit";
import { DreamScreen } from "./Dream";
import { IdentityScreen } from "./Identity";
import { MemoryScreen } from "./Memory";
import { ReviewScreen } from "./Review";
import { StreamScreen } from "./Stream";

function jsonResponse(body: unknown): Response {
  return new Response(JSON.stringify(body), {
    status: 200,
    headers: { "Content-Type": "application/json" },
  });
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

function streamEntry(
  input: Partial<StreamEntry> & Pick<StreamEntry, "id" | "kind" | "content">,
): StreamEntry {
  return {
    timestamp: 1,
    turn_id: "turn_1",
    audience: "alice",
    sender_entity_id: null,
    reply_target_entity_id: null,
    session_id: "default",
    compressed: false,
    ...input,
  };
}

function requestPath(request: RequestInfo | URL): string {
  return new URL(String(request), "http://test.invalid").pathname;
}

function semanticNodeFixture(id: string, label: string, description: string, episodeId: string) {
  return {
    id,
    kind: "proposition",
    label,
    description,
    domain: "runtime",
    aliases: [],
    confidence: 0.8,
    status: "active",
    source_episode_ids: [episodeId],
    source_count: 1,
    created_at: 1,
    updated_at: 2,
  };
}

function semanticEdgeFixture(id: string, fromNodeId: string, toNodeId: string, episodeId: string) {
  return {
    id,
    from_node_id: fromNodeId,
    to_node_id: toNodeId,
    relation: "contradicts",
    confidence: 0.72,
    evidence_episode_ids: [episodeId],
    source_count: 1,
    valid_from: 3,
    valid_to: null,
    invalidated_at: null,
    invalidated_by_edge_id: null,
    invalidated_by_review_id: null,
    invalidated_by_process: null,
    invalidated_reason: null,
  };
}

function dreamAuditRowFixture(
  input: Partial<MaintenanceAuditRow> & Pick<MaintenanceAuditRow, "id" | "targets">,
): MaintenanceAuditRow {
  return {
    run_id: "run_dreamrefs",
    process: "belief-reviser",
    action: "inspect_target",
    reversal: {},
    applied_at: 4,
    reverted_at: null,
    reverted_by: null,
    ...input,
  };
}

function dreamStateWithAuditRows(rows: MaintenanceAuditRow[]) {
  return {
    processes: [],
    schedule: [],
    audit_rows: rows,
    dream_reports: [],
    belief_revision_rows: [],
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

function installDreamAuditFetch(
  rows: MaintenanceAuditRow[],
  extra?: (url: URL, init?: RequestInit) => Promise<Response> | undefined,
): ReturnType<typeof vi.fn> {
  const fetchMock = vi.fn((request: RequestInfo | URL, init?: RequestInit) => {
    const url = new URL(String(request), "http://test.invalid");
    if (url.pathname === "/api/dream/state") {
      return Promise.resolve(jsonResponse(dreamStateWithAuditRows(rows)));
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

function jsonErrorResponse(status: number, message: string): Response {
  return new Response(JSON.stringify({ error: { status, message } }), {
    status,
    headers: { "Content-Type": "application/json" },
  });
}

function installFetch(): ReturnType<typeof vi.fn> {
  const fetchMock = vi.fn((request: RequestInfo | URL) => {
    // Client defaults to same-origin (relative) URLs; provide a base so URL parses them.
    const url = new URL(String(request), "http://test.invalid");
    if (url.pathname === "/api/stream") {
      return Promise.resolve(
        jsonResponse({
          entries: [
            streamEntry({
              id: "strm_att",
              timestamp: 3,
              kind: "user_image_attachment",
              sender_entity_id: "ent_sender11111111",
              reply_target_entity_id: "ent_reply111111111",
              content: { attachment_id: "att_1", media_type: "image/png" },
            }),
            streamEntry({
              id: "strm_obs",
              timestamp: 2,
              kind: "agent_observed",
              content: { reason: "observed action state" },
            }),
            streamEntry({ id: "strm_user", timestamp: 1, kind: "user_msg", content: "hello" }),
            streamEntry({
              id: "strm_suppressed",
              timestamp: 0.5,
              kind: "agent_suppressed",
              content: {
                reason: "finalizer_no_output",
                primary_no_output_reason: "low_value_echo",
                no_output_categories: ["closure"],
                structural_no_output_flags: ["with_open_question"],
              },
            }),
            streamEntry({
              id: "strm_dream_summary",
              timestamp: 0.4,
              kind: "dream_report",
              content: {
                run_id: "run_demo",
                processes: ["consolidator", "curator"],
                dry_run: false,
                planned_at: 3,
                changes: 2,
                tokens_used: 99,
                errors: [],
                budget_exhausted_processes: [],
                notes: [],
              },
            }),
            streamEntry({
              id: "strm_internal_summary",
              timestamp: 0.3,
              kind: "internal_event",
              content: {
                event: "frame_anomaly_gate",
                trigger: "scheduler",
                outcome_summary: "skipped by policy",
                source_stream_entry_ids: ["strm_user"],
              },
            }),
          ],
          next_cursor: null,
        }),
      );
    }
    if (url.pathname === "/api/attachments") {
      return Promise.resolve(
        jsonResponse([
          {
            id: "att_1",
            status: { active: false, quarantined: true, stream_active: false, parent_active: true },
          },
        ]),
      );
    }
    if (url.pathname === "/api/attachments/att_1") {
      return Promise.resolve(
        jsonResponse({
          attachment: {
            attachment_id: "att_1",
            sha256: "sha256abc",
            media_type: "image/png",
            byte_size: 10,
            width: 4,
            height: 4,
            storage_ref: "x",
            thumbnail_ref: null,
            perception_id: "imgp_1",
            text_embedding_ref: null,
            visual_embedding_ref: null,
            active: false,
            audience: "alice",
            created_turn_global: null,
            parent_entry_id: "strm_user",
            stream_entry_id: "strm_att",
            parent_turn_id: "turn_1",
            created_at: 1,
          },
          perception: { caption: "quarantined screenshot", active: false, perception_id: "imgp_1" },
          status: { active: false, quarantined: true, stream_active: false, parent_active: true },
        }),
      );
    }
    if (url.pathname.endsWith("/bytes")) {
      return Promise.resolve(new Response("", { status: 404 }));
    }
    if (url.pathname === "/api/memory/bands") {
      return Promise.resolve(
        jsonResponse({
          bands: [
            {
              id: "episodic",
              n: "01",
              name: "episodic",
              desc: "what happened",
              count: 0,
              growth: [1],
              stats: [],
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
              count: 1,
              growth: [1],
              stats: [{ k: "skills", v: 1 }],
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
        }),
      );
    }
    if (url.pathname === "/api/reviews") {
      return Promise.resolve(jsonResponse({ rows: [] }));
    }
    if (url.pathname === "/api/memory/bands/procedural") {
      return Promise.resolve(
        jsonResponse({
          band: "procedural",
          items: [
            {
              id: "skill_1",
              applies_when: "debugging migrations",
              approach: "read the failing test first",
              status: "active",
              alpha: 5,
              beta: 2,
              attempts: 7,
              successes: 5,
              failures: 2,
              sample_count: 3,
              source_episode_ids: ["ep_1"],
              last_used: null,
              last_successful: null,
              requires_manual_review: false,
              created_at: 1,
              updated_at: 1,
            },
          ],
        }),
      );
    }
    if (url.pathname === "/api/identity") {
      return Promise.resolve(
        jsonResponse({
          values: [],
          goals: [],
          traits: [],
          open_questions: [
            {
              id: "oq_1",
              question: "what remains unresolved?",
              urgency: 0.7,
              status: "abandoned",
              goal_id: null,
              source: "ruminator",
              created_at: 1,
              last_touched: 2,
              resolved_at: null,
              abandoned_at: 3,
              abandoned_reason: "no signal",
              resolution_note: null,
              unresolved_rumination_ticks: 4,
              last_ruminated_at: 2,
            },
          ],
          growth_markers: [],
          periods: [
            {
              id: "abp_current",
              label: "current arc",
              start_ts: 4,
              end_ts: null,
              narrative: "active period",
              key_episode_ids: [],
              themes: [],
              created_at: 4,
              last_updated: 4,
            },
            {
              id: "abp_old",
              label: "old arc",
              start_ts: 1,
              end_ts: 3,
              narrative: "closed period",
              key_episode_ids: [],
              themes: [],
              created_at: 1,
              last_updated: 3,
            },
          ],
          open_question_events: [],
        }),
      );
    }
    if (url.pathname === "/api/commitments") {
      return Promise.resolve(
        jsonResponse({
          commitments: [
            {
              id: "cm_active",
              text: "active rule",
              type: "rule",
              kind: "process_norm",
              enforcement_class: "advisory",
              critical_domain: null,
              state: "active",
              priority: 1,
              directive_family: "a",
              audience: "alice",
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
            },
            {
              id: "cm_revoked",
              text: "revoked rule",
              type: "boundary",
              kind: "audience_rule",
              enforcement_class: "critical",
              critical_domain: "audience_scope",
              state: "revoked",
              priority: 9,
              directive_family: "b",
              audience: "alice",
              made_to: null,
              about: null,
              committed_by: null,
              source: "manual",
              source_stream_entry_ids: [],
              created_at: 1,
              expires_at: null,
              expired_at: null,
              revoked_at: 2,
              revoked_reason: "old",
              superseded_by_id: null,
              canonicalized_by_artifact_entry_id: null,
              last_reinforced_at: 1,
            },
          ],
        }),
      );
    }
    if (url.pathname === "/api/state") {
      return Promise.resolve(
        jsonResponse({
          active_session: "default",
          audiences: ["alice"],
          counts: { turns: 1, commitments: 1, open_qs: 1, open_reviews: 1, dream_audit_rows: 1 },
          current_mood: {
            session_id: "default",
            valence: 0,
            arousal: 0,
            updated_at: 1,
            half_life_hours: 1,
            recent_triggers: [],
          },
          version: "test",
        }),
      );
    }
    if (url.pathname === "/api/shared-state") {
      return Promise.resolve(
        jsonResponse({
          audience: "alice",
          entries: [
            {
              id: "ss_1",
              audience_entity_id: "ent_1",
              state_key: null,
              kind: "locked",
              text: "alice likes terse answers",
              owner_entity_id: null,
              provenance_stream_entry_ids: ["strm_1"],
              last_updated_stream_entry_ids: ["strm_1"],
              created_at: 1,
              last_updated_at: 2,
              last_updated_turn_global: null,
              superseded_by_id: null,
              rank: 0,
              canonicalizes: {
                goal_ids: [],
                commitment_ids: ["cm_1"],
                action_ids: [],
                open_question_ids: [],
              },
            },
            {
              id: "ss_2",
              audience_entity_id: "ent_1",
              state_key: "decision.low",
              kind: "low_salience_live",
              text: "low salience route detail",
              owner_entity_id: null,
              provenance_stream_entry_ids: ["strm_1"],
              last_updated_stream_entry_ids: ["strm_1"],
              created_at: 1,
              last_updated_at: 2,
              last_updated_turn_global: null,
              superseded_by_id: null,
              rank: 1,
              canonicalizes: {
                goal_ids: [],
                commitment_ids: [],
                action_ids: [],
                open_question_ids: [],
              },
            },
            {
              id: "ss_3",
              audience_entity_id: "ent_1",
              state_key: "decision.dormant",
              kind: "dormant_live",
              text: "dormant route detail",
              owner_entity_id: null,
              provenance_stream_entry_ids: ["strm_1"],
              last_updated_stream_entry_ids: ["strm_1"],
              created_at: 1,
              last_updated_at: 2,
              last_updated_turn_global: null,
              superseded_by_id: null,
              rank: 2,
              canonicalizes: {
                goal_ids: [],
                commitment_ids: [],
                action_ids: [],
                open_question_ids: [],
              },
            },
            {
              id: "ss_4",
              audience_entity_id: "ent_1",
              state_key: "decision.legacy",
              kind: "pending",
              text: "legacy pending route detail",
              owner_entity_id: null,
              provenance_stream_entry_ids: ["strm_1"],
              last_updated_stream_entry_ids: ["strm_1"],
              created_at: 1,
              last_updated_at: 2,
              last_updated_turn_global: null,
              superseded_by_id: null,
              rank: 3,
              canonicalizes: {
                goal_ids: [],
                commitment_ids: [],
                action_ids: [],
                open_question_ids: [],
              },
            },
          ],
        }),
      );
    }
    if (url.pathname === "/api/dream/state") {
      return Promise.resolve(
        jsonResponse({
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
          schedule: [
            {
              process: "belief-reviser",
              scheduled_at: 4,
              source: "stream",
              stream_entry_id: "strm_dream",
            },
          ],
          audit_rows: [],
          belief_revision_rows: [
            {
              id: 1,
              kind: "belief_revision",
              refs: {
                target_type: "semantic_node",
                target_id: "sn_1",
                invalidated_edge_id: "se_1",
              },
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
        }),
      );
    }
    return Promise.resolve(new Response("{}", { status: 404 }));
  });
  vi.stubGlobal("fetch", fetchMock);
  return fetchMock;
}

afterEach(() => {
  vi.unstubAllGlobals();
});

describe("P2 screens", () => {
  it("renders stream live tail and attachment quarantine detail", async () => {
    const live = makeLiveSource();
    installFetch();
    renderWithInspector(
      <LiveEventsProvider value={live.live()}>
        <StreamScreen sessionId="default" />
      </LiveEventsProvider>,
    );

    expect(await screen.findByText(/user_image_attachment/)).toBeInTheDocument();
    expect((await screen.findAllByText("quarantined")).length).toBeGreaterThan(0);
    expect(await screen.findByText(/quarantined cascade/)).toBeInTheDocument();
    expect(await screen.findByText("deliberate silence")).toBeInTheDocument();
    expect(screen.getByText("reason finalizer no output")).toBeInTheDocument();
    expect(screen.getByText("primary low value echo")).toBeInTheDocument();
    expect(screen.getByText("category closure")).toBeInTheDocument();
    expect(screen.getByText("flag with open question")).toBeInTheDocument();

    const attachmentFilter = screen.getAllByText("user_image_attachment")[0]?.closest(".opt");
    expect(attachmentFilter).not.toBeNull();
    fireEvent.click(attachmentFilter!);
    await waitFor(() => {
      expect(screen.queryByText("[strm_att]")).not.toBeInTheDocument();
      expect(screen.getByText("[strm_obs]")).toBeInTheDocument();
    });

    act(() => {
      live.emit({
        type: "stream:append",
        ts: 5,
        entries: [
          streamEntry({
            id: "strm_live",
            timestamp: 5,
            kind: "dream_report",
            content: { processes: ["curator"] },
          }),
        ],
      });
    });

    await waitFor(() => {
      expect(screen.getAllByText(/dream_report/).length).toBeGreaterThan(0);
    });
  });

  it("opens inspector refs from stream provenance rows", async () => {
    const live = makeLiveSource();
    installFetch();
    const { container } = renderWithInspector(
      <LiveEventsProvider value={live.live()}>
        <StreamScreen sessionId="default" />
      </LiveEventsProvider>,
      { inspector: true },
    );

    expect(await screen.findByText("quarantined screenshot")).toBeInTheDocument();
    const provenance = container.querySelector(".stream-detail .det-body .props");
    expect(provenance).not.toBeNull();
    const provenanceView = within(provenance as HTMLElement);

    fireEvent.click(provenanceView.getByRole("button", { name: "jump to default" }));
    expect(await screen.findByRole("dialog", { name: "Session inspector" })).toBeInTheDocument();
    fireEvent.click(screen.getByRole("button", { name: "close inspector" }));

    fireEvent.click(provenanceView.getByRole("button", { name: "jump to turn_1" }));
    expect(
      await screen.findByRole("dialog", { name: "Turn evidence inspector" }),
    ).toBeInTheDocument();
    fireEvent.click(screen.getByRole("button", { name: "close inspector" }));

    fireEvent.click(provenanceView.getByRole("button", { name: "jump to ent_sender11111111" }));
    expect(await screen.findByRole("dialog", { name: "Entity inspector" })).toBeInTheDocument();
  });

  it("renders compact stream row summaries for object content", async () => {
    const live = makeLiveSource();
    installFetch();
    renderWithInspector(
      <LiveEventsProvider value={live.live()}>
        <StreamScreen sessionId="default" />
      </LiveEventsProvider>,
    );

    const dreamRow = (await screen.findByText("run run_demo")).closest(".stream-row");
    expect(dreamRow).not.toBeNull();
    expect(dreamRow).toHaveTextContent("consolidator, curator");
    expect(dreamRow).toHaveTextContent("2 changes");
    expect(dreamRow).toHaveTextContent("99 tok");
    expect(dreamRow).toHaveTextContent("0 errors");
    expect(dreamRow).not.toHaveTextContent('{"run_id"');

    const internalRow = (await screen.findByText("frame_anomaly_gate")).closest(".stream-row");
    expect(internalRow).not.toBeNull();
    expect(internalRow).toHaveTextContent("trigger scheduler");
    expect(internalRow).toHaveTextContent("outcome summary skipped by policy");
    expect(internalRow).toHaveTextContent("1 source refs");
    expect(internalRow).not.toHaveTextContent('{"event"');

    expect(screen.getByText("deliberate silence")).toBeInTheDocument();
    expect(screen.getByText("reason finalizer no output")).toBeInTheDocument();
  });

  it("refetches stream rows on WebSocket reconnect after the initial connection", async () => {
    const live = makeLiveSource();
    const fetchMock = installFetch();
    const { rerender } = renderWithInspector(
      <LiveEventsProvider value={live.live(1)}>
        <StreamScreen sessionId="default" />
      </LiveEventsProvider>,
    );

    await waitFor(() => {
      expect(
        fetchMock.mock.calls.filter((call) => requestPath(call[0]) === "/api/stream"),
      ).toHaveLength(1);
    });

    rerender(
      <LiveEventsProvider value={live.live(2)}>
        <StreamScreen sessionId="default" />
      </LiveEventsProvider>,
    );

    await waitFor(() => {
      expect(
        fetchMock.mock.calls.filter((call) => requestPath(call[0]) === "/api/stream"),
      ).toHaveLength(2);
    });
  });

  it("drills from memory overview into procedural records", async () => {
    installFetch();
    renderWithInspector(<MemoryScreen sessionId="default" />);

    const proceduralLabels = await screen.findAllByText("procedural");
    fireEvent.click(proceduralLabels[0]?.closest(".band-card") ?? proceduralLabels[0]!);

    expect((await screen.findAllByText("debugging migrations")).length).toBeGreaterThan(0);
    expect(screen.getAllByText(/alpha 5.0/).length).toBeGreaterThan(0);
  });

  it("renders full open-question lifecycle in identity", async () => {
    installFetch();
    renderWithInspector(<IdentityScreen />);

    expect(await screen.findByText("what remains unresolved?")).toBeInTheDocument();
    expect(screen.getAllByText("abandoned")[0]).toBeInTheDocument();
    expect(screen.getByText(/current period: current arc/)).toBeInTheDocument();
  });

  it("filters commitment lifecycle rows", async () => {
    installFetch();
    renderWithInspector(<CommitScreen />);

    expect((await screen.findAllByText("active rule")).length).toBeGreaterThan(0);
    fireEvent.click(screen.getByText("revoked"));

    await waitFor(() => {
      expect(screen.queryAllByText("active rule")).toHaveLength(0);
    });
    expect(screen.getAllByText("revoked rule").length).toBeGreaterThan(0);
  });

  it("links dream belief-revision rows to the unified review screen", async () => {
    const live = makeLiveSource();
    installFetch();
    const openReview = vi.fn();
    renderWithInspector(
      <LiveEventsProvider value={live.live()}>
        <DreamScreen onOpenReview={openReview} />
      </LiveEventsProvider>,
    );

    expect(await screen.findByText("belief revisions")).toBeInTheDocument();
    expect(screen.getByText("1 open")).toBeInTheDocument();
    fireEvent.click(screen.getByRole("button", { name: "open review" }));
    expect(openReview).toHaveBeenCalledTimes(1);
  });

  it("keeps dream card audit text selecting its card", async () => {
    const live = makeLiveSource();
    const fetchMock = vi.fn((request: RequestInfo | URL) => {
      const url = new URL(String(request), "http://test.invalid");
      if (url.pathname === "/api/dream/state") {
        return Promise.resolve(
          jsonResponse({
            processes: [
              {
                name: "belief-reviser",
                description: "default selected process",
                last_run_at: 4,
                last_status: "ok",
                last_audit_id: 1,
                budget: null,
                enabled: true,
              },
              {
                name: "curator",
                description: "audit click target",
                last_run_at: 5,
                last_status: "ok",
                last_audit_id: 2,
                budget: null,
                enabled: true,
              },
            ],
            schedule: [],
            audit_rows: [],
            dream_reports: [],
            belief_revision_rows: [],
            scheduler: {
              enabled: true,
              light_interval_ms: 1,
              heavy_interval_ms: 1,
              light_processes: [],
              heavy_processes: ["belief-reviser"],
              process_budgets: {},
            },
          }),
        );
      }
      return Promise.resolve(new Response("not found", { status: 404 }));
    });
    vi.stubGlobal("fetch", fetchMock);

    const { container } = renderWithInspector(
      <LiveEventsProvider value={live.live()}>
        <DreamScreen />
      </LiveEventsProvider>,
    );

    const targetCard = (await screen.findByText("audit click target")).closest(".dream-card");
    expect(container.querySelector(".panel-header .title")).toHaveTextContent("belief-reviser");
    expect(targetCard).not.toBeNull();

    fireEvent.click(within(targetCard as HTMLElement).getByText("2"));

    expect(container.querySelector(".panel-header .title")).toHaveTextContent("curator");
  });

  it("reverts only reversible unreverted dream audit rows", async () => {
    const live = makeLiveSource();
    const reversibleRow = {
      id: 7,
      run_id: "run_audit",
      process: "curator",
      action: "decay",
      targets: { episode_ids: ["ep_decay1"] },
      reversal: {
        decay: [
          {
            episode_id: "ep_decay1",
            old_salience: 0.8,
            new_salience: 0.4,
          },
        ],
        previous: [{ episode_id: "ep_decay1", tier: "T1" }],
      },
      applied_at: 4,
      reverted_at: null,
      reverted_by: null,
    };
    const nonReversibleRow = {
      ...reversibleRow,
      id: 8,
      run_id: "run_none",
      action: "noop",
      targets: { id: "no_reverse" },
      reversal: {},
    };
    const revertedRow = {
      ...reversibleRow,
      id: 9,
      process: "creator-directive-reconciler",
      action: "creator_directive_merge",
      reverted_at: 6,
      reverted_by: "demo_operator",
    };
    const dreamReport = {
      run_id: "run_audit",
      processes: ["curator"],
      dry_run: false,
      planned_at: 3,
      changes: 2,
      tokens_used: 99,
      errors: [{ process: "curator", message: "curator warning" }],
      budget_exhausted_processes: ["curator"],
      notes: ["Budget exhausted: curator"],
    };
    let stateCalls = 0;
    const fetchMock = vi.fn((request: RequestInfo | URL, init?: RequestInit) => {
      const url = new URL(String(request), "http://test.invalid");

      if (url.pathname === "/api/dream/state") {
        stateCalls += 1;
        return Promise.resolve(
          jsonResponse({
            processes: [],
            schedule: [],
            audit_rows:
              stateCalls === 1
                ? [reversibleRow, nonReversibleRow, revertedRow]
                : [{ ...reversibleRow, reverted_at: 7, reverted_by: "demo_operator" }],
            dream_reports: [dreamReport],
            belief_revision_rows: [],
            scheduler: {
              enabled: true,
              light_interval_ms: 1,
              heavy_interval_ms: 1,
              light_processes: [],
              heavy_processes: ["belief-reviser"],
              process_budgets: {},
            },
          }),
        );
      }

      if (url.pathname === "/api/dream/audit/7/revert" && init?.method === "POST") {
        return Promise.resolve(
          jsonResponse({ ...reversibleRow, reverted_at: 7, reverted_by: "demo_operator" }),
        );
      }

      if (url.pathname === "/api/dream/audit") {
        return Promise.resolve(jsonResponse({ rows: [] }));
      }

      return Promise.resolve(new Response("not found", { status: 404 }));
    });
    vi.stubGlobal("fetch", fetchMock);

    renderWithInspector(
      <LiveEventsProvider value={live.live()}>
        <DreamScreen />
      </LiveEventsProvider>,
    );

    const enabled = await screen.findByRole("button", { name: "revert audit 7" });
    expect(enabled).not.toBeDisabled();
    expect(screen.getByRole("button", { name: "revert audit 8" })).toBeDisabled();
    expect(screen.getByRole("button", { name: "revert audit 9" })).toBeDisabled();
    expect(screen.getAllByText("reverted").length).toBeGreaterThan(0);
    expect(screen.getByRole("row", { name: /audit run run_audit/i })).toBeInTheDocument();
    expect(screen.getByRole("row", { name: /audit run run_none/i })).toBeInTheDocument();
    expect(screen.getByText("2 changes")).toBeInTheDocument();
    expect(screen.getByText("99 tok")).toBeInTheDocument();
    expect(screen.getByText("1 errors")).toBeInTheDocument();
    expect(screen.getByText("budget curator")).toBeInTheDocument();
    expect(screen.getByText("Budget exhausted: curator")).toBeInTheDocument();
    expect(screen.getByText("no matching dream_report in state window")).toBeInTheDocument();

    fireEvent.click(screen.getByRole("button", { name: "show audit 7 payload" }));
    expect(screen.getByText("applied target")).toBeInTheDocument();
    expect(screen.getByText("undo/change payload")).toBeInTheDocument();
    expect(screen.getByText("salience")).toBeInTheDocument();
    expect(screen.getByText("0.8")).toBeInTheDocument();
    expect(screen.getByText("0.4")).toBeInTheDocument();
    expect(screen.getByText("raw reversal JSON")).toBeInTheDocument();

    fireEvent.click(enabled);
    expect(screen.getByRole("dialog")).toHaveTextContent("revert maintenance change?");
    expect(screen.getByRole("dialog")).toHaveTextContent("curator");
    expect(screen.getByRole("dialog")).toHaveTextContent("episode ep_decay1");
    expect(
      fetchMock.mock.calls.some(
        ([request, init]) =>
          requestPath(request) === "/api/dream/audit/7/revert" && init?.method === "POST",
      ),
    ).toBe(false);

    fireEvent.click(screen.getByRole("button", { name: "confirm revert" }));

    await waitFor(() => {
      expect(
        fetchMock.mock.calls.some(
          ([request, init]) =>
            requestPath(request) === "/api/dream/audit/7/revert" && init?.method === "POST",
        ),
      ).toBe(true);
      expect(
        fetchMock.mock.calls.filter((call) => requestPath(call[0]) === "/api/dream/state"),
      ).toHaveLength(2);
      expect(fetchMock.mock.calls.some((call) => requestPath(call[0]) === "/api/dream/audit")).toBe(
        true,
      );
    });
  });

  it("opens the inspector from expanded dream audit target refs", async () => {
    const live = makeLiveSource();
    const nodeId = "semn_dreamtarget111";
    const row = dreamAuditRowFixture({
      id: 31,
      targets: { target_id: nodeId, related_ids: [nodeId, "review_42"] },
    });
    installDreamAuditFetch([row], (url) => {
      if (url.pathname === `/api/semantic/nodes/${nodeId}`) {
        return Promise.resolve(
          jsonResponse({
            node: semanticNodeFixture(
              nodeId,
              "Dream target node",
              "Loaded through the universal inspector.",
              "ep_dreamtarget1111",
            ),
          }),
        );
      }
      return undefined;
    });

    renderWithInspector(
      <LiveEventsProvider value={live.live()}>
        <DreamScreen />
      </LiveEventsProvider>,
      { inspector: true },
    );

    fireEvent.click(await screen.findByRole("button", { name: "show audit 31 payload" }));
    fireEvent.click(screen.getByRole("button", { name: `jump to ${nodeId}` }));

    expect(
      await screen.findByRole("dialog", { name: "Semantic node inspector" }),
    ).toBeInTheDocument();
    expect(await screen.findByText("Dream target node")).toBeInTheDocument();
  });

  it("keeps dream audit payload expansion working after target refs render", async () => {
    const live = makeLiveSource();
    const nodeId = "semn_dreamtoggle111";
    installDreamAuditFetch([
      dreamAuditRowFixture({
        id: 32,
        targets: { target_id: nodeId },
      }),
    ]);

    renderWithInspector(
      <LiveEventsProvider value={live.live()}>
        <DreamScreen />
      </LiveEventsProvider>,
    );

    fireEvent.click(await screen.findByRole("button", { name: "show audit 32 payload" }));
    expect(screen.getByText("applied target")).toBeInTheDocument();
    expect(screen.getByRole("button", { name: `jump to ${nodeId}` })).toBeInTheDocument();

    fireEvent.click(screen.getByRole("button", { name: "hide audit 32 payload" }));
    expect(screen.queryByText("applied target")).not.toBeInTheDocument();

    fireEvent.click(screen.getByRole("button", { name: "show audit 32 payload" }));
    expect(screen.getByText("applied target")).toBeInTheDocument();
  });

  it("opens nested dream audit target refs and keeps row expansion working", async () => {
    const live = makeLiveSource();
    const streamId = "strm_nestedtarget1";
    installDreamAuditFetch(
      [
        dreamAuditRowFixture({
          id: 34,
          targets: {
            overseer_flag: {
              cited_stream_ids: [streamId],
            },
          },
        }),
      ],
      (url) => {
        if (url.pathname === "/api/sessions") {
          return Promise.resolve(
            jsonResponse({
              sessions: [
                {
                  session_id: "default",
                  source_type: "demo",
                  source_external_id: null,
                  source_url: null,
                  label: "default",
                  audience_label: "alice",
                  audience_entity_id: null,
                  conversation_kind: "demo",
                  created_at: 1,
                  last_activity_at: 2,
                  last_turn_id: "turn_nested",
                  message_count: 1,
                  status: "active",
                  privacy_level: "payload_on",
                  participation_policy: "active",
                  audience_role: "operator",
                },
              ],
            }),
          );
        }
        if (url.pathname === "/api/stream") {
          return Promise.resolve(
            jsonResponse({
              entries: [
                streamEntry({
                  id: streamId,
                  kind: "internal_event",
                  content: { event: "nested target evidence" },
                }),
              ],
              next_cursor: null,
            }),
          );
        }
        return undefined;
      },
    );

    renderWithInspector(
      <LiveEventsProvider value={live.live()}>
        <DreamScreen />
      </LiveEventsProvider>,
      { inspector: true },
    );

    fireEvent.click(await screen.findByRole("button", { name: "show audit 34 payload" }));
    expect(screen.getByText("applied target")).toBeInTheDocument();
    fireEvent.click(screen.getByRole("button", { name: `jump to ${streamId}` }));

    expect(
      await screen.findByRole("dialog", { name: "Stream entry inspector" }),
    ).toBeInTheDocument();
    expect(await screen.findByText("internal_event")).toBeInTheDocument();

    fireEvent.click(screen.getByRole("button", { name: "hide audit 34 payload" }));
    expect(screen.queryByText("applied target")).not.toBeInTheDocument();
  });

  it("does not render IdRefs for unrecognized dream audit target values", async () => {
    const live = makeLiveSource();
    installDreamAuditFetch([
      dreamAuditRowFixture({
        id: 33,
        targets: { review_id: "42", target_id: "review_42" },
      }),
    ]);

    renderWithInspector(
      <LiveEventsProvider value={live.live()}>
        <DreamScreen />
      </LiveEventsProvider>,
    );

    fireEvent.click(await screen.findByRole("button", { name: "show audit 33 payload" }));

    expect(screen.getAllByText(/review_42/).length).toBeGreaterThan(0);
    expect(screen.queryByRole("button", { name: "jump to review_42" })).not.toBeInTheDocument();
    expect(screen.queryByRole("button", { name: "jump to 42" })).not.toBeInTheDocument();
  });

  it("renders unified review rows and resolves a generic row", async () => {
    const live = makeLiveSource();
    let reviewCalls = 0;
    const fetchMock = vi.fn((request: RequestInfo | URL, init?: RequestInit) => {
      const url = new URL(String(request), "http://test.invalid");
      if (url.pathname === "/api/reviews" && init?.method === undefined) {
        reviewCalls += 1;
        return Promise.resolve(
          jsonResponse({
            rows:
              reviewCalls === 1
                ? [
                    {
                      id: 41,
                      kind: "belief_revision",
                      refs: {
                        target_type: "semantic_node",
                        target_id: "semn_1111111111111111",
                        invalidated_edge_id: "seme_1111111111111111",
                        dependency_path_edge_ids: [],
                        surviving_support_edge_ids: [],
                        evidence_episode_ids: [],
                      },
                      reason: "dependency invalidated",
                      created_at: 4,
                      resolved_at: null,
                      resolution: null,
                    },
                  ]
                : [],
          }),
        );
      }
      if (url.pathname === "/api/creator-directives") {
        return Promise.resolve(jsonResponse({ directives: [] }));
      }
      if (url.pathname === "/api/reviews/41" && init?.method === "PATCH") {
        return Promise.resolve(
          jsonResponse({
            id: 41,
            kind: "belief_revision",
            refs: {},
            reason: "dependency invalidated",
            created_at: 4,
            resolved_at: 5,
            resolution: "dismiss",
          }),
        );
      }
      return Promise.resolve(new Response("not found", { status: 404 }));
    });
    vi.stubGlobal("fetch", fetchMock);

    renderWithInspector(
      <LiveEventsProvider value={live.live()}>
        <ReviewScreen />
      </LiveEventsProvider>,
    );

    expect(await screen.findByText("dependency invalidated")).toBeInTheDocument();
    fireEvent.click(screen.getByRole("button", { name: "dismiss" }));

    await waitFor(() => {
      expect(
        fetchMock.mock.calls.some(
          ([request, init]) =>
            requestPath(request) === "/api/reviews/41" && init?.method === "PATCH",
        ),
      ).toBe(true);
    });
    expect(await screen.findByText("no open review rows")).toBeInTheDocument();
  });

  it("opens review ref-list ids and keeps review select refs in sync", async () => {
    const live = makeLiveSource();
    const leftNodeId = "semn_selectleft1111";
    const rightNodeId = "semn_selectright111";
    const edgeId = "seme_selectedge1111";
    const episodeId = "ep_selectepisode11";
    const firstDirectiveId = "cdir_selectone1111";
    const secondDirectiveId = "cdir_selecttwo1111";
    const fetchMock = vi.fn((request: RequestInfo | URL) => {
      const url = new URL(String(request), "http://test.invalid");
      if (url.pathname === "/api/reviews") {
        return Promise.resolve(
          jsonResponse({
            rows: [
              {
                id: 61,
                kind: "contradiction",
                refs: {
                  node_ids: [leftNodeId, rightNodeId],
                  node_labels: ["Left selectable", "Right selectable"],
                  edge_id: edgeId,
                  episode_ids: [episodeId],
                },
                reason: "select a semantic winner",
                created_at: 4,
                resolved_at: null,
                resolution: null,
              },
              {
                id: 62,
                kind: "creator_directive_reconciliation",
                refs: {
                  subkind: "scope_equivalence",
                  directive_ids: [firstDirectiveId, secondDirectiveId],
                  members: [
                    {
                      id: firstDirectiveId,
                      scope_equivalence: {
                        disclosure_policy: {
                          content_scope: "allow_list",
                          mention_policy: "answer_if_asked",
                          allowed_entity_ids: ["ent_scopeallowed111"],
                          excluded_entity_ids: [],
                        },
                        activation_policy: {
                          scope: "allow_list",
                          allowed_entity_ids: ["ent_activation111"],
                          excluded_entity_ids: [],
                        },
                      },
                    },
                    {
                      id: secondDirectiveId,
                      scope_equivalence: {
                        disclosure_policy: {
                          content_scope: "public",
                          mention_policy: "answer_if_asked",
                          allowed_entity_ids: [],
                          excluded_entity_ids: [],
                        },
                        activation_policy: {
                          scope: "public",
                          allowed_entity_ids: [],
                          excluded_entity_ids: [],
                        },
                      },
                    },
                  ],
                  judgment: { rationale: "merge matching directive scopes" },
                },
                reason: "directive scopes need reconciliation",
                created_at: 5,
                resolved_at: null,
                resolution: null,
              },
            ],
          }),
        );
      }
      if (url.pathname === "/api/creator-directives") {
        return Promise.resolve(
          jsonResponse({
            directives: [
              {
                id: firstDirectiveId,
                kind: "subject_fact",
                text: "First directive.",
                source_session_id: "default",
                authorization_stream_entry_ids: [],
                content_source_stream_entry_ids: [],
                canonical_fact: "First directive.",
                operational_directive: null,
                activation_scope: "allow_list",
                activation_allowed_entity_ids: ["ent_activation111"],
                activation_excluded_entity_ids: [],
                content_scope: "allow_list",
                mention_policy: "answer_if_asked",
                status: "active",
                subject_kind: "entity",
                subject_entity_id: null,
                subject_entity_name: null,
                priority: 1,
                superseded_by_id: null,
                revoked_reason: null,
                created_at: 1,
                updated_at: 1,
              },
              {
                id: secondDirectiveId,
                kind: "subject_fact",
                text: "Second directive.",
                source_session_id: "default",
                authorization_stream_entry_ids: [],
                content_source_stream_entry_ids: [],
                canonical_fact: "Second directive.",
                operational_directive: null,
                activation_scope: "public",
                activation_allowed_entity_ids: [],
                activation_excluded_entity_ids: [],
                content_scope: "public",
                mention_policy: "answer_if_asked",
                status: "active",
                subject_kind: "entity",
                subject_entity_id: null,
                subject_entity_name: null,
                priority: 1,
                superseded_by_id: null,
                revoked_reason: null,
                created_at: 1,
                updated_at: 1,
              },
            ],
          }),
        );
      }
      return Promise.resolve(new Response("not found", { status: 404 }));
    });
    vi.stubGlobal("fetch", fetchMock);

    renderWithInspector(
      <LiveEventsProvider value={live.live()}>
        <ReviewScreen />
      </LiveEventsProvider>,
      { inspector: true },
    );

    const winnerSelect = (await screen.findByLabelText("winner node")) as HTMLSelectElement;
    const winnerControl = winnerSelect.parentElement as HTMLElement;
    expect(
      within(winnerControl).getByRole("button", { name: `jump to ${leftNodeId}` }),
    ).toBeInTheDocument();
    fireEvent.change(winnerSelect, { target: { value: rightNodeId } });
    expect(
      within(winnerControl).getByRole("button", { name: `jump to ${rightNodeId}` }),
    ).toBeInTheDocument();

    fireEvent.click(screen.getByRole("button", { name: `jump to ${edgeId}` }));
    expect(
      await screen.findByRole("dialog", { name: "Semantic edge inspector" }),
    ).toBeInTheDocument();
    fireEvent.click(screen.getByRole("button", { name: "close inspector" }));

    const survivorSelect = (await screen.findByLabelText("scope survivor")) as HTMLSelectElement;
    const survivorControl = survivorSelect.parentElement as HTMLElement;
    expect(
      within(survivorControl).getByRole("button", { name: `jump to ${firstDirectiveId}` }),
    ).toBeInTheDocument();
    fireEvent.change(survivorSelect, { target: { value: secondDirectiveId } });
    expect(
      within(survivorControl).getByRole("button", { name: `jump to ${secondDirectiveId}` }),
    ).toBeInTheDocument();
    expect(screen.getByRole("button", { name: "jump to ent_scopeallowed111" })).toBeInTheDocument();
  });

  it("drills into contradiction reviews with both semantic nodes and the edge", async () => {
    const live = makeLiveSource();
    const leftNodeId = "semn_aaaaaaaaaaaaaaaa";
    const rightNodeId = "semn_bbbbbbbbbbbbbbbb";
    const edgeId = "seme_cccccccccccccccc";
    const episodeId = "ep_dddddddddddddddd";
    const fetchMock = vi.fn((request: RequestInfo | URL) => {
      const url = new URL(String(request), "http://test.invalid");
      if (url.pathname === "/api/reviews") {
        return Promise.resolve(
          jsonResponse({
            rows: [
              {
                id: 51,
                kind: "contradiction",
                refs: {
                  node_ids: [leftNodeId, rightNodeId],
                  node_labels: ["Substrate claim", "Runtime claim"],
                  edge_id: edgeId,
                  episode_ids: [episodeId],
                },
                reason: "semantic contradiction requires operator review",
                created_at: 4,
                resolved_at: null,
                resolution: null,
              },
            ],
          }),
        );
      }
      if (url.pathname === "/api/creator-directives") {
        return Promise.resolve(jsonResponse({ directives: [] }));
      }
      if (url.pathname === `/api/semantic/nodes/${leftNodeId}`) {
        return Promise.resolve(
          jsonResponse({
            node: {
              id: leftNodeId,
              kind: "proposition",
              label: "Substrate claim",
              description: "The substrate is expected to stay online during the demo.",
              domain: "runtime",
              aliases: ["demo substrate"],
              confidence: 0.82,
              status: "active",
              source_episode_ids: [episodeId],
              source_count: 1,
              created_at: 1,
              updated_at: 2,
            },
          }),
        );
      }
      if (url.pathname === `/api/semantic/nodes/${rightNodeId}`) {
        return Promise.resolve(
          jsonResponse({
            node: {
              id: rightNodeId,
              kind: "proposition",
              label: "Runtime claim",
              description: "The runtime should be treated as offline for the demo.",
              domain: "runtime",
              aliases: [],
              confidence: 0.74,
              status: "active",
              source_episode_ids: [episodeId],
              source_count: 1,
              created_at: 1,
              updated_at: 2,
            },
          }),
        );
      }
      if (url.pathname === `/api/semantic/edges/${edgeId}`) {
        return Promise.resolve(
          jsonResponse({
            edge: {
              id: edgeId,
              from_node_id: leftNodeId,
              to_node_id: rightNodeId,
              relation: "contradicts",
              confidence: 0.72,
              evidence_episode_ids: [episodeId],
              source_count: 1,
              valid_from: 3,
              valid_to: null,
              invalidated_at: null,
              invalidated_by_edge_id: null,
              invalidated_by_review_id: null,
              invalidated_by_process: null,
              invalidated_reason: null,
            },
          }),
        );
      }
      return Promise.resolve(new Response("not found", { status: 404 }));
    });
    vi.stubGlobal("fetch", fetchMock);

    renderWithInspector(
      <LiveEventsProvider value={live.live()}>
        <ReviewScreen />
      </LiveEventsProvider>,
    );

    const winnerSelect = (await screen.findByLabelText("winner node")) as HTMLSelectElement;
    const optionTexts = Array.from(winnerSelect.options).map((option) => option.textContent ?? "");
    expect(optionTexts[0]).toMatch(/^Substrate claim \[semn_aaa.*aaaa\]$/);
    expect(optionTexts[1]).toMatch(/^Runtime claim \[semn_bbb.*bbbb\]$/);

    fireEvent.click(await screen.findByRole("button", { name: "drill" }));

    expect(
      await screen.findByText("The substrate is expected to stay online during the demo."),
    ).toBeInTheDocument();
    expect(
      screen.getByText("The runtime should be treated as offline for the demo."),
    ).toBeInTheDocument();
    expect(screen.getByText("contradicts")).toBeInTheDocument();
    expect(screen.getByText("confidence 0.72")).toBeInTheDocument();
    expect(screen.getAllByText(episodeId).length).toBeGreaterThan(0);

    await waitFor(() => {
      expect(
        fetchMock.mock.calls.some(
          (call) => requestPath(call[0]) === `/api/semantic/nodes/${leftNodeId}`,
        ),
      ).toBe(true);
      expect(
        fetchMock.mock.calls.some(
          (call) => requestPath(call[0]) === `/api/semantic/nodes/${rightNodeId}`,
        ),
      ).toBe(true);
      expect(
        fetchMock.mock.calls.some(
          (call) => requestPath(call[0]) === `/api/semantic/edges/${edgeId}`,
        ),
      ).toBe(true);
    });
  });

  it("renders the available node and edge when one drill-through node is unavailable", async () => {
    const live = makeLiveSource();
    const availableNodeId = "semn_available000000";
    const missingNodeId = "semn_missing00000000";
    const edgeId = "seme_partial0000000";
    const episodeId = "ep_partial000000000";
    const fetchMock = vi.fn((request: RequestInfo | URL) => {
      const url = new URL(String(request), "http://test.invalid");
      if (url.pathname === "/api/reviews") {
        return Promise.resolve(
          jsonResponse({
            rows: [
              {
                id: 52,
                kind: "contradiction",
                refs: {
                  node_ids: [availableNodeId, missingNodeId],
                  node_labels: ["Available claim", "Deleted claim"],
                  edge_id: edgeId,
                },
                reason: "semantic contradiction requires operator review",
                created_at: 4,
                resolved_at: null,
                resolution: null,
              },
            ],
          }),
        );
      }
      if (url.pathname === "/api/creator-directives") {
        return Promise.resolve(jsonResponse({ directives: [] }));
      }
      if (url.pathname === `/api/semantic/nodes/${availableNodeId}`) {
        return Promise.resolve(
          jsonResponse({
            node: semanticNodeFixture(
              availableNodeId,
              "Available claim",
              "Available semantic node description.",
              episodeId,
            ),
          }),
        );
      }
      if (url.pathname === `/api/semantic/nodes/${missingNodeId}`) {
        return Promise.resolve(jsonErrorResponse(404, "semantic node missing"));
      }
      if (url.pathname === `/api/semantic/edges/${edgeId}`) {
        return Promise.resolve(
          jsonResponse({
            edge: semanticEdgeFixture(edgeId, availableNodeId, missingNodeId, episodeId),
          }),
        );
      }
      return Promise.resolve(new Response("not found", { status: 404 }));
    });
    vi.stubGlobal("fetch", fetchMock);

    renderWithInspector(
      <LiveEventsProvider value={live.live()}>
        <ReviewScreen />
      </LiveEventsProvider>,
    );

    fireEvent.click(await screen.findByRole("button", { name: "drill" }));

    expect(await screen.findByText("Available semantic node description.")).toBeInTheDocument();
    expect(screen.getByText("contradicts")).toBeInTheDocument();
    expect(screen.getByText("candidate 2 semantic node unavailable")).toBeInTheDocument();
    expect(screen.getAllByText(missingNodeId).length).toBeGreaterThan(0);
    expect(screen.getByText("semantic node missing")).toBeInTheDocument();
  });

  it("renders both drill-through nodes when the semantic edge is unavailable", async () => {
    const live = makeLiveSource();
    const leftNodeId = "semn_leftedge000000";
    const rightNodeId = "semn_rightedge00000";
    const edgeId = "seme_missing0000000";
    const episodeId = "ep_edge00000000000";
    const fetchMock = vi.fn((request: RequestInfo | URL) => {
      const url = new URL(String(request), "http://test.invalid");
      if (url.pathname === "/api/reviews") {
        return Promise.resolve(
          jsonResponse({
            rows: [
              {
                id: 53,
                kind: "contradiction",
                refs: {
                  node_ids: [leftNodeId, rightNodeId],
                  node_labels: ["Left claim", "Right claim"],
                  edge_id: edgeId,
                },
                reason: "semantic contradiction requires operator review",
                created_at: 4,
                resolved_at: null,
                resolution: null,
              },
            ],
          }),
        );
      }
      if (url.pathname === "/api/creator-directives") {
        return Promise.resolve(jsonResponse({ directives: [] }));
      }
      if (url.pathname === `/api/semantic/nodes/${leftNodeId}`) {
        return Promise.resolve(
          jsonResponse({
            node: semanticNodeFixture(
              leftNodeId,
              "Left claim",
              "Left semantic node description.",
              episodeId,
            ),
          }),
        );
      }
      if (url.pathname === `/api/semantic/nodes/${rightNodeId}`) {
        return Promise.resolve(
          jsonResponse({
            node: semanticNodeFixture(
              rightNodeId,
              "Right claim",
              "Right semantic node description.",
              episodeId,
            ),
          }),
        );
      }
      if (url.pathname === `/api/semantic/edges/${edgeId}`) {
        return Promise.resolve(jsonErrorResponse(404, "semantic edge missing"));
      }
      return Promise.resolve(new Response("not found", { status: 404 }));
    });
    vi.stubGlobal("fetch", fetchMock);

    renderWithInspector(
      <LiveEventsProvider value={live.live()}>
        <ReviewScreen />
      </LiveEventsProvider>,
    );

    fireEvent.click(await screen.findByRole("button", { name: "drill" }));

    expect(await screen.findByText("Left semantic node description.")).toBeInTheDocument();
    expect(screen.getByText("Right semantic node description.")).toBeInTheDocument();
    expect(screen.getByText("semantic edge unavailable")).toBeInTheDocument();
    expect(screen.getAllByText(edgeId).length).toBeGreaterThan(0);
    expect(screen.getByText("semantic edge missing")).toBeInTheDocument();
  });

  it("renders new insight review details without embeddings or raw internal ids", async () => {
    const live = makeLiveSource();
    const fetchMock = vi.fn((request: RequestInfo | URL) => {
      const url = new URL(String(request), "http://test.invalid");
      if (url.pathname === "/api/reviews") {
        return Promise.resolve(
          jsonResponse({
            rows: [
              {
                id: 42,
                kind: "new_insight",
                refs: {
                  node_ids: ["semn_abcdefghijklmnop"],
                  episode_ids: ["ep_abcdefghijklmnop", "ep_bcdefghijklmnopq"],
                  entity_ids: ["ent_abcdefghijklmnop"],
                  evidence_cluster_key: "scope:private|cluster:maya",
                  evidence_cluster_size: 3,
                  reflector_pending_insight: {
                    target: {
                      mode: "insert",
                      node: {
                        id: "semn_abcdefghijklmnop",
                        kind: "proposition",
                        label: "Maya preference",
                        description: "Maya prefers concise project memory summaries.",
                        domain: null,
                        aliases: [],
                        confidence: 0.8,
                        source_episode_ids: ["ep_abcdefghijklmnop"],
                        created_at: 1,
                        updated_at: 1,
                        last_verified_at: 1,
                        embedding: Array.from({ length: 24 }, (_, index) => index / 100),
                        archived: false,
                        superseded_by: null,
                        status: "active",
                        corrected_by: null,
                        superseded_at: null,
                      },
                    },
                    candidate_support_edges: [
                      {
                        id: "seme_abcdefghijklmnop",
                        insight_node_id: "semn_abcdefghijklmnop",
                        target_node_id: "semn_bcdefghijklmnopq",
                        source_episode_ids: ["ep_bcdefghijklmnopq"],
                        confidence: 0.7,
                      },
                    ],
                    evidence_cluster: {
                      key: "scope:private|cluster:maya",
                      episode_ids: ["ep_abcdefghijklmnop", "ep_bcdefghijklmnopq"],
                      size: 3,
                    },
                  },
                },
                reason: "low-confidence reflector insight",
                created_at: 4,
                resolved_at: null,
                resolution: null,
              },
            ],
          }),
        );
      }
      if (url.pathname === "/api/creator-directives") {
        return Promise.resolve(jsonResponse({ directives: [] }));
      }
      return Promise.resolve(new Response("not found", { status: 404 }));
    });
    vi.stubGlobal("fetch", fetchMock);

    const { container } = renderWithInspector(
      <LiveEventsProvider value={live.live()}>
        <ReviewScreen />
      </LiveEventsProvider>,
    );

    expect(await screen.findByText("Maya preference")).toBeInTheDocument();
    expect(screen.getByText("Maya prefers concise project memory summaries.")).toBeInTheDocument();

    const text = container.textContent ?? "";
    expect(text).toContain("evidence cluster size");
    expect(text).toContain("episode count");
    expect(text).toContain("3");
    expect(text).toContain("2");
    expect(text).not.toContain("vector(");
    expect(
      screen.getByRole("button", { name: "jump to semn_abcdefghijklmnop" }),
    ).toBeInTheDocument();
    expect(screen.getByRole("button", { name: "jump to ep_abcdefghijklmnop" })).toBeInTheDocument();
    expect(text).not.toContain("ent_");
  });

  it("refetches dream state on WebSocket reconnect after the initial connection", async () => {
    const live = makeLiveSource();
    const fetchMock = installFetch();
    const { rerender } = renderWithInspector(
      <LiveEventsProvider value={live.live(1)}>
        <DreamScreen />
      </LiveEventsProvider>,
    );

    await waitFor(() => {
      expect(
        fetchMock.mock.calls.filter((call) => requestPath(call[0]) === "/api/dream/state"),
      ).toHaveLength(1);
    });

    rerender(
      <LiveEventsProvider value={live.live(2)}>
        <DreamScreen />
      </LiveEventsProvider>,
    );

    await waitFor(() => {
      expect(
        fetchMock.mock.calls.filter((call) => requestPath(call[0]) === "/api/dream/state"),
      ).toHaveLength(2);
    });
  });

  it("refetches dream state and shows the maintenance tick indicator", async () => {
    const live = makeLiveSource();
    const fetchMock = installFetch();
    renderWithInspector(
      <LiveEventsProvider value={live.live()}>
        <DreamScreen />
      </LiveEventsProvider>,
    );

    await waitFor(() => {
      expect(
        fetchMock.mock.calls.filter((call) => requestPath(call[0]) === "/api/dream/state"),
      ).toHaveLength(1);
    });

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
    await waitFor(() => {
      expect(
        fetchMock.mock.calls.filter((call) => requestPath(call[0]) === "/api/dream/state"),
      ).toHaveLength(2);
    });
  });
});
