import { act, fireEvent, render, screen, waitFor } from "@testing-library/react";
import { afterEach, describe, expect, it, vi } from "vitest";

import type { LiveFrame, StreamEntry, WsState } from "../api/types";
import { LiveEventsProvider } from "../hooks/live-context";
import type { LiveEventHandler, LiveEvents } from "../hooks/use-live-events";
import { CommitScreen } from "./Commit";
import { DreamScreen } from "./Dream";
import { IdentityScreen } from "./Identity";
import { MemoryScreen } from "./Memory";
import { ReviewScreen } from "./Review";
import { SharedScreen } from "./Shared";
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
    render(
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

  it("refetches stream rows on WebSocket reconnect after the initial connection", async () => {
    const live = makeLiveSource();
    const fetchMock = installFetch();
    const { rerender } = render(
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
    render(<MemoryScreen sessionId="default" />);

    const proceduralLabels = await screen.findAllByText("procedural");
    fireEvent.click(proceduralLabels[0]?.closest(".band-card") ?? proceduralLabels[0]!);

    expect((await screen.findAllByText("debugging migrations")).length).toBeGreaterThan(0);
    expect(screen.getAllByText(/alpha 5.0/).length).toBeGreaterThan(0);
  });

  it("renders full open-question lifecycle in identity", async () => {
    installFetch();
    render(<IdentityScreen />);

    expect(await screen.findByText("what remains unresolved?")).toBeInTheDocument();
    expect(screen.getAllByText("abandoned")[0]).toBeInTheDocument();
    expect(screen.getByText(/current period: current arc/)).toBeInTheDocument();
  });

  it("filters commitment lifecycle rows", async () => {
    installFetch();
    render(<CommitScreen />);

    expect((await screen.findAllByText("active rule")).length).toBeGreaterThan(0);
    fireEvent.click(screen.getByText("revoked"));

    await waitFor(() => {
      expect(screen.queryAllByText("active rule")).toHaveLength(0);
    });
    expect(screen.getAllByText("revoked rule").length).toBeGreaterThan(0);
  });

  it("renders shared-state lifecycle entries from state audiences", async () => {
    installFetch();
    render(<SharedScreen sessionId="default" />);

    expect(await screen.findByText("alice likes terse answers")).toBeInTheDocument();
    expect(screen.getAllByText("locked")[0]).toBeInTheDocument();
    expect(screen.getByText("live - low salience")).toBeInTheDocument();
    expect(screen.getByText("live - dormant")).toBeInTheDocument();
    expect(screen.getByText("pending (legacy)")).toBeInTheDocument();
    expect(screen.getByText("live 2")).toBeInTheDocument();
    expect(screen.queryByText("pending 1")).not.toBeInTheDocument();
  });

  it("links dream belief-revision rows to the unified review screen", async () => {
    const live = makeLiveSource();
    installFetch();
    const openReview = vi.fn();
    render(
      <LiveEventsProvider value={live.live()}>
        <DreamScreen onOpenReview={openReview} />
      </LiveEventsProvider>,
    );

    expect(await screen.findByText("belief revisions")).toBeInTheDocument();
    expect(screen.getByText("1 open")).toBeInTheDocument();
    fireEvent.click(screen.getByRole("button", { name: "open review" }));
    expect(openReview).toHaveBeenCalledTimes(1);
  });

  it("reverts only reversible unreverted dream audit rows", async () => {
    const live = makeLiveSource();
    const reversibleRow = {
      id: 7,
      run_id: "run_audit",
      process: "creator-directive-reconciler",
      action: "creator_directive_merge",
      targets: { survivor_id: "cdir_survivor", superseded_ids: ["cdir_loser"] },
      reversal: { superseded: [{ id: "cdir_loser", expected_record_version: 2 }] },
      applied_at: 4,
      reverted_at: null,
      reverted_by: null,
    };
    const nonReversibleRow = {
      ...reversibleRow,
      id: 8,
      action: "noop",
      targets: { id: "no_reverse" },
      reversal: {},
    };
    const revertedRow = {
      ...reversibleRow,
      id: 9,
      reverted_at: 6,
      reverted_by: "demo_operator",
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

    render(
      <LiveEventsProvider value={live.live()}>
        <DreamScreen />
      </LiveEventsProvider>,
    );

    const enabled = await screen.findByRole("button", { name: "revert audit 7" });
    expect(enabled).not.toBeDisabled();
    expect(screen.getByRole("button", { name: "revert audit 8" })).toBeDisabled();
    expect(screen.getByRole("button", { name: "revert audit 9" })).toBeDisabled();
    expect(screen.getByText("auto-resolved")).toBeInTheDocument();
    expect(screen.getAllByText("reverted").length).toBeGreaterThan(0);

    fireEvent.click(enabled);

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

    render(
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

    const { container } = render(
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
    expect(text).not.toContain("semn_");
    expect(text).not.toContain("ep_");
    expect(text).not.toContain("ent_");
  });

  it("refetches dream state on WebSocket reconnect after the initial connection", async () => {
    const live = makeLiveSource();
    const fetchMock = installFetch();
    const { rerender } = render(
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
    render(
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
