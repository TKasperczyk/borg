import { act, fireEvent, render, screen, waitFor } from "@testing-library/react";
import { afterEach, describe, expect, it, vi } from "vitest";

import type { LiveFrame, StreamEntry, WsState } from "../api/types";
import { LiveEventsProvider } from "../hooks/live-context";
import type { LiveEventHandler, LiveEvents } from "../hooks/use-live-events";
import { CommitScreen } from "./Commit";
import { DreamScreen } from "./Dream";
import { GraphScreen } from "./Graph";
import { IdentityScreen } from "./Identity";
import { MemoryScreen } from "./Memory";
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
              content: { text: "observed action state" },
            }),
            streamEntry({ id: "strm_user", timestamp: 1, kind: "user_msg", content: "hello" }),
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
          counts: { turns: 1, commitments: 1, open_qs: 1, dream_audit_rows: 1 },
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
    if (url.pathname === "/api/semantic/graph") {
      return Promise.resolve(
        jsonResponse({
          nodes: [
            { id: "semn_alice", label: "alice", status: "active", kind: "entity", edge_count: 2 },
            { id: "semn_borg", label: "borg", status: "contested", kind: "entity", edge_count: 1 },
            {
              id: "semn_memory",
              label: "semantic memory",
              status: "contradicted",
              kind: "concept",
              edge_count: 1,
            },
          ],
          edges: [
            {
              id: "seme_support",
              source: "semn_alice",
              target: "semn_borg",
              type: "supports",
              weight: 0.9,
            },
            {
              id: "seme_contradict",
              source: "semn_borg",
              target: "semn_memory",
              type: "contradicts",
              weight: 0.4,
            },
          ],
          total_nodes: 3,
          total_edges: 2,
          rendered: { nodes: 3, edges: 2 },
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
    render(<SharedScreen />);

    expect(await screen.findByText("alice likes terse answers")).toBeInTheDocument();
    expect(screen.getAllByText("locked")[0]).toBeInTheDocument();
  });

  it("renders dream belief-revision rows", async () => {
    const live = makeLiveSource();
    installFetch();
    render(
      <LiveEventsProvider value={live.live()}>
        <DreamScreen />
      </LiveEventsProvider>,
    );

    expect(await screen.findByText("dependency invalidated")).toBeInTheDocument();
    expect(screen.getByText(/semantic_node:sn_1/)).toBeInTheDocument();
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

  it("renders semantic graph endpoint data", async () => {
    installFetch();
    const { container } = render(<GraphScreen />);

    expect(
      await screen.findByText(
        (content) => content.includes("3 nodes") && content.includes("showing 3 of 3"),
      ),
    ).toBeInTheDocument();
    await waitFor(() => {
      expect(container.querySelectorAll(".graph-node")).toHaveLength(3);
      expect(container.querySelectorAll(".graph-edge")).toHaveLength(2);
    });
  });
});
