import { fireEvent, render, screen, waitFor } from "@testing-library/react";
import type { ReactNode } from "react";
import { afterEach, describe, expect, it, vi } from "vitest";

import type { ReviewKind, ReviewResolution } from "../../api/types";
import { GENERIC_REVIEW_ACTIONS } from "../../lib/review-actions";
import { Inspector } from "./Inspector";
import { IdRef } from "./IdRef";
import { InspectorProvider, useInspector } from "./inspector-context";
import { ID_PREFIX_OBJECT_TYPES, resolveObjectType, type ObjectType } from "./inspector-id";
import { isWhySupported, objectRegistry, type ObjectModel } from "./inspector-registry";

function jsonResponse(body: unknown, status = 200): Response {
  return new Response(JSON.stringify(body), {
    status,
    headers: { "Content-Type": "application/json" },
  });
}

function jsonErrorResponse(status: number, message: string): Response {
  return jsonResponse({ error: { status, message } }, status);
}

function requestUrl(request: RequestInfo | URL): URL {
  return new URL(String(request), "http://test.invalid");
}

function requestBody(init: RequestInit | undefined): Record<string, unknown> {
  return JSON.parse(String(init?.body ?? "{}")) as Record<string, unknown>;
}

function semanticNodeFixture(id: string) {
  return {
    id,
    kind: "proposition",
    label: "Direct node",
    description: "Direct semantic node description.",
    domain: "runtime",
    aliases: [],
    confidence: 0.86,
    status: "active",
    source_episode_ids: ["ep_source1111111111"],
    source_count: 1,
    created_at: 1,
    updated_at: 2,
  };
}

function commitmentFixture(id: string) {
  return {
    id,
    text: "Stay inside sanctioned endpoints.",
    type: "rule",
    kind: "assistant_commitment",
    enforcement_class: "critical",
    critical_domain: null,
    state: "active",
    priority: 0.8,
    directive_family: "demo",
    audience: null,
    made_to: null,
    about: null,
    committed_by: null,
    source: "operator",
    source_stream_entry_ids: ["strm_commit_source"],
    created_at: 1,
    expires_at: null,
    expired_at: null,
    revoked_at: null,
    revoked_reason: null,
    superseded_by_id: "cmt_next1111111111",
    canonicalized_by_artifact_entry_id: "dart_commit111111111",
    last_reinforced_at: 2,
  };
}

function episodeFixture(id: string) {
  return {
    id,
    title: "Global episode",
    narrative: "Resolved outside the active session.",
    participants: [],
    location: null,
    start_time: 1,
    end_time: 2,
    audience: "alice",
    origin_audience_entity_ids: ["ent_aaaaaaaaaaaaaaaa"],
    origin_audience_refs: [
      { value: "ent_aaaaaaaaaaaaaaaa", id: "ent_aaaaaaaaaaaaaaaa", label: "Alice" },
    ],
    shared: false,
    disclosure_class: "relationship_private",
    disclosure_label: {
      disclosure_class: "relationship_private",
      origin_audience_entity_ids: ["ent_aaaaaaaaaaaaaaaa"],
      private_to_entity_ids: ["ent_aaaaaaaaaaaaaaaa"],
      public_to_entity_ids: [],
    },
    significance: 0.5,
    confidence: 0.8,
    tags: [],
    source_stream_ids: ["strm_global_source"],
    source_count: 1,
    lineage: { derived_from: [], supersedes: [] },
    emotional_arc: null,
    vector_dims: 4,
    created_at: 1,
    updated_at: 2,
  };
}

function reviewFixture(id: number) {
  return {
    id,
    kind: "contradiction",
    refs: {
      node_ids: ["semn_review111111111"],
      edge_id: "seme_review111111111",
      episode_ids: ["ep_review1111111111"],
      target_id: "cmt_review111111111",
    },
    reason: "review reason",
    created_at: 1,
    resolved_at: null,
    resolution: null,
  };
}

type InspectorReviewDispatchCase = {
  kind: ReviewKind;
  action: ReviewResolution;
  path: string;
  body: Record<string, unknown>;
};

const INSPECTOR_REVIEW_DISPATCH_CASES: InspectorReviewDispatchCase[] = [
  {
    kind: "correction",
    action: GENERIC_REVIEW_ACTIONS.correction[0]!,
    path: "/api/correction/reviews/31",
    body: { action: "accept" },
  },
  {
    kind: "belief_revision",
    action: GENERIC_REVIEW_ACTIONS.belief_revision[0]!,
    path: "/api/dream/review/31",
    body: { action: "dismiss" },
  },
  {
    kind: "new_insight",
    action: GENERIC_REVIEW_ACTIONS.new_insight[0]!,
    path: "/api/reviews/31",
    body: { action: "accept" },
  },
];

function stateFixture() {
  return {
    active_session: "default",
    audiences: ["bob"],
    counts: {
      turns: 0,
      commitments: 0,
      open_qs: 0,
      open_reviews: 0,
      dream_audit_rows: 0,
    },
    current_mood: {
      session_id: "default",
      valence: 0,
      arousal: 0,
      updated_at: 1,
      half_life_hours: 12,
      recent_triggers: [],
    },
    version: "test",
  };
}

function sessionFixture(audienceLabel: string, sessionId = "sess_origin111111") {
  return {
    session_id: sessionId,
    source_type: "demo",
    source_external_id: null,
    source_url: null,
    label: "origin",
    audience_label: audienceLabel,
    audience_entity_id: null,
    conversation_kind: "demo",
    created_at: 1,
    last_activity_at: 1,
    last_turn_id: null,
    message_count: 0,
    status: "active",
    privacy_level: "payload_off",
    participation_policy: "active",
    audience_role: "participant",
  };
}

function renderWithInspector(children: ReactNode) {
  const setView = vi.fn();
  const setSessionId = vi.fn();
  const result = render(
    <InspectorProvider
      setView={setView}
      setSessionId={setSessionId}
      sessionId="default"
      audience="alice"
    >
      {children}
      <Inspector />
    </InspectorProvider>,
  );
  return { ...result, setView, setSessionId };
}

function renderProviderOnly(children: ReactNode) {
  const setView = vi.fn();
  const setSessionId = vi.fn();
  const result = render(
    <InspectorProvider
      setView={setView}
      setSessionId={setSessionId}
      sessionId="default"
      audience="alice"
    >
      {children}
    </InspectorProvider>,
  );
  return { ...result, setView, setSessionId };
}

function OpenButton({ type, id }: { type: ObjectType; id: string }) {
  const inspector = useInspector();
  return (
    <button type="button" onClick={() => inspector.openObject({ type, id })}>
      open {id}
    </button>
  );
}

function OpenSourceProbe({ type, id }: { type: ObjectType; id: string }) {
  const inspector = useInspector();
  return (
    <>
      <button type="button" onClick={() => inspector.openObject({ type, id })}>
        open {id}
      </button>
      <button type="button" onClick={inspector.openInSourceScreen}>
        source
      </button>
    </>
  );
}

function HintOpenButton({ type, id, hint }: { type: ObjectType; id: string; hint: unknown }) {
  const inspector = useInspector();
  return (
    <button type="button" onClick={() => inspector.openObject({ type, id, hint })}>
      open hinted {id}
    </button>
  );
}

function StackProbe() {
  const inspector = useInspector();
  return (
    <div>
      <div data-testid="stack">
        {inspector.targets.map((target) => `${target.type}:${target.id}`).join(">") || "empty"}
      </div>
      <button
        type="button"
        onClick={() => inspector.openObject({ type: "semantic_edge", id: "seme_stack" })}
      >
        open edge
      </button>
      <button
        type="button"
        onClick={() => inspector.openObject({ type: "semantic_node", id: "semn_stack" })}
      >
        pivot node
      </button>
      <button type="button" onClick={inspector.back}>
        back
      </button>
      <button type="button" onClick={inspector.close}>
        close
      </button>
    </div>
  );
}

afterEach(() => {
  vi.unstubAllGlobals();
  vi.restoreAllMocks();
});

describe("resolveObjectType", () => {
  it("maps every mirrored id prefix to the exact object type", () => {
    for (const [prefix, type] of ID_PREFIX_OBJECT_TYPES) {
      expect(resolveObjectType(`${prefix}abc123`)).toBe(type);
    }
    expect(resolveObjectType("default")).toBe("session");
  });

  it("returns null for unknown, numeric, and unprefixed ids", () => {
    expect(resolveObjectType("42")).toBeNull();
    expect(resolveObjectType("review_42")).toBeNull();
    expect(resolveObjectType("semn")).toBeNull();
    expect(resolveObjectType("plain-id")).toBeNull();
  });
});

describe("objectRegistry", () => {
  it("resolves direct semantic nodes and derives schema-known pivots", async () => {
    const nodeId = "semn_direct11111111";
    const fetchMock = vi.fn((request: RequestInfo | URL) => {
      expect(requestUrl(request).pathname).toBe(`/api/semantic/nodes/${nodeId}`);
      return Promise.resolve(jsonResponse({ node: semanticNodeFixture(nodeId) }));
    });
    vi.stubGlobal("fetch", fetchMock);

    const model: ObjectModel = objectRegistry.semantic_node;
    const node = await model.fetch(nodeId, { sessionId: "default", audience: "alice" });

    expect(model.reliability).toBe("direct");
    expect(isRecordWithLabel(node)).toBe(true);
    expect(model.pivots(node)).toEqual([
      { type: "episode", id: "ep_source1111111111", fieldLabel: "source_episode_ids" },
    ]);
  });

  it("resolves commitments from the list endpoint without audience filtering", async () => {
    const commitment = commitmentFixture("cmt_list1111111111");
    const fetchMock = vi.fn((request: RequestInfo | URL) => {
      const url = requestUrl(request);
      expect(url.pathname).toBe("/api/commitments");
      expect(url.searchParams.get("state")).toBe("all");
      expect(url.searchParams.has("audience")).toBe(false);
      return Promise.resolve(jsonResponse({ commitments: [commitment] }));
    });
    vi.stubGlobal("fetch", fetchMock);

    const model: ObjectModel = objectRegistry.commitment;
    const resolved = await model.fetch(commitment.id, { sessionId: "default", audience: "alice" });

    expect(model.reliability).toBe("in_list");
    expect(resolved).toEqual(commitment);
    expect(model.pivots(resolved)).toEqual([
      { type: "stream_entry", id: "strm_commit_source", fieldLabel: "source_stream_entry_ids" },
      { type: "commitment", id: "cmt_next1111111111", fieldLabel: "superseded_by_id" },
      {
        type: "shared_state_entry",
        id: "dart_commit111111111",
        fieldLabel: "canonicalized_by_artifact_entry_id",
      },
    ]);
  });

  it("resolves memory-band ids globally instead of using the current session", async () => {
    const episode = episodeFixture("ep_global1111111111");
    const fetchMock = vi.fn((request: RequestInfo | URL) => {
      const url = requestUrl(request);
      expect(url.pathname).toBe("/api/memory/bands/episodic");
      expect(url.searchParams.has("session")).toBe(false);
      expect(url.searchParams.has("audience")).toBe(false);
      return Promise.resolve(
        jsonResponse({
          band: "episodic",
          items: [episode],
          next_cursor: null,
        }),
      );
    });
    vi.stubGlobal("fetch", fetchMock);

    const model: ObjectModel = objectRegistry.episode;
    const resolved = await model.fetch(episode.id, {
      sessionId: "sess_wrong111111",
      audience: "wrong-audience",
    });

    expect(resolved).toEqual(episode);
  });

  it("resolves shared-state ids across discovered audiences instead of current audience", async () => {
    const entry = {
      id: "dart_global11111111",
      audience_entity_id: "ent_bob1111111111",
      state_key: "demo",
      kind: "live",
      text: "global shared state",
      owner_entity_id: null,
      provenance_stream_entry_ids: [],
      last_updated_stream_entry_ids: [],
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
    };
    const requestedAudiences: string[] = [];
    vi.stubGlobal(
      "fetch",
      vi.fn((request: RequestInfo | URL) => {
        const url = requestUrl(request);
        if (url.pathname === "/api/sessions") {
          return Promise.resolve(jsonResponse({ sessions: [sessionFixture("alice")] }));
        }
        if (url.pathname === "/api/state") {
          return Promise.resolve(jsonResponse(stateFixture()));
        }
        if (url.pathname === "/api/shared-state") {
          const audience = url.searchParams.get("audience") ?? "";
          requestedAudiences.push(audience);
          if (audience === "self") {
            return Promise.resolve(jsonErrorResponse(500, "self failed"));
          }
          return Promise.resolve(
            jsonResponse({
              audience,
              entries: audience === "bob" ? [entry] : [],
            }),
          );
        }
        return Promise.reject(new Error(`unexpected fetch ${url.pathname}`));
      }),
    );

    const model: ObjectModel = objectRegistry.shared_state_entry;
    const resolved = await model.fetch(entry.id, {
      sessionId: "sess_wrong111111",
      audience: "wrong-audience",
    });

    expect(resolved).toEqual(entry);
    expect(requestedAudiences).toContain("bob");
    expect(requestedAudiences).not.toContain("wrong-audience");
  });

  it("resolves stream entries across sessions and ignores failed session fetches", async () => {
    const entry = {
      id: "strm_nondefault1111",
      timestamp: 1,
      kind: "user_msg",
      content: "non-default stream entry",
      sender_entity_id: null,
      reply_target_entity_id: null,
      session_id: "sess_nondefault111",
      compressed: false,
    };
    const requestedSessions: string[] = [];
    vi.stubGlobal(
      "fetch",
      vi.fn((request: RequestInfo | URL) => {
        const url = requestUrl(request);
        if (url.pathname === "/api/sessions") {
          return Promise.resolve(
            jsonResponse({ sessions: [sessionFixture("bob", "sess_nondefault111")] }),
          );
        }
        if (url.pathname === "/api/stream") {
          const session = url.searchParams.get("session") ?? "";
          requestedSessions.push(session);
          if (session === "default") {
            return Promise.resolve(jsonErrorResponse(500, "default failed"));
          }
          return Promise.resolve(jsonResponse({ entries: [entry], next_cursor: null }));
        }
        return Promise.reject(new Error(`unexpected fetch ${url.pathname}`));
      }),
    );

    const model: ObjectModel = objectRegistry.stream_entry;
    const resolved = await model.fetch(entry.id, {
      sessionId: "sess_wrong111111",
      audience: "wrong-audience",
    });

    expect(resolved).toEqual(entry);
    expect(requestedSessions).toContain("default");
    expect(requestedSessions).toContain("sess_nondefault111");
  });

  it("resolves turns across sessions instead of default-only history", async () => {
    const turn = {
      turn_id: "turn_nondefault",
      started_at: 1,
      audience: "bob",
      outcome: "emitted",
      suppression_reason: null,
    };
    vi.stubGlobal(
      "fetch",
      vi.fn((request: RequestInfo | URL) => {
        const url = requestUrl(request);
        if (url.pathname === "/api/sessions") {
          return Promise.resolve(
            jsonResponse({ sessions: [sessionFixture("bob", "sess_turn1111111")] }),
          );
        }
        if (url.pathname === "/api/turns") {
          const session = url.searchParams.get("session");
          return Promise.resolve(
            jsonResponse({
              rows: session === "sess_turn1111111" ? [turn] : [],
              next_cursor: null,
            }),
          );
        }
        return Promise.reject(new Error(`unexpected fetch ${url.pathname}`));
      }),
    );

    const model: ObjectModel = objectRegistry.turn;
    const resolved = await model.fetch(turn.turn_id, {
      sessionId: "sess_wrong111111",
      audience: "wrong-audience",
    });

    expect(resolved).toEqual(turn);
  });

  it("resolves image perceptions across sessions", async () => {
    const perception = {
      perception_id: "imgp_nondefault111",
      payload_id: "payload",
      attachment_id: "att_nondefault111",
      caption: "non-default image",
      image_kind: "screenshot",
      active: true,
      audience: "bob",
      visible_text: [],
      objects: [],
      people_or_roles: [],
      scene: "",
      colors_and_visual_attributes: [],
      spatial_relationships: [],
      possible_user_relevant_details: [],
      search_terms: [],
      uncertainties: [],
      embedding_status: "complete",
    };
    vi.stubGlobal(
      "fetch",
      vi.fn((request: RequestInfo | URL) => {
        const url = requestUrl(request);
        if (url.pathname === "/api/sessions") {
          return Promise.resolve(
            jsonResponse({ sessions: [sessionFixture("bob", "sess_image111111")] }),
          );
        }
        if (url.pathname === "/api/stream") {
          const session = url.searchParams.get("session");
          return Promise.resolve(
            jsonResponse({
              entries:
                session === "sess_image111111"
                  ? [
                      {
                        id: "strm_image1111111",
                        timestamp: 1,
                        kind: "user_image_attachment",
                        content: {
                          attachment_id: "att_nondefault111",
                          perception_id: perception.perception_id,
                        },
                        sender_entity_id: null,
                        reply_target_entity_id: null,
                        session_id: "sess_image111111",
                        compressed: false,
                      },
                    ]
                  : [],
              next_cursor: null,
            }),
          );
        }
        if (url.pathname === "/api/attachments/att_nondefault111") {
          return Promise.resolve(
            jsonResponse({
              attachment: {
                attachment_id: "att_nondefault111",
                sha256: "sha",
                media_type: "image/png",
                byte_size: 1,
                width: 1,
                height: 1,
                storage_ref: "x",
                thumbnail_ref: null,
                perception_id: perception.perception_id,
                text_embedding_ref: null,
                visual_embedding_ref: null,
                active: true,
                audience: "bob",
                created_turn_global: null,
                parent_entry_id: "strm_image1111111",
                stream_entry_id: "strm_image1111111",
                parent_turn_id: "turn_nondefault",
                created_at: 1,
              },
              perception,
              status: {
                active: true,
                quarantined: false,
                stream_active: true,
                parent_active: true,
              },
            }),
          );
        }
        return Promise.reject(new Error(`unexpected fetch ${url.pathname}`));
      }),
    );

    const model: ObjectModel = objectRegistry.image_perception;
    const resolved = await model.fetch(perception.perception_id, {
      sessionId: "sess_wrong111111",
      audience: "wrong-audience",
    });

    expect(resolved).toEqual(perception);
  });

  it("keeps creator directives out of the correctable/evidence set", () => {
    expect(objectRegistry.creator_directive.reliability).toBe("in_list");
    expect(objectRegistry.creator_directive.tabs).not.toContain("evidence");
    expect(isWhySupported("creator_directive")).toBe(false);
  });
});

describe("InspectorProvider", () => {
  it("maintains a target stack for open, pivot, back, and close", () => {
    renderProviderOnly(<StackProbe />);

    fireEvent.click(screen.getByRole("button", { name: "open edge" }));
    expect(screen.getByTestId("stack")).toHaveTextContent("semantic_edge:seme_stack");

    fireEvent.click(screen.getByRole("button", { name: "pivot node" }));
    expect(screen.getByTestId("stack")).toHaveTextContent(
      "semantic_edge:seme_stack>semantic_node:semn_stack",
    );

    fireEvent.click(screen.getByRole("button", { name: "back" }));
    expect(screen.getByTestId("stack")).toHaveTextContent("semantic_edge:seme_stack");

    fireEvent.click(screen.getByRole("button", { name: "close" }));
    expect(screen.getByTestId("stack")).toHaveTextContent("empty");
  });

  it("carries Governance tab intent when opening source screens", () => {
    const { rerender, setView } = renderProviderOnly(
      <OpenSourceProbe type="commitment" id="cmt_source111111" />,
    );

    fireEvent.click(screen.getByRole("button", { name: "open cmt_source111111" }));
    fireEvent.click(screen.getByRole("button", { name: "source" }));
    expect(setView).toHaveBeenLastCalledWith("governance", {
      governanceTab: "commitments",
    });

    rerender(
      <InspectorProvider
        setView={setView}
        setSessionId={vi.fn()}
        sessionId="default"
        audience="alice"
      >
        <OpenSourceProbe type="creator_directive" id="cdir_source111111" />
      </InspectorProvider>,
    );
    fireEvent.click(screen.getByRole("button", { name: "open cdir_source111111" }));
    fireEvent.click(screen.getByRole("button", { name: "source" }));
    expect(setView).toHaveBeenLastCalledWith("governance", {
      governanceTab: "shared_state",
    });

    rerender(
      <InspectorProvider
        setView={setView}
        setSessionId={vi.fn()}
        sessionId="default"
        audience="alice"
      >
        <OpenSourceProbe type="shared_state_entry" id="dart_source111111" />
      </InspectorProvider>,
    );
    fireEvent.click(screen.getByRole("button", { name: "open dart_source111111" }));
    fireEvent.click(screen.getByRole("button", { name: "source" }));
    expect(setView).toHaveBeenLastCalledWith("governance", {
      governanceTab: "shared_state",
    });
  });
});

describe("Inspector drawer", () => {
  it("renders a direct semantic node with registry-gated tabs and raw JSON", async () => {
    const nodeId = "semn_drawer11111111";
    vi.stubGlobal(
      "fetch",
      vi.fn((request: RequestInfo | URL) => {
        if (requestUrl(request).pathname === `/api/semantic/nodes/${nodeId}`) {
          return Promise.resolve(jsonResponse({ node: semanticNodeFixture(nodeId) }));
        }
        return Promise.reject(new Error(`unexpected fetch ${requestUrl(request).pathname}`));
      }),
    );

    renderWithInspector(<OpenButton type="semantic_node" id={nodeId} />);
    fireEvent.click(screen.getByRole("button", { name: `open ${nodeId}` }));

    expect(
      await screen.findByRole("dialog", { name: "Semantic node inspector" }),
    ).toBeInTheDocument();
    expect(await screen.findByText("Direct node")).toBeInTheDocument();
    expect(screen.getByRole("tab", { name: "Evidence" })).toBeInTheDocument();
    expect(screen.getByRole("tab", { name: "Actions" })).toBeInTheDocument();

    fireEvent.click(screen.getByRole("tab", { name: "Raw JSON" }));
    expect(await screen.findByText("source_episode_ids")).toBeInTheDocument();
  });

  it("renders disclosure labels in generic summary rows", async () => {
    const episode = episodeFixture("ep_drawer111111111");
    vi.stubGlobal(
      "fetch",
      vi.fn((request: RequestInfo | URL) => {
        if (requestUrl(request).pathname === "/api/memory/bands/episodic") {
          return Promise.resolve(
            jsonResponse({
              band: "episodic",
              items: [episode],
              next_cursor: null,
            }),
          );
        }
        return Promise.reject(new Error(`unexpected fetch ${requestUrl(request).pathname}`));
      }),
    );

    renderWithInspector(<OpenButton type="episode" id={episode.id} />);
    fireEvent.click(screen.getByRole("button", { name: `open ${episode.id}` }));

    expect(await screen.findByRole("dialog", { name: "Episode inspector" })).toBeInTheDocument();
    expect(await screen.findByText("Global episode")).toBeInTheDocument();
    expect(screen.getByText("labels")).toBeInTheDocument();
    expect(screen.getByText("private")).toHaveClass("tag", "purple");
  });

  it("renders a list-scoped commitment and its schema-known relationship chips", async () => {
    const commitment = commitmentFixture("cmt_drawer11111111");
    vi.stubGlobal(
      "fetch",
      vi.fn((request: RequestInfo | URL) => {
        if (requestUrl(request).pathname === "/api/commitments") {
          return Promise.resolve(jsonResponse({ commitments: [commitment] }));
        }
        return Promise.reject(new Error(`unexpected fetch ${requestUrl(request).pathname}`));
      }),
    );

    renderWithInspector(<OpenButton type="commitment" id={commitment.id} />);
    fireEvent.click(screen.getByRole("button", { name: `open ${commitment.id}` }));

    expect(await screen.findByText("Stay inside sanctioned endpoints.")).toBeInTheDocument();
    fireEvent.click(screen.getByRole("tab", { name: "Relationships" }));

    expect(await screen.findByText("source_stream_entry_ids")).toBeInTheDocument();
    expect(screen.getByTitle("Jump to strm_commit_source")).toBeInTheDocument();
  });

  it("renders a needs-backend placeholder without fetching", async () => {
    const fetchMock = vi.fn();
    vi.stubGlobal("fetch", fetchMock);

    renderWithInspector(<OpenButton type="entity" id="ent_missing11111111" />);
    fireEvent.click(screen.getByRole("button", { name: "open ent_missing11111111" }));

    expect(
      await screen.findByText("Entity does not have a direct resolver for ent_missing11111111."),
    ).toBeInTheDocument();
    expect(fetchMock).not.toHaveBeenCalled();
  });

  it("opens a sniffed object from an IdRef click", async () => {
    const nodeId = "semn_idref111111111";
    vi.stubGlobal(
      "fetch",
      vi.fn((request: RequestInfo | URL) => {
        if (requestUrl(request).pathname === `/api/semantic/nodes/${nodeId}`) {
          return Promise.resolve(jsonResponse({ node: semanticNodeFixture(nodeId) }));
        }
        return Promise.reject(new Error(`unexpected fetch ${requestUrl(request).pathname}`));
      }),
    );

    renderWithInspector(<IdRef id={nodeId} />);
    fireEvent.click(screen.getByRole("button", { name: `jump to ${nodeId}` }));

    expect(
      await screen.findByRole("dialog", { name: "Semantic node inspector" }),
    ).toBeInTheDocument();
    expect(await screen.findByText("Direct node")).toBeInTheDocument();
  });

  it("uses hints only as optimistic display data and still fetches the authoritative object", async () => {
    const nodeId = "semn_hint111111111";
    const fetchMock = vi.fn((request: RequestInfo | URL) => {
      if (requestUrl(request).pathname === `/api/semantic/nodes/${nodeId}`) {
        return Promise.resolve(jsonResponse({ node: semanticNodeFixture(nodeId) }));
      }
      return Promise.reject(new Error(`unexpected fetch ${requestUrl(request).pathname}`));
    });
    vi.stubGlobal("fetch", fetchMock);

    renderWithInspector(
      <HintOpenButton
        type="semantic_node"
        id={nodeId}
        hint={{ ...semanticNodeFixture(nodeId), label: "Stale hint node" }}
      />,
    );
    fireEvent.click(screen.getByRole("button", { name: `open hinted ${nodeId}` }));

    expect(await screen.findByText("Direct node")).toBeInTheDocument();
    await waitFor(() => expect(fetchMock).toHaveBeenCalledTimes(1));
  });

  it("degrades numeric reviews as list-only objects and hides evidence", async () => {
    const fetchMock = vi.fn((request: RequestInfo | URL) => {
      const path = requestUrl(request).pathname;
      if (path === "/api/reviews" || path === "/api/correction/reviews") {
        return Promise.resolve(jsonResponse({ rows: [] }));
      }
      return Promise.reject(new Error(`unexpected fetch ${path}`));
    });
    vi.stubGlobal("fetch", fetchMock);

    renderWithInspector(<OpenButton type="review" id="42" />);
    fireEvent.click(screen.getByRole("button", { name: "open 42" }));

    expect(
      await screen.findByText(
        "Review is available only from the currently loaded list; 42 was not found.",
      ),
    ).toBeInTheDocument();
    expect(screen.queryByRole("tab", { name: "Evidence" })).not.toBeInTheDocument();
    await waitFor(() => expect(fetchMock).toHaveBeenCalledTimes(2));
  });

  it("renders numeric review rows when they are present in review lists", async () => {
    const review = reviewFixture(7);
    vi.stubGlobal(
      "fetch",
      vi.fn((request: RequestInfo | URL) => {
        const path = requestUrl(request).pathname;
        if (path === "/api/reviews") {
          return Promise.resolve(jsonResponse({ rows: [review] }));
        }
        if (path === "/api/correction/reviews") {
          return Promise.resolve(jsonResponse({ rows: [] }));
        }
        return Promise.reject(new Error(`unexpected fetch ${path}`));
      }),
    );

    renderWithInspector(<OpenButton type="review" id="7" />);
    fireEvent.click(screen.getByRole("button", { name: "open 7" }));

    expect(await screen.findByText("review reason")).toBeInTheDocument();
    fireEvent.click(screen.getByRole("tab", { name: "Relationships" }));
    expect(await screen.findByText("refs.node_ids")).toBeInTheDocument();
  });

  it("uses the creator-directive reconciliation endpoint for that review kind", async () => {
    const review = {
      id: 9,
      kind: "creator_directive_reconciliation",
      refs: { directive_ids: ["cdir_one111111111", "cdir_two111111111"] },
      reason: "directive reconciliation",
      created_at: 1,
      resolved_at: null,
      resolution: null,
    };
    const fetchMock = vi.fn((request: RequestInfo | URL, init?: RequestInit) => {
      const url = requestUrl(request);
      if (url.pathname === "/api/reviews" && init?.method !== "POST") {
        return Promise.resolve(jsonResponse({ rows: [review] }));
      }
      if (url.pathname === "/api/correction/reviews") {
        return Promise.resolve(jsonResponse({ rows: [] }));
      }
      if (url.pathname === "/api/reviews/9/creator-directive-reconciliation") {
        expect(init?.method).toBe("POST");
        expect(JSON.parse(String(init?.body))).toEqual({ action: "keep" });
        return Promise.resolve(jsonResponse({ ...review, resolved_at: 2, resolution: "keep" }));
      }
      return Promise.reject(new Error(`unexpected fetch ${url.pathname}`));
    });
    vi.stubGlobal("fetch", fetchMock);

    renderWithInspector(<OpenButton type="review" id="9" />);
    fireEvent.click(screen.getByRole("button", { name: "open 9" }));
    fireEvent.click(await screen.findByRole("tab", { name: "Actions" }));
    fireEvent.click(await screen.findByRole("button", { name: "keep directives" }));
    fireEvent.click(await screen.findByRole("button", { name: "keep" }));

    await waitFor(() =>
      expect(fetchMock).toHaveBeenCalledWith(
        "/api/reviews/9/creator-directive-reconciliation",
        expect.objectContaining({ method: "POST" }),
      ),
    );
    expect(
      fetchMock.mock.calls.some(
        ([request, init]) =>
          requestUrl(request).pathname === "/api/reviews/9" && init?.method === "PATCH",
      ),
    ).toBe(false);
  });

  it.each(INSPECTOR_REVIEW_DISPATCH_CASES)(
    "routes Inspector review $kind $action through the shared dispatcher",
    async ({ kind, action, path, body }) => {
      const review = { ...reviewFixture(31), kind, reason: `${kind} inspector route` };
      const fetchMock = vi.fn((request: RequestInfo | URL, init?: RequestInit) => {
        const url = requestUrl(request);
        if (url.pathname === "/api/reviews" && init?.method === undefined) {
          return Promise.resolve(jsonResponse({ rows: [review] }));
        }
        if (url.pathname === "/api/correction/reviews" && init?.method === undefined) {
          return Promise.resolve(jsonResponse({ rows: [] }));
        }
        if (url.pathname === path) {
          return Promise.resolve(jsonResponse({ ...review, resolved_at: 2, resolution: action }));
        }
        return Promise.reject(new Error(`unexpected fetch ${url.pathname}`));
      });
      vi.stubGlobal("fetch", fetchMock);

      renderWithInspector(<OpenButton type="review" id="31" />);
      fireEvent.click(screen.getByRole("button", { name: "open 31" }));
      fireEvent.click(await screen.findByRole("tab", { name: "Actions" }));
      fireEvent.click(await screen.findByRole("button", { name: action }));
      const confirmButtons = await screen.findAllByRole("button", { name: action });
      fireEvent.click(confirmButtons.at(-1)!);

      await waitFor(() => {
        const call = fetchMock.mock.calls.find(
          ([request, init]) => requestUrl(request).pathname === path && init?.method !== undefined,
        );
        expect(call).toBeDefined();
        expect(call?.[1]?.method).toBe(
          kind === "creator_directive_reconciliation" ? "POST" : "PATCH",
        );
        expect(requestBody(call?.[1])).toEqual(body);
      });
      expect(
        fetchMock.mock.calls.some(
          ([request, init]) =>
            requestUrl(request).pathname === "/api/reviews/31" &&
            init?.method === "PATCH" &&
            path !== "/api/reviews/31",
        ),
      ).toBe(false);
    },
  );

  it("omits empty optional reason fields for commitment revoke actions", async () => {
    const commitment = commitmentFixture("cmt_revoke11111111");
    const fetchMock = vi.fn((request: RequestInfo | URL, init?: RequestInit) => {
      const url = requestUrl(request);
      if (url.pathname === "/api/commitments" && init?.method !== "POST") {
        return Promise.resolve(jsonResponse({ commitments: [commitment] }));
      }
      if (url.pathname === `/api/commitments/${commitment.id}/revoke`) {
        expect(init?.method).toBe("POST");
        expect(JSON.parse(String(init?.body))).toEqual({});
        return Promise.resolve(jsonResponse({ ...commitment, state: "revoked" }));
      }
      return Promise.reject(new Error(`unexpected fetch ${url.pathname}`));
    });
    vi.stubGlobal("fetch", fetchMock);

    renderWithInspector(<OpenButton type="commitment" id={commitment.id} />);
    fireEvent.click(screen.getByRole("button", { name: `open ${commitment.id}` }));
    fireEvent.click(await screen.findByRole("tab", { name: "Actions" }));
    fireEvent.click(await screen.findByRole("button", { name: "revoke commitment" }));
    fireEvent.click(await screen.findByRole("button", { name: "revoke" }));

    await waitFor(() =>
      expect(fetchMock).toHaveBeenCalledWith(
        `/api/commitments/${commitment.id}/revoke`,
        expect.objectContaining({ method: "POST" }),
      ),
    );
  });

  it("keeps the Inspector open when Escape closes a nested action modal", async () => {
    const commitment = commitmentFixture("cmt_escape11111111");
    vi.stubGlobal(
      "fetch",
      vi.fn((request: RequestInfo | URL) => {
        if (requestUrl(request).pathname === "/api/commitments") {
          return Promise.resolve(jsonResponse({ commitments: [commitment] }));
        }
        return Promise.reject(new Error(`unexpected fetch ${requestUrl(request).pathname}`));
      }),
    );

    renderWithInspector(<OpenButton type="commitment" id={commitment.id} />);
    fireEvent.click(screen.getByRole("button", { name: `open ${commitment.id}` }));
    fireEvent.click(await screen.findByRole("tab", { name: "Actions" }));
    fireEvent.click(await screen.findByRole("button", { name: "revoke commitment" }));

    await waitFor(() => expect(document.querySelector(".modal-title")).toBeInTheDocument());
    expect(document.querySelector(".modal-title")?.textContent).toContain("revoke");
    expect(document.querySelector(".modal-title")?.textContent).not.toContain(commitment.id);
    fireEvent.keyDown(window, { key: "Escape" });

    await waitFor(() => expect(document.querySelector(".modal-title")).not.toBeInTheDocument());
    expect(screen.getByRole("dialog", { name: "Commitment inspector" })).toBeInTheDocument();
  });

  it("degrades historical turn ledgers when the live cache no longer retains them", async () => {
    const turnId = "turn_historical";
    vi.stubGlobal(
      "fetch",
      vi.fn((request: RequestInfo | URL) => {
        if (requestUrl(request).pathname === "/api/sessions") {
          return Promise.resolve(jsonResponse({ sessions: [] }));
        }
        if (requestUrl(request).pathname === "/api/turns") {
          return Promise.resolve(
            jsonResponse({
              rows: [
                {
                  turn_id: turnId,
                  started_at: 1,
                  audience: "alice",
                  outcome: "emitted",
                  suppression_reason: null,
                },
              ],
              next_cursor: null,
            }),
          );
        }
        if (requestUrl(request).pathname === `/api/turns/${turnId}/ledger`) {
          return Promise.resolve(jsonErrorResponse(404, "not retained"));
        }
        return Promise.reject(new Error(`unexpected fetch ${requestUrl(request).pathname}`));
      }),
    );

    renderWithInspector(<OpenButton type="turn" id={turnId} />);
    fireEvent.click(screen.getByRole("button", { name: `open ${turnId}` }));
    fireEvent.click(await screen.findByRole("tab", { name: "Evidence" }));

    expect(await screen.findByText(/ledger not retained/)).toBeInTheDocument();
  });
});

function isRecordWithLabel(value: unknown): value is { label: string } {
  return typeof value === "object" && value !== null && "label" in value;
}
