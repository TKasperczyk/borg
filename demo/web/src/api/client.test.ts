import { afterEach, describe, expect, it, vi } from "vitest";

import {
  ApiError,
  attachmentBytesUrl,
  getSemanticGraph,
  getSemanticEdge,
  getLedger,
  getSemanticNode,
  getSessions,
  getStream,
  postTurn,
  subscribeClientFetchErrors,
} from "./client";

const unsubscribers: Array<() => void> = [];

afterEach(() => {
  for (const unsubscribe of unsubscribers.splice(0)) {
    unsubscribe();
  }
  vi.unstubAllGlobals();
});

function mockFetch(response: Response): ReturnType<typeof vi.fn> {
  const fetchMock = vi.fn(async () => response);
  vi.stubGlobal("fetch", fetchMock);
  return fetchMock;
}

describe("api client", () => {
  it("constructs stream query strings with backend kind names", async () => {
    const fetchMock = mockFetch(
      new Response(JSON.stringify({ entries: [], next_cursor: null }), {
        status: 200,
        headers: { "Content-Type": "application/json" },
      }),
    );

    await getStream({
      session: "sess_custom",
      audience: "alice",
      kinds: ["user_msg", "agent_msg", "user_image_attachment"],
      limit: 50,
    });

    const requested = String(fetchMock.mock.calls[0]?.[0]);
    // Default API base is same-origin (empty) so the browser sends a relative URL
    // that goes through the dev proxy / production host as configured.
    expect(requested.startsWith("/api/stream?")).toBe(true);
    const url = new URL(requested, "http://test.invalid");
    expect(url.pathname).toBe("/api/stream");
    expect(url.searchParams.get("session")).toBe("sess_custom");
    expect(url.searchParams.get("audience")).toBe("alice");
    expect(url.searchParams.get("kind")).toBe("user_msg,agent_msg,user_image_attachment");
    expect(url.searchParams.get("limit")).toBe("50");
  });

  it("posts turns with message, external id, audience, and session", async () => {
    const fetchMock = mockFetch(
      new Response(JSON.stringify({ ok: true, status: "enqueued", stream_entry_id: "strm_123" }), {
        status: 200,
        headers: { "Content-Type": "application/json" },
      }),
    );

    await postTurn({
      message: "hello",
      external_message_id: "msg_123",
      audience: "alice",
      session: "sess_custom",
    });

    const init = fetchMock.mock.calls[0]?.[1] as RequestInit;
    expect(fetchMock.mock.calls[0]?.[0]).toBe("/api/turn");
    expect(init.method).toBe("POST");
    expect(JSON.parse(String(init.body))).toEqual({
      message: "hello",
      external_message_id: "msg_123",
      audience: "alice",
      session: "sess_custom",
    });
  });

  it("fetches sessions from the registry endpoint", async () => {
    const fetchMock = mockFetch(
      new Response(JSON.stringify({ sessions: [] }), {
        status: 200,
        headers: { "Content-Type": "application/json" },
      }),
    );

    await getSessions();

    expect(fetchMock.mock.calls[0]?.[0]).toBe("/api/sessions");
  });

  it("normalizes disclosure metadata on fetched ledger entries", async () => {
    mockFetch(
      new Response(
        JSON.stringify({
          turn_id: "turn_ledger",
          ledger: {
            sections: [
              {
                id: "shared_state",
                label: "shared state",
                entries: [
                  {
                    id: "entry_private",
                    source_type: "shared_state",
                    session_scope: "global",
                    actor: "memory",
                    trust_rank: 1,
                    text: "private state",
                    state_metadata: {
                      disclosure_label: {
                        disclosure_class: "relationship_private",
                        origin_audience_entity_ids: ["ent_aaaaaaaaaaaaaaaa"],
                        private_to_entity_ids: ["ent_aaaaaaaaaaaaaaaa"],
                        public_to_entity_ids: [],
                      },
                      disclosure_note: "private source",
                      current_audience_entity_id: "ent_aaaaaaaaaaaaaaaa",
                    },
                  },
                ],
              },
            ],
            sharedState: null,
            transcriptIncluded: false,
            transcriptCompacted: false,
            originalTranscriptTokenEstimate: 0,
            compactedTranscriptEntryCount: 0,
            rawPreservedUserTranscriptEntryCount: 0,
            estimatedTokens: 0,
          },
        }),
        {
          status: 200,
          headers: { "Content-Type": "application/json" },
        },
      ),
    );

    const response = await getLedger("turn_ledger");
    const entry = response.ledger.sections[0]?.entries[0];

    expect(entry?.disclosure_label?.disclosure_class).toBe("relationship_private");
    expect(entry?.disclosure_note).toBe("private source");
    expect(entry?.current_audience_entity_id).toBe("ent_aaaaaaaaaaaaaaaa");
  });

  it("constructs semantic graph query strings", async () => {
    const fetchMock = mockFetch(
      new Response(
        JSON.stringify({
          nodes: [],
          edges: [],
          total_nodes: 0,
          total_edges: 0,
          rendered: { nodes: 0, edges: 0 },
        }),
        {
          status: 200,
          headers: { "Content-Type": "application/json" },
        },
      ),
    );

    await getSemanticGraph(300);

    const requested = String(fetchMock.mock.calls[0]?.[0]);
    const url = new URL(requested, "http://test.invalid");
    expect(url.pathname).toBe("/api/semantic/graph");
    expect(url.searchParams.get("limit")).toBe("300");
  });

  it("fetches semantic node detail by id", async () => {
    const fetchMock = mockFetch(
      new Response(
        JSON.stringify({
          node: {
            id: "semn_detail0000000",
            kind: "entity",
            label: "Detail node",
            description: "Detail node description",
            domain: null,
            aliases: [],
            confidence: 0.8,
            status: "active",
            source_episode_ids: ["ep_source000000000"],
            source_count: 1,
            created_at: 1,
            updated_at: 2,
          },
        }),
        {
          status: 200,
          headers: { "Content-Type": "application/json" },
        },
      ),
    );

    await expect(getSemanticNode("semn_detail0000000")).resolves.toMatchObject({
      id: "semn_detail0000000",
      label: "Detail node",
    });
    expect(fetchMock.mock.calls[0]?.[0]).toBe("/api/semantic/nodes/semn_detail0000000");
  });

  it("fetches semantic edge detail by id", async () => {
    const fetchMock = mockFetch(
      new Response(
        JSON.stringify({
          edge: {
            id: "seme_detail0000000",
            from_node_id: "semn_source0000000",
            to_node_id: "semn_target0000000",
            relation: "contradicts",
            confidence: 0.7,
            evidence_episode_ids: ["ep_source000000000"],
            source_count: 1,
            valid_from: 1,
            valid_to: null,
            invalidated_at: null,
            invalidated_by_edge_id: null,
            invalidated_by_review_id: null,
            invalidated_by_process: null,
            invalidated_reason: null,
          },
        }),
        {
          status: 200,
          headers: { "Content-Type": "application/json" },
        },
      ),
    );

    await expect(getSemanticEdge("seme_detail0000000")).resolves.toMatchObject({
      id: "seme_detail0000000",
      relation: "contradicts",
    });
    expect(fetchMock.mock.calls[0]?.[0]).toBe("/api/semantic/edges/seme_detail0000000");
  });

  it("requires audience when constructing attachment byte URLs", () => {
    const url = new URL(attachmentBytesUrl("att_1111111111111111", "alice"), "http://test.invalid");

    expect(url.pathname).toBe("/api/attachments/att_1111111111111111/bytes");
    expect(url.searchParams.get("audience")).toBe("alice");
  });

  it("throws structured errors on non-2xx responses", async () => {
    mockFetch(
      new Response(JSON.stringify({ error: { status: 400, message: "kind rejected" } }), {
        status: 400,
        headers: { "Content-Type": "application/json" },
      }),
    );

    await expect(getStream({ kinds: ["user_msg"], limit: 10 })).rejects.toMatchObject({
      status: 400,
      payload: { status: 400, message: "kind rejected" },
    });
  });

  it("emits ApiError fetch failures and stops after unsubscribe", async () => {
    const events: unknown[] = [];
    const unsubscribe = subscribeClientFetchErrors((event) => events.push(event));
    unsubscribers.push(unsubscribe);
    const fetchMock = vi.fn(
      async () =>
        new Response(JSON.stringify({ error: { message: "kind rejected" } }), {
          status: 400,
          headers: { "Content-Type": "application/json" },
        }),
    );
    vi.stubGlobal("fetch", fetchMock);

    await expect(getStream({ kinds: ["user_msg"], limit: 10 })).rejects.toMatchObject({
      status: 400,
      payload: { status: 400, message: "kind rejected" },
    });

    expect(events).toHaveLength(1);
    expect(events[0]).toMatchObject({
      endpoint: "/api/stream?kind=user_msg&limit=10",
      status: 400,
      message: "kind rejected",
    });

    unsubscribe();
    unsubscribers.pop();

    await expect(getStream({ kinds: ["user_msg"], limit: 10 })).rejects.toMatchObject({
      status: 400,
    });
    expect(events).toHaveLength(1);
  });

  it("keeps the original ApiError rejection when a subscriber throws", async () => {
    unsubscribers.push(
      subscribeClientFetchErrors(() => {
        throw new Error("observer failed");
      }),
    );
    const fetchMock = vi.fn(
      async () =>
        new Response(JSON.stringify({ error: { message: "server failed" } }), {
          status: 500,
          headers: { "Content-Type": "application/json" },
        }),
    );
    vi.stubGlobal("fetch", fetchMock);

    await expect(getSessions()).rejects.toMatchObject({
      status: 500,
      payload: { status: 500, message: "server failed" },
    });
  });

  it("emits network failures without changing the rejection", async () => {
    const cause = new Error("socket closed");
    const events: unknown[] = [];
    unsubscribers.push(subscribeClientFetchErrors((event) => events.push(event)));
    vi.stubGlobal(
      "fetch",
      vi.fn(async () => {
        throw cause;
      }),
    );

    await expect(getSessions()).rejects.toBe(cause);

    expect(events).toHaveLength(1);
    expect(events[0]).toMatchObject({
      endpoint: "/api/sessions",
      message: "socket closed",
      cause,
    });
  });
});
