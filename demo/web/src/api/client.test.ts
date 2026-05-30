import { afterEach, describe, expect, it, vi } from "vitest";

import {
  ApiError,
  attachmentBytesUrl,
  getSemanticGraph,
  getSessions,
  getStream,
  postTurn,
} from "./client";

afterEach(() => {
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
      new Response(
        JSON.stringify({ ok: true, status: "enqueued", stream_entry_id: "strm_123" }),
        {
          status: 200,
          headers: { "Content-Type": "application/json" },
        },
      ),
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
});
