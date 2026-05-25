import { afterEach, describe, expect, it, vi } from "vitest";

import { ApiError, getStream, postTurn } from "./client";

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
        headers: { "Content-Type": "application/json" }
      })
    );

    await getStream({
      audience: "alice",
      kinds: ["user_msg", "agent_msg", "user_image_attachment"],
      limit: 50
    });

    const url = new URL(String(fetchMock.mock.calls[0]?.[0]));
    expect(url.origin).toBe("http://localhost:7740");
    expect(url.pathname).toBe("/api/stream");
    expect(url.searchParams.get("audience")).toBe("alice");
    expect(url.searchParams.get("kind")).toBe("user_msg,agent_msg,user_image_attachment");
    expect(url.searchParams.get("limit")).toBe("50");
  });

  it("posts turns with message, audience, and stakes", async () => {
    const fetchMock = mockFetch(
      new Response(JSON.stringify({ ok: true, turn_id: "turn_123" }), {
        status: 200,
        headers: { "Content-Type": "application/json" }
      })
    );

    await postTurn({ message: "hello", audience: "alice", stakes: "low" });

    const init = fetchMock.mock.calls[0]?.[1] as RequestInit;
    expect(fetchMock.mock.calls[0]?.[0]).toBe("http://localhost:7740/api/turn");
    expect(init.method).toBe("POST");
    expect(JSON.parse(String(init.body))).toEqual({
      message: "hello",
      audience: "alice",
      stakes: "low"
    });
  });

  it("throws structured errors on non-2xx responses", async () => {
    mockFetch(
      new Response(JSON.stringify({ error: { status: 400, message: "kind rejected" } }), {
        status: 400,
        headers: { "Content-Type": "application/json" }
      })
    );

    await expect(getStream({ kinds: ["user_msg"], limit: 10 })).rejects.toMatchObject({
      status: 400,
      payload: { status: 400, message: "kind rejected" }
    });
  });
});
