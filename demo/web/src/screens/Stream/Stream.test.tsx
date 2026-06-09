import { act, fireEvent, screen, waitFor, within } from "@testing-library/react";
import { afterEach, describe, expect, it, vi } from "vitest";

import type { AttachmentMetadataResponse, LiveFrame, StreamEntry, WsState } from "../../api/types";
import { LiveEventsProvider } from "../../hooks/live-context";
import type { LiveEventHandler, LiveEvents } from "../../hooks/use-live-events";
import { renderWithInspector } from "../../test/inspector";
import { StreamScreen } from ".";

function jsonResponse(body: unknown): Response {
  return new Response(JSON.stringify(body), {
    status: 200,
    headers: { "Content-Type": "application/json" },
  });
}

function deferredResponse(): { promise: Promise<Response>; resolve: (response: Response) => void } {
  let resolve!: (response: Response) => void;
  const promise = new Promise<Response>((innerResolve) => {
    resolve = innerResolve;
  });
  return { promise, resolve };
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

function requestUrl(request: RequestInfo | URL): URL {
  return new URL(String(request), "http://test.invalid");
}

function streamFetch(
  firstEntries: readonly StreamEntry[],
  input: {
    nextCursor?: string | null;
    olderEntries?: readonly StreamEntry[];
    attachment?: AttachmentMetadataResponse;
  } = {},
): ReturnType<typeof vi.fn> {
  const fetchMock = vi.fn((request: RequestInfo | URL) => {
    const url = requestUrl(request);

    if (url.pathname === "/api/stream") {
      if (url.searchParams.get("before") !== null) {
        return Promise.resolve(
          jsonResponse({ entries: input.olderEntries ?? [], next_cursor: null }),
        );
      }
      return Promise.resolve(
        jsonResponse({ entries: firstEntries, next_cursor: input.nextCursor ?? null }),
      );
    }

    if (url.pathname === "/api/attachments") {
      return Promise.resolve(
        jsonResponse(
          input.attachment === undefined
            ? []
            : [
                {
                  id: input.attachment.attachment.attachment_id,
                  status: input.attachment.status,
                },
              ],
        ),
      );
    }

    if (input.attachment !== undefined) {
      const attachmentId = input.attachment.attachment.attachment_id;
      if (url.pathname === `/api/attachments/${attachmentId}`) {
        return Promise.resolve(jsonResponse(input.attachment));
      }
    }

    if (url.pathname.endsWith("/bytes")) {
      return Promise.resolve(new Response("", { status: 404 }));
    }

    return Promise.resolve(new Response("not found", { status: 404 }));
  });
  vi.stubGlobal("fetch", fetchMock);
  return fetchMock;
}

afterEach(() => {
  vi.unstubAllGlobals();
});

describe("Stream & Provenance", () => {
  it("renders grouped turn rows and the unclaimed maintenance lane", async () => {
    const live = makeLiveSource();
    streamFetch([
      streamEntry({
        id: "strm_agent",
        kind: "agent_msg",
        content: "agent reply",
        timestamp: 3,
        turn_id: "turn_a",
      }),
      streamEntry({
        id: "strm_maintenance",
        kind: "internal_event",
        content: { event: "index_backfill" },
        timestamp: 2,
        turn_id: undefined,
      }),
    ]);

    renderWithInspector(
      <LiveEventsProvider value={live.live()}>
        <StreamScreen sessionId="default" />
      </LiveEventsProvider>,
    );

    expect(await screen.findByRole("button", { name: "jump to turn_a" })).toBeInTheDocument();
    expect(screen.getByText("unclaimed / maintenance")).toBeInTheDocument();
    expect(screen.getAllByText("1 entries").length).toBeGreaterThanOrEqual(2);
  });

  it("keeps a collapsed group collapsed when a live append extends it", async () => {
    const live = makeLiveSource();
    streamFetch([
      streamEntry({
        id: "strm_first",
        kind: "user_msg",
        content: "first turn text",
        timestamp: 2,
        turn_id: "turn_live",
      }),
    ]);

    const { container } = renderWithInspector(
      <LiveEventsProvider value={live.live()}>
        <StreamScreen sessionId="default" />
      </LiveEventsProvider>,
    );
    const main = container.querySelector(".stream-main") as HTMLElement;

    fireEvent.click(await screen.findByRole("button", { name: "collapse turn_live" }));
    expect(within(main).queryByText("first turn text")).not.toBeInTheDocument();

    act(() => {
      live.emit({
        type: "stream:append",
        ts: 3,
        entries: [
          streamEntry({
            id: "strm_live_append",
            kind: "agent_msg",
            content: "live append text",
            timestamp: 3,
            turn_id: "turn_live",
          }),
        ],
      });
    });

    await waitFor(() => {
      expect(screen.getByText("2 entries")).toBeInTheDocument();
    });
    expect(within(main).queryByText("live append text")).not.toBeInTheDocument();
    expect(screen.getByRole("button", { name: "expand turn_live" })).toBeInTheDocument();
  });

  it("refetches with a kinds query when a kind chip changes", async () => {
    const live = makeLiveSource();
    const fetchMock = streamFetch([
      streamEntry({ id: "strm_user", kind: "user_msg", content: "hello", timestamp: 2 }),
      streamEntry({ id: "strm_agent", kind: "agent_msg", content: "reply", timestamp: 1 }),
    ]);

    renderWithInspector(
      <LiveEventsProvider value={live.live()}>
        <StreamScreen sessionId="default" />
      </LiveEventsProvider>,
    );

    fireEvent.click(await screen.findByRole("button", { name: /agent_msg/ }));

    await waitFor(() => {
      expect(
        fetchMock.mock.calls.filter((call) => requestUrl(call[0]).pathname === "/api/stream"),
      ).toHaveLength(2);
    });
    const secondStreamRequest = fetchMock.mock.calls
      .map((call) => requestUrl(call[0]))
      .filter((url) => url.pathname === "/api/stream")[1];
    const kinds = secondStreamRequest?.searchParams.get("kind")?.split(",") ?? [];
    expect(kinds).toContain("user_msg");
    expect(kinds).not.toContain("agent_msg");
  });

  it("applies structural filters to the loaded window and labels older availability honestly", async () => {
    const live = makeLiveSource();
    streamFetch(
      [
        streamEntry({
          id: "strm_active",
          kind: "agent_msg",
          content: "active row text",
          timestamp: 3,
        }),
        streamEntry({
          id: "strm_aborted",
          kind: "agent_suppressed",
          content: { reason: "finalizer_no_output" },
          timestamp: 2,
          turn_status: "aborted",
        }),
      ],
      { nextCursor: "cursor_older" },
    );

    renderWithInspector(
      <LiveEventsProvider value={live.live()}>
        <StreamScreen sessionId="default" />
      </LiveEventsProvider>,
    );

    expect(await screen.findAllByText("loaded window only · older entries available")).toHaveLength(
      2,
    );
    fireEvent.click(screen.getByRole("button", { name: /aborted-only/ }));

    await waitFor(() => {
      expect(screen.queryByText("active row text")).not.toBeInTheDocument();
    });
    expect(screen.getByText("deliberate silence")).toBeInTheDocument();
    expect(screen.getByText("1 window events")).toBeInTheDocument();
  });

  it("loads older pages into the current window", async () => {
    const live = makeLiveSource();
    const fetchMock = streamFetch(
      [streamEntry({ id: "strm_new", kind: "agent_msg", content: "new row", timestamp: 10 })],
      {
        nextCursor: "cursor_old",
        olderEntries: [
          streamEntry({ id: "strm_old", kind: "user_msg", content: "older row", timestamp: 1 }),
        ],
      },
    );

    renderWithInspector(
      <LiveEventsProvider value={live.live()}>
        <StreamScreen sessionId="default" />
      </LiveEventsProvider>,
    );

    fireEvent.click(await screen.findByRole("button", { name: "load older" }));

    expect(await screen.findByText("older row")).toBeInTheDocument();
    const olderRequest = fetchMock.mock.calls
      .map((call) => requestUrl(call[0]))
      .find((url) => url.searchParams.get("before") === "cursor_old");
    expect(olderRequest).toBeDefined();
  });

  it("discards a stale older page when server filters change in flight", async () => {
    const live = makeLiveSource();
    const older = deferredResponse();
    const fetchMock = vi.fn((request: RequestInfo | URL) => {
      const url = requestUrl(request);

      if (url.pathname === "/api/stream") {
        if (url.searchParams.get("before") === "cursor_old") {
          return older.promise;
        }

        const kinds = url.searchParams.get("kind")?.split(",") ?? null;
        if (kinds !== null && !kinds.includes("agent_msg")) {
          return Promise.resolve(
            jsonResponse({
              entries: [
                streamEntry({
                  id: "strm_refetched",
                  kind: "user_msg",
                  content: "refetched after filter",
                  timestamp: 8,
                }),
              ],
              next_cursor: null,
            }),
          );
        }

        return Promise.resolve(
          jsonResponse({
            entries: [
              streamEntry({
                id: "strm_new",
                kind: "agent_msg",
                content: "new row",
                timestamp: 10,
              }),
            ],
            next_cursor: "cursor_old",
          }),
        );
      }

      if (url.pathname === "/api/attachments") {
        return Promise.resolve(jsonResponse([]));
      }

      return Promise.resolve(new Response("not found", { status: 404 }));
    });
    vi.stubGlobal("fetch", fetchMock);

    renderWithInspector(
      <LiveEventsProvider value={live.live()}>
        <StreamScreen sessionId="default" />
      </LiveEventsProvider>,
    );

    fireEvent.click(await screen.findByRole("button", { name: "load older" }));
    fireEvent.click(screen.getByRole("button", { name: /agent_msg/ }));
    expect(await screen.findByText("refetched after filter")).toBeInTheDocument();

    await act(async () => {
      older.resolve(
        jsonResponse({
          entries: [
            streamEntry({
              id: "strm_stale_old",
              kind: "user_msg",
              content: "stale older row",
              timestamp: 1,
            }),
          ],
          next_cursor: null,
        }),
      );
      await older.promise;
    });

    await waitFor(() => {
      expect(screen.queryByText("stale older row")).not.toBeInTheDocument();
    });
  });

  it("invalidates attachment status from same-session live entries excluded by kind filters", async () => {
    const live = makeLiveSource();
    let quarantined = false;
    const attachment: AttachmentMetadataResponse = {
      attachment: {
        attachment_id: "att_status",
        sha256: "statushash",
        media_type: "image/png",
        byte_size: 20,
        width: 10,
        height: 10,
        storage_ref: "store",
        thumbnail_ref: null,
        perception_id: null,
        text_embedding_ref: null,
        visual_embedding_ref: null,
        active: true,
        audience: "alice",
        created_turn_global: null,
        parent_entry_id: "strm_status",
        stream_entry_id: "strm_status",
        parent_turn_id: "turn_1",
        created_at: 1,
      },
      perception: null,
      status: { active: true, quarantined: false, stream_active: true, parent_active: true },
    };
    const fetchMock = vi.fn((request: RequestInfo | URL) => {
      const url = requestUrl(request);

      if (url.pathname === "/api/stream") {
        return Promise.resolve(
          jsonResponse({
            entries: [
              streamEntry({
                id: "strm_status",
                kind: "user_image_attachment",
                content: { attachment_id: "att_status", media_type: "image/png" },
                timestamp: 5,
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
              id: "att_status",
              status: {
                active: !quarantined,
                quarantined,
                stream_active: !quarantined,
                parent_active: true,
              },
            },
          ]),
        );
      }

      if (url.pathname === "/api/attachments/att_status") {
        return Promise.resolve(jsonResponse(attachment));
      }

      if (url.pathname.endsWith("/bytes")) {
        return Promise.resolve(new Response("", { status: 404 }));
      }

      return Promise.resolve(new Response("not found", { status: 404 }));
    });
    vi.stubGlobal("fetch", fetchMock);

    renderWithInspector(
      <LiveEventsProvider value={live.live()}>
        <StreamScreen sessionId="default" />
      </LiveEventsProvider>,
    );

    expect(await screen.findByText("att_status · image/png")).toBeInTheDocument();
    fireEvent.click(screen.getByRole("button", { name: /internal_event/ }));
    await waitFor(() => {
      expect(screen.queryByText("quarantined")).not.toBeInTheDocument();
    });

    quarantined = true;
    act(() => {
      live.emit({
        type: "stream:append",
        ts: 6,
        entries: [
          streamEntry({
            id: "strm_status_event",
            kind: "internal_event",
            content: { attachment_id: "att_status", event: "attachment_quarantine" },
            timestamp: 6,
          }),
        ],
      });
    });

    await waitFor(() => {
      expect(screen.getAllByText("quarantined").length).toBeGreaterThan(0);
    });
  });

  it("clears selection when the selected entry is filtered out", async () => {
    const live = makeLiveSource();
    streamFetch([
      streamEntry({ id: "strm_plain", kind: "agent_msg", content: "plain selected", timestamp: 3 }),
      streamEntry({
        id: "strm_att",
        kind: "user_image_attachment",
        content: { attachment_id: "att_filter", media_type: "image/png" },
        timestamp: 2,
      }),
    ]);
    const { container } = renderWithInspector(
      <LiveEventsProvider value={live.live()}>
        <StreamScreen sessionId="default" />
      </LiveEventsProvider>,
    );

    expect(await screen.findByText("plain selected")).toBeInTheDocument();
    expect(container.querySelector(".stream-row.selected")).not.toBeNull();

    fireEvent.click(screen.getByRole("button", { name: /has attachment/ }));

    await waitFor(() => {
      expect(container.querySelector(".stream-row.selected")).toBeNull();
    });
    expect(screen.getByText("select a stream entry")).toBeInTheDocument();
  });

  it("uses entry index ordering for the loaded window selection", async () => {
    const live = makeLiveSource();
    streamFetch([
      streamEntry({
        id: "strm_newer_timestamp",
        kind: "agent_msg",
        content: "newer timestamp lower index",
        timestamp: 99,
        entry_index: 1,
      }),
      streamEntry({
        id: "strm_higher_index",
        kind: "user_msg",
        content: "higher index selected",
        timestamp: 10,
        entry_index: 2,
      }),
    ]);
    const { container } = renderWithInspector(
      <LiveEventsProvider value={live.live()}>
        <StreamScreen sessionId="default" />
      </LiveEventsProvider>,
    );

    expect(await screen.findByText("higher index selected")).toBeInTheDocument();
    expect(container.querySelector(".stream-row.selected")).toHaveTextContent(
      "higher index selected",
    );
  });

  it("renders attachment status booleans, metadata, and perception fields", async () => {
    const live = makeLiveSource();
    const attachment: AttachmentMetadataResponse = {
      attachment: {
        attachment_id: "att_full",
        sha256: "abcdef0123456789abcdef0123456789abcdef0123456789abcdef0123456789",
        media_type: "image/png",
        byte_size: 4096,
        width: 640,
        height: 480,
        storage_ref: "store",
        thumbnail_ref: null,
        perception_id: "imgp_full",
        text_embedding_ref: null,
        visual_embedding_ref: null,
        active: false,
        audience: null,
        created_turn_global: null,
        parent_entry_id: "strm_parent",
        stream_entry_id: "strm_att_full",
        parent_turn_id: "turn_parent",
        created_at: 5,
      },
      perception: {
        perception_id: "imgp_full",
        payload_id: "payload_1",
        attachment_id: "att_full",
        caption: "whiteboard deployment sketch",
        image_kind: "whiteboard",
        active: true,
        audience: null,
        visible_text: ["deploy diagram"],
        objects: ["terminal screenshot"],
        people_or_roles: ["operator"],
        scene: "release planning room",
        colors_and_visual_attributes: ["blue accents"],
        spatial_relationships: ["left of panel"],
        possible_user_relevant_details: ["contains release gate"],
        search_terms: ["release gate"],
        uncertainties: ["maybe old UI"],
        embedding_status: "complete",
      },
      status: { active: false, quarantined: true, stream_active: false, parent_active: true },
    };
    streamFetch(
      [
        streamEntry({
          id: "strm_att_full",
          kind: "user_image_attachment",
          content: { attachment_id: "att_full", media_type: "image/png" },
          timestamp: 4,
          audience: undefined,
        }),
      ],
      { attachment },
    );

    const { container } = renderWithInspector(
      <LiveEventsProvider value={live.live()}>
        <StreamScreen sessionId="default" />
      </LiveEventsProvider>,
    );

    expect(await screen.findByText("whiteboard deployment sketch")).toBeInTheDocument();
    expect(screen.getByText("preview unavailable")).toBeInTheDocument();
    expect(screen.getByText("no audience")).toBeInTheDocument();
    expect(screen.getByText("active false")).toBeInTheDocument();
    expect(screen.getAllByText("quarantined true").length).toBeGreaterThan(0);
    expect(screen.getByText("stream_active false")).toBeInTheDocument();
    expect(screen.getByText("parent_active true")).toBeInTheDocument();
    expect(screen.getByText("4.0 KB")).toBeInTheDocument();
    expect(
      screen.getByText("abcdef0123456789abcdef0123456789abcdef0123456789abcdef0123456789"),
    ).toBeInTheDocument();
    expect(screen.getByRole("button", { name: "jump to strm_parent" })).toBeInTheDocument();
    expect(screen.getByRole("button", { name: "jump to turn_parent" })).toBeInTheDocument();

    const detail = container.querySelector(".stream-detail .det-body");
    expect(detail).not.toBeNull();
    const detailView = within(detail as HTMLElement);
    expect(detailView.getByText("payload_1")).toBeInTheDocument();
    expect(detailView.getByText("whiteboard")).toBeInTheDocument();
    expect(detailView.getByText("complete")).toBeInTheDocument();
    expect(detailView.getByText("deploy diagram")).toBeInTheDocument();
    expect(detailView.getByText("terminal screenshot")).toBeInTheDocument();
    expect(detailView.getByText("operator")).toBeInTheDocument();
    expect(detailView.getByText("release planning room")).toBeInTheDocument();
    expect(detailView.getByText("blue accents")).toBeInTheDocument();
    expect(detailView.getByText("left of panel")).toBeInTheDocument();
    expect(detailView.getByText("contains release gate")).toBeInTheDocument();
    expect(detailView.getByText("release gate")).toBeInTheDocument();
    expect(detailView.getByText("maybe old UI")).toBeInTheDocument();
  });

  it("treats null attachment parent ids as absent", async () => {
    const live = makeLiveSource();
    const attachment: AttachmentMetadataResponse = {
      attachment: {
        attachment_id: "att_null_parent",
        sha256: "nullparenthash",
        media_type: "image/png",
        byte_size: 16,
        width: 4,
        height: 4,
        storage_ref: "store",
        thumbnail_ref: null,
        perception_id: null,
        text_embedding_ref: null,
        visual_embedding_ref: null,
        active: true,
        audience: "alice",
        created_turn_global: null,
        parent_entry_id: null,
        stream_entry_id: "strm_null_parent",
        parent_turn_id: null,
        created_at: 8,
      },
      perception: null,
      status: { active: true, quarantined: false, stream_active: true, parent_active: true },
    };
    streamFetch(
      [
        streamEntry({
          id: "strm_null_parent",
          kind: "user_image_attachment",
          content: { attachment_id: "att_null_parent", media_type: "image/png" },
          timestamp: 7,
        }),
      ],
      { attachment },
    );

    renderWithInspector(
      <LiveEventsProvider value={live.live()}>
        <StreamScreen sessionId="default" />
      </LiveEventsProvider>,
    );

    expect(await screen.findByText("att_null_parent")).toBeInTheDocument();
    expect(screen.queryByRole("button", { name: "jump to null" })).not.toBeInTheDocument();
    expect(screen.queryByRole("button", { name: "unknown object null" })).not.toBeInTheDocument();
  });
});
