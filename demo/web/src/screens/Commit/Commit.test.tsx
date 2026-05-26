import { fireEvent, render, screen, waitFor, within } from "@testing-library/react";
import { afterEach, describe, expect, it, vi } from "vitest";

import type { CommitmentItem } from "../../api/types";
import { CommitScreen } from ".";

function jsonResponse(body: unknown): Response {
  return new Response(JSON.stringify(body), {
    status: 200,
    headers: { "Content-Type": "application/json" },
  });
}

function requestPath(request: RequestInfo | URL): string {
  return new URL(String(request), "http://test.invalid").pathname;
}

function requestMethod(init?: RequestInit): string {
  return init?.method ?? "GET";
}

function commitment(input: Partial<CommitmentItem> = {}): CommitmentItem {
  return {
    id: "cmt_1111111111111111",
    text: "Prefer direct answers.",
    type: "rule",
    kind: "process_norm",
    enforcement_class: "advisory",
    critical_domain: null,
    state: "active",
    priority: 5,
    directive_family: "creator_guidance",
    audience: "Alice",
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
    ...input,
  };
}

afterEach(() => {
  vi.unstubAllGlobals();
});

describe("CommitScreen operator actions", () => {
  it("opens the add commitment modal, submits, and refetches", async () => {
    let commitments: CommitmentItem[] = [];
    const created = commitment({
      text: "Prefer direct answers when speaking with Alice.",
      priority: 7,
      made_to: "Tom",
      about: "Project Atlas",
      expires_at: new Date("2030-01-02T00:00:00").getTime(),
    });
    const fetchMock = vi.fn((request: RequestInfo | URL, init?: RequestInit) => {
      const path = requestPath(request);
      const method = requestMethod(init);
      if (path === "/api/commitments" && method === "GET") {
        return Promise.resolve(jsonResponse({ commitments }));
      }
      if (path === "/api/commitments" && method === "POST") {
        commitments = [created];
        return Promise.resolve(jsonResponse(created));
      }
      return Promise.resolve(new Response("{}", { status: 404 }));
    });
    vi.stubGlobal("fetch", fetchMock);

    render(<CommitScreen />);

    fireEvent.click(await screen.findByLabelText("add commitment"));
    expect(screen.getByText("marked as creator-authored advice")).toBeInTheDocument();
    expect(screen.queryByLabelText(/enforcement/i)).not.toBeInTheDocument();
    fireEvent.change(screen.getByLabelText(/^directive\s+\*$/i), {
      target: { value: "Prefer direct answers when speaking with Alice." },
    });
    fireEvent.change(screen.getByLabelText(/priority/i), { target: { value: "7" } });
    fireEvent.change(screen.getByLabelText(/audience:/i), { target: { value: "Alice" } });
    fireEvent.change(screen.getByLabelText(/made_to:/i), { target: { value: "Tom" } });
    fireEvent.change(screen.getByLabelText(/about:/i), { target: { value: "Project Atlas" } });
    fireEvent.change(screen.getByLabelText(/directive_family/i), {
      target: { value: "creator_guidance" },
    });
    fireEvent.change(screen.getByLabelText(/expires_at/i), { target: { value: "2030-01-02" } });
    fireEvent.click(screen.getByRole("button", { name: "save" }));

    await waitFor(() => {
      expect(
        fetchMock.mock.calls.filter(
          (call) => requestPath(call[0]) === "/api/commitments" && requestMethod(call[1]) === "GET",
        ),
      ).toHaveLength(2);
    });

    const postCall = fetchMock.mock.calls.find(
      (call) => requestPath(call[0]) === "/api/commitments" && requestMethod(call[1]) === "POST",
    );
    expect(postCall).toBeDefined();
    expect(JSON.parse(String((postCall?.[1] as RequestInit | undefined)?.body))).toEqual({
      type: "rule",
      kind: "process_norm",
      directive: "Prefer direct answers when speaking with Alice.",
      priority: 7,
      audience: "Alice",
      made_to: "Tom",
      about: "Project Atlas",
      directive_family: "creator_guidance",
      expires_at: new Date("2030-01-02T00:00:00").getTime(),
    });
  });

  it("confirms revoke, posts the reason, and refetches", async () => {
    let commitments = [commitment()];
    const fetchMock = vi.fn((request: RequestInfo | URL, init?: RequestInit) => {
      const path = requestPath(request);
      const method = requestMethod(init);
      if (path === "/api/commitments" && method === "GET") {
        return Promise.resolve(jsonResponse({ commitments }));
      }
      if (path === "/api/commitments/cmt_1111111111111111/revoke" && method === "POST") {
        commitments = [
          commitment({
            state: "revoked",
            revoked_at: 2,
            revoked_reason: "creator changed the instruction",
          }),
        ];
        return Promise.resolve(jsonResponse(commitments[0]));
      }
      return Promise.resolve(new Response("{}", { status: 404 }));
    });
    vi.stubGlobal("fetch", fetchMock);

    render(<CommitScreen />);

    expect((await screen.findAllByText("Prefer direct answers.")).length).toBeGreaterThan(0);
    fireEvent.click((await screen.findAllByRole("button", { name: "revoke" }))[0]!);
    fireEvent.change(screen.getByLabelText("reason"), {
      target: { value: "creator changed the instruction" },
    });
    fireEvent.click(within(screen.getByRole("dialog")).getByRole("button", { name: "revoke" }));

    await waitFor(() => {
      expect(
        fetchMock.mock.calls.filter(
          (call) => requestPath(call[0]) === "/api/commitments" && requestMethod(call[1]) === "GET",
        ),
      ).toHaveLength(2);
    });

    const postCall = fetchMock.mock.calls.find(
      (call) =>
        requestPath(call[0]) === "/api/commitments/cmt_1111111111111111/revoke" &&
        requestMethod(call[1]) === "POST",
    );
    expect(postCall).toBeDefined();
    expect(JSON.parse(String((postCall?.[1] as RequestInit | undefined)?.body))).toEqual({
      reason: "creator changed the instruction",
    });
  });
});
