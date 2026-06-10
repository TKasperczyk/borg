import { fireEvent, screen, waitFor, within } from "@testing-library/react";
import { renderWithInspector } from "../../test/inspector";
import { afterEach, describe, expect, it, vi } from "vitest";

import type { CommitmentItem } from "../../api/types";
import { CommitmentsTab } from "./CommitmentsTab";

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

function clickPill(label: string): void {
  const pill = screen.getAllByText(label).find((element) => element.classList.contains("pill"));
  expect(pill).toBeDefined();
  fireEvent.click(pill!);
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
  it("filters superseded commitments separately from explicitly revoked commitments", async () => {
    const active = commitment({
      id: "cmt_active111111111",
      text: "Active commitment.",
      directive_family: "active_family",
    });
    const survivor = commitment({
      id: "cmt_survivor111111",
      text: "Replacement commitment.",
      directive_family: "replacement_family",
    });
    const superseded = commitment({
      id: "cmt_superseded1111",
      text: "Superseded predecessor.",
      state: "revoked",
      directive_family: "replacement_family",
      superseded_by_id: survivor.id,
    });
    const revoked = commitment({
      id: "cmt_revoked1111111",
      text: "Explicitly revoked commitment.",
      state: "revoked",
      directive_family: "revoked_family",
      revoked_at: 2,
      revoked_reason: "operator retired it",
    });
    const fetchMock = vi.fn((request: RequestInfo | URL, init?: RequestInit) => {
      const path = requestPath(request);
      const method = requestMethod(init);
      if (path === "/api/commitments" && method === "GET") {
        return Promise.resolve(
          jsonResponse({ commitments: [active, survivor, superseded, revoked] }),
        );
      }
      return Promise.resolve(new Response("{}", { status: 404 }));
    });
    vi.stubGlobal("fetch", fetchMock);

    renderWithInspector(<CommitmentsTab />);

    expect((await screen.findAllByText("Active commitment.")).length).toBeGreaterThan(0);
    expect(screen.queryByText("Superseded predecessor.")).not.toBeInTheDocument();

    clickPill("superseded");

    expect((await screen.findAllByText("Superseded predecessor.")).length).toBeGreaterThan(0);
    expect(screen.queryByText("Explicitly revoked commitment.")).not.toBeInTheDocument();
    expect(screen.getAllByText("superseded").length).toBeGreaterThan(1);

    clickPill("revoked");

    expect((await screen.findAllByText("Explicitly revoked commitment.")).length).toBeGreaterThan(
      0,
    );
    await waitFor(() => {
      expect(screen.queryByText("Superseded predecessor.")).not.toBeInTheDocument();
    });
    expect(screen.getAllByText("revoked").length).toBeGreaterThan(1);
  });

  it("renders a supersession chain and navigates every chip", async () => {
    const survivor = commitment({
      id: "cmt_survivor111111",
      text: "Surviving @handle commitment.",
      directive_family: "mandatory_mention_handles",
      priority: 9,
      audience: "botarena",
    });
    const middle = commitment({
      id: "cmt_middle11111111",
      text: "Middle @handle commitment.",
      state: "revoked",
      directive_family: "mention_handle_format_botarena",
      priority: 8,
      audience: "botarena",
      superseded_by_id: survivor.id,
    });
    const sibling = commitment({
      id: "cmt_sibling1111111",
      text: "Sibling @handle commitment.",
      state: "revoked",
      directive_family: "mention_handle_format_botarena",
      priority: 8,
      audience: "botarena",
      superseded_by_id: survivor.id,
    });
    const oldest = commitment({
      id: "cmt_oldest11111111",
      text: "Oldest @handle commitment.",
      state: "revoked",
      directive_family: "mention_handle_format",
      priority: 6,
      audience: "Tom",
      superseded_by_id: middle.id,
    });
    const fetchMock = vi.fn((request: RequestInfo | URL, init?: RequestInit) => {
      const path = requestPath(request);
      const method = requestMethod(init);
      if (path === "/api/commitments" && method === "GET") {
        return Promise.resolve(jsonResponse({ commitments: [survivor, middle, sibling, oldest] }));
      }
      return Promise.resolve(new Response("{}", { status: 404 }));
    });
    vi.stubGlobal("fetch", fetchMock);

    renderWithInspector(<CommitmentsTab />);

    expect((await screen.findAllByText("Surviving @handle commitment.")).length).toBeGreaterThan(0);
    const firstChain = await screen.findByLabelText("supersession chain");
    for (const target of [oldest, middle, sibling, survivor]) {
      expect(
        within(firstChain).getAllByRole("button", { name: `jump to commitment ${target.id}` })
          .length,
      ).toBeGreaterThan(0);
    }

    for (const target of [oldest, middle, sibling, survivor]) {
      const chain = screen.getByLabelText("supersession chain");
      fireEvent.click(
        within(chain).getAllByRole("button", { name: `jump to commitment ${target.id}` })[0]!,
      );
      expect(
        await screen.findByRole("button", { name: `jump to ${target.id}` }),
      ).toBeInTheDocument();
    }

    expect(screen.getAllByText("all").some((element) => element.classList.contains("on"))).toBe(
      true,
    );
  });

  it("filters commitments by directive family and groups visible rows by family", async () => {
    const alphaFirst = commitment({
      id: "cmt_alpha11111111",
      text: "Alpha first.",
      directive_family: "family_alpha",
    });
    const alphaSecond = commitment({
      id: "cmt_alpha22222222",
      text: "Alpha second.",
      directive_family: "family_alpha",
      priority: 6,
    });
    const beta = commitment({
      id: "cmt_beta111111111",
      text: "Beta only.",
      directive_family: "family_beta",
    });
    const fetchMock = vi.fn((request: RequestInfo | URL, init?: RequestInit) => {
      const path = requestPath(request);
      const method = requestMethod(init);
      if (path === "/api/commitments" && method === "GET") {
        return Promise.resolve(jsonResponse({ commitments: [alphaFirst, alphaSecond, beta] }));
      }
      return Promise.resolve(new Response("{}", { status: 404 }));
    });
    vi.stubGlobal("fetch", fetchMock);

    renderWithInspector(<CommitmentsTab />);

    expect((await screen.findAllByText("Alpha first.")).length).toBeGreaterThan(0);
    clickPill("family groups");

    expect(screen.getByRole("row", { name: "family group family_alpha" })).toBeInTheDocument();
    expect(screen.getByRole("row", { name: "family group family_beta" })).toBeInTheDocument();

    clickPill("family_alpha");

    expect((await screen.findAllByText("Alpha second.")).length).toBeGreaterThan(0);
    expect(screen.queryByText("Beta only.")).not.toBeInTheDocument();
    expect(screen.getByRole("row", { name: "family group family_alpha" })).toBeInTheDocument();
    expect(screen.queryByRole("row", { name: "family group family_beta" })).not.toBeInTheDocument();
  });

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

    renderWithInspector(<CommitmentsTab />);

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

    renderWithInspector(<CommitmentsTab />);

    expect((await screen.findAllByText("Prefer direct answers.")).length).toBeGreaterThan(0);
    const revokeButton = (await screen.findAllByRole("button", { name: "revoke" }))[0]!;
    expect(revokeButton).toHaveClass("danger");
    fireEvent.click(revokeButton);
    fireEvent.change(screen.getByLabelText("reason"), {
      target: { value: "creator changed the instruction" },
    });
    const confirmRevoke = within(screen.getByRole("dialog")).getByRole("button", {
      name: "revoke",
    });
    expect(confirmRevoke).toHaveClass("danger");
    fireEvent.click(confirmRevoke);

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

  it("queues corrections with the existing patch and reason body", async () => {
    const current = commitment();
    const fetchMock = vi.fn((request: RequestInfo | URL, init?: RequestInit) => {
      const path = requestPath(request);
      const method = requestMethod(init);
      if (path === "/api/commitments" && method === "GET") {
        return Promise.resolve(jsonResponse({ commitments: [current] }));
      }
      if (path === "/api/correction/cmt_1111111111111111/correct" && method === "POST") {
        return Promise.resolve(jsonResponse({ id: 1 }));
      }
      return Promise.resolve(new Response("{}", { status: 404 }));
    });
    vi.stubGlobal("fetch", fetchMock);

    renderWithInspector(<CommitmentsTab />);

    expect((await screen.findAllByText("Prefer direct answers.")).length).toBeGreaterThan(0);
    fireEvent.click(screen.getByRole("button", { name: "correct" }));
    fireEvent.change(screen.getByLabelText("reason"), {
      target: { value: "operator correction" },
    });
    fireEvent.change(screen.getByLabelText("json patch"), {
      target: { value: '{ "directive": "Prefer concise answers.", "priority": 4 }' },
    });
    fireEvent.click(within(screen.getByRole("dialog")).getByRole("button", { name: "queue" }));

    await waitFor(() => {
      expect(
        fetchMock.mock.calls.filter(
          (call) => requestPath(call[0]) === "/api/commitments" && requestMethod(call[1]) === "GET",
        ),
      ).toHaveLength(2);
    });

    const postCall = fetchMock.mock.calls.find(
      (call) =>
        requestPath(call[0]) === "/api/correction/cmt_1111111111111111/correct" &&
        requestMethod(call[1]) === "POST",
    );
    expect(postCall).toBeDefined();
    expect(JSON.parse(String((postCall?.[1] as RequestInit | undefined)?.body))).toEqual({
      patch: {
        directive: "Prefer concise answers.",
        priority: 4,
      },
      reason: "operator correction",
    });
  });

  it("does not expose forget for commitments", async () => {
    const fetchMock = vi.fn((request: RequestInfo | URL, init?: RequestInit) => {
      const path = requestPath(request);
      const method = requestMethod(init);
      if (path === "/api/commitments" && method === "GET") {
        return Promise.resolve(jsonResponse({ commitments: [commitment()] }));
      }
      return Promise.resolve(new Response("{}", { status: 404 }));
    });
    vi.stubGlobal("fetch", fetchMock);

    renderWithInspector(<CommitmentsTab />);

    expect((await screen.findAllByText("Prefer direct answers.")).length).toBeGreaterThan(0);
    expect(screen.queryByRole("button", { name: "forget" })).not.toBeInTheDocument();
    expect(screen.getAllByRole("button", { name: "revoke" })).toHaveLength(2);
  });
});
