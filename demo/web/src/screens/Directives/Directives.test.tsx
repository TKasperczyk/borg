import { fireEvent, render, screen, waitFor, within } from "@testing-library/react";
import { afterEach, describe, expect, it, vi } from "vitest";

import type { CreatorDirectiveItem } from "../../api/types";
import { DirectivesScreen } from ".";

function jsonResponse(body: unknown): Response {
  return new Response(JSON.stringify(body), {
    status: 200,
    headers: { "Content-Type": "application/json" },
  });
}

function requestPath(request: RequestInfo | URL): string {
  return new URL(String(request), "http://test.invalid").pathname;
}

function requestStatus(request: RequestInfo | URL): string {
  return new URL(String(request), "http://test.invalid").searchParams.get("status") ?? "active";
}

function requestMethod(init?: RequestInit): string {
  return init?.method ?? "GET";
}

function clickPill(label: string): void {
  const pill = screen.getAllByText(label).find((element) => element.classList.contains("pill"));
  expect(pill).toBeDefined();
  fireEvent.click(pill!);
}

function directive(input: Partial<CreatorDirectiveItem> = {}): CreatorDirectiveItem {
  return {
    id: "cdir_1111111111111111",
    kind: "subject_fact",
    text: "Alice is the launch reviewer.",
    canonical_fact: "Alice is the launch reviewer.",
    operational_directive: null,
    activation_scope: "same_as_disclosure",
    activation_allowed_entity_ids: [],
    activation_excluded_entity_ids: [],
    content_scope: "public",
    mention_policy: "only_if_topic_raised",
    status: "active",
    subject_kind: "entity",
    subject_entity_id: "ent_1111111111111111",
    subject_entity_name: "Alice",
    priority: 6,
    superseded_by_id: null,
    revoked_reason: null,
    created_at: 1,
    updated_at: 1,
    ...input,
  };
}

afterEach(() => {
  vi.unstubAllGlobals();
});

describe("DirectivesScreen", () => {
  it("defaults to active creator directives", async () => {
    const active = directive();
    const revoked = directive({
      id: "cdir_2222222222222222",
      text: "Retired directive.",
      status: "revoked",
      revoked_reason: "obsolete",
      updated_at: 2,
    });
    const fetchMock = vi.fn((request: RequestInfo | URL, init?: RequestInit) => {
      const path = requestPath(request);
      const method = requestMethod(init);
      if (path === "/api/creator-directives" && method === "GET") {
        expect(requestStatus(request)).toBe("active");
        return Promise.resolve(jsonResponse({ directives: [active] }));
      }
      return Promise.resolve(new Response("{}", { status: 404 }));
    });
    vi.stubGlobal("fetch", fetchMock);

    render(<DirectivesScreen />);

    expect((await screen.findAllByText("Alice is the launch reviewer.")).length).toBeGreaterThan(0);
    expect(screen.queryByText("Retired directive.")).not.toBeInTheDocument();
    expect(screen.getAllByText("subject_fact").length).toBeGreaterThan(0);
    expect(screen.getAllByText("same_as_disclosure").length).toBeGreaterThan(0);
    expect(screen.getAllByText("public").length).toBeGreaterThan(0);
    expect(screen.getAllByText("only_if_topic_raised").length).toBeGreaterThan(0);
    expect(screen.getAllByText("active").length).toBeGreaterThan(0);
    expect(screen.getByText("Alice")).toBeInTheDocument();
    expect(screen.queryByLabelText("add directive")).not.toBeInTheDocument();
    expect(screen.queryByRole("button", { name: "forget" })).not.toBeInTheDocument();
    expect(screen.queryByRole("button", { name: "correct" })).not.toBeInTheDocument();
    expect(revoked.status).toBe("revoked");

    await waitFor(() => {
      expect(
        fetchMock.mock.calls.filter(
          (call) =>
            requestPath(call[0]) === "/api/creator-directives" && requestMethod(call[1]) === "GET",
        ),
      ).toHaveLength(1);
    });
  });

  it("can show all and revoked history rows", async () => {
    const active = directive({ id: "cdir_active11111111", text: "Active directive." });
    const replacement = directive({
      id: "cdir_replace111111",
      text: "Replacement directive.",
      priority: 9,
    });
    const revoked = directive({
      id: "cdir_revoked111111",
      text: "Revoked directive.",
      status: "revoked",
      revoked_reason: "replaced by newer policy",
      updated_at: 2,
    });
    const superseded = directive({
      id: "cdir_superseded111",
      text: "Superseded directive.",
      status: "superseded",
      superseded_by_id: replacement.id,
      updated_at: 3,
    });
    const fetchMock = vi.fn((request: RequestInfo | URL, init?: RequestInit) => {
      const path = requestPath(request);
      const method = requestMethod(init);
      if (path === "/api/creator-directives" && method === "GET") {
        const status = requestStatus(request);
        const directives =
          status === "all"
            ? [active, replacement, revoked, superseded]
            : status === "revoked"
              ? [revoked]
              : [active, replacement];
        return Promise.resolve(jsonResponse({ directives }));
      }
      return Promise.resolve(new Response("{}", { status: 404 }));
    });
    vi.stubGlobal("fetch", fetchMock);

    render(<DirectivesScreen />);

    expect(await screen.findByText("Active directive.")).toBeInTheDocument();
    expect(screen.queryByText("Revoked directive.")).not.toBeInTheDocument();

    clickPill("all");

    expect(await screen.findByText("Revoked directive.")).toBeInTheDocument();
    expect(screen.getByText("revoked: replaced by newer policy")).toBeInTheDocument();
    expect(screen.getByText(/superseded by:/)).toBeInTheDocument();

    clickPill("revoked");

    expect(await screen.findByText("Revoked directive.")).toBeInTheDocument();
    await waitFor(() => {
      expect(screen.queryByText("Active directive.")).not.toBeInTheDocument();
    });

    await waitFor(() => {
      expect(
        fetchMock.mock.calls.some(
          (call) =>
            requestPath(call[0]) === "/api/creator-directives" &&
            requestMethod(call[1]) === "GET" &&
            requestStatus(call[0]) === "revoked",
        ),
      ).toBe(true);
    });
  });

  it("jumps from a superseded row to its target directive", async () => {
    const replacement = directive({
      id: "cdir_replace111111",
      text: "Replacement directive.",
      priority: 9,
    });
    const superseded = directive({
      id: "cdir_superseded111",
      text: "Superseded directive.",
      status: "superseded",
      superseded_by_id: replacement.id,
      updated_at: 3,
    });
    const fetchMock = vi.fn((request: RequestInfo | URL, init?: RequestInit) => {
      const path = requestPath(request);
      const method = requestMethod(init);
      if (path === "/api/creator-directives" && method === "GET") {
        const status = requestStatus(request);
        const directives =
          status === "superseded"
            ? [superseded]
            : status === "all"
              ? [replacement, superseded]
              : [replacement];
        return Promise.resolve(jsonResponse({ directives }));
      }
      return Promise.resolve(new Response("{}", { status: 404 }));
    });
    vi.stubGlobal("fetch", fetchMock);

    render(<DirectivesScreen />);

    expect((await screen.findAllByText("Replacement directive.")).length).toBeGreaterThan(0);
    clickPill("superseded");

    expect((await screen.findAllByText("Superseded directive.")).length).toBeGreaterThan(0);
    fireEvent.click(
      screen.getAllByRole("button", { name: `jump to directive ${replacement.id}` })[0]!,
    );

    await waitFor(() => {
      expect(
        fetchMock.mock.calls.some(
          (call) =>
            requestPath(call[0]) === "/api/creator-directives" &&
            requestMethod(call[1]) === "GET" &&
            requestStatus(call[0]) === "all",
        ),
      ).toBe(true);
    });
    expect(await screen.findByText(replacement.id)).toBeInTheDocument();
    expect(
      screen.getAllByText("all").some((element) => element.classList.contains("on")),
    ).toBe(true);
  });

  it("confirms revoke, posts the reason, and refetches", async () => {
    const active = directive({
      id: "cdir_revoke1111111",
      text: "Directive to revoke.",
    });
    let directives = [active];
    const fetchMock = vi.fn((request: RequestInfo | URL, init?: RequestInit) => {
      const path = requestPath(request);
      const method = requestMethod(init);
      if (path === "/api/creator-directives" && method === "GET") {
        return Promise.resolve(jsonResponse({ directives }));
      }
      if (path === "/api/creator-directives/cdir_revoke1111111/revoke" && method === "POST") {
        directives = [];
        return Promise.resolve(
          jsonResponse({
            ...active,
            status: "revoked",
            revoked_reason: "creator retired obsolete guidance",
            updated_at: 2,
          }),
        );
      }
      return Promise.resolve(new Response("{}", { status: 404 }));
    });
    vi.stubGlobal("fetch", fetchMock);

    render(<DirectivesScreen />);

    expect((await screen.findAllByText("Directive to revoke.")).length).toBeGreaterThan(0);
    fireEvent.click(screen.getAllByRole("button", { name: "revoke" })[0]!);
    expect(within(screen.getByRole("dialog")).getByRole("button", { name: "revoke" })).toBeDisabled();
    fireEvent.change(screen.getByLabelText("reason"), {
      target: { value: "creator retired obsolete guidance" },
    });
    fireEvent.click(within(screen.getByRole("dialog")).getByRole("button", { name: "revoke" }));

    await waitFor(() => {
      expect(
        fetchMock.mock.calls.filter(
          (call) =>
            requestPath(call[0]) === "/api/creator-directives" && requestMethod(call[1]) === "GET",
        ),
      ).toHaveLength(2);
    });

    const postCall = fetchMock.mock.calls.find(
      (call) =>
        requestPath(call[0]) === "/api/creator-directives/cdir_revoke1111111/revoke" &&
        requestMethod(call[1]) === "POST",
    );
    expect(postCall).toBeDefined();
    expect(JSON.parse(String(postCall?.[1]?.body))).toEqual({
      reason: "creator retired obsolete guidance",
    });
  });

  it("confirms supersede, posts the replacement id, and refetches", async () => {
    const target = directive({
      id: "cdir_target1111111",
      text: "Directive to supersede.",
      priority: 9,
    });
    const replacement = directive({
      id: "cdir_replace111111",
      text: "Replacement directive.",
      priority: 8,
    });
    let directives = [target, replacement];
    const fetchMock = vi.fn((request: RequestInfo | URL, init?: RequestInit) => {
      const path = requestPath(request);
      const method = requestMethod(init);
      if (path === "/api/creator-directives" && method === "GET") {
        return Promise.resolve(jsonResponse({ directives }));
      }
      if (path === "/api/creator-directives/cdir_target1111111/supersede" && method === "POST") {
        directives = [replacement];
        return Promise.resolve(
          jsonResponse({
            ...target,
            status: "superseded",
            superseded_by_id: replacement.id,
            updated_at: 2,
          }),
        );
      }
      return Promise.resolve(new Response("{}", { status: 404 }));
    });
    vi.stubGlobal("fetch", fetchMock);

    render(<DirectivesScreen />);

    expect((await screen.findAllByText("Directive to supersede.")).length).toBeGreaterThan(0);
    fireEvent.click(screen.getAllByRole("button", { name: "supersede" })[0]!);
    fireEvent.change(screen.getByLabelText("replacement"), {
      target: { value: replacement.id },
    });
    fireEvent.click(
      within(screen.getByRole("dialog")).getByRole("button", { name: "supersede" }),
    );

    await waitFor(() => {
      expect(
        fetchMock.mock.calls.filter(
          (call) =>
            requestPath(call[0]) === "/api/creator-directives" && requestMethod(call[1]) === "GET",
        ),
      ).toHaveLength(2);
    });

    const postCall = fetchMock.mock.calls.find(
      (call) =>
        requestPath(call[0]) === "/api/creator-directives/cdir_target1111111/supersede" &&
        requestMethod(call[1]) === "POST",
    );
    expect(postCall).toBeDefined();
    expect(JSON.parse(String(postCall?.[1]?.body))).toEqual({
      replacement_id: replacement.id,
    });
  });

  it("does not offer revoke for non-active directives", async () => {
    const revoked = directive({
      id: "cdir_revoked111111",
      text: "Revoked directive.",
      status: "revoked",
      revoked_reason: "old",
      updated_at: 2,
    });
    const fetchMock = vi.fn((request: RequestInfo | URL, init?: RequestInit) => {
      const path = requestPath(request);
      const method = requestMethod(init);
      if (path === "/api/creator-directives" && method === "GET") {
        return Promise.resolve(
          jsonResponse({ directives: requestStatus(request) === "all" ? [revoked] : [] }),
        );
      }
      return Promise.resolve(new Response("{}", { status: 404 }));
    });
    vi.stubGlobal("fetch", fetchMock);

    render(<DirectivesScreen />);

    expect(await screen.findByText("no creator directives in filter")).toBeInTheDocument();
    clickPill("all");

    expect((await screen.findAllByText("Revoked directive.")).length).toBeGreaterThan(0);
    expect(screen.queryByRole("button", { name: "revoke" })).not.toBeInTheDocument();
    expect(screen.queryByRole("button", { name: "supersede" })).not.toBeInTheDocument();
  });
});
