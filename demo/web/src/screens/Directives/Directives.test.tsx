import { render, screen, waitFor } from "@testing-library/react";
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

function requestMethod(init?: RequestInit): string {
  return init?.method ?? "GET";
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
    created_at: 1,
    ...input,
  };
}

afterEach(() => {
  vi.unstubAllGlobals();
});

describe("DirectivesScreen", () => {
  it("renders creator directives from the read-only endpoint", async () => {
    const directives = [directive()];
    const fetchMock = vi.fn((request: RequestInfo | URL, init?: RequestInit) => {
      const path = requestPath(request);
      const method = requestMethod(init);
      if (path === "/api/creator-directives" && method === "GET") {
        return Promise.resolve(jsonResponse({ directives }));
      }
      return Promise.resolve(new Response("{}", { status: 404 }));
    });
    vi.stubGlobal("fetch", fetchMock);

    render(<DirectivesScreen />);

    expect((await screen.findAllByText("Alice is the launch reviewer.")).length).toBeGreaterThan(0);
    expect(screen.getAllByText("subject_fact").length).toBeGreaterThan(0);
    expect(screen.getAllByText("same_as_disclosure").length).toBeGreaterThan(0);
    expect(screen.getAllByText("public").length).toBeGreaterThan(0);
    expect(screen.getAllByText("only_if_topic_raised").length).toBeGreaterThan(0);
    expect(screen.getAllByText("active").length).toBeGreaterThan(0);
    expect(screen.getByText("Alice")).toBeInTheDocument();
    expect(screen.queryByLabelText("add directive")).not.toBeInTheDocument();
    expect(screen.queryByRole("button", { name: "revoke" })).not.toBeInTheDocument();
    expect(screen.queryByRole("button", { name: "forget" })).not.toBeInTheDocument();
    expect(screen.queryByRole("button", { name: "correct" })).not.toBeInTheDocument();

    await waitFor(() => {
      expect(
        fetchMock.mock.calls.filter(
          (call) =>
            requestPath(call[0]) === "/api/creator-directives" && requestMethod(call[1]) === "GET",
        ),
      ).toHaveLength(1);
    });
  });
});
