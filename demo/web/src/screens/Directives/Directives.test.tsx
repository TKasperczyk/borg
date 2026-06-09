import { fireEvent, screen, waitFor, within } from "@testing-library/react";
import { renderWithInspector } from "../../test/inspector";
import { afterEach, describe, expect, it, vi } from "vitest";

import type {
  CommitmentItem,
  CreatorDirectiveItem,
  SessionRecord,
  SharedStateEntry,
} from "../../api/types";
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
    source_session_id: "sess_directive1111",
    authorization_stream_entry_ids: ["strm_directive1111"],
    content_source_stream_entry_ids: ["strm_directive1111"],
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

function session(input: Partial<SessionRecord> = {}): SessionRecord {
  return {
    session_id: "sess_directive1111",
    source_type: "demo",
    source_external_id: null,
    source_url: null,
    label: "demo",
    audience_label: "alice",
    audience_entity_id: "ent_1111111111111111",
    conversation_kind: "demo",
    created_at: 1,
    last_activity_at: 1,
    last_turn_id: null,
    message_count: 0,
    status: "active",
    privacy_level: "payload_off",
    participation_policy: "active",
    audience_role: "participant",
    ...input,
  };
}

function commitment(input: Partial<CommitmentItem> = {}): CommitmentItem {
  return {
    id: "cmt_1111111111111111",
    text: "Retired commitment.",
    type: "rule",
    kind: "audience_rule",
    enforcement_class: "advisory",
    critical_domain: null,
    state: "revoked",
    priority: 7,
    directive_family: "demo_family",
    audience: "alice",
    made_to: null,
    about: null,
    committed_by: null,
    source: "online",
    source_stream_entry_ids: ["strm_directive1111"],
    created_at: 1,
    expires_at: null,
    expired_at: null,
    revoked_at: 2,
    revoked_reason: "canonicalized_by_artifact_entry_id=dart_1111111111111111",
    superseded_by_id: null,
    canonicalized_by_artifact_entry_id: "dart_1111111111111111",
    last_reinforced_at: 1,
    ...input,
  };
}

function sharedEntry(input: Partial<SharedStateEntry> = {}): SharedStateEntry {
  return {
    id: "dart_1111111111111111",
    audience_entity_id: "ent_1111111111111111",
    state_key: "rule.demo",
    kind: "locked",
    text: "Canonical shared-state row.",
    owner_entity_id: null,
    provenance_stream_entry_ids: ["strm_shared1111"],
    last_updated_stream_entry_ids: ["strm_shared1111"],
    created_at: 1,
    last_updated_at: 2,
    last_updated_turn_global: 3,
    superseded_by_id: null,
    rank: 0,
    canonicalizes: {
      goal_ids: [],
      commitment_ids: [],
      action_ids: [],
      open_question_ids: [],
    },
    ...input,
  };
}

type SupportFixture = {
  stateAudiences?: string[];
  sessions?: SessionRecord[];
  commitments?: CommitmentItem[];
  sharedStateByAudience?: Record<string, SharedStateEntry[]>;
};

function supportResponse(
  request: RequestInfo | URL,
  fixture: SupportFixture = {},
): Response | null {
  const url = new URL(String(request), "http://test.invalid");

  if (url.pathname === "/api/sessions") {
    return jsonResponse({ sessions: fixture.sessions ?? [] });
  }

  if (url.pathname === "/api/state") {
    return jsonResponse({
      active_session: url.searchParams.get("session") ?? "default",
      audiences: fixture.stateAudiences ?? [],
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
        half_life_hours: 24,
        recent_triggers: [],
      },
      version: "test",
    });
  }

  if (url.pathname === "/api/commitments") {
    return jsonResponse({ commitments: fixture.commitments ?? [] });
  }

  if (url.pathname === "/api/shared-state") {
    const audience = url.searchParams.get("audience") ?? "self";
    return jsonResponse({
      audience,
      entries: fixture.sharedStateByAudience?.[audience] ?? [],
    });
  }

  return null;
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
        expect(requestStatus(request)).toBe("all");
        return Promise.resolve(jsonResponse({ directives: [active, revoked] }));
      }
      const support = supportResponse(request);
      if (support !== null) {
        return Promise.resolve(support);
      }
      return Promise.resolve(new Response("{}", { status: 404 }));
    });
    vi.stubGlobal("fetch", fetchMock);

    renderWithInspector(<DirectivesScreen />);

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
        return Promise.resolve(
          jsonResponse({ directives: [active, replacement, revoked, superseded] }),
        );
      }
      const support = supportResponse(request);
      if (support !== null) {
        return Promise.resolve(support);
      }
      return Promise.resolve(new Response("{}", { status: 404 }));
    });
    vi.stubGlobal("fetch", fetchMock);

    renderWithInspector(<DirectivesScreen />);

    expect(await screen.findByText("Active directive.")).toBeInTheDocument();
    expect(screen.queryByText("Revoked directive.")).not.toBeInTheDocument();

    clickPill("all");

    expect((await screen.findAllByText("Revoked directive.")).length).toBeGreaterThan(0);
    expect(screen.getByText("revoked: replaced by newer policy")).toBeInTheDocument();
    expect(screen.getByText(/superseded by:/)).toBeInTheDocument();

    clickPill("revoked");

    expect((await screen.findAllByText("Revoked directive.")).length).toBeGreaterThan(0);
    await waitFor(() => {
      expect(screen.queryByText("Active directive.")).not.toBeInTheDocument();
    });

    expect(screen.getAllByText("revoked").some((element) => element.classList.contains("on"))).toBe(
      true,
    );
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
        return Promise.resolve(jsonResponse({ directives: [replacement, superseded] }));
      }
      const support = supportResponse(request);
      if (support !== null) {
        return Promise.resolve(support);
      }
      return Promise.resolve(new Response("{}", { status: 404 }));
    });
    vi.stubGlobal("fetch", fetchMock);

    renderWithInspector(<DirectivesScreen />);

    expect((await screen.findAllByText("Replacement directive.")).length).toBeGreaterThan(0);
    clickPill("superseded");

    expect((await screen.findAllByText("Superseded directive.")).length).toBeGreaterThan(0);
    fireEvent.click(
      screen.getAllByRole("button", { name: `jump to directive ${replacement.id}` })[0]!,
    );

    expect(await screen.findByText(replacement.id)).toBeInTheDocument();
    expect(screen.getAllByText("all").some((element) => element.classList.contains("on"))).toBe(
      true,
    );
  });

  it("renders structurally related shared-state lifecycle and keeps uncorrelated rows visible", async () => {
    const selected = directive({
      id: "cdir_policy111111",
      kind: "response_policy",
      text: "Selected policy.",
      canonical_fact: null,
      operational_directive: "Selected policy.",
      source_session_id: "sess_tom111111111",
      authorization_stream_entry_ids: ["strm_rule_source"],
      content_source_stream_entry_ids: ["strm_rule_source"],
      priority: 80,
    });
    const retiredCommitment = commitment({
      id: "cmt_retired111111",
      state: "revoked",
      source_stream_entry_ids: ["strm_rule_source"],
      canonicalized_by_artifact_entry_id: "dart_rule11111111",
      revoked_reason: "canonicalized_by_artifact_entry_id=dart_rule11111111",
    });
    const sharedSourceOnlyCommitment = commitment({
      id: "cmt_sourceonly111",
      state: "revoked",
      source_stream_entry_ids: ["strm_other_commitment"],
      canonicalized_by_artifact_entry_id: "dart_sourceonly11",
      revoked_reason: "canonicalized_by_artifact_entry_id=dart_sourceonly11",
    });
    const related = sharedEntry({
      id: "dart_rule11111111",
      state_key: "rule.botarena.language",
      text: "Locked lifecycle value.",
      provenance_stream_entry_ids: ["strm_shared_source"],
      last_updated_stream_entry_ids: ["strm_shared_source"],
      canonicalizes: {
        goal_ids: [],
        commitment_ids: [retiredCommitment.id],
        action_ids: [],
        open_question_ids: [],
      },
    });
    const sharedSourceOnly = sharedEntry({
      id: "dart_sourceonly11",
      state_key: "rule.shared-source-only",
      text: "Shared-source-only lifecycle value.",
      provenance_stream_entry_ids: ["strm_rule_source"],
      last_updated_stream_entry_ids: ["strm_rule_source"],
      canonicalizes: {
        goal_ids: [],
        commitment_ids: [sharedSourceOnlyCommitment.id],
        action_ids: [],
        open_question_ids: [],
      },
    });
    const uncorrelated = sharedEntry({
      id: "dart_uncorrelated1",
      state_key: "uncorrelated.thread",
      text: "Uncorrelated lifecycle value.",
      provenance_stream_entry_ids: ["strm_uncorrelated"],
      last_updated_stream_entry_ids: ["strm_uncorrelated"],
      canonicalizes: {
        goal_ids: [],
        commitment_ids: [sharedSourceOnlyCommitment.id],
        action_ids: [],
        open_question_ids: [],
      },
    });
    const fixture: SupportFixture = {
      sessions: [
        session({ session_id: "sess_tom111111111", audience_label: "Tom" }),
        session({
          session_id: "sess_arena111111",
          audience_label: "botarena_thread:test",
          audience_entity_id: "ent_arena111111111",
        }),
      ],
      commitments: [retiredCommitment, sharedSourceOnlyCommitment],
      sharedStateByAudience: {
        Tom: [related, sharedSourceOnly, uncorrelated],
        "botarena_thread:test": [],
      },
    };
    const fetchMock = vi.fn((request: RequestInfo | URL, init?: RequestInit) => {
      const path = requestPath(request);
      const method = requestMethod(init);
      if (path === "/api/creator-directives" && method === "GET") {
        return Promise.resolve(jsonResponse({ directives: [selected] }));
      }
      const support = supportResponse(request, fixture);
      if (support !== null) {
        return Promise.resolve(support);
      }
      return Promise.resolve(new Response("{}", { status: 404 }));
    });
    vi.stubGlobal("fetch", fetchMock);

    renderWithInspector(<DirectivesScreen />);

    expect(await screen.findByText("rule.botarena.language")).toBeInTheDocument();
    expect(screen.getByText(/related via canonicalized commitment/)).toHaveTextContent("revoked");
    expect(
      screen.getByRole("button", { name: `inspect canonical target ${retiredCommitment.id}` }),
    ).toHaveTextContent("revoked");
    expect(screen.getByText("rule.shared-source-only")).toBeInTheDocument();
    expect(screen.getByText(/related via shared source/)).toBeInTheDocument();
    expect(
      screen.getAllByText("canonicalized commitment is revoked while selected directive is active"),
    ).toHaveLength(1);
    expect(screen.getByText("uncorrelated.thread")).toBeInTheDocument();
    expect(
      screen.getByText(/empty shared-state audiences: self, botarena_thread:test/),
    ).toBeInTheDocument();
  });

  it("discovers shared-state rows from self and active session state audiences", async () => {
    const selected = directive({
      id: "cdir_stateaud11111",
      text: "Directive with non-session lifecycle rows.",
    });
    const selfEntry = sharedEntry({
      id: "dart_self111111111",
      state_key: "self.identity.row",
      text: "Self lifecycle row.",
    });
    const activeStateEntry = sharedEntry({
      id: "dart_active111111",
      state_key: "thread.active.row",
      text: "Active session lifecycle row.",
    });
    const fixture: SupportFixture = {
      stateAudiences: ["thread:active"],
      sessions: [session({ audience_label: "alice" })],
      sharedStateByAudience: {
        self: [selfEntry],
        "thread:active": [activeStateEntry],
        alice: [],
      },
    };
    const fetchMock = vi.fn((request: RequestInfo | URL, init?: RequestInit) => {
      const path = requestPath(request);
      const method = requestMethod(init);
      if (path === "/api/creator-directives" && method === "GET") {
        return Promise.resolve(jsonResponse({ directives: [selected] }));
      }
      const support = supportResponse(request, fixture);
      if (support !== null) {
        return Promise.resolve(support);
      }
      return Promise.resolve(new Response("{}", { status: 404 }));
    });
    vi.stubGlobal("fetch", fetchMock);

    renderWithInspector(<DirectivesScreen sessionId="sess_active111111" />);

    expect(await screen.findByText("self.identity.row")).toBeInTheDocument();
    expect(screen.getByText("thread.active.row")).toBeInTheDocument();
    expect(
      fetchMock.mock.calls.some(
        (call) =>
          requestPath(call[0]) === "/api/state" &&
          new URL(String(call[0]), "http://test.invalid").searchParams.get("session") ===
            "sess_active111111",
      ),
    ).toBe(true);
  });

  it("distinguishes superseded canonicalized commitments from explicitly revoked ones", async () => {
    const selected = directive({
      id: "cdir_policy222222",
      kind: "response_policy",
      text: "Selected superseded policy.",
      authorization_stream_entry_ids: ["strm_superseded_source"],
      content_source_stream_entry_ids: ["strm_superseded_source"],
    });
    const replacement = commitment({
      id: "cmt_replacement111",
      text: "Replacement commitment.",
      state: "active",
      source_stream_entry_ids: ["strm_replacement"],
      revoked_at: null,
      revoked_reason: null,
      canonicalized_by_artifact_entry_id: null,
    });
    const superseded = commitment({
      id: "cmt_superseded111",
      state: "revoked",
      source_stream_entry_ids: ["strm_superseded_source"],
      canonicalized_by_artifact_entry_id: "dart_superseded1",
      superseded_by_id: replacement.id,
      revoked_reason: null,
    });
    const related = sharedEntry({
      id: "dart_superseded1",
      state_key: "rule.superseded",
      canonicalizes: {
        goal_ids: [],
        commitment_ids: [superseded.id],
        action_ids: [],
        open_question_ids: [],
      },
    });
    const fixture: SupportFixture = {
      sessions: [session({ audience_label: "Tom" })],
      commitments: [replacement, superseded],
      sharedStateByAudience: {
        self: [],
        Tom: [related],
      },
    };
    const fetchMock = vi.fn((request: RequestInfo | URL, init?: RequestInit) => {
      const path = requestPath(request);
      const method = requestMethod(init);
      if (path === "/api/creator-directives" && method === "GET") {
        return Promise.resolve(jsonResponse({ directives: [selected] }));
      }
      const support = supportResponse(request, fixture);
      if (support !== null) {
        return Promise.resolve(support);
      }
      return Promise.resolve(new Response("{}", { status: 404 }));
    });
    vi.stubGlobal("fetch", fetchMock);

    renderWithInspector(<DirectivesScreen />);

    expect(await screen.findByText("rule.superseded")).toBeInTheDocument();
    expect(screen.getByText(/related via canonicalized commitment/)).toHaveTextContent(
      "superseded",
    );
    expect(
      screen.getByText("canonicalized commitment is superseded while selected directive is active"),
    ).toBeInTheDocument();
    expect(
      screen.queryByText("canonicalized commitment is revoked while selected directive is active"),
    ).not.toBeInTheDocument();
  });

  it("shows an honest notice when session audience discovery reaches the server cap", async () => {
    const selected = directive();
    const cappedSessions = Array.from({ length: 1000 }, (_, index) =>
      session({
        session_id: `sess_cap${String(index).padStart(4, "0")}`,
        audience_label: "Tom",
      }),
    );
    const fixture: SupportFixture = {
      sessions: cappedSessions,
      sharedStateByAudience: {
        Tom: [],
      },
    };
    const fetchMock = vi.fn((request: RequestInfo | URL, init?: RequestInit) => {
      const path = requestPath(request);
      const method = requestMethod(init);
      if (path === "/api/creator-directives" && method === "GET") {
        return Promise.resolve(jsonResponse({ directives: [selected] }));
      }
      const support = supportResponse(request, fixture);
      if (support !== null) {
        return Promise.resolve(support);
      }
      return Promise.resolve(new Response("{}", { status: 404 }));
    });
    vi.stubGlobal("fetch", fetchMock);

    renderWithInspector(<DirectivesScreen />);

    expect(
      await screen.findByText(/audience discovery reached the 1000-session server cap/),
    ).toBeInTheDocument();
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
      const support = supportResponse(request);
      if (support !== null) {
        return Promise.resolve(support);
      }
      return Promise.resolve(new Response("{}", { status: 404 }));
    });
    vi.stubGlobal("fetch", fetchMock);

    renderWithInspector(<DirectivesScreen />);

    expect((await screen.findAllByText("Directive to revoke.")).length).toBeGreaterThan(0);
    fireEvent.click(screen.getAllByRole("button", { name: "revoke" })[0]!);
    expect(
      within(screen.getByRole("dialog")).getByRole("button", { name: "revoke" }),
    ).toBeDisabled();
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
      const support = supportResponse(request);
      if (support !== null) {
        return Promise.resolve(support);
      }
      return Promise.resolve(new Response("{}", { status: 404 }));
    });
    vi.stubGlobal("fetch", fetchMock);

    renderWithInspector(<DirectivesScreen />);

    expect((await screen.findAllByText("Directive to supersede.")).length).toBeGreaterThan(0);
    fireEvent.click(screen.getAllByRole("button", { name: "supersede" })[0]!);
    fireEvent.change(screen.getByLabelText("replacement"), {
      target: { value: replacement.id },
    });
    fireEvent.click(within(screen.getByRole("dialog")).getByRole("button", { name: "supersede" }));

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
        return Promise.resolve(jsonResponse({ directives: [revoked] }));
      }
      const support = supportResponse(request);
      if (support !== null) {
        return Promise.resolve(support);
      }
      return Promise.resolve(new Response("{}", { status: 404 }));
    });
    vi.stubGlobal("fetch", fetchMock);

    renderWithInspector(<DirectivesScreen />);

    expect(await screen.findByText("no creator directives in filter")).toBeInTheDocument();
    clickPill("all");

    expect((await screen.findAllByText("Revoked directive.")).length).toBeGreaterThan(0);
    expect(screen.queryByRole("button", { name: "revoke" })).not.toBeInTheDocument();
    expect(screen.queryByRole("button", { name: "supersede" })).not.toBeInTheDocument();
  });
});
