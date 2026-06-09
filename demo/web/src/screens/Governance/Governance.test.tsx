import { fireEvent, screen, waitFor, within } from "@testing-library/react";
import { useState } from "react";
import { afterEach, describe, expect, it, vi } from "vitest";

import type {
  CommitmentItem,
  CreatorDirectiveItem,
  ReviewRow,
  SessionRecord,
  SharedStateEntry,
  StateSnapshot,
} from "../../api/types";
import { renderWithInspector } from "../../test/inspector";
import type { GovernanceTabId } from "../../routes";
import { GovernanceScreen } from ".";

function jsonResponse(body: unknown): Response {
  return new Response(JSON.stringify(body), {
    status: 200,
    headers: { "Content-Type": "application/json" },
  });
}

function requestUrl(request: RequestInfo | URL): URL {
  return new URL(String(request), "http://test.invalid");
}

function callsForPath(fetchMock: ReturnType<typeof vi.fn>, path: string): number {
  return fetchMock.mock.calls.filter((call) => requestUrl(call[0]).pathname === path).length;
}

function session(input: Partial<SessionRecord> = {}): SessionRecord {
  return {
    session_id: "sess_scope111111",
    source_type: "demo",
    source_external_id: null,
    source_url: null,
    label: "scope lab",
    audience_label: "alice",
    audience_entity_id: "ent_alice111111",
    conversation_kind: "demo",
    created_at: 1,
    last_activity_at: 2,
    last_turn_id: "turn_scope111111",
    message_count: 7,
    status: "active",
    privacy_level: "payload_off",
    participation_policy: "active",
    audience_role: "participant",
    ...input,
  };
}

function stateSnapshot(): StateSnapshot {
  return {
    active_session: "sess_scope111111",
    audiences: ["alice"],
    counts: {
      turns: 0,
      commitments: 1,
      open_qs: 0,
      open_reviews: 1,
      dream_audit_rows: 0,
    },
    current_mood: {
      session_id: "sess_scope111111",
      valence: 0,
      arousal: 0,
      updated_at: 1,
      half_life_hours: 24,
      recent_triggers: [],
    },
    version: "test",
  };
}

function commitment(input: Partial<CommitmentItem> = {}): CommitmentItem {
  return {
    id: "cmt_scope11111111",
    text: "Keep scope labels visible.",
    type: "rule",
    kind: "process_norm",
    enforcement_class: "critical",
    critical_domain: "governance",
    state: "active",
    priority: 8,
    directive_family: "scope",
    audience: "alice",
    made_to: null,
    about: null,
    committed_by: null,
    source: "manual",
    source_stream_entry_ids: ["strm_scope111111"],
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

function directive(input: Partial<CreatorDirectiveItem> = {}): CreatorDirectiveItem {
  return {
    id: "cdir_scope111111",
    kind: "subject_fact",
    text: "Alice owns launch review.",
    source_session_id: "sess_scope111111",
    authorization_stream_entry_ids: ["strm_scope111111"],
    content_source_stream_entry_ids: ["strm_scope111111"],
    canonical_fact: "Alice owns launch review.",
    operational_directive: null,
    activation_scope: "same_as_disclosure",
    activation_allowed_entity_ids: [],
    activation_excluded_entity_ids: [],
    content_scope: "public",
    mention_policy: "only_if_topic_raised",
    status: "active",
    subject_kind: "entity",
    subject_entity_id: "ent_alice111111",
    subject_entity_name: "Alice",
    priority: 6,
    superseded_by_id: null,
    revoked_reason: null,
    created_at: 1,
    updated_at: 1,
    ...input,
  };
}

function sharedEntry(input: Partial<SharedStateEntry> = {}): SharedStateEntry {
  return {
    id: "dart_scope111111",
    audience_entity_id: "ent_alice111111",
    state_key: "scope.alice",
    kind: "locked",
    text: "Scoped lifecycle value.",
    owner_entity_id: null,
    provenance_stream_entry_ids: ["strm_scope111111"],
    last_updated_stream_entry_ids: ["strm_scope111111"],
    created_at: 1,
    last_updated_at: 2,
    last_updated_turn_global: 3,
    superseded_by_id: null,
    rank: 0,
    canonicalizes: {
      goal_ids: [],
      commitment_ids: ["cmt_scope11111111"],
      action_ids: [],
      open_question_ids: [],
    },
    ...input,
  };
}

function review(): ReviewRow {
  return {
    id: 42,
    kind: "commitment_reconciliation",
    refs: { commitment_ids: ["cmt_scope11111111"], session_id: "sess_scope111111" },
    reason: "linked scope review",
    created_at: 1,
    resolved_at: null,
    resolution: null,
  };
}

afterEach(() => {
  vi.unstubAllGlobals();
});

describe("GovernanceScreen", () => {
  it("renders tabs, matrix aggregates, entity handles, and the shared policy editor", async () => {
    const sessions = [session()];
    const commitments = [commitment()];
    const directives = [directive()];
    const fetchMock = vi.fn((request: RequestInfo | URL, init?: RequestInit) => {
      const url = requestUrl(request);

      if (url.pathname === "/api/commitments") {
        return Promise.resolve(jsonResponse({ commitments }));
      }
      if (url.pathname === "/api/creator-directives") {
        return Promise.resolve(jsonResponse({ directives }));
      }
      if (url.pathname === "/api/sessions" && init?.method !== "POST") {
        return Promise.resolve(jsonResponse({ sessions }));
      }
      if (url.pathname === "/api/state") {
        return Promise.resolve(jsonResponse(stateSnapshot()));
      }
      if (url.pathname === "/api/shared-state") {
        return Promise.resolve(
          jsonResponse({
            audience: url.searchParams.get("audience") ?? "self",
            entries: url.searchParams.get("audience") === "alice" ? [sharedEntry()] : [],
          }),
        );
      }
      if (url.pathname === "/api/reviews") {
        return Promise.resolve(jsonResponse({ rows: [review()] }));
      }
      if (url.pathname === "/api/sessions/sess_scope111111/participation") {
        return Promise.resolve(jsonResponse({ ...sessions[0], participation_policy: "muted" }));
      }

      return Promise.resolve(new Response("{}", { status: 404 }));
    });
    vi.stubGlobal("fetch", fetchMock);
    const onSessionPolicyChanged = vi.fn(async () => undefined);

    function Harness() {
      const [activeTab, setActiveTab] = useState<GovernanceTabId>("commitments");
      return (
        <GovernanceScreen
          sessionId="sess_scope111111"
          activeSessionId="sess_scope111111"
          activeTab={activeTab}
          sessions={sessions}
          creator={null}
          operatorChatError={null}
          onSelectSession={vi.fn()}
          onOpenOperatorChat={vi.fn()}
          onSetCreatorByName={vi.fn()}
          onSessionPolicyChanged={onSessionPolicyChanged}
          onTabChange={setActiveTab}
        />
      );
    }

    renderWithInspector(<Harness />);

    expect(screen.getByRole("tab", { name: "Commitments" })).toHaveAttribute(
      "aria-selected",
      "true",
    );
    expect((await screen.findAllByText("Keep scope labels visible.")).length).toBeGreaterThan(0);
    expect(callsForPath(fetchMock, "/api/commitments")).toBe(1);
    expect(callsForPath(fetchMock, "/api/shared-state")).toBe(0);

    fireEvent.click(screen.getByRole("tab", { name: "Directives & shared state" }));
    expect((await screen.findAllByText("Alice owns launch review.")).length).toBeGreaterThan(0);
    expect(within(screen.getByRole("tabpanel")).getByText("scope.alice")).toBeInTheDocument();
    const sharedStateCallsAfterDirectives = callsForPath(fetchMock, "/api/shared-state");
    expect(sharedStateCallsAfterDirectives).toBeGreaterThan(0);
    expect(callsForPath(fetchMock, "/api/commitments")).toBe(1);

    fireEvent.click(screen.getByRole("tab", { name: "Commitments" }));
    fireEvent.click(screen.getByRole("tab", { name: "Directives & shared state" }));
    expect((await screen.findAllByText("Alice owns launch review.")).length).toBeGreaterThan(0);
    expect(callsForPath(fetchMock, "/api/shared-state")).toBe(sharedStateCallsAfterDirectives);

    fireEvent.click(screen.getByRole("tab", { name: "Scope matrix" }));
    expect(await screen.findByText("scope lab")).toBeInTheDocument();
    expect(screen.getAllByText("1 active / 1 critical").length).toBeGreaterThan(0);
    expect(screen.getByText("1 rows")).toBeInTheDocument();
    expect(screen.getAllByText("1 linked open").length).toBeGreaterThan(0);
    expect(screen.getByText(/not recall gates or output controls/i)).toBeInTheDocument();

    fireEvent.click(screen.getByRole("tab", { name: "Sessions & entities" }));
    expect(await screen.findByText("entities known from sessions/directives")).toBeInTheDocument();
    expect(
      screen.getAllByRole("button", { name: "jump to ent_alice111111" }).length,
    ).toBeGreaterThan(0);
    expect(screen.getByText(/no generic entity create/i)).toBeInTheDocument();

    fireEvent.click(screen.getByRole("button", { name: "participation policy active" }));
    fireEvent.change(screen.getByLabelText("participation policy selection"), {
      target: { value: "muted" },
    });
    fireEvent.change(screen.getByLabelText("participation policy reason"), {
      target: { value: "operator requested quiet" },
    });
    fireEvent.click(screen.getByRole("button", { name: "apply" }));

    await waitFor(() => expect(onSessionPolicyChanged).toHaveBeenCalled());
    const postCall = fetchMock.mock.calls.find(
      (call) => requestUrl(call[0]).pathname === "/api/sessions/sess_scope111111/participation",
    );
    expect(postCall).toBeDefined();
    expect(JSON.parse(String(postCall?.[1]?.body))).toEqual({
      policy: "muted",
      reason: "operator requested quiet",
    });
  });
});
