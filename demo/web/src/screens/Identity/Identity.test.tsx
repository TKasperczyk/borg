import { fireEvent, screen, waitFor, within } from "@testing-library/react";
import { afterEach, describe, expect, it, vi } from "vitest";

import type { IdentityResponse } from "../../api/types";
import { renderWithInspector } from "../../test/inspector";
import { IdentityScreen } from ".";

const EVIDENCE_EPISODE_ID = "ep_evidence111111";

function jsonResponse(body: unknown): Response {
  return new Response(JSON.stringify(body), {
    status: 200,
    headers: { "Content-Type": "application/json" },
  });
}

function requestPath(request: RequestInfo | URL): string {
  return new URL(String(request), "http://test.invalid").pathname;
}

function episodeBandResponse() {
  return {
    band: "episodic",
    mode: "browse",
    items: [
      {
        id: EVIDENCE_EPISODE_ID,
        title: "Evidence episode",
        narrative: "Evidence episode narrative",
        participants: ["operator"],
        location: null,
        start_time: 1,
        end_time: 1,
        audience: null,
        significance: 0.5,
        confidence: 0.8,
        tags: ["identity"],
        source_stream_ids: ["strm_evidence111111"],
        source_count: 1,
        lineage: { derived_from: [], supersedes: [] },
        emotional_arc: null,
        vector_dims: 4,
        created_at: 1,
        updated_at: 1,
      },
    ],
    next_cursor: null,
  };
}

function identityStudioResponse(): IdentityResponse {
  return {
    values: [
      {
        id: "val_established111111",
        label: "Careful stewardship",
        description: "protect continuity while changing the interface",
        priority: 0.9,
        created_at: 10,
        last_affirmed: 40,
        state: "established",
        confidence: 0.82,
        support_count: 3,
        contradiction_count: 1,
        evidence_episode_ids: [EVIDENCE_EPISODE_ID],
      },
      {
        id: "val_candidate111111",
        label: "Creative rigor",
        description: "explore without losing evidence",
        priority: 0.55,
        created_at: 12,
        last_affirmed: null,
        state: "candidate",
        confidence: 0.61,
        support_count: 2,
        contradiction_count: 0,
        evidence_episode_ids: [],
      },
    ],
    goals: [
      {
        id: "goal_active111111",
        description: "ship the studio",
        priority: 1,
        status: "active",
        progress_notes: "layout is underway",
        created_at: 10,
        target_at: 90,
      },
      {
        id: "goal_blocked111111",
        description: "validate blocked lane",
        priority: 0.7,
        status: "blocked",
        progress_notes: "waiting on review",
        created_at: 11,
        target_at: null,
      },
      {
        id: "goal_done111111",
        description: "close old scaffold",
        priority: 0.3,
        status: "done",
        progress_notes: null,
        created_at: 12,
        target_at: null,
      },
      {
        id: "goal_abandoned111111",
        description: "drop obsolete layout",
        priority: 0.2,
        status: "abandoned",
        progress_notes: null,
        created_at: 13,
        target_at: null,
      },
    ],
    traits: [
      {
        id: "trt_established111111",
        label: "methodical",
        strength: 0.8,
        state: "established",
        confidence: 0.77,
        support_count: 4,
        contradiction_count: 1,
        evidence_episode_ids: [EVIDENCE_EPISODE_ID],
      },
      {
        id: "trt_candidate111111",
        label: "adaptive",
        strength: 0.5,
        state: "candidate",
        confidence: 0.58,
        support_count: 1,
        contradiction_count: 0,
        evidence_episode_ids: [],
      },
    ],
    open_questions: [
      {
        id: "oq_alpha1111111111",
        question: "alpha unresolved question?",
        urgency: 0.95,
        status: "open",
        goal_id: "goal_active111111",
        source: "ruminator",
        created_at: 10,
        last_touched: 30,
        resolved_at: null,
        abandoned_at: null,
        abandoned_reason: null,
        resolution_note: null,
        unresolved_rumination_ticks: 4,
        last_ruminated_at: 29,
      },
      {
        id: "oq_beta2222222222",
        question: "beta unresolved question?",
        urgency: 0.4,
        status: "open",
        goal_id: null,
        source: "reflector",
        created_at: 11,
        last_touched: 20,
        resolved_at: null,
        abandoned_at: null,
        abandoned_reason: null,
        resolution_note: null,
        unresolved_rumination_ticks: 1,
        last_ruminated_at: null,
      },
    ],
    growth_markers: [
      {
        id: "grw_111111111111",
        ts: 70,
        category: "workflow",
        what_changed: "adopted studio review",
        before_description: "reactive patches",
        after_description: "studio review",
        evidence_episode_ids: [EVIDENCE_EPISODE_ID],
        confidence: 0.74,
        source_process: "self-narrator",
        created_at: 70,
      },
    ],
    periods: [
      {
        id: "abp_current111111",
        label: "current arc",
        start_ts: 200,
        end_ts: null,
        narrative: "working in the studio shape",
        key_episode_ids: [EVIDENCE_EPISODE_ID],
        themes: ["identity", "studio"],
        created_at: 200,
        last_updated: 210,
      },
      {
        id: "abp_old111111",
        label: "old arc",
        start_ts: 100,
        end_ts: 150,
        narrative: "flat identity grid",
        key_episode_ids: [],
        themes: [],
        created_at: 100,
        last_updated: 150,
      },
    ],
    open_question_events: [
      {
        id: 1,
        record_type: "open_question",
        record_id: "oq_alpha1111111111",
        action: "create",
        old_value: null,
        new_value: { status: "open", urgency: 0.5, question: "alpha event created" },
        reason: null,
        provenance: { kind: "offline", process: "ruminator" },
        review_item_id: null,
        overwrite_without_review: false,
        ts: 40,
      },
      {
        id: 2,
        record_type: "open_question",
        record_id: "oq_alpha1111111111",
        action: "update",
        old_value: { status: "open", urgency: 0.5, question: "alpha event created" },
        new_value: { status: "open", urgency: 0.95, question: "alpha event updated" },
        reason: "rumination bump",
        provenance: { kind: "offline", process: "ruminator" },
        review_item_id: null,
        overwrite_without_review: false,
        ts: 50,
      },
      {
        id: 3,
        record_type: "open_question",
        record_id: "oq_beta2222222222",
        action: "create",
        old_value: null,
        new_value: { status: "open", urgency: 0.4, question: "beta event created" },
        reason: null,
        provenance: { kind: "online", process: "reflector" },
        review_item_id: 7,
        overwrite_without_review: false,
        ts: 60,
      },
    ],
  };
}

function installIdentityFetch(identity = identityStudioResponse()) {
  const fetchMock = vi.fn((request: RequestInfo | URL, init?: RequestInit) => {
    const path = requestPath(request);
    if (path === "/api/identity" && init?.method === undefined) {
      return Promise.resolve(jsonResponse(identity));
    }
    if (path === "/api/memory/bands/episodic" && init?.method === undefined) {
      return Promise.resolve(jsonResponse(episodeBandResponse()));
    }
    return Promise.resolve(new Response("{}", { status: 404 }));
  });
  vi.stubGlobal("fetch", fetchMock);
  return fetchMock;
}

afterEach(() => {
  vi.unstubAllGlobals();
});

describe("Identity Studio", () => {
  it("renders snapshot counts, latest period, and recent open-question event label", async () => {
    installIdentityFetch();
    renderWithInspector(<IdentityScreen />);

    const header = await screen.findByLabelText("self snapshot header");
    expect(header).toHaveTextContent("current autobiographical period:");
    expect(header).toHaveTextContent("current arc");
    expect(within(header).getByLabelText("snapshot values")).toHaveTextContent(
      "1 established · 1 candidate",
    );
    expect(within(header).getByLabelText("snapshot traits")).toHaveTextContent(
      "1 established · 1 candidate",
    );
    expect(within(header).getByLabelText("snapshot goals")).toHaveTextContent(
      "1 active · 1 blocked · 1 done · 1 abandoned",
    );
    expect(within(header).getByLabelText("snapshot open questions")).toHaveTextContent("2 open");
    expect(within(header).getByLabelText("snapshot growth markers")).toHaveTextContent("1");
    expect(within(header).getByLabelText("snapshot periods")).toHaveTextContent("2");
    expect(within(header).getByLabelText("recent open-question event")).toHaveTextContent(
      "recent open-question event",
    );
    expect(within(header).getByLabelText("recent open-question event")).toHaveTextContent("create");
  });

  it("groups values and traits by state and places goals in status lanes", async () => {
    installIdentityFetch();
    renderWithInspector(<IdentityScreen />);

    const establishedValues = await screen.findByLabelText("established values");
    const candidateValues = screen.getByLabelText("candidate values");
    expect(establishedValues).toHaveTextContent("Careful stewardship");
    expect(establishedValues).not.toHaveTextContent("Creative rigor");
    expect(candidateValues).toHaveTextContent("Creative rigor");

    expect(screen.getByLabelText("established traits")).toHaveTextContent("methodical");
    expect(screen.getByLabelText("candidate traits")).toHaveTextContent("adaptive");

    expect(screen.getByLabelText("active goals")).toHaveTextContent("ship the studio");
    const blockedLane = screen.getByLabelText("blocked goals");
    expect(blockedLane).toHaveTextContent("validate blocked lane");
    expect(within(blockedLane).queryByRole("button", { name: "complete" })).toBeNull();
    expect(within(blockedLane).queryByRole("button", { name: "progress" })).toBeNull();
    expect(within(blockedLane).queryByText("writes live self-band")).toBeNull();
    expect(screen.getByLabelText("done goals")).toHaveTextContent("close old scaffold");
    expect(screen.getByLabelText("abandoned goals")).toHaveTextContent("drop obsolete layout");
  });

  it("omits optional value and trait metrics when sparse metadata is absent", async () => {
    const identity = identityStudioResponse();
    identity.values = [
      {
        id: "val_sparse111111",
        label: "Sparse value",
        description: "no affirmation metadata",
        priority: 0.4,
        created_at: 10,
        state: "established",
        confidence: 0.5,
        support_count: 1,
        contradiction_count: 0,
        evidence_episode_ids: [],
      } as unknown as IdentityResponse["values"][number],
    ];
    identity.traits = [
      {
        id: "trt_sparse111111",
        label: "patient",
        state: "candidate",
        confidence: 0.44,
        support_count: 1,
        contradiction_count: 0,
        evidence_episode_ids: [],
      } as unknown as IdentityResponse["traits"][number],
    ];

    installIdentityFetch(identity);
    renderWithInspector(<IdentityScreen />);

    const valueTitle = await screen.findByText("Sparse value");
    const valueCard = valueTitle.closest("[data-testid='identity-value-card']");
    expect(valueCard).not.toBeNull();
    expect(valueCard as HTMLElement).not.toHaveTextContent("last affirmed");

    const traitTitle = screen.getByText("patient");
    const traitCard = traitTitle.closest("[data-testid='identity-trait-card']");
    expect(traitCard).not.toBeNull();
    expect(traitCard as HTMLElement).not.toHaveTextContent("strength");
    expect(traitCard as HTMLElement).toHaveTextContent("confidence");
  });

  it("selects queue questions, filters the selected timeline, and keeps all events available", async () => {
    installIdentityFetch();
    renderWithInspector(<IdentityScreen />);

    const detail = await screen.findByLabelText("selected open question detail");
    expect(detail).toHaveTextContent("alpha unresolved question?");

    const eventSection = screen.getByLabelText("open question events history");
    expect(eventSection).toHaveTextContent("alpha event updated");
    expect(eventSection).not.toHaveTextContent("beta event created");

    fireEvent.click(screen.getByRole("button", { name: "select open question oq_beta2…2222" }));

    await waitFor(() => {
      expect(screen.getByLabelText("selected open question detail")).toHaveTextContent(
        "beta unresolved question?",
      );
    });
    expect(screen.getByLabelText("open question events history")).toHaveTextContent(
      "beta event created",
    );
    expect(screen.getByLabelText("open question events history")).not.toHaveTextContent(
      "alpha event updated",
    );

    fireEvent.click(screen.getByRole("button", { name: "all events" }));
    expect(screen.getByLabelText("open question events history")).toHaveTextContent(
      "alpha event updated",
    );
    expect(screen.getByLabelText("open question events history")).toHaveTextContent(
      "beta event created",
    );
  });

  it("renders growth before/after detail and autobiographical period metadata", async () => {
    installIdentityFetch();
    renderWithInspector(<IdentityScreen />);

    const growth = (await screen.findByTestId("growth-marker-row")) as HTMLElement;
    expect(growth).toHaveTextContent("workflow");
    expect(growth).toHaveTextContent("adopted studio review");
    expect(growth).toHaveTextContent("reactive patches -> studio review");
    expect(growth).toHaveTextContent("self-narrator");

    const periods = screen.getAllByTestId("autobiographical-period-row");
    expect(periods[0]).toHaveTextContent("current arc");
    expect(periods[0]).toHaveTextContent("identity");
    expect(periods[0]).toHaveTextContent("studio");
    expect(periods[1]).toHaveTextContent("old arc");
  });

  it("opens the inspector from evidence IdRefs", async () => {
    installIdentityFetch();
    renderWithInspector(<IdentityScreen />, { inspector: true });

    fireEvent.click((await screen.findAllByLabelText(`jump to ${EVIDENCE_EPISODE_ID}`))[0]!);

    expect(await screen.findByRole("dialog", { name: "Episode inspector" })).toBeInTheDocument();
    expect(await screen.findByText("Evidence episode")).toBeInTheDocument();
  });
});
