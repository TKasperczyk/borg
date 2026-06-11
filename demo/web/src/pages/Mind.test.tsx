import { cleanup, fireEvent, render, screen, waitFor } from "@testing-library/react";

import type {
  ApiState,
  Commitment,
  CreatorDirective,
  IdentityResponse,
  MemoryBandSummary,
  SemanticGraphResponse,
} from "../api/types";
import { MindPage, commitmentExpiresSoon } from "./Mind";

const now = Date.UTC(2026, 5, 11, 12);

function state(): ApiState {
  return {
    active_session: "default",
    audiences: [],
    counts: {
      turns: 0,
      commitments: 1,
      open_qs: 1,
      open_reviews: 0,
      dream_audit_rows: 0,
    },
    current_mood: {
      session_id: "default",
      valence: 0.4,
      arousal: 0.6,
      updated_at: now,
      half_life_hours: 8,
      recent_triggers: [],
    },
    version: "0.1.0",
  };
}

function identity(): IdentityResponse {
  return {
    values: [
      {
        id: "val_1",
        label: "honesty over closure",
        description: "honesty over closure",
        priority: 0.9,
        state: "established",
        confidence: 0.8,
        created_at: now,
        last_affirmed: now,
        established_at: now,
        support_count: 2,
        contradiction_count: 0,
      },
    ],
    goals: [
      {
        id: "goal_1",
        description: "finish arena log consolidation",
        priority: 0.72,
        status: "active",
        progress_notes: null,
        last_progress_ts: null,
        created_at: now,
        target_at: null,
      },
    ],
    traits: [
      {
        id: "trait_1",
        label: "measured",
        strength: 0.8,
        state: "established",
        confidence: 0.9,
        established_at: now,
        last_reinforced: now,
      },
    ],
    open_questions: [
      {
        id: "oq_1",
        question: "is the Jun 14 deadline still true?",
        urgency: 0.7,
        status: "open",
        source: "contradiction",
        goal_id: null,
        created_at: now,
        last_touched: now,
        related_episode_ids: [],
        related_semantic_node_ids: [],
        resolution_note: null,
        resolved_at: null,
        abandoned_reason: null,
        abandoned_at: null,
      },
    ],
    growth_markers: [
      {
        id: "gm_1",
        ts: now,
        category: "understanding",
        what_changed: "first deliberate silence accepted",
        confidence: 0.6,
        source_process: "manual",
      },
    ],
    periods: [
      {
        id: "period_1",
        label: "consolidation",
        start_ts: now,
        end_ts: null,
        narrative: "current period",
        themes: [],
        created_at: now,
        last_updated: now,
      },
    ],
    open_question_events: [],
  };
}

const directives: CreatorDirective[] = [
  {
    id: "cd_1",
    kind: "subject_fact",
    text: "The operator is the creator.",
    canonical_fact: "The operator is the creator.",
    operational_directive: null,
    activation_scope: "same_as_disclosure",
    content_scope: "public",
    mention_policy: "answer_if_asked",
    status: "active",
    subject_kind: "entity",
    subject_entity_id: "ent_1",
    subject_entity_name: "operator",
    priority: 0.9,
    superseded_by_id: null,
    revoked_reason: null,
    created_at: now,
    updated_at: now,
  },
  {
    id: "cd_2",
    kind: "response_policy",
    text: "Never surface internal record IDs.",
    canonical_fact: null,
    operational_directive: "Never surface internal record IDs.",
    activation_scope: "same_as_disclosure",
    content_scope: "operator_only",
    mention_policy: "never_mention",
    status: "active",
    subject_kind: "system",
    subject_entity_id: null,
    subject_entity_name: null,
    priority: 0.8,
    superseded_by_id: null,
    revoked_reason: null,
    created_at: now,
    updated_at: now,
  },
  {
    id: "cd_3",
    kind: "routing_instruction",
    text: "Old routing instruction.",
    canonical_fact: null,
    operational_directive: "Old routing instruction.",
    activation_scope: "same_as_disclosure",
    content_scope: "public",
    mention_policy: "only_if_topic_raised",
    status: "superseded",
    subject_kind: "system",
    subject_entity_id: null,
    subject_entity_name: null,
    priority: 0.2,
    superseded_by_id: "cd_2",
    revoked_reason: null,
    created_at: now,
    updated_at: now,
  },
];

function commitment(overrides: Partial<Commitment> = {}): Commitment {
  return {
    id: "cm_1",
    text: "Deploy notes disclosed only to the operator.",
    type: "boundary",
    kind: "boundary",
    enforcement_class: "critical",
    critical_domain: "privacy",
    state: "active",
    priority: 7,
    directive_family: "deploy_notes",
    audience: "operator",
    made_to: null,
    about: null,
    committed_by: null,
    source: "manual",
    created_at: now,
    expires_at: now + 60 * 60 * 1000,
    expired_at: null,
    revoked_at: null,
    revoked_reason: null,
    superseded_by_id: null,
    last_reinforced_at: now,
    ...overrides,
  };
}

const bands: MemoryBandSummary[] = [
  "episodic",
  "semantic",
  "procedural",
  "affective",
  "self",
  "commitments",
  "social",
  "relational",
].map((id, index) => ({
  id: id as MemoryBandSummary["id"],
  n: String(index + 1).padStart(2, "0"),
  name: id,
  desc: `${id} desc`,
  count: index + 1,
  stats: [{ k: "items", v: index + 1 }],
}));

const graph: SemanticGraphResponse = {
  nodes: [
    {
      id: "n1",
      label: "operator",
      display_label: "operator",
      status: "active",
      kind: "entity",
      edge_count: 1,
    },
    {
      id: "n2",
      label: "deadline = jun 14",
      display_label: "deadline = jun 14",
      status: "contested",
      kind: "proposition",
      edge_count: 1,
    },
  ],
  edges: [{ id: "edge_1", source: "n1", target: "n2", type: "contradicts", weight: 0.5 }],
  total_nodes: 2,
  total_edges: 1,
  rendered: { nodes: 2, edges: 1 },
};

function json(body: unknown, status = 200): Response {
  return new Response(JSON.stringify(body), {
    status,
    headers: { "Content-Type": "application/json" },
  });
}

type Deferred<T> = {
  promise: Promise<T>;
  resolve: (value: T) => void;
};

function deferred<T>(): Deferred<T> {
  let resolve!: (value: T) => void;
  const promise = new Promise<T>((nextResolve) => {
    resolve = nextResolve;
  });
  return { promise, resolve };
}

type RenderOptions = {
  deferGoalPatch?: boolean;
  deferNode2Detail?: boolean;
};

function renderMind(path = "/mind", options: RenderOptions = {}) {
  window.history.pushState({}, "", path);
  const requests: Array<{ url: string; method: string; body: unknown }> = [];
  const goalPatch = deferred<Response>();
  const node2Detail = deferred<Response>();

  vi.spyOn(globalThis, "fetch").mockImplementation(async (input, init) => {
    const url = String(input);
    const method = init?.method ?? "GET";
    requests.push({
      url,
      method,
      body: init?.body === undefined ? null : JSON.parse(String(init.body)),
    });

    if (url === "/api/state") {
      return json(state());
    }
    if (url === "/api/identity") {
      return json(identity());
    }
    if (url.startsWith("/api/identity/goals/goal_1") && method === "PATCH") {
      return options.deferGoalPatch ? goalPatch.promise : json({ ok: true });
    }
    if (url.startsWith("/api/creator-directives")) {
      if (method === "POST") {
        return json(directives[0]);
      }
      return json({ directives });
    }
    if (url.startsWith("/api/commitments")) {
      if (method === "POST") {
        return json(commitment({ state: "revoked" }));
      }
      return json({ commitments: [commitment()] });
    }
    if (url.startsWith("/api/memory/bands/episodic")) {
      if (url.includes("cursor=c2")) {
        return json({
          band: "episodic",
          mode: "browse",
          items: [
            { id: "ep_1", title: "first episode", created_at: now },
            { id: "ep_2", title: "second episode", created_at: now },
          ],
          next_cursor: null,
        });
      }
      return json({
        band: "episodic",
        mode: "browse",
        items: [{ id: "ep_1", title: "first episode", created_at: now }],
        next_cursor: "c2",
      });
    }
    if (url.startsWith("/api/memory/bands/affective")) {
      return json({
        band: "affective",
        mode: "browse",
        current: state().current_mood,
        history: [
          {
            session_id: "default",
            valence: 0.1,
            arousal: 0.2,
            updated_at: now - 1000,
            half_life_hours: 8,
            recent_triggers: [],
          },
        ],
      });
    }
    if (url === "/api/memory/bands") {
      return json({ bands });
    }
    if (url.startsWith("/api/semantic/graph")) {
      return json(graph);
    }
    if (url.startsWith("/api/semantic/nodes/n1")) {
      return json({
        node: {
          id: "n1",
          kind: "entity",
          label: "operator",
          display_label: "operator",
          description: "creator entity",
          domain: null,
          aliases: [],
          confidence: 0.9,
          status: "active",
          source_episode_ids: ["ep_1"],
          source_count: 1,
          created_at: now,
          updated_at: now,
        },
      });
    }
    if (url.startsWith("/api/semantic/nodes/n2")) {
      const response = json({
        node: {
          id: "n2",
          kind: "proposition",
          label: "deadline = jun 14",
          display_label: "deadline = jun 14",
          description: "deadline proposition",
          domain: "schedule",
          aliases: [],
          confidence: 0.6,
          status: "contested",
          source_episode_ids: ["ep_2"],
          source_count: 1,
          created_at: now,
          updated_at: now,
        },
      });
      if (options.deferNode2Detail) {
        return node2Detail.promise;
      }
      return response;
    }
    if (url.startsWith("/api/correction/semantic-edges/edge_1/invalidate")) {
      return json({ ok: true });
    }

    return json({ message: "not found" }, 404);
  });

  render(<MindPage />);
  return { requests, goalPatch, node2Detail };
}

describe("Mind page", () => {
  afterEach(() => {
    vi.restoreAllMocks();
  });

  it("posts exact real identity action payloads", async () => {
    const { requests } = renderMind();

    await screen.findByText("finish arena log consolidation");
    fireEvent.click(screen.getByText("BUMP +"));
    await waitFor(() =>
      expect(requests).toContainEqual(
        expect.objectContaining({
          url: "/api/identity/goals/goal_1",
          method: "PATCH",
          body: { action: "progress", progress: 82, note: "operator bump +0.1" },
        }),
      ),
    );

    expect(screen.getByText("BLOCK")).toBeTruthy();
    fireEvent.click(screen.getByText("BLOCK"));
    fireEvent.change(screen.getByPlaceholderText("reason"), { target: { value: "not useful" } });
    fireEvent.click(screen.getByText("CONFIRM"));
    await waitFor(() =>
      expect(requests).toContainEqual(
        expect.objectContaining({
          url: "/api/identity/goals/goal_1",
          method: "PATCH",
          body: { action: "block", note: "not useful" },
        }),
      ),
    );

    fireEvent.click(screen.getByText("RESOLVE"));
    fireEvent.change(screen.getByPlaceholderText("resolution"), { target: { value: "deadline moved" } });
    fireEvent.click(screen.getAllByText("CONFIRM").at(-1)!);
    await waitFor(() =>
      expect(requests).toContainEqual(
        expect.objectContaining({
          url: "/api/identity/open-questions/oq_1",
          method: "PATCH",
          body: { action: "resolve", resolution: "deadline moved" },
        }),
      ),
    );
  });

  it("uses block copy for inspector goal actions", async () => {
    renderMind("/mind/inspect/identity");

    expect(await screen.findByText("BUMP / BLOCK")).toBeTruthy();
  });

  it("guards inline mutations while a request is pending", async () => {
    const { requests, goalPatch } = renderMind("/mind", { deferGoalPatch: true });

    await screen.findByText("finish arena log consolidation");
    const bump = screen.getByText("BUMP +") as HTMLButtonElement;
    fireEvent.click(bump);
    await waitFor(() => expect(bump.disabled).toBe(true));

    fireEvent.click(bump);
    expect(
      requests.filter((request) => request.url === "/api/identity/goals/goal_1" && request.method === "PATCH"),
    ).toHaveLength(1);

    goalPatch.resolve(json({ ok: true }));
    await waitFor(() => expect(bump.disabled).toBe(false));
  });

  it("supersedes directives with a replacement id and renders superseded rows inactive", async () => {
    const { requests } = renderMind();

    await screen.findByText("The operator is the creator.");
    fireEvent.click(screen.getAllByText("all")[0]!);
    expect(screen.getByText("Old routing instruction.").className).toContain("ledger-text-inactive");

    fireEvent.click(screen.getAllByText("SUPERSEDE")[0]!);
    fireEvent.click(screen.getByText("CONFIRM"));
    await waitFor(() =>
      expect(requests).toContainEqual(
        expect.objectContaining({
          url: "/api/creator-directives/cd_1/supersede",
          method: "POST",
          body: { replacement_id: "cd_2" },
        }),
      ),
    );
  });

  it("renders commitment enforcement, computes expiring soon structurally, and revokes with reason", async () => {
    expect(commitmentExpiresSoon(commitment(), now)).toBe(true);
    expect(commitmentExpiresSoon(commitment({ expires_at: null }), now)).toBe(false);

    const { requests } = renderMind();
    await screen.findByText("Deploy notes disclosed only to the operator.");
    expect(screen.getByText("CRITICAL")).toBeTruthy();

    fireEvent.click(screen.getAllByText("REVOKE").at(-1)!);
    fireEvent.change(screen.getByPlaceholderText("reason"), { target: { value: "changed" } });
    fireEvent.click(screen.getByText("CONFIRM"));

    await waitFor(() =>
      expect(requests).toContainEqual(
        expect.objectContaining({
          url: "/api/commitments/cm_1/revoke",
          method: "POST",
          body: { reason: "changed" },
        }),
      ),
    );
  });

  it("renders band cards from real-shaped summaries and deep-links inspector bands", async () => {
    renderMind("/mind/inspect/semantic");

    expect(await screen.findByText("semantic")).toBeTruthy();
    expect(screen.getByText("02 / semantic")).toBeTruthy();
  });

  it("appends episodic pages and renders affective special-case rows", async () => {
    renderMind("/mind/inspect/episodic");

    expect(await screen.findByText("first episode")).toBeTruthy();
    fireEvent.click(screen.getByText("LOAD MORE"));
    expect(await screen.findByText("second episode")).toBeTruthy();
    expect(screen.getAllByText("first episode")).toHaveLength(1);

    cleanup();
    renderMind("/mind/inspect/affective");
    expect(await screen.findByText(/valence 0.40/)).toBeTruthy();
    expect(await screen.findByText(/valence 0.10/)).toBeTruthy();
  });

  it("renders graph detail from real fields and invalidates selected edges", async () => {
    const { requests } = renderMind();

    expect(await screen.findAllByText("operator")).toHaveLength(2);
    expect(await screen.findByText(/confidence 0.90/)).toBeTruthy();
    fireEvent.click(screen.getByText("INVALIDATE EDGE"));
    fireEvent.change(screen.getByPlaceholderText("reason"), { target: { value: "bad edge" } });
    fireEvent.click(screen.getByText("POST"));

    await waitFor(() =>
      expect(requests).toContainEqual(
        expect.objectContaining({
          url: "/api/correction/semantic-edges/edge_1/invalidate",
          method: "POST",
          body: { reason: "bad edge" },
        }),
      ),
    );
  });

  it("suppresses stale graph detail while a newly selected node loads", async () => {
    renderMind("/mind", { deferNode2Detail: true });

    expect(await screen.findByText(/confidence 0.90/)).toBeTruthy();
    fireEvent.click(screen.getByLabelText("select deadline = jun 14"));

    expect(await screen.findByText("loading node detail…")).toBeTruthy();
    expect(screen.queryByText(/confidence 0.90/)).toBeNull();
    expect(screen.queryByText("creator entity")).toBeNull();
  });
});
