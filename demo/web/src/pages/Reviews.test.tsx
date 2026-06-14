import { act, fireEvent, render, screen, waitFor } from "@testing-library/react";

import type {
  Commitment,
  CreatorDirective,
  EpisodeDetail,
  IdentityResponse,
  ReviewRow,
  SemanticEdgeDetail,
  SemanticNodeDetail,
} from "../api/types";
import { ReviewsPage, actionsForReviewKind, mergeReviewRows, reviewKindColor } from "./Reviews";

const now = Date.UTC(2026, 5, 11, 12);

function row(input: Partial<ReviewRow> & Pick<ReviewRow, "id" | "kind" | "reason" | "refs">): ReviewRow {
  return {
    created_at: now - 6 * 60 * 60 * 1000,
    resolved_at: null,
    resolution: null,
    ...input,
  };
}

const contradiction = row({
  id: 412,
  kind: "contradiction",
  reason: "Two believed arena deadlines cannot both be true.",
  refs: {
    node_ids: ["semn_a", "semn_b"],
    node_labels: ["deadline = jun 14", "deadline = jun 21"],
    edge_id: "seme_contradiction",
    vector_similarity: 0.42,
  },
});

const directiveReview = row({
  id: 425,
  kind: "creator_directive_reconciliation",
  reason: "Two active creator directives conflict.",
  refs: {
    target_type: "creator_directive_reconciliation",
    directive_ids: ["cd_1", "cd_2"],
    members: [
      { id: "cd_1", family_key: {}, scope_equivalence: {} },
      { id: "cd_2", family_key: {}, scope_equivalence: {} },
    ],
    judgment: {
      member_ids: ["cd_1", "cd_2"],
      verdict: "conflicting",
      resolution: "supersede_to_survivor",
      survivor_id: "cd_1",
      loser_ids: ["cd_2"],
      confidence: "high",
      rationale: "structural conflict",
    },
  },
});

const beliefRevision = row({
  id: 421,
  kind: "belief_revision",
  reason: "Proposed weakening from the belief reviser.",
  refs: {
    target_type: "semantic_node",
    target_id: "semn_c",
    invalidated_edge_id: "seme_1",
    dependency_path_edge_ids: ["seme_1"],
    surviving_support_edge_ids: [],
    evidence_episode_ids: ["ep_1"],
  },
});

const correction = row({
  id: 423,
  kind: "correction",
  reason: "Operator correction: stale phone number.",
  refs: {
    target_type: "semantic_node",
    target_id: "semn_stale",
    patch: { archived: true },
    operator_reason: "stale",
  },
});

const proposedPeriodNarrative =
  "This proposed narrative starts with a first-person account of the reviewable change, then keeps going long enough that the console should clamp it before expansion while still preserving the exact proposed text for the operator.";

const periodPatchReview = row({
  id: 430,
  kind: "identity_inconsistency",
  reason: "Autobiographical period narrative should be patched.",
  refs: {
    target_type: "autobiographical_period",
    target_id: "period_current",
    repair_op: "patch",
    patch: {
      narrative: proposedPeriodNarrative,
      themes: ["continuity", "repair"],
      key_episode_ids: ["ep_1", "ep_2"],
      future_text_list: ["structural patch field", "future patch list"],
      disclosure_label: {
        disclosure_class: "relationship_private",
        origin_audience_entity_ids: ["ent_operator"],
        private_to_entity_ids: ["ent_operator"],
      },
      internal_blob: {
        internal_blob_key: "patch blob value must stay hidden",
      },
      embedding: [0.123, 0.456, 0.789],
    },
    evidence_episode_ids: ["ep_1"],
  },
});

const newInsightInsert = row({
  id: 431,
  kind: "new_insight",
  reason: "New reflected insight needs review.",
  refs: {
    node_ids: ["semn_new_insight"],
    episode_ids: ["ep_1"],
    evidence_cluster_key: "cluster:new",
    evidence_cluster_size: 1,
    reflector_pending_insight: {
      target: {
        mode: "insert",
        node: {
          id: "semn_new_insight",
          kind: "claim",
          label: "Borg values rollback planning",
          description:
            "The evidence-backed reasoning says Borg repeatedly treats rollback plans as part of deployment care.",
          aliases: [],
          confidence: 0.73,
          source_episode_ids: ["ep_1"],
          review_tags: ["rollback evidence", "deployment care"],
          created_at: now,
          updated_at: now,
          last_verified_at: now,
          embedding: [0.111, 0.222, 0.333],
          archived: false,
          superseded_by: null,
          status: "candidate_reviewable",
          corrected_by: null,
          superseded_at: null,
          internal_payload: {
            internal_payload_key: "node payload value must stay hidden",
          },
          audit_vector: [0.9, 0.8],
        },
      },
      candidate_support_edges: [],
      evidence_cluster: {
        key: "cluster:new",
        episode_ids: ["ep_1"],
        size: 1,
      },
    },
  },
});

const newInsightUpdate = row({
  id: 432,
  kind: "new_insight",
  reason: "Existing insight should be updated.",
  refs: {
    node_ids: ["semn_update_insight"],
    episode_ids: ["ep_2"],
    evidence_cluster_key: "cluster:update",
    evidence_cluster_size: 1,
    reflector_pending_insight: {
      target: {
        mode: "update",
        node_id: "semn_update_insight",
        patch: {
          description: "Updated evidence says rollback planning is a durable preference.",
          confidence: 0.82,
          source_episode_ids: ["ep_2"],
          revision_notes: ["structural update note"],
          status: "needs-review-status",
          last_verified_at: now,
          embedding: [0.444, 0.555, 0.666],
          archived: false,
          internal_payload: {
            update_payload_key: "update payload value must stay hidden",
          },
        },
      },
      candidate_support_edges: [],
      evidence_cluster: {
        key: "cluster:update",
        episode_ids: ["ep_2"],
        size: 1,
      },
    },
  },
});

const commitmentReview = row({
  id: 427,
  kind: "commitment_reconciliation",
  reason: "Two commitments overlap.",
  refs: {
    target_type: "commitment_reconciliation",
    commitment_ids: ["cmt_1", "cmt_2"],
    members: [
      { id: "cmt_1", kind: "promise", type: "promise", directive_family: "weekly" },
      { id: "cmt_2", kind: "promise", type: "promise", directive_family: "weekly" },
    ],
    judgment: {
      commitment_ids: ["cmt_1", "cmt_2"],
      resolution: "keep_independent",
      survivor_commitment_id: null,
      superseded_commitment_ids: [],
      reason: "both are distinct",
    },
  },
});

const resolvedDuplicate = row({
  id: 418,
  kind: "duplicate",
  reason: "Two semantic nodes duplicate the same fact.",
  refs: { node_ids: ["semn_a", "semn_b"] },
  resolved_at: now - 60_000,
  resolution: "dismiss",
});

const nodes: Record<string, SemanticNodeDetail> = {
  semn_a: {
    id: "semn_a",
    kind: "claim",
    label: "deadline = jun 14",
    display_label: "deadline = jun 14",
    description:
      "1997 science-fiction film praised in the thread for aging well because its gene-paranoia now reads as prophetic.",
    domain: "cinema",
    aliases: [],
    confidence: 0.71,
    status: "contested",
    source_episode_ids: ["ep_1", "ep_2"],
    source_count: 2,
    origin_audience_refs: [{ value: "ent_operator", id: "ent_operator", label: "operator" }],
    disclosure_class: "relationship_private",
    disclosure_label: {
      disclosure_class: "relationship_private",
      origin_audience_entity_ids: ["ent_operator"],
      private_to_entity_ids: ["ent_operator"],
      public_to_entity_ids: [],
    },
    created_at: now,
    updated_at: now,
  },
  semn_b: {
    id: "semn_b",
    kind: "claim",
    label: "deadline = jun 21",
    display_label: "deadline = jun 21",
    description: "",
    domain: null,
    aliases: [],
    confidence: 0.66,
    status: "contested",
    source_episode_ids: ["ep_2", "ep_3", "ep_4", "ep_5", "ep_6", "ep_7"],
    source_count: 6,
    created_at: now,
    updated_at: now,
  },
  semn_update_insight: {
    id: "semn_update_insight",
    kind: "claim",
    label: "Rollback planning matters",
    display_label: "Rollback planning matters",
    description: "Current evidence only says rollback planning matters sometimes.",
    domain: "operations",
    aliases: [],
    confidence: 0.51,
    status: "active",
    source_episode_ids: ["ep_3"],
    source_count: 1,
    created_at: now,
    updated_at: now,
  },
};

const edges: Record<string, SemanticEdgeDetail> = {
  seme_contradiction: {
    id: "seme_contradiction",
    from_node_id: "semn_a",
    to_node_id: "semn_b",
    relation: "contradicts",
    confidence: 0.4,
    evidence_episode_ids: ["ep_4", "ep_8"],
    source_count: 2,
    valid_from: now,
    valid_to: null,
    invalidated_at: null,
    invalidated_reason: null,
  },
};

const episodes: Record<string, EpisodeDetail> = Object.fromEntries(
  Array.from({ length: 8 }, (_, index) => {
    const id = `ep_${index + 1}`;
    return [
      id,
      {
        id,
        title: `Evidence ${index + 1}`,
        narrative: `Episode ${index + 1} narrative with enough context to decide the review.`,
        start_time: now - index * 60_000,
        end_time: now - index * 60_000,
        participant_refs: [],
        origin_audience_refs: [],
        disclosure_class: "public",
      } satisfies EpisodeDetail,
    ];
  }),
);

const directives: CreatorDirective[] = [
  {
    id: "cd_1",
    kind: "response_policy",
    text: "Use Europe/Warsaw for scheduling.",
    canonical_fact: null,
    operational_directive: "Use Europe/Warsaw for scheduling.",
    activation_scope: "same_as_disclosure",
    content_scope: "operator_only",
    mention_policy: "answer_if_asked",
    status: "active",
    subject_kind: "system",
    subject_entity_id: null,
    subject_entity_name: null,
    priority: 0.9,
    superseded_by_id: null,
    revoked_reason: null,
    created_at: now,
    updated_at: now,
  },
  {
    id: "cd_2",
    kind: "response_policy",
    text: "Schedule everything in UTC.",
    canonical_fact: null,
    operational_directive: "Schedule everything in UTC.",
    activation_scope: "same_as_disclosure",
    content_scope: "public",
    mention_policy: "answer_if_asked",
    status: "active",
    subject_kind: "system",
    subject_entity_id: null,
    subject_entity_name: null,
    priority: 0.4,
    superseded_by_id: null,
    revoked_reason: null,
    created_at: now - 1000,
    updated_at: now - 1000,
  },
];

const commitments: Commitment[] = [];

const identity: IdentityResponse = {
  values: [],
  goals: [],
  traits: [],
  open_questions: [],
  growth_markers: [],
  periods: [
    {
      id: "period_current",
      label: "2026-Q2",
      start_ts: now - 10_000,
      end_ts: null,
      narrative: "Current period narrative before review.",
      key_episode_ids: ["ep_3"],
      disclosure_label: {
        disclosure_class: "relationship_private",
        origin_audience_entity_ids: ["ent_operator"],
        private_to_entity_ids: ["ent_operator"],
      },
      themes: ["continuity"],
      created_at: now - 10_000,
      last_updated: now - 1_000,
    },
  ],
  open_question_events: [],
};

function json(body: unknown, status = 200): Response {
  return new Response(JSON.stringify(body), {
    status,
    headers: { "Content-Type": "application/json" },
  });
}

function deferred<T>() {
  let resolve!: (value: T) => void;
  const promise = new Promise<T>((done) => {
    resolve = done;
  });
  return { promise, resolve };
}

function renderReviews(
  options: {
    cdError?: boolean;
    failingEpisodes?: string[];
    mainRows?: ReviewRow[];
    correctionRows?: ReviewRow[];
    identity?: IdentityResponse;
  } = {},
) {
  const failingEpisodes = new Set(options.failingEpisodes ?? []);
  const requests: Array<{ url: string; method: string; body: unknown }> = [];
  const mainRows = options.mainRows ?? [
    contradiction,
    directiveReview,
    beliefRevision,
    correction,
    commitmentReview,
    resolvedDuplicate,
  ];
  const correctionRows = options.correctionRows ?? [correction];
  vi.spyOn(globalThis, "fetch").mockImplementation(async (input, init) => {
    const url = String(input);
    const method = init?.method ?? "GET";
    requests.push({
      url,
      method,
      body: init?.body === undefined ? null : JSON.parse(String(init.body)),
    });

    if (url === "/api/reviews?open_only=false") {
      return json({
        rows: mainRows,
      });
    }
    if (url === "/api/correction/reviews") {
      return json({ rows: correctionRows });
    }
    if (url.startsWith("/api/semantic/nodes/")) {
      const id = decodeURIComponent(url.slice("/api/semantic/nodes/".length));
      return json({ node: nodes[id] });
    }
    if (url.startsWith("/api/semantic/edges/")) {
      const id = decodeURIComponent(url.slice("/api/semantic/edges/".length));
      return edges[id] === undefined ? json({ message: "not found" }, 404) : json({ edge: edges[id] });
    }
    if (url.startsWith("/api/episodes/")) {
      const id = decodeURIComponent(url.slice("/api/episodes/".length));
      if (failingEpisodes.has(id) || episodes[id] === undefined) {
        return json({ message: "not found" }, 404);
      }
      return json({ episode: episodes[id] });
    }
    if (url.startsWith("/api/creator-directives")) {
      return json({ directives });
    }
    if (url.startsWith("/api/commitments")) {
      return json({ commitments });
    }
    if (url === "/api/identity") {
      return json(options.identity ?? identity);
    }
    if (url === "/api/reviews/412" && method === "PATCH") {
      return json({ ...contradiction, resolved_at: now, resolution: "supersede" });
    }
    if (url === "/api/reviews/425/creator-directive-reconciliation" && method === "POST") {
      if (options.cdError) {
        return json({ message: "creator directive reconciliation changed before apply" }, 409);
      }
      return json({ ...directiveReview, resolved_at: now, resolution: "accept" });
    }
    if (url === "/api/dream/review/421" && method === "PATCH") {
      return json({ ...beliefRevision, resolved_at: now, resolution: "dismiss" });
    }
    if (url === "/api/correction/semn_stale/why") {
      return json({
        target_type: "semantic_node",
        record: { id: "semn_stale", label: "stale phone", status: "active" },
        direct_edges: [],
      });
    }
    if (url === "/api/correction/reviews/423" && method === "PATCH") {
      return json({ ...correction, resolved_at: now, resolution: "accept" });
    }

    return json({ message: "not found" }, 404);
  });

  render(<ReviewsPage />);
  return { requests };
}

describe("Reviews page", () => {
  afterEach(() => {
    vi.restoreAllMocks();
  });

  it("maps kind colors and per-kind action sets from the server contract", () => {
    expect(reviewKindColor("contradiction")).toBe("var(--error-bright)");
    expect(reviewKindColor("duplicate")).toBe("var(--gold)");
    expect(reviewKindColor("correction")).toBe("var(--blue)");
    expect(actionsForReviewKind("belief_revision")).toEqual(["dismiss"]);
    expect(actionsForReviewKind("creator_directive_reconciliation")).toEqual(["supersede", "keep"]);
    expect(actionsForReviewKind("commitment_reconciliation")).toEqual(["dismiss", "reject", "accept", "keep"]);
    expect(actionsForReviewKind("skill_split")).toEqual(["accept", "reject"]);
  });

  it("merges correction feed duplicates by kind/id", () => {
    expect(mergeReviewRows([correction], [correction])).toHaveLength(1);
  });

  it("renders action sets for selected review kinds", async () => {
    renderReviews();

    expect(await screen.findByText("5 OPEN")).toBeTruthy();
    expect(screen.getByRole("button", { name: "SUPERSEDE -> WINNER" })).toBeTruthy();

    fireEvent.click(screen.getByText("Proposed weakening from the belief reviser."));
    expect(await screen.findByRole("button", { name: "DISMISS" })).toBeTruthy();
    expect(screen.queryByRole("button", { name: "ACCEPT" })).toBeNull();

    fireEvent.click(screen.getByText("Two active creator directives conflict."));
    expect(await screen.findByRole("button", { name: "SUPERSEDE FAMILY -> SURVIVOR" })).toBeTruthy();
    expect(screen.getByRole("button", { name: "KEEP ALL" })).toBeTruthy();
  });

  it("renders node-pair evidence context from semantic details and episodes", async () => {
    const { requests } = renderReviews();

    expect(await screen.findByText("5 OPEN")).toBeTruthy();
    expect(
      screen.getByText(
        "1997 science-fiction film praised in the thread for aging well because its gene-paranoia now reads as prophetic.",
      ),
    ).toBeTruthy();
    expect(screen.getByText("claim · cinema · confidence 0.71 · contested")).toBeTruthy();
    expect(screen.getByText(/recorded JUN 11 · updated JUN 11 · semn_a/)).toBeTruthy();
    expect(screen.getByText("origin operator")).toBeTruthy();
    expect(screen.getByText("relationship_private")).toBeTruthy();
    expect(screen.getByText("deadline = jun 21")).toBeTruthy();
    expect(screen.queryByText("second deadline")).toBeNull();

    expect(await screen.findByText("Evidence 1")).toBeTruthy();
    expect(screen.getByText("Episode 1 narrative with enough context to decide the review.")).toBeTruthy();
    expect(screen.getByText("2 more not shown")).toBeTruthy();
    expect(screen.getByText("contradicts edge · confidence 0.40 · recorded JUN 11")).toBeTruthy();

    const episodeRequests = requests.filter((request) => request.url.startsWith("/api/episodes/"));
    expect(episodeRequests.map((request) => request.url)).toEqual([
      "/api/episodes/ep_1",
      "/api/episodes/ep_2",
      "/api/episodes/ep_3",
      "/api/episodes/ep_4",
      "/api/episodes/ep_5",
      "/api/episodes/ep_6",
    ]);
    expect(episodeRequests.filter((request) => request.url === "/api/episodes/ep_2")).toHaveLength(1);
    expect(requests).toContainEqual(
      expect.objectContaining({ url: "/api/semantic/edges/seme_contradiction", method: "GET" }),
    );
  });

  it("renders patch proposed values with current comparison and expandable long text", async () => {
    renderReviews({
      mainRows: [periodPatchReview],
      correctionRows: [],
      identity,
    });

    expect(await screen.findAllByText("Autobiographical period narrative should be patched.")).toHaveLength(2);

    expect(await screen.findByText("Current period narrative before review.")).toBeTruthy();
    expect(screen.getByText(proposedPeriodNarrative)).toBeTruthy();
    expect(screen.getAllByText("continuity").length).toBeGreaterThan(0);
    expect(screen.getByText("repair")).toBeTruthy();
    expect(screen.getAllByText("ep_1").length).toBeGreaterThan(0);
    expect(screen.getByText("ep_2")).toBeTruthy();
    expect(screen.getByText("structural patch field")).toBeTruthy();
    expect(screen.getByText("future patch list")).toBeTruthy();
    expect(screen.getAllByText("relationship_private").length).toBeGreaterThan(0);
    expect(screen.queryByText("narrative, themes, key_episode_ids, disclosure_label, embedding")).toBeNull();
    expect(screen.queryByText("embedding")).toBeNull();
    expect(screen.queryByText("0.123")).toBeNull();
    expect(screen.queryByText("INTERNAL BLOB")).toBeNull();
    expect(screen.queryByText("internal_blob_key")).toBeNull();
    expect(screen.queryByText("patch blob value must stay hidden")).toBeNull();

    const toggle = screen.getByRole("button", { name: proposedPeriodNarrative }) as HTMLButtonElement;
    expect(toggle.getAttribute("aria-expanded")).toBe("false");
    fireEvent.click(toggle);
    expect(toggle.getAttribute("aria-expanded")).toBe("true");

    expect(await screen.findByText("Evidence 1")).toBeTruthy();
    expect(screen.getByText("Episode 1 narrative with enough context to decide the review.")).toBeTruthy();
  });

  it("renders pending insight content, confidence, and evidence while skipping vectors", async () => {
    renderReviews({
      mainRows: [newInsightInsert],
      correctionRows: [],
    });

    expect(await screen.findAllByText("New reflected insight needs review.")).toHaveLength(2);

    expect((await screen.findAllByText("Borg values rollback planning")).length).toBeGreaterThan(0);
    expect(
      screen.getByText(
        "The evidence-backed reasoning says Borg repeatedly treats rollback plans as part of deployment care.",
      ),
    ).toBeTruthy();
    expect(screen.getByText("claim")).toBeTruthy();
    expect(screen.getByText("0.73")).toBeTruthy();
    expect(screen.getByText("candidate_reviewable")).toBeTruthy();
    expect(screen.getByText("rollback evidence")).toBeTruthy();
    expect(screen.getByText("deployment care")).toBeTruthy();
    const proposedSection = screen.getByText("PROPOSED CONTENT").closest("section")!;
    const proposedText = proposedSection.textContent ?? "";
    const labelIndex = proposedText.indexOf("LABEL");
    const descriptionIndex = proposedText.indexOf("DESCRIPTION");
    const kindIndex = proposedText.indexOf("KIND");
    const confidenceIndex = proposedText.indexOf("CONFIDENCE");
    const statusIndex = proposedText.indexOf("STATUS");
    expect(labelIndex).toBeGreaterThanOrEqual(0);
    expect(descriptionIndex).toBeGreaterThanOrEqual(0);
    expect(kindIndex).toBeGreaterThanOrEqual(0);
    expect(confidenceIndex).toBeGreaterThanOrEqual(0);
    expect(statusIndex).toBeGreaterThanOrEqual(0);
    expect(labelIndex).toBeLessThan(descriptionIndex);
    expect(descriptionIndex).toBeLessThan(kindIndex);
    expect(kindIndex).toBeLessThan(confidenceIndex);
    expect(confidenceIndex).toBeLessThan(statusIndex);
    expect(await screen.findByText("Evidence 1")).toBeTruthy();
    expect(screen.getByText("Episode 1 narrative with enough context to decide the review.")).toBeTruthy();
    expect(screen.queryByText("embedding")).toBeNull();
    expect(screen.queryByText("0.111")).toBeNull();
    expect(screen.queryByText("semn_new_insight")).toBeNull();
    expect(screen.queryByText("CREATED AT")).toBeNull();
    expect(screen.queryByText("UPDATED AT")).toBeNull();
    expect(screen.queryByText("LAST VERIFIED AT")).toBeNull();
    expect(screen.queryByText("INTERNAL PAYLOAD")).toBeNull();
    expect(screen.queryByText("internal_payload_key")).toBeNull();
    expect(screen.queryByText("node payload value must stay hidden")).toBeNull();
    expect(screen.queryByText("AUDIT VECTOR")).toBeNull();
    expect(screen.queryByText("0.9")).toBeNull();
  });

  it("renders current-to-proposed comparison for pending insight updates", async () => {
    renderReviews({
      mainRows: [newInsightUpdate],
      correctionRows: [],
    });

    expect(await screen.findAllByText("Existing insight should be updated.")).toHaveLength(2);

    expect(await screen.findByText(/Rollback planning matters/)).toBeTruthy();
    expect(screen.getByText("Current evidence only says rollback planning matters sometimes.")).toBeTruthy();
    expect(screen.getByText("Updated evidence says rollback planning is a durable preference.")).toBeTruthy();
    expect(screen.getByText("structural update note")).toBeTruthy();
    expect(screen.getByText("needs-review-status")).toBeTruthy();
    expect(screen.getByText("0.51")).toBeTruthy();
    expect(screen.getByText("0.82")).toBeTruthy();
    expect(screen.queryByText("LABEL")).toBeNull();
    expect(screen.queryByText("KIND")).toBeNull();
    expect(screen.queryByText("LAST VERIFIED AT")).toBeNull();
    expect(screen.queryByText("INTERNAL PAYLOAD")).toBeNull();
    expect(screen.queryByText("update_payload_key")).toBeNull();
    expect(screen.queryByText("update payload value must stay hidden")).toBeNull();
    expect(screen.queryByText("0.444")).toBeNull();
  });

  it("falls back to proposed-only rendering when a current insight target is missing", async () => {
    const missingTargetInsight = row({
      id: 433,
      kind: "new_insight",
      reason: "Missing current insight should still render.",
      refs: {
        node_ids: ["semn_missing_current"],
        episode_ids: ["ep_1"],
        evidence_cluster_key: "cluster:missing",
        evidence_cluster_size: 1,
        reflector_pending_insight: {
          target: {
            mode: "update",
            node_id: "semn_missing_current",
            patch: {
              description: "Proposed-only description remains reviewable.",
              confidence: 0.61,
              source_episode_ids: ["ep_1"],
              last_verified_at: now,
              embedding: [0.9, 0.8, 0.7],
              archived: false,
            },
          },
          candidate_support_edges: [],
          evidence_cluster: {
            key: "cluster:missing",
            episode_ids: ["ep_1"],
            size: 1,
          },
        },
      },
    });

    renderReviews({
      mainRows: [missingTargetInsight],
      correctionRows: [],
    });

    expect(await screen.findAllByText("Missing current insight should still render.")).toHaveLength(2);

    expect(await screen.findByText("Proposed-only description remains reviewable.")).toBeTruthy();
    expect(screen.getAllByText("PROPOSED ONLY").length).toBeGreaterThan(0);
    expect(screen.queryByText("0.9")).toBeNull();
  });

  it("does not render stale node, edge, or episode evidence after switching selection", async () => {
    const reviewA = row({
      id: 501,
      kind: "contradiction",
      reason: "Review A stale pair.",
      refs: {
        node_ids: ["semn_stale_a", "semn_stale_b"],
        node_labels: ["stale A", "stale B"],
        edge_id: "seme_stale",
      },
    });
    const reviewB = row({
      id: 502,
      kind: "duplicate",
      reason: "Review B active pair.",
      refs: {
        node_ids: ["semn_current_a", "semn_current_b"],
        node_labels: ["current A", "current B"],
        edge_id: "seme_current",
      },
    });
    const baseNodeA = nodes.semn_a!;
    const baseNodeB = nodes.semn_b!;
    const baseEdge = edges.seme_contradiction!;
    const baseEpisodeA = episodes.ep_1!;
    const baseEpisodeB = episodes.ep_2!;
    const staleNodeA = {
      ...baseNodeA,
      id: "semn_stale_a",
      label: "stale A",
      display_label: "stale A",
      description: "A stale description that must not survive selection changes.",
      source_episode_ids: ["ep_stale"],
    } satisfies SemanticNodeDetail;
    const staleNodeB = {
      ...baseNodeB,
      id: "semn_stale_b",
      label: "stale B",
      display_label: "stale B",
      source_episode_ids: [],
    } satisfies SemanticNodeDetail;
    const currentNodeA = {
      ...baseNodeA,
      id: "semn_current_a",
      label: "current A",
      display_label: "current A",
      description: "B current description.",
      source_episode_ids: ["ep_current"],
    } satisfies SemanticNodeDetail;
    const currentNodeB = {
      ...baseNodeB,
      id: "semn_current_b",
      label: "current B",
      display_label: "current B",
      source_episode_ids: [],
    } satisfies SemanticNodeDetail;
    const staleEdge = {
      ...baseEdge,
      id: "seme_stale",
      from_node_id: "semn_stale_a",
      to_node_id: "semn_stale_b",
      confidence: 0.21,
      evidence_episode_ids: ["ep_stale_edge"],
    } satisfies SemanticEdgeDetail;
    const currentEdge = {
      ...baseEdge,
      id: "seme_current",
      from_node_id: "semn_current_a",
      to_node_id: "semn_current_b",
      relation: "related_to",
      confidence: 0.82,
      evidence_episode_ids: [],
    } satisfies SemanticEdgeDetail;
    const staleEpisode = {
      ...baseEpisodeA,
      id: "ep_stale",
      title: "Stale episode title",
      narrative: "Stale episode narrative.",
    } satisfies EpisodeDetail;
    const currentEpisode = {
      ...baseEpisodeB,
      id: "ep_current",
      title: "Current episode title",
      narrative: "Current episode narrative.",
    } satisfies EpisodeDetail;
    const staleNodeDefers = new Map<string, ReturnType<typeof deferred<Response>>>();
    const staleEdgeDeferred = deferred<Response>();
    const staleEpisodeDeferred = deferred<Response>();
    let staleEpisodeRequested = false;

    vi.spyOn(globalThis, "fetch").mockImplementation(async (input, init) => {
      const url = String(input);
      const method = init?.method ?? "GET";

      if (url === "/api/reviews?open_only=false") {
        return json({ rows: [reviewA, reviewB] });
      }
      if (url === "/api/correction/reviews") {
        return json({ rows: [] });
      }
      if (url.startsWith("/api/creator-directives")) {
        return json({ directives: [] });
      }
      if (url.startsWith("/api/commitments")) {
        return json({ commitments: [] });
      }
      if (url === "/api/semantic/nodes/semn_stale_a") {
        const pending = deferred<Response>();
        staleNodeDefers.set("semn_stale_a", pending);
        return pending.promise;
      }
      if (url === "/api/semantic/nodes/semn_stale_b") {
        const pending = deferred<Response>();
        staleNodeDefers.set("semn_stale_b", pending);
        return pending.promise;
      }
      if (url === "/api/semantic/edges/seme_stale") {
        return staleEdgeDeferred.promise;
      }
      if (url === "/api/episodes/ep_stale") {
        staleEpisodeRequested = true;
        return staleEpisodeDeferred.promise;
      }
      if (url === "/api/episodes/ep_stale_edge") {
        return json({ episode: { ...staleEpisode, id: "ep_stale_edge", title: "Stale edge episode" } });
      }
      if (url === "/api/semantic/nodes/semn_current_a") {
        return json({ node: currentNodeA });
      }
      if (url === "/api/semantic/nodes/semn_current_b") {
        return json({ node: currentNodeB });
      }
      if (url === "/api/semantic/edges/seme_current") {
        return json({ edge: currentEdge });
      }
      if (url === "/api/episodes/ep_current") {
        return json({ episode: currentEpisode });
      }

      return json({ message: `unexpected ${method} ${url}` }, 404);
    });

    render(<ReviewsPage />);

    expect(await screen.findByText("2 OPEN")).toBeTruthy();
    await waitFor(() => {
      expect(staleNodeDefers.size).toBe(2);
    });

    await act(async () => {
      staleNodeDefers.get("semn_stale_a")!.resolve(json({ node: staleNodeA }));
      staleNodeDefers.get("semn_stale_b")!.resolve(json({ node: staleNodeB }));
      staleEdgeDeferred.resolve(json({ edge: staleEdge }));
      await Promise.resolve();
    });
    expect(await screen.findByText("A stale description that must not survive selection changes.")).toBeTruthy();
    expect(screen.getByText("contradicts edge · confidence 0.21 · recorded JUN 11")).toBeTruthy();
    await waitFor(() => expect(staleEpisodeRequested).toBe(true));

    fireEvent.click(screen.getByText("Review B active pair."));

    expect(await screen.findByText("B current description.")).toBeTruthy();
    expect(screen.getByText("related_to edge · confidence 0.82 · recorded JUN 11")).toBeTruthy();
    expect(screen.queryByText("A stale description that must not survive selection changes.")).toBeNull();
    expect(screen.queryByText("contradicts edge · confidence 0.21 · recorded JUN 11")).toBeNull();
    expect(screen.queryByText("Stale edge episode")).toBeNull();

    await act(async () => {
      staleEpisodeDeferred.resolve(json({ episode: staleEpisode }));
      await Promise.resolve();
    });

    expect(await screen.findByText("Current episode title")).toBeTruthy();
    expect(screen.queryByText("Stale episode title")).toBeNull();
    expect(screen.queryByText("Stale episode narrative.")).toBeNull();
  });

  it("does not render stale proposed evidence or current target after switching selection", async () => {
    const staleReview = row({
      id: 601,
      kind: "new_insight",
      reason: "Stale proposed insight.",
      refs: {
        node_ids: ["semn_stale_update"],
        episode_ids: ["ep_stale_proposed"],
        evidence_cluster_key: "cluster:stale",
        evidence_cluster_size: 1,
        reflector_pending_insight: {
          target: {
            mode: "update",
            node_id: "semn_stale_update",
            patch: {
              description: "Stale proposed description.",
              confidence: 0.9,
              source_episode_ids: ["ep_stale_proposed"],
              last_verified_at: now,
              embedding: [0, 1, 0, 1],
              archived: false,
            },
          },
          candidate_support_edges: [],
          evidence_cluster: {
            key: "cluster:stale",
            episode_ids: ["ep_stale_proposed"],
            size: 1,
          },
        },
      },
    });
    const currentReview = row({
      id: 602,
      kind: "new_insight",
      reason: "Current proposed insight.",
      refs: {
        node_ids: ["semn_current_update"],
        episode_ids: ["ep_current_proposed"],
        evidence_cluster_key: "cluster:current",
        evidence_cluster_size: 1,
        reflector_pending_insight: {
          target: {
            mode: "update",
            node_id: "semn_current_update",
            patch: {
              description: "Current proposed description.",
              confidence: 0.8,
              source_episode_ids: ["ep_current_proposed"],
              last_verified_at: now,
              embedding: [1, 0, 1, 0],
              archived: false,
            },
          },
          candidate_support_edges: [],
          evidence_cluster: {
            key: "cluster:current",
            episode_ids: ["ep_current_proposed"],
            size: 1,
          },
        },
      },
    });
    const staleNode = {
      ...nodes.semn_update_insight!,
      id: "semn_stale_update",
      label: "Stale current node",
      display_label: "Stale current node",
      description: "Stale current description.",
    } satisfies SemanticNodeDetail;
    const currentNode = {
      ...nodes.semn_update_insight!,
      id: "semn_current_update",
      label: "Current node",
      display_label: "Current node",
      description: "Current node description.",
    } satisfies SemanticNodeDetail;
    const staleNodeDeferred = deferred<Response>();
    const staleEpisodeDeferred = deferred<Response>();

    vi.spyOn(globalThis, "fetch").mockImplementation(async (input, init) => {
      const url = String(input);
      const method = init?.method ?? "GET";

      if (url === "/api/reviews?open_only=false") {
        return json({ rows: [staleReview, currentReview] });
      }
      if (url === "/api/correction/reviews") {
        return json({ rows: [] });
      }
      if (url.startsWith("/api/creator-directives")) {
        return json({ directives: [] });
      }
      if (url.startsWith("/api/commitments")) {
        return json({ commitments: [] });
      }
      if (url === "/api/semantic/nodes/semn_stale_update") {
        return staleNodeDeferred.promise;
      }
      if (url === "/api/episodes/ep_stale_proposed") {
        return staleEpisodeDeferred.promise;
      }
      if (url === "/api/semantic/nodes/semn_current_update") {
        return json({ node: currentNode });
      }
      if (url === "/api/episodes/ep_current_proposed") {
        return json({
          episode: {
            ...episodes.ep_1!,
            id: "ep_current_proposed",
            title: "Current proposed episode",
            narrative: "Current proposed episode narrative.",
          },
        });
      }

      return json({ message: `unexpected ${method} ${url}` }, 404);
    });

    render(<ReviewsPage />);

    expect(await screen.findAllByText("Stale proposed insight.")).toHaveLength(2);
    expect(await screen.findByText("Stale proposed description.")).toBeTruthy();

    fireEvent.click(screen.getAllByText("Current proposed insight.")[0]!);

    expect(await screen.findByText("Current proposed description.")).toBeTruthy();
    expect(await screen.findByText("Current node description.")).toBeTruthy();

    await act(async () => {
      staleNodeDeferred.resolve(json({ node: staleNode }));
      staleEpisodeDeferred.resolve(
        json({
          episode: {
            ...episodes.ep_1!,
            id: "ep_stale_proposed",
            title: "Stale proposed episode",
            narrative: "Stale proposed episode narrative.",
          },
        }),
      );
      await Promise.resolve();
    });

    expect(await screen.findByText("Current proposed episode")).toBeTruthy();
    expect(screen.queryByText("Stale current description.")).toBeNull();
    expect(screen.queryByText("Stale proposed episode")).toBeNull();
    expect(screen.queryByText("Stale proposed episode narrative.")).toBeNull();
  });

  it("keeps node-pair degraded evidence quiet", async () => {
    renderReviews({ failingEpisodes: ["ep_1", "ep_2", "ep_3", "ep_4", "ep_5", "ep_6"] });

    expect(await screen.findByText("evidence unavailable")).toBeTruthy();

    fireEvent.click(screen.getByRole("button", { name: "RESOLVED 1" }));
    expect(await screen.findAllByText("Two semantic nodes duplicate the same fact.")).toHaveLength(2);

    await waitFor(() => expect(screen.queryByText(/edge · confidence/)).toBeNull());
  });

  it("requires generic winners and posts winner_node_id", async () => {
    const { requests } = renderReviews();

    expect(await screen.findAllByText("Two believed arena deadlines cannot both be true.")).toHaveLength(2);
    const supersede = screen.getByRole("button", { name: "SUPERSEDE -> WINNER" }) as HTMLButtonElement;
    expect(supersede.disabled).toBe(true);

    fireEvent.click(screen.getByText("deadline = jun 14"));
    await waitFor(() => expect(supersede.disabled).toBe(false));
    fireEvent.click(supersede);

    await waitFor(() =>
      expect(requests).toContainEqual(
        expect.objectContaining({
          url: "/api/reviews/412",
          method: "PATCH",
          body: { action: "supersede", winner_node_id: "semn_a" },
        }),
      ),
    );
  });

  it("posts creator-directive survivor_id through the distinct route", async () => {
    const { requests } = renderReviews();

    fireEvent.click(await screen.findByText("Two active creator directives conflict."));
    const supersede = await screen.findByRole("button", { name: "SUPERSEDE FAMILY -> SURVIVOR" });
    expect((supersede as HTMLButtonElement).disabled).toBe(true);

    fireEvent.click(screen.getByText("Use Europe/Warsaw for scheduling."));
    await waitFor(() => expect((supersede as HTMLButtonElement).disabled).toBe(false));
    fireEvent.click(supersede);

    await waitFor(() =>
      expect(requests).toContainEqual(
        expect.objectContaining({
          url: "/api/reviews/425/creator-directive-reconciliation",
          method: "POST",
          body: { action: "supersede", survivor_id: "cd_1" },
        }),
      ),
    );
  });

  it("routes belief revision dismiss through the dream review endpoint", async () => {
    const { requests } = renderReviews();

    fireEvent.click(await screen.findByText("Proposed weakening from the belief reviser."));
    fireEvent.click(await screen.findByRole("button", { name: "DISMISS" }));

    await waitFor(() =>
      expect(requests).toContainEqual(
        expect.objectContaining({
          url: "/api/dream/review/421",
          method: "PATCH",
          body: { action: "dismiss" },
        }),
      ),
    );
    expect(requests.some((request) => request.url === "/api/reviews/421")).toBe(false);
    expect(screen.queryByRole("button", { name: "DISMISS" })).toBeNull();
  });

  it("routes correction accept through the correction review endpoint", async () => {
    const { requests } = renderReviews();

    fireEvent.click(await screen.findByText("Operator correction: stale phone number."));
    fireEvent.change(screen.getByPlaceholderText("optional note"), { target: { value: "apply it" } });
    fireEvent.click(screen.getByRole("button", { name: "ACCEPT" }));

    await waitFor(() =>
      expect(requests).toContainEqual(
        expect.objectContaining({
          url: "/api/correction/reviews/423",
          method: "PATCH",
          body: { action: "accept", note: "apply it" },
        }),
      ),
    );
    expect(requests.some((request) => request.url === "/api/reviews/423")).toBe(false);
  });

  it("surfaces server errors verbatim", async () => {
    renderReviews({ cdError: true });

    fireEvent.click(await screen.findByText("Two active creator directives conflict."));
    fireEvent.click(screen.getByRole("button", { name: "KEEP ALL" }));

    expect(await screen.findByText("creator directive reconciliation changed before apply")).toBeTruthy();
  });

  it("renders correction why provenance inline", async () => {
    renderReviews();

    fireEvent.click(await screen.findByText("Operator correction: stale phone number."));
    fireEvent.click(screen.getByRole("button", { name: "WHY?" }));

    expect(await screen.findByText("WHY")).toBeTruthy();
    expect(screen.getByText("target_type")).toBeTruthy();
    expect(screen.getByText("semantic_node")).toBeTruthy();
  });

  it("filters by kind and open/resolved state, and hides actions for resolved reviews", async () => {
    renderReviews();

    expect(await screen.findByRole("button", { name: "ALL 5" })).toBeTruthy();
    fireEvent.click(screen.getByRole("button", { name: "CONTRADICTION 1" }));
    expect(screen.getAllByText("Two believed arena deadlines cannot both be true.")).toHaveLength(2);
    expect(screen.queryByText("Two active creator directives conflict.")).toBeNull();

    fireEvent.click(screen.getByRole("button", { name: "ALL 5" }));
    fireEvent.click(screen.getByRole("button", { name: "RESOLVED 1" }));
    expect(await screen.findAllByText("Two semantic nodes duplicate the same fact.")).toHaveLength(2);
    expect(
      screen.getByText(
        (_, element) =>
          element?.className === "review-resolved-banner" &&
          element.textContent?.includes("DISMISS") === true,
      ),
    ).toBeTruthy();
    expect(screen.queryByRole("button", { name: "DISMISS" })).toBeNull();
  });
});
