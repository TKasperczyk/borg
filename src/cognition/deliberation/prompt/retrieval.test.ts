import { describe, expect, it } from "vitest";

import {
  createEpisodeFixture,
  createRetrievalScoreFixture,
} from "../../../offline/test-support.js";
import type { SemanticEdge, SemanticNode } from "../../../memory/semantic/index.js";
import type {
  EvidenceItem,
  RetrievalConfidence,
  RetrievedEpisode,
  RetrievedSemantic,
} from "../../../retrieval/index.js";
import { publicMemoryDisclosureLabel } from "../../../retrieval/index.js";
import { ManualClock } from "../../../util/clock.js";
import { DEFAULT_PLAN_REQUESTED_VERIFICATION_MEMBERSHIP_TOKEN_BUDGET } from "../constants.js";
import {
  PLAN_REQUESTED_VERIFICATION_MEMBERSHIP_BUDGET_MARKER,
  summarizeRetrievalConfidence,
  renderPlanRequestedVerificationNotCompleted,
  renderPlanRequestedVerificationRetrieval,
  summarizeRetrievedEpisodes,
  summarizeRetrievedEvidence,
  summarizeSemanticContext,
} from "./retrieval.js";

function verificationEvidence(id: string, text: string): EvidenceItem {
  return {
    id,
    source: "episodic",
    recallIntentId: "intent-verification",
    score: 0.8,
    text,
    matchedTerms: [],
  } as unknown as EvidenceItem;
}

function makeRetrievalConfidence(
  overrides: Partial<RetrievalConfidence> = {},
): RetrievalConfidence {
  return {
    overall: overrides.overall ?? 0,
    evidenceStrength: overrides.evidenceStrength ?? 0,
    coverage: overrides.coverage ?? 0,
    sourceDiversity: overrides.sourceDiversity ?? 0,
    contradictionPresent: overrides.contradictionPresent ?? false,
    sampleSize: overrides.sampleSize ?? 0,
    coverageExpected: overrides.coverageExpected ?? 5,
    diversitySources: overrides.diversitySources ?? 0,
    diversitySampleSize: overrides.diversitySampleSize ?? 0,
  };
}

function makeNode(overrides: Partial<SemanticNode> = {}): SemanticNode {
  return {
    id: overrides.id ?? ("semn_aaaaaaaaaaaaaaaa" as SemanticNode["id"]),
    kind: overrides.kind ?? "proposition",
    label: overrides.label ?? "Atlas claim",
    description: overrides.description ?? "A claim about Atlas deployment state.",
    domain: overrides.domain ?? null,
    aliases: overrides.aliases ?? [],
    observation_metadata: overrides.observation_metadata ?? null,
    confidence: overrides.confidence ?? 0.7,
    source_episode_ids:
      overrides.source_episode_ids ??
      (["ep_aaaaaaaaaaaaaaaa"] as SemanticNode["source_episode_ids"]),
    created_at: overrides.created_at ?? 0,
    updated_at: overrides.updated_at ?? 0,
    last_verified_at: overrides.last_verified_at ?? 0,
    embedding: overrides.embedding ?? Float32Array.from([1, 0, 0, 0]),
    archived: overrides.archived ?? false,
    superseded_by: overrides.superseded_by ?? null,
    status: overrides.status ?? "active",
    corrected_by: overrides.corrected_by ?? null,
    superseded_at: overrides.superseded_at ?? null,
  };
}

function makeClosedEdge(overrides: Partial<SemanticEdge> = {}): SemanticEdge {
  return {
    id: overrides.id ?? ("seme_aaaaaaaaaaaaaaaa" as SemanticEdge["id"]),
    from_node_id:
      overrides.from_node_id ?? ("semn_aaaaaaaaaaaaaaaa" as SemanticEdge["from_node_id"]),
    to_node_id: overrides.to_node_id ?? ("semn_bbbbbbbbbbbbbbbb" as SemanticEdge["to_node_id"]),
    relation: overrides.relation ?? "supports",
    confidence: overrides.confidence ?? 0.7,
    evidence_episode_ids:
      overrides.evidence_episode_ids ??
      (["ep_aaaaaaaaaaaaaaaa"] as SemanticEdge["evidence_episode_ids"]),
    created_at: overrides.created_at ?? Date.UTC(2024, 0, 1),
    last_verified_at: overrides.last_verified_at ?? Date.UTC(2024, 0, 1),
    valid_from: overrides.valid_from ?? Date.UTC(2024, 0, 1),
    valid_to: overrides.valid_to ?? Date.UTC(2024, 0, 10),
    invalidated_at: overrides.invalidated_at ?? Date.UTC(2024, 0, 12),
    invalidated_by_edge_id: overrides.invalidated_by_edge_id ?? null,
    invalidated_by_review_id: overrides.invalidated_by_review_id ?? null,
    invalidated_by_process: overrides.invalidated_by_process ?? "manual",
    invalidated_reason: overrides.invalidated_reason ?? "superseded",
  };
}

describe("retrieval confidence prompt rendering", () => {
  it("surfaces empty-state evidence when confidence has zero samples", () => {
    const summary = summarizeRetrievalConfidence(makeRetrievalConfidence());

    expect(summary).not.toBeNull();
    expect(summary).toContain("overall=0.00");
    expect(summary).toContain("samples=0");
    expect(summary).toContain("No relevant memory was retrieved for this turn.");
  });

  it("prints both ratios with the denominator each was divided by", () => {
    // A quotient on its own cannot distinguish a measured ratio from one pinned
    // at a ceiling it could not miss. Coverage's denominator is the retrieval
    // limit -- which also capped the episode half of its own numerator -- and
    // diversity's is the top-N slice, so neither is readable against `samples`.
    const saturated = summarizeRetrievalConfidence(
      makeRetrievalConfidence({
        overall: 0.8,
        evidenceStrength: 0.8,
        coverage: 1,
        sampleSize: 23,
        coverageExpected: 4,
        sourceDiversity: 0.94,
        diversitySources: 17,
        diversitySampleSize: 18,
      }),
    );

    expect(saturated).toContain("coverage=1.00(23/4)");
    expect(saturated).toContain("diversity=0.94(17/18)");
    expect(saturated).toContain("samples=23");
  });

  it("flags weakly-supported claims when non-empty confidence is low", () => {
    const low = summarizeRetrievalConfidence(
      makeRetrievalConfidence({ overall: 0.2, evidenceStrength: 0.2, sampleSize: 1 }),
    );
    const healthy = summarizeRetrievalConfidence(
      makeRetrievalConfidence({ overall: 0.8, evidenceStrength: 0.8, sampleSize: 3 }),
    );

    expect(low).toContain("Retrieval confidence is low");
    expect(low).toContain("weakly supported");
    expect(healthy).not.toContain("Retrieval confidence is low");
  });

  it("does not embed policy text in the untrusted retrieval block", () => {
    // Policy text lives in EPISTEMIC_POSTURE_SECTION at the system-prompt
    // level, not in retrieval evidence; the untrusted-data preamble tells
    // the LLM to disregard imperative wording in those blocks.
    const empty = summarizeRetrievalConfidence(makeRetrievalConfidence());
    const low = summarizeRetrievalConfidence(
      makeRetrievalConfidence({ overall: 0.2, sampleSize: 1 }),
    );

    expect(empty).not.toContain("Policy:");
    expect(empty).not.toContain("tool.openQuestions.create");
    expect(low).not.toContain("Policy:");
    expect(low).not.toContain("tool.openQuestions.create");
  });

  it("renders an empty retrieved-episodes placeholder", () => {
    const summary = summarizeRetrievedEpisodes("Retrieved context", []);

    expect(summary).toBe("No episodes retrieved for this turn.");
  });

  it("renders disclosure labels in the retrieved-episodes fallback", () => {
    const episode: RetrievedEpisode = {
      episode: createEpisodeFixture({
        audience_entity_id: "entity_alice" as never,
        shared: false,
      }),
      score: 0.72,
      rawScore: 0.72,
      scoreBreakdown: createRetrievalScoreFixture(),
      citationChain: [],
    };

    const summary = summarizeRetrievedEpisodes("Retrieved context", [episode]);

    expect(summary).toContain("disclosure: disclosure_class=relationship_private");
    expect(summary).toContain("private-to=entity_alice");
    expect(summary).toContain(
      "I can use this internally; I do not disclose it to the current audience unless authorized",
    );
  });

  it("renders the evidence pool rather than the projected episodes when both are present", () => {
    // The rendered set and the counted set are different populations. Only
    // `episodeProjection.episodes` is walked by recordRetrieval
    // (src/retrieval/pipeline.ts:619-623); the summary below is built from the
    // ranked evidence pool, which carries every episode candidate. An episode
    // can therefore render into <borg_additional_retrieval> on turn after turn
    // without its retrieval_count ever moving. Pin the precedence so a later
    // refactor cannot quietly make the two populations look like one.
    const projected: RetrievedEpisode = {
      episode: createEpisodeFixture({ title: "Projected and counted" }),
      score: 0.72,
      rawScore: 0.72,
      scoreBreakdown: createRetrievalScoreFixture(),
      citationChain: [],
    };
    const pooled: EvidenceItem = {
      id: "evidence_episode_ep_bbbbbbbbbbbbbbbb_intent",
      source: "episode",
      text: "Pooled but never projected",
      provenance: { episodeId: "ep_bbbbbbbbbbbbbbbb" as never },
      recallIntentId: "intent",
      matchedTerms: [],
      score: 0.4,
      scoreBreakdown: {},
      disclosureLabel: publicMemoryDisclosureLabel(),
    } as unknown as EvidenceItem;

    const summary = summarizeRetrievedEvidence(
      "Additional retrieval",
      { evidence: [pooled], episodes: [projected] },
      1_000,
    );

    expect(summary).toContain("Pooled but never projected");
    expect(summary).not.toContain("Projected and counted");
  });

  it("renders disclosure labels on episode evidence items", () => {
    const evidence: EvidenceItem = {
      id: "evidence_episode_ep_aaaaaaaaaaaaaaaa_intent",
      source: "episode",
      text: "Alice private planning: private launch details.",
      provenance: {
        episodeId: "ep_aaaaaaaaaaaaaaaa" as never,
      },
      recallIntentId: "intent",
      matchedTerms: [],
      score: 0.8,
      scoreBreakdown: {},
      disclosureLabel: {
        disclosureClass: "relationship_private",
        originAudienceEntityIds: ["entity_alice" as never],
        privateToEntityIds: ["entity_alice" as never],
        publicToEntityIds: [],
      },
    };

    const summary = summarizeRetrievedEvidence(
      "Retrieved context",
      { evidence: [evidence] },
      1_000,
    );

    expect(summary).toContain("disclosure_class=relationship_private");
    expect(summary).toContain("private-to=entity_alice");
    expect(summary).toContain(
      "I can use this internally; I do not disclose it to the current audience unless authorized",
    );
  });

  it("renders disclosure labels on open-question evidence items", () => {
    const evidence: EvidenceItem = {
      id: "evidence_open_question_oq_aaaaaaaaaaaaaaaa_intent",
      source: "open_question",
      text: "Should I ask Alice about the private launch timing?",
      provenance: {
        openQuestionId: "oq_aaaaaaaaaaaaaaaa" as never,
      },
      recallIntentId: "intent",
      matchedTerms: [],
      score: 0.8,
      scoreBreakdown: {},
      disclosureLabel: {
        disclosureClass: "relationship_private",
        originAudienceEntityIds: ["entity_alice" as never],
        privateToEntityIds: ["entity_alice" as never],
        publicToEntityIds: [],
      },
    };

    const summary = summarizeRetrievedEvidence(
      "Retrieved context",
      { evidence: [evidence] },
      1_000,
    );

    expect(summary).toContain("Should I ask Alice about the private launch timing?");
    expect(summary).toContain("disclosure_class=relationship_private");
    expect(summary).toContain("private-to=entity_alice");
    expect(summary).toContain(
      "I can use this internally; I do not disclose it to the current audience unless authorized",
    );
  });

  it("renders disclosure labels on open-question fallback rows", () => {
    const summary = summarizeRetrievedEvidence(
      "Retrieved context",
      {
        episodes: [],
        semantic: null,
        openQuestions: [
          {
            id: "oq_aaaaaaaaaaaaaaaa",
            question: "Should I ask Alice about the private launch timing?",
            urgency: 0.72,
            audience_entity_id: "entity_alice" as never,
          },
        ],
      },
      1_000,
    );

    expect(summary).toContain("Should I ask Alice about the private launch timing?");
    expect(summary).toContain("disclosure_class=relationship_private");
    expect(summary).toContain("private-to=entity_alice");
    expect(summary).toContain(
      "I can use this internally; I do not disclose it to the current audience unless authorized",
    );
  });

  it("renders partial-source metadata on semantic evidence items", () => {
    const evidence: EvidenceItem = {
      id: "evidence_semantic_node_semn_aaaaaaaaaaaaaaaa_intent",
      source: "semantic_node",
      text: "Atlas mixed visibility: Atlas node backed by visible and hidden sources.",
      provenance: {
        nodeId: "semn_aaaaaaaaaaaaaaaa" as never,
      },
      recallIntentId: "intent",
      matchedTerms: [],
      score: 0.8,
      scoreBreakdown: {},
      source_episode_ids: ["ep_aaaaaaaaaaaaaaaa" as never],
      partial_source_visibility: true,
      source_visibility_fraction: 0.5,
    };

    const summary = summarizeRetrievedEvidence(
      "Retrieved context",
      { evidence: [evidence] },
      1_000,
    );

    expect(summary).toContain("sources=ep_aaaaaaaaaaaaaaaa");
    expect(summary).toContain("partial_sources=true");
    expect(summary).toContain("visible_fraction=0.50");
  });

  it("renders semantic disclosure labels with source-detail notes on evidence items", () => {
    const evidence: EvidenceItem = {
      id: "evidence_semantic_node_semn_private_intent",
      source: "semantic_node",
      text: "Alice private semantic claim.",
      provenance: {
        nodeId: "semn_aaaaaaaaaaaaaaaa" as never,
      },
      recallIntentId: "intent",
      matchedTerms: [],
      score: 0.8,
      scoreBreakdown: {},
      source_episode_ids: ["ep_aaaaaaaaaaaaaaaa" as never],
      disclosureLabel: {
        disclosureClass: "relationship_private",
        originAudienceEntityIds: ["ent_alice" as never],
        privateToEntityIds: ["ent_alice" as never],
        publicToEntityIds: [],
      },
    };

    const summary = summarizeRetrievedEvidence(
      "Retrieved context",
      { evidence: [evidence] },
      1_000,
    );

    expect(summary).toContain("disclosure_class=relationship_private");
    expect(summary).toContain("private-to=ent_alice");
    expect(summary).toContain(
      "supported by private source episodes; I can use this internally; I do not reveal source details to the current audience unless authorized",
    );
  });
});

describe("plan-requested compact terminal retrieval", () => {
  it("carries requested payload exactly, accounting after XML escaping", () => {
    const payload = `<verified attr="x">${'&<>"'.repeat(40)}</verified>`;
    const rendered = renderPlanRequestedVerificationRetrieval(
      {
        evidence: [verificationEvidence("evidence:exact", payload)],
        episodes: [],
        semantic: {
          matched_node_ids: [],
          matched_nodes: [],
          supports: [],
          contradicts: [],
          categories: [],
          support_hits: [],
          causal_hits: [],
          contradiction_hits: [],
          category_hits: [],
        },
        open_questions: [],
      } as never,
      2_000,
    );

    expect(rendered).toContain('handle="evidence:exact"');
    expect(rendered).toContain('payload_status="exact"');
    expect(rendered).toContain(
      "disclosure_class=unknown origin_audience=none private-to=none public-to=none",
    );
    expect(rendered).toContain(
      `payload_total_chars="${JSON.stringify({ text: payload, matched_terms: [], image_label: null, image_origin_frame: null, image_unavailable_reason: null }).length}"`,
    );
    expect(rendered).toContain("&lt;verified attr=\\&quot;x\\&quot;&gt;");
    expect(rendered).not.toContain("HEAD+TAIL EXCERPT");
  });

  it("keeps fallback source handles even when unified evidence is also present", () => {
    const rendered = renderPlanRequestedVerificationRetrieval(
      {
        evidence: [verificationEvidence("evidence:mixed", "evidence payload")],
        episodes: [],
        semantic: {
          matched_node_ids: [],
          matched_nodes: [],
          supports: [],
          contradicts: [],
          categories: [],
          support_hits: [],
          causal_hits: [],
          contradiction_hits: [],
          category_hits: [],
        },
        open_questions: [
          {
            id: "oq_mixed",
            question: "What must still be checked?",
            status: "open",
            urgency: 0.8,
            source: "user",
            audience_entity_id: null,
            goal_id: null,
            resolution_note: null,
            abandoned_reason: null,
          },
        ],
      } as never,
      2_000,
    );

    expect(rendered).toContain('handle="evidence:mixed"');
    expect(rendered).toContain('handle="oq_mixed"');
    expect(rendered).toContain('rows_total="2"');
    expect(rendered).toContain("<omitted_count>0</omitted_count>");
  });

  it("keeps under-budget membership rows byte-stable", () => {
    const input = {
      evidence: [
        verificationEvidence("evidence:zeta", "first payload"),
        verificationEvidence("evidence:alpha", "second payload"),
      ],
      episodes: [],
      semantic: {
        matched_node_ids: [],
        matched_nodes: [],
        supports: [],
        contradicts: [],
        categories: [],
        support_hits: [],
        causal_hits: [],
        contradiction_hits: [],
        category_hits: [],
      },
      open_questions: [],
    } as never;
    const rendered = renderPlanRequestedVerificationRetrieval(input, 2_000);
    const renderedWithUnboundedMembership = renderPlanRequestedVerificationRetrieval(
      input,
      2_000,
      Number.MAX_SAFE_INTEGER,
    );
    const sourceRows = (value: string) => value.match(/^  <verification_source.*$/gm);

    expect(sourceRows(rendered)).toEqual(sourceRows(renderedWithUnboundedMembership));
    expect(rendered).toContain('complete_membership="true" rows_total="2"');
    expect(rendered).toContain("<omitted_count>0</omitted_count>");
    expect(rendered).not.toContain(`${PLAN_REQUESTED_VERIFICATION_MEMBERSHIP_BUDGET_MARKER}=\"`);
  });

  it("fully enumerates the 300-row commitment and goal scale at the default budget", () => {
    const rendered = renderPlanRequestedVerificationRetrieval(
      {
        evidence: Array.from({ length: 300 }, (_unused, index) =>
          verificationEvidence(`evidence:${index}`, "z".repeat(40)),
        ),
        episodes: [],
        semantic: {
          matched_node_ids: [],
          matched_nodes: [],
          supports: [],
          contradicts: [],
          categories: [],
          support_hits: [],
          causal_hits: [],
          contradiction_hits: [],
          category_hits: [],
        },
        open_questions: [],
      } as never,
      2_000,
    );

    const membershipTokens = Number(/membership_tokens="(\d+)"/.exec(rendered)?.[1]);
    expect(membershipTokens).toBe(43_923);
    expect(membershipTokens).toBeLessThan(
      DEFAULT_PLAN_REQUESTED_VERIFICATION_MEMBERSHIP_TOKEN_BUDGET,
    );
    expect(rendered).toContain('complete_membership="true" rows_total="300"');
    expect(rendered.match(/<verification_source /g)).toHaveLength(300);
    expect(rendered).toContain('handle="evidence:0"');
    expect(rendered).toContain('handle="evidence:299"');
    expect(rendered).toContain("<omitted_count>0</omitted_count>");
  });

  it("keeps every source handle and reports a structurally incomplete check instead of an excerpt", () => {
    const rendered = renderPlanRequestedVerificationRetrieval(
      {
        evidence: [
          verificationEvidence("evidence:one", "x".repeat(10_000)),
          verificationEvidence("evidence:two", "y".repeat(10_000)),
        ],
        episodes: [],
        semantic: {
          matched_node_ids: [],
          matched_nodes: [],
          supports: [],
          contradicts: [],
          categories: [],
          support_hits: [],
          causal_hits: [],
          contradiction_hits: [],
          category_hits: [],
        },
        open_questions: [],
      } as never,
      200,
    );

    expect(rendered).toContain('handle="evidence:one"');
    expect(rendered).toContain('handle="evidence:two"');
    expect(rendered.match(/payload_status="check_not_completed_budget"/g)).toHaveLength(2);
    expect(rendered).toContain('payload_included_chars="0"');
    expect(rendered).toContain("<omitted_count>0</omitted_count>");
    expect(rendered).not.toContain("HEAD+TAIL EXCERPT");
  });

  it("still completes affordable checks when the handle list alone exceeds the budget", () => {
    const rendered = renderPlanRequestedVerificationRetrieval(
      {
        evidence: Array.from({ length: 400 }, (_unused, index) =>
          verificationEvidence(`evidence:${index}`, "z".repeat(40)),
        ),
        episodes: [],
        semantic: {
          matched_node_ids: [],
          matched_nodes: [],
          supports: [],
          contradicts: [],
          categories: [],
          support_hits: [],
          causal_hits: [],
          contradiction_hits: [],
          category_hits: [],
        },
        open_questions: [],
      } as never,
      2_000,
      100_000,
    );

    const membershipTokens = Number(/membership_tokens="(\d+)"/.exec(rendered)?.[1]);
    const payloadTokens = Number(/payload_tokens_included="(\d+)"/.exec(rendered)?.[1]);
    expect(membershipTokens).toBeGreaterThan(2_000);
    expect(payloadTokens).toBeLessThanOrEqual(2_000);
    expect(rendered).toContain('rows_total="400"');
    expect((rendered.match(/payload_status="exact"/g) ?? []).length).toBeGreaterThan(0);
    expect(rendered).toContain("<omitted_count>0</omitted_count>");
  });

  it("enumerates an ordered prefix and flags the exact membership remainder", () => {
    const firstTwo = [
      verificationEvidence("evidence:zeta", "first payload"),
      verificationEvidence("evidence:alpha", "second payload"),
    ];
    const semantic = {
      matched_node_ids: [],
      matched_nodes: [],
      supports: [],
      contradicts: [],
      categories: [],
      support_hits: [],
      causal_hits: [],
      contradiction_hits: [],
      category_hits: [],
    };
    const firstTwoRendered = renderPlanRequestedVerificationRetrieval(
      {
        evidence: firstTwo,
        episodes: [],
        semantic,
        open_questions: [],
      } as never,
      2_000,
    );
    const firstTwoMembershipTokens = Number(
      /membership_tokens="(\d+)"/.exec(firstTwoRendered)?.[1],
    );
    const rendered = renderPlanRequestedVerificationRetrieval(
      {
        evidence: [
          ...firstTwo,
          verificationEvidence("evidence:middle", "third payload"),
          verificationEvidence("evidence:beta", "fourth payload"),
        ],
        episodes: [],
        semantic,
        open_questions: [],
      } as never,
      2_000,
      firstTwoMembershipTokens,
    );
    const enumeratedHandles = [...rendered.matchAll(/<verification_source handle="([^"]+)"/g)].map(
      (match) => match[1],
    );
    const sourceRows = (value: string) => value.match(/^  <verification_source.*$/gm);

    expect(enumeratedHandles).toEqual(["evidence:zeta", "evidence:alpha"]);
    expect(sourceRows(rendered)).toEqual(sourceRows(firstTwoRendered));
    expect(rendered).toContain('complete_membership="false" rows_total="4"');
    expect(rendered).toContain(`membership_target_tokens="${firstTwoMembershipTokens}"`);
    expect(rendered).toContain(`membership_tokens="${firstTwoMembershipTokens}"`);
    expect(rendered).toContain(`${PLAN_REQUESTED_VERIFICATION_MEMBERSHIP_BUDGET_MARKER}="2"`);
    expect(rendered).toContain(
      `<${PLAN_REQUESTED_VERIFICATION_MEMBERSHIP_BUDGET_MARKER}>2</${PLAN_REQUESTED_VERIFICATION_MEMBERSHIP_BUDGET_MARKER}>`,
    );
    expect(rendered).toContain("<omitted_count>2</omitted_count>");
    expect(rendered).not.toContain('handle="evidence:middle"');
    expect(rendered).not.toContain('handle="evidence:beta"');
    expect(rendered).toContain(
      "complete_membership=true means every one of rows_total handles and its structural fields is enumerated",
    );
    expect(rendered).toContain(
      `${PLAN_REQUESTED_VERIFICATION_MEMBERSHIP_BUDGET_MARKER}=N and its same-named marker carry the exact un-enumerated remainder`,
    );
    expect(rendered).toContain(
      "Payloads are priced against payload_target_tokens alone and never consume the membership budget; membership never consumes the payload budget",
    );
  });

  it("renders an unavailable plan-requested check with a handle and zero payload", () => {
    const rendered = renderPlanRequestedVerificationNotCompleted();

    expect(rendered).toContain('handle="plan:verification_steps"');
    expect(rendered).toContain('payload_status="check_not_completed_retrieval_unavailable"');
    expect(rendered).toContain('payload_included_chars="0"');
    expect(rendered).toContain('payload_total_chars="0"');
    expect(rendered).toContain('payload_json=""');
    expect(rendered).toContain("<check_not_completed_count>1</check_not_completed_count>");
  });
});

describe("semantic retrieval prompt rendering", () => {
  it("renders semantic disclosure labels in the semantic-context fallback", () => {
    const root = makeNode({
      id: "semn_aaaaaaaaaaaaaaaa" as SemanticNode["id"],
      label: "Alice private claim",
      description: "A semantic claim backed by Alice-private source episodes.",
      source_episode_ids: ["ep_aaaaaaaaaaaaaaaa" as never],
    }) as RetrievedSemantic["matched_nodes"][number];
    root.disclosureLabel = {
      disclosureClass: "relationship_private",
      originAudienceEntityIds: ["ent_alice" as never],
      privateToEntityIds: ["ent_alice" as never],
      publicToEntityIds: [],
    };
    const edge = makeClosedEdge({
      from_node_id: root.id,
      to_node_id: "semn_bbbbbbbbbbbbbbbb" as SemanticNode["id"],
      evidence_episode_ids: ["ep_aaaaaaaaaaaaaaaa" as never],
      valid_to: null,
      invalidated_at: null,
    }) as RetrievedSemantic["support_hits"][number]["edgePath"][number];
    edge.disclosureLabel = root.disclosureLabel;
    const support = makeNode({
      id: "semn_bbbbbbbbbbbbbbbb" as SemanticNode["id"],
      label: "Alice private support",
      description: "A supporting claim backed by Alice-private evidence.",
      source_episode_ids: ["ep_aaaaaaaaaaaaaaaa" as never],
    }) as RetrievedSemantic["support_hits"][number]["node"];
    support.disclosureLabel = root.disclosureLabel;

    const summary = summarizeSemanticContext(
      {
        as_of: null,
        matched_node_ids: [root.id],
        matched_nodes: [root],
        supports: [],
        contradicts: [],
        categories: [],
        support_hits: [
          {
            root_node_id: root.id,
            node: support,
            edgePath: [edge],
          },
        ],
        causal_hits: [],
        contradiction_hits: [],
        category_hits: [],
      },
      1_000,
    );

    expect(summary).toContain("disclosure_class=relationship_private");
    expect(summary).toContain("private-to=ent_alice");
    expect(summary).toContain(
      "supported by private source episodes; I can use this internally; I do not reveal source details to the current audience unless authorized",
    );
  });

  it("tags closed path edges for historical as-of context", () => {
    const root = makeNode({
      id: "semn_aaaaaaaaaaaaaaaa" as SemanticNode["id"],
      kind: "entity",
      label: "Atlas",
      description: "Atlas deployment service.",
    });
    const support = makeNode({
      id: "semn_bbbbbbbbbbbbbbbb" as SemanticNode["id"],
      label: "Rerun install",
      description: "Rerun pnpm install before deploying Atlas.",
    });
    const edge = makeClosedEdge({
      from_node_id: root.id,
      to_node_id: support.id,
    });
    const summary = summarizeSemanticContext(
      {
        as_of: Date.UTC(2024, 0, 5),
        matched_node_ids: [root.id],
        matched_nodes: [root],
        supports: [support],
        contradicts: [],
        categories: [],
        support_hits: [
          {
            root_node_id: root.id,
            node: support,
            edgePath: [edge],
          },
        ],
        causal_hits: [],
        contradiction_hits: [],
        category_hits: [],
      } satisfies RetrievedSemantic,
      1_000,
    );

    expect(summary).toContain("[valid 2024-01-01..2024-01-10, closed 2024-01-12]");
  });

  it("does not render closed path edges in current mode and marks historical direct matches", () => {
    const root = makeNode({
      id: "semn_aaaaaaaaaaaaaaaa" as SemanticNode["id"],
      kind: "entity",
      label: "Atlas",
      description: "Atlas deployment service.",
    });
    const support = makeNode({
      id: "semn_bbbbbbbbbbbbbbbb" as SemanticNode["id"],
      label: "Rerun install",
      description: "Rerun pnpm install before deploying Atlas.",
    });
    const historical = {
      ...makeNode({
        id: "semn_cccccccccccccccc" as SemanticNode["id"],
        label: "Closed Atlas proposition",
        description: "A proposition whose support is no longer current.",
      }),
      historical: true,
    };
    const summary = summarizeSemanticContext(
      {
        matched_node_ids: [root.id, historical.id],
        matched_nodes: [root, historical],
        supports: [support],
        contradicts: [],
        categories: [],
        support_hits: [
          {
            root_node_id: root.id,
            node: support,
            edgePath: [
              makeClosedEdge({
                from_node_id: root.id,
                to_node_id: support.id,
              }),
            ],
          },
        ],
        causal_hits: [],
        contradiction_hits: [],
        category_hits: [],
      } satisfies RetrievedSemantic,
      1_000,
    );

    expect(summary).toContain("Closed Atlas proposition [historical]");
    expect(summary).not.toContain("-[supports");
    expect(summary).not.toContain("[valid 2024-01-01..2024-01-10");
  });

  it("uses injected current time when filtering current-mode closed edges", () => {
    const clock = new ManualClock(Date.UTC(2024, 0, 5));
    const root = makeNode({
      id: "semn_aaaaaaaaaaaaaaaa" as SemanticNode["id"],
      kind: "entity",
      label: "Atlas",
      description: "Atlas deployment service.",
    });
    const support = makeNode({
      id: "semn_bbbbbbbbbbbbbbbb" as SemanticNode["id"],
      label: "Rerun install",
      description: "Rerun pnpm install before deploying Atlas.",
    });
    const edge = makeClosedEdge({
      from_node_id: root.id,
      to_node_id: support.id,
      valid_to: Date.UTC(2024, 0, 10),
    });
    const retrievedSemantic = {
      matched_node_ids: [root.id],
      matched_nodes: [root],
      supports: [support],
      contradicts: [],
      categories: [],
      support_hits: [
        {
          root_node_id: root.id,
          node: support,
          edgePath: [edge],
        },
      ],
      causal_hits: [],
      contradiction_hits: [],
      category_hits: [],
    } satisfies RetrievedSemantic;

    const beforeClose = summarizeSemanticContext(retrievedSemantic, 1_000, clock.now());
    clock.set(Date.UTC(2024, 0, 11));
    const afterClose = summarizeSemanticContext(retrievedSemantic, 1_000, clock.now());

    expect(beforeClose).toContain("-[supports");
    expect(afterClose).not.toContain("-[supports");
  });

  it("renders causal semantic hits in a separate bucket", () => {
    const root = makeNode({
      id: "semn_aaaaaaaaaaaaaaaa" as SemanticNode["id"],
      kind: "entity",
      label: "Atlas",
      description: "Atlas deployment service.",
    });
    const effect = makeNode({
      id: "semn_bbbbbbbbbbbbbbbb" as SemanticNode["id"],
      label: "Rollback pressure",
      description: "Atlas rollback pressure rises after failed deploys.",
    });
    const edge = makeClosedEdge({
      from_node_id: root.id,
      to_node_id: effect.id,
      relation: "causes",
      valid_to: Date.UTC(2099, 0, 1),
    });
    const summary = summarizeSemanticContext(
      {
        matched_node_ids: [root.id],
        matched_nodes: [root],
        supports: [],
        contradicts: [],
        categories: [],
        support_hits: [],
        causal_hits: [
          {
            root_node_id: root.id,
            node: effect,
            edgePath: [edge],
          },
        ],
        contradiction_hits: [],
        category_hits: [],
      } satisfies RetrievedSemantic,
      1_000,
      Date.UTC(2024, 0, 5),
    );

    expect(summary).toContain("causal:");
    expect(summary).toContain("-[causes");
  });

  it("labels under-review direct semantic matches", () => {
    const underReview = {
      ...makeNode({
        label: "Atlas claim under review",
      }),
      under_review: {
        review_id: 1,
        reason: "Supporting semantic edge was invalidated; target needs re-evaluation",
        reason_code: "support_chain_collapsed",
        invalidated_edge_id: "seme_aaaaaaaaaaaaaaaa",
        disclosureLabel: publicMemoryDisclosureLabel(),
      },
    } satisfies RetrievedSemantic["matched_nodes"][number];
    const summary = summarizeSemanticContext(
      {
        matched_node_ids: [underReview.id],
        matched_nodes: [underReview],
        supports: [],
        contradicts: [],
        categories: [],
        support_hits: [],
        causal_hits: [],
        contradiction_hits: [],
        category_hits: [],
      } satisfies RetrievedSemantic,
      1_000,
    );

    expect(summary).toContain("[under re-evaluation: support_chain_collapsed]");
    expect(summary).toContain("Atlas claim under review");
  });

  it("labels non-active semantic nodes with status metadata", () => {
    const superseded = makeNode({
      label: "Four night itinerary",
      description: "The itinerary has four nights in San Sebastian.",
      status: "superseded",
      corrected_by: "semn_bbbbbbbbbbbbbbbb" as SemanticNode["id"],
      superseded_at: 12_345,
    });
    const summary = summarizeSemanticContext(
      {
        matched_node_ids: [superseded.id],
        matched_nodes: [superseded],
        supports: [],
        contradicts: [],
        categories: [],
        support_hits: [],
        causal_hits: [],
        contradiction_hits: [],
        category_hits: [],
      } satisfies RetrievedSemantic,
      1_000,
    );

    expect(summary).toContain("[status=superseded, t=12345]");
    expect(summary).not.toContain("semn_bbbbbbbbbbbbbbbb");
    expect(summary).toContain("Four night itinerary");
  });

  it("does not label nodes without an open under-review marker", () => {
    const closedReviewNode = makeNode({
      label: "Closed review claim",
    });
    const summary = summarizeSemanticContext(
      {
        matched_node_ids: [closedReviewNode.id],
        matched_nodes: [closedReviewNode],
        supports: [],
        contradicts: [],
        categories: [],
        support_hits: [],
        causal_hits: [],
        contradiction_hits: [],
        category_hits: [],
      } satisfies RetrievedSemantic,
      1_000,
    );

    expect(summary).toContain("Closed review claim");
    expect(summary).not.toContain("[under re-evaluation:");
  });

  it("labels multiple under-review semantic nodes inline", () => {
    const first = {
      ...makeNode({
        id: "semn_bbbbbbbbbbbbbbbb" as SemanticNode["id"],
        label: "First weak claim",
      }),
      under_review: {
        review_id: 1,
        reason: "First support was invalidated",
        reason_code: "evidence_invalidated",
        invalidated_edge_id: "seme_bbbbbbbbbbbbbbbb",
        disclosureLabel: publicMemoryDisclosureLabel(),
      },
    } satisfies RetrievedSemantic["matched_nodes"][number];
    const second = {
      ...makeNode({
        id: "semn_cccccccccccccccc" as SemanticNode["id"],
        label: "Second weak claim",
      }),
      under_review: {
        review_id: 2,
        reason: "Second support was invalidated",
        reason_code: "support_chain_collapsed",
        invalidated_edge_id: "seme_cccccccccccccccc",
        disclosureLabel: publicMemoryDisclosureLabel(),
      },
    } satisfies RetrievedSemantic["matched_nodes"][number];
    const summary = summarizeSemanticContext(
      {
        matched_node_ids: [first.id, second.id],
        matched_nodes: [first, second],
        supports: [],
        contradicts: [],
        categories: [],
        support_hits: [],
        causal_hits: [],
        contradiction_hits: [],
        category_hits: [],
      } satisfies RetrievedSemantic,
      1_000,
    );

    expect(summary?.match(/\[under re-evaluation:/g)).toHaveLength(2);
    expect(summary).toContain("First weak claim");
    expect(summary).toContain("Second weak claim");
  });
});
