import { describe, expect, it, vi } from "vitest";

import {
  createEpisodeFixture,
  createRetrievalScoreFixture,
} from "../../../offline/test-support.js";
import { MEMORY_DISCLOSURE_CLASSES } from "../../../memory/common/disclosure-label.js";
import type { SemanticEdge, SemanticNode } from "../../../memory/semantic/index.js";
import type {
  EvidenceItem,
  RetrievalConfidence,
  RetrievedEpisode,
  RetrievedSemantic,
} from "../../../retrieval/index.js";
import {
  publicMemoryDisclosureLabel,
  unknownMemoryDisclosureLabel,
} from "../../../retrieval/index.js";
import { ManualClock } from "../../../util/clock.js";
import type { CommitmentId } from "../../../util/ids.js";
import {
  DEFAULT_PLAN_REQUESTED_VERIFICATION_MEMBERSHIP_TOKEN_BUDGET,
  DELIBERATION_S2_CONFIDENCE_FLOOR,
} from "../constants.js";
import { chooseDeliberationPath } from "../path-selector.js";
import {
  PLAN_REQUESTED_VERIFICATION_MEMBERSHIP_CARVE_OUT_OVERFLOW_ERROR,
  PLAN_REQUESTED_VERIFICATION_MEMBERSHIP_CARVE_OUT_OVERFLOW_MARKER,
  PLAN_REQUESTED_VERIFICATION_MEMBERSHIP_CARVE_OUT_REQUIRED_TOKENS_ATTRIBUTE,
  PLAN_REQUESTED_VERIFICATION_MEMBERSHIP_CARVE_OUT_ROWS_ATTRIBUTE,
  PLAN_REQUESTED_VERIFICATION_MEMBERSHIP_BUDGET_MARKER,
  PLAN_REQUESTED_VERIFICATION_MEMBERSHIP_DISCLOSURE_REMAINDER_MARKER,
  PLAN_REQUESTED_VERIFICATION_MEMBERSHIP_ERROR_ATTRIBUTE,
  PLAN_REQUESTED_VERIFICATION_MEMBERSHIP_NOT_OBSERVED_STATUS,
  PLAN_REQUESTED_VERIFICATION_MEMBERSHIP_ORDER,
  PLAN_REQUESTED_VERIFICATION_MEMBERSHIP_ORDER_ATTRIBUTE,
  PLAN_REQUESTED_VERIFICATION_PAYLOAD_ORDER,
  PLAN_REQUESTED_VERIFICATION_PAYLOAD_ORDER_ATTRIBUTE,
  PLAN_REQUESTED_VERIFICATION_MEMBERSHIP_REMAINDER_TOTAL_ATTRIBUTE,
  PLAN_REQUESTED_VERIFICATION_MEMBERSHIP_STATUS_ATTRIBUTE,
  PLAN_REQUESTED_VERIFICATION_RETRIEVAL_STATUS_ATTRIBUTE,
  PLAN_REQUESTED_VERIFICATION_RETRIEVAL_UNAVAILABLE_STATUS,
  PLAN_REQUESTED_VERIFICATION_ROWS_TOTAL_AS_OF_ATTRIBUTE,
  summarizeRetrievalConfidence,
  renderPlanRequestedVerificationNotCompleted,
  renderPlanRequestedVerificationRetrieval,
  summarizeRetrievedEpisodes,
  summarizeRetrievedEvidence,
  summarizeSemanticContext,
} from "./retrieval.js";

function verificationEvidence(
  id: string,
  text: string,
  overrides: Partial<EvidenceItem> = {},
): EvidenceItem {
  return {
    id,
    source: "episode",
    recallIntentId: "intent-verification",
    score: 0.8,
    text,
    matchedTerms: [],
    scoreBreakdown: {},
    disclosureLabel: publicMemoryDisclosureLabel(),
    ...overrides,
  };
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
    semanticSampleSize: overrides.semanticSampleSize ?? 0,
    coverageExpected: overrides.coverageExpected ?? 5,
    diversitySources: overrides.diversitySources ?? 0,
    diversitySampleSize: overrides.diversitySampleSize ?? 0,
    evidenceEpisodeStrength: overrides.evidenceEpisodeStrength ?? 0,
    evidenceSemanticStrength: overrides.evidenceSemanticStrength ?? 0,
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
    // diversity's is the top-N slice *plus* the semantic hit count, which is
    // bounded by neither, so neither is readable against `samples`.
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
        semanticSampleSize: 13,
        evidenceEpisodeStrength: 0,
        evidenceSemanticStrength: 0,
      }),
    );

    expect(saturated).toContain("coverage=1.00(23/4)");
    expect(saturated).toContain("diversity=0.94(17/18)");
    expect(saturated).toContain("samples=23(episodes=10+semantic=13)");
  });

  it("splits `samples` so diversity's smaller denominator is attributable, not just visible", () => {
    // `samples` counts every episode; diversity's denominator counts only the
    // top-N slice, plus the semantic hits both include. Bare, the two figures
    // differ by an amount the page cannot explain, and the readings available
    // to a reader are "different populations" or "off by one" -- neither of
    // which is what happened. Printing the shared semantic term makes both
    // halves recoverable: episodes on one side, the slice on the other.
    const summary = summarizeRetrievalConfidence(
      makeRetrievalConfidence({
        overall: 0.69,
        evidenceStrength: 1,
        coverage: 1,
        sampleSize: 27,
        semanticSampleSize: 21,
        coverageExpected: 6,
        sourceDiversity: 0.81,
        diversitySources: 21,
        diversitySampleSize: 26,
      }),
    );

    expect(summary).toContain("samples=27(episodes=6+semantic=21)");
    expect(summary).toContain("diversity=0.81(21/26)");
    // 26 - 21 = 5 episodes in the slice against 6 retrieved: the one-episode
    // gap to `samples` is the cap binding, and it is now derivable on the page.
    expect(summary).toContain("`diversity` divides by the top-N episode slice plus the same semantic");
  });

  it("prints the evidence addends and marks the line when their sum was clamped", () => {
    // `evidenceStrength` saturates like the two ratios but has no denominator
    // to give it away: it is a clamped sum of two independently-clamped parts.
    // A 1.00 that overshot and a 1.00 that landed exactly print identically
    // unless the addends and the pre-clamp sum ship with it.
    const clamped = summarizeRetrievalConfidence(
      makeRetrievalConfidence({
        overall: 0.9,
        evidenceStrength: 1,
        sampleSize: 6,
        evidenceEpisodeStrength: 0.86,
        evidenceSemanticStrength: 0.29,
      }),
    );

    expect(clamped).toContain("evidence=1.00(ep=0.86/1.00+sem=0.29/0.30,raw=1.15,clamped)");

    const measured = summarizeRetrievalConfidence(
      makeRetrievalConfidence({
        overall: 0.9,
        evidenceStrength: 0.98,
        sampleSize: 6,
        evidenceEpisodeStrength: 0.7,
        evidenceSemanticStrength: 0.28,
      }),
    );

    expect(measured).toContain("evidence=0.98(ep=0.70/1.00+sem=0.28/0.30)");
    expect(measured).not.toContain("clamped");
  });

  it("prints the routing floor beside overall, on every turn, with the ladder it sits in", () => {
    // `overall` is the only field here anything downstream acts on, and its
    // boundary lived in a second literal in the path selector. Bare, the number
    // is a quantity with no scale: a reader cannot tell 0.69 from 0.44 in
    // consequence. Rendered floor-or-no-floor, because printing the floor only
    // on turns that cross it would make crossing look like the only way the
    // path is ever decided -- it is the fourth test of four.
    const high = summarizeRetrievalConfidence(
      makeRetrievalConfidence({ overall: 0.69, sampleSize: 21 }),
    );
    const low = summarizeRetrievalConfidence(
      makeRetrievalConfidence({ overall: 0.31, sampleSize: 4 }),
    );

    expect(high).toContain("overall=0.69(s2_floor=0.45)");
    expect(low).toContain("overall=0.31(s2_floor=0.45)");
    for (const rendered of [high, low]) {
      expect(rendered).toContain("fourth test in the path ladder");
    }
  });

  it("pins the rendered routing floor to the constant the path ladder tests", () => {
    // Two literals at one value are indistinguishable from one boundary until
    // they drift, and the render asserts this number IS the routing floor.
    const rendered = summarizeRetrievalConfidence(
      makeRetrievalConfidence({ overall: 0.69, sampleSize: 21 }),
    );

    expect(rendered).toContain(`s2_floor=${DELIBERATION_S2_CONFIDENCE_FLOOR.toFixed(2)}`);
    expect(
      chooseDeliberationPath(
        "relational",
        "medium",
        [],
        false,
        makeRetrievalConfidence({
          overall: DELIBERATION_S2_CONFIDENCE_FLOOR - 0.01,
          sampleSize: 4,
        }),
      ).path,
    ).toBe("system_2");
    expect(
      chooseDeliberationPath(
        "relational",
        "medium",
        [],
        false,
        makeRetrievalConfidence({ overall: DELIBERATION_S2_CONFIDENCE_FLOOR, sampleSize: 4 }),
      ).path,
    ).toBe("system_1");
  });

  it("prints both evidence addends against their own ceilings so a pinned one is readable", () => {
    // The semantic addend is a fixed scale times a sigmoid that saturates once a
    // few supported matches exist, so it reaches the scale and stays there. Bare,
    // that is indistinguishable from a measurement that happened to land there,
    // and it hides that the field saturates at `ep >= 1 - sem` rather than at 1.
    // The episode addend carries its bound for the same reason plus one more:
    // bounding only one of two addends argues by silence that the other is
    // unbounded, which is the failure the semantic bound was added to remove.
    const pinned = summarizeRetrievalConfidence(
      makeRetrievalConfidence({
        overall: 0.69,
        evidenceStrength: 1,
        sampleSize: 26,
        evidenceEpisodeStrength: 0.74,
        evidenceSemanticStrength: 0.3,
      }),
    );

    expect(pinned).toContain("ep=0.74/1.00");
    expect(pinned).toContain("sem=0.30/0.30");

    const unpinned = summarizeRetrievalConfidence(
      makeRetrievalConfidence({
        overall: 0.5,
        evidenceStrength: 0.55,
        sampleSize: 3,
        evidenceEpisodeStrength: 0.36,
        evidenceSemanticStrength: 0.19,
      }),
    );

    expect(unpinned).toContain("ep=0.36/1.00");
    expect(unpinned).toContain("sem=0.19/0.30");
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
        evidence: [
          verificationEvidence("evidence:exact", payload, {
            disclosureLabel: unknownMemoryDisclosureLabel(),
          }),
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

  it("separates the serialized cost from the content length the canonical count pairs with", () => {
    // The directive carries characters JSON.stringify has to escape. payload_total_chars
    // absorbs that escaping, so the residual against commitment_directive_chars stops
    // being the `${type}: ` prefix and becomes an amount nothing on the page discloses.
    const directive = 'never quote "operator" context or\nbreak the line';
    const text = `rule: ${directive}`;
    const rendered = renderPlanRequestedVerificationRetrieval(
      {
        evidence: [
          verificationEvidence("evidence:commitment", text, {
            provenance: { commitmentId: "cmt_escape" as CommitmentId },
            commitment_enforcement_class: "critical",
            commitment_directive_chars: directive.length,
          }),
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
      2_000,
    );

    const totalChars = JSON.stringify({
      text,
      matched_terms: [],
      image_label: null,
      image_origin_frame: null,
      image_unavailable_reason: null,
    }).length;
    // The escaping is real: the total overshoots the wrapper-plus-content sum.
    expect(totalChars).toBeGreaterThan(107 + text.length);
    expect(rendered).toContain(`payload_total_chars="${totalChars}"`);
    expect(rendered).toContain(`payload_text_chars="${text.length}"`);
    expect(rendered).toContain(`commitment_directive_chars="${directive.length}"`);
    // What remains between the printed content length and the canonical count is exactly
    // the type prefix, with no serialization term hidden inside it.
    expect(text.length - directive.length).toBe("rule: ".length);
  });

  it("says none rather than zero when a payload has no text field", () => {
    const rendered = renderPlanRequestedVerificationRetrieval(
      {
        evidence: [],
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
          { id: "oq_1", question: "does the residual decompose?", urgency: 0.5 },
        ],
      } as never,
      2_000,
    );

    expect(rendered).toContain('source_class="open_question"');
    expect(rendered).toContain('payload_text_chars="none"');
    expect(rendered).not.toContain('payload_text_chars="0"');
  });

  it("stamps the exact membership total at read time with its prompt-clock offset", () => {
    const rowsTotalReadAtMs = Date.UTC(2026, 7, 24, 16, 38, 0, 250);
    const rendered = renderPlanRequestedVerificationRetrieval(
      {
        evidence: [verificationEvidence("evidence:dated", "dated payload")],
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
      DEFAULT_PLAN_REQUESTED_VERIFICATION_MEMBERSHIP_TOKEN_BUDGET,
      {
        rowsTotalReadAtMs,
        currentTimeMs: rowsTotalReadAtMs - 2_500,
      },
    );

    expect(rendered).toContain(
      `${PLAN_REQUESTED_VERIFICATION_ROWS_TOTAL_AS_OF_ATTRIBUTE}="2026-08-24T16:38:00.250Z"`,
    );
    expect(rendered).toContain(
      "Read at 2026-08-24T16:38:00.250Z, 2500ms after the current_time_ms at the top of this prompt",
    );
    expect(rendered).toContain("rows_total is exact as of that read, not as of now");
  });

  it.each(MEMORY_DISCLOSURE_CLASSES)(
    "accounts for omitted disclosure_class=%s without treating the label as a gate",
    (disclosureClass) => {
      const rendered = renderPlanRequestedVerificationRetrieval(
        {
          evidence: [
            verificationEvidence(`evidence:${disclosureClass}`, "ordinary", {
              disclosureLabel: {
                disclosureClass,
                originAudienceEntityIds: [],
                privateToEntityIds: [],
                publicToEntityIds: [],
              },
            }),
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
        2_000,
        0,
      );

      expect(rendered).not.toContain(`${PLAN_REQUESTED_VERIFICATION_MEMBERSHIP_ERROR_ATTRIBUTE}="`);
      expect(rendered).toContain(
        `${PLAN_REQUESTED_VERIFICATION_MEMBERSHIP_CARVE_OUT_ROWS_ATTRIBUTE}="0"`,
      );
      expect(rendered).toContain(`${PLAN_REQUESTED_VERIFICATION_MEMBERSHIP_BUDGET_MARKER}="1"`);
      expect(rendered).toContain(
        `<${PLAN_REQUESTED_VERIFICATION_MEMBERSHIP_DISCLOSURE_REMAINDER_MARKER} ${disclosureClass}="1" />`,
      );
    },
  );

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
    expect(rendered).toContain('check_not_completed_count="0"');
    expect(rendered).toContain("<check_not_completed_count>0</check_not_completed_count>");
  });

  it("reports zero membership tokens for zero candidates", () => {
    const rendered = renderPlanRequestedVerificationRetrieval(
      {
        evidence: [],
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
      0,
    );

    expect(rendered).toContain('complete_membership="true" rows_total="0"');
    expect(rendered).toContain('membership_tokens="0"');
    expect(rendered).toContain('membership_carve_out_rows_total="0"');
    expect(rendered).toContain("<omitted_count>0</omitted_count>");
    expect(rendered).toContain("<check_not_completed_count>0</check_not_completed_count>");
    expect(rendered).not.toContain(`${PLAN_REQUESTED_VERIFICATION_MEMBERSHIP_BUDGET_MARKER}="`);
    expect(rendered).not.toContain("<verification_source ");
  });

  it("flags exactly one omitted row with its exact disclosure-class remainder", () => {
    const rendered = renderPlanRequestedVerificationRetrieval(
      {
        evidence: [verificationEvidence("evidence:only", "one row")],
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
      0,
    );

    expect(rendered).toContain('complete_membership="false" rows_total="1"');
    expect(rendered).toContain(`${PLAN_REQUESTED_VERIFICATION_MEMBERSHIP_BUDGET_MARKER}="1"`);
    expect(rendered).toContain(
      `<${PLAN_REQUESTED_VERIFICATION_MEMBERSHIP_BUDGET_MARKER} ${PLAN_REQUESTED_VERIFICATION_MEMBERSHIP_REMAINDER_TOTAL_ATTRIBUTE}="1">`,
    );
    expect(rendered).toContain(
      `<${PLAN_REQUESTED_VERIFICATION_MEMBERSHIP_DISCLOSURE_REMAINDER_MARKER} public="1" />`,
    );
    expect(rendered).toContain("<omitted_count>1</omitted_count>");
    expect(rendered).toContain("<check_not_completed_count>0</check_not_completed_count>");
  });

  it("breaks a truncated remainder down by every disclosure class actually omitted", () => {
    const labelledEvidence = (
      id: string,
      disclosureClass: "relationship_private" | "self_private" | "unknown",
    ) =>
      verificationEvidence(id, "labelled row", {
        disclosureLabel: {
          disclosureClass,
          originAudienceEntityIds: [],
          privateToEntityIds: [],
          publicToEntityIds: [],
        },
      });
    const rendered = renderPlanRequestedVerificationRetrieval(
      {
        evidence: [
          labelledEvidence("evidence:relationship", "relationship_private"),
          labelledEvidence("evidence:self", "self_private"),
          labelledEvidence("evidence:unknown-a", "unknown"),
          labelledEvidence("evidence:unknown-b", "unknown"),
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
      2_000,
      0,
    );

    expect(rendered).toContain(`${PLAN_REQUESTED_VERIFICATION_MEMBERSHIP_BUDGET_MARKER}="4"`);
    expect(rendered).toContain(
      `<${PLAN_REQUESTED_VERIFICATION_MEMBERSHIP_DISCLOSURE_REMAINDER_MARKER} relationship_private="1" self_private="1" unknown="2" />`,
    );
    const breakdownAttributes = new RegExp(
      `<${PLAN_REQUESTED_VERIFICATION_MEMBERSHIP_DISCLOSURE_REMAINDER_MARKER} ([^>]+) />`,
    ).exec(rendered)?.[1];
    const breakdownTotal = [...(breakdownAttributes ?? "").matchAll(/="(\d+)"/g)].reduce(
      (sum, match) => sum + Number(match[1]),
      0,
    );
    expect(breakdownTotal).toBe(4);
  });

  it("fully enumerates the 300-row commitment and goal scale at the default budget", () => {
    const rendered = renderPlanRequestedVerificationRetrieval(
      {
        evidence: Array.from({ length: 300 }, (_unused, index) => {
          const idSuffix = index.toString(36).padStart(16, "0");
          const commitmentId = `cmt_${idSuffix}`;
          return verificationEvidence(
            `evidence_commitment_${commitmentId}_recall_known_term_0`,
            "z".repeat(40),
            {
              source: "commitment",
              provenance: { commitmentId: commitmentId as never },
              disclosureLabel: unknownMemoryDisclosureLabel(),
              commitment_enforcement_class: "advisory",
              commitment_critical_domain: null,
            },
          );
        }),
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
    // 58_800, up from 57_000: every row now also carries payload_text_chars, ~24 chars
    // per row against the membership budget. Paid on all rows, not just commitment ones,
    // because absence would otherwise be the only way to say "this payload has no text".
    expect(membershipTokens).toBe(58_800);
    expect(membershipTokens).toBeLessThan(
      DEFAULT_PLAN_REQUESTED_VERIFICATION_MEMBERSHIP_TOKEN_BUDGET,
    );
    expect(rendered).toContain('complete_membership="true" rows_total="300"');
    expect(rendered.match(/<verification_source /g)).toHaveLength(300);
    expect(rendered).toContain(
      'handle="evidence_commitment_cmt_0000000000000000_recall_known_term_0"',
    );
    expect(rendered).toContain(
      'handle="evidence_commitment_cmt_000000000000008b_recall_known_term_0"',
    );
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
    expect(rendered).toContain('check_not_completed_count="2"');
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
      `<${PLAN_REQUESTED_VERIFICATION_MEMBERSHIP_BUDGET_MARKER} ${PLAN_REQUESTED_VERIFICATION_MEMBERSHIP_REMAINDER_TOTAL_ATTRIBUTE}="2">`,
    );
    expect(rendered).toContain(
      `<${PLAN_REQUESTED_VERIFICATION_MEMBERSHIP_DISCLOSURE_REMAINDER_MARKER} public="2" />`,
    );
    expect(rendered).toContain("<omitted_count>2</omitted_count>");
    expect(rendered).toContain("<check_not_completed_count>0</check_not_completed_count>");
    expect(rendered).not.toContain('handle="evidence:middle"');
    expect(rendered).not.toContain('handle="evidence:beta"');
    expect(rendered).toContain(
      "complete_membership=true means every one of rows_total handles and its structural fields is enumerated",
    );
    expect(rendered).toContain(
      `${PLAN_REQUESTED_VERIFICATION_MEMBERSHIP_BUDGET_MARKER}=N and its same-named marker's ${PLAN_REQUESTED_VERIFICATION_MEMBERSHIP_REMAINDER_TOTAL_ATTRIBUTE}=N carry the exact un-enumerated remainder`,
    );
    expect(rendered).toContain(
      `${PLAN_REQUESTED_VERIFICATION_MEMBERSHIP_DISCLOSURE_REMAINDER_MARKER} has one attribute per disclosure_class present among omitted rows`,
    );
    expect(rendered).toContain(
      "Payloads are priced against payload_target_tokens alone and never consume the membership budget, and membership never consumes the payload budget",
    );
    expect(rendered).toContain(
      `${PLAN_REQUESTED_VERIFICATION_PAYLOAD_ORDER_ATTRIBUTE}=${PLAN_REQUESTED_VERIFICATION_PAYLOAD_ORDER} means payload admission walks these enumerated rows in the order they are rendered here`,
    );
    expect(rendered).toContain(
      "rows that are not enumerated never consume payload budget",
    );
  });

  it("prices payloads in rendered order so the carve-out is not starved by rows above or omitted", () => {
    const ordinarySmall = verificationEvidence("evidence:ordinary-small", "small ordinary");
    const ordinaryHuge = verificationEvidence("evidence:ordinary-huge", "x".repeat(4_000));
    const criticalSmall = verificationEvidence("evidence:critical-small", "small critical", {
      source: "commitment",
      provenance: { commitmentId: "cmt_critical_small" as never },
      commitment_enforcement_class: "critical",
    });
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
    const enumeratedOnly = renderPlanRequestedVerificationRetrieval(
      {
        evidence: [ordinarySmall, criticalSmall],
        episodes: [],
        semantic,
        open_questions: [],
      } as never,
      Number.MAX_SAFE_INTEGER,
      Number.MAX_SAFE_INTEGER,
    );
    const membershipTokens = Number(/membership_tokens="(\d+)"/.exec(enumeratedOnly)?.[1]);
    const rendered = renderPlanRequestedVerificationRetrieval(
      {
        evidence: [ordinarySmall, ordinaryHuge, criticalSmall],
        episodes: [],
        semantic,
        open_questions: [],
      } as never,
      1_060,
      membershipTokens,
    );
    const payloadStatuses = [
      ...rendered.matchAll(
        /<verification_source handle="([^"]+)"[^>]*payload_status="([a-z_]+)"/g,
      ),
    ].map((match) => [match[1], match[2]]);

    // The huge row is omitted from membership, so it never competes for payload,
    // and the carve-out is priced ahead of the ordinary row rendered below it.
    expect(payloadStatuses).toEqual([
      ["evidence:critical-small", "exact"],
      ["evidence:ordinary-small", "exact"],
    ]);
    expect(rendered).toContain(
      `${PLAN_REQUESTED_VERIFICATION_PAYLOAD_ORDER_ATTRIBUTE}="${PLAN_REQUESTED_VERIFICATION_PAYLOAD_ORDER}"`,
    );
    expect(rendered).toContain("<check_not_completed_count>0</check_not_completed_count>");
    expect(rendered).toContain('payload_tokens_included="62"');
  });

  it("enumerates critical commitments first while preserving both partition orders", () => {
    const ordinaryA = verificationEvidence("evidence:ordinary-a", "ordinary A");
    const criticalA = verificationEvidence("evidence:critical-a", "critical A", {
      source: "commitment",
      provenance: { commitmentId: "cmt_critical_a" as never },
      commitment_enforcement_class: "critical",
      commitment_critical_domain: "privacy",
    });
    const ordinaryB = verificationEvidence("evidence:ordinary-b", "ordinary B");
    const restrictive = verificationEvidence("evidence:restrictive", "restrictive", {
      disclosureLabel: unknownMemoryDisclosureLabel(),
    });
    const criticalB = verificationEvidence("evidence:critical-b", "critical B", {
      source: "commitment",
      provenance: { commitmentId: "cmt_critical_b" as never },
      commitment_enforcement_class: "critical",
      commitment_critical_domain: "audience_scope",
    });
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
    const protectedPlusFirstOrdinary = renderPlanRequestedVerificationRetrieval(
      {
        evidence: [criticalA, criticalB, ordinaryA],
        episodes: [],
        semantic,
        open_questions: [],
      } as never,
      2_000,
      Number.MAX_SAFE_INTEGER,
    );
    const protectedPlusFirstOrdinaryTokens = Number(
      /membership_tokens="(\d+)"/.exec(protectedPlusFirstOrdinary)?.[1],
    );
    const rendered = renderPlanRequestedVerificationRetrieval(
      {
        evidence: [ordinaryA, criticalA, ordinaryB, restrictive, criticalB],
        episodes: [],
        semantic,
        open_questions: [],
      } as never,
      2_000,
      protectedPlusFirstOrdinaryTokens,
    );
    const enumeratedHandles = [...rendered.matchAll(/<verification_source handle="([^"]+)"/g)].map(
      (match) => match[1],
    );

    expect(enumeratedHandles).toEqual([
      "evidence:critical-a",
      "evidence:critical-b",
      "evidence:ordinary-a",
    ]);
    expect(rendered).toContain(
      `${PLAN_REQUESTED_VERIFICATION_MEMBERSHIP_ORDER_ATTRIBUTE}="${PLAN_REQUESTED_VERIFICATION_MEMBERSHIP_ORDER}"`,
    );
    expect(rendered).toContain('commitment_enforcement_class="critical"');
    expect(rendered).toContain('commitment_critical_domain="privacy"');
    expect(rendered).toContain(
      `${PLAN_REQUESTED_VERIFICATION_MEMBERSHIP_CARVE_OUT_ROWS_ATTRIBUTE}="2"`,
    );
    expect(rendered).toContain(`${PLAN_REQUESTED_VERIFICATION_MEMBERSHIP_BUDGET_MARKER}="2"`);
    expect(rendered).toContain(
      `<${PLAN_REQUESTED_VERIFICATION_MEMBERSHIP_DISCLOSURE_REMAINDER_MARKER} public="1" unknown="1" />`,
    );
    expect(rendered).not.toContain('handle="evidence:ordinary-b"');
    expect(rendered).not.toContain('handle="evidence:restrictive"');
    expect(rendered).toContain("Disclosure labels do not affect membership admission or ordering");
    expect(rendered).toContain("The renderer performs no content-based selection or new ranking");
  });

  it("fails the whole check with exact counts when carve-out rows alone exceed budget", () => {
    const critical = verificationEvidence("evidence:critical", "critical", {
      source: "commitment",
      provenance: { commitmentId: "cmt_critical" as never },
      commitment_enforcement_class: "critical",
      commitment_critical_domain: "privacy",
    });
    const restrictive = verificationEvidence("evidence:restrictive", "restrictive", {
      disclosureLabel: unknownMemoryDisclosureLabel(),
    });
    const ordinary = verificationEvidence("evidence:ordinary", "ordinary");
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
    const carveOutOnly = renderPlanRequestedVerificationRetrieval(
      {
        evidence: [critical],
        episodes: [],
        semantic,
        open_questions: [],
      } as never,
      2_000,
      Number.MAX_SAFE_INTEGER,
    );
    const carveOutRequiredTokens = Number(/membership_tokens="(\d+)"/.exec(carveOutOnly)?.[1]);
    const onMembershipCarveOutOverflow = vi.fn();
    const rendered = renderPlanRequestedVerificationRetrieval(
      {
        evidence: [ordinary, critical, restrictive],
        episodes: [],
        semantic,
        open_questions: [],
      } as never,
      2_000,
      carveOutRequiredTokens - 1,
      { onMembershipCarveOutOverflow },
    );

    expect(onMembershipCarveOutOverflow).toHaveBeenCalledWith({
      rowsTotal: 3,
      carveOutRowsTotal: 1,
      carveOutRequiredTokens,
      membershipTargetTokens: carveOutRequiredTokens - 1,
    });
    expect(rendered).toContain(
      `${PLAN_REQUESTED_VERIFICATION_MEMBERSHIP_ERROR_ATTRIBUTE}="${PLAN_REQUESTED_VERIFICATION_MEMBERSHIP_CARVE_OUT_OVERFLOW_ERROR}"`,
    );
    expect(rendered).toContain(
      `${PLAN_REQUESTED_VERIFICATION_MEMBERSHIP_CARVE_OUT_ROWS_ATTRIBUTE}="1"`,
    );
    expect(rendered).toContain(
      `${PLAN_REQUESTED_VERIFICATION_MEMBERSHIP_CARVE_OUT_REQUIRED_TOKENS_ATTRIBUTE}="${carveOutRequiredTokens}"`,
    );
    expect(rendered).toContain(`${PLAN_REQUESTED_VERIFICATION_MEMBERSHIP_BUDGET_MARKER}="3"`);
    expect(rendered).toContain(
      `<${PLAN_REQUESTED_VERIFICATION_MEMBERSHIP_BUDGET_MARKER} ${PLAN_REQUESTED_VERIFICATION_MEMBERSHIP_REMAINDER_TOTAL_ATTRIBUTE}="3">`,
    );
    expect(rendered).toContain(
      `<${PLAN_REQUESTED_VERIFICATION_MEMBERSHIP_DISCLOSURE_REMAINDER_MARKER} public="2" unknown="1" />`,
    );
    expect(rendered).not.toContain("<verification_source ");
    expect(rendered).toContain('payload_tokens_included="0"');
    expect(rendered).toContain("<omitted_count>3</omitted_count>");
    expect(rendered).toContain('check_not_completed_count="0"');
    expect(rendered).toContain("<check_not_completed_count>0</check_not_completed_count>");
    expect(rendered).toMatch(
      new RegExp(
        `<${PLAN_REQUESTED_VERIFICATION_MEMBERSHIP_CARVE_OUT_OVERFLOW_MARKER}[^>]+/>\\n</plan_requested_verification_retrieval>$`,
      ),
    );
  });

  it("renders an unavailable plan-requested check with a handle and zero payload", () => {
    const rendered = renderPlanRequestedVerificationNotCompleted();

    expect(rendered).toContain(
      `${PLAN_REQUESTED_VERIFICATION_RETRIEVAL_STATUS_ATTRIBUTE}="${PLAN_REQUESTED_VERIFICATION_RETRIEVAL_UNAVAILABLE_STATUS}"`,
    );
    expect(rendered).toContain(
      `${PLAN_REQUESTED_VERIFICATION_MEMBERSHIP_STATUS_ATTRIBUTE}="${PLAN_REQUESTED_VERIFICATION_MEMBERSHIP_NOT_OBSERVED_STATUS}"`,
    );
    expect(rendered).toContain('handle="plan:verification_steps"');
    expect(rendered).toContain('payload_status="check_not_completed_retrieval_unavailable"');
    expect(rendered).toContain('payload_included_chars="0"');
    expect(rendered).toContain('payload_total_chars="0"');
    expect(rendered).toContain('payload_json=""');
    expect(rendered).not.toContain("complete_membership=");
    expect(rendered).not.toContain("rows_total=");
    expect(rendered).not.toContain("<omitted_count>");
    expect(rendered).not.toContain(`${PLAN_REQUESTED_VERIFICATION_MEMBERSHIP_BUDGET_MARKER}=`);
    expect(rendered).not.toContain(`<${PLAN_REQUESTED_VERIFICATION_MEMBERSHIP_BUDGET_MARKER}`);
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
