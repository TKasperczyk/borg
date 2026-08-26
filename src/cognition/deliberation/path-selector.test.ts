import { describe, expect, it, vi } from "vitest";

import type {
  RetrievalConfidence,
  RetrievedContradictionRouting,
  RetrievedEpisode,
} from "../../retrieval/index.js";
import type { TurnTracer } from "../../tracing/tracer.js";

import { ContradictionRoutingCooldown } from "./contradiction-routing-cooldown.js";
import { chooseDeliberationPath } from "./path-selector.js";

const CONTRADICTION_FINGERPRINT = "fingerprint-linked-contradiction";
const OPEN_QUESTION_FINGERPRINT = "open_question:oq_aaaaaaaaaaaaaaaa";

function makeEpisode(score: number, tags: string[] = []): RetrievedEpisode {
  return {
    episode: {
      id: "epi_aaaaaaaaaaaaaaaa" as RetrievedEpisode["episode"]["id"],
      title: "title",
      narrative: "narrative",
      participants: ["team"],
      location: null,
      start_time: 0,
      end_time: 1,
      source_stream_ids: [
        "strm_aaaaaaaaaaaaaaaa" as RetrievedEpisode["episode"]["source_stream_ids"][number],
      ],
      significance: 0.8,
      tags,
      confidence: 0.8,
      lineage: {
        derived_from: [],
        supersedes: [],
      },
      emotional_arc: null,
      embedding: Float32Array.from([1, 0, 0, 0]),
      created_at: 0,
      updated_at: 0,
    },
    score,
    rawScore: score,
    scoreBreakdown: {
      similarity: score,
      decayedSalience: 0.3,
      heat: 0,
      goalRelevance: 0,
      valueAlignment: 0,
      timeRelevance: 0,
      moodBoost: 0,
      socialRelevance: 0,
      entityRelevance: 0,
      suppressionPenalty: 0,
    },
    citationChain: [],
  };
}

function makeConfidence(overall: number, contradictionPresent = false): RetrievalConfidence {
  return {
    overall,
    evidenceStrength: overall,
    coverage: 1,
    sourceDiversity: 1,
    contradictionPresent,
    sampleSize: 5,
    semanticSampleSize: 0,
    coverageExpected: 5,
    diversitySources: 5,
    diversitySampleSize: 5,
    evidenceEpisodeStrength: 0,
    evidenceSemanticStrength: 0,
  };
}

function makeContradictionRouting(): RetrievedContradictionRouting {
  return {
    contradictions: [
      {
        edgeId: "edg_aaaaaaaaaaaaaaaa",
        nodeIds: ["sem_aaaaaaaaaaaaaaaa", "sem_bbbbbbbbbbbbbbbb"],
        sourceEpisodeIds: ["ep_aaaaaaaaaaaaaaaa"],
        validUntil: null,
        sessionScope: "unknown",
        linkedOpenQuestionIds: ["oq_aaaaaaaaaaaaaaaa"],
        fingerprint: CONTRADICTION_FINGERPRINT,
      },
    ],
  };
}

describe("chooseDeliberationPath", () => {
  it("uses RetrievalConfidence.overall when provided, not the relevance-score average", () => {
    // Relevance score average is high, but epistemic confidence is low.
    // Should route to S2 because the epistemic signal is what matters.
    const highRelevance = [makeEpisode(0.9), makeEpisode(0.9)];

    const decision = chooseDeliberationPath(
      "problem_solving",
      "low",
      highRelevance,
      false,
      makeConfidence(0.2),
    );

    expect(decision.path).toBe("system_2");
    expect(decision.reason).toMatch(/low retrieval confidence/i);
  });

  it("routes to S1 when epistemic confidence is high, regardless of score", () => {
    // Low relevance-score average but high epistemic confidence.
    const lowRelevance = [makeEpisode(0.1), makeEpisode(0.2)];

    const decision = chooseDeliberationPath(
      "problem_solving",
      "low",
      lowRelevance,
      false,
      makeConfidence(0.9),
    );

    expect(decision.path).toBe("system_1");
  });

  it("routes from explicit low retrieval confidence without averaging scores", () => {
    const decision = chooseDeliberationPath(
      "problem_solving",
      "low",
      [makeEpisode(0.9)],
      false,
      makeConfidence(0.2),
    );

    expect(decision.path).toBe("system_2");
    expect(decision.reason).toMatch(/low retrieval confidence/i);
  });

  it("routes to S2 when reflective mode is active regardless of confidence", () => {
    const decision = chooseDeliberationPath(
      "reflective",
      "low",
      [makeEpisode(0.9)],
      false,
      makeConfidence(0.95),
    );

    expect(decision.path).toBe("system_2");
  });

  it("routes to S1 in idle mode when confidence is high", () => {
    const decision = chooseDeliberationPath(
      "idle",
      "low",
      [makeEpisode(0.9)],
      false,
      makeConfidence(0.9),
    );

    expect(decision.path).toBe("system_1");
  });

  it("routes to S1 when contradiction is present without an operational OQ force", () => {
    const decision = chooseDeliberationPath(
      "problem_solving",
      "low",
      [makeEpisode(0.9)],
      true,
      makeConfidence(0.9, true),
    );

    expect(decision.path).toBe("system_1");
    expect(decision.contradiction_tier).toBe("confidence_penalty");
  });

  it("does not force S2 when only confidence.contradictionPresent is set", () => {
    const decision = chooseDeliberationPath(
      "problem_solving",
      "low",
      [makeEpisode(0.9)],
      false,
      makeConfidence(0.9, true),
    );

    expect(decision.path).toBe("system_1");
    expect(decision.contradiction_tier).toBe("confidence_penalty");
  });

  it("ignores warning/recommended tags as contradiction cues", () => {
    const decision = chooseDeliberationPath(
      "problem_solving",
      "low",
      [makeEpisode(0.9, ["warning"]), makeEpisode(0.9, ["recommended"])],
      false,
      makeConfidence(0.9, false),
    );

    expect(decision.path).toBe("system_1");
  });

  it("routes to S2 for high stakes even with confident retrieval", () => {
    const decision = chooseDeliberationPath(
      "problem_solving",
      "high",
      [makeEpisode(0.9)],
      false,
      makeConfidence(0.95),
    );

    expect(decision.path).toBe("system_2");
    expect(decision.reason).toMatch(/high-stakes/i);
  });

  it("escalates idle mode to S2 when stakes are high", () => {
    // Sprint 53: idle was a hard early return that bypassed the high-stakes
    // and contradiction checks below it. A misclassified high-stakes idle
    // turn must still take the deeper path.
    const decision = chooseDeliberationPath(
      "idle",
      "high",
      [makeEpisode(0.9)],
      false,
      makeConfidence(0.95),
    );

    expect(decision.path).toBe("system_2");
    expect(decision.reason).toMatch(/high-stakes/i);
  });

  it("does not escalate idle mode when only retrieved context contradicts", () => {
    const decision = chooseDeliberationPath(
      "idle",
      "low",
      [makeEpisode(0.9)],
      true,
      makeConfidence(0.9, true),
    );

    expect(decision.path).toBe("system_1");
    expect(decision.contradiction_tier).toBe("confidence_penalty");
  });

  it("honors a forced S2 routing override and traces the base path", () => {
    const emit = vi.fn<TurnTracer["emit"]>();
    const tracer = {
      enabled: true,
      includePayloads: false,
      emit,
    } satisfies TurnTracer;
    const decision = chooseDeliberationPath(
      "problem_solving",
      "low",
      [makeEpisode(0.9)],
      false,
      makeConfidence(0.9),
      {
        tracer,
        turnId: "turn-forced",
      },
      {
        forceSystem2: true,
        reason: "open_question_contradiction",
        forcedBy: "open_question_contradiction",
        oqIds: ["oq_aaaaaaaaaaaaaaaa"],
        contradictionFingerprints: [CONTRADICTION_FINGERPRINT],
        openQuestions: [
          {
            id: "oq_aaaaaaaaaaaaaaaa" as never,
            question: "Which itinerary claim is current?",
            source: "contradiction",
          },
        ],
        audienceEntityId: null,
        isOperational: true,
      },
      {
        routing: makeContradictionRouting(),
      },
    );

    expect(decision).toMatchObject({
      path: "system_2",
      reason: "open_question_contradiction",
      forced_by: "open_question_contradiction",
      contradiction_tier: "s2_forced",
    });
    expect(emit).toHaveBeenCalledWith("deliberation.path.transitioned", {
      turnId: "turn-forced",
      perceptionMode: "problem_solving",
      isOperational: true,
      audienceEntityId: null,
      openQuestionIds: ["oq_aaaaaaaaaaaaaaaa"],
      openQuestionSources: ["contradiction"],
      openQuestionLocalHandleMap: {
        contradiction_1: "oq_aaaaaaaaaaaaaaaa",
      },
      contradictionFingerprints: [CONTRADICTION_FINGERPRINT],
      basePath: "system_1",
      baseReason: "Retrieval confidence is strong enough for a direct response.",
      forcedPath: "system_2",
    });
    expect(emit).toHaveBeenCalledWith("deliberation.path.completed", {
      turnId: "turn-forced",
      path: "system_2",
      reason: "open_question_contradiction",
      confidenceOverall: 0.9,
      contradictionPresent: false,
      forced_by: "open_question_contradiction",
      contradiction_tier: "s2_forced",
      contradiction_fingerprints: [CONTRADICTION_FINGERPRINT],
      contradiction_cooldown_demoted: false,
    });
  });

  it("keeps v55 P2 forced S2 for an operational OQ-linked contradiction", () => {
    const decision = chooseDeliberationPath(
      "problem_solving",
      "low",
      [makeEpisode(0.9)],
      true,
      makeConfidence(0.9, true),
      undefined,
      {
        forceSystem2: true,
        reason: "open_question_contradiction",
        forcedBy: "open_question_contradiction",
        oqIds: ["oq_aaaaaaaaaaaaaaaa"],
        openQuestions: [
          {
            id: "oq_aaaaaaaaaaaaaaaa" as never,
            question: "Which itinerary claim is current?",
            source: "contradiction",
          },
        ],
        audienceEntityId: null,
        isOperational: true,
      },
      {
        routing: makeContradictionRouting(),
      },
    );

    expect(decision).toMatchObject({
      path: "system_2",
      forced_by: "open_question_contradiction",
      contradiction_tier: "s2_forced",
      contradiction_fingerprints: [CONTRADICTION_FINGERPRINT],
    });
  });

  it("demotes a repeated forced contradiction within the cooldown window", () => {
    const emit = vi.fn<TurnTracer["emit"]>();
    const tracer = {
      enabled: true,
      includePayloads: false,
      emit,
    } satisfies TurnTracer;
    const cooldown = new ContradictionRoutingCooldown();
    const routingOverride = {
      forceSystem2: true,
      reason: "open_question_contradiction" as const,
      forcedBy: "open_question_contradiction" as const,
      oqIds: ["oq_aaaaaaaaaaaaaaaa"],
      contradictionFingerprints: [OPEN_QUESTION_FINGERPRINT],
      openQuestions: [
        {
          id: "oq_aaaaaaaaaaaaaaaa" as never,
          question: "Which itinerary claim is current?",
          source: "contradiction" as const,
        },
      ],
      audienceEntityId: null,
      isOperational: true,
    };

    const first = chooseDeliberationPath(
      "problem_solving",
      "low",
      [makeEpisode(0.9)],
      true,
      makeConfidence(0.9, true),
      {
        tracer,
        turnId: "turn-first",
      },
      routingOverride,
      {
        cooldown,
        audienceKey: "audience",
        currentTurn: 10,
        cooldownTurns: 5,
      },
    );
    const second = chooseDeliberationPath(
      "problem_solving",
      "low",
      [makeEpisode(0.9)],
      true,
      makeConfidence(0.9, true),
      {
        tracer,
        turnId: "turn-second",
      },
      routingOverride,
      {
        cooldown,
        audienceKey: "audience",
        currentTurn: 12,
        cooldownTurns: 5,
      },
    );

    expect(first.path).toBe("system_2");
    expect(second).toMatchObject({
      path: "system_1",
      forced_by: null,
      contradiction_tier: "s2_recommended",
      contradiction_cooldown_demoted: true,
    });
    expect(emit).toHaveBeenCalledWith("deliberation.contradiction_routing.transitioned", {
      turnId: "turn-second",
      fingerprint: OPEN_QUESTION_FINGERPRINT,
      last_forced_turn: 10,
      current_turn: 12,
    });
  });

  it("preserves the natural S2 reason when the override does not change the path", () => {
    const emit = vi.fn<TurnTracer["emit"]>();
    const tracer = {
      enabled: true,
      includePayloads: false,
      emit,
    } satisfies TurnTracer;
    const decision = chooseDeliberationPath(
      "reflective",
      "low",
      [makeEpisode(0.9)],
      false,
      makeConfidence(0.9),
      {
        tracer,
        turnId: "turn-natural-s2",
      },
      {
        forceSystem2: true,
        reason: "open_question_contradiction",
        forcedBy: "open_question_contradiction",
        oqIds: ["oq_aaaaaaaaaaaaaaaa"],
        contradictionFingerprints: [CONTRADICTION_FINGERPRINT],
        openQuestions: [
          {
            id: "oq_aaaaaaaaaaaaaaaa" as never,
            question: "Which itinerary claim is current?",
            source: "contradiction",
            localHandle: "contradiction_1",
          },
        ],
        audienceEntityId: null,
        isOperational: true,
      },
      {
        routing: makeContradictionRouting(),
      },
    );

    expect(decision).toMatchObject({
      path: "system_2",
      reason: "Reflective mode always takes the deeper reasoning path.",
      forced_by: null,
      contradiction_tier: "s2_forced",
    });
    expect(emit).toHaveBeenCalledWith(
      "deliberation.path.completed",
      expect.objectContaining({
        turnId: "turn-natural-s2",
        path: "system_2",
        reason: "Reflective mode always takes the deeper reasoning path.",
        forced_by: null,
        contradiction_tier: "s2_forced",
      }),
    );
    expect(emit).toHaveBeenCalledWith(
      "deliberation.path.transitioned",
      expect.objectContaining({
        turnId: "turn-natural-s2",
        basePath: "system_2",
        baseReason: "Reflective mode always takes the deeper reasoning path.",
        contradictionFingerprints: [CONTRADICTION_FINGERPRINT],
      }),
    );
  });
});
